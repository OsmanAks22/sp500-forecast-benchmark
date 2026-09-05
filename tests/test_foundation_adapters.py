"""The two hazards in KronosAdapter that produce wrong numbers rather than errors.

Kronos is the only model in the roster whose output cannot be checked by reading
it. Chronos-2's quantile axis is verified against the checkpoint's own level list
and TTM's shape against its config, but a set of sampled paths is just plausible
numbers - there is no property of the array that says whether the sampler was
given the temperature it was meant to have, or whether the sample axis it carries
is real. So both are pinned here instead.

Neither test needs a GPU or the kronos package.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_foundation_under_test", ROOT / "scripts" / "run_foundation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

SAMPLE_COUNT = 200


def _fake_kronos_module(signature_fn=None):
    """A stand-in for ``model.kronos`` carrying its own ``np`` binding."""
    module = types.ModuleType("fake_kronos")
    module.np = np
    if signature_fn is not None:
        module.auto_regressive_inference = signature_fn
    return module


def _real_signature(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len,
                    clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5,
                    verbose=False):
    """The signature as installed at model/kronos.py:389."""
    return None


# --------------------------------------------------------------------------
# 1. the numpy shim must not escape the kronos module
# --------------------------------------------------------------------------

def test_proxy_passes_through_only_the_sample_axis_reduction():
    proxy = runner._NumpyProxy(np, SAMPLE_COUNT)

    # The reduction that would destroy the predictive distribution: returned whole.
    paths = np.zeros((3, SAMPLE_COUNT, 1, 6))
    assert proxy.mean(paths, axis=1).shape == (3, SAMPLE_COUNT, 1, 6)

    # A 4-d axis-1 mean whose sample extent does not match is a different call
    # and must still reduce.
    other = np.zeros((3, 7, 1, 6))
    assert proxy.mean(other, axis=1).shape == (3, 1, 6)

    # Any other reduction is ordinary numpy.
    assert proxy.mean(np.array([1.0, 3.0])) == 2.0
    assert proxy.mean(paths, axis=0).shape == (SAMPLE_COUNT, 1, 6)


def test_proxy_delegates_every_other_numpy_attribute():
    proxy = runner._NumpyProxy(np, SAMPLE_COUNT)
    assert proxy.sqrt(np.array([4.0]))[0] == 2.0
    assert proxy.ndarray is np.ndarray
    assert proxy.float64 is np.float64


def test_the_shim_does_not_leak_into_the_numpy_module(monkeypatch):
    """The point of the proxy.

    Setting ``kronos_module.np.mean`` would set the attribute on the numpy module
    object, which every other module shares, so an unrelated ``np.mean`` would
    hit the shim for the duration. Rebinding the *name* inside kronos's namespace
    cannot reach anyone else's binding.
    """
    adapter = runner.KronosAdapter(
        our_grid=(0.1, 0.5, 0.9), context=512, device="cpu", batch=8,
        sample_count=SAMPLE_COUNT)
    adapter.kronos_module = _fake_kronos_module()

    real_mean = np.mean
    probe = np.zeros((2, SAMPLE_COUNT, 3, 4))

    with adapter._keep_sample_axis(SAMPLE_COUNT):
        # Inside kronos's namespace the proxy is in force ...
        assert adapter.kronos_module.np.mean(probe, axis=1).shape == probe.shape
        # ... and everywhere else numpy is untouched.
        assert np.mean is real_mean
        assert np.mean(probe, axis=1).shape == (2, 3, 4)

    assert adapter.kronos_module.np is np, "the real module must be restored"
    assert np.mean is real_mean


def test_the_binding_is_restored_even_if_the_call_raises():
    adapter = runner.KronosAdapter(
        our_grid=(0.1, 0.5, 0.9), context=512, device="cpu", batch=8,
        sample_count=SAMPLE_COUNT)
    adapter.kronos_module = _fake_kronos_module()

    with pytest.raises(RuntimeError):
        with adapter._keep_sample_axis(SAMPLE_COUNT):
            raise RuntimeError("decode blew up")
    assert adapter.kronos_module.np is np


# --------------------------------------------------------------------------
# 2. the sampling arguments must be passed by name, and checked
# --------------------------------------------------------------------------

def test_the_installed_signature_is_accepted(capsys):
    module = _fake_kronos_module(_real_signature)
    runner._assert_kronos_signature(module)          # must not raise
    assert "all passed by keyword" in capsys.readouterr().out


def test_every_parameter_the_runner_names_is_in_the_installed_signature():
    """The list is not aspirational - it is what predict_bars actually passes."""
    params = set(runner.inspect.signature(_real_signature).parameters)
    assert set(runner.KRONOS_INFERENCE_PARAMS) <= params


def test_a_renamed_parameter_fails_loudly_rather_than_shifting_silently():
    """The failure mode this replaces.

    Positionally, a parameter renamed or inserted upstream would hand the
    temperature to top_k and return samples that look entirely reasonable. By
    keyword the same change cannot even be called, and this check catches it at
    load rather than at the call.
    """
    def moved(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len,
              clip=5, temperature=1.0, top_k=0, top_p=0.99, sample_count=5,
              verbose=False):
        return None

    module = _fake_kronos_module(moved)
    with pytest.raises(SystemExit, match="missing parameter"):
        runner._assert_kronos_signature(module)


def test_positional_only_parameters_are_rejected():
    def positional(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, /,
                   clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5,
                   verbose=False):
        return None

    module = _fake_kronos_module(positional)
    with pytest.raises(SystemExit, match="positional-only"):
        runner._assert_kronos_signature(module)


def test_a_missing_function_says_why_predict_is_not_a_substitute():
    with pytest.raises(SystemExit, match="averages the sample axis"):
        runner._assert_kronos_signature(_fake_kronos_module())


# --------------------------------------------------------------------------
# 3. the memory model: sequences resident are batch x sample_count
# --------------------------------------------------------------------------

def _adapter(sample_count, paths_per_call, budget):
    return runner.KronosAdapter(
        our_grid=(0.1, 0.5, 0.9), context=512, device="cpu", batch=256,
        sample_count=sample_count, paths_per_call=paths_per_call,
        sequence_budget=budget)


def test_sample_chunks_sum_to_sample_count_and_respect_the_ceiling():
    for sample_count in (100, 200, 500, 1000):
        for width in (1, 16, 64, 128):
            adapter = _adapter(sample_count, width, 512)
            chunks = adapter.sample_chunks()
            assert sum(chunks) == sample_count, (sample_count, width, chunks)
            assert max(chunks) <= adapter.paths_per_call
            assert all(c > 0 for c in chunks)


def test_paths_per_call_never_exceeds_sample_count():
    """At S=100 a 128-wide chunk would ask for more paths than exist."""
    adapter = _adapter(100, 128, 512)
    assert adapter.paths_per_call == 100
    assert adapter.sample_chunks() == [100]


def test_resident_sequences_stay_under_the_budget_at_every_sample_count():
    """The invariant the OOM violated.

    The old code chunked only the origin dimension, so the product
    origin_batch * sample_count grew with the sweep. It must now be flat.
    """
    for sample_count in (100, 200, 500, 1000):
        adapter = _adapter(sample_count, 64, 512)
        resident = adapter.origin_batch * adapter.paths_per_call
        assert resident <= 512, (sample_count, resident)


def test_the_old_sizing_would_have_exceeded_the_card_and_the_new_one_does_not():
    """Anchored on the configuration that actually OOMed on the T4."""
    adapter = _adapter(100, 64, 512)
    old_style_origin_batch = 4096 // 100          # the previous per_call formula
    old_resident = old_style_origin_batch * 100   # sample dim was not chunked
    assert old_resident == 4000
    new_resident = adapter.origin_batch * adapter.paths_per_call
    assert new_resident <= 512
    assert new_resident < old_resident / 4


def test_origin_batch_shrinks_when_the_budget_is_cut_after_an_oom():
    """forecast_fold reads origin_batch each iteration, so the cut must show."""
    adapter = _adapter(500, 64, 512)
    before = adapter.origin_batch
    adapter.sequence_budget //= 2
    assert adapter.origin_batch == before // 2
    assert adapter.origin_batch >= 1


def test_origin_batch_never_falls_below_one():
    adapter = _adapter(500, 64, 8)      # budget smaller than one chunk
    assert adapter.origin_batch == 1


def test_oom_is_recognised_across_torch_versions():
    """The except clause cannot name torch.OutOfMemoryError - it may not exist.

    ``_is_oom`` is only the second half of the test: the ``except RuntimeError``
    clause narrows the type, and this decides whether a RuntimeError that got
    that far is a memory failure. It is deliberately not a type check, because
    the class is named differently across torch versions and is absent before
    2.4 - naming it in an except clause would raise AttributeError at the moment
    the handler is needed.
    """
    class OutOfMemoryError(RuntimeError):        # how torch 2.4+ spells it
        pass

    assert runner._is_oom(OutOfMemoryError("Tried to allocate 1.95 GiB"))
    assert runner._is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not runner._is_oom(RuntimeError("shape mismatch"))
    # Anything that is not a RuntimeError is re-raised by the caller before this
    # is consulted, so the predicate is never asked about one.
    assert isinstance(OutOfMemoryError("x"), RuntimeError)


def test_the_proxy_matches_the_chunk_extent_not_the_full_sample_count():
    """The trap introduced by chunking.

    Once the sample dimension is chunked the returned array's sample axis is the
    chunk width. A proxy still matching on the full sample_count would leave the
    reduction in place and the sample axis would be silently collapsed.
    """
    adapter = _adapter(500, 64, 512)
    adapter.kronos_module = _fake_kronos_module()
    chunk = adapter.sample_chunks()[0]
    assert chunk == 64 and chunk != adapter.sample_count

    block = np.zeros((4, chunk, 1, 6))
    with adapter._keep_sample_axis(chunk):
        assert adapter.kronos_module.np.mean(block, axis=1).shape == block.shape
    # Matching on the full sample_count instead would have reduced it away.
    with adapter._keep_sample_axis(adapter.sample_count):
        assert adapter.kronos_module.np.mean(block, axis=1).shape == (4, 1, 6)
