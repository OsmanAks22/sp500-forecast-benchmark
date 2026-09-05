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


def test_backoff_walks_down_to_a_size_that_fits_rather_than_halving_once():
    """The bug this replaces: one halving, then a crash.

    On the T4 both 4,000 and 2,000 sequences failed, so a single halving from an
    over-large start still OOMs and the run died with work left it could have
    done. The loop must keep backing off until something runs.
    """
    adapter = _adapter(500, 64, 512)
    fits_at = 128
    attempts = []

    def fake_predict_bars(windows, x_stamps, y_stamps, last_close):
        attempts.append(adapter.origin_batch * adapter.paths_per_call)
        if attempts[-1] > fits_at:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.95 GiB")
        return np.zeros((len(windows), 3)), {}

    adapter._predict_bars = fake_predict_bars
    levels, _ = adapter.predict_bars(np.zeros((2, 4, 6)), np.zeros((2, 4, 3)),
                                     np.zeros((2, 1, 3)), np.ones(2))
    assert levels.shape == (2, 3)
    assert attempts[0] > fits_at, "the first attempt should be the oversized one"
    assert attempts[-1] <= fits_at
    assert len(attempts) > 2, f"must back off more than once, got {attempts}"
    assert adapter.budget_was_reduced


def test_backoff_stops_at_the_floor_instead_of_looping_forever():
    adapter = _adapter(500, 64, 128)

    def always_oom(windows, x_stamps, y_stamps, last_close):
        raise RuntimeError("CUDA out of memory")

    adapter._predict_bars = always_oom
    with pytest.raises(SystemExit, match="floor budget"):
        adapter.predict_bars(np.zeros((2, 4, 6)), np.zeros((2, 4, 3)),
                             np.zeros((2, 1, 3)), np.ones(2))
    assert adapter.sequence_budget == runner.MIN_KRONOS_SEQUENCE_BUDGET


def test_a_non_memory_runtime_error_is_not_swallowed_by_the_backoff():
    adapter = _adapter(500, 64, 512)

    def wrong_shape(windows, x_stamps, y_stamps, last_close):
        raise RuntimeError("shape mismatch: expected 3 dims")

    adapter._predict_bars = wrong_shape
    with pytest.raises(RuntimeError, match="shape mismatch"):
        adapter.predict_bars(np.zeros((2, 4, 6)), np.zeros((2, 4, 3)),
                             np.zeros((2, 1, 3)), np.ones(2))
    assert not adapter.budget_was_reduced


# --------------------------------------------------------------------------
# 4. TimesFM: the method name and the channel layout are discovered, not assumed
# --------------------------------------------------------------------------

class _Surface:
    """An object exposing a chosen subset of plausible inference methods."""

    def __init__(self, *names):
        for name in names:
            setattr(self, name, lambda *a, **k: None)


def test_discovery_prefers_the_batched_entry_point():
    name, _ = runner._discover_inference_method(
        _Surface("predict", "predict_batch", "compile"),
        runner.TIMESFM_METHOD_PREFERENCE, "TimesFM3Forecaster")
    assert name == "predict_batch"


def test_discovery_falls_back_through_the_preference_order():
    name, _ = runner._discover_inference_method(
        _Surface("forecast", "compile"),
        runner.TIMESFM_METHOD_PREFERENCE, "TimesFM3Forecaster")
    assert name == "forecast"


def test_discovery_failure_attaches_the_namespace_instead_of_guessing():
    """What the AttributeError on Kaggle should have produced the first time."""
    with pytest.raises(SystemExit) as excinfo:
        runner._discover_inference_method(
            _Surface("compile", "load_checkpoint"),
            runner.TIMESFM_METHOD_PREFERENCE, "TimesFM3Forecaster")
    message = str(excinfo.value)
    assert "public attributes" in message
    assert "compile" in message and "load_checkpoint" in message


def test_the_pinned_checkpoint_is_never_left_to_the_config_default():
    """The default is a different model generation with a compatible shape."""
    assert runner.TIMESFM_CHECKPOINT == "google/timesfm-2.5-200m-pytorch"
    assert "3.0" not in runner.TIMESFM_CHECKPOINT


def _timesfm_adapter():
    adapter = runner.TimesFMAdapter(
        our_grid=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
        context=512, device="cpu", batch=32)
    adapter.native = list(runner.TIMESFM_NATIVE_GRID)
    adapter.channel_layout = None
    return adapter


def test_layout_resolved_for_a_ten_channel_head_with_a_point_channel():
    """timesfm_2p5_torch.py:53 - len(quantiles) + 1 channels, median at 5."""
    adapter = _timesfm_adapter()
    levels = np.asarray(runner.TIMESFM_NATIVE_GRID)
    rows = np.stack([np.concatenate([[0.5], levels]) for _ in range(4)])
    point = np.full(4, 0.5)                     # the median level's value
    assert adapter._resolve_layout(rows, point) == 1


def test_layout_resolved_for_a_nine_channel_head_indexed_by_level():
    """_ModelConfig.median_quantile_index 4 - one channel per level."""
    adapter = _timesfm_adapter()
    levels = np.asarray(runner.TIMESFM_NATIVE_GRID)
    rows = np.stack([levels for _ in range(4)])
    point = np.full(4, 0.5)
    assert adapter._resolve_layout(rows, point) == 0


def test_an_unverifiable_layout_raises_rather_than_shifting_every_level():
    """The silent failure: an offset that is wrong by one changes no shape."""
    adapter = _timesfm_adapter()
    rows = np.stack([np.arange(10, dtype=float) for _ in range(4)])
    point = np.full(4, 999.0)                   # matches no channel
    with pytest.raises(AssertionError, match="could not identify"):
        adapter._resolve_layout(rows, point)


def test_an_unexpected_channel_count_is_rejected():
    adapter = _timesfm_adapter()
    rows = np.zeros((4, 5))
    with pytest.raises(AssertionError, match="neither"):
        adapter._resolve_layout(rows, np.zeros(4))


# --------------------------------------------------------------------------
# 5. fp16 is a numerics change to a sampler, so it has to be evidenced
# --------------------------------------------------------------------------

def _rows(fp32_values, fp16_values, sample_count=100):
    rows = []
    for precision, values in (("fp32", fp32_values), ("fp16", fp16_values)):
        for i, crps in enumerate(values):
            rows.append({"sample_count": sample_count, "precision": precision,
                         "repeat": i, "crps": crps, "seconds": 1.0,
                         "n_origins": 253, "seconds_per_origin": 0.004,
                         "projected_full_run_gpu_hours": 10.0,
                         "peak_gpu_gib": 3.0, "sequences_per_call": 512,
                         "budget_reduced_after_oom": False})
    return rows


def test_a_shift_smaller_than_the_noise_floor_is_declared_within_noise():
    # fp32 disagrees with itself by 4e-6; fp16 sits 1e-6 from its mean.
    rows = _rows([0.003700, 0.003704], [0.003703, 0.003701])
    verdict = runner._precision_verdicts(rows, repeats=2)
    assert verdict["within_noise"] is True
    assert "verified against fp32" in verdict["sentence"]
    assert "within the" in verdict["sentence"]


def test_a_shift_larger_than_the_noise_floor_is_declared_not_verified():
    # fp32 is stable to 1e-6, fp16 is 5e-5 away - an order of magnitude clear.
    rows = _rows([0.003700, 0.003701], [0.003750, 0.003751])
    verdict = runner._precision_verdicts(rows, repeats=2)
    assert verdict["within_noise"] is False
    assert "NOT verified" in verdict["sentence"]
    assert "not free" in verdict["sentence"]


def test_the_verdict_sentence_carries_both_figures_and_the_noise():
    """The sentence is what makes the choice defensible, so it must be specific."""
    rows = _rows([0.003700, 0.003704], [0.003702, 0.003703])
    sentence = runner._precision_verdicts(rows, repeats=2)["sentence"]
    assert "0.0037" in sentence                      # both CRPS figures
    assert "%" in sentence                           # the relative shift
    assert "fold 0" in sentence and "S=100" in sentence
    assert "same origins" in sentence


def test_one_run_per_precision_refuses_to_declare_anything():
    """A bare difference with no noise floor is the number that misleads."""
    rows = _rows([0.003700], [0.003750])
    verdict = runner._precision_verdicts(rows, repeats=1)
    assert verdict["within_noise"] is None
    assert verdict["sentence"] is None
    assert any("not measured" in line for line in verdict["report"])


def test_the_noise_floor_is_the_wider_of_the_two_spreads():
    """Conservative: a larger floor makes 'within noise' harder to earn."""
    rows = _rows([0.003700, 0.003701], [0.003700, 0.003760])
    report = "\n".join(runner._precision_verdicts(rows, repeats=2)["report"])
    assert "0.000060" in report, report


def test_missing_a_precision_produces_no_verdict():
    rows = _rows([0.0037, 0.0037], [])
    verdict = runner._precision_verdicts(rows, repeats=2)
    assert verdict["sentence"] is None


def test_autocast_is_off_for_fp32_and_off_on_cpu():
    """The scope of the numerics change is exactly one call, on CUDA, in fp16."""
    fp32 = _adapter(100, 64, 512)
    fp32.precision = "fp32"
    fp32.torch = types.SimpleNamespace(autocast=lambda *a, **k: "AUTOCAST",
                                       float16="f16")
    assert fp32._autocast() != "AUTOCAST"

    on_cpu = _adapter(100, 64, 512)          # device is cpu in the helper
    on_cpu.precision = "fp16"
    on_cpu.torch = fp32.torch
    assert on_cpu._autocast() != "AUTOCAST"


def test_fp16_on_cuda_enables_autocast():
    adapter = runner.KronosAdapter(
        our_grid=(0.1, 0.5, 0.9), context=512, device="cuda", batch=8,
        sample_count=100, precision="fp16")
    captured = {}

    def fake_autocast(device, dtype=None):
        captured["device"], captured["dtype"] = device, dtype
        return "AUTOCAST"

    adapter.torch = types.SimpleNamespace(autocast=fake_autocast, float16="f16")
    assert adapter._autocast() == "AUTOCAST"
    assert captured == {"device": "cuda", "dtype": "f16"}


def test_precision_and_its_verification_are_recorded_in_grid_json():
    adapter = runner.KronosAdapter(
        our_grid=(0.01, 0.5, 0.99), context=512, device="cuda", batch=8,
        sample_count=100, precision="fp16",
        verification="fp16 verified against fp32 on fold 0")
    note = adapter.grid_note()
    assert note["precision"] == "fp16"
    assert "verified against fp32" in note["precision_verification"]
    assert "fp64" in note["precision_scope"]


def test_an_unverified_fp16_run_says_so_in_grid_json():
    """Silence would read as verified; the artefact has to state the absence."""
    adapter = runner.KronosAdapter(
        our_grid=(0.01, 0.5, 0.99), context=512, device="cuda", batch=8,
        sample_count=100, precision="fp16", verification=None)
    note = adapter.grid_note()
    assert "NOT VERIFIED" in note["precision_verification"]


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
