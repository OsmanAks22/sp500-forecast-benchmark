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

    with adapter._keep_sample_axis():
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
        with adapter._keep_sample_axis():
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
