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
import json
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


# --------------------------------------------------------------------------
# 6. a load-only call must not overwrite a completed run's timing
# --------------------------------------------------------------------------

def _full_record(**over):
    rec = {"n_folds": 4, "n_forecasts": 83450, "partial": False,
           "seconds_per_fold": {"0": 38.7, "1": 37.5, "2": 38.2, "3": 140.7},
           "forecasts_per_fold": {"0": 12650, "1": 12600, "2": 12550, "3": 45650},
           "seconds_steady_state_total": 255.1,
           "context": 512, "device": "cuda", "sample_count": None,
           "precision": None, "paths_per_call": None,
           "checkpoint": "ibm-granite/granite-timeseries-ttm-r2",
           "checkpoint_revision": "abc123", "recorded_at": "2026-09-06T10:00:00Z"}
    rec.update(over)
    return rec


def _load_only_record(**over):
    """What run_model writes when every fold already exists on disk."""
    rec = _full_record(n_folds=0, n_forecasts=0, seconds_per_fold={},
                       forecasts_per_fold={}, seconds_steady_state_total=0,
                       recorded_at="2026-09-06T12:00:00Z")
    rec.update(over)
    return rec


def test_a_load_only_call_does_not_erase_a_completed_record():
    """The TTM regression: 4 folds and 83,450 forecasts became 0 and 0."""
    merged = runner._merge_timing(_full_record(), _load_only_record())
    assert merged["n_folds"] == 4
    assert merged["n_forecasts"] == 83450
    assert merged["seconds_steady_state_total"] == 255.1
    assert merged["seconds_per_fold"]["3"] == 140.7
    assert "executed no folds" in merged["last_invocation_note"]
    assert merged["last_invocation_ran_no_folds_at"] == "2026-09-06T12:00:00Z"


def test_a_load_only_call_on_an_empty_slot_still_records_the_absence():
    merged = runner._merge_timing(None, _load_only_record())
    assert merged["n_folds"] == 0
    assert "last_invocation_note" not in merged


def test_re_running_one_fold_updates_it_and_keeps_the_others():
    partial = _full_record(n_folds=1, n_forecasts=12650,
                           seconds_per_fold={"0": 30.0},
                           forecasts_per_fold={"0": 12650},
                           seconds_steady_state_total=30.0)
    merged = runner._merge_timing(_full_record(), partial)
    assert merged["seconds_per_fold"] == {"0": 30.0, "1": 37.5, "2": 38.2, "3": 140.7}
    assert merged["n_folds"] == 4
    assert merged["n_forecasts"] == 83450
    assert merged["merged_with_prior_folds"] == ["1", "2", "3"]
    assert merged["seconds_steady_state_total"] == pytest.approx(246.4)


def test_a_run_under_different_settings_replaces_rather_than_merges():
    """Per-fold times from another configuration are not the same measurement."""
    other = _full_record(precision="fp16", n_folds=2, n_forecasts=25250,
                         seconds_per_fold={"0": 9.0, "1": 9.1},
                         forecasts_per_fold={"0": 12650, "1": 12600})
    merged = runner._merge_timing(_full_record(precision="fp32"), other)
    assert merged["replaced_prior_record"] is True
    assert "precision" in merged["replaced_because"]
    assert merged["seconds_per_fold"] == {"0": 9.0, "1": 9.1}


def test_a_changed_checkpoint_revision_also_forces_a_replace():
    """Same repo id, moved branch - not the same weights, not the same timing."""
    other = _full_record(checkpoint_revision="def456", n_folds=1,
                         seconds_per_fold={"0": 40.0},
                         forecasts_per_fold={"0": 12650})
    merged = runner._merge_timing(_full_record(checkpoint_revision="abc123"), other)
    assert merged["replaced_prior_record"] is True
    assert "checkpoint_revision" in merged["replaced_because"]


def test_record_timing_on_disk_preserves_the_completed_run(tmp_path):
    """End to end through the file, which is where the loss actually happened."""
    path = tmp_path / "timings.json"
    runner.record_timing(path, "TTM", _full_record())
    runner.record_timing(path, "TTM", _load_only_record())
    written = json.loads(path.read_text())["TTM"]
    assert written["n_folds"] == 4 and written["n_forecasts"] == 83450


def test_other_models_in_the_file_are_untouched(tmp_path):
    path = tmp_path / "timings.json"
    runner.record_timing(path, "Chronos-2", _full_record())
    runner.record_timing(path, "TTM", _load_only_record())
    written = json.loads(path.read_text())
    assert written["Chronos-2"]["n_folds"] == 4
    assert set(written) == {"Chronos-2", "TTM"}


# --------------------------------------------------------------------------
# 7. provenance: which weights, which code
# --------------------------------------------------------------------------

def test_every_adapter_declares_its_checkpoint():
    for cls in (runner.ChronosAdapter, runner.TimesFMAdapter, runner.TTMAdapter,
                runner.KronosAdapter):
        assert cls.checkpoint, f"{cls.name} declares no checkpoint"


def test_timesfm_declares_the_pinned_checkpoint_not_the_library_default():
    assert runner.TimesFMAdapter.checkpoint == runner.TIMESFM_CHECKPOINT
    assert "3.0" not in runner.TimesFMAdapter.checkpoint


def test_a_revision_read_from_the_cache_is_reported_with_its_source(tmp_path,
                                                                    monkeypatch):
    repo = "amazon/chronos-2"
    refs = tmp_path / ("models--" + repo.replace("/", "--")) / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("3f9c1a2b7d4e5f60718293a4b5c6d7e8f9012345\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    sha, source = runner._hf_cache_revision(repo)
    assert sha == "3f9c1a2b7d4e5f60718293a4b5c6d7e8f9012345"
    assert "hf cache" in source


def test_a_missing_revision_is_distinguishable_from_one_nobody_looked_for():
    """A null revision must say which kind of null it is."""
    assert runner._hf_cache_revision("no/such-model")[1] == "not in the local hf cache"
    assert runner._hf_cache_revision(None)[1] == "no checkpoint declared"


def test_kronos_provenance_carries_both_checkpoints_and_the_clone():
    adapter = runner.KronosAdapter(
        our_grid=(0.1, 0.5, 0.9), context=512, device="cpu", batch=8,
        sample_count=100, kronos_path=str(ROOT))
    prov = adapter.provenance()
    assert prov["checkpoint"] == "NeoQuasar/Kronos-small"
    assert prov["tokenizer_checkpoint"] == "NeoQuasar/Kronos-Tokenizer-base"
    assert prov["library"] == "kronos (git clone)"
    # Cloned, not installed, so its version is the commit of the working copy.
    assert prov["library_version"] and len(prov["library_version"]) == 40


def test_the_timing_payload_would_carry_provenance_fields():
    prov = runner.ChronosAdapter((0.1, 0.5, 0.9), 512, "cpu", 32).provenance()
    for field in ("checkpoint", "checkpoint_revision", "library",
                  "library_version"):
        assert field in prov


# --------------------------------------------------------------------------
# 8. merging against a record written before forecasts_per_fold existed
# --------------------------------------------------------------------------

def _legacy_record(**over):
    """A prior written before forecasts_per_fold was added - no row attribution."""
    rec = _full_record()
    rec.pop("forecasts_per_fold")
    rec.update(over)
    return rec


def test_a_legacy_prior_does_not_have_its_total_overwritten_by_one_fold():
    """Re-measuring one fold must not write 12,650 over 83,450."""
    partial = _full_record(n_folds=1, n_forecasts=12650,
                           seconds_per_fold={"0": 30.0},
                           forecasts_per_fold={"0": 12650},
                           seconds_steady_state_total=30.0)
    merged = runner._merge_timing(_legacy_record(), partial)
    assert merged["n_forecasts"] == 83450, "the completed total must stand"
    assert merged["n_folds"] == 4
    assert merged["n_forecasts_from_legacy_prior"] is True
    assert "attributed per fold" in merged["n_forecasts_note"]


def test_a_legacy_prior_still_gains_rows_from_a_genuinely_new_fold():
    legacy = _legacy_record(seconds_per_fold={"0": 38.7, "1": 37.5},
                            n_folds=2, n_forecasts=25250)
    new = _full_record(n_folds=1, n_forecasts=12550,
                       seconds_per_fold={"2": 40.0},
                       forecasts_per_fold={"2": 12550})
    merged = runner._merge_timing(legacy, new)
    assert merged["n_forecasts"] == 25250 + 12550
    assert merged["n_folds"] == 3


def test_a_row_set_that_disagrees_with_the_fold_set_raises():
    """A record that contradicts itself is worse than a failure."""
    prior = _full_record(seconds_per_fold={"0": 1.0, "1": 2.0},
                         forecasts_per_fold={"0": 10}, n_folds=2)
    new = _full_record(seconds_per_fold={"2": 3.0},
                       forecasts_per_fold={"2": 30}, n_folds=1)
    with pytest.raises(ValueError, match="inconsistent"):
        runner._merge_timing(prior, new)


def test_a_different_input_series_forces_a_replace():
    """Returns and prices are different questions, not different speeds."""
    prior = _full_record(input_series="return")
    other = _full_record(input_series="price", n_folds=1,
                         seconds_per_fold={"0": 40.0},
                         forecasts_per_fold={"0": 12650})
    merged = runner._merge_timing(prior, other)
    assert merged["replaced_prior_record"] is True
    assert "input_series" in merged["replaced_because"]


def test_input_series_is_part_of_run_identity():
    assert "input_series" in runner.RUN_IDENTITY_FIELDS


# --------------------------------------------------------------------------
# 9. the checks that would have caught the TimesFM run
# --------------------------------------------------------------------------

TIMESFM_25_CONFIG = {
    "architectures": ["TimesFmModelForPrediction"],
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "horizon_length": 128, "quantile_horizon_length": 1024,
}


def test_the_25_checkpoint_implies_ten_channels_not_nine():
    """The whole bug in one assertion: 9 observed, 10 declared."""
    adapter = _timesfm_adapter()
    assert adapter.expected_channels(TIMESFM_25_CONFIG) == 10


def test_a_three_point_zero_style_config_implies_one_channel_per_level():
    config = {"architectures": ["TimesFM3Torch"],
              "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    assert _timesfm_adapter().expected_channels(config) == 9


def test_a_channel_count_disagreeing_with_the_checkpoint_raises(monkeypatch):
    adapter = _timesfm_adapter()
    monkeypatch.setattr(runner, "_checkpoint_config", lambda repo: TIMESFM_25_CONFIG)
    with pytest.raises(SystemExit, match="9 output channels"):
        adapter.assert_channels_match_checkpoint(9)
    adapter.assert_channels_match_checkpoint(10)        # must not raise


def test_a_missing_checkpoint_config_skips_rather_than_guesses(monkeypatch):
    adapter = _timesfm_adapter()
    monkeypatch.setattr(runner, "_checkpoint_config", lambda repo: None)
    adapter.assert_channels_match_checkpoint(9)         # must not raise


def test_chronos_expects_its_own_twenty_one_levels():
    adapter = runner.ChronosAdapter((0.1, 0.5, 0.9), 512, "cpu", 32)
    config = {"chronos_config": {"quantiles": [0.01] * 21}}
    assert adapter.expected_channels(config) == 21
    assert adapter.expected_channels({}) is None


class _FakeModule:
    """Stands in for torch.nn.Module so the recorder can be exercised."""

    def load_state_dict(self, state_dict, *a, **k):
        return self._result


def _fake_torch(result):
    module_cls = type("Module", (), {"load_state_dict": _FakeModule.load_state_dict,
                                     "_result": result})
    return types.SimpleNamespace(nn=types.SimpleNamespace(Module=module_cls))


def test_a_clean_load_passes_and_restores_the_method():
    ok = types.SimpleNamespace(missing_keys=[], unexpected_keys=[])
    torch = _fake_torch(ok)
    original = torch.nn.Module.load_state_dict
    rec = runner._StateDictLoadRecorder(torch)
    with rec:
        torch.nn.Module().load_state_dict({})
    assert torch.nn.Module.load_state_dict is original
    rec.assert_clean("Chronos-2", "amazon/chronos-2")    # must not raise


def test_missing_or_unexpected_keys_fail_the_run():
    bad = types.SimpleNamespace(missing_keys=["blk.0.w"], unexpected_keys=["enc.b"])
    torch = _fake_torch(bad)
    rec = runner._StateDictLoadRecorder(torch)
    with rec:
        torch.nn.Module().load_state_dict({})
    with pytest.raises(SystemExit, match="did not load cleanly"):
        rec.assert_clean("TimesFM-2.5", "google/timesfm-2.5-200m-pytorch")


def test_the_recorder_restores_the_method_even_when_the_load_raises():
    torch = _fake_torch(types.SimpleNamespace(missing_keys=[], unexpected_keys=[]))
    original = torch.nn.Module.load_state_dict
    with pytest.raises(RuntimeError):
        with runner._StateDictLoadRecorder(torch):
            raise RuntimeError("download failed")
    assert torch.nn.Module.load_state_dict is original


# --------------------------------------------------------------------------
# 10. the scale guardrail, arm-aware
# --------------------------------------------------------------------------

def _levels_with_sigma(sigma, n=64):
    """A frozen-grid block whose 80% interval implies the given sigma."""
    grid = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
    half = sigma * runner.Q90_Q10_TO_SIGMA / 2
    row = np.zeros(len(grid))
    row[grid.index(0.1)] = -half
    row[grid.index(0.9)] = +half
    return np.tile(row, (n, 1)), grid


def _band_adapter(arm="return"):
    adapter = runner.ChronosAdapter(
        (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99), 512, "cpu", 32)
    adapter.arm = arm
    return adapter


def test_a_realistic_return_sigma_passes():
    levels, _ = _levels_with_sigma(0.0157)              # Chronos-2, measured
    _band_adapter().assert_sigma_in_band(levels)        # must not raise


def test_the_timesfm_scale_would_have_been_rejected():
    """0.0906/day = 144% annualised, the run that reached the parquet."""
    levels, _ = _levels_with_sigma(0.0906)
    with pytest.raises(SystemExit, match="outside the return-arm band"):
        _band_adapter().assert_sigma_in_band(levels)


def test_an_implausibly_tight_band_is_also_rejected():
    levels, _ = _levels_with_sigma(0.0001)
    with pytest.raises(SystemExit, match="outside the return-arm band"):
        _band_adapter().assert_sigma_in_band(levels)


def test_the_volatility_arm_has_its_own_band():
    """A single global band would never fire on the volatility arm."""
    ret_low, ret_high = runner.SIGMA_BANDS["return"]
    vol_low, vol_high = runner.SIGMA_BANDS["volatility"]
    assert (vol_low, vol_high) != (ret_low, ret_high)
    # A spread plausible for the 21-session RV target but far too tight to be a
    # daily return: passes on the volatility arm, fails on the return arm.
    levels, _ = _levels_with_sigma(0.001)
    _band_adapter("volatility").assert_sigma_in_band(levels)
    with pytest.raises(SystemExit):
        _band_adapter("return").assert_sigma_in_band(levels)


def test_a_point_only_model_skips_the_band_rather_than_failing():
    """TTM has no 80% interval; absence is not a scale error."""
    levels, _ = _levels_with_sigma(0.02)
    levels[:] = np.nan
    _band_adapter().assert_sigma_in_band(levels)        # must not raise


def test_implied_sigma_uses_the_normal_80_percent_span():
    assert runner.Q90_Q10_TO_SIGMA == pytest.approx(2.5631, abs=1e-4)
    assert runner.implied_daily_sigma(np.array([-1.28155]),
                                      np.array([1.28155])) == pytest.approx(1.0, abs=1e-4)


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
