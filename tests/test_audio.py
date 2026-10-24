from __future__ import annotations

import numpy as np
import pytest

from src.audio import TARGET_RMS_DBFS, TARGET_SR, make_windows, preprocess, silence_stats


def assert_valid_audio(result) -> None:
    assert result.waveform.dtype == np.float32
    assert result.waveform.ndim == 1
    assert np.isfinite(result.waveform).all()
    assert result.original_sr > 0


@pytest.mark.parametrize("duration_s", [0.5, 2.0, 10.0, 15.0])
def test_synthetic_signals_are_fixed_size_and_finite(duration_s: float) -> None:
    source_sr = 16_000
    t = np.arange(int(duration_s * source_sr), dtype=np.float32) / source_sr
    result = preprocess(0.2 * np.sin(2 * np.pi * 220 * t), sr=source_sr)
    assert_valid_audio(result)
    windows, warnings = make_windows(result.waveform)
    assert all(window.shape == (4 * TARGET_SR,) for window in windows)
    assert all(window.dtype == np.float32 for window in windows)
    assert np.isfinite(np.concatenate(windows)).all()
    if duration_s < 4:
        assert "padded" in warnings
    if duration_s < 1:
        assert "too_short" in warnings
    if duration_s > 10:
        assert "truncated" in warnings


def test_empty_array_does_not_crash() -> None:
    result = preprocess(np.array([], dtype=np.float32), sr=16_000)
    assert_valid_audio(result)
    assert result.waveform.size == 0
    assert "empty" in result.warnings
    windows, warnings = make_windows(result.waveform)
    assert warnings == ["too_short", "padded"]
    assert windows[0].shape == (4 * TARGET_SR,)


def test_pure_silence_warns_without_divide_by_zero() -> None:
    result = preprocess(np.zeros(2 * TARGET_SR, dtype=np.float32), sr=TARGET_SR)
    assert_valid_audio(result)
    assert "near_silent" in result.warnings
    assert np.all(result.waveform == 0)
    rms_dbfs, leading, trailing = silence_stats(result.waveform)
    assert np.isneginf(rms_dbfs)
    assert leading == pytest.approx(2.0, abs=1 / TARGET_SR)
    assert trailing == pytest.approx(2.0, abs=1 / TARGET_SR)


def test_resampling_and_stereo_mixing() -> None:
    source_sr = 8_000
    t = np.arange(source_sr, dtype=np.float32) / source_sr
    stereo = np.column_stack((np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 500 * t)))
    result = preprocess(stereo, sr=source_sr)
    assert_valid_audio(result)
    assert result.waveform.shape == (TARGET_SR,)
    assert result.original_sr == source_sr

    source_sr = 44_100
    t = np.arange(source_sr, dtype=np.float32) / source_sr
    result = preprocess(np.sin(2 * np.pi * 440 * t), sr=source_sr)
    assert_valid_audio(result)
    assert result.waveform.shape == (TARGET_SR,)


def test_asymmetric_stereo_supports_soundfile_and_librosa_layouts() -> None:
    source_sr = 8_000
    t = np.arange(source_sr, dtype=np.float32) / source_sr
    left = np.sin(2 * np.pi * 300 * t)
    right = np.sin(2 * np.pi * 700 * t)
    soundfile_layout = np.column_stack((left, right))
    librosa_layout = np.vstack((left, right))

    soundfile_result = preprocess(soundfile_layout, sr=source_sr)
    librosa_result = preprocess(librosa_layout, sr=source_sr)
    assert np.allclose(soundfile_result.waveform, librosa_result.waveform, atol=2e-4)


def test_preprocess_equalizes_loudness_but_manifest_features_can_remain_different() -> None:
    source_sr = TARGET_SR
    t = np.arange(2 * source_sr, dtype=np.float32) / source_sr
    quiet = 10 ** (-30 / 20) * np.sin(2 * np.pi * 220 * t)
    loud = 10 ** (-10 / 20) * np.sin(2 * np.pi * 220 * t)
    quiet_processed = preprocess(quiet, sr=source_sr)
    loud_processed = preprocess(loud, sr=source_sr)
    quiet_rms, _, _ = silence_stats(quiet_processed.waveform)
    loud_rms, _, _ = silence_stats(loud_processed.waveform)
    assert quiet_rms == pytest.approx(TARGET_RMS_DBFS, abs=0.1)
    assert loud_rms == pytest.approx(TARGET_RMS_DBFS, abs=0.1)


def test_nan_and_clipping_are_sanitized_and_warned() -> None:
    wave = np.array([np.nan, np.inf, -np.inf, 1.0, -1.0, 0.5], dtype=np.float32)
    result = preprocess(wave, sr=TARGET_SR)
    assert_valid_audio(result)
    assert "clipped" in result.warnings


def test_trim_silence_keeps_margin_and_never_goes_below_one_second() -> None:
    sr = TARGET_SR
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    t = np.arange(int(1.2 * sr), dtype=np.float32) / sr
    tone = 0.2 * np.sin(2 * np.pi * 220 * t)
    raw = np.concatenate((silence, tone, silence))
    result = preprocess(raw, sr=sr, trim_silence=True)
    assert result.waveform.shape[0] / sr >= 1.0
    assert result.waveform.shape[0] / sr < raw.shape[0] / sr

    short = preprocess(tone[: int(0.8 * sr)], sr=sr, trim_silence=True)
    assert short.waveform.shape[0] == int(0.8 * sr)


def test_peak_normalized_audio_does_not_warn_from_a_few_peak_samples() -> None:
    wave = np.zeros(2_000, dtype=np.float32)
    wave[1_000] = 1.0
    result = preprocess(wave, sr=TARGET_SR)
    assert "clipped" not in result.warnings
