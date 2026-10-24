"""Shared audio preprocessing for training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf

# All preprocessing constants live here so training, evaluation, and inference share them.
TARGET_SR = 16_000
TARGET_RMS_DBFS = -20.0       # Fixed RMS target after preprocessing.
MAX_GAIN_DB = 20.0            # Prevent near-silent audio from being amplified excessively.
NEAR_SILENT_DBFS = -60.0      # Below this, leave gain at 1.0 and warn.
SILENCE_THRESHOLD_DBFS = -50.0  # Fixed frame-RMS threshold used by silence_stats().
SILENCE_FRAME_S = 0.020       # 20 ms energy frames for leading/trailing silence.
TRIM_RELATIVE_DB = 35.0        # Silence is this far below the loudest frame RMS.
TRIM_MARGIN_S = 0.050          # Keep this much audio around detected speech.
TRIM_MIN_DURATION_S = 1.0      # Never trim a clip below this duration.
CLIP_THRESHOLD = 0.999         # Per-sample clipping threshold.
CLIP_FRACTION_THRESHOLD = 0.001  # Warn only when more than 0.1% samples clip.

ArrayLike = Union[np.ndarray, list[float], tuple[float, ...]]


@dataclass(frozen=True)
class PreprocessedAudio:
    """Normalized mono waveform and metadata returned by preprocess()."""

    waveform: np.ndarray
    original_duration_s: float
    original_sr: int
    warnings: list[str]


def _dbfs(rms: float) -> float:
    if rms <= 0.0 or not np.isfinite(rms):
        return float("-inf")
    return float(20.0 * np.log10(rms))


def _as_mono(array: np.ndarray) -> np.ndarray:
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError("audio array must be 1D or 2D audio")
    # soundfile uses (samples, channels); librosa with mono=False uses
    # (channels, samples). A small leading dimension is the reliable signal
    # for the latter layout in ordinary audio arrays.
    if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
        return np.mean(array, axis=0)
    return np.mean(array, axis=1)


def _load_audio(path_or_array: Union[str, Path, ArrayLike], sr: int | None) -> tuple[np.ndarray, int]:
    if isinstance(path_or_array, (str, Path)):
        waveform, file_sr = sf.read(str(path_or_array), dtype="float32", always_2d=True)
        return _as_mono(np.asarray(waveform, dtype=np.float32)), int(file_sr)
    if sr is None:
        sr = TARGET_SR
    return _as_mono(np.asarray(path_or_array, dtype=np.float32)), int(sr)


def load_mono_resampled(path_or_array: Union[str, Path, ArrayLike], sr: int | None = None) -> tuple[np.ndarray, int]:
    """Return sanitized mono audio resampled to TARGET_SR, before normalization."""
    waveform, original_sr = _load_audio(path_or_array, sr)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if waveform.size and original_sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=TARGET_SR).astype(np.float32)
    return waveform, original_sr


def _trim_silence(waveform: np.ndarray) -> np.ndarray:
    """Trim relative silence while preserving a margin and a one-second minimum."""
    minimum_samples = int(round(TRIM_MIN_DURATION_S * TARGET_SR))
    if waveform.size < minimum_samples:
        return waveform

    frame = max(1, int(round(SILENCE_FRAME_S * TARGET_SR)))
    frame_count = int(np.ceil(len(waveform) / frame))
    frame_rms = np.array(
        [
            np.sqrt(
                np.mean(
                    np.square(waveform[i * frame : min((i + 1) * frame, len(waveform))], dtype=np.float64)
                )
            )
            for i in range(frame_count)
        ],
        dtype=np.float64,
    )
    loudest = float(np.max(frame_rms)) if frame_rms.size else 0.0
    if loudest <= 0.0 or not np.isfinite(loudest):
        return waveform

    threshold = loudest * (10.0 ** (-TRIM_RELATIVE_DB / 20.0))
    non_silent = frame_rms >= threshold
    if not np.any(non_silent):
        return waveform

    first = int(np.flatnonzero(non_silent)[0]) * frame
    last = min(len(waveform), (int(np.flatnonzero(non_silent)[-1]) + 1) * frame)
    margin = int(round(TRIM_MARGIN_S * TARGET_SR))
    start = max(0, first - margin)
    end = min(len(waveform), last + margin)
    if end - start < minimum_samples:
        return waveform
    return waveform[start:end]


def preprocess(
    path_or_array: Union[str, Path, ArrayLike],
    sr: int | None = None,
    trim_silence: bool = False,
) -> PreprocessedAudio:
    """Load, sanitize, resample, mono-mix, and RMS-normalize an audio clip."""
    waveform, original_sr = load_mono_resampled(path_or_array, sr)
    original_duration_s = float(len(waveform) / original_sr) if original_sr else 0.0
    warnings: list[str] = []

    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if waveform.size == 0:
        warnings.append("empty")
        return PreprocessedAudio(np.zeros(0, dtype=np.float32), original_duration_s, original_sr, warnings)

    if float(np.mean(np.abs(waveform))) <= np.finfo(np.float32).eps:
        warnings.append("near_silent")

    rms_before_resample = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if _dbfs(rms_before_resample) <= NEAR_SILENT_DBFS and "near_silent" not in warnings:
        warnings.append("near_silent")

    clipped_fraction = float(np.mean(np.abs(waveform) >= CLIP_THRESHOLD))
    if clipped_fraction > CLIP_FRACTION_THRESHOLD:
        warnings.append("clipped")

    if trim_silence:
        waveform = _trim_silence(waveform)

    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    rms_dbfs = _dbfs(rms)
    if rms > 0.0 and rms_dbfs >= NEAR_SILENT_DBFS:
        requested_gain_db = TARGET_RMS_DBFS - rms_dbfs
        gain_db = min(requested_gain_db, MAX_GAIN_DB)
        waveform = waveform * np.float32(10.0 ** (gain_db / 20.0))

    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return PreprocessedAudio(waveform, original_duration_s, original_sr, warnings)


def make_windows(
    wave: np.ndarray,
    win_s: float = 4.0,
    hop_s: float = 2.0,
    max_s: float = 10.0,
) -> tuple[list[np.ndarray], list[str]]:
    """Return fixed-size windows and preprocessing warnings for a waveform."""
    wave = np.nan_to_num(np.asarray(wave, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    win = max(1, int(round(win_s * TARGET_SR)))
    hop = max(1, int(round(hop_s * TARGET_SR)))
    max_samples = max(win, int(round(max_s * TARGET_SR)))
    warnings: list[str] = []
    duration_s = len(wave) / TARGET_SR

    if duration_s < 1.0:
        warnings.append("too_short")
    if len(wave) < win:
        warnings.append("padded")
        if len(wave) == 0:
            wave = np.zeros(win, dtype=np.float32)
        else:
            wave = np.tile(wave, int(np.ceil(win / len(wave))))[:win]
        return [wave.astype(np.float32, copy=False)], warnings

    if len(wave) > max_samples:
        warnings.append("truncated")
        wave = wave[:max_samples]

    starts = range(0, max(1, len(wave) - win + 1), hop)
    windows = [wave[start : start + win].astype(np.float32, copy=False) for start in starts if len(wave[start : start + win]) == win]
    if not windows:
        windows = [wave[:win].astype(np.float32, copy=False)]
    return windows, warnings


def random_crop(wave: np.ndarray, win_s: float = 4.0) -> np.ndarray:
    """Take a random fixed-size crop, repeat-tiling clips shorter than the crop."""
    wave = np.nan_to_num(np.asarray(wave, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    win = max(1, int(round(win_s * TARGET_SR)))
    if len(wave) == 0:
        return np.zeros(win, dtype=np.float32)
    if len(wave) < win:
        return np.tile(wave, int(np.ceil(win / len(wave))))[:win].astype(np.float32, copy=False)
    start = int(np.random.randint(0, len(wave) - win + 1))
    return wave[start : start + win].astype(np.float32, copy=True)


def silence_stats(wave: np.ndarray) -> tuple[float, float, float]:
    """Return RMS dBFS and leading/trailing silence durations.

    Silence is defined by 20 ms frame RMS below SILENCE_THRESHOLD_DBFS.
    """
    wave = np.nan_to_num(np.asarray(wave, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if wave.size == 0:
        return float("-inf"), 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))
    frame = max(1, int(round(SILENCE_FRAME_S * TARGET_SR)))
    threshold = 10.0 ** (SILENCE_THRESHOLD_DBFS / 20.0)
    frame_count = int(np.ceil(len(wave) / frame))
    frame_rms = np.array(
        [np.sqrt(np.mean(np.square(wave[i * frame : min((i + 1) * frame, len(wave))], dtype=np.float64))) for i in range(frame_count)],
        dtype=np.float64,
    )
    silent = frame_rms < threshold
    leading_frames = 0
    while leading_frames < len(silent) and silent[leading_frames]:
        leading_frames += 1
    trailing_frames = 0
    while trailing_frames < len(silent) and silent[len(silent) - 1 - trailing_frames]:
        trailing_frames += 1
    leading_s = min(float(leading_frames * frame / TARGET_SR), len(wave) / TARGET_SR)
    trailing_s = min(float(trailing_frames * frame / TARGET_SR), len(wave) / TARGET_SR)
    return _dbfs(rms), leading_s, trailing_s


def peak_dbfs(wave: np.ndarray) -> float:
    """Return peak absolute amplitude in dBFS for an already prepared waveform."""
    wave = np.nan_to_num(np.asarray(wave, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if wave.size == 0:
        return float("-inf")
    return _dbfs(float(np.max(np.abs(wave))))
