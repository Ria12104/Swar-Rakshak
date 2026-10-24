"""Synthetic smoke demo for the shared Phase 1 audio path."""

from __future__ import annotations

import numpy as np

from src.audio import TARGET_SR, make_windows, preprocess


for duration_s in (2.0, 4.0, 10.0):
    t = np.arange(int(duration_s * TARGET_SR), dtype=np.float32) / TARGET_SR
    raw = 0.2 * np.sin(2 * np.pi * 220 * t)
    processed = preprocess(raw, sr=TARGET_SR)
    windows, window_warnings = make_windows(processed.waveform)
    print(
        f"duration={duration_s:.1f}s "
        f"waveform_shape={processed.waveform.shape} "
        f"windows={[window.shape for window in windows]} "
        f"preprocess_warnings={processed.warnings} "
        f"window_warnings={window_warnings}"
    )
