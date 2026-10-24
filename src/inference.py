"""Model loading and window-averaged clip scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .audio import make_windows, preprocess
from .features import LogMelFrontEnd
from .models.cnn import CNNBaseline


def load_cnn_checkpoint(checkpoint_path: str | Path, device: torch.device) -> CNNBaseline:
    """Load a Phase 2 CNN checkpoint without requiring training code."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CNNBaseline().to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def _score_windows(
    windows: list[np.ndarray],
    model: CNNBaseline,
    frontend: LogMelFrontEnd,
    device: torch.device,
    batch_size: int,
) -> float:
    logits: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(np.stack(windows[start : start + batch_size])).to(device)
            logits.append(model(frontend(batch)))
    return float(torch.cat(logits).mean().item())


def score_manifest_dataframe(
    manifest_df: pd.DataFrame,
    model: CNNBaseline,
    device: torch.device,
    batch_size: int = 16,
    trim_silence: bool = False,
) -> pd.DataFrame:
    """Score each clip by the mean logit over its 4 s windows."""
    frontend = LogMelFrontEnd().to(device)
    rows: list[dict[str, object]] = []
    for _, row in manifest_df.iterrows():
        processed = preprocess(str(row["path"]), trim_silence=trim_silence)
        windows, _ = make_windows(processed.waveform)
        score = _score_windows(windows, model, frontend, device, batch_size)
        rows.append(
            {
                "path": str(row["path"]),
                "label": int(row["label"]),
                "score": score,
                "n_windows": len(windows),
            }
        )
    return pd.DataFrame(rows, columns=["path", "label", "score", "n_windows"])
