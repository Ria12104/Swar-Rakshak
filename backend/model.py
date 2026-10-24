"""Model adapters for the Phase 3 backend scoring contract."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from src.audio import make_windows
from src.features import LogMelFrontEnd
from src.inference import load_cnn_checkpoint
from src.models.cnn import CNNBaseline


class ScoringModel(Protocol):
    """Stable model interface used by REST and WebSocket endpoints."""

    model_version: str

    def load(self) -> "ScoringModel":
        ...

    def score(self, waveform_16k: np.ndarray) -> float:
        """
        Return the mean synthetic logit for a 16 kHz waveform.

        A future WavLM adapter can implement this same interface without
        changing the API layer.
        """
        ...


class CNNScoringModel:
    """Phase 2 CNN adapter that exposes raw synthetic logits."""

    def __init__(self, checkpoint_path: str | Path, device: torch.device | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_version = self.checkpoint_path.parent.name or self.checkpoint_path.stem
        self.model: CNNBaseline | None = None
        self.frontend: LogMelFrontEnd | None = None

    def load(self) -> "CNNScoringModel":
        self.model = load_cnn_checkpoint(self.checkpoint_path, self.device)
        self.frontend = LogMelFrontEnd().to(self.device)
        return self

    def score(self, waveform_16k: np.ndarray) -> float:
        if self.model is None or self.frontend is None:
            raise RuntimeError("model is not loaded")
        windows, _ = make_windows(waveform_16k, win_s=4.0, hop_s=2.0, max_s=10.0)
        batch = torch.from_numpy(np.stack(windows)).to(self.device)
        with torch.inference_mode():
            logits: Tensor = self.model(self.frontend(batch))
        return float(logits.mean().item())
