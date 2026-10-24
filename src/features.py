"""GPU-friendly log-Mel features shared by training and inference."""

from __future__ import annotations

import torch
import torchaudio
from torch import Tensor, nn

from .audio import TARGET_SR

N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
LOG_EPS = 1e-6


class LogMelFrontEnd(nn.Module):
    """Compute per-utterance normalized log-Mel spectrograms."""

    def __init__(
        self,
        sample_rate: int = TARGET_SR,
        n_fft: int = N_FFT,
        win_length: int = WIN_LENGTH,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            center=True,
        )

    def forward(self, waveforms: Tensor) -> Tensor:
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        if self.mel.spectrogram.window.device != waveforms.device:
            self.to(waveforms.device)
        log_mel = torch.log(self.mel(waveforms.float()) + LOG_EPS)
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(LOG_EPS)
        return ((log_mel - mean) / std).unsqueeze(1)


def extract_log_mel(waveforms: Tensor, device: torch.device | str | None = None) -> Tensor:
    """Convenience wrapper returning [batch, 1, 80, frames]."""
    if device is not None:
        waveforms = waveforms.to(device)
    frontend = LogMelFrontEnd().to(waveforms.device)
    return frontend(waveforms)
