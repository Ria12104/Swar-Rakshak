"""Upload decoding and metadata helpers for the Phase 3 backend."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


class AudioDecodeError(ValueError):
    """Raised when an uploaded audio file cannot be decoded."""


@dataclass(frozen=True)
class DecodedAudio:
    waveform: np.ndarray
    sample_rate: int
    channels: int
    format_name: str
    duration_s: float

    @property
    def format_label(self) -> str:
        channel_label = "mono" if self.channels == 1 else "stereo" if self.channels == 2 else f"{self.channels} channels"
        return f"{self.format_name}, {self.sample_rate} Hz, {channel_label}"


def _format_name(filename: str, detected: str | None = None) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix or (detected or "audio").lower()


def _decode_soundfile(data: bytes, filename: str) -> DecodedAudio:
    buffer = BytesIO(data)
    info = sf.info(buffer)
    buffer.seek(0)
    waveform, sample_rate = sf.read(buffer, dtype="float32", always_2d=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    channels = int(waveform.shape[1]) if waveform.ndim == 2 else 1
    mono_or_channels = waveform[:, 0] if channels == 1 else waveform
    return DecodedAudio(
        waveform=mono_or_channels,
        sample_rate=int(sample_rate),
        channels=channels,
        format_name=_format_name(filename, str(info.format)),
        duration_s=float(len(waveform) / sample_rate) if sample_rate else 0.0,
    )


def _ffprobe(data: bytes) -> tuple[int, int, str | None]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name",
        "-of",
        "json",
        "-i",
        "pipe:0",
    ]
    result = subprocess.run(command, input=data, capture_output=True, check=False)
    if result.returncode != 0:
        raise AudioDecodeError("corrupt or unreadable audio")
    try:
        stream = json.loads(result.stdout.decode("utf-8"))["streams"][0]
        return int(stream["sample_rate"]), int(stream["channels"]), str(stream.get("codec_name") or "audio")
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioDecodeError("corrupt or unreadable audio") from exc


def _decode_ffmpeg(data: bytes, filename: str) -> DecodedAudio:
    sample_rate, channels, codec_name = _ffprobe(data)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "pipe:1",
    ]
    result = subprocess.run(command, input=data, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        raise AudioDecodeError("corrupt or unreadable audio")
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if channels > 1:
        samples = samples[: (len(samples) // channels) * channels].reshape(-1, channels)
    return DecodedAudio(
        waveform=samples.astype(np.float32, copy=False),
        sample_rate=sample_rate,
        channels=channels,
        format_name=_format_name(filename, codec_name),
        duration_s=float(len(samples) / sample_rate) if sample_rate else 0.0,
    )


def decode_audio(data: bytes, filename: str) -> DecodedAudio:
    """Decode an upload while preserving original rate/channel metadata."""
    if not data:
        raise AudioDecodeError("empty audio upload")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise AudioDecodeError("unsupported audio format; use wav, flac, mp3, or ogg")
    try:
        return _decode_soundfile(data, filename)
    except Exception as soundfile_error:
        if suffix not in {".mp3", ".ogg"}:
            raise AudioDecodeError("corrupt or unreadable audio") from soundfile_error
        try:
            return _decode_ffmpeg(data, filename)
        except (AudioDecodeError, FileNotFoundError, OSError) as ffmpeg_error:
            raise AudioDecodeError("corrupt or unreadable audio; ffmpeg is required for this format") from ffmpeg_error
