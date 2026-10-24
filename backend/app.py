"""FastAPI REST and 500 ms WebSocket backend for Swar Rakshak."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.audio import TARGET_SR, preprocess

from .audio import AudioDecodeError, DecodedAudio, decode_audio
from .model import CNNScoringModel, ScoringModel


DEFAULT_MODEL_PATH = "runs/smoke_500/best.pt"
DEFAULT_UNCERTAIN_LOW = 0.40
DEFAULT_SYNTHETIC_HIGH = 0.70
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MIN_DURATION_S = 1.0
MAX_DURATION_S = 30.0
STREAM_HOP_S = 0.5
STREAM_WINDOW_S = 4.0


@dataclass(frozen=True)
class Thresholds:
    uncertain_low: float
    synthetic_high: float
    placeholder: bool


@dataclass(frozen=True)
class BackendSettings:
    model_path: Path
    thresholds: Thresholds
    trim_silence: bool
    max_upload_bytes: int
    allowed_origins: tuple[str, ...]
    logs_dir: Path

    @classmethod
    def from_env(cls) -> "BackendSettings":
        model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
        thresholds = _load_thresholds(model_path)
        extra_origins = tuple(
            origin.strip()
            for origin in os.getenv("EXTRA_ORIGINS", "").split(",")
            if origin.strip()
        )
        origins = tuple(dict.fromkeys(("http://localhost:5173", "http://127.0.0.1:5173", *extra_origins)))
        return cls(
            model_path=model_path,
            thresholds=thresholds,
            trim_silence=os.getenv("TRIM_SILENCE", "false").lower() in {"1", "true", "yes"},
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(MAX_UPLOAD_BYTES))),
            allowed_origins=origins,
            logs_dir=Path(os.getenv("LOGS_DIR", "logs")),
        )


def _parse_threshold(value: object, default: float, assume_logit: bool = False) -> float:
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return default
    if assume_logit or numeric < 0.0 or numeric > 1.0:
        numeric = 1.0 / (1.0 + math.exp(-numeric))
    return float(np.clip(numeric, 0.0, 1.0))


def _load_thresholds(model_path: Path) -> Thresholds:
    env_low = os.getenv("UNCERTAIN_LOW")
    env_high = os.getenv("SYNTHETIC_HIGH")
    source: dict[str, object] | None = None
    threshold_path = os.getenv("THRESHOLDS_PATH")
    candidate = Path(threshold_path) if threshold_path else model_path.parent / "thresholds.json"
    if env_low is None or env_high is None:
        if candidate.exists():
            try:
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    source = loaded
            except (OSError, json.JSONDecodeError):
                source = None
    low_value = env_low if env_low is not None else (source or {}).get("uncertain_low")
    high_value = env_high if env_high is not None else (source or {}).get("synthetic_high")
    low_is_logit = False
    high_is_logit = False
    if source is not None:
        # Also understand the Phase 2 choose_threshold() names. Those files
        # contain logits, so _parse_threshold() applies sigmoid before the
        # REST/WebSocket probability contract uses them.
        if low_value is None and "eer_threshold" in source:
            low_value = source["eer_threshold"]
            low_is_logit = True
        if high_value is None and "fpr_5pct_threshold" in source:
            high_value = source["fpr_5pct_threshold"]
            high_is_logit = True
    placeholder = low_value is None or high_value is None
    return Thresholds(
        uncertain_low=_parse_threshold(low_value, DEFAULT_UNCERTAIN_LOW, low_is_logit),
        synthetic_high=_parse_threshold(high_value, DEFAULT_SYNTHETIC_HIGH, high_is_logit),
        placeholder=placeholder,
    )


def verdict_for_score(score: float, thresholds: Thresholds) -> str:
    """Apply the exact genuine/uncertain/synthetic boundary rule."""
    if score < thresholds.uncertain_low:
        return "genuine"
    if score >= thresholds.synthetic_high:
        return "synthetic"
    return "uncertain"


def _model_version(model: ScoringModel) -> str:
    return str(getattr(model, "model_version", "model"))


def _warning_list(decoded: DecodedAudio, processed_warnings: list[str]) -> list[str]:
    warnings = list(dict.fromkeys(processed_warnings))
    if decoded.sample_rate != TARGET_SR:
        warnings.append("resampled")
    if decoded.duration_s < 2.0:
        warnings.append("too_short")
    if "near_silent" in warnings and "silent" not in warnings:
        warnings.append("silent")
    return list(dict.fromkeys(warnings))


def _score_window(model: ScoringModel, waveform: np.ndarray) -> float:
    return float(np.clip(float(model.score(waveform.astype(np.float32, copy=False))), 0.0, 1.0))


def _bin_scores(model: ScoringModel, waveform_16k: np.ndarray, duration_s: float) -> list[float]:
    """Return one score per 0.5 s bin, using 4 s/0.5 s sliding windows."""
    waveform = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
    bin_count = max(1, int(math.ceil(duration_s / STREAM_HOP_S - 1e-9)))
    window_samples = int(round(STREAM_WINDOW_S * TARGET_SR))
    hop_samples = int(round(STREAM_HOP_S * TARGET_SR))
    if len(waveform) <= window_samples:
        score = _score_window(model, waveform)
        return [score] * bin_count

    starts = list(range(0, len(waveform) - window_samples + 1, hop_samples))
    final_start = len(waveform) - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    window_scores = [
        _score_window(model, waveform[start : start + window_samples])
        for start in starts
    ]
    scores: list[float] = []
    for index in range(bin_count):
        bin_start = index * hop_samples
        bin_end = min(len(waveform), bin_start + hop_samples)
        covering = [
            window_scores[window_index]
            for window_index, start in enumerate(starts)
            if start < bin_end and start + window_samples > bin_start
        ]
        if not covering:
            nearest = min(range(len(starts)), key=lambda window_index: abs(starts[window_index] - bin_start))
            covering = [window_scores[nearest]]
        scores.append(float(np.mean(covering)))
    return scores


def _stream_bin_scores(score: float, seconds: float) -> list[float]:
    count = max(1, int(math.floor(min(STREAM_WINDOW_S, seconds) / STREAM_HOP_S + 1e-9)))
    return [score] * count


def _stream_log_path(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = logs_dir / f"stream_{timestamp}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["seq", "t", "inter_update_ms", "processing_ms", "buffer_s"])
    return path


def _append_stream_log(path: Path, values: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(values)


def _decode_upload(data: bytes, filename: str, max_bytes: int) -> DecodedAudio:
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"audio upload exceeds {max_bytes // (1024 * 1024)} MB")
    try:
        return decode_audio(data, filename)
    except AudioDecodeError as exc:
        message = str(exc)
        status = 415 if message.startswith("unsupported") else 400
        raise HTTPException(status_code=status, detail=message) from exc


def _validate_duration(duration_s: float) -> None:
    if duration_s <= 0.0:
        raise HTTPException(status_code=400, detail="audio is empty")
    if duration_s < MIN_DURATION_S:
        raise HTTPException(status_code=400, detail="audio duration must be at least 1 second")
    if duration_s > MAX_DURATION_S:
        raise HTTPException(status_code=400, detail="audio duration must not exceed 30 seconds")


def create_app(model: ScoringModel | None = None, settings: BackendSettings | None = None) -> FastAPI:
    """Create an app; tests may inject a tiny random-weight scoring model."""
    resolved_settings = settings or BackendSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.model is None:
            adapter = CNNScoringModel(resolved_settings.model_path)
            app.state.model = adapter.load()
        yield

    app = FastAPI(title="Swar Rakshak Backend", lifespan=lifespan)
    app.state.model = model
    app.state.settings = resolved_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
    )

    def current_model() -> ScoringModel:
        active_model = app.state.model
        if active_model is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return active_model

    @app.get("/health")
    async def health() -> dict[str, object]:
        active_model = current_model()
        return {
            "status": "ok",
            "model_version": _model_version(active_model),
            "device": str(getattr(active_model, "device", "cpu")),
            "thresholds": {
                "uncertain_low": resolved_settings.thresholds.uncertain_low,
                "synthetic_high": resolved_settings.thresholds.synthetic_high,
            },
            "thresholds_placeholder": resolved_settings.thresholds.placeholder,
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, object]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="uploaded file needs a filename")
        data = await file.read(resolved_settings.max_upload_bytes + 1)
        decoded = _decode_upload(data, file.filename, resolved_settings.max_upload_bytes)
        _validate_duration(decoded.duration_s)
        processed = preprocess(decoded.waveform, sr=decoded.sample_rate, trim_silence=resolved_settings.trim_silence)
        active_model = current_model()
        bins = _bin_scores(active_model, processed.waveform, decoded.duration_s)
        score = float(np.mean(bins))
        thresholds = resolved_settings.thresholds
        return {
            "score": score,
            "smoothed_score": score,
            "verdict": verdict_for_score(score, thresholds),
            "duration": decoded.duration_s,
            "duration_s": decoded.duration_s,
            "format": decoded.format_label,
            "model_version": _model_version(active_model),
            "warnings": _warning_list(decoded, processed.warnings),
            "bin_scores": bins,
            "bin_s": STREAM_HOP_S,
            "source": "upload",
            "thresholds_placeholder": thresholds.placeholder,
        }

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin and origin not in resolved_settings.allowed_origins:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        active_model = app.state.model
        if active_model is None:
            await websocket.send_json({"type": "error", "message": "model is not loaded"})
            await websocket.close(code=1011)
            return

        configured = False
        sample_rate = 0
        expected_samples = 0
        buffer = np.zeros(0, dtype=np.float32)
        total_samples = 0
        raw_scores: list[float] = []
        result_seq = 0
        latest_job: tuple[np.ndarray, float, float] | None = None
        processing_task: asyncio.Task[None] | None = None
        log_path: Path | None = None
        last_result_wall: float | None = None
        loop = asyncio.get_running_loop()

        async def send_error(message: str) -> None:
            try:
                await websocket.send_json({"type": "error", "message": message})
            except Exception:
                pass

        def score_snapshot(snapshot: np.ndarray) -> float:
            processed = preprocess(snapshot, sr=sample_rate, trim_silence=resolved_settings.trim_silence)
            return _score_window(active_model, processed.waveform)

        async def drain_latest() -> None:
            nonlocal latest_job, result_seq, last_result_wall
            while latest_job is not None:
                snapshot, seconds, received_wall = latest_job
                latest_job = None
                started = time.perf_counter()
                try:
                    score = await loop.run_in_executor(None, score_snapshot, snapshot)
                except Exception as exc:
                    await send_error(f"scoring failed: {exc}")
                    continue
                processing_ms = (time.perf_counter() - started) * 1000.0
                if latest_job is not None:
                    continue
                raw_scores.append(score)
                smoothed = float(np.mean(raw_scores[-3:]))
                now = time.perf_counter()
                inter_update_ms = 0.0 if last_result_wall is None else (now - last_result_wall) * 1000.0
                last_result_wall = now
                t = float(seconds)
                payload: dict[str, Any] = {
                    "type": "result",
                    "seq": result_seq,
                    "t": t,
                    "score": score,
                    "smoothed_score": smoothed,
                    "verdict": verdict_for_score(smoothed, resolved_settings.thresholds),
                    "bin_scores": _stream_bin_scores(score, t),
                    "seconds": t,
                    "latency_ms": (time.perf_counter() - received_wall) * 1000.0,
                    "model_version": _model_version(active_model),
                    "source": "mic",
                }
                result_seq += 1
                if log_path is not None:
                    _append_stream_log(log_path, [result_seq - 1, t, inter_update_ms, processing_ms, len(snapshot) / sample_rate])
                try:
                    await websocket.send_json(payload)
                except Exception:
                    return

        async def ensure_drain() -> None:
            nonlocal processing_task
            if processing_task is None or processing_task.done():
                processing_task = asyncio.create_task(drain_latest())

        async def flush() -> None:
            nonlocal processing_task
            if latest_job is not None:
                await ensure_drain()
            if processing_task is not None:
                try:
                    await processing_task
                except Exception as exc:
                    await send_error(f"scoring failed: {exc}")
                processing_task = None

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                binary = message.get("bytes")
                if text is not None:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        await send_error("text messages must be valid JSON")
                        continue
                    message_type = payload.get("type") if isinstance(payload, dict) else None
                    if message_type == "config":
                        if configured:
                            await send_error("config may only be sent once")
                            continue
                        try:
                            sample_rate = int(payload["sample_rate"])
                            channels = int(payload["channels"])
                            sample_format = str(payload["format"])
                        except (KeyError, TypeError, ValueError):
                            await send_error("config requires sample_rate, channels, and format")
                            continue
                        if sample_rate <= 0 or channels != 1 or sample_format != "int16":
                            await send_error("config must specify mono int16 PCM and a positive sample_rate")
                            continue
                        expected_samples = int(round(sample_rate * STREAM_HOP_S))
                        configured = True
                        log_path = _stream_log_path(resolved_settings.logs_dir)
                        continue
                    if message_type == "stop":
                        if not configured:
                            await send_error("send config before stop")
                        await flush()
                        await websocket.close(code=1000)
                        return
                    await send_error("expected config or stop JSON message")
                    continue

                if binary is not None:
                    if not configured:
                        await send_error("send config before binary audio frames")
                        continue
                    if len(binary) % 2:
                        await send_error("audio frame has an odd byte length")
                        continue
                    samples = np.frombuffer(binary, dtype="<i2")
                    if len(samples) != expected_samples:
                        await send_error(f"audio frame must contain {expected_samples} int16 samples")
                        continue
                    frame = samples.astype(np.float32) / 32768.0
                    buffer = np.concatenate((buffer, frame))
                    max_samples = int(round(sample_rate * STREAM_WINDOW_S))
                    if len(buffer) > max_samples:
                        buffer = buffer[-max_samples:]
                    total_samples += len(frame)
                    seconds = total_samples / sample_rate
                    seconds = round(seconds / STREAM_HOP_S) * STREAM_HOP_S
                    if seconds < 2.0:
                        await websocket.send_json({"type": "warmup", "seconds": float(seconds)})
                    else:
                        latest_job = (buffer.copy(), float(seconds), time.perf_counter())
                        await ensure_drain()
                    continue

                await send_error("unsupported WebSocket message")
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            await send_error(f"stream error: {exc}")
        finally:
            if processing_task is not None and not processing_task.done():
                processing_task.cancel()

    return app


app = create_app()
