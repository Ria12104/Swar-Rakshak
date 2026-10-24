from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app import BackendSettings, Thresholds, _load_thresholds, create_app, verdict_for_score


class FixtureModel:
    model_version = "fixture-random-cnn"
    device = "cpu"

    def __init__(self, score_value: float = 0.55) -> None:
        self.score_value = score_value

    def load(self) -> "FixtureModel":
        return self

    def score(self, waveform_16k: np.ndarray) -> float:
        assert waveform_16k.ndim == 1
        assert waveform_16k.dtype == np.float32
        return self.score_value


def _wav_bytes(
    duration_s: float = 2.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    amplitude: float = 0.2,
) -> bytes:
    t = np.arange(int(duration_s * sample_rate), dtype=np.float32) / sample_rate
    mono = amplitude * np.sin(2 * np.pi * 220.0 * t)
    waveform = np.column_stack((mono, mono * 0.8)) if channels == 2 else mono
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return buffer.getvalue()


def _client(score: float = 0.55) -> TestClient:
    settings = BackendSettings(
        model_path=Path("fixture.pt"),
        thresholds=Thresholds(0.40, 0.70, False),
        trim_silence=False,
        max_upload_bytes=10 * 1024 * 1024,
        allowed_origins=("http://localhost:5173", "http://127.0.0.1:5173"),
        logs_dir=Path("logs"),
    )
    return TestClient(create_app(FixtureModel(score), settings))


def test_health_has_model_and_thresholds() -> None:
    with _client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "fixture-random-cnn"
    assert body["thresholds"] == {"uncertain_low": 0.4, "synthetic_high": 0.7}
    assert body["thresholds_placeholder"] is False


def test_probability_threshold_file_matches_sigmoid_model_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "cnn_full" / "best.pt"
    model_path.parent.mkdir()
    (model_path.parent / "thresholds.json").write_text(
        json.dumps({"uncertain_low": 0.123, "synthetic_high": 0.987}),
        encoding="utf-8",
    )
    monkeypatch.delenv("UNCERTAIN_LOW", raising=False)
    monkeypatch.delenv("SYNTHETIC_HIGH", raising=False)
    monkeypatch.delenv("THRESHOLDS_PATH", raising=False)
    thresholds = _load_thresholds(model_path)
    assert thresholds.placeholder is False
    assert thresholds.uncertain_low == pytest.approx(0.123)
    assert thresholds.synthetic_high == pytest.approx(0.987)


def test_predict_returns_contract_and_original_format() -> None:
    with _client(0.55) as client:
        response = client.post("/predict", files={"file": ("stereo.wav", _wav_bytes(2.0, 44_100, 2), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "score",
        "smoothed_score",
        "verdict",
        "duration",
        "duration_s",
        "format",
        "model_version",
        "warnings",
        "bin_scores",
        "bin_s",
        "source",
        "thresholds_placeholder",
    }
    assert body["format"] == "wav, 44100 Hz, stereo"
    assert body["duration"] == body["duration_s"]
    assert len(body["bin_scores"]) == 4
    assert body["score"] == body["smoothed_score"] == 0.55
    assert body["bin_s"] == 0.5
    assert body["source"] == "upload"
    assert "resampled" in body["warnings"]


@pytest.mark.parametrize(
    ("filename", "payload", "expected_status"),
    [
        ("empty.wav", b"", 400),
        ("broken.wav", b"not a wav", 400),
        ("audio.txt", b"not audio", 415),
    ],
)
def test_predict_rejects_empty_corrupt_and_unsupported(filename: str, payload: bytes, expected_status: int) -> None:
    with _client() as client:
        response = client.post("/predict", files={"file": (filename, payload, "application/octet-stream")})
    assert response.status_code == expected_status
    assert response.json()["detail"]


def test_predict_handles_silence_and_duration_limit() -> None:
    with _client() as client:
        silent = client.post("/predict", files={"file": ("silent.wav", _wav_bytes(1.0, amplitude=0.0), "audio/wav")})
        too_long = client.post("/predict", files={"file": ("long.wav", _wav_bytes(31.0), "audio/wav")})
    assert silent.status_code == 200
    assert "silent" in silent.json()["warnings"]
    assert too_long.status_code == 400


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.399, "genuine"), (0.40, "uncertain"), (0.699, "uncertain"), (0.70, "synthetic")],
)
def test_verdict_boundaries(score: float, expected: str) -> None:
    assert verdict_for_score(score, Thresholds(0.40, 0.70, False)) == expected


@pytest.mark.parametrize("sample_rate", [16_000, 48_000])
def test_stream_warmup_results_and_clean_stop(tmp_path: Path, sample_rate: int) -> None:
    settings = BackendSettings(
        model_path=Path("fixture.pt"),
        thresholds=Thresholds(0.40, 0.70, False),
        trim_silence=False,
        max_upload_bytes=10 * 1024 * 1024,
        allowed_origins=("http://localhost:5173", "http://127.0.0.1:5173"),
        logs_dir=tmp_path / "logs",
    )
    with TestClient(create_app(FixtureModel(0.55), settings)) as client:
        with client.websocket_connect("/stream") as socket:
            socket.send_text(json.dumps({"type": "config", "sample_rate": sample_rate, "channels": 1, "format": "int16"}))
            expected_results = []
            samples_per_hop = int(round(sample_rate * 0.5))
            chunk = (np.zeros(samples_per_hop, dtype="<i2")).tobytes()
            for frame_index in range(8):
                socket.send_bytes(chunk)
                message = socket.receive_json()
                if frame_index < 3:
                    assert message["type"] == "warmup"
                    assert message["seconds"] == pytest.approx((frame_index + 1) * 0.5)
                else:
                    assert message["type"] == "result"
                    expected_results.append(message)
            assert [item["seq"] for item in expected_results] == list(range(len(expected_results)))
            for item in expected_results:
                assert item["t"] * 2 == pytest.approx(round(item["t"] * 2))
                assert len(item["bin_scores"]) == min(8, int(item["t"] / 0.5))
                assert item["verdict"] == "uncertain"
                assert item["source"] == "mic"
            socket.send_text(json.dumps({"type": "stop"}))


def test_stream_handles_bad_frame_and_binary_before_config() -> None:
    with _client() as client:
        with client.websocket_connect("/stream") as socket:
            socket.send_bytes(b"\x00")
            assert socket.receive_json()["type"] == "error"
            socket.send_text(json.dumps({"type": "config", "sample_rate": 16_000, "channels": 1, "format": "int16"}))
            socket.send_bytes(b"\x00")
            assert socket.receive_json()["type"] == "error"
            socket.send_text(json.dumps({"type": "stop"}))


def test_cors_preflight_allows_frontend_and_ngrok_header() -> None:
    with _client() as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, ngrok-skip-browser-warning",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "ngrok-skip-browser-warning" in response.headers["access-control-allow-headers"].lower()
