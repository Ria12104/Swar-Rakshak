from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src.evaluate import apply_frozen_threshold, choose_threshold, compute_metrics, evaluate_scores
from src.features import LogMelFrontEnd
from src.models.cnn import CNNBaseline, count_parameters
from src.train_cnn import train_cnn


def test_metrics_and_thresholds(tmp_path: Path) -> None:
    labels = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(labels, perfect)
    assert metrics["eer"] == pytest.approx(0.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)

    rng = np.random.default_rng(42)
    random_metrics = compute_metrics(np.tile(labels, 50), rng.random(200))
    assert 0.25 < random_metrics["eer"] < 0.75

    inverted = compute_metrics(labels, perfect[::-1])
    assert inverted["eer"] > 0.5

    thresholds_path = tmp_path / "thresholds.json"
    thresholds = choose_threshold(labels, perfect, thresholds_path)
    assert thresholds_path.exists()
    assert json.loads(thresholds_path.read_text()) == thresholds
    assert apply_frozen_threshold(perfect, thresholds["eer_threshold"]).tolist() == [0, 0, 1, 1]


def test_attack_breakdown_uses_attack_spoofs_against_bonafides() -> None:
    manifest = pd.DataFrame(
        {
            "path": ["b1.wav", "b2.wav", "a1.wav", "a2.wav", "c1.wav"],
            "label": [0, 0, 1, 1, 1],
            "attack_id": [None, None, "A01", "A01", "A02"],
            "duration": [2.0, 3.0, 4.0, 5.0, 7.0],
        }
    )
    scores = pd.DataFrame({"path": manifest["path"], "score": [0.1, 0.2, 0.8, 0.9, 0.7]})
    result = evaluate_scores(manifest, scores, group_by=("attack_id", "duration"))
    assert set(result["by_attack_id"]) == {"A01", "A02"}
    assert "by_duration" in result


def test_features_output_shape_and_model_size() -> None:
    frontend = LogMelFrontEnd()
    waveforms = torch.randn(2, 16_000 * 4)
    features = frontend(waveforms)
    assert features.shape[0] == 2
    assert features.shape[1:] == (1, 80, features.shape[-1])
    assert torch.isfinite(features).all()

    model = CNNBaseline()
    logits = model(features)
    assert logits.shape == (2,)
    assert count_parameters(model) < 1_000_000


def test_one_training_step_on_fake_data() -> None:
    frontend = LogMelFrontEnd()
    model = CNNBaseline()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    waveforms = torch.randn(2, 16_000 * 4)
    labels = torch.tensor([0.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(frontend(waveforms)), labels)
    loss.backward()
    optimizer.step()
    assert np.isfinite(loss.item())


def _write_fake_clip(path: Path, amplitude: float, frequency: float) -> None:
    sr = 16_000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, amplitude * np.sin(2 * np.pi * frequency * t), sr)


def test_one_epoch_smoke_train_on_fake_manifest(tmp_path: Path) -> None:
    rows = []
    for split in ("train", "dev"):
        for index, label in enumerate((0, 1)):
            path = tmp_path / f"{split}_{index}.wav"
            _write_fake_clip(path, 0.1 + 0.1 * label, 220.0 + 100 * label)
            rows.append({"path": str(path), "label": label, "split": split})
    frame = pd.DataFrame(rows)
    train_path = tmp_path / "train.csv"
    dev_path = tmp_path / "dev.csv"
    frame[frame["split"] == "train"].to_csv(train_path, index=False)
    frame[frame["split"] == "dev"].to_csv(dev_path, index=False)
    config = tmp_path / "cnn.yaml"
    config.write_text(
        "\n".join(
            [
                "name: smoke",
                "seed: 7",
                f"train_manifest: {train_path}",
                f"dev_manifest: {dev_path}",
                f"runs_dir: {tmp_path / 'runs'}",
                "epochs: 1",
                "patience: 5",
                "batch_size: 2",
                "num_workers: 0",
                "learning_rate: 0.001",
                "weight_decay: 0.0001",
                "trim_silence: false",
                "device: cpu",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = train_cnn(config, limit=2, epochs=1, device_override="cpu")
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "dev_scores.csv").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "run_summary.json").exists()
