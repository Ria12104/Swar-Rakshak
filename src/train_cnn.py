"""Train the Phase 2 CNN baseline on train/dev manifests only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .audio import random_crop, preprocess
from .evaluate import evaluate_scores, write_metrics
from .features import LogMelFrontEnd
from .inference import score_manifest_dataframe
from .models.cnn import CNNBaseline, count_parameters
from .sampling import stratified_subset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AudioTrainDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, frame: pd.DataFrame, trim_silence: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.trim_silence = trim_silence

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.frame.iloc[index]
        processed = preprocess(str(row["path"]), trim_silence=self.trim_silence)
        crop = random_crop(processed.waveform)
        return torch.from_numpy(crop), torch.tensor(float(row["label"]), dtype=torch.float32)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _manifest_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _make_loader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    num_workers_override: int | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = AudioTrainDataset(frame, bool(config.get("trim_silence", False)))
    labels = frame["label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = np.array([1.0 / counts[label] if counts[label] else 0.0 for label in labels], dtype=np.float64)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    num_workers = int(config.get("num_workers", 0)) if num_workers_override is None else num_workers_override
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 16)),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: CNNBaseline,
    frontend: LogMelFrontEnd,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    frontend.train()
    total_loss = 0.0
    amp_enabled = device.type == "cuda"
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(frontend(waveforms))
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * len(labels)
    return total_loss / max(1, len(loader.dataset))


def train_cnn(
    config_path: str | Path = "configs/cnn.yaml",
    run_name: str | None = None,
    limit: int | None = None,
    epochs: int | None = None,
    device_override: str | None = None,
) -> Path:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle) or {}
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = _device(device_override or str(config.get("device", "auto")))
    train_path = Path(config.get("train_manifest", "manifests/train.csv"))
    dev_path = Path(config.get("dev_manifest", "manifests/dev.csv"))
    train_frame = pd.read_csv(train_path)
    dev_frame = pd.read_csv(dev_path)
    if limit is not None:
        train_frame = stratified_subset(train_frame, limit, seed, ("label",))
        dev_frame = stratified_subset(dev_frame, limit, seed, ("label",))
    if train_frame["label"].nunique() < 2 or dev_frame["label"].nunique() < 2:
        raise ValueError("training and dev manifests must each contain both labels")

    force_single_worker = (
        device.type == "cpu"
        or (limit is not None and limit <= 8)
        or "PYTEST_CURRENT_TEST" in os.environ
    )
    effective_num_workers = 0 if force_single_worker else int(config.get("num_workers", 0))

    name = run_name or str(config.get("name", "cnn_baseline"))
    run_dir = Path(config.get("runs_dir", "runs")) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = _manifest_hash([train_path, dev_path])
    saved_config = dict(config)
    saved_config.update(
        {
            "manifest_hash": manifest_hash,
            "seed": seed,
            "device": str(device),
            "num_workers": effective_num_workers,
        }
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(saved_config, sort_keys=False), encoding="utf-8")
    (run_dir / "manifest_hash.txt").write_text(manifest_hash + "\n", encoding="utf-8")

    loader = _make_loader(train_frame, config, seed, effective_num_workers)
    model = CNNBaseline().to(device)
    frontend = LogMelFrontEnd().to(device)
    optimizer = AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    max_epochs = min(20, int(epochs if epochs is not None else config.get("epochs", 20)))
    patience = int(config.get("patience", 5))
    best_eer = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    best_metrics: dict[str, object] = {}
    best_scores = pd.DataFrame()

    for epoch in range(1, max_epochs + 1):
        loss = train_one_epoch(model, frontend, loader, optimizer, criterion, device, scaler)
        scores = score_manifest_dataframe(
            dev_frame,
            model,
            device,
            batch_size=int(config.get("batch_size", 16)),
            trim_silence=bool(config.get("trim_silence", False)),
        )
        metrics = evaluate_scores(dev_frame, scores)["overall"]
        assert isinstance(metrics, dict)
        eer = float(metrics["eer"])
        print(f"epoch={epoch} loss={loss:.5f} dev_eer={eer:.5f} dev_auc={float(metrics['roc_auc']):.5f}")
        if eer < best_eer:
            best_eer = eer
            best_epoch = epoch
            stale_epochs = 0
            best_metrics = {"epoch": epoch, "train_loss": loss, **metrics}
            best_scores = scores
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "config": saved_config,
                    "parameter_count": count_parameters(model),
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    wall_time = time.perf_counter() - started
    best_scores.to_csv(run_dir / "dev_scores.csv", index=False)
    write_metrics(best_metrics, run_dir / "metrics.json")
    checkpoint_size = (run_dir / "best.pt").stat().st_size / (1024 * 1024)
    summary = {
        "parameter_count": count_parameters(model),
        "checkpoint_size_mb": checkpoint_size,
        "training_wall_time_s": wall_time,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "manifest_hash": manifest_hash,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_dir = train_cnn(args.config, args.name, args.limit, args.epochs, args.device)
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
