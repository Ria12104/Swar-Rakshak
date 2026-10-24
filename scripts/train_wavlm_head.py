"""Train a small learnable head on frozen WavLM embedding shards."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import compute_metrics  # noqa: E402


N_CHUNKS = 8
N_LAYERS = 13
HIDDEN_SIZE = 768
SEGMENT_CHUNKS = 2


class WavLMHead(nn.Module):
    """Layer-attention plus an MLP over four one-second clip segments."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, n_layers: int = N_LAYERS) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, embeddings: Tensor, return_segment_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if embeddings.ndim != 4 or embeddings.shape[1:] != (N_CHUNKS, N_LAYERS, HIDDEN_SIZE):
            raise ValueError(
                f"expected embeddings [batch, {N_CHUNKS}, {N_LAYERS}, {HIDDEN_SIZE}], got {tuple(embeddings.shape)}"
            )
        layer_weights = torch.softmax(self.layer_logits, dim=0)
        chunk_embeddings = (embeddings * layer_weights.view(1, 1, -1, 1)).sum(dim=2)
        segment_embeddings = chunk_embeddings.reshape(
            embeddings.shape[0], N_CHUNKS // SEGMENT_CHUNKS, SEGMENT_CHUNKS, HIDDEN_SIZE
        ).mean(dim=2)
        segment_logits = self.mlp(segment_embeddings).squeeze(-1)
        clip_logits = segment_logits.mean(dim=1)
        if return_segment_logits:
            return clip_logits, segment_logits
        return clip_logits


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shard_files(directory: str | Path) -> list[Path]:
    paths = sorted(Path(directory).glob("*_shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no WavLM shard files found under {directory}")
    return paths


def _iter_batches(
    shard_files: list[Path],
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    generator = np.random.default_rng(seed)
    for shard_path in shard_files:
        with np.load(shard_path, allow_pickle=False) as shard:
            embeddings = shard["embeddings"]
            labels = shard["labels"].astype(np.float32, copy=False)
            if embeddings.ndim != 4 or embeddings.shape[1:] != (N_CHUNKS, N_LAYERS, HIDDEN_SIZE):
                raise ValueError(f"unexpected embedding shape in {shard_path}: {embeddings.shape}")
            order = np.arange(len(labels))
            if shuffle:
                generator.shuffle(order)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch_embeddings = torch.from_numpy(embeddings[indices].astype(np.float32, copy=False)).to(device)
                batch_labels = torch.from_numpy(labels[indices]).to(device)
                yield batch_embeddings, batch_labels


def _evaluate(
    model: WavLMHead,
    shard_files: list[Path],
    batch_size: int,
    device: torch.device,
) -> tuple[float, dict[str, float | list[float]]]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for embeddings, batch_labels in _iter_batches(shard_files, batch_size, device, False, 0):
            logits.append(model(embeddings).detach().cpu().numpy())
            labels.append(batch_labels.detach().cpu().numpy())
    y = np.concatenate(labels).astype(np.int64)
    probabilities = torch.sigmoid(torch.from_numpy(np.concatenate(logits))).numpy()
    metrics = compute_metrics(y.tolist(), probabilities.tolist())
    eer = metrics["eer"]
    if not isinstance(eer, float):
        raise RuntimeError("EER metric was not scalar")
    return eer, metrics


def train_head(
    train_dir: str | Path,
    dev_dir: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    device: str | None = None,
    seed: int = 42,
) -> Path:
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")
    _seed_everything(seed)
    active_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_shards = _shard_files(train_dir)
    dev_shards = _shard_files(dev_dir)
    model = WavLMHead().to(active_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_eer = float("inf")
    stale_epochs = 0
    best_metrics: dict[str, object] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for embeddings, labels in _iter_batches(train_shards, batch_size, active_device, True, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(embeddings), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        dev_eer, metrics = _evaluate(model, dev_shards, batch_size, active_device)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch={epoch} train_loss={train_loss:.5f} dev_eer={dev_eer:.5f}", flush=True)
        if dev_eer < best_eer:
            best_eer = dev_eer
            stale_epochs = 0
            best_metrics = {"epoch": epoch, "train_loss": train_loss, **metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "device": str(active_device),
                    "metrics": best_metrics,
                },
                output_path / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    (output_path / "metrics.json").write_text(json.dumps(best_metrics, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--dev-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_head(
        args.train_dir,
        args.dev_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
