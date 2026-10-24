"""Extract frozen WavLM layer/chunk embeddings from an audio manifest."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio import TARGET_SR, make_windows, preprocess  # noqa: E402
from src.sampling import stratified_subset  # noqa: E402


MODEL_NAME = "microsoft/wavlm-base-plus"
WINDOW_S = 4.0
CHUNK_S = 0.5
WAVLM_FRAME_RATE = 50
N_CHUNKS = 8
FRAMES_PER_CHUNK = 25
N_FRAMES = N_CHUNKS * FRAMES_PER_CHUNK
N_LAYERS = 13
HIDDEN_SIZE = 768
SHARD_SIZE = 500


@dataclass(frozen=True)
class ClipRecord:
    path: str
    label: int
    attack_id: str


class AudioClipDataset(Dataset[tuple[Tensor, int, str, str]]):
    """Read and preprocess clips in DataLoader workers."""

    def __init__(self, frame: pd.DataFrame) -> None:
        required = {"path", "label"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
        self.records = [
            ClipRecord(
                path=str(row["path"]),
                label=int(row["label"]),
                attack_id="" if pd.isna(row.get("attack_id")) else str(row.get("attack_id", "")),
            )
            for _, row in frame.reset_index(drop=True).iterrows()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str, str]:
        record = self.records[index]
        processed = preprocess(record.path)
        windows, _ = make_windows(processed.waveform, win_s=WINDOW_S, hop_s=2.0, max_s=10.0)
        # The shard contract is one fixed four-second window per clip. The
        # shared make_windows() rule pads short clips and truncates long ones.
        waveform = windows[0]
        return torch.from_numpy(waveform), record.label, record.path, record.attack_id


def _collate_clips(
    batch: list[tuple[Tensor, int, str, str]],
) -> tuple[Tensor, Tensor, list[str], list[str]]:
    waveforms, labels, paths, attack_ids = zip(*batch)
    return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long), list(paths), list(attack_ids)


def _load_model(model_name: str, device: torch.device, fp16: bool = True) -> torch.nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - dependency guard for CLI users
        raise RuntimeError("transformers is required; install requirements.txt first") from exc

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    if fp16 and device.type == "cuda":
        model.half()
    return model


def _get_hidden_states(output: Any) -> tuple[Tensor, ...]:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None and isinstance(output, dict):
        hidden_states = output.get("hidden_states")
    if hidden_states is None:
        raise RuntimeError("WavLM output did not contain hidden_states")
    hidden_states = tuple(hidden_states)
    if len(hidden_states) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} WavLM hidden-state layers, got {len(hidden_states)}")
    return hidden_states


def _normalize_frame_count(layer: Tensor) -> Tensor:
    """Make approximately-50-fps WavLM output exactly 200 frames."""
    if layer.ndim != 3 or layer.shape[-1] != HIDDEN_SIZE:
        raise ValueError(f"expected hidden state [batch, frames, {HIDDEN_SIZE}], got {tuple(layer.shape)}")
    frame_count = int(layer.shape[1])
    if frame_count == N_FRAMES:
        return layer
    if frame_count < N_FRAMES:
        if frame_count == 0:
            return torch.zeros(layer.shape[0], N_FRAMES, layer.shape[2], dtype=layer.dtype, device=layer.device)
        padding = layer[:, -1:, :].expand(-1, N_FRAMES - frame_count, -1)
        return torch.cat((layer, padding), dim=1)
    # WavLM's convolutional frontend can produce 199 frames for 4 s. For
    # unusual input/model lengths, adaptive average pooling preserves the
    # temporal mean while restoring the fixed 50-fps contract.
    return torch.nn.functional.adaptive_avg_pool1d(layer.transpose(1, 2), N_FRAMES).transpose(1, 2)


def hidden_states_to_embeddings(hidden_states: Iterable[Tensor]) -> Tensor:
    """Mean-pool 13 layers into [batch, 8 half-second chunks, 13, 768]."""
    normalized = [_normalize_frame_count(layer) for layer in tuple(hidden_states)]
    if len(normalized) != N_LAYERS:
        raise ValueError(f"expected {N_LAYERS} hidden-state layers, got {len(normalized)}")
    stacked = torch.stack(normalized, dim=2)  # [B, 200, 13, 768]
    embeddings = stacked.reshape(stacked.shape[0], N_CHUNKS, FRAMES_PER_CHUNK, N_LAYERS, HIDDEN_SIZE)
    return embeddings.float().mean(dim=2).to(dtype=torch.float16)


def extract_batch(model: torch.nn.Module, waveforms: Tensor, device: torch.device, fp16: bool = True) -> Tensor:
    """Run one fixed-length waveform batch through frozen WavLM."""
    inputs = waveforms.to(device, non_blocking=True)
    use_fp16 = fp16 and device.type == "cuda"
    if use_fp16:
        inputs = inputs.half()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
        output = model(inputs, output_hidden_states=True)
    return hidden_states_to_embeddings(_get_hidden_states(output)).cpu()


def _shard_path(out_dir: Path, split_name: str, shard_number: int) -> Path:
    return out_dir / f"{split_name}_shard_{shard_number:05d}.npz"


def _index_rows_from_shard(
    shard_path: Path,
    expected_paths: list[str] | None = None,
) -> list[dict[str, object]]:
    with np.load(shard_path, allow_pickle=False) as shard:
        paths = [str(value) for value in shard["paths"].tolist()]
        labels = shard["labels"].astype(np.int64).tolist()
        attack_ids = [str(value) for value in shard["attack_id"].tolist()]
    if expected_paths is not None and paths != expected_paths:
        raise ValueError(f"existing shard does not match the selected manifest rows: {shard_path}")
    return [
        {"path": path, "label": int(label), "attack_id": attack_id, "shard": shard_path.name, "row": row}
        for row, (path, label, attack_id) in enumerate(zip(paths, labels, attack_ids))
    ]


def _write_shard(
    shard_path: Path,
    embeddings: list[np.ndarray],
    labels: list[int],
    paths: list[str],
    attack_ids: list[str],
) -> None:
    np.savez_compressed(
        shard_path,
        embeddings=np.concatenate(embeddings, axis=0).astype(np.float16, copy=False),
        labels=np.asarray(labels, dtype=np.int64),
        paths=np.asarray(paths, dtype=str),
        attack_id=np.asarray(attack_ids, dtype=str),
    )


def extract_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    split_name: str = "train",
    degrade: str = "none",
    batch_size: int = 8,
    num_workers: int = 4,
    resume: bool = False,
    seed: int = 42,
    model: torch.nn.Module | None = None,
    model_name: str = MODEL_NAME,
    device: str | None = None,
    shard_size: int = SHARD_SIZE,
) -> Path:
    """Extract and shard a manifest, returning the written index path."""
    if degrade != "none":
        raise NotImplementedError("--degrade train_pool is reserved for a later phase; use --degrade none")
    if batch_size <= 0 or num_workers < 0 or shard_size <= 0:
        raise ValueError("batch_size and shard_size must be positive; num_workers must be non-negative")

    manifest = pd.read_csv(manifest_path)
    if "path" not in manifest.columns or "label" not in manifest.columns:
        raise ValueError("manifest must contain path and label columns")
    selected = stratified_subset(manifest, limit, seed, ("label",)).reset_index(drop=True)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if selected.empty:
        index_path = output_dir / f"{split_name}_index.csv"
        pd.DataFrame(columns=["path", "label", "attack_id", "shard", "row"]).to_csv(index_path, index=False)
        print("extracted_clips=0 total_seconds=0.00 clips_per_sec=0.00", flush=True)
        return index_path
    index_rows: list[dict[str, object]] = []
    active_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    active_model = model if model is not None else _load_model(model_name, active_device, fp16=True)
    active_model.eval()
    started = time.perf_counter()
    processed_clips = 0
    batch_count = 0

    for shard_number, start in enumerate(range(0, len(selected), shard_size)):
        shard_frame = selected.iloc[start : start + shard_size].copy()
        shard_file = _shard_path(output_dir, split_name, shard_number)
        expected_paths = shard_frame["path"].astype(str).tolist()
        if resume and shard_file.exists():
            index_rows.extend(_index_rows_from_shard(shard_file, expected_paths))
            processed_clips += len(shard_frame)
            continue

        dataset = AudioClipDataset(shard_frame)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=active_device.type == "cuda",
            persistent_workers=num_workers > 0,
            collate_fn=_collate_clips,
        )
        embeddings: list[np.ndarray] = []
        labels: list[int] = []
        paths: list[str] = []
        attack_ids: list[str] = []
        for waveforms, batch_labels, batch_paths, batch_attack_ids in loader:
            batch_embeddings = extract_batch(active_model, waveforms, active_device, fp16=True).numpy()
            embeddings.append(batch_embeddings)
            labels.extend(int(value) for value in batch_labels.tolist())
            paths.extend(batch_paths)
            attack_ids.extend(batch_attack_ids)
            processed_clips += len(batch_paths)
            batch_count += 1
            if batch_count % 50 == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(f"batches={batch_count} clips={processed_clips} clips_per_sec={processed_clips / elapsed:.2f}", flush=True)

        _write_shard(shard_file, embeddings, labels, paths, attack_ids)
        index_rows.extend(_index_rows_from_shard(shard_file, expected_paths))

    index_path = output_dir / f"{split_name}_index.csv"
    pd.DataFrame(index_rows, columns=["path", "label", "attack_id", "shard", "row"]).to_csv(index_path, index=False)
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(f"extracted_clips={processed_clips} total_seconds={elapsed:.2f} clips_per_sec={processed_clips / elapsed:.2f}", flush=True)
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--degrade", choices=("none", "train_pool"), default="none")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()
    extract_manifest(
        args.manifest,
        args.out_dir,
        limit=args.limit,
        split_name=args.split_name,
        degrade=args.degrade,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=args.resume,
        seed=args.seed,
        model_name=args.model_name,
        device=args.device,
        shard_size=args.shard_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
