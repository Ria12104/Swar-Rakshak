from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.extract_wavlm import extract_manifest, hidden_states_to_embeddings
from scripts.train_wavlm_head import WavLMHead, train_head


class FakeOutput:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class FakeWavLM(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, output_hidden_states: bool = False) -> FakeOutput:
        assert output_hidden_states is True
        batch_size = inputs.shape[0]
        hidden_states = tuple(
            torch.full((batch_size, 199, 768), float(layer), device=inputs.device)
            for layer in range(13)
        )
        return FakeOutput(hidden_states)


class ExplodingWavLM(torch.nn.Module):
    def eval(self) -> "ExplodingWavLM":
        return self

    def forward(self, inputs: torch.Tensor, output_hidden_states: bool = False) -> FakeOutput:
        raise AssertionError("resume should not run the model for existing shards")


def _write_clip(path: Path, seconds: float) -> None:
    sample_rate = 16_000
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, 0.2 * np.sin(2 * np.pi * 220.0 * t), sample_rate)


def test_hidden_states_are_mean_pooled_to_fixed_chunk_shape() -> None:
    hidden_states = tuple(torch.randn(2, 199, 768) for _ in range(13))
    result = hidden_states_to_embeddings(hidden_states)
    assert result.shape == (2, 8, 13, 768)
    assert result.dtype == torch.float16


def test_extraction_short_clip_shape_and_resume(tmp_path: Path) -> None:
    short_path = tmp_path / "short.wav"
    long_path = tmp_path / "long.wav"
    _write_clip(short_path, 0.8)
    _write_clip(long_path, 4.0)
    manifest = tmp_path / "manifest.csv"
    import pandas as pd

    pd.DataFrame(
        {
            "path": [str(short_path), str(long_path)],
            "label": [0, 1],
            "attack_id": [None, "A01"],
        }
    ).to_csv(manifest, index=False)
    out_dir = tmp_path / "features"

    index_path = extract_manifest(
        manifest,
        out_dir,
        split_name="dev",
        batch_size=2,
        num_workers=0,
        model=FakeWavLM(),
        device="cpu",
        shard_size=2,
    )
    assert index_path.exists()
    with np.load(out_dir / "dev_shard_00000.npz", allow_pickle=False) as shard:
        assert shard["embeddings"].shape == (2, 8, 13, 768)
        assert shard["embeddings"].dtype == np.float16
        assert shard["labels"].tolist() == [0, 1]

    extract_manifest(
        manifest,
        out_dir,
        split_name="dev",
        batch_size=2,
        num_workers=0,
        resume=True,
        model=ExplodingWavLM(),
        device="cpu",
        shard_size=2,
    )


def test_wavlm_head_uses_four_one_second_segments() -> None:
    model = WavLMHead()
    embeddings = torch.randn(3, 8, 13, 768)
    clip_logits, segment_logits = model(embeddings, return_segment_logits=True)
    assert clip_logits.shape == (3,)
    assert segment_logits.shape == (3, 4)
    assert torch.softmax(model.layer_logits, dim=0).sum() == torch.tensor(1.0)


def test_train_head_smoke_on_tiny_shards(tmp_path: Path) -> None:
    for split in ("train", "dev"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        np.savez_compressed(
            split_dir / f"{split}_shard_00000.npz",
            embeddings=np.random.default_rng(7).standard_normal((4, 8, 13, 768)).astype(np.float16),
            labels=np.array([0, 1, 0, 1], dtype=np.int64),
            paths=np.array([f"{split}_{i}.wav" for i in range(4)]),
            attack_id=np.array(["", "A01", "", "A02"]),
        )
    output = train_head(
        tmp_path / "train",
        tmp_path / "dev",
        tmp_path / "head_run",
        epochs=1,
        batch_size=2,
        patience=1,
        device="cpu",
    )
    assert (output / "best.pt").exists()
    assert (output / "metrics.json").exists()
