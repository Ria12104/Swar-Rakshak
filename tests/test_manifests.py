from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.build_manifests import build_manifests, parse_asvspoof_protocol, parse_itw_metadata
from src.validate_manifests import validate_manifests


def _write_audio(path: Path, sample_rate: int = 8_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sf.write(path, 0.2 * np.sin(2 * np.pi * 220 * t), sample_rate)


def test_parsers_are_pure_and_tolerant() -> None:
    asv = parse_asvspoof_protocol("\nLA_0001 LA_T_0001 - - bonafide\nLA_0002 LA_T_0002 - A01 spoof\n", "train")
    assert asv["label"].tolist() == [0, 1]
    assert pd.isna(asv.loc[0, "attack_id"])
    assert asv.loc[1, "attack_id"] == "A01"

    itw = parse_itw_metadata("file,speaker,label\na.wav,s1,bona-fide\nb.wav,s2,spoof\n")
    assert itw["label"].tolist() == [0, 1]
    assert itw["split"].tolist() == ["test", "test"]


def test_fake_tree_build_and_validate_end_to_end(tmp_path: Path) -> None:
    asv_root = tmp_path / "asv"
    protocol_root = asv_root / "ASVspoof2019_LA_cm_protocols"
    protocol_root.mkdir(parents=True)
    split_rows = {
        "train": [("LA_0001", "LA_T_0001", "-", "bonafide"), ("LA_0002", "LA_T_0002", "A01", "spoof")],
        "dev": [("LA_0101", "LA_D_0101", "-", "bonafide"), ("LA_0102", "LA_D_0102", "A02", "spoof")],
        "eval": [("LA_0201", "LA_E_0201", "-", "bonafide"), ("LA_0202", "LA_E_0202", "A03", "spoof")],
    }
    for split, rows in split_rows.items():
        flac_dir = asv_root / f"ASVspoof2019_LA_{split}" / "flac"
        for speaker, file_id, attack, label in rows:
            _write_audio(flac_dir / f"{file_id}.flac")
        protocol = "\n".join(f"{speaker} {file_id} - {attack} {label}" for speaker, file_id, attack, label in rows)
        (protocol_root / f"ASVspoof2019.LA.cm.{split}.txt").write_text(protocol + "\n", encoding="utf-8")

    itw_root = tmp_path / "itw"
    itw_root.mkdir()
    _write_audio(itw_root / "wild_real.wav", sample_rate=16_000)
    _write_audio(itw_root / "wild_fake.wav", sample_rate=16_000)
    (itw_root / "meta.csv").write_text(
        "file,speaker,label\nwild_real.wav,wild_speaker_1,bona-fide\nwild_fake.wav,wild_speaker_2,spoof\n",
        encoding="utf-8",
    )

    config = tmp_path / "config.yaml"
    config.write_text(f"asvspoof_la_root: {asv_root}\nitw_root: {itw_root}\n", encoding="utf-8")
    output = tmp_path / "manifests"
    build_manifests(config, output, workers=1)

    for name in ("train", "dev", "eval", "itw_test"):
        frame = pd.read_csv(output / f"{name}.csv")
        assert list(frame.columns) == [
            "path", "label", "dataset", "split", "speaker_id", "attack_id", "duration",
            "sample_rate", "rms_dbfs", "leading_silence_s", "trailing_silence_s",
        ]
        assert len(frame) == 2
        assert frame["path"].map(Path).map(Path.exists).all()
    assert validate_manifests(output)
