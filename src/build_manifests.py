"""Build ASVspoof LA and In-the-Wild CSV manifests."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

from .audio import preprocess, silence_stats

MANIFEST_COLUMNS = [
    "path",
    "label",
    "dataset",
    "split",
    "speaker_id",
    "attack_id",
    "duration",
    "sample_rate",
    "rms_dbfs",
    "leading_silence_s",
    "trailing_silence_s",
]


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Path]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = ("asvspoof_la_root", "itw_root")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"config.yaml is missing required paths: {', '.join(missing)}")
    return {key: Path(str(config[key])).expanduser() for key in required}


def _require_data_paths(config: dict[str, Path]) -> None:
    missing = [f"{key}={value}" for key, value in config.items() if not value.exists()]
    if missing:
        raise FileNotFoundError(
            "Configured dataset path(s) do not exist. Extract the datasets and fill config.yaml first: "
            + "; ".join(missing)
        )


def _protocol_paths(root: Path) -> dict[str, Path]:
    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    if not protocol_dir.exists():
        raise FileNotFoundError(f"ASVspoof protocol directory not found: {protocol_dir}")
    result: dict[str, Path] = {}
    for split in ("train", "dev", "eval"):
        matches = sorted(protocol_dir.glob(f"*cm.{split}*"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one ASVspoof *cm.{split}* protocol in {protocol_dir}, found {len(matches)}: {matches}"
            )
        result[split] = matches[0]
    return result


def _label_from_asv(value: str) -> int:
    value = value.strip().lower()
    if value == "bonafide":
        return 0
    if value == "spoof":
        return 1
    raise ValueError(f"Unknown ASVspoof label {value!r}; expected bonafide or spoof")


def parse_asvspoof_protocol(text: str, split: str, audio_root: str | Path | None = None) -> pd.DataFrame:
    """Parse ASVspoof protocol text into rows; no audio I/O occurs here."""
    rows: list[dict[str, object]] = []
    root = Path(audio_root) if audio_root is not None else None
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed ASVspoof protocol line {line_number}: {line!r}")
        speaker_id, file_id, attack_id, key = parts[0], parts[1], parts[3], parts[-1]
        path = _resolve_asv_path(root, split, file_id) if root is not None else file_id
        rows.append(
            {
                "path": str(path),
                "file_id": file_id,
                "label": _label_from_asv(key),
                "dataset": "asvspoof2019_la",
                "split": split,
                "speaker_id": speaker_id,
                "attack_id": None if attack_id == "-" else attack_id,
            }
        )
    return pd.DataFrame(rows)


def _resolve_asv_path(root: Path | None, split: str, file_id: str) -> Path:
    if root is None:
        return Path(file_id)
    expected = root / f"ASVspoof2019_LA_{split}" / "flac" / f"{file_id}.flac"
    if expected.exists():
        return expected.resolve()
    matches = sorted(root.rglob(f"{file_id}.flac"))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        return expected.resolve()
    raise RuntimeError(f"Multiple audio files found for ASVspoof file id {file_id}: {matches}")


def _normalise_column_name(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum())


def _detect_column(columns: Iterable[str], candidates: set[str], description: str) -> str:
    normalised = {_normalise_column_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    raise ValueError(f"Could not detect In-the-Wild {description} column; found columns: {list(columns)}")


def _label_from_itw(value: object) -> int:
    normalised = _normalise_column_name(str(value))
    if normalised in {"bonafide", "bonafied", "genuine", "real", "human"}:
        return 0
    if normalised in {"spoof", "synthetic", "fake", "deepfake", "generated"}:
        return 1
    raise ValueError(f"Unknown In-the-Wild label value {value!r}; expected bona-fide/bonafide or spoof")


def parse_itw_metadata(text: str, audio_root: str | Path | None = None) -> pd.DataFrame:
    """Parse In-the-Wild CSV text into rows; no audio feature extraction occurs here."""
    from io import StringIO

    metadata = pd.read_csv(StringIO(text))
    file_col = _detect_column(metadata.columns, {"file", "filename", "filepath", "path", "audio", "audiopath"}, "file")
    speaker_col = _detect_column(metadata.columns, {"speaker", "speakerid", "speakername", "id"}, "speaker")
    label_col = _detect_column(metadata.columns, {"label", "class", "target", "isfake"}, "label")
    print(f"Detected In-the-Wild columns: file={file_col!r}, speaker={speaker_col!r}, label={label_col!r}")
    root = Path(audio_root) if audio_root is not None else None
    rows: list[dict[str, object]] = []
    for row_number, row in metadata.iterrows():
        raw_file = str(row[file_col])
        path = _resolve_itw_path(root, raw_file) if root is not None else raw_file
        rows.append(
            {
                "path": str(path),
                "file_id": Path(raw_file).stem,
                "label": _label_from_itw(row[label_col]),
                "dataset": "in_the_wild",
                "split": "test",
                "speaker_id": str(row[speaker_col]),
                "attack_id": None,
            }
        )
    return pd.DataFrame(rows)


def _resolve_itw_path(root: Path | None, raw_file: str) -> Path:
    candidate = Path(raw_file)
    if root is None:
        return candidate
    direct = candidate if candidate.is_absolute() else root / candidate
    if direct.exists():
        return direct.resolve()
    matches = sorted(root.rglob(candidate.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        return direct.resolve()
    raise RuntimeError(f"Multiple In-the-Wild files found for {raw_file}: {matches}")


def _find_meta_csv(root: Path) -> Path:
    direct = root / "meta.csv"
    if direct.exists():
        return direct
    matches = sorted(root.rglob("meta.csv"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Expected exactly one meta.csv under {root}, found {len(matches)}")


def inspect_datasets(config_path: str | Path = "config.yaml") -> None:
    config = load_config(config_path)
    _require_data_paths(config)
    asv_root, itw_root = config["asvspoof_la_root"], config["itw_root"]
    protocols = _protocol_paths(asv_root)
    print("ASVspoof protocol previews:")
    for split, protocol in protocols.items():
        print(f"\n[{split}] {protocol}")
        lines = protocol.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[:5]) or "<empty>")
    print("\nASVspoof audio paths:")
    for path in sorted(asv_root.rglob("*.flac"))[:3]:
        print(path)
    meta = _find_meta_csv(itw_root)
    print(f"\n[In-the-Wild] {meta}")
    print("\n".join(meta.read_text(encoding="utf-8", errors="replace").splitlines()[:5]) or "<empty>")
    print("\nIn-the-Wild audio paths:")
    for path in sorted(itw_root.rglob("*.wav"))[:3]:
        print(path)


def _feature_row(item: tuple[str, str]) -> dict[str, object]:
    path, file_id = item
    info = sf.info(path)
    processed = preprocess(path)
    rms_dbfs, leading_s, trailing_s = silence_stats(processed.waveform)
    return {
        "path": path,
        "file_id": file_id,
        "duration": float(info.duration),
        "sample_rate": int(info.samplerate),
        "rms_dbfs": rms_dbfs,
        "leading_silence_s": leading_s,
        "trailing_silence_s": trailing_s,
    }


def _cache_key(path: str) -> tuple[int, int]:
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(cache_path)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def _feature_rows(rows: pd.DataFrame, cache_path: Path, workers: int | None = None) -> pd.DataFrame:
    cache = _load_cache(cache_path)
    cached: dict[str, dict[str, object]] = {}
    if not cache.empty and {"path", "size", "mtime_ns"}.issubset(cache.columns):
        for _, row in cache.iterrows():
            cached[str(row["path"])] = row.to_dict()

    pending: list[tuple[str, str]] = []
    feature_by_path: dict[str, dict[str, object]] = {}
    missing = [str(row["path"]) for _, row in rows.iterrows() if not Path(str(row["path"])).exists()]
    if missing:
        raise FileNotFoundError(
            "Audio files referenced by metadata/protocol are missing; first missing paths: "
            + ", ".join(missing[:5])
        )
    for _, row in rows.iterrows():
        path = str(row["path"])
        size, mtime_ns = _cache_key(path)
        old = cached.get(path)
        if old is not None and int(old.get("size", -1)) == size and int(old.get("mtime_ns", -1)) == mtime_ns:
            feature_by_path[path] = old
        else:
            pending.append((path, str(row["file_id"])))

    if pending:
        worker_count = workers if workers is not None else min(4, max(1, os.cpu_count() or 1))
        if worker_count <= 1:
            computed = [_feature_row(item) for item in tqdm(pending, desc="audio features")]
        else:
            with mp.Pool(processes=worker_count) as pool:
                computed = list(tqdm(pool.imap(_feature_row, pending), total=len(pending), desc="audio features"))
        for feature in computed:
            path = str(feature["path"])
            size, mtime_ns = _cache_key(path)
            feature["size"] = size
            feature["mtime_ns"] = mtime_ns
            feature_by_path[path] = feature

    cache_rows = list(feature_by_path.values())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cache_rows).to_csv(cache_path, index=False)
    enriched = rows.copy()
    for column in ("duration", "sample_rate", "rms_dbfs", "leading_silence_s", "trailing_silence_s"):
        enriched[column] = enriched["path"].map(lambda path: feature_by_path[str(path)][column])
    return enriched


def _write_manifest(rows: pd.DataFrame, output_path: Path) -> None:
    output = rows.copy()
    for column in MANIFEST_COLUMNS:
        if column not in output:
            output[column] = None
    output[MANIFEST_COLUMNS].to_csv(output_path, index=False)


def build_manifests(config_path: str | Path = "config.yaml", output_dir: str | Path = "manifests", limit: int | None = None, workers: int | None = None) -> None:
    config = load_config(config_path)
    _require_data_paths(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    asv_root, itw_root = config["asvspoof_la_root"], config["itw_root"]
    protocol_paths = _protocol_paths(asv_root)
    all_rows: dict[str, pd.DataFrame] = {}
    for split, protocol in protocol_paths.items():
        parsed = parse_asvspoof_protocol(protocol.read_text(encoding="utf-8", errors="replace"), split, asv_root)
        if limit is not None:
            parsed = parsed.head(limit)
        all_rows[split] = parsed

    meta = _find_meta_csv(itw_root)
    itw_rows = parse_itw_metadata(meta.read_text(encoding="utf-8", errors="replace"), itw_root)
    if limit is not None:
        itw_rows = itw_rows.head(limit)
    all_rows["itw_test"] = itw_rows

    combined = pd.concat(all_rows.values(), ignore_index=True)
    combined = _feature_rows(combined, output / ".feature_cache.csv", workers=workers)
    for split in ("train", "dev", "eval"):
        _write_manifest(combined[combined["split"] == split], output / f"{split}.csv")
    _write_manifest(combined[combined["split"] == "test"], output / "itw_test.csv")
    print(f"Wrote manifests to {output.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "build"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="config.yaml")
        if name == "build":
            subparser.add_argument("--limit", type=int, default=None)
            subparser.add_argument("--workers", type=int, default=None)
            subparser.add_argument("--output-dir", default="manifests")
    args = parser.parse_args()
    if args.command == "inspect":
        inspect_datasets(args.config)
    else:
        build_manifests(args.config, args.output_dir, args.limit, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
