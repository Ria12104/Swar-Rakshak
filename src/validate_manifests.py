"""Validate Phase 1 CSV manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .build_manifests import MANIFEST_COLUMNS

EXPECTED_TOTALS = {"train": 25_380, "dev": 24_844, "eval": 71_237}
MANIFEST_NAMES = ("train", "dev", "eval", "itw_test")


def _read_manifests(manifests_dir: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for name in MANIFEST_NAMES:
        path = manifests_dir / f"{name}.csv"
        if path.exists():
            result[name] = pd.read_csv(path)
        else:
            result[name] = pd.DataFrame(columns=MANIFEST_COLUMNS)
    return result


def validate_manifests(manifests_dir: str | Path = "manifests") -> bool:
    root = Path(manifests_dir)
    frames = _read_manifests(root)
    failures: list[str] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        print(f"{name}: {'PASS' if passed else 'FAIL'}{(' — ' + detail) if detail else ''}")
        if not passed:
            failures.append(name)

    all_frames = [frame for frame in frames.values() if not frame.empty]
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=MANIFEST_COLUMNS)
    required_without_nullable_attack = [column for column in MANIFEST_COLUMNS if column != "attack_id"]

    missing_columns = {
        name: [column for column in MANIFEST_COLUMNS if column not in frame.columns]
        for name, frame in frames.items()
        if not frame.empty and any(column not in frame.columns for column in MANIFEST_COLUMNS)
    }
    check("required columns present", not missing_columns, str(missing_columns))

    missing_paths = [str(path) for path in combined.get("path", pd.Series(dtype=str)).dropna() if not Path(str(path)).exists()]
    check("every path exists", not missing_paths, ", ".join(missing_paths[:3]))

    labels = set(pd.to_numeric(combined.get("label", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    bad_labels = labels - {0, 1}
    check("labels only in {0,1}", not bad_labels, f"unexpected labels: {sorted(bad_labels)}")

    file_ids = combined["path"].map(lambda value: Path(str(value)).stem) if "path" in combined else pd.Series(dtype=str)
    duplicates = file_ids[file_ids.duplicated()].unique().tolist()
    check("no duplicate file ids", not duplicates, f"duplicates: {duplicates[:3]}")

    split_frames = {name: frames[name] for name in ("train", "dev", "eval")}
    file_overlap = set()
    speaker_overlap = set()
    for left_name, right_name in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        left = split_frames[left_name]
        right = split_frames[right_name]
        left_files = set(left["path"].astype(str)) if "path" in left else set()
        right_files = set(right["path"].astype(str)) if "path" in right else set()
        left_speakers = set(left["speaker_id"].dropna().astype(str)) if "speaker_id" in left else set()
        right_speakers = set(right["speaker_id"].dropna().astype(str)) if "speaker_id" in right else set()
        file_overlap |= left_files & right_files
        speaker_overlap |= left_speakers & right_speakers
    check("no file overlap between train/dev/eval", not file_overlap, f"overlap: {sorted(file_overlap)[:3]}")
    check("no speaker overlap between train/dev/eval", not speaker_overlap, f"overlap: {sorted(speaker_overlap)[:3]}")

    missing_values = []
    for column in required_without_nullable_attack:
        if column in combined and combined[column].isna().any():
            missing_values.append(column)
    check("no NaN in required columns", not missing_values, f"columns: {missing_values}")

    print("\nCounts by dataset/split/class:")
    if combined.empty:
        print("<no rows>")
    else:
        counts = combined.assign(label=pd.to_numeric(combined["label"], errors="coerce")).groupby(
            ["dataset", "split", "label"], dropna=False
        ).size().rename("count").reset_index()
        print(counts.to_string(index=False))

    print("\nOfficial ASVspoof 2019 LA reference totals (reference only):")
    for split, expected in EXPECTED_TOTALS.items():
        actual = len(frames[split])
        print(f"{split}: actual={actual}, expected_approx={expected}")
        if actual != expected:
            print(f"WARNING: {split} count differs from the official reference; this is not a failure.")
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-dir", default="manifests")
    args = parser.parse_args()
    return 0 if validate_manifests(args.manifests_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
