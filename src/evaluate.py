"""Reusable evaluation metrics, threshold selection, and report writing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def _as_arrays(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("labels and scores must be one-dimensional arrays with equal length")
    if len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("metrics require both classes: label 0 genuine and label 1 synthetic")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return y, s


def _roc_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    eer_threshold = float(thresholds[eer_index])
    return eer, eer_threshold, float(roc_auc_score(labels, scores))


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    valid = np.flatnonzero(fpr <= target_fpr)
    return float(np.max(tpr[valid])) if len(valid) else 0.0


def _bootstrap_eer(labels: np.ndarray, scores: np.ndarray, resamples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, len(labels), size=len(labels))
        sample_labels, sample_scores = labels[indices], scores[indices]
        if set(np.unique(sample_labels)) != {0, 1}:
            continue
        values.append(_roc_metrics(sample_labels, sample_scores)[0])
    if not values:
        raise ValueError("bootstrap produced no resamples containing both classes")
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def compute_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    bootstrap_resamples: int = 0,
    seed: int = 42,
) -> dict[str, float | list[float]]:
    """Compute EER, ROC-AUC, and TPR at fixed FPR operating points."""
    y, s = _as_arrays(labels, scores)
    eer, _, auc = _roc_metrics(y, s)
    metrics: dict[str, float | list[float]] = {
        "eer": eer,
        "roc_auc": auc,
        "tpr_at_fpr_1pct": _tpr_at_fpr(y, s, 0.01),
        "tpr_at_fpr_5pct": _tpr_at_fpr(y, s, 0.05),
    }
    if bootstrap_resamples:
        metrics["eer_ci_95"] = _bootstrap_eer(y, s, bootstrap_resamples, seed)
    return metrics


def _duration_bucket(duration: float) -> str:
    if duration < 2.0:
        return "<2 s"
    if duration < 4.0:
        return "2-4 s"
    if duration <= 6.0:
        return "4-6 s"
    return ">6 s"


def evaluate_scores(
    manifest_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    group_by: str | Sequence[str] | None = None,
    bootstrap_resamples: int = 0,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate scores and optional attack, duration, or condition breakdowns."""
    required_manifest = {"path", "label"}
    required_scores = {"path", "score"}
    if not required_manifest.issubset(manifest_df.columns) or not required_scores.issubset(scores_df.columns):
        raise ValueError("manifest needs path/label and scores need path/score")
    joined = manifest_df.drop(columns=["label"], errors="ignore").merge(
        scores_df[["path", "score"]], on="path", how="inner", validate="one_to_one"
    )
    labels = manifest_df.set_index("path").loc[joined["path"], "label"].to_numpy()
    joined = joined.assign(label=labels)
    result: dict[str, object] = {
        "overall": compute_metrics(labels, joined["score"], bootstrap_resamples, seed)
    }
    if group_by is None:
        return result
    groups = [group_by] if isinstance(group_by, str) else list(group_by)
    if "attack_id" in groups:
        attack_metrics: dict[str, object] = {}
        bonafide = joined[labels == 0]
        for attack_id in manifest_df.loc[manifest_df["label"] == 1, "attack_id"].dropna().unique():
            paths = set(manifest_df.loc[manifest_df["attack_id"] == attack_id, "path"])
            attack_rows = joined[joined["path"].isin(paths)]
            subset = pd.concat([bonafide, attack_rows], ignore_index=True)
            if subset["label"].nunique() == 2:
                attack_metrics[str(attack_id)] = compute_metrics(
                    subset["label"], subset["score"], bootstrap_resamples, seed
                )
        result["by_attack_id"] = attack_metrics
    if "duration" in groups:
        if "duration" not in manifest_df.columns:
            raise ValueError("duration breakdown requires manifest duration")
        durations = manifest_df.set_index("path").loc[joined["path"], "duration"].astype(float)
        joined = joined.assign(duration=durations.to_numpy(), duration_bucket=durations.map(_duration_bucket).to_numpy())
        result["by_duration"] = {
            bucket: compute_metrics(group["label"], group["score"], bootstrap_resamples, seed)
            for bucket, group in joined.groupby("duration_bucket")
            if group["label"].nunique() == 2
        }
    if "condition" in groups:
        if "condition" not in manifest_df.columns:
            raise ValueError("condition breakdown requires manifest condition")
        conditions = manifest_df.set_index("path").loc[joined["path"], "condition"]
        joined = joined.assign(condition=conditions.to_numpy())
        result["by_condition"] = {
            str(condition): compute_metrics(group["label"], group["score"], bootstrap_resamples, seed)
            for condition, group in joined.groupby("condition", dropna=False)
            if group["label"].nunique() == 2
        }
    return result


def choose_threshold(
    dev_labels: Sequence[int],
    dev_scores: Sequence[float],
    output_path: str | Path = "thresholds.json",
) -> dict[str, float]:
    """Choose and persist the EER and <=5% FPR thresholds from dev only."""
    labels, scores = _as_arrays(dev_labels, dev_scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    allowed = np.flatnonzero(fpr <= 0.05)
    best_5_index = int(allowed[np.argmax(tpr[allowed])])
    result = {
        "eer_threshold": float(thresholds[eer_index]),
        "fpr_5pct_threshold": float(thresholds[best_5_index]),
    }
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def apply_frozen_threshold(scores: Sequence[float], threshold: float) -> np.ndarray:
    """Apply a frozen synthetic-score threshold, returning labels 0/1."""
    return (np.asarray(scores, dtype=np.float64) >= float(threshold)).astype(np.int64)


def write_metrics(metrics: dict[str, object], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _print_metrics(metrics: dict[str, object]) -> None:
    overall = metrics["overall"]
    assert isinstance(overall, dict)
    print("metric                 value")
    print("----------------------  -----")
    for key, value in overall.items():
        print(f"{key:22}  {value}")
    for key, value in metrics.items():
        if key != "overall":
            print(f"\n{key}:")
            print(json.dumps(value, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", default="metrics.json")
    parser.add_argument("--group-by", action="append", choices=["attack_id", "duration", "condition"])
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--choose-threshold", default=None, metavar="PATH")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    scores = pd.read_csv(args.scores)
    metrics = evaluate_scores(manifest, scores, args.group_by, args.bootstrap)
    write_metrics(metrics, args.output)
    _print_metrics(metrics)
    if args.choose_threshold:
        labels = manifest.set_index("path").loc[scores["path"], "label"]
        thresholds = choose_threshold(labels, scores["score"], args.choose_threshold)
        print("\nthresholds:")
        print(json.dumps(thresholds, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
