"""Check silence/loudness correlations and dev trim-silence sensitivity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import evaluate_scores  # noqa: E402
from src.inference import load_cnn_checkpoint, score_manifest_dataframe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default="manifests/train.csv")
    parser.add_argument("--dev-manifest", default="manifests/dev.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train = pd.read_csv(args.train_manifest)
    dev = pd.read_csv(args.dev_manifest)
    correlations = {
        column: float(train[column].corr(train["label"]))
        for column in ("leading_silence_s", "trailing_silence_s", "duration", "rms_dbfs")
    }
    print("manifest correlation with label")
    print("feature                 correlation")
    for feature, value in correlations.items():
        print(f"{feature:23} {value:.5f}")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model = load_cnn_checkpoint(args.checkpoint, device)
    rows = []
    for trim in (False, True):
        scores = score_manifest_dataframe(dev, model, device, args.batch_size, trim)
        metrics = evaluate_scores(dev, scores)["overall"]
        assert isinstance(metrics, dict)
        rows.append({"trim_silence": trim, "eer": metrics["eer"], "roc_auc": metrics["roc_auc"], "tpr_at_fpr_1pct": metrics["tpr_at_fpr_1pct"], "tpr_at_fpr_5pct": metrics["tpr_at_fpr_5pct"]})
    print("\ndev metrics")
    print(pd.DataFrame(rows).to_string(index=False))
    # A metric change alone is not proof that the model used a silence shortcut.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
