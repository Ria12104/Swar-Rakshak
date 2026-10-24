"""Score a manifest with a CNN checkpoint using mean window logits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import load_cnn_checkpoint, score_manifest_dataframe  # noqa: E402
from src.sampling import stratified_subset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="scores.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratified-subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    if args.limit is not None:
        frame = stratified_subset(frame, args.limit, args.seed, ("label",))
    if args.stratified_subset is not None:
        frame = stratified_subset(frame, args.stratified_subset, args.seed, ("label", "attack_id"))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model = load_cnn_checkpoint(args.checkpoint, device)
    scores = score_manifest_dataframe(frame, model, device, args.batch_size, args.trim_silence)
    scores.to_csv(args.output, index=False)
    print(f"wrote {len(scores)} scores to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
