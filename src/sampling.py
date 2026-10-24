"""Small deterministic sampling helpers for smoke tests and scoring."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def stratified_subset(
    frame: pd.DataFrame,
    n: int | None,
    seed: int = 42,
    by: Sequence[str] = ("label",),
) -> pd.DataFrame:
    """Take a reproducible proportional sample with at least one row per group."""
    if n is None or n >= len(frame):
        return frame.copy()
    if n < 0:
        raise ValueError("subset size must be non-negative")
    if n == 0 or frame.empty:
        return frame.iloc[:0].copy()
    columns = [column for column in by if column in frame.columns]
    if not columns:
        return frame.sample(n=n, random_state=seed).reset_index(drop=True)
    keys = frame[columns].copy()
    for column in columns:
        keys[column] = keys[column].fillna("<null>")
    grouped = {key: frame.loc[group.index] for key, group in keys.groupby(columns, dropna=False, sort=True)}
    if n < len(grouped):
        return frame.sample(n=n, random_state=seed).reset_index(drop=True)
    total = len(frame)
    quotas = {key: max(1, int(len(group) * n / total)) for key, group in grouped.items()}
    while sum(quotas.values()) > n:
        candidates = [key for key, quota in quotas.items() if quota > 1]
        key = max(candidates, key=lambda item: (quotas[item], str(item)))
        quotas[key] -= 1
    while sum(quotas.values()) < n:
        candidates = [key for key, group in grouped.items() if quotas[key] < len(group)]
        key = max(candidates, key=lambda item: (len(grouped[item]) * n / total - quotas[item], str(item)))
        quotas[key] += 1
    sampled = [
        group.sample(n=quotas[key], random_state=seed + index)
        for index, (key, group) in enumerate(grouped.items())
    ]
    return pd.concat(sampled).sample(frac=1.0, random_state=seed).reset_index(drop=True)
