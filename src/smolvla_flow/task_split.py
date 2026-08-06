"""Deterministic episode splitting and full action-window accounting."""

from __future__ import annotations

import random
from typing import Iterable


def count_full_windows(length: int, *, chunk_size: int = 50, stride: int = 4) -> int:
    """Return the number of complete ``chunk_size`` windows in one episode."""

    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if chunk_size < 1 or stride < 1:
        raise ValueError("chunk_size and stride must be positive")
    if length < chunk_size:
        return 0
    return 1 + (length - chunk_size) // stride


def split_episode_indices(
    episode_indices: Iterable[int],
    *,
    seed: int = 0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[int]]:
    """Create a deterministic, disjoint train/validation/test split.

    The largest-remainder allocation keeps the requested ratios close while
    making the 45 task-0 episodes a 31/7/7 split.  Episode IDs are shuffled
    with a local RNG so global random state is untouched.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0.0 <= train_ratio <= 1.0 or not 0.0 <= val_ratio <= 1.0:
        raise ValueError("train_ratio and val_ratio must be in [0, 1]")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    ordered = sorted(int(index) for index in episode_indices)
    if len(set(ordered)) != len(ordered):
        raise ValueError("episode_indices must be unique")
    if len(ordered) < 3:
        raise ValueError("at least three episodes are required for train/val/test")

    fractions = (train_ratio, val_ratio, 1.0 - train_ratio - val_ratio)
    raw_counts = [len(ordered) * fraction for fraction in fractions]
    counts = [int(value) for value in raw_counts]
    remainder = len(ordered) - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    if any(count == 0 for count in counts):
        raise ValueError("ratios must allocate at least one episode to every split")

    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    train_end = counts[0]
    val_end = train_end + counts[1]
    return {
        "train": sorted(shuffled[:train_end]),
        "validation": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


__all__ = ["count_full_windows", "split_episode_indices"]
