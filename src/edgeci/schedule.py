"""Deterministic order-balanced benchmark schedule generation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Invocation:
    """One planned base or head binary invocation."""

    arm: str
    pair_index: int
    block_index: int
    position: int
    is_warmup: bool


def _validate_count(value: int, name: str, *, allow_zero: bool) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")


def generate_schedule(
    n_pairs: int = 20,
    n_warmup_pairs: int = 2,
    seed: str = "",
) -> list[Invocation]:
    """Create the complete warm-up and measured ABBA/BAAB schedule.

    Measured pairs are grouped two at a time.  Every block therefore contains
    one base-first pair and one head-first pair, while the seeded choice controls
    whether the block is ABBA or BAAB.

    Args:
        n_pairs: Positive, even number of measured base/head pairs.
        n_warmup_pairs: Number of discarded warm-up pairs.
        seed: Arbitrary deterministic schedule identifier.

    Returns:
        Fully materialized invocation order, with warm-ups first.

    Raises:
        TypeError: If a count or seed has the wrong type.
        ValueError: If the measured pair count is odd or a count is invalid.
    """

    _validate_count(n_pairs, "n_pairs", allow_zero=False)
    _validate_count(n_warmup_pairs, "n_warmup_pairs", allow_zero=True)
    if n_pairs % 2:
        raise ValueError("n_pairs must be even to form balanced ABBA/BAAB blocks")
    if not isinstance(seed, str):
        raise TypeError("seed must be a string")

    schedule: list[Invocation] = []
    for warmup_index in range(n_warmup_pairs):
        arms = ("base", "head") if warmup_index % 2 == 0 else ("head", "base")
        for position, arm in enumerate(arms):
            schedule.append(
                Invocation(
                    arm=arm,
                    pair_index=-(warmup_index + 1),
                    block_index=-(warmup_index + 1),
                    position=position,
                    is_warmup=True,
                )
            )

    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest, byteorder="big", signed=False))
    for block_index in range(n_pairs // 2):
        first_pair_index = block_index * 2
        if rng.getrandbits(1) == 0:
            arms = ("base", "head", "head", "base")
        else:
            arms = ("head", "base", "base", "head")

        for position, arm in enumerate(arms):
            pair_index = first_pair_index if position < 2 else first_pair_index + 1
            schedule.append(
                Invocation(
                    arm=arm,
                    pair_index=pair_index,
                    block_index=block_index,
                    position=position,
                    is_warmup=False,
                )
            )
    return schedule

