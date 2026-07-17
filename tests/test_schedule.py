from __future__ import annotations

import pytest

from edgeci.schedule import generate_schedule


def _measured(seed: str = "schedule-test"):
    return [entry for entry in generate_schedule(seed=seed) if not entry.is_warmup]


def test_schedule_is_deterministic() -> None:
    assert generate_schedule(seed="same") == generate_schedule(seed="same")


def test_different_seeds_change_schedule() -> None:
    assert _measured("base-commit") != _measured("head-commit")


def test_default_schedule_has_twenty_of_each_arm() -> None:
    measured = _measured()

    assert len(measured) == 40
    assert sum(entry.arm == "base" for entry in measured) == 20
    assert sum(entry.arm == "head" for entry in measured) == 20
    assert {entry.pair_index for entry in measured} == set(range(20))


def test_each_block_is_abba_or_baab() -> None:
    measured = _measured()

    for block_index in range(10):
        block = [entry for entry in measured if entry.block_index == block_index]
        assert [entry.position for entry in block] == [0, 1, 2, 3]
        assert [entry.arm for entry in block] in [
            ["base", "head", "head", "base"],
            ["head", "base", "base", "head"],
        ]
        assert sum(entry.arm == "base" for entry in block) == 2
        assert sum(entry.arm == "head" for entry in block) == 2


def test_warmups_alternate_pair_order() -> None:
    warmups = [entry for entry in generate_schedule() if entry.is_warmup]

    assert [entry.arm for entry in warmups] == ["base", "head", "head", "base"]


@pytest.mark.parametrize("pairs", [1, 3, 19])
def test_odd_pair_count_is_rejected(pairs: int) -> None:
    with pytest.raises(ValueError, match="even"):
        generate_schedule(n_pairs=pairs)

