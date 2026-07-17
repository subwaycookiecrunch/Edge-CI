"""CLI exit behavior for shadow and enforced verdicts."""

from __future__ import annotations

import click
import pytest

from edgeci.cli import _exit_for_verdict
from edgeci.verdict import SessionVerdict


def _verdict(overall: str, *, experimental: bool) -> SessionVerdict:
    return SessionVerdict(
        overall=overall,
        metrics=[],
        experimental=experimental,
        abort_reason=None,
        warnings=[],
    )


@pytest.mark.parametrize("overall", ["PASS", "FAIL", "INCONCLUSIVE"])
def test_experimental_compare_verdicts_are_shadow_only(overall: str) -> None:
    _exit_for_verdict(
        _verdict(overall, experimental=True),
        shadow_if_experimental=True,
    )


@pytest.mark.parametrize(
    ("overall", "expected_code"),
    [("FAIL", 1), ("INCONCLUSIVE", 2)],
)
def test_enrolled_compare_verdicts_block(overall: str, expected_code: int) -> None:
    with pytest.raises(click.exceptions.Exit) as raised:
        _exit_for_verdict(
            _verdict(overall, experimental=False),
            shadow_if_experimental=True,
        )

    assert raised.value.exit_code == expected_code


@pytest.mark.parametrize(
    ("overall", "expected_code"),
    [("FAIL", 1), ("INCONCLUSIVE", 2)],
)
def test_calibration_still_enforces_experimental_verdicts(
    overall: str,
    expected_code: int,
) -> None:
    with pytest.raises(click.exceptions.Exit) as raised:
        _exit_for_verdict(
            _verdict(overall, experimental=True),
            shadow_if_experimental=False,
        )

    assert raised.value.exit_code == expected_code
