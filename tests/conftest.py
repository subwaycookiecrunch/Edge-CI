"""Shared pytest fixtures for EdgeCI."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def fixture_dir() -> Path:
    """Return directory containing static test fixtures."""

    return Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_jsonl(fixture_dir: Path) -> str:
    """Return realistic llama-bench JSONL fixture text."""

    return (fixture_dir / "sample_bench_output.jsonl").read_text(encoding="utf-8")


@pytest.fixture()
def sample_config_path(fixture_dir: Path) -> Path:
    """Return sample TOML configuration path."""

    return fixture_dir / "sample_config.toml"
