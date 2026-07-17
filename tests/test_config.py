"""Tests for EdgeCI TOML configuration loading."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from edgeci.config import ConfigError, EdgeCIConfig, apply_overrides, find_config, load_config


def test_default_config_values(tmp_path: Path) -> None:
    """Defaults implement the documented protocol."""

    config = load_config(start_dir=tmp_path)

    assert config == EdgeCIConfig()
    assert config.model.path is None
    assert config.benchmark.prompt_tokens == 512
    assert config.benchmark.generate_tokens == 128
    assert config.benchmark.pairs == 20
    assert config.benchmark.warmup_pairs == 2
    assert config.benchmark.gap_seconds == 15.0
    assert config.benchmark.timeout_minutes == 60.0
    assert config.budgets.tg == pytest.approx(0.05)
    assert config.budgets.pp == pytest.approx(0.05)
    assert config.preflight.thermal_settle_seconds == 60.0
    assert config.preflight.idle_cpu_threshold == pytest.approx(0.20)
    assert config.preflight.post_build_cooldown == 120.0
    assert config.preflight.preflight_timeout == 600.0
    assert config.report.format == "all"
    assert config.report.output_dir == Path("edgeci-results")
    with pytest.raises(FrozenInstanceError):
        config.benchmark.pairs = 10  # type: ignore[misc]


def test_loads_all_toml_sections_and_resolves_relative_paths(tmp_path: Path) -> None:
    """TOML values replace defaults and paths anchor to the TOML directory."""

    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        """
[model]
path = "models/tiny.gguf"

[benchmark]
prompt_tokens = 256
generate_tokens = 64
pairs = 12
warmup_pairs = 4
gap_seconds = 3.5
timeout_minutes = 30

[budgets]
tg = 0.03
pp = 0.07

[preflight]
thermal_settle_seconds = 10
idle_cpu_threshold = 0.15
post_build_cooldown = 45
preflight_timeout = 120

[report]
format = "json"
output_dir = "results"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model.path == (tmp_path / "models/tiny.gguf").resolve()
    assert config.benchmark.prompt_tokens == 256
    assert config.benchmark.generate_tokens == 64
    assert config.benchmark.pairs == 12
    assert config.benchmark.warmup_pairs == 4
    assert config.benchmark.gap_seconds == pytest.approx(3.5)
    assert config.benchmark.timeout_minutes == pytest.approx(30)
    assert config.budgets.tg == pytest.approx(0.03)
    assert config.budgets.pp == pytest.approx(0.07)
    assert config.preflight.thermal_settle_seconds == pytest.approx(10)
    assert config.preflight.idle_cpu_threshold == pytest.approx(0.15)
    assert config.preflight.post_build_cooldown == pytest.approx(45)
    assert config.preflight.preflight_timeout == pytest.approx(120)
    assert config.report.format == "json"
    assert config.report.output_dir == (tmp_path / "results").resolve()


def test_searches_current_and_parent_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nearest parent .edgeci.toml is found from a nested working directory."""

    config_path = tmp_path / ".edgeci.toml"
    config_path.write_text("[benchmark]\npairs = 8\n", encoding="utf-8")
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert find_config() == config_path
    assert load_config().benchmark.pairs == 8


def test_cli_overrides_take_precedence_without_erasing_file_values(
    tmp_path: Path,
) -> None:
    """Flat and nested CLI overrides win while ``None`` means unspecified."""

    config_path = tmp_path / ".edgeci.toml"
    config_path.write_text(
        """
[benchmark]
pairs = 8
gap_seconds = 9
[report]
format = "markdown"
""".strip(),
        encoding="utf-8",
    )
    model = tmp_path / "override.gguf"

    config = load_config(
        config_path,
        overrides={
            "model": model,
            "pairs": 14,
            "benchmark.gap_seconds": None,
            "preflight": {"idle_cpu_threshold": 0.1},
            "fmt": "terminal",
            "output": tmp_path / "cli-results",
        },
    )

    assert config.model.path == model
    assert config.benchmark.pairs == 14
    assert config.benchmark.gap_seconds == pytest.approx(9)
    assert config.preflight.idle_cpu_threshold == pytest.approx(0.1)
    assert config.report.format == "terminal"
    assert config.report.output_dir == tmp_path / "cli-results"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"pairs": 3}, "must be even"),
        ({"pairs": 2}, "at least 4"),
        ({"gap_seconds": float("nan")}, "must be finite"),
        ({"timeout_minutes": float("inf")}, "must be finite"),
        ({"budgets.tg": 1.0}, "less than 1"),
        ({"report.format": "xml"}, "must be one of"),
        ({"not_a_setting": 1}, "unknown configuration override"),
    ],
)
def test_rejects_invalid_overrides(override: dict[str, object], message: str) -> None:
    """Protocol-breaking and unknown values fail with actionable errors."""

    with pytest.raises(ConfigError, match=message):
        apply_overrides(EdgeCIConfig(), override)
