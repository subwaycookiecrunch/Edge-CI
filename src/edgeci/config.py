"""Typed configuration loading for EdgeCI."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:  # pragma: no cover - branch depends on interpreter version
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]


CONFIG_FILENAME = ".edgeci.toml"
REPORT_FORMATS = frozenset({"terminal", "markdown", "json", "all"})


class ConfigError(ValueError):
    """Raised when an EdgeCI configuration is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    """Model input configuration.

    Attributes:
        path: Path to a local GGUF model, or ``None`` when not configured.
    """

    path: Path | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parameters passed to each benchmark invocation."""

    prompt_tokens: int = 512
    generate_tokens: int = 128
    pairs: int = 20
    warmup_pairs: int = 2
    gap_seconds: float = 15.0
    timeout_minutes: float = 60.0


@dataclass(frozen=True)
class BudgetsConfig:
    """Maximum tolerated throughput regressions as fractions."""

    tg: float = 0.05
    pp: float = 0.05


@dataclass(frozen=True)
class PreflightConfig:
    """Health thresholds used before and between measurements."""

    thermal_settle_seconds: float = 60.0
    idle_cpu_threshold: float = 0.20
    post_build_cooldown: float = 120.0
    preflight_timeout: float = 600.0


@dataclass(frozen=True)
class ReportConfig:
    """Result serialization preferences."""

    format: str = "all"
    output_dir: Path = Path("./edgeci-results")


@dataclass(frozen=True)
class EdgeCIConfig:
    """Complete immutable EdgeCI configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    budgets: BudgetsConfig = field(default_factory=BudgetsConfig)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    @property
    def model_path(self) -> Path | None:
        """Return the configured model path."""

        return self.model.path

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of this configuration."""

        result = asdict(self)
        model_path = result["model"]["path"]
        output_dir = result["report"]["output_dir"]
        result["model"]["path"] = str(model_path) if model_path is not None else None
        result["report"]["output_dir"] = str(output_dir)
        return result


_SECTION_KEYS: Mapping[str, frozenset[str]] = MappingProxyType({
    "model": frozenset({"path"}),
    "benchmark": frozenset(
        {
            "prompt_tokens",
            "generate_tokens",
            "pairs",
            "warmup_pairs",
            "gap_seconds",
            "timeout_minutes",
        }
    ),
    "budgets": frozenset({"tg", "pp"}),
    "preflight": frozenset(
        {
            "thermal_settle_seconds",
            "idle_cpu_threshold",
            "post_build_cooldown",
            "preflight_timeout",
        }
    ),
    "report": frozenset({"format", "output_dir"}),
})


def find_config(
    start_dir: Path | str | None = None,
    filename: str = CONFIG_FILENAME,
) -> Path | None:
    """Find a configuration file in a directory or one of its parents.

    Args:
        start_dir: Search starting point. Defaults to the current directory.
            A file path starts the search in its parent directory.
        filename: Configuration filename to locate.

    Returns:
        Resolved path to the nearest configuration file, or ``None``.
    """

    current = Path.cwd() if start_dir is None else Path(start_dir).expanduser()
    if current.exists() and current.is_file():
        current = current.parent
    current = current.resolve()
    while True:
        candidate = current / filename
        if candidate.is_file():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


def load_config(
    path: Path | str | None = None,
    *,
    start_dir: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> EdgeCIConfig:
    """Load, validate, and optionally override EdgeCI configuration.

    When ``path`` is omitted, the nearest ``.edgeci.toml`` is searched for
    from ``start_dir`` (or the current directory). No discovered file means
    the documented defaults are returned.

    Args:
        path: Explicit TOML file. Missing explicit files are errors.
        start_dir: Starting directory for implicit parent search.
        overrides: CLI-style overrides, either nested or dot-delimited.
        cli_overrides: Backward-compatible alias for ``overrides``.

    Returns:
        Validated immutable configuration.

    Raises:
        ConfigError: If TOML, keys, values, or override syntax are invalid.
    """

    if overrides is not None and cli_overrides is not None:
        raise ConfigError("provide only one of overrides or cli_overrides")
    selected_overrides = overrides if overrides is not None else cli_overrides

    config_path: Path | None
    if path is None:
        config_path = find_config(start_dir)
    else:
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise ConfigError(f"configuration file does not exist: {config_path}")

    raw: Mapping[str, Any] = {}
    if config_path is not None:
        try:
            with config_path.open("rb") as stream:
                loaded = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot load configuration {config_path}: {exc}") from exc
        raw = loaded

    config = _config_from_mapping(
        raw,
        base_dir=config_path.parent if config_path is not None else None,
    )
    if selected_overrides:
        config = apply_overrides(config, selected_overrides)
    return config


def apply_overrides(
    config: EdgeCIConfig,
    overrides: Mapping[str, Any],
) -> EdgeCIConfig:
    """Return a new configuration with CLI values applied.

    ``None`` values are ignored so an unspecified Click option does not erase
    a file value. Keys may be nested (``{"benchmark": {"pairs": 10}}``),
    dot-delimited (``{"benchmark.pairs": 10}``), or common CLI aliases such
    as ``model``, ``output``, and ``format``.

    Args:
        config: Base configuration.
        overrides: Override mapping.

    Returns:
        A validated replacement configuration.

    Raises:
        ConfigError: If a key or value is invalid.
    """

    merged = config.to_dict()
    for raw_key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and raw_key in _SECTION_KEYS:
            for child_key, child_value in value.items():
                if child_value is not None:
                    _set_override(merged, f"{raw_key}.{child_key}", child_value)
            continue
        _set_override(merged, _canonical_override_key(raw_key), value)
    return _config_from_mapping(merged)


def _canonical_override_key(key: str) -> str:
    aliases = {
        "model": "model.path",
        "output": "report.output_dir",
        "output_dir": "report.output_dir",
        "format": "report.format",
        "fmt": "report.format",
    }
    if key in aliases:
        return aliases[key]
    if "." in key:
        return key
    matches = [section for section, keys in _SECTION_KEYS.items() if key in keys]
    if len(matches) == 1:
        return f"{matches[0]}.{key}"
    return key


def _set_override(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if len(parts) != 2 or parts[0] not in _SECTION_KEYS:
        raise ConfigError(f"unknown configuration override: {key}")
    section, field_name = parts
    if field_name not in _SECTION_KEYS[section]:
        raise ConfigError(f"unknown configuration override: {key}")
    data[section][field_name] = value


def _config_from_mapping(
    raw: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> EdgeCIConfig:
    _validate_mapping_shape(raw)
    model = raw.get("model", {})
    benchmark = raw.get("benchmark", {})
    budgets = raw.get("budgets", {})
    preflight = raw.get("preflight", {})
    report = raw.get("report", {})

    model_path_value = model.get("path")
    if model_path_value is not None and not isinstance(model_path_value, (str, Path)):
        raise ConfigError("model.path must be a filesystem path string")
    model_path = None
    if model_path_value not in (None, ""):
        model_path = _path(model_path_value, "model.path")
        if base_dir is not None and not model_path.is_absolute():
            model_path = (base_dir / model_path).resolve()

    config = EdgeCIConfig(
        model=ModelConfig(path=model_path),
        benchmark=BenchmarkConfig(
            prompt_tokens=_positive_int(
                benchmark.get("prompt_tokens", BenchmarkConfig.prompt_tokens),
                "benchmark.prompt_tokens",
            ),
            generate_tokens=_positive_int(
                benchmark.get("generate_tokens", BenchmarkConfig.generate_tokens),
                "benchmark.generate_tokens",
            ),
            pairs=_positive_int(
                benchmark.get("pairs", BenchmarkConfig.pairs), "benchmark.pairs"
            ),
            warmup_pairs=_nonnegative_int(
                benchmark.get("warmup_pairs", BenchmarkConfig.warmup_pairs),
                "benchmark.warmup_pairs",
            ),
            gap_seconds=_nonnegative_float(
                benchmark.get("gap_seconds", BenchmarkConfig.gap_seconds),
                "benchmark.gap_seconds",
            ),
            timeout_minutes=_positive_float(
                benchmark.get("timeout_minutes", BenchmarkConfig.timeout_minutes),
                "benchmark.timeout_minutes",
            ),
        ),
        budgets=BudgetsConfig(
            tg=_budget(budgets.get("tg", BudgetsConfig.tg), "budgets.tg"),
            pp=_budget(budgets.get("pp", BudgetsConfig.pp), "budgets.pp"),
        ),
        preflight=PreflightConfig(
            thermal_settle_seconds=_nonnegative_float(
                preflight.get(
                    "thermal_settle_seconds", PreflightConfig.thermal_settle_seconds
                ),
                "preflight.thermal_settle_seconds",
            ),
            idle_cpu_threshold=_fraction(
                preflight.get("idle_cpu_threshold", PreflightConfig.idle_cpu_threshold),
                "preflight.idle_cpu_threshold",
            ),
            post_build_cooldown=_nonnegative_float(
                preflight.get("post_build_cooldown", PreflightConfig.post_build_cooldown),
                "preflight.post_build_cooldown",
            ),
            preflight_timeout=_positive_float(
                preflight.get("preflight_timeout", PreflightConfig.preflight_timeout),
                "preflight.preflight_timeout",
            ),
        ),
        report=ReportConfig(
            format=_report_format(report.get("format", ReportConfig.format)),
            output_dir=_config_path(
                report.get("output_dir", ReportConfig.output_dir),
                "report.output_dir",
                base_dir,
            ),
        ),
    )
    if config.benchmark.pairs % 2:
        raise ConfigError(
            "benchmark.pairs must be even so every block contains AB and BA pairs"
        )
    if config.benchmark.pairs < 4:
        raise ConfigError(
            "benchmark.pairs must be at least 4 to form two balanced blocks"
        )
    return config


def _validate_mapping_shape(raw: Mapping[str, Any]) -> None:
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a TOML table")
    unknown_sections = set(raw) - set(_SECTION_KEYS)
    if unknown_sections:
        names = ", ".join(sorted(map(str, unknown_sections)))
        raise ConfigError(f"unknown configuration section(s): {names}")
    for section, allowed_keys in _SECTION_KEYS.items():
        values = raw.get(section, {})
        if not isinstance(values, Mapping):
            raise ConfigError(f"{section} must be a TOML table")
        unknown_keys = set(values) - set(allowed_keys)
        if unknown_keys:
            names = ", ".join(sorted(map(str, unknown_keys)))
            raise ConfigError(f"unknown key(s) in [{section}]: {names}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _number(value, name)
    if number <= 0:
        raise ConfigError(f"{name} must be greater than zero")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = _number(value, name)
    if number < 0:
        raise ConfigError(f"{name} must be zero or greater")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return value


def _budget(value: Any, name: str) -> float:
    number = _number(value, name)
    if not 0 <= number < 1:
        raise ConfigError(f"{name} must be at least 0 and less than 1")
    return number


def _fraction(value: Any, name: str) -> float:
    number = _number(value, name)
    if not 0 <= number <= 1:
        raise ConfigError(f"{name} must be between 0 and 1")
    return number


def _report_format(value: Any) -> str:
    if not isinstance(value, str) or value not in REPORT_FORMATS:
        choices = ", ".join(sorted(REPORT_FORMATS))
        raise ConfigError(f"report.format must be one of: {choices}")
    return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or str(value) == "":
        raise ConfigError(f"{name} must be a non-empty filesystem path")
    return Path(value).expanduser()


def _config_path(value: Any, name: str, base_dir: Path | None) -> Path:
    path = _path(value, name)
    if base_dir is not None and not path.is_absolute():
        return (base_dir / path).resolve()
    return path
