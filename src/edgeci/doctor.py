"""Structured testbed readiness checks for EdgeCI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import EdgeCIConfig
from .lock import EdgeCILock
from .probe import (
    HardwareFingerprint,
    MemoryPressure,
    ThermalState,
    detect_llama_bench,
    get_cpu_load,
    get_hardware_fingerprint,
    get_memory_state,
    get_power_state,
    get_thermal_state,
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one doctor readiness check."""

    name: str
    passed: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class DoctorReport:
    """Hardware fingerprint and all readiness check outcomes."""

    hardware: HardwareFingerprint
    checks: tuple[CheckResult, ...]

    @property
    def ready(self) -> bool:
        """Return whether every required check passed."""

        return all(check.passed for check in self.checks if check.required)

    def get_check(self, name: str) -> CheckResult | None:
        """Return a check by case-insensitive name, if present."""

        target = name.casefold()
        return next(
            (check for check in self.checks if check.name.casefold() == target),
            None,
        )


def run_doctor(
    config: EdgeCIConfig,
    *,
    binary_path: Path | str | None = None,
    cpu_sample_seconds: float = 1.0,
) -> DoctorReport:
    """Run all seven EdgeCI testbed readiness checks.

    Args:
        config: Thresholds and model path to validate.
        binary_path: Optional explicit llama-bench binary. Otherwise ``PATH``
            is searched.
        cpu_sample_seconds: Delay between CPU samples.

    Returns:
        A presentation-independent doctor report.
    """

    with EdgeCILock():
        return _run_doctor_unlocked(
            config,
            binary_path=binary_path,
            cpu_sample_seconds=cpu_sample_seconds,
        )


def _run_doctor_unlocked(
    config: EdgeCIConfig,
    *,
    binary_path: Path | str | None,
    cpu_sample_seconds: float,
) -> DoctorReport:
    """Collect readiness evidence while the caller owns the machine lock."""

    hardware = get_hardware_fingerprint()
    checks: list[CheckResult] = []

    hardware_ready = hardware.is_apple_silicon
    checks.append(
        CheckResult(
            name="Hardware",
            passed=hardware_ready,
            detail=(
                hardware.summary
                if hardware_ready
                else f"unsupported host: {hardware.chip_name}; Apple Silicon required"
            ),
        )
    )

    thermal = get_thermal_state()
    thermal_name = str(getattr(thermal, "name", thermal)).casefold()
    thermal_ready = thermal == ThermalState.NOMINAL or thermal_name == "nominal"
    checks.append(
        CheckResult(
            name="Thermal",
            passed=thermal_ready,
            detail=f"thermal state: {thermal_name}",
        )
    )

    power = get_power_state()
    power_ready = power.ac_connected and not power.low_power_mode
    source = "AC power" if power.ac_connected else "battery power or unknown source"
    lpm = "Low Power Mode on" if power.low_power_mode else "Low Power Mode off"
    checks.append(
        CheckResult(
            name="Power",
            passed=power_ready,
            detail=f"{source}; {lpm}",
        )
    )

    memory = get_memory_state()
    pressure = memory.pressure_level
    pressure_name = str(getattr(pressure, "value", pressure)).casefold()
    checks.append(
        CheckResult(
            name="Memory",
            passed=pressure == MemoryPressure.NORMAL or pressure_name == "normal",
            detail=(
                f"pressure: {pressure_name}; "
                f"available: {_format_bytes(memory.available_bytes)}; "
                f"pageouts: {memory.pageouts}"
            ),
        )
    )

    cpu_busy = get_cpu_load(cpu_sample_seconds)
    cpu_threshold = config.preflight.idle_cpu_threshold
    checks.append(
        CheckResult(
            name="CPU",
            passed=cpu_busy < cpu_threshold,
            detail=(
                f"busy: {cpu_busy:.1%}; required below {cpu_threshold:.1%}"
            ),
        )
    )

    llama_bench = detect_llama_bench(binary_path)
    checks.append(
        CheckResult(
            name="llama-bench",
            passed=llama_bench is not None,
            detail=(
                f"{llama_bench.path} ({llama_bench.version})"
                if llama_bench is not None
                else "llama-bench not found or not executable"
            ),
        )
    )

    configured_model_path = config.model.path
    model_path = Path(configured_model_path) if configured_model_path is not None else None
    model_ready, model_detail = _check_model(model_path)
    checks.append(
        CheckResult(
            name="Model",
            passed=model_ready,
            detail=model_detail,
        )
    )

    return DoctorReport(hardware=hardware, checks=tuple(checks))


def _format_bytes(value: int) -> str:
    if value <= 0:
        return "unknown"
    gib = value / (1024**3)
    return f"{gib:.1f} GiB"


def _check_model(model_path: Path | None) -> tuple[bool, str]:
    """Validate configured model existence, readability, and GGUF header."""

    if model_path is None:
        return False, "no GGUF model configured"
    if not model_path.is_file():
        return False, f"model file not found: {model_path}"
    try:
        with model_path.open("rb") as stream:
            magic = stream.read(4)
    except OSError as exc:
        return False, f"model file is not readable: {model_path} ({exc})"
    if magic != b"GGUF":
        return False, f"model file does not have GGUF header: {model_path}"
    return True, str(model_path)


__all__ = ("CheckResult", "DoctorReport", "run_doctor")
