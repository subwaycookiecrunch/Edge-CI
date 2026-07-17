"""Read-only macOS environment sensors used by EdgeCI."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import re
import signal
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable


class ProbeError(RuntimeError):
    """Raised when a requested environment value cannot be parsed."""


class ProbeDeadlineError(ProbeError):
    """Raised when a sensor subprocess reaches a comparison deadline."""


class ThermalState(IntEnum):
    """Values reported by ``NSProcessInfo.thermalState``."""

    UNKNOWN = -1
    NOMINAL = 0
    FAIR = 1
    SERIOUS = 2
    CRITICAL = 3


class MemoryPressure(str, Enum):
    """Coarse memory pressure derived from ``vm_stat``."""

    UNKNOWN = "unknown"
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class PowerState:
    """Current power-source and Low Power Mode state."""

    ac_connected: bool
    low_power_mode: bool


@dataclass(frozen=True)
class MemoryState:
    """Virtual-memory counters and a conservative pressure assessment."""

    page_size: int
    free_pages: int
    pageouts: int
    compressed_pages: int
    inactive_pages: int = 0
    speculative_pages: int = 0
    total_bytes: int = 0
    pressure_level: MemoryPressure = MemoryPressure.UNKNOWN

    @property
    def free_bytes(self) -> int:
        """Return completely free memory in bytes."""

        return self.free_pages * self.page_size

    @property
    def available_bytes(self) -> int:
        """Return free, inactive, and speculative memory in bytes."""

        pages = self.free_pages + self.inactive_pages + self.speculative_pages
        return pages * self.page_size

    @property
    def compressed_bytes(self) -> int:
        """Return memory held by the compressor in bytes."""

        return self.compressed_pages * self.page_size

    @property
    def has_pressure(self) -> bool:
        """Return whether memory is pressured or could not be assessed."""

        return self.pressure_level is not MemoryPressure.NORMAL


@dataclass(frozen=True)
class HardwareFingerprint:
    """Stable hardware and operating-system provenance for a run."""

    chip_name: str
    logical_cpu_count: int
    performance_core_count: int
    efficiency_core_count: int
    memory_bytes: int
    gpu_core_count: int
    macos_version: str
    os_build: str
    model_identifier: str

    @property
    def is_apple_silicon(self) -> bool:
        """Return whether the fingerprint describes an Apple Silicon Mac."""

        chip = self.chip_name.casefold()
        arm_machine = platform.machine().casefold() in {"arm64", "aarch64"}
        return arm_machine and (
            chip.startswith("apple ") or platform.system() == "Darwin"
        )

    @property
    def summary(self) -> str:
        """Return a compact human-readable hardware summary."""

        memory_gib = self.memory_bytes / (1024**3) if self.memory_bytes else 0.0
        memory_text = f"{memory_gib:.0f} GB" if memory_gib else "unknown memory"
        gpu_text = f"{self.gpu_core_count}-core GPU" if self.gpu_core_count else "GPU unknown"
        os_text = f"macOS {self.macos_version}" if self.macos_version else "macOS unknown"
        return f"{self.chip_name} · {memory_text} · {gpu_text} · {os_text}"


@dataclass(frozen=True)
class LlamaBenchInfo:
    """Detected llama-bench executable, build text, and CLI capabilities."""

    path: Path
    version: str
    help_text: str = ""


class _MachTimebaseInfo(ctypes.Structure):
    """Native scale factors returned by ``mach_timebase_info``."""

    _fields_ = (("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32))


def make_continuous_clock() -> Callable[[], float]:
    """Return a monotonic clock that advances while macOS is asleep.

    ``time.monotonic()`` maps to an uptime clock on macOS and pauses during
    system sleep. EdgeCI hard deadlines instead use ``mach_continuous_time``.
    Non-macOS hosts use ``time.monotonic`` so parsing and unit tests remain
    portable, although benchmarking itself is macOS-only.

    Returns:
        Zero-argument callable returning continuous seconds.

    Raises:
        ProbeError: If the native continuous clock cannot be initialized on
            macOS.
    """

    if platform.system() != "Darwin":
        return time.monotonic
    try:
        library_path = ctypes.util.find_library("System")
        if not library_path:
            library_path = "/usr/lib/libSystem.B.dylib"
        library = ctypes.CDLL(library_path)
        counter = library.mach_continuous_time
        counter.argtypes = ()
        counter.restype = ctypes.c_uint64
        timebase = library.mach_timebase_info
        timebase.argtypes = (ctypes.POINTER(_MachTimebaseInfo),)
        timebase.restype = ctypes.c_int
        info = _MachTimebaseInfo()
        if timebase(ctypes.byref(info)) != 0 or info.denom == 0:
            raise ProbeError("mach_timebase_info failed")
        scale = info.numer / info.denom / 1_000_000_000.0
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise ProbeError(f"cannot initialize mach_continuous_time: {exc}") from exc

    def continuous_seconds() -> float:
        return float(counter()) * scale

    return continuous_seconds


def get_thermal_state() -> ThermalState:
    """Return current macOS thermal state.

    Native binding failures return ``ThermalState.UNKNOWN`` so callers can
    fail closed without crashing on unsupported hosts.
    """

    value = _process_info_integer("thermalState")
    try:
        return ThermalState(value)
    except (TypeError, ValueError):
        return ThermalState.UNKNOWN


def get_power_state(
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> PowerState:
    """Return AC connection and Low Power Mode state.

    Args:
        deadline: Optional absolute comparison deadline.
        continuous_now: Sleep-inclusive clock associated with ``deadline``.
    """

    try:
        output = _run_text(
            ("pmset", "-g", "ps"),
            timeout=5.0,
            deadline=deadline,
            continuous_now=continuous_now,
        )
        ac_connected = bool(re.search(r"(?:Now drawing from|')\s*'?AC Power", output, re.I))
    except ProbeDeadlineError:
        raise
    except ProbeError:
        ac_connected = False

    low_power_value = _process_info_bool("isLowPowerModeEnabled")
    if low_power_value is None:
        low_power_value = _pmset_low_power_mode(
            deadline=deadline, continuous_now=continuous_now
        )
    # Unknown LPM state fails closed: treating it as enabled prevents a run.
    low_power_mode = True if low_power_value is None else low_power_value
    return PowerState(ac_connected=ac_connected, low_power_mode=low_power_mode)


def get_memory_state(
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> MemoryState:
    """Parse ``vm_stat`` and assess current memory pressure.

    Args:
        deadline: Optional absolute comparison deadline.
        continuous_now: Sleep-inclusive clock associated with ``deadline``.
    """

    try:
        output = _run_text(
            ("vm_stat",),
            timeout=5.0,
            deadline=deadline,
            continuous_now=continuous_now,
        )
        page_size, counters = _parse_vm_stat(output)
    except ProbeDeadlineError:
        raise
    except ProbeError:
        return MemoryState(
            page_size=0,
            free_pages=0,
            pageouts=0,
            compressed_pages=0,
            pressure_level=MemoryPressure.UNKNOWN,
        )

    total_bytes = _sysctl_int(
        "hw.memsize", deadline=deadline, continuous_now=continuous_now
    ) or _physical_memory_fallback()
    free_pages = counters.get("Pages free", 0)
    inactive_pages = counters.get("Pages inactive", 0)
    speculative_pages = counters.get("Pages speculative", 0)
    available_pages = free_pages + inactive_pages + speculative_pages
    pressure = _assess_memory_pressure(available_pages, page_size, total_bytes)
    return MemoryState(
        page_size=page_size,
        free_pages=free_pages,
        pageouts=counters.get("Pageouts", 0),
        compressed_pages=counters.get("Pages occupied by compressor", 0),
        inactive_pages=inactive_pages,
        speculative_pages=speculative_pages,
        total_bytes=total_bytes,
        pressure_level=pressure,
    )


def get_cpu_load(
    sample_seconds: float = 1.0,
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> float:
    """Return CPU busy fraction from the second ``top`` sample.

    Args:
        sample_seconds: Delay between the two samples.
        deadline: Optional absolute comparison deadline.
        continuous_now: Sleep-inclusive clock associated with ``deadline``.

    Returns:
        Busy CPU fraction in the inclusive range 0.0 through 1.0. Sensor
        failure returns 1.0 so health gates fail conservatively.
    """

    if sample_seconds < 0:
        raise ValueError("sample_seconds must be zero or greater")
    interval = max(sample_seconds, 0.1)
    try:
        output = _run_text(
            ("top", "-l", "2", "-n", "0", "-s", f"{interval:g}"),
            timeout=max(10.0, interval * 3.0 + 5.0),
            deadline=deadline,
            continuous_now=continuous_now,
        )
        idle_matches = re.findall(
            r"CPU usage:.*?([0-9]+(?:\.[0-9]+)?)%\s+idle", output, re.I
        )
        if len(idle_matches) < 2:
            raise ProbeError("top output did not contain two CPU samples")
        idle_fraction = float(idle_matches[-1]) / 100.0
        return min(1.0, max(0.0, 1.0 - idle_fraction))
    except ProbeDeadlineError:
        raise
    except (ProbeError, ValueError):
        return 1.0


def get_cpu_busy_fraction(
    sample_interval: float = 1.0,
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> float:
    """Compatibility wrapper returning CPU busy fraction."""

    return get_cpu_load(
        sample_interval, deadline=deadline, continuous_now=continuous_now
    )


def get_hardware_fingerprint(
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> HardwareFingerprint:
    """Collect hardware and operating-system provenance using macOS tools.

    Args:
        deadline: Optional absolute comparison deadline.
        continuous_now: Sleep-inclusive clock associated with ``deadline``.
    """

    overview = _hardware_overview(
        deadline=deadline, continuous_now=continuous_now
    )
    chip_name = _sysctl_text(
        "machdep.cpu.brand_string",
        deadline=deadline,
        continuous_now=continuous_now,
    )
    if not chip_name:
        chip_name = str(overview.get("chip_type", ""))
    if not chip_name:
        chip_name = platform.processor() or "Unknown chip"
    overview_cpu_counts = _overview_cpu_counts(overview)
    logical_cpus = (
        _sysctl_int(
            "hw.ncpu", deadline=deadline, continuous_now=continuous_now
        )
        or overview_cpu_counts[0]
        or (os.cpu_count() or 0)
    )
    performance_cpus = _sysctl_int(
        "hw.perflevel0.logicalcpu",
        deadline=deadline,
        continuous_now=continuous_now,
    )
    if performance_cpus <= 0:
        performance_cpus = overview_cpu_counts[1] or logical_cpus
    efficiency_cpus = _sysctl_int(
        "hw.perflevel1.logicalcpu",
        deadline=deadline,
        continuous_now=continuous_now,
    )
    if efficiency_cpus <= 0:
        efficiency_cpus = overview_cpu_counts[2]
    if efficiency_cpus <= 0 and logical_cpus > performance_cpus:
        efficiency_cpus = logical_cpus - performance_cpus
    memory_bytes = (
        _sysctl_int(
            "hw.memsize", deadline=deadline, continuous_now=continuous_now
        )
        or _parse_memory_size(str(overview.get("physical_memory", "")))
        or _physical_memory_fallback()
    )
    os_build = _sysctl_text(
        "kern.osversion", deadline=deadline, continuous_now=continuous_now
    ) or _sw_vers_build(deadline=deadline, continuous_now=continuous_now)
    model_identifier = _sysctl_text(
        "hw.model", deadline=deadline, continuous_now=continuous_now
    ) or str(
        overview.get("machine_model", "")
    )
    return HardwareFingerprint(
        chip_name=chip_name,
        logical_cpu_count=logical_cpus,
        performance_core_count=performance_cpus,
        efficiency_core_count=max(0, efficiency_cpus),
        memory_bytes=memory_bytes,
        gpu_core_count=_gpu_core_count(
            deadline=deadline, continuous_now=continuous_now
        ),
        macos_version=platform.mac_ver()[0],
        os_build=os_build,
        model_identifier=model_identifier,
    )


def detect_llama_bench(
    binary_path: Path | str | None = None,
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> LlamaBenchInfo | None:
    """Locate llama-bench and extract its version from ``--help`` output.

    Args:
        binary_path: Optional explicit executable. When absent, ``PATH`` is
            searched.
        deadline: Optional absolute comparison deadline.
        continuous_now: Sleep-inclusive clock associated with ``deadline``.

    Returns:
        Executable information, or ``None`` when no runnable binary exists.
    """

    if binary_path is None:
        discovered = shutil.which("llama-bench")
        if discovered is None:
            return None
        path = Path(discovered)
    else:
        candidate = Path(binary_path).expanduser()
        discovered = shutil.which(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            path = candidate
        elif discovered is not None:
            path = Path(discovered)
        else:
            return None

    path = path.resolve()
    try:
        completed = _run_command(
            (str(path), "--help"),
            timeout=10.0,
            deadline=deadline,
            continuous_now=continuous_now,
        )
        help_text = "\n".join((completed.stdout, completed.stderr)).strip()
    except ProbeDeadlineError:
        raise
    except ProbeError:
        return None
    if completed.returncode != 0:
        return None
    recognizable = re.search(r"\bllama-bench\b", help_text, re.I) or all(
        marker in help_text for marker in ("--n-prompt", "--n-gen", "--output")
    )
    if not recognizable:
        return None
    return LlamaBenchInfo(
        path=path,
        version=_extract_llama_bench_version(help_text),
        help_text=help_text,
    )


def wait_for_thermal_nominal(
    settle_seconds: float,
    timeout: float,
) -> tuple[bool, float]:
    """Wait for a continuous period of nominal thermal state.

    Any non-nominal or unknown reading resets the settlement timer. The sensor
    is polled at approximately 1 Hz.

    Args:
        settle_seconds: Required uninterrupted nominal duration.
        timeout: Overall deadline in seconds.

    Returns:
        ``(success, elapsed_seconds)``.
    """

    if settle_seconds < 0:
        raise ValueError("settle_seconds must be zero or greater")
    if timeout < 0:
        raise ValueError("timeout must be zero or greater")

    start = time.monotonic()
    nominal_since: float | None = None
    while True:
        state = get_thermal_state()
        now = time.monotonic()
        elapsed = now - start
        if state == ThermalState.NOMINAL:
            if nominal_since is None:
                nominal_since = now
            if now - nominal_since >= settle_seconds:
                return True, elapsed
        else:
            nominal_since = None

        if elapsed >= timeout:
            return False, elapsed
        time.sleep(min(1.0, max(0.0, timeout - elapsed)))


def _process_info_integer(selector_name: str) -> int | None:
    value = _process_info_message(selector_name, ctypes.c_long)
    return int(value) if value is not None else None


def _process_info_bool(selector_name: str) -> bool | None:
    value = _process_info_message(selector_name, ctypes.c_bool)
    return bool(value) if value is not None else None


def _process_info_message(selector_name: str, result_type: Any) -> Any | None:
    if platform.system() != "Darwin":
        return None
    try:
        foundation_path = ctypes.util.find_library("Foundation")
        objc_path = ctypes.util.find_library("objc")
        if not foundation_path or not objc_path:
            return None
        ctypes.CDLL(foundation_path)
        objc = ctypes.CDLL(objc_path)

        objc.objc_getClass.argtypes = (ctypes.c_char_p,)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = (ctypes.c_char_p,)
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        objc.objc_msgSend.restype = ctypes.c_void_p

        process_info_class = objc.objc_getClass(b"NSProcessInfo")
        process_info_selector = objc.sel_registerName(b"processInfo")
        process_info = objc.objc_msgSend(process_info_class, process_info_selector)
        if not process_info:
            return None

        target_selector = objc.sel_registerName(selector_name.encode("ascii"))
        responds_selector = objc.sel_registerName(b"respondsToSelector:")
        objc.objc_msgSend.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        objc.objc_msgSend.restype = ctypes.c_bool
        if not objc.objc_msgSend(process_info, responds_selector, target_selector):
            return None

        objc.objc_msgSend.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        objc.objc_msgSend.restype = result_type
        return objc.objc_msgSend(process_info, target_selector)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _run_command(
    command: tuple[str, ...],
    timeout: float,
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a read-only probe with a sleep-inclusive hard deadline."""

    if timeout <= 0.0:
        raise ProbeError("probe timeout must be greater than zero")
    clock = continuous_now
    if clock is None:
        clock = make_continuous_clock() if deadline is not None else time.monotonic
    started = clock()
    command_deadline = started + timeout
    effective_deadline = min(
        command_deadline, deadline if deadline is not None else command_deadline
    )
    comparison_deadline_first = (
        deadline is not None and deadline <= command_deadline
    )
    if clock() >= effective_deadline:
        _raise_probe_timeout(command, timeout, comparison_deadline_first)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        raise ProbeError(f"cannot execute {command[0]}: {exc}") from exc

    try:
        while True:
            remaining = effective_deadline - clock()
            if remaining <= 0.0:
                _terminate_probe_process(process)
                _raise_probe_timeout(command, timeout, comparison_deadline_first)
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
            if clock() >= effective_deadline:
                _raise_probe_timeout(command, timeout, comparison_deadline_first)
            return subprocess.CompletedProcess(
                args=command,
                returncode=int(process.returncode),
                stdout=stdout,
                stderr=stderr,
            )
    except BaseException:
        # The group leader may have exited while a descendant still owns a
        # captured pipe, so poll() alone cannot prove the group is gone.
        _terminate_probe_process(process)
        raise


def _raise_probe_timeout(
    command: tuple[str, ...], timeout: float, comparison_deadline_first: bool
) -> None:
    """Raise the correctly classified timeout for a probe process."""

    if comparison_deadline_first:
        raise ProbeDeadlineError(
            f"comparison deadline expired while running {command[0]}"
        )
    raise ProbeError(f"{command[0]} timed out after {timeout:g} seconds")


def _terminate_probe_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Kill and reap a probe process group without leaking descendants."""

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            process.poll()
    try:
        return process.communicate(timeout=0.5)
    except subprocess.TimeoutExpired as exc:
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                process.poll()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.poll()
        return stdout, stderr


def _timeout_text(value: str | bytes | None) -> str:
    """Normalize partial text collected while terminating a probe."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _run_text(
    command: tuple[str, ...],
    timeout: float,
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> str:
    completed = _run_command(
        command,
        timeout,
        deadline=deadline,
        continuous_now=continuous_now,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise ProbeError(f"{command[0]} failed: {detail}")
    return completed.stdout


def _parse_vm_stat(output: str) -> tuple[int, dict[str, int]]:
    page_match = re.search(r"page size of\s+([0-9,]+)\s+bytes", output, re.I)
    if page_match is None:
        raise ProbeError("vm_stat output did not contain page size")
    page_size = int(page_match.group(1).replace(",", ""))
    counters: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"\s*([^:]+):\s*([0-9,]+)\.?\s*$", line)
        if match:
            counters[match.group(1).strip()] = int(match.group(2).replace(",", ""))
    if "Pages free" not in counters:
        raise ProbeError("vm_stat output did not contain free pages")
    return page_size, counters


def _assess_memory_pressure(
    available_pages: int,
    page_size: int,
    total_bytes: int,
) -> MemoryPressure:
    if page_size <= 0 or total_bytes <= 0:
        return MemoryPressure.UNKNOWN
    available_fraction = (available_pages * page_size) / total_bytes
    if available_fraction >= 0.05:
        return MemoryPressure.NORMAL
    if available_fraction >= 0.02:
        return MemoryPressure.WARNING
    return MemoryPressure.CRITICAL


def _sysctl_text(
    name: str,
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> str:
    try:
        return _run_text(
            ("sysctl", "-n", name),
            timeout=5.0,
            deadline=deadline,
            continuous_now=continuous_now,
        ).strip()
    except ProbeDeadlineError:
        raise
    except ProbeError:
        return ""


def _sysctl_int(
    name: str,
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> int:
    value = _sysctl_text(
        name, deadline=deadline, continuous_now=continuous_now
    )
    try:
        return int(value)
    except ValueError:
        return 0


def _gpu_core_count(
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> int:
    try:
        raw = _run_text(
            ("system_profiler", "SPDisplaysDataType", "-json"),
            timeout=30.0,
            deadline=deadline,
            continuous_now=continuous_now,
        )
        payload = json.loads(raw)
    except ProbeDeadlineError:
        raise
    except (ProbeError, json.JSONDecodeError):
        return 0

    def find_core_value(value: Any) -> int:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).casefold()
                if "core" in key_text:
                    match = re.search(r"\d+", str(child))
                    if match:
                        return int(match.group())
                found = find_core_value(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_core_value(child)
                if found:
                    return found
        return 0

    return find_core_value(payload)


def _hardware_overview(
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> Mapping[str, Any]:
    try:
        raw = _run_text(
            ("system_profiler", "SPHardwareDataType", "-json"),
            timeout=30.0,
            deadline=deadline,
            continuous_now=continuous_now,
        )
        payload = json.loads(raw)
    except ProbeDeadlineError:
        raise
    except (ProbeError, json.JSONDecodeError):
        return {}
    entries = payload.get("SPHardwareDataType", [])
    if isinstance(entries, list) and entries and isinstance(entries[0], Mapping):
        return entries[0]
    return {}


def _overview_cpu_counts(overview: Mapping[str, Any]) -> tuple[int, int, int]:
    text = str(overview.get("number_processors", ""))
    numbers = [int(value) for value in re.findall(r"\d+", text)]
    if len(numbers) >= 3:
        return numbers[0], numbers[1], numbers[2]
    if numbers:
        return numbers[0], 0, 0
    return 0, 0, 0


def _parse_memory_size(value: str) -> int:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt])b", value, re.I)
    if match is None:
        return 0
    powers = {"k": 1, "m": 2, "g": 3, "t": 4}
    return int(float(match.group(1)) * (1024 ** powers[match.group(2).casefold()]))


def _physical_memory_fallback() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return 0


def _sw_vers_build(
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> str:
    try:
        return _run_text(
            ("sw_vers", "-buildVersion"),
            timeout=5.0,
            deadline=deadline,
            continuous_now=continuous_now,
        ).strip()
    except ProbeDeadlineError:
        raise
    except ProbeError:
        return ""


def _pmset_low_power_mode(
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> bool | None:
    try:
        source_output = _run_text(
            ("pmset", "-g", "ps"),
            timeout=5.0,
            deadline=deadline,
            continuous_now=continuous_now,
        )
        profiles_output = _run_text(
            ("pmset", "-g", "custom"),
            timeout=5.0,
            deadline=deadline,
            continuous_now=continuous_now,
        )
    except ProbeDeadlineError:
        raise
    except ProbeError:
        return None
    source_match = re.search(
        r"Now drawing from\s+'?([^'\n]+? Power)'?\s*$",
        source_output,
        re.M | re.I,
    )
    if source_match is None:
        return None
    active_source = source_match.group(1).strip()
    section_match = re.search(
        rf"(?ms)^\s*{re.escape(active_source)}:\s*$\n"
        rf"(.*?)(?=^\s*[^\n:]+ Power:\s*$|\Z)",
        profiles_output,
    )
    if section_match is None:
        return None
    values = re.findall(
        r"^\s*lowpowermode\s+([01])\s*$",
        section_match.group(1),
        re.M | re.I,
    )
    if not values:
        return None
    return values[-1] == "1"


def _extract_llama_bench_version(help_text: str) -> str:
    patterns = (
        r"(?im)^\s*(?:llama(?:\.cpp|-bench)?\s+)?version\s*[:=]?\s*(.+?)\s*$",
        r"(?im)^\s*build(?:\s+version)?\s*[:=]\s*(.+?)\s*$",
        r"(?im)\b(build\s+\d+\s*\([0-9a-f]{7,40}\))\b",
    )
    for pattern in patterns:
        match = re.search(pattern, help_text)
        if match:
            return match.group(1).strip()
    return "unknown"


__all__ = (
    "HardwareFingerprint",
    "LlamaBenchInfo",
    "MemoryPressure",
    "MemoryState",
    "PowerState",
    "ProbeDeadlineError",
    "ProbeError",
    "ThermalState",
    "detect_llama_bench",
    "get_cpu_busy_fraction",
    "get_cpu_load",
    "get_hardware_fingerprint",
    "get_memory_state",
    "get_power_state",
    "get_thermal_state",
    "make_continuous_clock",
    "wait_for_thermal_nominal",
)
