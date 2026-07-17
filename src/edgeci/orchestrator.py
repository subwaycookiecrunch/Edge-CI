"""Paired base/head benchmark orchestration for EdgeCI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .adapter import (
    BenchOutputError,
    BenchSample,
    parse_bench_output,
    validate_config_match,
)
from .config import EdgeCIConfig
from .hashing import hash_file
from .lock import EdgeCILock
from .probe import (
    HardwareFingerprint,
    MemoryState,
    PowerState,
    ProbeDeadlineError,
    ThermalState,
    detect_llama_bench,
    get_cpu_load,
    get_hardware_fingerprint,
    get_memory_state,
    get_power_state,
    get_thermal_state,
    make_continuous_clock,
)
from .runner import (
    LlamaBenchContract,
    RunResult,
    RunnerError,
    get_performance_core_count,
    parse_llama_bench_contract,
    run_llama_bench,
)
from .schedule import Invocation, generate_schedule


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Environment readings captured around one benchmark invocation.

    Attributes:
        timestamp: Wall-clock timestamp in UTC.
        monotonic_seconds: Monotonic clock value used to detect sleep/wake gaps.
        thermal: Foundation thermal state.
        power: AC and Low Power Mode state.
        memory: Virtual-memory counters and pressure assessment.
        cpu_busy_fraction: CPU busy fraction in the inclusive range 0 through 1.
        continuous_seconds: Sleep-inclusive monotonic timestamp used to detect
            system suspension without relying on adjustable wall time.
    """

    timestamp: datetime
    monotonic_seconds: float
    thermal: ThermalState
    power: PowerState
    memory: MemoryState
    cpu_busy_fraction: float
    continuous_seconds: float = 0.0


@dataclass(frozen=True)
class MeasuredInvocation:
    """One completed or failed llama-bench process invocation."""

    invocation: Invocation
    arm: str
    bench_samples: tuple[BenchSample, ...]
    env_before: EnvironmentSnapshot
    env_after: EnvironmentSnapshot
    wall_time: float
    stdout: str
    stderr: str
    exit_code: int
    command: tuple[str, ...]
    timed_out: bool = False
    deadline_expired: bool = False
    external_events: tuple[str, ...] = ()
    parse_error: str | None = None
    precondition_error: str | None = None

    def sample_for(self, test_type: str) -> BenchSample:
        """Return sample for ``test_type``.

        Args:
            test_type: llama-bench test identifier, normally ``pp`` or ``tg``.

        Raises:
            LookupError: If exactly one matching sample is not present.
        """

        matches = [sample for sample in self.bench_samples if sample.test_type == test_type]
        if len(matches) != 1:
            raise LookupError(
                f"expected one {test_type!r} sample for {self.arm} invocation, "
                f"found {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True)
class ProgressUpdate:
    """Structured update emitted during comparison execution."""

    phase: str
    completed: int
    total: int
    invocation: Invocation | None
    message: str
    elapsed_seconds: float


@dataclass(frozen=True)
class ComparisonSession:
    """Complete provenance and raw evidence for one comparison session."""

    schedule: tuple[Invocation, ...]
    warmup_results: tuple[MeasuredInvocation, ...]
    measured_results: tuple[MeasuredInvocation, ...]
    hardware: HardwareFingerprint
    base_binary_path: Path
    head_binary_path: Path
    model_path: Path
    base_binary_sha: str
    head_binary_sha: str
    model_sha: str
    config: EdgeCIConfig
    seed: str
    start_time: datetime
    end_time: datetime
    preflight_duration: float
    contaminated_blocks: tuple[int, ...]
    abort_reason: str | None

    @property
    def duration_seconds(self) -> float:
        """Return wall-clock session duration in seconds."""

        return max(0.0, (self.end_time - self.start_time).total_seconds())

    @property
    def complete(self) -> bool:
        """Return whether all planned measurements completed cleanly."""

        expected = self.config.benchmark.pairs * 2
        return (
            self.abort_reason is None
            and not self.contaminated_blocks
            and len(self.measured_results) == expected
            and all(result.exit_code == 0 for result in self.measured_results)
        )


ProgressCallback = Callable[[ProgressUpdate], None]
FileIdentity = tuple[int, int, int, int, int]


class ComparisonError(RuntimeError):
    """Raised for an invalid comparison request or harness contract failure."""


def capture_environment(
    *,
    cpu_sample_seconds: float = 1.0,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> EnvironmentSnapshot:
    """Capture all per-invocation environment sensors.

    Args:
        cpu_sample_seconds: Sampling interval passed to the CPU probe.
        deadline: Optional absolute comparison deadline.
        continuous_now: Optional shared sleep-inclusive clock.

    Returns:
        Timestamped environment snapshot.
    """

    clock = continuous_now or make_continuous_clock()
    if deadline is not None and clock() >= deadline:
        raise ProbeDeadlineError(
            "comparison deadline expired before environment capture"
        )
    timestamp = datetime.now(timezone.utc)
    monotonic_seconds = time.monotonic()
    continuous_seconds = clock()
    thermal = get_thermal_state()
    power = get_power_state(deadline=deadline, continuous_now=clock)
    memory = get_memory_state(deadline=deadline, continuous_now=clock)
    cpu_busy_fraction = get_cpu_load(
        sample_seconds=cpu_sample_seconds,
        deadline=deadline,
        continuous_now=clock,
    )
    if deadline is not None and clock() >= deadline:
        raise ProbeDeadlineError(
            "comparison deadline expired during environment capture"
        )
    return EnvironmentSnapshot(
        timestamp=timestamp,
        monotonic_seconds=monotonic_seconds,
        thermal=thermal,
        power=power,
        memory=memory,
        cpu_busy_fraction=cpu_busy_fraction,
        continuous_seconds=continuous_seconds,
    )


def run_comparison(
    base_binary: Path,
    head_binary: Path,
    model_path: Path,
    config: EdgeCIConfig,
    *,
    seed: str = "",
    on_progress: ProgressCallback | None = None,
) -> ComparisonSession:
    """Run a deterministic, sequential paired base/head comparison.

    No failed process is retried and no completed measurement is removed based on
    its observed throughput. External power or sleep/wake events contaminate the
    containing block and make the session inconclusive.

    Args:
        base_binary: Base ``llama-bench`` executable.
        head_binary: Head ``llama-bench`` executable.
        model_path: GGUF model used by both arms.
        config: Validated EdgeCI configuration.
        seed: Stable schedule identifier. A generated timestamp is used if empty.
        on_progress: Optional structured progress callback.

    Returns:
        Session containing provenance, schedule, raw samples, and abort evidence.

    Raises:
        ComparisonError: If an input path is invalid.
        EdgeCILockError: If another EdgeCI process owns the machine lock.
    """

    continuous_now = make_continuous_clock()
    start_time = datetime.now(timezone.utc)
    started_continuous = continuous_now()
    total_seconds = config.benchmark.timeout_minutes * 60.0
    if total_seconds <= 0.0:
        raise ComparisonError("benchmark.timeout_minutes must be greater than zero")
    deadline = started_continuous + total_seconds

    base_binary = _validate_executable(base_binary, "base binary")
    head_binary = _validate_executable(head_binary, "head binary")
    model_path = _validate_file(model_path, "model")
    effective_seed = seed or start_time.isoformat()
    schedule = tuple(
        generate_schedule(
            n_pairs=config.benchmark.pairs,
            n_warmup_pairs=config.benchmark.warmup_pairs,
            seed=effective_seed,
        )
    )
    warmup_results: list[MeasuredInvocation] = []
    measured_results: list[MeasuredInvocation] = []
    contaminated_blocks: set[int] = set()
    abort_reason: str | None = None
    preflight_duration = 0.0

    def emit(
        phase: str,
        completed: int,
        total: int,
        invocation: Invocation | None,
        message: str,
    ) -> None:
        if on_progress is not None:
            on_progress(
                ProgressUpdate(
                    phase=phase,
                    completed=completed,
                    total=total,
                    invocation=invocation,
                    message=message,
                    elapsed_seconds=continuous_now() - started_continuous,
                )
            )

    with EdgeCILock():
        emit("provenance", 0, len(schedule), None, "Hashing binaries and model")
        provenance_paths = {
            "base binary": base_binary,
            "head binary": head_binary,
            "model": model_path,
        }
        provenance_identities = _capture_file_identities(provenance_paths)
        try:
            base_binary_sha = hash_file(
                base_binary, deadline=deadline, continuous_now=continuous_now
            )
            head_binary_sha = hash_file(
                head_binary, deadline=deadline, continuous_now=continuous_now
            )
            model_sha = hash_file(
                model_path, deadline=deadline, continuous_now=continuous_now
            )
        except TimeoutError as exc:
            raise ComparisonError(str(exc)) from exc
        try:
            base_contract, head_contract = _inspect_llama_bench_contracts(
                base_binary,
                head_binary,
                deadline=deadline,
                continuous_now=continuous_now,
            )
            performance_threads = get_performance_core_count(
                deadline=deadline, continuous_now=continuous_now
            )
        except ProbeDeadlineError as exc:
            raise ComparisonError(str(exc)) from exc
        provenance_problem = _provenance_change(
            provenance_paths, provenance_identities
        )
        if provenance_problem is not None:
            abort_reason = f"ERROR_PROVENANCE: {provenance_problem}"
        try:
            hardware = get_hardware_fingerprint(
                deadline=deadline, continuous_now=continuous_now
            )
        except ProbeDeadlineError as exc:
            raise ComparisonError(str(exc)) from exc

        if abort_reason is None:
            preflight_started = continuous_now()
            emit("preflight", 0, len(schedule), None, "Waiting for stable testbed")
            preflight_ok, preflight_reason = _run_preflight(
                base_binary,
                head_binary,
                config,
                deadline,
                continuous_now,
            )
            preflight_duration = continuous_now() - preflight_started
            if not preflight_ok:
                abort_reason = preflight_reason
            else:
                provenance_problem = _provenance_change(
                    provenance_paths, provenance_identities
                )
                if provenance_problem is not None:
                    abort_reason = f"ERROR_PROVENANCE: {provenance_problem}"

        previous_result: MeasuredInvocation | None = None
        if abort_reason is None:
            for schedule_index, invocation in enumerate(schedule):
                phase = "warmup" if invocation.is_warmup else "measurement"
                completed = len(warmup_results) + len(measured_results)
                emit(
                    phase,
                    completed,
                    len(schedule),
                    invocation,
                    f"Running {invocation.arm} invocation",
                )

                if previous_result is not None:
                    gate_ok, gate_reason = _inter_invocation_gate(
                        config,
                        deadline,
                        previous_result.env_after,
                        continuous_now,
                    )
                    if not gate_ok:
                        abort_reason = gate_reason
                        if (
                            gate_reason.startswith("INCONCLUSIVE_EXTERNAL")
                            and previous_result is not None
                            and not previous_result.invocation.is_warmup
                        ):
                            contaminated_blocks.add(
                                previous_result.invocation.block_index
                            )
                        break

                if continuous_now() >= deadline:
                    abort_reason = "INCONCLUSIVE_DEADLINE: comparison timeout"
                    break

                try:
                    result = _execute_invocation(
                        invocation,
                        base_binary,
                        head_binary,
                        model_path,
                        config,
                        deadline,
                        continuous_now,
                        performance_threads,
                        (
                            base_contract
                            if invocation.arm == "base"
                            else head_contract
                        ),
                    )
                except Exception as exc:  # Evidence preserved as explicit harness abort.
                    abort_reason = f"ERROR_HARNESS: {type(exc).__name__}: {exc}"
                    break

                if invocation.is_warmup:
                    warmup_results.append(result)
                else:
                    measured_results.append(result)
                previous_result = result

                if result.external_events:
                    if not invocation.is_warmup:
                        contaminated_blocks.add(invocation.block_index)
                    abort_reason = "INCONCLUSIVE_EXTERNAL: " + ", ".join(
                        result.external_events
                    )
                    break
                if result.timed_out:
                    abort_reason = (
                        f"INCONCLUSIVE_DEADLINE: {invocation.arm} llama-bench timeout"
                    )
                    break
                if result.deadline_expired:
                    abort_reason = "INCONCLUSIVE_DEADLINE: comparison timeout"
                    break
                if result.precondition_error is not None:
                    abort_reason = result.precondition_error
                    break
                if result.exit_code != 0:
                    detail = result.stderr.strip().splitlines()
                    suffix = f": {detail[-1]}" if detail else ""
                    abort_reason = (
                        f"ERROR_INVOCATION: {invocation.arm} exited "
                        f"{result.exit_code}{suffix}"
                    )
                    break
                if result.parse_error is not None:
                    abort_reason = f"ERROR_OUTPUT: {result.parse_error}"
                    break
                if not result.bench_samples:
                    abort_reason = (
                        f"ERROR_OUTPUT: {invocation.arm} produced no benchmark samples"
                    )
                    break

                emit(
                    phase,
                    schedule_index + 1,
                    len(schedule),
                    invocation,
                    f"Completed {invocation.arm} invocation",
                )

        provenance_problem = _provenance_change(
            provenance_paths, provenance_identities
        )
        if provenance_problem is not None:
            abort_reason = f"ERROR_PROVENANCE: {provenance_problem}"

        if abort_reason is None:
            try:
                reference = measured_results[0].bench_samples
                for result in measured_results[1:]:
                    validate_config_match(reference, result.bench_samples)
                _validate_arm_build_identity(measured_results)
            except Exception as exc:
                abort_reason = f"ERROR_CONFIG_MISMATCH: {exc}"

        emit(
            "complete" if abort_reason is None else "aborted",
            len(warmup_results) + len(measured_results),
            len(schedule),
            None,
            abort_reason or "Comparison complete",
        )

    return ComparisonSession(
        schedule=schedule,
        warmup_results=tuple(warmup_results),
        measured_results=tuple(measured_results),
        hardware=hardware,
        base_binary_path=base_binary,
        head_binary_path=head_binary,
        model_path=model_path,
        base_binary_sha=base_binary_sha,
        head_binary_sha=head_binary_sha,
        model_sha=model_sha,
        config=config,
        seed=effective_seed,
        start_time=start_time,
        end_time=datetime.now(timezone.utc),
        preflight_duration=preflight_duration,
        contaminated_blocks=tuple(sorted(contaminated_blocks)),
        abort_reason=abort_reason,
    )


def _validate_file(path: Path, label: str) -> Path:
    """Resolve and validate an input file path."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ComparisonError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _validate_executable(path: Path, label: str) -> Path:
    """Resolve and validate an executable input path."""

    resolved = _validate_file(path, label)
    if not os.access(resolved, os.X_OK):
        raise ComparisonError(f"{label} is not executable: {resolved}")
    return resolved


def _inspect_llama_bench_contracts(
    base_binary: Path,
    head_binary: Path,
    *,
    deadline: float,
    continuous_now: Callable[[], float],
) -> tuple[LlamaBenchContract, LlamaBenchContract]:
    """Probe each distinct binary once and validate its locked CLI contract."""

    contracts: dict[Path, LlamaBenchContract] = {}
    for label, binary in (
        ("base binary", base_binary),
        ("head binary", head_binary),
    ):
        if binary in contracts:
            continue
        info = detect_llama_bench(
            binary,
            deadline=deadline,
            continuous_now=continuous_now,
        )
        if info is None:
            raise ComparisonError(
                f"{label} is not a recognized llama-bench: {binary}"
            )
        try:
            contracts[binary] = parse_llama_bench_contract(info.help_text)
        except RunnerError as exc:
            raise ComparisonError(f"{label} {exc}") from exc
    return contracts[base_binary], contracts[head_binary]


def _capture_file_identities(
    paths: dict[str, Path],
) -> dict[str, FileIdentity]:
    """Capture immutable provenance metadata for all benchmark inputs."""

    return {label: _file_identity(path) for label, path in paths.items()}


def _file_identity(path: Path) -> FileIdentity:
    """Return metadata used to detect replacement or mutation of one file."""

    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _provenance_change(
    paths: dict[str, Path], expected: dict[str, FileIdentity]
) -> str | None:
    """Describe the first changed or unavailable provenance input."""

    for label, path in paths.items():
        try:
            observed = _file_identity(path)
        except OSError as exc:
            return f"{label} became unavailable: {exc}"
        if observed != expected[label]:
            return f"{label} changed after hashing: {path}"
    return None


def _validate_arm_build_identity(results: list[MeasuredInvocation]) -> None:
    """Require a stable emitted build identity within each comparison arm."""

    arm_identities: dict[str, tuple[str, int]] = {}
    for result in results:
        identities = {
            (sample.build_commit, sample.build_number)
            for sample in result.bench_samples
        }
        if len(identities) != 1:
            raise ComparisonError(
                f"{result.arm} emitted inconsistent build identities in one run"
            )
        identity = next(iter(identities))
        previous = arm_identities.setdefault(result.arm, identity)
        if previous != identity:
            raise ComparisonError(
                f"{result.arm} build identity changed during comparison: "
                f"{previous!r} != {identity!r}"
            )


def _run_preflight(
    base_binary: Path,
    head_binary: Path,
    config: EdgeCIConfig,
    session_deadline: float,
    continuous_now: Callable[[], float],
) -> tuple[bool, str | None]:
    """Wait until every preflight condition stays healthy for settle window."""

    preflight_deadline = continuous_now() + config.preflight.preflight_timeout
    session_limits_preflight = session_deadline <= preflight_deadline
    timeout_deadline = min(session_deadline, preflight_deadline)
    newest_mtime = max(base_binary.stat().st_mtime, head_binary.stat().st_mtime)
    cooldown_remaining = max(
        0.0, config.preflight.post_build_cooldown - (time.time() - newest_mtime)
    )
    if cooldown_remaining and not _sleep_until(
        cooldown_remaining, timeout_deadline, continuous_now
    ):
        if session_limits_preflight:
            return False, "INCONCLUSIVE_DEADLINE: timeout during post-build cooldown"
        return False, "INCONCLUSIVE_PREFLIGHT: post-build cooldown timeout"

    nominal_since: float | None = None
    last_pageouts: int | None = None
    last_problem = "testbed did not stabilize"
    while continuous_now() < timeout_deadline:
        try:
            power = get_power_state(
                deadline=timeout_deadline, continuous_now=continuous_now
            )
            memory = get_memory_state(
                deadline=timeout_deadline, continuous_now=continuous_now
            )
            thermal = get_thermal_state()
            cpu_busy = get_cpu_load(
                sample_seconds=1.0,
                deadline=timeout_deadline,
                continuous_now=continuous_now,
            )
        except ProbeDeadlineError:
            if session_limits_preflight:
                return False, "INCONCLUSIVE_DEADLINE: timeout during preflight"
            return False, "INCONCLUSIVE_PREFLIGHT: sensor sampling timeout"
        problems: list[str] = []
        if not power.ac_connected:
            problems.append("AC power is disconnected")
        if power.low_power_mode:
            problems.append("Low Power Mode is enabled")
        if not _memory_is_healthy(memory):
            problems.append(f"memory pressure is {_pressure_name(memory)}")
        if last_pageouts is not None and memory.pageouts != last_pageouts:
            problems.append("page-outs increased during preflight")
        last_pageouts = memory.pageouts
        if not _thermal_is_nominal(thermal):
            problems.append(f"thermal state is {_enum_name(thermal)}")
        if cpu_busy > config.preflight.idle_cpu_threshold:
            problems.append(
                f"CPU busy {cpu_busy:.1%} exceeds "
                f"{config.preflight.idle_cpu_threshold:.1%}"
            )

        now = time.monotonic()
        if problems:
            nominal_since = None
            last_problem = "; ".join(problems)
        else:
            if nominal_since is None:
                nominal_since = now
            if now - nominal_since >= config.preflight.thermal_settle_seconds:
                return True, None
        _sleep_until(1.0, timeout_deadline, continuous_now)

    if continuous_now() >= session_deadline:
        return False, "INCONCLUSIVE_DEADLINE: timeout during preflight"
    return False, f"INCONCLUSIVE_PREFLIGHT: {last_problem}"


def _inter_invocation_gate(
    config: EdgeCIConfig,
    session_deadline: float,
    previous_snapshot: EnvironmentSnapshot,
    continuous_now: Callable[[], float],
) -> tuple[bool, str | None]:
    """Observe fixed gap, external events, and resource recovery."""

    gap_deadline = continuous_now() + config.benchmark.gap_seconds
    while continuous_now() < gap_deadline:
        if continuous_now() >= session_deadline:
            return False, "INCONCLUSIVE_DEADLINE: timeout during invocation gap"
        if _sleep_detected_since(previous_snapshot, continuous_now):
            return False, "INCONCLUSIVE_EXTERNAL: sleep/wake during invocation gap"
        try:
            power = get_power_state(
                deadline=session_deadline, continuous_now=continuous_now
            )
        except ProbeDeadlineError:
            return False, "INCONCLUSIVE_DEADLINE: timeout during invocation gap"
        if not power.ac_connected:
            return False, "INCONCLUSIVE_EXTERNAL: AC power disconnected"
        if power.low_power_mode:
            return False, "INCONCLUSIVE_EXTERNAL: Low Power Mode enabled"
        remaining = min(gap_deadline, session_deadline) - continuous_now()
        if remaining > 0:
            time.sleep(min(1.0, remaining))

    resource_deadline = continuous_now() + 600.0
    session_limits_recovery = session_deadline <= resource_deadline
    recovery_deadline = min(session_deadline, resource_deadline)
    last_resource = "resource recovery"
    last_pageouts = previous_snapshot.memory.pageouts
    while continuous_now() < recovery_deadline:
        if _sleep_detected_since(previous_snapshot, continuous_now):
            return False, "INCONCLUSIVE_EXTERNAL: sleep/wake during resource recovery"
        try:
            power = get_power_state(
                deadline=recovery_deadline, continuous_now=continuous_now
            )
        except ProbeDeadlineError:
            if session_limits_recovery:
                return False, "INCONCLUSIVE_DEADLINE: timeout during resource recovery"
            return False, "INCONCLUSIVE_RESOURCE: resource recovery timeout"
        if not power.ac_connected:
            return False, "INCONCLUSIVE_EXTERNAL: AC power disconnected"
        if power.low_power_mode:
            return False, "INCONCLUSIVE_EXTERNAL: Low Power Mode enabled"
        thermal = get_thermal_state()
        try:
            memory = get_memory_state(
                deadline=recovery_deadline, continuous_now=continuous_now
            )
        except ProbeDeadlineError:
            if session_limits_recovery:
                return False, "INCONCLUSIVE_DEADLINE: timeout during resource recovery"
            return False, "INCONCLUSIVE_RESOURCE: resource recovery timeout"
        thermal_ok = _thermal_is_nominal(thermal)
        memory_ok = _memory_is_healthy(memory)
        pageouts_stable = memory.pageouts == last_pageouts
        last_pageouts = memory.pageouts
        if thermal_ok and memory_ok and pageouts_stable:
            return True, None
        if not thermal_ok:
            last_resource = f"thermal state {_enum_name(thermal)}"
        elif not memory_ok:
            last_resource = f"memory pressure {_pressure_name(memory)}"
        elif not pageouts_stable:
            last_resource = "active page-outs"
        _sleep_until(1.0, recovery_deadline, continuous_now)
    if continuous_now() >= session_deadline:
        return False, "INCONCLUSIVE_DEADLINE: timeout during resource recovery"
    return False, f"INCONCLUSIVE_RESOURCE: {last_resource} recovery timeout"


def _execute_invocation(
    invocation: Invocation,
    base_binary: Path,
    head_binary: Path,
    model_path: Path,
    config: EdgeCIConfig,
    deadline: float,
    continuous_now: Callable[[], float],
    performance_threads: int,
    cli_contract: LlamaBenchContract,
) -> MeasuredInvocation:
    """Execute, parse, and snapshot one scheduled process."""

    binary = base_binary if invocation.arm == "base" else head_binary
    capture_deadline_expired = continuous_now() >= deadline
    if capture_deadline_expired:
        env_before = _deadline_environment(continuous_now)
    else:
        try:
            env_before = capture_environment(
                deadline=deadline, continuous_now=continuous_now
            )
        except ProbeDeadlineError:
            capture_deadline_expired = True
            env_before = _deadline_environment(continuous_now)
    if capture_deadline_expired or continuous_now() >= deadline:
        return MeasuredInvocation(
            invocation=invocation,
            arm=invocation.arm,
            bench_samples=(),
            env_before=env_before,
            env_after=env_before,
            wall_time=0.0,
            stdout="",
            stderr="comparison deadline expired before process launch",
            exit_code=124,
            command=(str(binary),),
            deadline_expired=True,
        )
    external_events: list[str] = []
    if not env_before.power.ac_connected:
        external_events.append("AC power disconnected before invocation")
    if env_before.power.low_power_mode:
        external_events.append("Low Power Mode enabled before invocation")
    if external_events:
        return MeasuredInvocation(
            invocation=invocation,
            arm=invocation.arm,
            bench_samples=(),
            env_before=env_before,
            env_after=env_before,
            wall_time=0.0,
            stdout="",
            stderr="benchmark not launched after external power event",
            exit_code=125,
            command=(str(binary),),
            external_events=tuple(external_events),
        )
    resource_problem: str | None = None
    if not _thermal_is_nominal(env_before.thermal):
        resource_problem = f"thermal state {_enum_name(env_before.thermal)}"
    elif not _memory_is_healthy(env_before.memory):
        resource_problem = f"memory pressure {_pressure_name(env_before.memory)}"
    if resource_problem is not None:
        return MeasuredInvocation(
            invocation=invocation,
            arm=invocation.arm,
            bench_samples=(),
            env_before=env_before,
            env_after=env_before,
            wall_time=0.0,
            stdout="",
            stderr="benchmark not launched because resources became unhealthy",
            exit_code=125,
            command=(str(binary),),
            precondition_error=(
                f"INCONCLUSIVE_RESOURCE: {resource_problem} before invocation"
            ),
        )
    run_result: RunResult = run_llama_bench(
        binary,
        model_path,
        config.benchmark,
        threads=performance_threads,
        deadline=deadline,
        continuous_now=continuous_now,
        contract=cli_contract,
    )
    post_capture_deadline = (
        run_result.deadline_expired or continuous_now() >= deadline
    )
    if post_capture_deadline:
        env_after = env_before
        observed_external_events: tuple[str, ...] = ()
    else:
        try:
            env_after = capture_environment(
                deadline=deadline, continuous_now=continuous_now
            )
            observed_external_events = _detect_external_events(
                env_before, env_after
            )
        except ProbeDeadlineError:
            post_capture_deadline = True
            env_after = env_before
            observed_external_events = ()
    samples: tuple[BenchSample, ...] = ()
    parse_error: str | None = None
    if run_result.exit_code == 0:
        try:
            samples = tuple(parse_bench_output(run_result.stdout))
            _validate_invocation_samples(samples, config)
        except (BenchOutputError, ComparisonError) as exc:
            parse_error = str(exc)
    return MeasuredInvocation(
        invocation=invocation,
        arm=invocation.arm,
        bench_samples=samples,
        env_before=env_before,
        env_after=env_after,
        wall_time=run_result.duration_seconds,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
        exit_code=run_result.exit_code,
        command=tuple(str(part) for part in run_result.command),
        timed_out=run_result.timed_out,
        deadline_expired=(
            run_result.deadline_expired
            or post_capture_deadline
            or (continuous_now() >= deadline and not run_result.timed_out)
        ),
        external_events=observed_external_events,
        parse_error=parse_error,
    )


def _deadline_environment(
    continuous_now: Callable[[], float],
) -> EnvironmentSnapshot:
    """Build a fail-closed snapshot without starting another sensor process."""

    return EnvironmentSnapshot(
        timestamp=datetime.now(timezone.utc),
        monotonic_seconds=time.monotonic(),
        thermal=ThermalState.UNKNOWN,
        power=PowerState(ac_connected=False, low_power_mode=True),
        memory=MemoryState(
            page_size=0,
            free_pages=0,
            pageouts=0,
            compressed_pages=0,
        ),
        cpu_busy_fraction=1.0,
        continuous_seconds=continuous_now(),
    )


def _validate_invocation_samples(
    samples: tuple[BenchSample, ...], config: EdgeCIConfig
) -> None:
    """Validate workload coverage and Metal backend before storing a result."""

    expected = {
        ("pp", config.benchmark.prompt_tokens),
        ("tg", config.benchmark.generate_tokens),
    }
    actual = {(sample.test_type, sample.test_size) for sample in samples}
    if actual != expected or len(samples) != 2:
        raise ComparisonError(
            "llama-bench output workload mismatch: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )
    backends = {sample.backend for sample in samples}
    if backends != {"Metal"}:
        raise ComparisonError(
            f"EdgeCI requires Metal output; observed {sorted(backends)}"
        )


def _detect_external_events(
    before: EnvironmentSnapshot,
    after: EnvironmentSnapshot,
) -> tuple[str, ...]:
    """Detect only external events allowed to contaminate a block."""

    events: list[str] = []
    if before.power.ac_connected and not after.power.ac_connected:
        events.append("AC power disconnected during invocation")
    if not before.power.low_power_mode and after.power.low_power_mode:
        events.append("Low Power Mode enabled during invocation")
    wall_elapsed = (after.timestamp - before.timestamp).total_seconds()
    monotonic_elapsed = after.monotonic_seconds - before.monotonic_seconds
    if before.continuous_seconds > 0.0 and after.continuous_seconds > 0.0:
        suspended_elapsed = (
            after.continuous_seconds
            - before.continuous_seconds
            - monotonic_elapsed
        )
    else:
        suspended_elapsed = wall_elapsed - monotonic_elapsed
    if suspended_elapsed > 5.0:
        events.append("possible sleep/wake event")
    return tuple(events)


def _sleep_until(
    seconds: float,
    deadline: float,
    continuous_now: Callable[[], float],
) -> bool:
    """Sleep for requested duration without crossing a deadline."""

    if seconds <= 0:
        return continuous_now() < deadline
    remaining = deadline - continuous_now()
    if remaining <= 0:
        return False
    time.sleep(min(seconds, remaining))
    return seconds <= remaining


def _sleep_detected_since(
    snapshot: EnvironmentSnapshot,
    continuous_now: Callable[[], float],
) -> bool:
    """Detect suspended wall time since a prior environment snapshot."""

    monotonic_elapsed = time.monotonic() - snapshot.monotonic_seconds
    if snapshot.continuous_seconds > 0.0:
        total_elapsed = continuous_now() - snapshot.continuous_seconds
    else:
        total_elapsed = (
            datetime.now(timezone.utc) - snapshot.timestamp
        ).total_seconds()
    return total_elapsed - monotonic_elapsed > 2.0


def _thermal_is_nominal(state: object) -> bool:
    """Return whether a thermal enum-like value is nominal."""

    name = _enum_name(state)
    return name == "nominal" or state == 0


def _memory_is_healthy(state: MemoryState) -> bool:
    """Return whether memory pressure assessment permits another invocation."""

    return _pressure_name(state) in {"normal", "nominal", "ok"}


def _pressure_name(state: MemoryState) -> str:
    """Normalize a memory pressure enum or string for comparisons."""

    pressure = getattr(state, "pressure_level", getattr(state, "pressure", "unknown"))
    return _enum_name(pressure)


def _enum_name(value: object) -> str:
    """Normalize enum-like values to lowercase names."""

    return str(getattr(value, "name", value)).lower()
