"""Single-invocation llama-bench subprocess runner."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import BenchmarkConfig
from .probe import (
    ProbeDeadlineError,
    ProbeError,
    _run_command,
    make_continuous_clock,
)


class RunnerError(RuntimeError):
    """Raised when benchmark inputs prevent an invocation from starting."""


LLAMA_BENCH_CONTRACT_VERSION = "edgeci-llamacpp-metal-v0.1"


# These are the options whose values define EdgeCI's locked workload.  Aliases
# are accepted because llama-bench has used both short and long spellings, but
# every semantic option must be present in ``--help`` before a comparison may
# start.  Options with no explicit neutral representation (for example RPC,
# tensor overrides, and fit-target) intentionally remain absent.
_REQUIRED_OPTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("model", ("-m", "--model")),
    ("n_prompt", ("-p", "--n-prompt")),
    ("n_gen", ("-n", "--n-gen")),
    ("n_depth", ("-d", "--n-depth")),
    ("repetitions", ("-r", "--repetitions")),
    ("batch_size", ("-b", "--batch-size")),
    ("ubatch_size", ("-ub", "--ubatch-size")),
    ("cache_type_k", ("-ctk", "--cache-type-k")),
    ("cache_type_v", ("-ctv", "--cache-type-v")),
    ("n_gpu_layers", ("-ngl", "--n-gpu-layers")),
    ("n_cpu_moe", ("-ncmoe", "--n-cpu-moe")),
    ("split_mode", ("-sm", "--split-mode")),
    ("main_gpu", ("-mg", "--main-gpu")),
    ("tensor_split", ("-ts", "--tensor-split")),
    ("no_kv_offload", ("-nkvo", "--no-kv-offload")),
    ("flash_attn", ("-fa", "--flash-attn")),
    ("device", ("-dev", "--device")),
    ("mmap", ("-mmp", "--mmap")),
    ("direct_io", ("-dio", "--direct-io")),
    ("embeddings", ("-embd", "--embeddings")),
    ("no_op_offload", ("-nopo", "--no-op-offload")),
    ("no_host", ("--no-host",)),
    ("priority", ("--prio",)),
    ("delay", ("--delay",)),
    ("cpu_mask", ("-C", "--cpu-mask")),
    ("cpu_strict", ("--cpu-strict",)),
    ("poll", ("--poll",)),
    ("threads", ("-t", "--threads")),
    ("output", ("-o", "--output")),
)

_OPTION_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])(--?[A-Za-z][A-Za-z0-9-]*)(?![A-Za-z0-9_-])"
)


@dataclass(frozen=True)
class LlamaBenchContract:
    """Known llama-bench option spellings for EdgeCI's locked protocol."""

    version: str
    selected_options: tuple[tuple[str, str], ...]

    def option(self, semantic_name: str) -> str:
        """Return the supported CLI spelling selected for one semantic option."""

        for name, option in self.selected_options:
            if name == semantic_name:
                return option
        raise RunnerError(
            f"llama-bench contract {self.version!r} has no "
            f"{semantic_name!r} option"
        )


def parse_llama_bench_contract(help_text: str) -> LlamaBenchContract:
    """Validate ``--help`` and select one spelling for every pinned option.

    A missing option is a contract incompatibility, not permission to inherit
    that binary's default.  This keeps base and head on the same semantic
    workload even when aliases differ between revisions.

    Args:
        help_text: Combined stdout/stderr from one successful ``--help`` probe.

    Returns:
        Immutable mapping from protocol semantics to supported CLI options.

    Raises:
        RunnerError: If the help output cannot honor the locked protocol.
    """

    if not isinstance(help_text, str) or not help_text.strip():
        raise RunnerError("llama-bench --help returned no capability text")
    supported = _declared_help_options(help_text)
    selected: list[tuple[str, str]] = []
    missing: list[str] = []
    for semantic_name, aliases in _REQUIRED_OPTION_ALIASES:
        option = next((alias for alias in aliases if alias in supported), None)
        if option is None:
            missing.append(f"{semantic_name} ({'/'.join(aliases)})")
        else:
            selected.append((semantic_name, option))
    if missing:
        raise RunnerError(
            "incompatible llama-bench CLI contract; missing required options: "
            + ", ".join(missing)
        )
    return LlamaBenchContract(
        version=LLAMA_BENCH_CONTRACT_VERSION,
        selected_options=tuple(selected),
    )


def canonical_llama_bench_contract() -> LlamaBenchContract:
    """Return the canonical spellings used by direct runner callers.

    Comparison orchestration must use :func:`parse_llama_bench_contract` on the
    already-probed binaries.  This fallback keeps the lower-level runner API
    compatible while still passing every pinned value explicitly; an older
    binary will fail rather than silently use defaults.
    """

    return LlamaBenchContract(
        version=LLAMA_BENCH_CONTRACT_VERSION,
        selected_options=tuple(
            (semantic_name, aliases[0])
            for semantic_name, aliases in _REQUIRED_OPTION_ALIASES
        ),
    )


def build_llama_bench_command(
    binary_path: Path,
    model_path: Path,
    config: BenchmarkConfig,
    threads: int,
    *,
    contract: LlamaBenchContract,
) -> tuple[str, ...]:
    """Build one fully pinned llama-bench argv vector."""

    values: tuple[tuple[str, str], ...] = (
        ("model", str(model_path)),
        ("n_prompt", str(config.prompt_tokens)),
        ("n_gen", str(config.generate_tokens)),
        ("n_depth", "0"),
        ("repetitions", "1"),
        ("batch_size", "2048"),
        ("ubatch_size", "512"),
        ("cache_type_k", "f16"),
        ("cache_type_v", "f16"),
        ("n_gpu_layers", "-1"),
        ("n_cpu_moe", "0"),
        ("split_mode", "layer"),
        ("main_gpu", "0"),
        ("tensor_split", "0"),
        ("no_kv_offload", "0"),
        ("flash_attn", "auto"),
        ("device", "auto"),
        ("mmap", "1"),
        ("direct_io", "0"),
        ("embeddings", "0"),
        ("no_op_offload", "0"),
        ("no_host", "0"),
        ("priority", "0"),
        ("delay", "0"),
        ("cpu_mask", "0x0"),
        ("cpu_strict", "0"),
        ("poll", "50"),
        ("threads", str(threads)),
        ("output", "jsonl"),
    )
    command: list[str] = [str(binary_path)]
    for semantic_name, value in values:
        command.extend((contract.option(semantic_name), value))
    return tuple(command)


def _declared_help_options(help_text: str) -> frozenset[str]:
    """Extract option tokens from declaration lines in CLI help text."""

    options: set[str] = set()
    for raw_line in help_text.splitlines():
        declaration = raw_line.lstrip()
        if not declaration.startswith("-"):
            continue
        # Argument metavariables begin after ``<``.  For switches without one,
        # multiple spaces conventionally delimit the prose description.
        declaration = declaration.split("<", 1)[0]
        declaration = re.split(r"\s{2,}", declaration, maxsplit=1)[0]
        options.update(_OPTION_TOKEN.findall(declaration))
    return frozenset(options)


@dataclass(frozen=True)
class RunResult:
    """Captured outcome of one llama-bench process."""

    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    command: tuple[str, ...]
    timed_out: bool = False
    deadline_expired: bool = False

    @property
    def succeeded(self) -> bool:
        """Return whether the process exited successfully before timeout."""

        return self.exit_code == 0 and not self.timed_out


def run_llama_bench(
    binary_path: Path,
    model_path: Path,
    config: BenchmarkConfig,
    *,
    threads: int | None = None,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
    contract: LlamaBenchContract | None = None,
) -> RunResult:
    """Execute one structured llama-bench invocation.

    EdgeCI controls repetition and scheduling externally, so this function
    always uses ``-r 1`` and never retries a failed or timed-out measurement.

    Args:
        binary_path: llama-bench executable.
        model_path: Local GGUF model.
        config: Benchmark token sizes and timeout.
        threads: Performance-core thread count override.
        deadline: Optional absolute sleep-inclusive deadline supplied by an
            enclosing comparison session.
        continuous_now: Clock associated with ``deadline``.
        contract: Capability-validated option spellings. Direct low-level
            callers receive canonical spellings, but comparison orchestration
            always supplies a contract parsed from the binary's help output.

    Returns:
        Captured stdout, stderr, exit status, wall time, and exact command.

    Raises:
        RunnerError: If an input path or thread count is invalid.
    """

    if continuous_now is None:
        try:
            continuous_now = make_continuous_clock()
        except ProbeError as exc:
            raise RunnerError(str(exc)) from exc
    if deadline is not None and continuous_now() >= deadline:
        return RunResult(
            stdout="",
            stderr="comparison deadline expired before input validation",
            exit_code=124,
            duration_seconds=0.0,
            command=(str(binary_path),),
            deadline_expired=True,
        )

    binary = _require_file(binary_path, "llama-bench binary")
    if not os.access(binary, os.X_OK):
        raise RunnerError(f"llama-bench binary is not executable: {binary}")
    model = _require_file(model_path, "model")
    try:
        with model.open("rb") as stream:
            magic = stream.read(4)
    except OSError as exc:
        raise RunnerError(f"cannot read model header {model}: {exc}") from exc
    if magic != b"GGUF":
        raise RunnerError(f"model does not have a GGUF header: {model}")
    try:
        selected_threads = (
            get_performance_core_count(
                deadline=deadline, continuous_now=continuous_now
            )
            if threads is None
            else threads
        )
    except ProbeDeadlineError as exc:
        return RunResult(
            stdout="",
            stderr=str(exc),
            exit_code=124,
            duration_seconds=0.0,
            command=(str(binary),),
            deadline_expired=True,
        )
    if (
        isinstance(selected_threads, bool)
        or not isinstance(selected_threads, int)
        or selected_threads <= 0
    ):
        raise RunnerError("threads must be a positive integer")
    if config.prompt_tokens <= 0 or config.generate_tokens <= 0:
        raise RunnerError("prompt_tokens and generate_tokens must be positive")

    command = build_llama_bench_command(
        binary,
        model,
        config,
        selected_threads,
        contract=contract or canonical_llama_bench_contract(),
    )
    timeout_seconds = config.timeout_minutes * 60.0
    if timeout_seconds <= 0:
        raise RunnerError("timeout_minutes must be greater than zero")
    started = continuous_now()
    configured_deadline = started + timeout_seconds
    effective_deadline = min(
        configured_deadline,
        deadline if deadline is not None else configured_deadline,
    )
    session_deadline_first = (
        deadline is not None and deadline <= configured_deadline
    )
    if continuous_now() >= effective_deadline:
        return RunResult(
            stdout="",
            stderr="deadline expired before llama-bench process launch",
            exit_code=124,
            duration_seconds=continuous_now() - started,
            command=command,
            deadline_expired=session_deadline_first,
            timed_out=not session_deadline_first,
        )
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
        duration = continuous_now() - started
        return RunResult(
            stdout="",
            stderr=f"cannot execute llama-bench: {exc}",
            exit_code=127,
            duration_seconds=duration,
            command=command,
        )

    try:
        while True:
            remaining = effective_deadline - continuous_now()
            if remaining <= 0.0:
                return _kill_timed_out_process(
                    process,
                    command,
                    started,
                    continuous_now,
                    timeout_seconds,
                    deadline_expired=session_deadline_first,
                )
            try:
                stdout, stderr = process.communicate(timeout=min(0.5, remaining))
            except subprocess.TimeoutExpired:
                continue
            duration = continuous_now() - started
            if continuous_now() >= effective_deadline:
                timeout_detail = (
                    "llama-bench exceeded the comparison deadline"
                    if session_deadline_first
                    else (
                        "llama-bench exceeded hard deadline of "
                        f"{timeout_seconds:g} seconds"
                    )
                )
                stderr = f"{stderr.rstrip()}\n{timeout_detail}".lstrip()
                return RunResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=124,
                    duration_seconds=duration,
                    command=command,
                    timed_out=not session_deadline_first,
                    deadline_expired=session_deadline_first,
                )
            return RunResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=int(process.returncode),
                duration_seconds=duration,
                command=command,
            )
    except BaseException:
        # Popen owns a new process group. Reap the complete tree before
        # propagating Ctrl-C, cancellation, or another asynchronous exception.
        _terminate_process_group(process)
        raise


def get_performance_core_count(
    *,
    deadline: float | None = None,
    continuous_now: Callable[[], float] | None = None,
) -> int:
    """Return logical performance-core count or fail closed.

    Args:
        deadline: Optional absolute comparison deadline.
        continuous_now: Sleep-inclusive clock associated with ``deadline``.
    """

    diagnostics: list[str] = []
    try:
        completed = _run_command(
            ("sysctl", "-n", "hw.perflevel0.logicalcpu"),
            timeout=5.0,
            deadline=deadline,
            continuous_now=continuous_now,
        )
        if completed.returncode == 0:
            count = int(completed.stdout.strip())
            if count > 0:
                return count
            diagnostics.append("sysctl returned a non-positive count")
        else:
            diagnostics.append(f"sysctl exited {completed.returncode}")
    except ProbeDeadlineError:
        raise
    except (ProbeError, ValueError) as exc:
        diagnostics.append(f"sysctl failed: {exc}")
    try:
        completed = _run_command(
            ("system_profiler", "SPHardwareDataType", "-json"),
            timeout=30.0,
            deadline=deadline,
            continuous_now=continuous_now,
        )
        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
            entries = payload.get("SPHardwareDataType", [])
            if isinstance(entries, list) and entries and isinstance(entries[0], dict):
                processor_text = str(entries[0].get("number_processors", ""))
                counts = [int(value) for value in re.findall(r"\d+", processor_text)]
                if len(counts) >= 3 and counts[1] > 0:
                    return counts[1]
            diagnostics.append("system_profiler did not report performance cores")
        else:
            diagnostics.append(f"system_profiler exited {completed.returncode}")
    except ProbeDeadlineError:
        raise
    except (ProbeError, ValueError, json.JSONDecodeError) as exc:
        diagnostics.append(f"system_profiler failed: {exc}")
    detail = "; ".join(diagnostics)
    raise RunnerError(
        "cannot determine Apple performance-core count from sysctl or system_profiler"
        + (f" ({detail})" if detail else "")
    )


def _require_file(path: Path, description: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise RunnerError(f"{description} does not exist or is not a file: {candidate}")
    return candidate


def _kill_timed_out_process(
    process: subprocess.Popen[str],
    command: tuple[str, ...],
    started: float,
    continuous_now: Callable[[], float],
    timeout_seconds: float,
    *,
    deadline_expired: bool,
) -> RunResult:
    """Kill a process at its sleep-inclusive deadline and retain its output."""

    stdout, stderr = _terminate_process_group(process)
    detail = (
        "comparison deadline expired; llama-bench was killed"
        if deadline_expired
        else f"llama-bench timed out after {timeout_seconds:g} seconds"
    )
    stderr = f"{stderr.rstrip()}\n{detail}".lstrip()
    return RunResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=124,
        duration_seconds=continuous_now() - started,
        command=command,
        timed_out=not deadline_expired,
        deadline_expired=deadline_expired,
    )


def _terminate_process_group(
    process: subprocess.Popen[str],
) -> tuple[str, str]:
    """Force-stop and reap a detached benchmark process group."""

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            process.poll()
    try:
        stdout, stderr = process.communicate(timeout=0.5)
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
    """Normalize partial subprocess output captured during timeout cleanup."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


__all__ = (
    "LLAMA_BENCH_CONTRACT_VERSION",
    "LlamaBenchContract",
    "RunResult",
    "RunnerError",
    "build_llama_bench_command",
    "canonical_llama_bench_contract",
    "get_performance_core_count",
    "parse_llama_bench_contract",
    "run_llama_bench",
)
