"""Parse and validate structured output produced by ``llama-bench``."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


ConfigValue = str | int | float | bool | None


class BenchOutputError(ValueError):
    """Raised when ``llama-bench`` output is malformed or inconsistent."""


@dataclass(frozen=True)
class BenchSample:
    """One prompt-processing or token-generation benchmark measurement."""

    model_filename: str
    model_size: int
    model_n_params: int
    n_gpu_layers: int
    n_batch: int
    n_ubatch: int
    type_k: str
    type_v: str
    n_threads: int
    test_type: str
    test_size: int
    tokens_per_second: float
    build_commit: str
    build_number: int
    backend: str
    device_description: str
    runtime_configuration: tuple[tuple[str, ConfigValue], ...] = ()


def _expect_string(record: Mapping[str, Any], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BenchOutputError(
            f"line {line_number}: field {field!r} must be a non-empty string"
        )
    return value


def _expect_int(record: Mapping[str, Any], field: str, line_number: int) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BenchOutputError(f"line {line_number}: field {field!r} must be an integer")
    return value


def _expect_number(record: Mapping[str, Any], field: str, line_number: int) -> float:
    value = record.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BenchOutputError(f"line {line_number}: field {field!r} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise BenchOutputError(f"line {line_number}: field {field!r} must be finite")
    return converted


def _backend(record: Mapping[str, Any], line_number: int) -> str:
    explicit = record.get("backend")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise BenchOutputError(
                f"line {line_number}: field 'backend' must be a non-empty string"
            )
        normalized_explicit = explicit.strip().casefold()
        if normalized_explicit in {"metal", "mtl"}:
            return "Metal"
        return explicit.strip()

    backends = record.get("backends")
    if backends is not None:
        if not isinstance(backends, str) or not backends.strip():
            raise BenchOutputError(
                f"line {line_number}: field 'backends' must be a non-empty string"
            )
        names = [name.strip() for name in backends.replace(";", ",").split(",")]
        normalized = {
            "metal" if name.casefold() == "mtl" else name.casefold()
            for name in names
            if name
        }
        if "metal" in normalized:
            return "Metal"
        if normalized <= {"cpu", "blas"}:
            return "CPU"
        return names[0]

    candidates = (
        ("metal", "Metal"),
        ("cuda", "CUDA"),
        ("vulkan", "Vulkan"),
        ("sycl", "SYCL"),
        ("kompute", "Kompute"),
        ("rpc", "RPC"),
    )
    active: list[str] = []
    for field, name in candidates:
        value = record.get(field, False)
        if not isinstance(value, bool):
            raise BenchOutputError(f"line {line_number}: field {field!r} must be boolean")
        if value:
            active.append(name)

    if active:
        if "Metal" in active:
            return "Metal"
        return active[0]

    # llama-bench does not emit an explicit ``cpu`` flag.  No accelerator flag
    # therefore means the CPU backend, including BLAS-accelerated CPU runs.
    return "CPU"


def _test_identity(record: Mapping[str, Any], line_number: int) -> tuple[str, int]:
    """Normalize legacy and current llama-bench workload fields."""

    explicit = record.get("test")
    if explicit is not None:
        test_type = _expect_string(record, "test", line_number)
        if test_type not in {"pp", "tg"}:
            raise BenchOutputError(
                f"line {line_number}: field 'test' must be 'pp' or 'tg', "
                f"got {test_type!r}"
            )
        size_field = "n_prompt" if test_type == "pp" else "n_gen"
        test_size = _expect_int(record, size_field, line_number)
    else:
        n_prompt = _expect_int(record, "n_prompt", line_number)
        n_gen = _expect_int(record, "n_gen", line_number)
        if n_prompt > 0 and n_gen == 0:
            test_type, test_size = "pp", n_prompt
        elif n_gen > 0 and n_prompt == 0:
            test_type, test_size = "tg", n_gen
        else:
            raise BenchOutputError(
                f"line {line_number}: expected one pp or tg workload; "
                f"n_prompt={n_prompt}, n_gen={n_gen}"
            )
    if test_size <= 0:
        raise BenchOutputError(
            f"line {line_number}: {test_type} test size must be greater than zero"
        )
    return test_type, test_size


def _throughput(record: Mapping[str, Any], line_number: int) -> float:
    """Normalize legacy ``t_s`` and current ``avg_ts`` throughput fields."""

    field = "t_s" if "t_s" in record else "avg_ts"
    value = _expect_number(record, field, line_number)
    if value <= 0.0:
        raise BenchOutputError(
            f"line {line_number}: field {field!r} must be greater than zero"
        )
    return value


def _device_description(
    record: Mapping[str, Any], backend: str, line_number: int
) -> str:
    """Normalize device identity across llama-bench output generations."""

    if "device_description" in record:
        return _expect_string(record, "device_description", line_number)
    preferred = "cpu_info" if backend == "CPU" else "gpu_info"
    fallback = "gpu_info" if preferred == "cpu_info" else "cpu_info"
    value = record.get(preferred) or record.get(fallback)
    if not isinstance(value, str) or not value.strip():
        raise BenchOutputError(
            f"line {line_number}: current schema requires non-empty "
            "'gpu_info' or 'cpu_info'"
        )
    return value


def _optional_value(
    record: Mapping[str, Any],
    field: str,
    default: ConfigValue,
    expected_types: tuple[type, ...],
    line_number: int,
    *,
    aliases: tuple[str, ...] = (),
) -> ConfigValue:
    """Read and type-check an optional runtime-configuration value."""

    selected = next((name for name in (field, *aliases) if name in record), None)
    if selected is None:
        return default
    value = record[selected]
    if isinstance(value, bool) and bool not in expected_types:
        valid = False
    else:
        valid = isinstance(value, expected_types)
    if not valid:
        expected = "/".join(item.__name__ for item in expected_types)
        raise BenchOutputError(
            f"line {line_number}: field {selected!r} must be {expected}"
        )
    return value


def _normalized_tensor_split(value: ConfigValue) -> str:
    """Normalize equivalent tensor-split spellings such as ``0``/``0.00``."""

    text = str(value)
    try:
        parts = [part for part in text.replace(",", "/").split("/") if part]
        return "/".join(f"{float(part):g}" for part in parts)
    except ValueError:
        return text


def _backend_configuration(
    record: Mapping[str, Any], line_number: int
) -> tuple[str, bool]:
    """Normalize full backend capabilities and GPU acceleration state."""

    current = record.get("backends")
    accelerator_names = {"metal", "cuda", "vulkan", "sycl", "kompute", "rpc"}
    if current is not None:
        if not isinstance(current, str) or not current.strip():
            raise BenchOutputError(
                f"line {line_number}: field 'backends' must be a non-empty string"
            )
        capabilities = {
            "metal" if name.strip().casefold() == "mtl" else name.strip().casefold()
            for name in current.replace(";", ",").split(",")
            if name.strip()
        }
        return ",".join(sorted(capabilities)), bool(
            capabilities & accelerator_names
        )

    capabilities: set[str] = set()
    for field in accelerator_names:
        value = record.get(field, False)
        if not isinstance(value, bool):
            raise BenchOutputError(
                f"line {line_number}: field {field!r} must be boolean"
            )
        if value:
            capabilities.add(field)
    blas = record.get("blas", False)
    if not isinstance(blas, bool):
        raise BenchOutputError(
            f"line {line_number}: field 'blas' must be boolean"
        )
    if blas:
        capabilities.add("blas")
    if not capabilities:
        capabilities.add("cpu")

    emitted_gpu_state = record.get("gpu_blas")
    if emitted_gpu_state is None:
        gpu_acceleration = bool(capabilities & accelerator_names)
    elif isinstance(emitted_gpu_state, bool):
        gpu_acceleration = emitted_gpu_state
    else:
        raise BenchOutputError(
            f"line {line_number}: field 'gpu_blas' must be boolean"
        )
    return ",".join(sorted(capabilities)), gpu_acceleration


def _direct_io_enabled(record: Mapping[str, Any], line_number: int) -> bool:
    """Normalize positive current and negative legacy direct-I/O fields."""

    if "use_direct_io" in record:
        return bool(
            _optional_value(
                record, "use_direct_io", False, (bool,), line_number
            )
        )
    if "no_direct_io" in record:
        disabled = _optional_value(
            record, "no_direct_io", True, (bool,), line_number
        )
        return not bool(disabled)
    return False


def _runtime_configuration(
    record: Mapping[str, Any], line_number: int
) -> tuple[tuple[str, ConfigValue], ...]:
    """Capture all emitted performance-affecting settings in canonical form."""

    flash = _optional_value(
        record, "flash_attn", "auto", (str, int, bool), line_number
    )
    if isinstance(flash, bool):
        flash = "on" if flash else "off"
    elif isinstance(flash, int):
        flash = {-1: "auto", 0: "off", 1: "on"}.get(flash, flash)
    no_op = _optional_value(
        record, "no_op_offload", False, (bool, int), line_number
    )
    no_host = _optional_value(record, "no_host", False, (bool, int), line_number)
    tensor_split = _optional_value(
        record, "tensor_split", "0", (str, int, float), line_number
    )
    backend_capabilities, gpu_acceleration = _backend_configuration(
        record, line_number
    )
    configuration: tuple[tuple[str, ConfigValue], ...] = (
        ("backend_capabilities", backend_capabilities),
        ("gpu_acceleration", gpu_acceleration),
        ("cpu_mask", _optional_value(record, "cpu_mask", "0x0", (str,), line_number)),
        ("cpu_strict", _optional_value(record, "cpu_strict", False, (bool,), line_number)),
        ("poll", _optional_value(record, "poll", 50, (int,), line_number)),
        ("n_cpu_moe", _optional_value(record, "n_cpu_moe", 0, (int,), line_number)),
        ("split_mode", _optional_value(record, "split_mode", "layer", (str,), line_number)),
        ("main_gpu", _optional_value(record, "main_gpu", 0, (int,), line_number)),
        ("no_kv_offload", _optional_value(record, "no_kv_offload", False, (bool,), line_number)),
        ("flash_attn", flash),
        ("devices", _optional_value(record, "devices", "auto", (str,), line_number, aliases=("device",))),
        ("tensor_split", _normalized_tensor_split(tensor_split)),
        ("tensor_buft_overrides", _optional_value(record, "tensor_buft_overrides", "none", (str,), line_number)),
        ("use_mmap", _optional_value(record, "use_mmap", True, (bool,), line_number)),
        ("use_direct_io", _direct_io_enabled(record, line_number)),
        ("embeddings", _optional_value(record, "embeddings", False, (bool,), line_number)),
        ("no_op_offload", bool(no_op)),
        ("no_host", bool(no_host)),
        ("fit_target", _optional_value(record, "fit_target", 0, (int,), line_number)),
        ("fit_min_ctx", _optional_value(record, "fit_min_ctx", 0, (int,), line_number)),
        ("n_depth", _optional_value(record, "n_depth", 0, (int,), line_number)),
    )
    return configuration


def _parse_record(record: Mapping[str, Any], line_number: int) -> BenchSample:
    # Validate stable identity/configuration fields first. This keeps malformed
    # output diagnostics deterministic rather than dependent on test type.
    model_filename = _expect_string(record, "model_filename", line_number)
    model_size = _expect_int(record, "model_size", line_number)
    model_n_params = _expect_int(record, "model_n_params", line_number)
    n_gpu_layers = _expect_int(record, "n_gpu_layers", line_number)
    n_batch = _expect_int(record, "n_batch", line_number)
    n_ubatch = _expect_int(record, "n_ubatch", line_number)
    type_k = _expect_string(record, "type_k", line_number)
    type_v = _expect_string(record, "type_v", line_number)
    n_threads = _expect_int(record, "n_threads", line_number)
    build_commit = _expect_string(record, "build_commit", line_number)
    build_number = _expect_int(record, "build_number", line_number)
    backend = _backend(record, line_number)
    device_description = _device_description(record, backend, line_number)
    test_type, test_size = _test_identity(record, line_number)
    tokens_per_second = _throughput(record, line_number)
    runtime_configuration = _runtime_configuration(record, line_number)

    sample = BenchSample(
        model_filename=model_filename,
        model_size=model_size,
        model_n_params=model_n_params,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        type_k=type_k,
        type_v=type_v,
        n_threads=n_threads,
        test_type=test_type,
        test_size=test_size,
        tokens_per_second=tokens_per_second,
        build_commit=build_commit,
        build_number=build_number,
        backend=backend,
        device_description=device_description,
        runtime_configuration=runtime_configuration,
    )

    non_negative = {
        "model_size": sample.model_size,
        "model_n_params": sample.model_n_params,
        "n_batch": sample.n_batch,
        "n_ubatch": sample.n_ubatch,
        "n_threads": sample.n_threads,
        "build_number": sample.build_number,
    }
    for field, value in non_negative.items():
        if value < 0:
            raise BenchOutputError(
                f"line {line_number}: field {field!r} must not be negative"
            )
    if sample.n_threads == 0:
        raise BenchOutputError(f"line {line_number}: field 'n_threads' must be positive")
    return sample


def parse_bench_output(text: str) -> list[BenchSample]:
    """Parse newline-delimited JSON emitted by one ``llama-bench`` invocation.

    Args:
        text: Complete standard output from ``llama-bench -o jsonl``.

    Returns:
        Parsed samples in their original output order.

    Raises:
        BenchOutputError: If output is empty, malformed, or violates the schema.
    """

    if not isinstance(text, str):
        raise BenchOutputError("benchmark output must be text")

    samples: list[BenchSample] = []
    seen_tests: set[tuple[str, int]] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchOutputError(
                f"line {line_number}: invalid JSON ({exc.msg} at column {exc.colno})"
            ) from exc
        if not isinstance(value, dict):
            raise BenchOutputError(f"line {line_number}: expected a JSON object")
        sample = _parse_record(value, line_number)
        test_key = (sample.test_type, sample.test_size)
        if test_key in seen_tests:
            raise BenchOutputError(
                f"line {line_number}: duplicate {sample.test_type}{sample.test_size} sample"
            )
        seen_tests.add(test_key)
        samples.append(sample)

    if not samples:
        raise BenchOutputError("benchmark output contained no JSONL samples")
    return samples


def _config_signature(sample: BenchSample) -> tuple[object, ...]:
    return (
        sample.test_type,
        sample.test_size,
        sample.model_filename,
        sample.model_size,
        sample.model_n_params,
        sample.n_gpu_layers,
        sample.n_batch,
        sample.n_ubatch,
        sample.type_k,
        sample.type_v,
        sample.n_threads,
        sample.backend,
        sample.device_description,
        sample.runtime_configuration,
    )


def validate_config_match(
    base_samples: Iterable[BenchSample], head_samples: Iterable[BenchSample]
) -> None:
    """Verify base and head samples used the same model, flags, and backend.

    Build identity and measured throughput are intentionally excluded: those are
    expected to differ between the two comparison arms.

    Args:
        base_samples: Samples produced by the base binary.
        head_samples: Samples produced by the head binary.

    Raises:
        BenchOutputError: If either side is empty or any configuration differs.
    """

    base = list(base_samples)
    head = list(head_samples)
    if not base or not head:
        raise BenchOutputError("base and head must each contain at least one sample")

    base_by_test = {(sample.test_type, sample.test_size): sample for sample in base}
    head_by_test = {(sample.test_type, sample.test_size): sample for sample in head}
    if len(base_by_test) != len(base) or len(head_by_test) != len(head):
        raise BenchOutputError("base and head samples must not contain duplicate tests")
    if base_by_test.keys() != head_by_test.keys():
        raise BenchOutputError(
            "base/head test workloads differ: "
            f"base={sorted(base_by_test)}, head={sorted(head_by_test)}"
        )

    for test_key in sorted(base_by_test):
        base_signature = _config_signature(base_by_test[test_key])
        head_signature = _config_signature(head_by_test[test_key])
        if base_signature != head_signature:
            field_names = (
                "test_type",
                "test_size",
                "model_filename",
                "model_size",
                "model_n_params",
                "n_gpu_layers",
                "n_batch",
                "n_ubatch",
                "type_k",
                "type_v",
                "n_threads",
                "backend",
                "device_description",
                "runtime_configuration",
            )
            differences = [
                f"{name}: {left!r} != {right!r}"
                for name, left, right in zip(
                    field_names, base_signature, head_signature, strict=True
                )
                if left != right
            ]
            raise BenchOutputError(
                f"base/head configuration mismatch for {test_key[0]}{test_key[1]} "
                f"({'; '.join(differences)})"
            )


# Short, discoverable alias for callers that name the wire format directly.
parse_jsonl = parse_bench_output
