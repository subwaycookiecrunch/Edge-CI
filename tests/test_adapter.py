from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from edgeci.adapter import BenchOutputError, parse_bench_output, validate_config_match


FIXTURE = Path(__file__).parent / "fixtures" / "sample_bench_output.jsonl"


def test_parse_valid_realistic_jsonl() -> None:
    samples = parse_bench_output(FIXTURE.read_text(encoding="utf-8"))

    assert [(sample.test_type, sample.test_size) for sample in samples] == [
        ("pp", 512),
        ("tg", 128),
    ]
    assert samples[0].backend == "Metal"
    assert samples[0].tokens_per_second == pytest.approx(812.45)
    assert samples[1].tokens_per_second == pytest.approx(54.21)


def test_parse_current_llama_bench_jsonl_schema() -> None:
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    for record in records:
        # Current b10052 Apple Silicon builds identify Metal as ``MTL``.
        record["backends"] = "MTL,BLAS"
        record["avg_ts"] = record.pop("t_s")
        record["use_direct_io"] = not record.pop("no_direct_io")
        record["n_cpu_moe"] = 0
        record["devices"] = "auto"
        record["tensor_buft_overrides"] = "none"
        record["no_op_offload"] = 0
        record["no_host"] = False
        record["fit_target"] = 0
        record["fit_min_ctx"] = 0
        record["n_depth"] = 0
        record["flash_attn"] = -1
        record.pop("test")
        record.pop("device_description")
        for field in (
            "cuda",
            "vulkan",
            "kompute",
            "metal",
            "sycl",
            "rpc",
            "blas",
            "gpu_blas",
        ):
            record.pop(field)

    samples = parse_bench_output("\n".join(json.dumps(record) for record in records))

    assert [(sample.test_type, sample.test_size) for sample in samples] == [
        ("pp", 512),
        ("tg", 128),
    ]
    assert all(sample.backend == "Metal" for sample in samples)
    assert samples[0].device_description == "Apple M5"
    runtime = dict(samples[0].runtime_configuration)
    assert runtime["backend_capabilities"] == "blas,metal"
    assert runtime["gpu_acceleration"] is True
    legacy = parse_bench_output(FIXTURE.read_text(encoding="utf-8"))
    validate_config_match(legacy, samples)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("not-json", "invalid JSON"),
        ("[]", "expected a JSON object"),
        ('{"test":"pp"}', "model_filename"),
        ("\n\n", "no JSONL samples"),
    ],
)
def test_rejects_malformed_jsonl(text: str, expected: str) -> None:
    with pytest.raises(BenchOutputError, match=expected):
        parse_bench_output(text)


def test_config_match_accepts_identical_configuration() -> None:
    samples = parse_bench_output(FIXTURE.read_text(encoding="utf-8"))
    head = [replace(sample, build_commit="def5678", tokens_per_second=1.0) for sample in samples]

    validate_config_match(samples, reversed(head))


def test_config_match_rejects_different_backend() -> None:
    samples = parse_bench_output(FIXTURE.read_text(encoding="utf-8"))
    head = [replace(sample, backend="CPU") for sample in samples]

    with pytest.raises(BenchOutputError, match="backend"):
        validate_config_match(samples, head)


def test_config_match_rejects_runtime_flag_difference() -> None:
    samples = parse_bench_output(FIXTURE.read_text(encoding="utf-8"))
    changed = dict(samples[0].runtime_configuration)
    changed["use_mmap"] = False
    head = [
        replace(sample, runtime_configuration=tuple(changed.items()))
        for sample in samples
    ]

    with pytest.raises(BenchOutputError, match="runtime_configuration"):
        validate_config_match(samples, head)


def test_config_match_rejects_backend_capability_difference() -> None:
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    samples = parse_bench_output(FIXTURE.read_text(encoding="utf-8"))
    for record in records:
        record["blas"] = False
    changed = parse_bench_output(
        "\n".join(json.dumps(record) for record in records)
    )

    with pytest.raises(BenchOutputError, match="runtime_configuration"):
        validate_config_match(samples, changed)


def test_boolean_flash_attention_schema_is_normalized() -> None:
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    for record in records:
        record["flash_attn"] = False

    samples = parse_bench_output(
        "\n".join(json.dumps(record) for record in records)
    )

    assert dict(samples[0].runtime_configuration)["flash_attn"] == "off"


def test_legacy_multi_backend_capabilities_preserve_metal_primary() -> None:
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    for record in records:
        record["rpc"] = True

    samples = parse_bench_output(
        "\n".join(json.dumps(record) for record in records)
    )

    assert all(sample.backend == "Metal" for sample in samples)
    assert "rpc" in dict(samples[0].runtime_configuration)[
        "backend_capabilities"
    ]
