"""End-to-end orchestration coverage without Apple hardware or real sleeps."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

from edgeci import orchestrator
from edgeci.config import (
    BenchmarkConfig,
    EdgeCIConfig,
    ModelConfig,
    PreflightConfig,
    ReportConfig,
)
from edgeci.probe import (
    HardwareFingerprint,
    LlamaBenchInfo,
    MemoryPressure,
    MemoryState,
    PowerState,
    ThermalState,
)
from edgeci.report import build_canonical_report
from edgeci.runner import RunResult, canonical_llama_bench_contract
from edgeci.verdict import SessionVerdict


def test_run_comparison_records_complete_deterministic_session(
    tmp_path: Path,
    monkeypatch,
    sample_jsonl: str,
) -> None:
    """Drive schedule -> probes -> runner -> parser -> persisted report shape."""

    base = tmp_path / "base-llama-bench"
    head = tmp_path / "head-llama-bench"
    model = tmp_path / "model.gguf"
    for binary in (base, head):
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
    model.write_bytes(b"GGUFdeterministic-integration-fixture")

    config = EdgeCIConfig(
        model=ModelConfig(path=model),
        benchmark=BenchmarkConfig(
            prompt_tokens=512,
            generate_tokens=128,
            pairs=2,
            warmup_pairs=1,
            gap_seconds=0.0,
            timeout_minutes=1.0,
        ),
        preflight=PreflightConfig(
            thermal_settle_seconds=0.0,
            idle_cpu_threshold=0.20,
            post_build_cooldown=0.0,
            preflight_timeout=10.0,
        ),
        report=ReportConfig(output_dir=tmp_path / "results"),
    )

    healthy_memory = MemoryState(
        page_size=16_384,
        free_pages=100_000,
        pageouts=7,
        compressed_pages=1_000,
        total_bytes=16 * 1024**3,
        pressure_level=MemoryPressure.NORMAL,
    )
    hardware = HardwareFingerprint(
        chip_name="Apple M5",
        logical_cpu_count=10,
        performance_core_count=4,
        efficiency_core_count=6,
        memory_bytes=16 * 1024**3,
        gpu_core_count=10,
        macos_version="26.0",
        os_build="25A1",
        model_identifier="MacBookAirFixture",
    )

    # A constant clock is sufficient because every fake operation is immediate;
    # it also proves no path depends on sleeping to make progress.
    monkeypatch.setattr(orchestrator, "make_continuous_clock", lambda: lambda: 100.0)
    monkeypatch.setattr(orchestrator.time, "monotonic", lambda: 50.0)
    def fail_on_sleep(_seconds: float) -> None:
        raise AssertionError("orchestrator unexpectedly attempted a real sleep")

    monkeypatch.setattr(orchestrator.time, "sleep", fail_on_sleep)
    monkeypatch.setattr(orchestrator, "EdgeCILock", nullcontext)
    monkeypatch.setattr(
        orchestrator,
        "detect_llama_bench",
        lambda path, **_: LlamaBenchInfo(path=path, version="fixture-bench"),
    )
    monkeypatch.setattr(
        orchestrator,
        "parse_llama_bench_contract",
        lambda _help_text: canonical_llama_bench_contract(),
    )
    monkeypatch.setattr(orchestrator, "get_performance_core_count", lambda **_: 4)
    monkeypatch.setattr(orchestrator, "get_hardware_fingerprint", lambda **_: hardware)
    monkeypatch.setattr(orchestrator, "get_power_state", lambda **_: PowerState(True, False))
    monkeypatch.setattr(orchestrator, "get_memory_state", lambda **_: healthy_memory)
    monkeypatch.setattr(orchestrator, "get_thermal_state", lambda: ThermalState.NOMINAL)
    monkeypatch.setattr(orchestrator, "get_cpu_load", lambda **_: 0.05)

    fixture_records = [json.loads(line) for line in sample_jsonl.splitlines()]
    launched: list[Path] = []

    def fake_runner(
        binary_path: Path,
        model_path: Path,
        benchmark: BenchmarkConfig,
        **_: object,
    ) -> RunResult:
        assert model_path == model.resolve()
        assert benchmark is config.benchmark
        launched.append(binary_path)
        is_base = binary_path == base.resolve()
        commit = "base123" if is_base else "head456"
        build_number = 100 if is_base else 101
        throughput_scale = 1.0 if is_base else 0.98
        records = []
        for source in fixture_records:
            record = dict(source)
            record["build_commit"] = commit
            record["build_number"] = build_number
            record["t_s"] = record["t_s"] * throughput_scale
            records.append(record)
        stdout = "\n".join(json.dumps(record) for record in records)
        return RunResult(
            stdout=stdout,
            stderr="",
            exit_code=0,
            duration_seconds=0.25,
            command=(str(binary_path), "-o", "jsonl"),
        )

    monkeypatch.setattr(orchestrator, "run_llama_bench", fake_runner)

    progress: list[orchestrator.ProgressUpdate] = []
    session = orchestrator.run_comparison(
        base,
        head,
        model,
        config,
        seed="integration-seed",
        on_progress=progress.append,
    )

    expected_arms = ["base", "head", "head", "base", "base", "head"]
    assert [invocation.arm for invocation in session.schedule] == expected_arms
    assert [path.name for path in launched] == [
        "base-llama-bench",
        "head-llama-bench",
        "head-llama-bench",
        "base-llama-bench",
        "base-llama-bench",
        "head-llama-bench",
    ]
    assert session.complete
    assert session.abort_reason is None
    assert len(session.warmup_results) == 2
    assert len(session.measured_results) == 4
    assert session.contaminated_blocks == ()
    assert session.seed == "integration-seed"
    assert session.hardware == hardware
    assert session.base_binary_sha == hashlib.sha256(base.read_bytes()).hexdigest()
    assert session.head_binary_sha == hashlib.sha256(head.read_bytes()).hexdigest()
    assert session.model_sha == hashlib.sha256(model.read_bytes()).hexdigest()

    all_results = (*session.warmup_results, *session.measured_results)
    assert [result.arm for result in all_results] == expected_arms
    assert all(result.sample_for("pp").backend == "Metal" for result in all_results)
    assert all(result.sample_for("tg").test_size == 128 for result in all_results)
    assert {
        (result.arm, result.sample_for("tg").build_commit)
        for result in all_results
    } == {("base", "base123"), ("head", "head456")}
    for result in all_results:
        for environment in (result.env_before, result.env_after):
            assert environment.thermal is ThermalState.NOMINAL
            assert environment.power == PowerState(True, False)
            assert environment.memory.pressure_level is MemoryPressure.NORMAL
            assert environment.cpu_busy_fraction == 0.05

    assert progress[0].phase == "provenance"
    assert progress[-1].phase == "complete"
    assert progress[-1].completed == len(session.schedule)

    # This is the exact object graph consumed by JSON/report persistence.
    verdict = SessionVerdict(
        overall="PASS",
        metrics=[],
        experimental=True,
        abort_reason=None,
        warnings=[],
    )
    report = build_canonical_report(session, analysis=None, verdict=verdict)
    json.dumps(report, allow_nan=False)
    assert report["schedule_seed"] == "integration-seed"
    assert len(report["schedule"]) == 6
    assert len(report["warmup_measurements"]) == 2
    assert len(report["raw_measurements"]) == 4
    assert report["binaries"]["base"]["build_commit"] == "base123"
    assert report["binaries"]["head"]["build_commit"] == "head456"
    assert report["model"]["sha256"] == session.model_sha
    assert report["config"]["benchmark"]["pairs"] == 2
