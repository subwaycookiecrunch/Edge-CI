"""Tests for canonical, terminal, and Markdown report generation."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from rich.console import Console

from edgeci.adapter import BenchSample
from edgeci.config import EdgeCIConfig, ModelConfig
from edgeci.orchestrator import ComparisonSession, EnvironmentSnapshot, MeasuredInvocation
from edgeci.probe import (
    HardwareFingerprint,
    MemoryPressure,
    MemoryState,
    PowerState,
    ThermalState,
)
from edgeci.report import (
    build_canonical_report,
    generate_json_report,
    generate_markdown_report,
    render_terminal_report,
    write_reports,
)
from edgeci.schedule import Invocation
from edgeci.stats import AnalysisResult, MetricAnalysis
from edgeci.verdict import MetricVerdict, SessionVerdict


def _sample(test_type: str, rate: float, *, commit: str, build: int) -> BenchSample:
    return BenchSample(
        model_filename="test-model.gguf",
        model_size=4_912_345_678,
        model_n_params=8_030_000_000,
        n_gpu_layers=-1,
        n_batch=2048,
        n_ubatch=512,
        type_k="f16",
        type_v="f16",
        n_threads=6,
        test_type=test_type,
        test_size=128 if test_type == "tg" else 512,
        tokens_per_second=rate,
        build_commit=commit,
        build_number=build,
        backend="Metal",
        device_description="Apple M5",
    )


def _environment(timestamp: datetime) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        timestamp=timestamp,
        monotonic_seconds=100.0,
        thermal=ThermalState.NOMINAL,
        power=PowerState(ac_connected=True, low_power_mode=False),
        memory=MemoryState(
            page_size=16_384,
            free_pages=100_000,
            pageouts=0,
            compressed_pages=10_000,
            total_bytes=24 * 1024**3,
            pressure_level=MemoryPressure.NORMAL,
        ),
        cpu_busy_fraction=0.04,
    )


def _metric(name: str, base: float, head: float) -> MetricAnalysis:
    return MetricAnalysis(
        metric_name=name,
        base_values=[base, base * 1.01],
        head_values=[head, head * 1.01],
        log_ratios=[0.12, 0.11],
        ab_ratios=[0.12],
        ba_ratios=[0.11],
        block_averages=[0.115],
        point_estimate=0.115,
        bootstrap_interval=(0.115, 0.10, 0.13),
        block_t_interval=(0.115, 0.09, 0.14),
        base_geometric_mean=base,
        head_geometric_mean=head,
        base_cv=0.007,
        head_cv=0.008,
        log_ratio_sd=0.018,
        median_change=-10.9,
        iqr=(-11.5, -10.2),
        block_drift=0.01,
    )


def _fixture_objects(tmp_path: Path) -> tuple[ComparisonSession, AnalysisResult, SessionVerdict]:
    start = datetime(2026, 7, 17, 1, 2, 3, tzinfo=timezone.utc)
    base_invocation = Invocation("base", 0, 0, 0, False)
    head_invocation = Invocation("head", 0, 0, 1, False)
    environment = _environment(start)
    base_result = MeasuredInvocation(
        invocation=base_invocation,
        arm="base",
        bench_samples=(
            _sample("pp", 812.0, commit="abc1234", build=9991),
            _sample("tg", 54.2, commit="abc1234", build=9991),
        ),
        env_before=environment,
        env_after=replace(environment, monotonic_seconds=101.0),
        wall_time=1.0,
        stdout="{}",
        stderr="",
        exit_code=0,
        command=("/base/llama-bench", "-o", "jsonl"),
    )
    head_result = MeasuredInvocation(
        invocation=head_invocation,
        arm="head",
        bench_samples=(
            _sample("pp", 806.0, commit="def5678", build=10002),
            _sample("tg", 48.1, commit="def5678", build=10002),
        ),
        env_before=replace(environment, monotonic_seconds=102.0),
        env_after=replace(environment, monotonic_seconds=103.0),
        wall_time=1.1,
        stdout="{}",
        stderr="",
        exit_code=0,
        command=("/head/llama-bench", "-o", "jsonl"),
    )
    session = ComparisonSession(
        schedule=(base_invocation, head_invocation),
        warmup_results=(),
        measured_results=(base_result, head_result),
        hardware=HardwareFingerprint(
            chip_name="Apple M5",
            logical_cpu_count=10,
            performance_core_count=6,
            efficiency_core_count=4,
            memory_bytes=24 * 1024**3,
            gpu_core_count=10,
            macos_version="26.1",
            os_build="25B78",
            model_identifier="Mac16,1",
        ),
        base_binary_path=tmp_path / "base-llama-bench",
        head_binary_path=tmp_path / "head-llama-bench",
        model_path=tmp_path / "test-model.gguf",
        base_binary_sha="a" * 64,
        head_binary_sha="b" * 64,
        model_sha="c" * 64,
        config=EdgeCIConfig(model=ModelConfig(tmp_path / "test-model.gguf")),
        seed="report-test",
        start_time=start,
        end_time=start + timedelta(minutes=28, seconds=14),
        preflight_duration=60.0,
        contaminated_blocks=(),
        abort_reason=None,
    )
    analysis = AnalysisResult(
        tg=_metric("tg128", 54.2, 48.1),
        pp=_metric("pp512", 812.0, 806.0),
        n_valid_pairs=20,
        n_contaminated_blocks=0,
    )
    metric_verdicts = [
        MetricVerdict(
            metric_name="tg128",
            verdict="FAIL",
            budget=0.05,
            budget_boundary=0.051293,
            bootstrap_lower=0.10,
            bootstrap_upper=0.13,
            block_t_lower=0.09,
            block_t_upper=0.14,
            human_change_pct=-11.3,
            human_interval_lower_pct=-13.1,
            human_interval_upper_pct=-9.4,
        ),
        MetricVerdict(
            metric_name="pp512",
            verdict="PASS",
            budget=0.05,
            budget_boundary=0.051293,
            bootstrap_lower=-0.008,
            bootstrap_upper=0.021,
            block_t_lower=-0.006,
            block_t_upper=0.019,
            human_change_pct=-0.7,
            human_interval_lower_pct=-2.1,
            human_interval_upper_pct=0.8,
        ),
    ]
    verdict = SessionVerdict(
        overall="FAIL",
        metrics=metric_verdicts,
        experimental=True,
        abort_reason=None,
        warnings=["tg128: example variability warning"],
    )
    return session, analysis, verdict


def test_json_report_is_valid_and_complete(tmp_path: Path) -> None:
    session, analysis, verdict = _fixture_objects(tmp_path)

    payload = json.loads(generate_json_report(session, analysis, verdict))

    assert payload["protocol_version"] == "0.1.0"
    assert payload["timestamp"] == "2026-07-17T01:02:03+00:00"
    assert payload["duration_seconds"] == 1694.0
    assert payload["schedule_seed"] == "report-test"
    assert payload["hardware"]["chip_name"] == "Apple M5"
    assert payload["binaries"]["base"]["build_commit"] == "abc1234"
    assert payload["binaries"]["head"]["build_number"] == 10002
    assert payload["model"]["filename"] == "test-model.gguf"
    assert len(payload["raw_measurements"]) == 2
    assert payload["raw_measurements"][0]["env_before"]["thermal"] == 0
    assert payload["analysis"]["tg"]["metric_name"] == "tg128"
    assert payload["verdict"]["overall"] == "FAIL"
    assert payload["warnings"] == ["tg128: example variability warning"]


def test_canonical_report_serializes_paths_and_enums(tmp_path: Path) -> None:
    session, analysis, verdict = _fixture_objects(tmp_path)

    payload = build_canonical_report(session, analysis, verdict)

    assert payload["model"]["path"] == str(tmp_path / "test-model.gguf")
    assert payload["config"]["model"]["path"] == str(tmp_path / "test-model.gguf")
    json.dumps(payload, allow_nan=False)


def test_terminal_report_contains_decision_evidence_and_environment(tmp_path: Path) -> None:
    session, analysis, verdict = _fixture_objects(tmp_path)
    stream = StringIO()
    console = Console(file=stream, color_system=None, force_terminal=False, width=120)

    returned = render_terminal_report(session, analysis, verdict, console=console)
    displayed = stream.getvalue()

    for text in (
        "regression detected",
        "tg128",
        "FAIL",
        "Apple M5",
        "Backend: Metal",
        "Thermal: nominal at captured samples",
        "Memory: no pressure at captured samples",
    ):
        assert text in displayed
        assert text in returned


def test_terminal_report_does_not_call_unknown_memory_healthy(tmp_path: Path) -> None:
    session, analysis, verdict = _fixture_objects(tmp_path)
    unknown_memory = replace(
        session.measured_results[0].env_before.memory,
        pressure_level=MemoryPressure.UNKNOWN,
    )
    first = session.measured_results[0]
    changed_environment = replace(first.env_before, memory=unknown_memory)
    changed_first = replace(
        first,
        env_before=changed_environment,
        env_after=changed_environment,
    )
    changed_session = replace(
        session,
        measured_results=(changed_first, *session.measured_results[1:]),
    )
    stream = StringIO()
    console = Console(file=stream, color_system=None, force_terminal=False)

    render_terminal_report(changed_session, analysis, verdict, console=console)

    assert "Memory: unknown at one or more captured samples" in stream.getvalue()


def test_markdown_report_is_github_flavored_and_collapsible(tmp_path: Path) -> None:
    session, analysis, verdict = _fixture_objects(tmp_path)

    markdown = generate_markdown_report(session, analysis, verdict)

    assert markdown.startswith("## EdgeCI — ❌ regression detected")
    assert "| Metric | Base | Head | Head vs. base | Budget | Verdict |" in markdown
    assert "<details>" in markdown
    assert "<summary>Raw pair data and environment details</summary>" in markdown
    assert "`test-model.gguf`" in markdown
    assert "- Backend: Metal" in markdown
    assert markdown.rstrip().endswith("_Protocol version 0.1.0_")


def test_aborted_report_omits_analysis_without_crashing(tmp_path: Path) -> None:
    session, _, _ = _fixture_objects(tmp_path)
    aborted = replace(session, abort_reason="INCONCLUSIVE_RESOURCE: thermal recovery timeout")
    verdict = SessionVerdict(
        overall="INCONCLUSIVE",
        metrics=[],
        experimental=True,
        abort_reason=aborted.abort_reason,
        warnings=[],
    )

    payload = json.loads(generate_json_report(aborted, None, verdict))
    markdown = generate_markdown_report(aborted, None, verdict)

    assert payload["analysis"] is None
    assert payload["abort_reason"] == aborted.abort_reason
    assert "thermal recovery timeout" in markdown
    assert "No metric analysis" in markdown


def test_write_reports_persists_all_formats_and_schedule(tmp_path: Path) -> None:
    session, analysis, verdict = _fixture_objects(tmp_path)
    console = Console(file=StringIO(), color_system=None, force_terminal=False)

    written = write_reports(
        tmp_path / "results",
        session,
        analysis,
        verdict,
        fmt="all",
        console=console,
    )

    assert set(written) == {"json", "markdown", "schedule"}
    assert all(path.is_file() for path in written.values())
    schedule = json.loads(written["schedule"].read_text(encoding="utf-8"))
    assert schedule["seed"] == "report-test"
    assert [item["arm"] for item in schedule["schedule"]] == ["base", "head"]
