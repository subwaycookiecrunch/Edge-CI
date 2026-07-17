"""Canonical JSON, terminal, and Markdown reporting for EdgeCI."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console, RenderableType

    from .orchestrator import ComparisonSession
    from .stats import AnalysisResult
    from .verdict import SessionVerdict


PROTOCOL_VERSION = "0.1.0"


def to_serializable(value: Any) -> Any:
    """Convert nested EdgeCI values into strict JSON-compatible values.

    Dataclasses, enums, paths, timestamps, mappings, and sequences are handled
    recursively. Non-finite floats become ``None`` so generated JSON remains
    standards compliant.

    Args:
        value: Value to convert.

    Returns:
        A value accepted by :func:`json.dumps` with ``allow_nan=False``.
    """

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return to_serializable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_serializable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [to_serializable(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_serializable(item) for item in value]
    return str(value)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_sample_for_arm(session: Any, arm: str) -> Any | None:
    for measured in (
        *(_get(session, "measured_results", ()) or ()),
        *(_get(session, "warmup_results", ()) or ()),
    ):
        measured_arm = _get(measured, "arm") or _get(_get(measured, "invocation"), "arm")
        if measured_arm != arm:
            continue
        samples = _get(measured, "bench_samples", ())
        if samples:
            return samples[0]
        sample = _get(measured, "bench_sample")
        if sample is not None:
            return sample
    return None


def _path_from_session(session: Any, field_name: str, config_section: str | None = None) -> str:
    path = _get(session, field_name)
    if path is None and config_section:
        section = _get(_get(session, "config"), config_section)
        path = _get(section, "path")
    return str(path) if path is not None else ""


def _binary_record(session: Any, arm: str) -> dict[str, Any]:
    sample = _first_sample_for_arm(session, arm)
    path = _path_from_session(session, f"{arm}_binary_path")
    return {
        "path": path,
        "sha256": _get(session, f"{arm}_binary_sha", ""),
        "build_commit": _get(sample, "build_commit", ""),
        "build_number": _get(sample, "build_number"),
    }


def _analysis_record(analysis: Any | None) -> dict[str, Any] | None:
    if analysis is None:
        return None
    return to_serializable(analysis)


def build_canonical_report(
    session: ComparisonSession,
    analysis: AnalysisResult | None,
    verdict: SessionVerdict,
) -> dict[str, Any]:
    """Build the canonical machine-readable EdgeCI result.

    Args:
        session: Completed or aborted comparison session.
        analysis: Statistical analysis, or ``None`` for an incomplete session.
        verdict: Tri-state session verdict.

    Returns:
        Strictly JSON-compatible report dictionary.
    """

    start = _get(session, "start_time")
    end = _get(session, "end_time")
    if isinstance(start, datetime) and isinstance(end, datetime):
        duration = max(0.0, (end - start).total_seconds())
    else:
        duration = float(_get(session, "duration_seconds", 0.0) or 0.0)

    model_path = _path_from_session(session, "model_path", "model")
    config = _get(session, "config")
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": to_serializable(start),
        "end_timestamp": to_serializable(end),
        "duration_seconds": duration,
        "preflight_duration_seconds": float(
            _get(session, "preflight_duration", 0.0) or 0.0
        ),
        "hardware": to_serializable(_get(session, "hardware")),
        "binaries": {
            "base": _binary_record(session, "base"),
            "head": _binary_record(session, "head"),
        },
        "model": {
            "path": model_path,
            "sha256": _get(session, "model_sha", ""),
            "filename": Path(model_path).name if model_path else "",
        },
        "config": to_serializable(config),
        "schedule_seed": str(
            _get(session, "schedule_seed", _get(session, "seed", "")) or ""
        ),
        "schedule": to_serializable(_get(session, "schedule", ())),
        "warmup_measurements": to_serializable(_get(session, "warmup_results", ())),
        "raw_measurements": to_serializable(_get(session, "measured_results", ())),
        "analysis": _analysis_record(analysis),
        "verdict": to_serializable(verdict),
        "warnings": to_serializable(_get(verdict, "warnings", ())),
        "contaminated_blocks": to_serializable(
            _get(session, "contaminated_blocks", ())
        ),
        "abort_reason": _get(session, "abort_reason"),
    }
    return to_serializable(report)


def generate_json_report(
    session: ComparisonSession,
    analysis: AnalysisResult | None,
    verdict: SessionVerdict,
) -> str:
    """Render the canonical report as indented JSON.

    Args:
        session: Completed or aborted comparison session.
        analysis: Statistical analysis, or ``None`` when unavailable.
        verdict: Tri-state session verdict.

    Returns:
        JSON document ending with a newline.
    """

    payload = build_canonical_report(session, analysis, verdict)
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def _metric_analyses(analysis: Any | None) -> list[Any]:
    if analysis is None:
        return []
    return [metric for metric in (_get(analysis, "tg"), _get(analysis, "pp")) if metric]


def _metric_verdicts(verdict: Any) -> dict[str, Any]:
    return {
        str(_get(metric, "metric_name")): metric
        for metric in (_get(verdict, "metrics", ()) or ())
    }


def _format_rate(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if abs(number) >= 100:
        return f"{number:.0f} t/s"
    return f"{number:.1f} t/s"


def _format_pct(value: Any, *, signed: bool = True, decimals: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if abs(number) < 0.5 * 10 ** (-decimals):
        number = 0.0
    magnitude = f"{abs(number):.{decimals}f}%"
    if number < 0:
        return f"−{magnitude}"
    if signed and number > 0:
        return f"+{magnitude}"
    return magnitude


def _change_cell(metric_verdict: Any) -> str:
    if metric_verdict is None:
        return "—"
    point = _format_pct(_get(metric_verdict, "human_change_pct"))
    lower = _format_pct(_get(metric_verdict, "human_interval_lower_pct"))
    upper = _format_pct(_get(metric_verdict, "human_interval_upper_pct"))
    return f"{point} [{lower}, {upper}]"


def _status_icon(status: str) -> str:
    return {"PASS": "✅", "FAIL": "❌", "INCONCLUSIVE": "⚪"}.get(status, "⚪")


def _status_title(verdict: Any) -> str:
    status = str(_get(verdict, "overall", "INCONCLUSIVE"))
    phrase = {
        "PASS": "non-inferior under protocol",
        "FAIL": "regression detected",
        "INCONCLUSIVE": "inconclusive",
    }.get(status, "inconclusive")
    suffix = " (experimental)" if _get(verdict, "experimental", False) else ""
    return f"EdgeCI — {_status_icon(status)} {phrase}{suffix}"


def _summary(verdict: Any) -> str:
    abort_reason = _get(verdict, "abort_reason")
    if abort_reason:
        return f"Comparison stopped before a verdict could be established: {abort_reason}"
    metrics = list(_metric_verdicts(verdict).values())
    failed = next((metric for metric in metrics if _get(metric, "verdict") == "FAIL"), None)
    if failed is not None:
        name = str(_get(failed, "metric_name", "metric"))
        label = "Decode throughput" if name.startswith("tg") else "Prompt processing"
        change = float(_get(failed, "human_change_pct", 0.0) or 0.0)
        budget = float(_get(failed, "budget", 0.0) or 0.0) * 100.0
        direction = "fell" if change < 0 else "changed"
        return (
            f"{label} {direction} {abs(change):.1f}%; both intervals exceed "
            f"the {budget:g}% regression budget."
        )
    overall = _get(verdict, "overall", "INCONCLUSIVE")
    if overall == "PASS":
        return "Both throughput metrics remain inside their preregistered regression budgets."
    return "The confidence intervals do not support a conclusive pass or fail."


def _hardware_summary(session: Any) -> str:
    hardware = _get(session, "hardware")
    summary = _get(hardware, "summary")
    if summary:
        return str(summary)
    chip = _get(hardware, "chip_name", "Unknown Apple Silicon")
    memory = _get(hardware, "memory_bytes") or _get(hardware, "memory_size")
    memory_text = f"{float(memory) / (1024**3):.0f} GB" if memory else "unknown memory"
    macos = _get(hardware, "macos_version", "unknown macOS")
    return f"{chip} · {memory_text} · macOS {macos}"


def _build_summary(session: Any) -> str:
    records = [_binary_record(session, arm) for arm in ("base", "head")]

    def label(record: Mapping[str, Any]) -> str:
        number = record.get("build_number")
        commit = str(record.get("build_commit") or "unknown")
        build = f"b{number}" if number is not None else "build"
        return f"{build} ({commit[:12]})"

    return f"llama.cpp {label(records[0])} → {label(records[1])}"


def _model_summary(session: Any) -> str:
    model_path = _path_from_session(session, "model_path", "model")
    model_sha = str(_get(session, "model_sha", ""))
    short_sha = f"{model_sha[:8]}…" if len(model_sha) > 8 else model_sha or "unknown"
    return f"Model: {Path(model_path).name or 'unknown'} (sha: {short_sha})"


def _backend_summary(session: Any) -> str:
    """Return the validated runtime backend represented in session samples."""

    backends = {
        str(backend)
        for arm in ("base", "head")
        if (sample := _first_sample_for_arm(session, arm)) is not None
        if (backend := _get(sample, "backend"))
    }
    if not backends:
        return "Backend: unavailable"
    label = "Backend" if len(backends) == 1 else "Backends"
    return f"{label}: {', '.join(sorted(backends))}"


def _state_name(value: Any) -> str:
    if value is None:
        return "unknown"
    name = getattr(value, "name", None)
    if name:
        return str(name).lower()
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _environment_summary(session: Any) -> tuple[str, str, str]:
    environments: list[Any] = []
    for measured in _get(session, "measured_results", ()) or ():
        environments.extend(
            environment
            for environment in (
                _get(measured, "env_before"),
                _get(measured, "env_after"),
            )
            if environment is not None
        )
    thermal_states = {
        _state_name(_get(environment, "thermal_state", _get(environment, "thermal")))
        for environment in environments
    }
    thermal_unknown = "unknown" in thermal_states
    thermal_states.discard("unknown")
    if not environments:
        thermal = "Thermal: no samples"
    elif thermal_unknown:
        thermal = "Thermal: unknown at one or more captured samples"
    elif thermal_states <= {"nominal", "0"}:
        thermal = "Thermal: nominal at captured samples"
    else:
        thermal = f"Thermal states: {', '.join(sorted(thermal_states))}"

    pressure_states: set[str] = set()
    power_disconnect = False
    for environment in environments:
        memory = _get(environment, "memory") or _get(environment, "memory_state")
        pressure_states.add(
            _state_name(_get(memory, "pressure_level", _get(environment, "memory_pressure")))
        )
        power = _get(environment, "power") or _get(environment, "power_state")
        if power is not None and _get(power, "ac_connected") is False:
            power_disconnect = True
    pressure_unknown = "unknown" in pressure_states
    pressure_states.discard("unknown")
    pressured = pressure_states - {"normal", "nominal", "0"}
    if not environments:
        memory_text = "Memory: no samples"
        power_text = "Power: no samples"
    else:
        if pressure_unknown:
            memory_text = "Memory: unknown at one or more captured samples"
        elif pressured:
            memory_text = f"Memory pressure: {', '.join(sorted(pressured))}"
        else:
            memory_text = "Memory: no pressure at captured samples"
        power_text = (
            "Power: disconnect observed"
            if power_disconnect
            else "Power: AC at captured samples"
        )
    return thermal, memory_text, power_text


def _duration_text(session: Any) -> str:
    start = _get(session, "start_time")
    end = _get(session, "end_time")
    seconds = (end - start).total_seconds() if isinstance(start, datetime) and isinstance(end, datetime) else 0
    seconds = max(0, int(round(seconds)))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m{remainder:02d}s" if hours else f"{minutes:d}m{remainder:02d}s"


def _terminal_renderables(
    session: Any,
    analysis: Any | None,
    verdict: Any,
    output_path: Path | None,
) -> list[RenderableType]:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    overall = str(_get(verdict, "overall", "INCONCLUSIVE"))
    style = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "yellow"}.get(
        overall, "yellow"
    )
    renderables: list[RenderableType] = [
        Panel(_summary(verdict), title=_status_title(verdict), border_style=style)
    ]

    metrics = _metric_analyses(analysis)
    metric_verdicts = _metric_verdicts(verdict)
    if metrics:
        table = Table(show_header=True, header_style="bold")
        for column, justify in (
            ("Metric", "left"),
            ("Base", "right"),
            ("Head", "right"),
            ("Head vs. base", "right"),
            ("Budget", "right"),
            ("Verdict", "left"),
        ):
            table.add_column(column, justify=justify)
        for metric in metrics:
            name = str(_get(metric, "metric_name", "metric"))
            metric_verdict = metric_verdicts.get(name)
            status = str(_get(metric_verdict, "verdict", "INCONCLUSIVE"))
            budget = float(_get(metric_verdict, "budget", 0.0) or 0.0) * 100.0
            table.add_row(
                name,
                _format_rate(_get(metric, "base_geometric_mean")),
                _format_rate(_get(metric, "head_geometric_mean")),
                _change_cell(metric_verdict),
                _format_pct(-budget, signed=False, decimals=0),
                f"{_status_icon(status)} {status}",
            )
        renderables.append(table)

    evidence_lines: list[str] = []
    if analysis is not None:
        pairs = int(_get(analysis, "n_valid_pairs", 0) or 0)
        block_count = len(_get(_metric_analyses(analysis)[0], "block_averages", ())) if metrics else 0
        evidence_lines.append(
            f"{pairs} pairs · {block_count} balanced ABBA/BAAB blocks · bootstrap + block-t"
        )
        for metric in metrics:
            evidence_lines.append(
                f"{_get(metric, 'metric_name', 'metric')}: log-ratio SD "
                f"{_format_pct(float(_get(metric, 'log_ratio_sd', 0.0) or 0.0) * 100, signed=False)} "
                f"· Base CV {_format_pct(float(_get(metric, 'base_cv', 0.0) or 0.0) * 100, signed=False)} "
                f"· Head CV {_format_pct(float(_get(metric, 'head_cv', 0.0) or 0.0) * 100, signed=False)}"
            )
    else:
        evidence_lines.append("No statistical analysis: comparison did not complete.")
    warnings = list(_get(verdict, "warnings", ()) or ())
    evidence_lines.extend(f"Warning: {warning}" for warning in warnings)
    renderables.append(Panel("\n".join(evidence_lines), title="Evidence"))

    contaminated = len(_get(session, "contaminated_blocks", ()) or ())
    thermal_text, memory_text, power_text = _environment_summary(session)
    environment_lines = [
        _hardware_summary(session),
        _backend_summary(session),
        _build_summary(session),
        _model_summary(session),
        thermal_text,
        memory_text,
        power_text,
        f"External-event invalidations: {contaminated} block(s)",
        f"Duration: {_duration_text(session)}",
    ]
    if output_path is not None:
        environment_lines.append(f"Output: {output_path}")
    renderables.append(
        Panel(Group(*(Text(line) for line in environment_lines)), title="Environment")
    )
    return renderables


def render_terminal_report(
    session: ComparisonSession,
    analysis: AnalysisResult | None,
    verdict: SessionVerdict,
    *,
    console: Console | None = None,
    output_path: Path | None = None,
) -> str:
    """Print a Rich terminal report and return its plain-text rendering.

    Args:
        session: Completed or aborted comparison session.
        analysis: Statistical analysis, or ``None`` when unavailable.
        verdict: Tri-state session verdict.
        console: Optional Rich console used for output.
        output_path: Optional JSON result path displayed in the report.

    Returns:
        Plain-text rendering useful for logs and tests.
    """

    from rich.console import Console

    renderables = _terminal_renderables(session, analysis, verdict, output_path)
    target = console or Console()
    for renderable in renderables:
        target.print(renderable)

    stream = StringIO()
    recorder = Console(file=stream, color_system=None, force_terminal=False, width=120)
    for renderable in renderables:
        recorder.print(renderable)
    return stream.getvalue()


def _markdown_metric_table(analysis: Any | None, verdict: Any) -> list[str]:
    metrics = _metric_analyses(analysis)
    if not metrics:
        return ["_No metric analysis is available for this session._"]
    metric_verdicts = _metric_verdicts(verdict)
    lines = [
        "| Metric | Base | Head | Head vs. base | Budget | Verdict |",
        "|:--|--:|--:|--:|--:|:--|",
    ]
    for metric in metrics:
        name = str(_get(metric, "metric_name", "metric"))
        metric_verdict = metric_verdicts.get(name)
        status = str(_get(metric_verdict, "verdict", "INCONCLUSIVE"))
        budget = float(_get(metric_verdict, "budget", 0.0) or 0.0) * 100.0
        lines.append(
            "| {name} | {base} | {head} | {change} | {budget} | {icon} **{status}** |".format(
                name=name,
                base=_format_rate(_get(metric, "base_geometric_mean")),
                head=_format_rate(_get(metric, "head_geometric_mean")),
                change=_change_cell(metric_verdict),
                budget=_format_pct(-budget, signed=False, decimals=0),
                icon=_status_icon(status),
                status=status,
            )
        )
    return lines


def _markdown_pair_data(analysis: Any | None) -> list[str]:
    if analysis is None:
        return ["No raw pair analysis available."]
    lines: list[str] = []
    for metric in _metric_analyses(analysis):
        name = str(_get(metric, "metric_name", "metric"))
        base_values = list(_get(metric, "base_values", ()) or ())
        head_values = list(_get(metric, "head_values", ()) or ())
        lines.extend(
            [
                f"#### `{name}` pairs",
                "",
                "| Pair | Base (t/s) | Head (t/s) | Head vs. base |",
                "|--:|--:|--:|--:|",
            ]
        )
        for index, (base, head) in enumerate(zip(base_values, head_values), start=1):
            change = ((float(head) / float(base)) - 1.0) * 100 if float(base) else float("nan")
            lines.append(
                f"| {index} | {float(base):.3f} | {float(head):.3f} | {_format_pct(change)} |"
            )
        lines.append("")
    return lines


def generate_markdown_report(
    session: ComparisonSession,
    analysis: AnalysisResult | None,
    verdict: SessionVerdict,
) -> str:
    """Render a GitHub-flavored Markdown report.

    Args:
        session: Completed or aborted comparison session.
        analysis: Statistical analysis, or ``None`` when unavailable.
        verdict: Tri-state session verdict.

    Returns:
        Markdown document ending with a newline.
    """

    lines = [
        f"## {_status_title(verdict)}",
        "",
        f"> {_summary(verdict)}",
        "",
        *_markdown_metric_table(analysis, verdict),
        "",
        "### Evidence",
        "",
    ]
    if analysis is None:
        lines.append("Comparison ended before enough measurements were collected for analysis.")
    else:
        pairs = int(_get(analysis, "n_valid_pairs", 0) or 0)
        blocks = len(_get(_metric_analyses(analysis)[0], "block_averages", ())) if _metric_analyses(analysis) else 0
        lines.append(
            f"{pairs} valid pairs across {blocks} balanced ABBA/BAAB blocks; "
            "stratified bootstrap and block-t intervals were evaluated."
        )
        lines.append("")
        lines.append("| Metric | Log-ratio SD | Base CV | Head CV | Block drift |")
        lines.append("|:--|--:|--:|--:|--:|")
        for metric in _metric_analyses(analysis):
            lines.append(
                "| {name} | {sd} | {base_cv} | {head_cv} | {drift} |".format(
                    name=_get(metric, "metric_name", "metric"),
                    sd=_format_pct(float(_get(metric, "log_ratio_sd", 0.0) or 0.0) * 100, signed=False),
                    base_cv=_format_pct(float(_get(metric, "base_cv", 0.0) or 0.0) * 100, signed=False),
                    head_cv=_format_pct(float(_get(metric, "head_cv", 0.0) or 0.0) * 100, signed=False),
                    drift=_format_pct(float(_get(metric, "block_drift", 0.0) or 0.0) * 100),
                )
            )

    warnings = list(_get(verdict, "warnings", ()) or ())
    if warnings:
        lines.extend(["", "### Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    model_path = _path_from_session(session, "model_path", "model")
    model_sha = str(_get(session, "model_sha", ""))
    thermal_text, memory_text, power_text = _environment_summary(session)
    lines.extend(
        [
            "",
            "<details>",
            "<summary>Raw pair data and environment details</summary>",
            "",
            *_markdown_pair_data(analysis),
            "#### Environment",
            "",
            f"- Hardware: {_hardware_summary(session)}",
            f"- {_backend_summary(session)}",
            f"- Builds: {_build_summary(session)}",
            f"- Model: `{Path(model_path).name or 'unknown'}` (`{model_sha or 'unknown'}`)",
            f"- {thermal_text}",
            f"- {memory_text}",
            f"- {power_text}",
            f"- Contaminated blocks: {len(_get(session, 'contaminated_blocks', ()) or ())}",
            f"- Duration: {_duration_text(session)}",
            "",
            "</details>",
            "",
            f"_Protocol version {PROTOCOL_VERSION}_",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    output_dir: Path,
    session: ComparisonSession,
    analysis: AnalysisResult | None,
    verdict: SessionVerdict,
    *,
    fmt: str = "all",
    console: Console | None = None,
) -> dict[str, Path]:
    """Write selected report formats and optionally print terminal output.

    Args:
        output_dir: Destination directory for file reports.
        session: Completed or aborted comparison session.
        analysis: Statistical analysis, or ``None`` when unavailable.
        verdict: Tri-state session verdict.
        fmt: ``terminal``, ``markdown``, ``json``, or ``all``.
        console: Optional Rich console for terminal output.

    Returns:
        Mapping of written format names to paths.

    Raises:
        ValueError: If ``fmt`` is not supported.
    """

    if fmt not in {"terminal", "markdown", "json", "all"}:
        raise ValueError(f"Unsupported report format: {fmt}")
    destination = Path(output_dir).expanduser()
    written: dict[str, Path] = {}
    destination.mkdir(parents=True, exist_ok=True)
    schedule_path = destination / "schedule.json"
    schedule_payload = {
        "seed": str(_get(session, "schedule_seed", _get(session, "seed", "")) or ""),
        "schedule": to_serializable(_get(session, "schedule", ())),
    }
    schedule_path.write_text(
        json.dumps(schedule_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written["schedule"] = schedule_path
    if fmt in {"json", "all"}:
        json_path = destination / "result.json"
        json_path.write_text(generate_json_report(session, analysis, verdict), encoding="utf-8")
        written["json"] = json_path
    if fmt in {"markdown", "all"}:
        markdown_path = destination / "report.md"
        markdown_path.write_text(
            generate_markdown_report(session, analysis, verdict), encoding="utf-8"
        )
        written["markdown"] = markdown_path
    if fmt in {"terminal", "all"}:
        render_terminal_report(
            session,
            analysis,
            verdict,
            console=console,
            output_path=written.get("json"),
        )
    return written
