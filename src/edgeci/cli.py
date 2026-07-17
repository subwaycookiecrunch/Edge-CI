"""Command-line interface for EdgeCI."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from . import __version__
from .calibrate import (
    calibration_false_fail_rate,
    get_enrollment_status,
    load_calibration_history,
    run_calibration,
)
from .config import EdgeCIConfig, load_config
from .doctor import DoctorReport, run_doctor
from .lock import EdgeCILock
from .orchestrator import ComparisonSession, ProgressUpdate, run_comparison
from .probe import get_hardware_fingerprint
from .report import render_terminal_report, write_reports
from .stats import AnalysisResult, analyze_session
from .verdict import SessionVerdict, determine_verdict


def _console() -> Console:
    return Console()


def _load_cli_config(
    config_path: Path | None,
    *,
    model: Path | None = None,
    output: Path | None = None,
    fmt: str | None = None,
) -> EdgeCIConfig:
    overrides: dict[str, Any] = {}
    if model is not None:
        overrides["model"] = model
    if output is not None:
        overrides["output"] = output
    if fmt is not None:
        overrides["format"] = fmt
    return load_config(config_path, overrides=overrides)


def _render_doctor(report: DoctorReport, console: Console) -> None:
    table = Table(title="EdgeCI doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        status = "✅ PASS" if check.passed else "❌ FAIL"
        style = "green" if check.passed else "red"
        table.add_row(check.name, f"[{style}]{status}[/{style}]", check.detail)
    console.print(table)
    if report.ready:
        console.print(Panel("All required checks passed.", title="READY", border_style="green"))
    else:
        failed = ", ".join(check.name for check in report.checks if check.required and not check.passed)
        console.print(
            Panel(
                f"Required checks failed: {failed}",
                title="NOT READY",
                border_style="red",
            )
        )


def _aborted_verdict(session: ComparisonSession, *, experimental: bool) -> SessionVerdict:
    reason = session.abort_reason or "INCONCLUSIVE_RESOURCE: comparison incomplete"
    return SessionVerdict(
        overall="INCONCLUSIVE",
        metrics=[],
        experimental=experimental,
        abort_reason=reason,
        warnings=[],
    )


def _exit_for_verdict(
    verdict: SessionVerdict,
    *,
    shadow_if_experimental: bool,
) -> None:
    if shadow_if_experimental and verdict.experimental:
        return
    code = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}.get(verdict.overall, 2)
    if code:
        raise click.exceptions.Exit(code)


def _comparison_error(message: str) -> NoReturn:
    """Render an operational compare error and use the inconclusive exit code."""

    click.echo(f"Error: {message}", err=True)
    raise click.exceptions.Exit(2)


def _comparison_summary(config: EdgeCIConfig, base: Path, head: Path, model: Path) -> str:
    benchmark = config.benchmark
    return (
        f"Base: {base}\n"
        f"Head: {head}\n"
        f"Model: {model}\n"
        f"Protocol: {benchmark.pairs} measured pairs + "
        f"{benchmark.warmup_pairs} warm-up pairs, "
        f"pp{benchmark.prompt_tokens}/tg{benchmark.generate_tokens}, "
        f"{benchmark.gap_seconds:g}s gap"
    )


@click.group()
@click.version_option(version=__version__, prog_name="edgeci")
def main() -> None:
    """EdgeCI — Performance release gate for on-device AI inference."""


@main.command()
@click.option(
    "--model",
    "-m",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to GGUF model.",
)
@click.option(
    "--config",
    "config_path",
    "-c",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to EdgeCI TOML configuration.",
)
def doctor(model: Path | None, config_path: Path | None) -> None:
    """Check if this machine is ready for benchmarking."""

    console = _console()
    try:
        config = _load_cli_config(config_path, model=model)
        with console.status("Sampling testbed health…", spinner="dots"):
            report = run_doctor(config)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    _render_doctor(report, console)
    if not report.ready:
        raise click.exceptions.Exit(1)


@main.command()
@click.option(
    "--base",
    "-b",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Base llama-bench binary.",
)
@click.option(
    "--head",
    "-H",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Head llama-bench binary.",
)
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to GGUF model.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Report directory (default: config or ./edgeci-results).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["terminal", "markdown", "json", "all"]),
    default=None,
    help="Report format (default: config or all).",
)
@click.option(
    "--config",
    "config_path",
    "-c",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to EdgeCI TOML configuration.",
)
@click.option("--seed", default="", help="Deterministic schedule seed.")
def compare(
    base: Path,
    head: Path,
    model: Path,
    output: Path | None,
    fmt: str | None,
    config_path: Path | None,
    seed: str,
) -> None:
    """Run a paired base/head performance comparison."""

    console = _console()
    try:
        config = _load_cli_config(config_path, model=model, output=output, fmt=fmt)
        with console.status("Running readiness checks…", spinner="dots"):
            readiness = run_doctor(config, binary_path=base)
    except Exception as exc:
        _comparison_error(str(exc))
    if not readiness.ready:
        _render_doctor(readiness, console)
        raise click.exceptions.Exit(2)

    console.print(
        Panel(
            _comparison_summary(config, base, head, model),
            title="Starting EdgeCI comparison",
            border_style="cyan",
        )
    )
    total = (config.benchmark.pairs + config.benchmark.warmup_pairs) * 2
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task("Preparing testbed", total=total)

    def on_progress(update: ProgressUpdate) -> None:
        description = update.message
        completed = min(total, max(0, update.completed))
        progress.update(task_id, completed=completed, description=description)

    try:
        with progress:
            session = run_comparison(
                base,
                head,
                model,
                config,
                seed=seed,
                on_progress=on_progress,
            )
            progress.update(
                task_id,
                completed=min(
                    total,
                    len(session.warmup_results) + len(session.measured_results),
                ),
            )
    except Exception as exc:
        _comparison_error(f"comparison could not start: {exc}")

    enrollment_warning: str | None = None
    try:
        enrollment = get_enrollment_status(
            session.hardware,
            model_sha=session.model_sha,
            config=session.config,
        )
        experimental = not enrollment.enrolled
    except Exception as exc:
        experimental = True
        enrollment_warning = f"calibration state unavailable: {exc}"
    analysis: AnalysisResult | None
    if session.complete:
        try:
            analysis = analyze_session(session)
            verdict = determine_verdict(
                analysis,
                config.budgets,
                experimental=experimental,
            )
        except Exception as exc:
            session = replace(
                session,
                abort_reason=f"ERROR_ANALYSIS: {type(exc).__name__}: {exc}",
            )
            analysis = None
            verdict = _aborted_verdict(session, experimental=experimental)
    else:
        analysis = None
        verdict = _aborted_verdict(session, experimental=experimental)
    if enrollment_warning is not None:
        verdict = replace(
            verdict,
            warnings=[*verdict.warnings, enrollment_warning],
        )

    output_dir = config.report.output_dir
    report_format = config.report.format
    try:
        written = write_reports(
            output_dir,
            session,
            analysis,
            verdict,
            fmt=report_format,
            console=console,
        )
    except OSError as exc:
        _comparison_error(f"cannot write report: {exc}")
    if report_format not in {"terminal", "all"}:
        console.print(
            "Wrote " + ", ".join(str(path) for path in written.values()),
            markup=False,
        )
    _exit_for_verdict(verdict, shadow_if_experimental=True)


@main.command()
@click.option(
    "--binary",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Primary llama-bench binary.",
)
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to GGUF model.",
)
@click.option(
    "--mode",
    type=click.Choice(["same-path", "copy", "equivalent-build"]),
    default="same-path",
    show_default=True,
)
@click.option(
    "--equivalent-binary",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Second same-source build for equivalent-build mode.",
)
@click.option(
    "--config",
    "config_path",
    "-c",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to EdgeCI TOML configuration.",
)
def calibrate(
    binary: Path,
    model: Path,
    mode: str,
    equivalent_binary: Path | None,
    config_path: Path | None,
) -> None:
    """Run A/A calibration to validate testbed reliability."""

    if mode == "equivalent-build" and equivalent_binary is None:
        raise click.UsageError("--equivalent-binary is required for equivalent-build mode")
    if mode != "equivalent-build" and equivalent_binary is not None:
        raise click.UsageError("--equivalent-binary is only valid with equivalent-build mode")
    console = _console()
    try:
        config = _load_cli_config(config_path, model=model)
        with console.status("Running A/A calibration…", spinner="dots"):
            result = run_calibration(
                binary,
                model,
                config,
                mode=mode,
                equivalent_binary_path=equivalent_binary,
            )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    render_terminal_report(
        result.session,
        result.analysis,
        result.verdict,
        console=console,
        output_path=result.history_path,
    )
    if not result.clean:
        console.print(
            Panel(
                "\n".join(result.clean_reasons),
                title="CALIBRATION NOT CLEAN",
                border_style="yellow",
            )
        )
    enrollment = result.enrollment
    if enrollment.enrolled:
        title = "RUNNER ENROLLED"
        body = (
            f"{enrollment.clean_sessions} clean sessions across "
            f"{enrollment.distinct_days} days. Expires {enrollment.expires_at}."
        )
        style = "green"
    else:
        title = "RUNNER NOT ENROLLED"
        body = (
            f"{enrollment.reason}. Progress: {enrollment.clean_sessions}/5 clean sessions "
            f"across {enrollment.distinct_days}/2 days."
        )
        style = "yellow"
    console.print(Panel(body, title=title, border_style=style))
    console.print(f"Observed A/A false-fail rate: {result.false_fail_rate:.1%}")
    if result.verdict.overall == "PASS" and not result.clean:
        raise click.exceptions.Exit(2)
    _exit_for_verdict(result.verdict, shadow_if_experimental=False)


@main.command()
def status() -> None:
    """Show testbed enrollment and calibration history."""

    console = _console()
    try:
        with EdgeCILock():
            hardware = get_hardware_fingerprint()
            enrollment = get_enrollment_status(hardware)
            history = load_calibration_history()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    style = "green" if enrollment.enrolled else "yellow"
    state = "ENROLLED" if enrollment.enrolled else "NOT ENROLLED"
    expiry = enrollment.expires_at.isoformat() if enrollment.expires_at else "—"
    console.print(
        Panel(
            f"{hardware.summary}\n"
            f"State: {state}\n"
            f"Reason: {enrollment.reason}\n"
            f"Clean sessions: {enrollment.clean_sessions}/5 across "
            f"{enrollment.distinct_days}/2 days\n"
            f"Expiry: {expiry}\n"
            f"False-fail rate: {calibration_false_fail_rate(history):.1%}",
            title="EdgeCI runner status",
            border_style=style,
        )
    )
    if not history:
        console.print("No calibration sessions recorded.")
        return

    table = Table(title="Calibration history", show_header=True, header_style="bold")
    table.add_column("Date")
    table.add_column("Mode")
    table.add_column("tg effect", justify="right")
    table.add_column("pp effect", justify="right")
    table.add_column("Verdict")
    for record in history[-10:]:
        metrics = record.get("metrics", {})
        tg = next(
            (value for key, value in metrics.items() if str(key).startswith("tg")),
            {},
        )
        pp = next(
            (value for key, value in metrics.items() if str(key).startswith("pp")),
            {},
        )
        overall = str(record.get("overall", "INCONCLUSIVE"))
        table.add_row(
            str(record.get("session_date", "—")),
            str(record.get("mode", "—")),
            _history_effect(tg),
            _history_effect(pp),
            f"{_verdict_icon(overall)} {overall}",
        )
    console.print(table)


def _history_effect(metric: Any) -> str:
    if not isinstance(metric, dict) or metric.get("null_effect_pct") is None:
        return "—"
    value = float(metric["null_effect_pct"])
    return f"{value:+.2f}%"


def _verdict_icon(verdict: str) -> str:
    return {"PASS": "✅", "FAIL": "❌", "INCONCLUSIVE": "⚪"}.get(verdict, "⚪")


if __name__ == "__main__":  # pragma: no cover
    main()
