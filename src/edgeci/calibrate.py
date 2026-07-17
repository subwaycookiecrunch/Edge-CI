"""A/A calibration history and runner-enrollment management."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .report import PROTOCOL_VERSION, to_serializable

if TYPE_CHECKING:
    from .config import EdgeCIConfig
    from .orchestrator import ComparisonSession
    from .probe import HardwareFingerprint
    from .stats import AnalysisResult
    from .verdict import SessionVerdict


@dataclass(frozen=True)
class EnrollmentStatus:
    """Current enrollment state for an EdgeCI runner."""

    enrolled: bool
    reason: str
    enrolled_at: datetime | None
    expires_at: datetime | None
    clean_sessions: int
    distinct_days: int
    false_fail_rate: float


@dataclass(frozen=True)
class CalibrationResult:
    """Result and persisted state produced by one A/A calibration."""

    session: ComparisonSession
    analysis: AnalysisResult | None
    verdict: SessionVerdict
    history_path: Path
    enrollment: EnrollmentStatus
    false_fail_rate: float
    clean: bool
    clean_reasons: tuple[str, ...]


class CalibrationStateError(RuntimeError):
    """Raised when persisted calibration state cannot be read or written."""


def _storage_root(storage_dir: Path | None = None) -> Path:
    return Path(storage_dir).expanduser() if storage_dir is not None else Path.home() / ".edgeci"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _hardware_key(hardware: HardwareFingerprint | dict[str, Any] | Any) -> str:
    payload = to_serializable(hardware)
    if not isinstance(payload, dict):
        payload = {"fingerprint": payload}
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"macos_version", "os_build"}
    }
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _os_build(hardware: Any) -> str:
    if isinstance(hardware, dict):
        return str(hardware.get("os_build", ""))
    return str(getattr(hardware, "os_build", ""))


def _calibration_context_key(model_sha: str, config: Any) -> str:
    """Hash the model and measurement protocol covered by enrollment."""

    serialized = to_serializable(config)
    if isinstance(serialized, dict):
        protocol_config = {
            name: serialized.get(name)
            for name in ("benchmark", "budgets", "preflight")
        }
    else:
        protocol_config = serialized
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_sha": str(model_sha),
        "config": protocol_config,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metric_summary(metric: Any) -> dict[str, Any]:
    bootstrap = tuple(getattr(metric, "bootstrap_interval", (0.0, 0.0, 0.0)))
    block_t = tuple(getattr(metric, "block_t_interval", (0.0, 0.0, 0.0)))
    theta = float(getattr(metric, "point_estimate", 0.0))
    return {
        "point_estimate": theta,
        "null_effect_pct": math.expm1(-theta) * 100.0,
        "bootstrap_interval": bootstrap,
        "block_t_interval": block_t,
        "bootstrap_width": float(bootstrap[2] - bootstrap[1]) if len(bootstrap) >= 3 else None,
        "block_t_width": float(block_t[2] - block_t[1]) if len(block_t) >= 3 else None,
        "base_cv": float(getattr(metric, "base_cv", 0.0)),
        "head_cv": float(getattr(metric, "head_cv", 0.0)),
        "log_ratio_sd": float(getattr(metric, "log_ratio_sd", 0.0)),
    }


def _arm_build_commits(session: Any, arm: str) -> set[str]:
    """Collect non-empty emitted build commits for one calibration arm."""

    commits: set[str] = set()
    results = (
        *(getattr(session, "warmup_results", ()) or ()),
        *(getattr(session, "measured_results", ()) or ()),
    )
    for result in results:
        if getattr(result, "arm", None) != arm:
            continue
        for sample in getattr(result, "bench_samples", ()) or ():
            commit = str(getattr(sample, "build_commit", "")).strip()
            if commit:
                commits.add(commit)
    return commits


def _calibration_record(
    session: ComparisonSession,
    analysis: AnalysisResult | None,
    verdict: SessionVerdict,
    mode: str,
) -> dict[str, Any]:
    hardware = getattr(session, "hardware", None)
    start_time = getattr(session, "start_time", None)
    timestamp = start_time if isinstance(start_time, datetime) else _utc_now()
    metrics: dict[str, dict[str, Any]] = {}
    analyzed_metrics: list[Any] = []
    if analysis is not None:
        for metric in (getattr(analysis, "tg", None), getattr(analysis, "pp", None)):
            if metric is not None:
                analyzed_metrics.append(metric)
                metrics[str(getattr(metric, "metric_name", "metric"))] = _metric_summary(metric)
    overall = str(getattr(verdict, "overall", "INCONCLUSIVE"))
    warnings = list(getattr(verdict, "warnings", ()) or ())
    config = getattr(session, "config", None)
    model_sha = str(getattr(session, "model_sha", ""))
    budgets = getattr(config, "budgets", None)
    clean_reasons: list[str] = []
    if overall != "PASS":
        clean_reasons.append(f"protocol verdict was {overall}")
    if getattr(session, "abort_reason", None) is not None:
        clean_reasons.append("session was aborted")
    if warnings:
        clean_reasons.append("diagnostic warnings were present")
    if len(analyzed_metrics) != 2:
        clean_reasons.append("both metrics were not analyzed")
    base_sha = str(getattr(session, "base_binary_sha", ""))
    head_sha = str(getattr(session, "head_binary_sha", ""))
    if mode in {"same-path", "copy"} and base_sha != head_sha:
        clean_reasons.append(f"{mode} calibration binaries were not byte-identical")
    if mode == "equivalent-build":
        base_commits = _arm_build_commits(session, "base")
        head_commits = _arm_build_commits(session, "head")
        if (
            len(base_commits) != 1
            or len(head_commits) != 1
            or base_commits != head_commits
        ):
            clean_reasons.append(
                "equivalent-build binaries did not report one shared source commit"
            )
    for metric in analyzed_metrics:
        metric_name = str(getattr(metric, "metric_name", "metric"))
        budget_name = "tg" if metric_name.startswith("tg") else "pp"
        budget = float(getattr(budgets, budget_name, 0.05))
        summary = metrics[metric_name]
        if abs(float(summary["null_effect_pct"])) > budget * 100.0:
            clean_reasons.append(
                f"{metric_name} null effect exceeded {budget:.1%}"
            )
        for interval_name in ("bootstrap_interval", "block_t_interval"):
            interval = tuple(summary[interval_name])
            if len(interval) < 3 or not float(interval[1]) <= 0.0 <= float(interval[2]):
                clean_reasons.append(
                    f"{metric_name} {interval_name} excluded zero"
                )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": timestamp.isoformat(),
        "session_date": timestamp.date().isoformat(),
        "mode": mode,
        "hardware": to_serializable(hardware),
        "hardware_key": _hardware_key(hardware),
        "os_build": _os_build(hardware),
        "model_sha": model_sha,
        "context_key": _calibration_context_key(model_sha, config),
        "binaries": {
            "base_sha256": str(getattr(session, "base_binary_sha", "")),
            "head_sha256": str(getattr(session, "head_binary_sha", "")),
        },
        "config": to_serializable(config),
        "overall": overall,
        "clean": not clean_reasons,
        "clean_reasons": clean_reasons,
        "abort_reason": getattr(session, "abort_reason", None),
        "metrics": metrics,
        "warnings": to_serializable(warnings),
    }


def load_calibration_history(storage_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load calibration summaries in chronological order.

    Args:
        storage_dir: Optional EdgeCI state root, primarily for tests.

    Returns:
        Parsed calibration records ordered by timestamp.

    Raises:
        CalibrationStateError: If a history file is malformed.
    """

    history_dir = _storage_root(storage_dir) / "calibrations"
    if not history_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(history_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationStateError(f"Cannot read calibration history {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise CalibrationStateError(f"Calibration history {path} is not a JSON object")
        records.append(value)
    records.sort(key=lambda record: str(record.get("timestamp", "")))
    return records


def calibration_false_fail_rate(records: list[dict[str, Any]]) -> float:
    """Calculate the observed A/A false-fail rate.

    Args:
        records: Calibration history records.

    Returns:
        Fraction of completed A/A sessions whose overall verdict was ``FAIL``.
    """

    completed = [
        record
        for record in records
        if record.get("overall") in {"PASS", "FAIL", "INCONCLUSIVE"}
        and not record.get("abort_reason")
    ]
    if not completed:
        return 0.0
    return sum(record.get("overall") == "FAIL" for record in completed) / len(completed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(to_serializable(payload), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as exc:
        raise CalibrationStateError(f"Cannot write calibration state {path}: {exc}") from exc


def _read_enrollment(storage_dir: Path | None = None) -> dict[str, Any] | None:
    path = _storage_root(storage_dir) / "enrollment.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationStateError(f"Cannot read enrollment state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CalibrationStateError(f"Enrollment state {path} is not a JSON object")
    return value


def get_enrollment_status(
    hardware: HardwareFingerprint | None = None,
    *,
    model_sha: str | None = None,
    config: Any | None = None,
    storage_dir: Path | None = None,
    now: datetime | None = None,
) -> EnrollmentStatus:
    """Return enrollment state, applying expiry and machine-change checks.

    Args:
        hardware: Current hardware fingerprint. When provided, hardware and OS
            changes invalidate persisted enrollment.
        model_sha: Current model digest for workload-scoped validation.
        config: Current EdgeCI protocol configuration.
        storage_dir: Optional EdgeCI state root.
        now: Optional current UTC timestamp for deterministic tests.

    Returns:
        Validated enrollment status.
    """

    current = (now or _utc_now()).astimezone(timezone.utc)
    state = _read_enrollment(storage_dir)
    records = load_calibration_history(storage_dir)
    false_fail_rate = calibration_false_fail_rate(records)
    if state is None:
        return EnrollmentStatus(False, "Five clean sessions across two days required", None, None, 0, 0, false_fail_rate)

    enrolled_at = _parse_datetime(state.get("enrolled_at"))
    expires_at = _parse_datetime(state.get("expires_at"))
    clean_sessions = int(state.get("clean_sessions", 0) or 0)
    distinct_days = int(state.get("distinct_days", 0) or 0)
    if not state.get("enrolled", False):
        return EnrollmentStatus(
            False,
            str(state.get("reason", "Five clean sessions across two days required")),
            enrolled_at,
            expires_at,
            clean_sessions,
            distinct_days,
            false_fail_rate,
        )
    if expires_at is None or current >= expires_at:
        return EnrollmentStatus(False, "Enrollment expired", enrolled_at, expires_at, clean_sessions, distinct_days, false_fail_rate)
    if hardware is not None:
        if state.get("hardware_key") != _hardware_key(hardware):
            return EnrollmentStatus(False, "Hardware changed", enrolled_at, expires_at, clean_sessions, distinct_days, false_fail_rate)
        if state.get("os_build") != _os_build(hardware):
            return EnrollmentStatus(False, "OS build changed", enrolled_at, expires_at, clean_sessions, distinct_days, false_fail_rate)
    if model_sha is not None and config is not None:
        expected_context = _calibration_context_key(model_sha, config)
        if state.get("context_key") != expected_context:
            return EnrollmentStatus(
                False,
                "Model or protocol configuration changed",
                enrolled_at,
                expires_at,
                clean_sessions,
                distinct_days,
                false_fail_rate,
            )
    return EnrollmentStatus(True, "Runner enrolled", enrolled_at, expires_at, clean_sessions, distinct_days, false_fail_rate)


def update_enrollment(
    hardware: HardwareFingerprint,
    *,
    model_sha: str | None = None,
    config: Any | None = None,
    storage_dir: Path | None = None,
    now: datetime | None = None,
) -> EnrollmentStatus:
    """Recompute and persist enrollment from matching calibration history.

    Args:
        hardware: Current runner hardware fingerprint.
        model_sha: Model digest calibrated by the qualifying sessions.
        config: Protocol configuration calibrated by the qualifying sessions.
        storage_dir: Optional EdgeCI state root.
        now: Optional current UTC timestamp for deterministic tests.

    Returns:
        Newly evaluated enrollment state.
    """

    current = (now or _utc_now()).astimezone(timezone.utc)
    records = load_calibration_history(storage_dir)
    hardware_key = _hardware_key(hardware)
    os_build = _os_build(hardware)
    context_key = (
        _calibration_context_key(model_sha, config)
        if model_sha is not None and config is not None
        else None
    )
    cutoff = current - timedelta(days=14)
    matching_clean: list[dict[str, Any]] = []
    for record in records:
        timestamp = _parse_datetime(record.get("timestamp"))
        if (
            record.get("clean") is True
            and record.get("hardware_key") == hardware_key
            and record.get("os_build") == os_build
            and (
                context_key is None
                or record.get("context_key") == context_key
            )
            and timestamp is not None
            and cutoff < timestamp <= current + timedelta(minutes=5)
        ):
            matching_clean.append(record)
    days = {str(record.get("session_date", "")) for record in matching_clean if record.get("session_date")}
    qualifying = len(matching_clean) >= 5 and len(days) >= 2
    clean_timestamps = [
        timestamp
        for record in matching_clean
        if (timestamp := _parse_datetime(record.get("timestamp"))) is not None
    ]
    latest_clean = max(clean_timestamps) if clean_timestamps else None
    expires_at = (
        latest_clean + timedelta(days=14)
        if qualifying and latest_clean is not None
        else None
    )
    enrolled = bool(expires_at is not None and current < expires_at)
    previous = _read_enrollment(storage_dir)
    previous_enrolled_at = _parse_datetime(previous.get("enrolled_at")) if previous else None
    same_runner = bool(
        previous
        and previous.get("hardware_key") == hardware_key
        and previous.get("os_build") == os_build
        and previous.get("context_key") == context_key
    )
    enrolled_at = previous_enrolled_at if enrolled and same_runner and previous_enrolled_at else (current if enrolled else None)
    reason = (
        "Runner enrolled"
        if enrolled
        else (
            "Calibration enrollment expired"
            if qualifying
            else "Five clean sessions across two days required"
        )
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "enrolled": enrolled,
        "reason": reason,
        "enrolled_at": enrolled_at,
        "expires_at": expires_at,
        "hardware_key": hardware_key,
        "os_build": os_build,
        "model_sha": model_sha,
        "context_key": context_key,
        "clean_sessions": len(matching_clean),
        "distinct_days": len(days),
        "false_fail_rate": calibration_false_fail_rate(records),
    }
    _write_json(_storage_root(storage_dir) / "enrollment.json", payload)
    return get_enrollment_status(
        hardware,
        model_sha=model_sha,
        config=config,
        storage_dir=storage_dir,
        now=current,
    )


def _save_calibration_record(
    record: dict[str, Any], storage_dir: Path | None = None
) -> Path:
    timestamp = _parse_datetime(record.get("timestamp")) or _utc_now()
    filename = timestamp.strftime("%Y%m%dT%H%M%S%fZ.json")
    path = _storage_root(storage_dir) / "calibrations" / filename
    _write_json(path, record)
    return path


def run_calibration(
    binary_path: Path,
    model_path: Path,
    config: EdgeCIConfig,
    mode: str = "same-path",
    *,
    equivalent_binary_path: Path | None = None,
    storage_dir: Path | None = None,
) -> CalibrationResult:
    """Run and persist a full A/A calibration comparison.

    Args:
        binary_path: Primary llama-bench binary.
        model_path: GGUF model used by both arms.
        config: Explicit EdgeCI protocol configuration.
        mode: ``same-path``, ``copy``, or ``equivalent-build``.
        equivalent_binary_path: Second independently built binary required by
            ``equivalent-build`` mode.
        storage_dir: Optional EdgeCI state root, primarily for tests.

    Returns:
        Calibration result with updated runner enrollment.

    Raises:
        ValueError: If mode or equivalent-build arguments are invalid.
    """

    from .orchestrator import run_comparison
    from .stats import analyze_session
    from .verdict import SessionVerdict, determine_verdict

    primary = Path(binary_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if mode not in {"same-path", "copy", "equivalent-build"}:
        raise ValueError(f"Unsupported calibration mode: {mode}")
    if mode == "equivalent-build" and equivalent_binary_path is None:
        raise ValueError("equivalent-build mode requires equivalent_binary_path")
    if mode != "equivalent-build" and equivalent_binary_path is not None:
        raise ValueError(
            "equivalent_binary_path is only valid in equivalent-build mode"
        )

    def evaluate(base: Path, head: Path) -> tuple[Any, Any | None, Any]:
        session = run_comparison(base, head, model, config, seed="")
        if getattr(session, "abort_reason", None):
            verdict = SessionVerdict(
                overall="INCONCLUSIVE",
                metrics=[],
                experimental=True,
                abort_reason=getattr(session, "abort_reason"),
                warnings=[],
            )
            return session, None, verdict
        analysis = analyze_session(session)
        verdict = determine_verdict(
            analysis,
            config.budgets,
            abort_reason=getattr(session, "abort_reason", None),
            experimental=True,
        )
        return session, analysis, verdict

    if mode == "copy":
        with tempfile.TemporaryDirectory(prefix="edgeci-calibration-") as temporary:
            copied = Path(temporary) / primary.name
            shutil.copy2(primary, copied)
            session, analysis, verdict = evaluate(primary, copied)
    elif mode == "equivalent-build":
        if equivalent_binary_path is None:
            raise ValueError("equivalent-build mode requires equivalent_binary_path")
        equivalent = Path(equivalent_binary_path).expanduser().resolve()
        session, analysis, verdict = evaluate(primary, equivalent)
    else:
        session, analysis, verdict = evaluate(primary, primary)

    record = _calibration_record(session, analysis, verdict, mode)
    history_path = _save_calibration_record(record, storage_dir)
    history = load_calibration_history(storage_dir)
    false_fail_rate = calibration_false_fail_rate(history)
    enrollment = update_enrollment(
        getattr(session, "hardware"),
        model_sha=str(getattr(session, "model_sha", "")),
        config=getattr(session, "config"),
        storage_dir=storage_dir,
    )
    return CalibrationResult(
        session=session,
        analysis=analysis,
        verdict=verdict,
        history_path=history_path,
        enrollment=enrollment,
        false_fail_rate=false_fail_rate,
        clean=bool(record["clean"]),
        clean_reasons=tuple(str(reason) for reason in record["clean_reasons"]),
    )
