"""Tri-state non-inferiority decisions for EdgeCI analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .stats import AnalysisResult, MetricAnalysis


@dataclass(frozen=True)
class MetricVerdict:
    """Decision and human-scale effect for one benchmark metric."""

    metric_name: str
    verdict: str
    budget: float
    budget_boundary: float
    bootstrap_lower: float
    bootstrap_upper: float
    block_t_lower: float
    block_t_upper: float
    human_change_pct: float
    human_interval_lower_pct: float
    human_interval_upper_pct: float


@dataclass(frozen=True)
class SessionVerdict:
    """Overall EdgeCI decision plus metric evidence and diagnostics."""

    overall: str
    metrics: list[MetricVerdict]
    experimental: bool
    abort_reason: str | None
    warnings: list[str]


def _validated_budget(budget: float) -> float:
    if not isinstance(budget, (int, float)) or isinstance(budget, bool):
        raise TypeError("budget must be numeric")
    value = float(budget)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("budget must be finite and in the interval [0, 1)")
    return value


def evaluate_metric(metric: MetricAnalysis, budget: float) -> MetricVerdict:
    """Apply both preregistered intervals to one throughput budget.

    Args:
        metric: Statistical analysis on the log ``base/head`` scale.
        budget: Allowed fractional throughput loss, such as ``0.05``.

    Returns:
        PASS only when both intervals are below the boundary, FAIL only when
        both are above it, and INCONCLUSIVE otherwise.
    """

    checked_budget = _validated_budget(budget)
    boundary = -math.log1p(-checked_budget)
    _, bootstrap_lower, bootstrap_upper = metric.bootstrap_interval
    _, block_t_lower, block_t_upper = metric.block_t_interval

    bootstrap_fail = bootstrap_lower > boundary
    block_t_fail = block_t_lower > boundary
    bootstrap_pass = bootstrap_upper < boundary
    block_t_pass = block_t_upper < boundary
    if bootstrap_fail and block_t_fail:
        decision = "FAIL"
    elif bootstrap_pass and block_t_pass:
        decision = "PASS"
    else:
        decision = "INCONCLUSIVE"

    envelope_log_lower = min(bootstrap_lower, block_t_lower)
    envelope_log_upper = max(bootstrap_upper, block_t_upper)
    return MetricVerdict(
        metric_name=metric.metric_name,
        verdict=decision,
        budget=checked_budget,
        budget_boundary=boundary,
        bootstrap_lower=bootstrap_lower,
        bootstrap_upper=bootstrap_upper,
        block_t_lower=block_t_lower,
        block_t_upper=block_t_upper,
        human_change_pct=math.expm1(-metric.point_estimate) * 100.0,
        # Throughput conversion reverses endpoints because exp(-x) decreases.
        human_interval_lower_pct=math.expm1(-envelope_log_upper) * 100.0,
        human_interval_upper_pct=math.expm1(-envelope_log_lower) * 100.0,
    )


def _budget_pair(
    budgets: object | None,
    positional_pp_budget: float | None,
    tg_budget: float | None,
    pp_budget: float | None,
) -> tuple[float, float]:
    resolved_tg: Any = None
    resolved_pp: Any = None

    if isinstance(budgets, (int, float)) and not isinstance(budgets, bool):
        resolved_tg = budgets
        resolved_pp = positional_pp_budget
    elif budgets is not None:
        source: Any = getattr(budgets, "budgets", budgets)
        if isinstance(source, Mapping):
            resolved_tg = source.get("tg")
            resolved_pp = source.get("pp")
        else:
            resolved_tg = getattr(source, "tg", None)
            resolved_pp = getattr(source, "pp", None)

    if tg_budget is not None:
        resolved_tg = tg_budget
    if pp_budget is not None:
        resolved_pp = pp_budget
    if resolved_tg is None:
        resolved_tg = 0.05
    if resolved_pp is None:
        resolved_pp = 0.05
    return _validated_budget(resolved_tg), _validated_budget(resolved_pp)


def _interval_direction(interval: tuple[float, float, float]) -> int:
    _, lower, upper = interval
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return 0


def _metric_warnings(metric: MetricAnalysis) -> list[str]:
    warnings: list[str] = []
    if metric.base_cv > 0.05 or metric.head_cv > 0.05:
        warnings.append(
            f"{metric.metric_name}: high variability "
            f"(base CV {metric.base_cv:.1%}, head CV {metric.head_cv:.1%}; limit 5.0%)"
        )
    if abs(metric.block_drift) > 0.05:
        warnings.append(
            f"{metric.metric_name}: block drift {metric.block_drift:+.1%} exceeds 5.0%"
        )

    bootstrap_direction = _interval_direction(metric.bootstrap_interval)
    block_t_direction = _interval_direction(metric.block_t_interval)
    bootstrap_point = metric.bootstrap_interval[0]
    block_t_point = metric.block_t_interval[0]
    opposing_points = bootstrap_point * block_t_point < 0.0
    opposing_intervals = (
        bootstrap_direction != 0
        and block_t_direction != 0
        and bootstrap_direction != block_t_direction
    )
    if opposing_points or opposing_intervals:
        warnings.append(
            f"{metric.metric_name}: bootstrap and block-t disagree on effect direction"
        )
    return warnings


def determine_verdict(
    analysis: AnalysisResult,
    budgets: object | None = None,
    positional_pp_budget: float | None = None,
    *,
    tg_budget: float | None = None,
    pp_budget: float | None = None,
    abort_reason: str | None = None,
    experimental: bool = True,
) -> SessionVerdict:
    """Compose tg/pp metric decisions into a session verdict.

    ``budgets`` may be a budgets dataclass, an EdgeCI config containing one, a
    mapping, or a numeric tg budget followed by a positional pp budget.  Keyword
    overrides take precedence.

    Args:
        analysis: Completed paired session analysis.
        budgets: Budget container, mapping, tg number, or ``None`` for defaults.
        positional_pp_budget: Optional pp budget when ``budgets`` is numeric.
        tg_budget: Explicit token-generation budget override.
        pp_budget: Explicit prompt-processing budget override.
        abort_reason: Resource or execution abort reason, if any.
        experimental: Whether to mark this runner as experimental.

    Returns:
        Overall tri-state verdict with non-verdict-changing warnings.
    """

    resolved_tg, resolved_pp = _budget_pair(
        budgets, positional_pp_budget, tg_budget, pp_budget
    )
    metric_verdicts = [
        evaluate_metric(analysis.tg, resolved_tg),
        evaluate_metric(analysis.pp, resolved_pp),
    ]

    if abort_reason is not None:
        overall = "INCONCLUSIVE"
    elif any(metric.verdict == "FAIL" for metric in metric_verdicts):
        overall = "FAIL"
    elif any(metric.verdict == "INCONCLUSIVE" for metric in metric_verdicts):
        overall = "INCONCLUSIVE"
    else:
        overall = "PASS"

    warnings = _metric_warnings(analysis.tg) + _metric_warnings(analysis.pp)
    return SessionVerdict(
        overall=overall,
        metrics=metric_verdicts,
        experimental=bool(experimental),
        abort_reason=abort_reason,
        warnings=warnings,
    )


# Compatibility names for callers that prefer decision-oriented verbs.
decide_verdict = determine_verdict
build_verdict = determine_verdict

