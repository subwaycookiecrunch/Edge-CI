from __future__ import annotations

from dataclasses import replace

import pytest

from edgeci.stats import AnalysisResult, MetricAnalysis
from edgeci.verdict import determine_verdict, evaluate_metric


def _metric(
    *,
    name: str = "tg128",
    point: float = 0.02,
    bootstrap: tuple[float, float] = (0.01, 0.03),
    block_t: tuple[float, float] = (0.00, 0.04),
    base_cv: float = 0.01,
    head_cv: float = 0.01,
    drift: float = 0.0,
) -> MetricAnalysis:
    return MetricAnalysis(
        metric_name=name,
        base_values=[100.0] * 20,
        head_values=[98.0] * 20,
        log_ratios=[point] * 20,
        ab_ratios=[point] * 10,
        ba_ratios=[point] * 10,
        block_averages=[point] * 10,
        point_estimate=point,
        bootstrap_interval=(point, *bootstrap),
        block_t_interval=(point, *block_t),
        base_geometric_mean=100.0,
        head_geometric_mean=98.0,
        base_cv=base_cv,
        head_cv=head_cv,
        log_ratio_sd=0.0,
        median_change=-0.02,
        iqr=(-0.02, -0.02),
        block_drift=drift,
    )


def _analysis(tg: MetricAnalysis, pp: MetricAnalysis | None = None) -> AnalysisResult:
    return AnalysisResult(
        tg=tg,
        pp=pp or replace(tg, metric_name="pp512"),
        n_valid_pairs=20,
        n_contaminated_blocks=0,
    )


def test_fail_when_both_intervals_exceed_budget() -> None:
    metric = _metric(point=0.10, bootstrap=(0.08, 0.12), block_t=(0.07, 0.13))

    result = evaluate_metric(metric, 0.05)

    assert result.verdict == "FAIL"
    assert result.human_change_pct == pytest.approx(-9.516, abs=0.01)


def test_pass_when_both_intervals_below_budget() -> None:
    metric = _metric(bootstrap=(-0.01, 0.04), block_t=(0.00, 0.05))

    assert evaluate_metric(metric, 0.05).verdict == "PASS"


def test_inconclusive_when_intervals_straddle_budget() -> None:
    metric = _metric(bootstrap=(0.03, 0.08), block_t=(0.02, 0.09))

    assert evaluate_metric(metric, 0.05).verdict == "INCONCLUSIVE"


def test_inconclusive_when_interval_methods_disagree() -> None:
    metric = _metric(bootstrap=(0.08, 0.12), block_t=(0.00, 0.04))

    assert evaluate_metric(metric, 0.05).verdict == "INCONCLUSIVE"


def test_any_metric_failure_makes_overall_failure() -> None:
    failed = _metric(point=0.1, bootstrap=(0.08, 0.12), block_t=(0.07, 0.13))
    passed = _metric(name="pp512", bootstrap=(-0.02, 0.03), block_t=(-0.01, 0.04))

    verdict = determine_verdict(_analysis(failed, passed), {"tg": 0.05, "pp": 0.05})

    assert verdict.overall == "FAIL"


def test_abort_overrides_metric_failure() -> None:
    failed = _metric(point=0.1, bootstrap=(0.08, 0.12), block_t=(0.07, 0.13))

    verdict = determine_verdict(_analysis(failed), abort_reason="thermal recovery timeout")

    assert verdict.overall == "INCONCLUSIVE"
    assert verdict.abort_reason == "thermal recovery timeout"


def test_high_cv_and_block_drift_generate_warnings() -> None:
    noisy = _metric(base_cv=0.06, drift=0.08)

    verdict = determine_verdict(_analysis(noisy))

    assert any("high variability" in warning for warning in verdict.warnings)
    assert any("block drift" in warning for warning in verdict.warnings)

