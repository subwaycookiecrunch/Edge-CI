"""Pure-Python paired statistical analysis for EdgeCI sessions."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .adapter import BenchSample


class AnalysisError(ValueError):
    """Raised when a session cannot support the preregistered paired analysis."""


@dataclass(frozen=True)
class MetricAnalysis:
    """Complete raw data, estimates, intervals, and diagnostics for one metric."""

    metric_name: str
    base_values: list[float]
    head_values: list[float]
    log_ratios: list[float]
    ab_ratios: list[float]
    ba_ratios: list[float]
    block_averages: list[float]
    point_estimate: float
    bootstrap_interval: tuple[float, float, float]
    block_t_interval: tuple[float, float, float]
    base_geometric_mean: float
    head_geometric_mean: float
    base_cv: float
    head_cv: float
    log_ratio_sd: float
    median_change: float
    iqr: tuple[float, float]
    block_drift: float


@dataclass(frozen=True)
class AnalysisResult:
    """Paired analyses for token generation and prompt processing."""

    tg: MetricAnalysis
    pp: MetricAnalysis
    n_valid_pairs: int
    n_contaminated_blocks: int


@dataclass(frozen=True)
class _Pair:
    pair_index: int
    block_index: int
    order: str
    base: Any
    head: Any


def quantile_type7(sorted_data: list[float], p: float) -> float:
    """Return a Hyndman-Fan type-7 quantile from already sorted values.

    Args:
        sorted_data: Non-empty values in ascending order.
        p: Probability in the closed interval ``[0, 1]``.

    Returns:
        Interpolated quantile.
    """

    if not sorted_data:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    h = (len(sorted_data) - 1) * p
    lo = int(h)
    hi = lo + 1
    fraction = h - lo
    if hi >= len(sorted_data):
        return float(sorted_data[-1])
    return float(sorted_data[lo] + fraction * (sorted_data[hi] - sorted_data[lo]))


def _checked_values(values: Iterable[float], name: str) -> list[float]:
    checked: list[float] = []
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name}[{index}] must be numeric")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name}[{index}] must be finite")
        checked.append(converted)
    if not checked:
        raise ValueError(f"{name} must not be empty")
    return checked


def compute_log_ratios(
    base_values: Iterable[float], head_values: Iterable[float]
) -> list[float]:
    """Compute paired throughput effects as ``ln(base / head)``.

    Args:
        base_values: Positive base-arm throughput measurements.
        head_values: Positive head-arm throughput measurements.

    Returns:
        Log-ratios where positive values indicate a head regression.
    """

    base = _checked_values(base_values, "base_values")
    head = _checked_values(head_values, "head_values")
    if len(base) != len(head):
        raise ValueError("base_values and head_values must have equal lengths")
    if any(value <= 0.0 for value in base + head):
        raise ValueError("throughput measurements must be greater than zero")
    return [math.log(base_value / head_value) for base_value, head_value in zip(base, head)]


def stratified_bootstrap(
    ab_ratios: list[float],
    ba_ratios: list[float],
    n_resamples: int = 50_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute the order-stratified bootstrap point and 95% interval.

    Args:
        ab_ratios: Base-first paired log-ratios.
        ba_ratios: Head-first paired log-ratios.
        n_resamples: Number of independent stratified resamples.
        seed: Integer random seed.

    Returns:
        ``(point, lower_95, upper_95)`` on the log-effect scale.
    """

    ab = _checked_values(ab_ratios, "ab_ratios")
    ba = _checked_values(ba_ratios, "ba_ratios")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    point = 0.5 * (statistics.fmean(ab) + statistics.fmean(ba))
    rng = random.Random(seed)
    n_ab = len(ab)
    n_ba = len(ba)
    estimates = [0.0] * n_resamples
    for resample_index in range(n_resamples):
        ab_mean = sum(ab[rng.randrange(n_ab)] for _ in range(n_ab)) / n_ab
        ba_mean = sum(ba[rng.randrange(n_ba)] for _ in range(n_ba)) / n_ba
        estimates[resample_index] = 0.5 * (ab_mean + ba_mean)
    estimates.sort()
    return (
        point,
        quantile_type7(estimates, 0.025),
        quantile_type7(estimates, 0.975),
    )


def block_t_interval(block_averages: list[float]) -> tuple[float, float, float]:
    """Compute the preregistered t interval over balanced block averages.

    Args:
        block_averages: At least two block-level average log-ratios. Default
            protocol supplies ten blocks and uses the preregistered
            ``t(0.975, 9) = 2.262157`` constant.

    Returns:
        ``(point, lower_95, upper_95)`` on the log-effect scale.
    """

    blocks = _checked_values(block_averages, "block_averages")
    if len(blocks) < 2:
        raise ValueError("block-t interval requires at least two blocks")
    mean_block = statistics.fmean(blocks)
    sd_block = math.sqrt(
        sum((value - mean_block) ** 2 for value in blocks) / (len(blocks) - 1)
    )
    t_critical = _t_critical_975(len(blocks) - 1)
    margin = t_critical * sd_block / math.sqrt(len(blocks))
    return mean_block, mean_block - margin, mean_block + margin


def _t_critical_975(degrees_of_freedom: int) -> float:
    """Return two-sided 95% Student-t critical value without SciPy."""

    # Values from the standard t table. Keeping the df=9 constant exact
    # preserves the preregistered 20-pair protocol while supporting explicitly
    # configured even pair counts without silently using the wrong coverage.
    table = (
        0.0,
        12.706205,
        4.302653,
        3.182446,
        2.776445,
        2.570582,
        2.446912,
        2.364624,
        2.306004,
        2.262157,
        2.228139,
        2.200985,
        2.178813,
        2.160369,
        2.144787,
        2.131450,
        2.119905,
        2.109816,
        2.100922,
        2.093024,
        2.085963,
        2.079614,
        2.073873,
        2.068658,
        2.063899,
        2.059539,
        2.055529,
        2.051831,
        2.048407,
        2.045230,
        2.042272,
    )
    if degrees_of_freedom < len(table):
        return table[degrees_of_freedom]

    # Cornish-Fisher expansion around N(0, 1), accurate for df > 30.
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def _sample_for(result: Any, test_type: str) -> BenchSample:
    accessor = getattr(result, "sample_for", None)
    if callable(accessor):
        try:
            sample = accessor(test_type)
        except (KeyError, LookupError, ValueError) as exc:
            raise AnalysisError(f"invocation has no {test_type!r} sample") from exc
        if isinstance(sample, BenchSample):
            return sample
        if hasattr(sample, "test_type") and hasattr(sample, "tokens_per_second"):
            return sample

    samples = getattr(result, "bench_samples", None)
    if samples is None:
        single = getattr(result, "bench_sample", None)
        samples = () if single is None else (single,)
    matching = [sample for sample in samples if getattr(sample, "test_type", None) == test_type]
    if len(matching) != 1:
        raise AnalysisError(
            f"invocation must contain exactly one {test_type!r} sample; found {len(matching)}"
        )
    return matching[0]


def _extract_pairs(session: Any) -> list[_Pair]:
    contaminated = {int(index) for index in getattr(session, "contaminated_blocks", [])}
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    positions: dict[tuple[int, int], dict[str, int]] = {}

    for result in getattr(session, "measured_results", []):
        invocation = getattr(result, "invocation", result)
        if bool(getattr(invocation, "is_warmup", False)):
            continue
        try:
            block_index = int(getattr(invocation, "block_index"))
            pair_index = int(getattr(invocation, "pair_index"))
            position = int(getattr(invocation, "position"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise AnalysisError("measured result lacks valid schedule metadata") from exc
        if block_index in contaminated:
            continue
        arm = getattr(result, "arm", getattr(invocation, "arm", None))
        if arm not in {"base", "head"}:
            raise AnalysisError(f"invalid comparison arm {arm!r}")
        key = (block_index, pair_index)
        bucket = grouped.setdefault(key, {})
        position_bucket = positions.setdefault(key, {})
        if arm in bucket:
            raise AnalysisError(
                f"duplicate {arm} result for block {block_index}, pair {pair_index}"
            )
        bucket[arm] = result
        position_bucket[arm] = position

    if not grouped:
        raise AnalysisError("session contains no valid measured pairs")

    pairs: list[_Pair] = []
    for (block_index, pair_index), arms in sorted(grouped.items()):
        if set(arms) != {"base", "head"}:
            missing = {"base", "head"} - set(arms)
            raise AnalysisError(
                f"incomplete block {block_index}, pair {pair_index}; missing {sorted(missing)}"
            )
        pair_positions = positions[(block_index, pair_index)]
        order = "AB" if pair_positions["base"] < pair_positions["head"] else "BA"
        pairs.append(
            _Pair(
                pair_index=pair_index,
                block_index=block_index,
                order=order,
                base=arms["base"],
                head=arms["head"],
            )
        )
    return pairs


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean_value = statistics.fmean(values)
    if mean_value == 0.0:
        raise AnalysisError("coefficient of variation is undefined for a zero mean")
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / mean_value


def _geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _analyze_pairs(pairs: Sequence[_Pair], test_type: str) -> MetricAnalysis:
    base_samples = [_sample_for(pair.base, test_type) for pair in pairs]
    head_samples = [_sample_for(pair.head, test_type) for pair in pairs]
    sizes = {
        int(getattr(sample, "test_size")) for sample in base_samples + head_samples
    }
    if len(sizes) != 1:
        raise AnalysisError(f"{test_type} samples do not share one workload size: {sizes}")
    test_size = sizes.pop()

    base_values = [float(sample.tokens_per_second) for sample in base_samples]
    head_values = [float(sample.tokens_per_second) for sample in head_samples]
    log_ratios = compute_log_ratios(base_values, head_values)

    block_ratios: dict[int, dict[str, float]] = {}
    block_throughputs: dict[int, list[float]] = {}
    ab_ratios: list[float] = []
    ba_ratios: list[float] = []
    for pair, ratio, base_value, head_value in zip(
        pairs, log_ratios, base_values, head_values, strict=True
    ):
        order_bucket = block_ratios.setdefault(pair.block_index, {})
        if pair.order in order_bucket:
            raise AnalysisError(
                f"block {pair.block_index} contains duplicate {pair.order} pairs"
            )
        order_bucket[pair.order] = ratio
        if pair.order == "AB":
            ab_ratios.append(ratio)
        else:
            ba_ratios.append(ratio)
        block_throughputs.setdefault(pair.block_index, []).extend(
            (base_value, head_value)
        )

    for block_index, order_bucket in block_ratios.items():
        if set(order_bucket) != {"AB", "BA"}:
            raise AnalysisError(
                f"block {block_index} is not order-balanced; found {sorted(order_bucket)}"
            )
    block_averages = [
        statistics.fmean((orders["AB"], orders["BA"]))
        for _, orders in sorted(block_ratios.items())
    ]

    bootstrap = stratified_bootstrap(ab_ratios, ba_ratios)
    block_t = block_t_interval(block_averages)
    changes = sorted(math.exp(-ratio) - 1.0 for ratio in log_ratios)
    median_change = statistics.median(changes)
    iqr = (quantile_type7(changes, 0.25), quantile_type7(changes, 0.75))

    block_performance = [
        _geometric_mean(values) for _, values in sorted(block_throughputs.items())
    ]
    first_half = block_performance[: len(block_performance) // 2]
    second_half = block_performance[len(block_performance) // 2 :]
    if not first_half or not second_half:
        block_drift = 0.0
    else:
        first_median = statistics.median(first_half)
        second_median = statistics.median(second_half)
        block_drift = second_median / first_median - 1.0

    return MetricAnalysis(
        metric_name=f"{test_type}{test_size}",
        base_values=base_values,
        head_values=head_values,
        log_ratios=log_ratios,
        ab_ratios=ab_ratios,
        ba_ratios=ba_ratios,
        block_averages=block_averages,
        point_estimate=bootstrap[0],
        bootstrap_interval=bootstrap,
        block_t_interval=block_t,
        base_geometric_mean=_geometric_mean(base_values),
        head_geometric_mean=_geometric_mean(head_values),
        base_cv=_coefficient_of_variation(base_values),
        head_cv=_coefficient_of_variation(head_values),
        log_ratio_sd=statistics.stdev(log_ratios) if len(log_ratios) > 1 else 0.0,
        median_change=median_change,
        iqr=iqr,
        block_drift=block_drift,
    )


def analyze_session(session: Any) -> AnalysisResult:
    """Run paired tg and pp analyses for a completed comparison session.

    The function depends on the session's public shape instead of importing the
    orchestrator module, avoiding an import cycle.  Whole contaminated blocks are
    excluded before pairing; performance-based outlier removal is never applied.

    Args:
        session: Object exposing ``measured_results`` and ``contaminated_blocks``.

    Returns:
        Complete paired analysis for both benchmark workloads.

    Raises:
        AnalysisError: If results are incomplete, unbalanced, or malformed.
    """

    pairs = _extract_pairs(session)
    tg = _analyze_pairs(pairs, "tg")
    pp = _analyze_pairs(pairs, "pp")
    return AnalysisResult(
        tg=tg,
        pp=pp,
        n_valid_pairs=len(pairs),
        n_contaminated_blocks=len(
            {int(index) for index in getattr(session, "contaminated_blocks", [])}
        ),
    )
