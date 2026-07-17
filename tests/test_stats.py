from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from edgeci.adapter import BenchSample
from edgeci.schedule import Invocation, generate_schedule
from edgeci.stats import (
    analyze_session,
    block_t_interval,
    compute_log_ratios,
    quantile_type7,
    stratified_bootstrap,
)


@dataclass(frozen=True)
class FakeResult:
    invocation: Invocation
    arm: str
    bench_samples: tuple[BenchSample, ...]

    def sample_for(self, test_type: str) -> BenchSample:
        return next(sample for sample in self.bench_samples if sample.test_type == test_type)


def _sample(test_type: str, value: float) -> BenchSample:
    return BenchSample(
        model_filename="model.gguf",
        model_size=1,
        model_n_params=1,
        n_gpu_layers=-1,
        n_batch=2048,
        n_ubatch=512,
        type_k="f16",
        type_v="f16",
        n_threads=6,
        test_type=test_type,
        test_size=128 if test_type == "tg" else 512,
        tokens_per_second=value,
        build_commit="abc",
        build_number=1,
        backend="Metal",
        device_description="Apple M5",
    )


def _session(*, head_tg: float, head_pp: float = 200.0):
    results = []
    for invocation in generate_schedule(n_warmup_pairs=0, seed="stats"):
        tg = 100.0 if invocation.arm == "base" else head_tg
        pp = 200.0 if invocation.arm == "base" else head_pp
        results.append(
            FakeResult(invocation, invocation.arm, (_sample("pp", pp), _sample("tg", tg)))
        )
    return SimpleNamespace(measured_results=tuple(results), contaminated_blocks=())


def test_compute_log_ratios_known_values() -> None:
    assert compute_log_ratios([100.0, 50.0], [50.0, 100.0]) == pytest.approx(
        [math.log(2.0), math.log(0.5)]
    )


def test_quantile_type7_known_values() -> None:
    values = [0.0, 10.0, 20.0, 30.0, 40.0]

    assert quantile_type7(values, 0.25) == 10.0
    assert quantile_type7(values, 0.5) == 20.0
    assert quantile_type7(values, 0.975) == pytest.approx(39.0)


def test_bootstrap_interval_is_reasonable() -> None:
    ab = [0.01, 0.02, 0.03, 0.04, 0.05]
    ba = [0.00, 0.01, 0.02, 0.03, 0.04]

    point, lower, upper = stratified_bootstrap(ab, ba, n_resamples=5_000)

    assert point == pytest.approx(0.025)
    assert lower < point < upper
    assert (lower, upper) == pytest.approx((0.014, 0.036), abs=0.005)


def test_block_t_interval_known_constant_effect() -> None:
    assert block_t_interval([0.1] * 10) == pytest.approx((0.1, 0.1, 0.1))


def test_block_t_interval_uses_correct_nondefault_degrees_of_freedom() -> None:
    point, lower, upper = block_t_interval([0.0, 1.0])

    assert point == pytest.approx(0.5)
    assert (lower, upper) == pytest.approx((-5.8531025, 6.8531025))


def test_known_ten_percent_regression_is_recovered() -> None:
    analysis = analyze_session(_session(head_tg=90.0))
    expected = math.log(100.0 / 90.0)

    assert analysis.n_valid_pairs == 20
    assert analysis.tg.point_estimate == pytest.approx(expected)
    assert analysis.tg.bootstrap_interval == pytest.approx((expected,) * 3)
    assert analysis.tg.block_t_interval == pytest.approx((expected,) * 3)


def test_identical_values_produce_intervals_containing_zero() -> None:
    analysis = analyze_session(_session(head_tg=100.0, head_pp=200.0))

    for metric in (analysis.tg, analysis.pp):
        assert metric.bootstrap_interval[1] <= 0.0 <= metric.bootstrap_interval[2]
        assert metric.block_t_interval[1] <= 0.0 <= metric.block_t_interval[2]
        assert metric.base_cv == 0.0
        assert metric.head_cv == 0.0


def test_block_drift_tracks_common_throughput_decay() -> None:
    results = []
    for invocation in generate_schedule(n_warmup_pairs=0, seed="drift"):
        scale = 1.0 if invocation.block_index < 5 else 0.9
        results.append(
            FakeResult(
                invocation,
                invocation.arm,
                (_sample("pp", 200.0 * scale), _sample("tg", 100.0 * scale)),
            )
        )
    session = SimpleNamespace(
        measured_results=tuple(results), contaminated_blocks=()
    )

    analysis = analyze_session(session)

    assert analysis.tg.block_drift == pytest.approx(-0.10)
    assert analysis.pp.block_drift == pytest.approx(-0.10)
