# EdgeCI validation note — 2026-07-17

Records the first physical Apple Silicon validation of EdgeCI 0.1.0.
Execution and decision semantics only — not customer demand evidence or a
production-ready false-alert rate.

## Testbed

- Apple M5 MacBook Air, 16 GB unified memory, 8-core GPU
- macOS 26.5.2 (build 25F84)
- AC power; Low Power Mode off
- `llama.cpp` b10052, commit `b2dd28a3b`
- Official macOS arm64 archive SHA-256:
  `20cce6aa20b0823d847ede7755c59abb8dc1462a06694714dc01ce9851d16e24`
- Local 1.8 GB GGUF, SHA-256 prefix `278bc2d9`
- Workloads: `pp512` and `tg128`
- 20 measured base/head pairs plus two discarded warm-up pairs
- 10 balanced ABBA/BAAB blocks; 15 seconds between invocations
- 5% non-inferiority budgets

## Null control

The exact same `llama-bench` binary was used under hidden base/head labels.

| Metric | Base | Head | Paired change (95% confirmatory envelope) | Result |
|:--|--:|--:|--:|:--|
| `tg128` | 59.4 t/s | 59.4 t/s | +0.1% [−1.0%, +1.2%] | PASS |
| `pp512` | 1395 t/s | 1391 t/s | −0.3% [−1.0%, +0.4%] | PASS |

- Duration: 17m23s
- Invalidated blocks: 0
- `tg128` base/head CV: 2.1% / 1.9%
- `pp512` base/head CV: 0.7% / 1.3%
- Captured thermal state: nominal
- Captured memory pressure: normal

Individual identical-binary generation pairs ranged from about −5.1% to +6.7%.
The balanced paired protocol nevertheless placed the aggregate effect close to
zero with an interval of roughly ±1%. A single unpaired threshold comparison
would have produced materially different answers depending on the selected run.

The report remained **experimental** — one clean session doesn't enroll a
testbed. EdgeCI needs five clean calibration sessions across two days.

## Synthetic 6% canary

A wrapper multiplied the head binary's reported throughput by
0.94. Not a real runtime bug — just exercises the statistics and verdict
path. 20 pairs with shortened 0.25-second gaps.

| Metric | Paired change (95% confirmatory envelope) | Result |
|:--|--:|:--|
| `tg128` | −5.0% [−6.5%, −3.5%] | INCONCLUSIVE |
| `pp512` | −5.9% [−6.2%, −5.6%] | FAIL |

Overall result: **FAIL (experimental)**. Prompt processing had both
confirmatory intervals beyond the −5% budget. Generation overlapped the policy
boundary and correctly remained inconclusive. Thermal samples included both
nominal and fair states; the report exposed the transition rather than deleting
completed measurements.

## Runtime-contract finding

The first physical run failed before measurement because current b10052 reports
the Apple GPU backend as `MTL,BLAS`, while the alpha recognized only `Metal`.
EdgeCI now canonicalizes `MTL` to `Metal`, records GPU acceleration correctly,
and regression-tests the current schema. This is the kind of upstream contract
drift a runtime-specific adapter must fail closed on.

EdgeCI also now probes `llama-bench --help` once per distinct binary, resolves
known option aliases, and pins every semantic in its v0.1 workload contract.
If either revision cannot honor that contract, comparison stops before warm-up
or measurement instead of silently inheriting different defaults. A follow-up
four-pair M5 smoke against b10052 completed with the pinned contract, Metal
acceleration, zero invalidated blocks, and an appropriately INCONCLUSIVE
overall result at that small sample size (`pp512` independently passed).

## What this does not prove

- It is one fanless Mac, one local model, and one clean null session.
- The synthetic canary is not a detected `llama.cpp` regression.
- No external developer has installed or repeated the run.
- No actively cooled Mac has been tested.
- No false-fail rate or detection-power claim is justified yet.
- The default protocol takes long enough to be a release-qualification job,
  not necessarily an every-commit job.

Next evidence required: independent A/A runs on fanless and actively cooled
Macs, a reproduced known regression, repeat use in an external repository, and
a paid release-qualification pilot.
