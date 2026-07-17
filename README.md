# EdgeCI

Performance release gate for on-device AI inference on Apple Silicon.

Runs paired `llama-bench` measurements (base vs. head), balances execution order with ABBA/BAAB scheduling, and produces a statistically grounded **PASS**, **FAIL**, or **INCONCLUSIVE** verdict.

Scope is intentionally narrow: two `llama.cpp` Metal builds, one Mac, one local GGUF model. Not a benchmark suite, not a dashboard, not a hosted service.

## Requirements

- Apple Silicon Mac on AC power (Low Power Mode off, nominal thermals, low background load)
- Python ≥ 3.10
- `llama-bench` binaries built with Metal
- One local GGUF model — EdgeCI never downloads anything or makes network calls

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify:

```bash
edgeci --version
edgeci doctor --model /path/to/model.gguf
```

## Quick start

Copy [`examples/edgeci.toml.example`](examples/edgeci.toml.example) to `.edgeci.toml`, set your model path, then check runner health:

```bash
edgeci doctor
```

Run a comparison:

```bash
edgeci compare \
  --base /path/to/base/llama-bench \
  --head /path/to/head/llama-bench \
  --model /path/to/model.gguf
```

Results go to `./edgeci-results/` by default:

- `result.json` — machine-readable canonical result with provenance and raw measurements
- `report.md` — GitHub-flavored Markdown for PR comments
- `schedule.json` — deterministic invocation order and seed

### Exit codes

| Code | Meaning |
|---:|:---|
| `0` | PASS |
| `1` | FAIL |
| `2` | INCONCLUSIVE |

Unenrolled runners operate in shadow mode: `edgeci compare` returns `0` regardless of verdict to avoid blocking CI with uncalibrated evidence. The report contains the real verdict. Calibration commands always use the codes above.

Read the report even on nonzero exits. FAIL means evidence of regression. INCONCLUSIVE means the protocol couldn't establish either direction.

## Configuration

EdgeCI walks up from the current directory looking for `.edgeci.toml`. `--config` overrides this; CLI flags override file values.

```toml
[model]
path = "/absolute/path/to/model.gguf"

[benchmark]
prompt_tokens = 512
generate_tokens = 128
pairs = 20
warmup_pairs = 2
gap_seconds = 15
timeout_minutes = 60

[budgets]
tg = 0.05
pp = 0.05

[preflight]
thermal_settle_seconds = 60
idle_cpu_threshold = 0.20
post_build_cooldown = 120
preflight_timeout = 600

[report]
format = "all"
output_dir = "./edgeci-results"
```

The `0.05` budgets tolerate up to 5% throughput loss for token generation (`tg`) and prompt processing (`pp`). Lower = stricter. Changing pair count or timing changes measurement power; recalibrate on the target runner.

`timeout_minutes` caps the full comparison (validation, hashing, probes, preflight, measurement, recovery). EdgeCI kills the subprocess group on deadline or interrupt.

## Commands

### `edgeci doctor`

Checks seven prerequisites without running a benchmark:

1. Apple Silicon hardware
2. Nominal thermal state
3. AC power, Low Power Mode off
4. Normal memory pressure
5. CPU load below threshold
6. `llama-bench` available
7. GGUF model exists

```bash
edgeci doctor --config .edgeci.toml
```

### `edgeci compare`

Runs preflight, discarded warm-ups, and measured pairs sequentially. `--seed` reproduces schedule order:

```bash
edgeci compare \
  --base ./build-base/bin/llama-bench \
  --head ./build-head/bin/llama-bench \
  --model /models/model.gguf \
  --seed pr-1842-attempt-1 \
  --format all \
  --output ./edgeci-results
```

Do not run hardware-probing commands concurrently. EdgeCI holds a machine-wide lock at `/tmp/edgeci.lock`.

### `edgeci calibrate`

Full protocol as an A/A test. A healthy runner should show effects near zero and no false FAIL.

```bash
# Same binary path, hidden labels
edgeci calibrate --binary ./llama-bench --model /models/model.gguf

# Byte-for-byte copy
edgeci calibrate --binary ./llama-bench --model /models/model.gguf --mode copy

# Two independent builds of the same source
edgeci calibrate \
  --binary ./build-a/bin/llama-bench \
  --equivalent-binary ./build-b/bin/llama-bench \
  --model /models/model.gguf \
  --mode equivalent-build
```

Calibration sessions are stored in `~/.edgeci/calibrations/`. Five clean sessions across two or more days enrolls the runner. Enrollment expires 14 days after the latest qualifying session and invalidates on OS, hardware, model, or protocol changes.

A session is clean only when both null-effect intervals include zero, effects stay within budgets, no diagnostic warnings fire, and the production protocol returns PASS. Large apparent A/A speedups don't qualify — they'd enroll a biased runner.

### `edgeci status`

Hardware fingerprint, enrollment state, expiry, recent calibration history, observed A/A false-fail rate:

```bash
edgeci status
```

## How the verdict works

The schedule is generated before execution. Each balanced block = one AB pair + one BA pair, arranged as ABBA or BAAB. Base and head never run concurrently.

Per metric, EdgeCI computes paired log-ratios `ln(base / head)`, balances AB/BA order, then evaluates two preregistered 95% intervals:

- **Stratified bootstrap** over AB and BA ratios independently
- **Block-t** over balanced block averages

For throughput, positive log-effect = head is slower. Both intervals must fall below the regression boundary → PASS. Both lower bounds must exceed it → FAIL. Any disagreement or overlap → INCONCLUSIVE. Overall: FAIL if either metric fails, INCONCLUSIVE if either is inconclusive, PASS only when both pass.

High CV, large block drift, and interval disagreement produce warnings. Warnings never silently change the verdict. Completed measurements are never dropped for looking like outliers — only observable external events can contaminate a block.

## Reproducible runner practice

- Dedicate the Mac during a run
- AC power, awake, Low Power Mode off
- Kill indexing, backups, builds, browsers, heavy background work
- Same model file and benchmark parameters for both binaries
- Let EdgeCI handle post-build cooldown and thermal settling
- Don't delete results because thermals changed after an invocation started — that's data
- Recalibrate after OS updates, hardware changes, or protocol changes

## GitHub Actions

[`examples/workflow.yml.example`](examples/workflow.yml.example) is a two-job PR workflow. Benchmarking runs on a labeled self-hosted Apple Silicon runner with read-only permissions. A separate hosted job downloads the report and posts a PR comment with `pull-requests: write`.

Before using it:

1. Set `EDGECI_MODEL_PATH` to an absolute model path on the runner
2. Set `EDGECI_WHEELHOUSE` to a runner-owned read-only directory with reviewed EdgeCI 0.1.0 and dependency wheels
3. Keep `.edgeci.toml` on the protected base branch — the template never reads protocol settings from PR head
4. Install build tools (Python 3, CMake, Ninja, compiler) on the Mac
5. Create an `edgeci-approved` environment with required reviewers
6. Restrict the runner group to this repository/workflow; Actions runner ≥ 2.329.0

Fork PRs are rejected at the job boundary. PR code still runs on the self-hosted Mac after environment approval — use a dedicated runner with no sensitive credentials or network access.

## Validation

The [first M5 validation](docs/validation-2026-07-17.md) records a 20-pair null control, a synthetic boundary canary, observed variance, and limitations. Alpha evidence from one machine — not a cross-hardware false-alert rate claim.

## Development

```bash
pip install -e '.[test]'
pytest
```

The statistical core uses only the Python standard library (no NumPy/SciPy). The tool makes no network calls.

## License

MIT. See [`LICENSE`](LICENSE).
