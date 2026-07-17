"""Tests for the pinned llama-bench CLI contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgeci import orchestrator
from edgeci.config import BenchmarkConfig
from edgeci.orchestrator import ComparisonError
from edgeci.probe import LlamaBenchInfo
from edgeci.runner import (
    RunnerError,
    build_llama_bench_command,
    parse_llama_bench_contract,
)


CURRENT_HELP = """
usage: llama-bench [options]
options:
  -r, --repetitions <n>                     number of repetitions
  --prio <-1|0|1|2|3>                       process priority
  --delay <0...N>                           delay between tests
  -o, --output <csv|json|jsonl|md|sql>      output format
test parameters:
  -m, --model <filename>
  -p, --n-prompt <n>
  -n, --n-gen <n>
  -d, --n-depth <n>
  -b, --batch-size <n>
  -ub, --ubatch-size <n>
  -ctk, --cache-type-k <t>
  -ctv, --cache-type-v <t>
  -t, --threads <n>
  -C, --cpu-mask <hex,hex>
  --cpu-strict <0|1>
  --poll <0...100>
  -ngl, --n-gpu-layers <n>
  -ncmoe, --n-cpu-moe <n>
  -sm, --split-mode <none|layer|row|tensor>
  -mg, --main-gpu <i>
  -nkvo, --no-kv-offload <0|1>
  -fa, --flash-attn <on|off|auto>
  -dev, --device <dev0/dev1/...>
  -mmp, --mmap <0|1>
  -dio, --direct-io <0|1>
  -embd, --embeddings <0|1>
  -ts, --tensor-split <ts0/ts1/..>
  -nopo, --no-op-offload <0|1>
  --no-host <0|1>
"""


def test_command_pins_complete_neutral_workload() -> None:
    """Every stable performance-affecting default is explicit in argv."""

    contract = parse_llama_bench_contract(CURRENT_HELP)

    command = build_llama_bench_command(
        Path("/base/llama-bench"),
        Path("/models/shipping.gguf"),
        BenchmarkConfig(prompt_tokens=256, generate_tokens=64),
        6,
        contract=contract,
    )

    assert command == (
        "/base/llama-bench",
        "-m", "/models/shipping.gguf",
        "-p", "256",
        "-n", "64",
        "-d", "0",
        "-r", "1",
        "-b", "2048",
        "-ub", "512",
        "-ctk", "f16",
        "-ctv", "f16",
        "-ngl", "-1",
        "-ncmoe", "0",
        "-sm", "layer",
        "-mg", "0",
        "-ts", "0",
        "-nkvo", "0",
        "-fa", "auto",
        "-dev", "auto",
        "-mmp", "1",
        "-dio", "0",
        "-embd", "0",
        "-nopo", "0",
        "--no-host", "0",
        "--prio", "0",
        "--delay", "0",
        "-C", "0x0",
        "--cpu-strict", "0",
        "--poll", "50",
        "-t", "6",
        "-o", "jsonl",
    )
    assert "--no-warmup" not in command


def test_missing_capability_fails_closed() -> None:
    """A binary cannot silently inherit a changed flash-attention default."""

    incompatible_help = CURRENT_HELP.replace(
        "  -fa, --flash-attn <on|off|auto>\n", ""
    )

    with pytest.raises(RunnerError, match=r"flash_attn \(-fa/--flash-attn\)"):
        parse_llama_bench_contract(incompatible_help)


def test_comparison_probes_identical_binary_help_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A/A same-path calibration does not re-probe help per arm or invocation."""

    binary = tmp_path / "llama-bench"
    calls: list[Path] = []

    def fake_detect(path: Path, **_: object) -> LlamaBenchInfo:
        calls.append(path)
        return LlamaBenchInfo(path=path, version="test", help_text=CURRENT_HELP)

    monkeypatch.setattr(orchestrator, "detect_llama_bench", fake_detect)

    base_contract, head_contract = orchestrator._inspect_llama_bench_contracts(
        binary,
        binary,
        deadline=100.0,
        continuous_now=lambda: 0.0,
    )

    assert calls == [binary]
    assert base_contract is head_contract


def test_comparison_rejects_incompatible_head_during_contract_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each distinct arm is probed once and mismatch stops before measurement."""

    base = tmp_path / "base-llama-bench"
    head = tmp_path / "head-llama-bench"
    calls: list[Path] = []

    def fake_detect(path: Path, **_: object) -> LlamaBenchInfo:
        calls.append(path)
        help_text = (
            CURRENT_HELP
            if path == base
            else CURRENT_HELP.replace("  --poll <0...100>\n", "")
        )
        return LlamaBenchInfo(path=path, version="test", help_text=help_text)

    monkeypatch.setattr(orchestrator, "detect_llama_bench", fake_detect)

    with pytest.raises(
        ComparisonError,
        match=r"head binary incompatible llama-bench CLI contract.*poll",
    ):
        orchestrator._inspect_llama_bench_contracts(
            base,
            head,
            deadline=100.0,
            continuous_now=lambda: 0.0,
        )

    assert calls == [base, head]
