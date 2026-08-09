#!/usr/bin/env python3
"""Audit every selected LLVM object in the Qwen3-0.6B W8 AOT bundle."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelSpec:
    group: str
    relative_dir: str
    symbol: str
    expected_sdot: int = 0
    expected_round: int = 0
    expected_umax: int = -1
    expected_fmaxnm: int = -1
    expected_fminnm: int = -1
    max_stack: int = 0


@dataclass(frozen=True)
class AuditResult:
    group: str
    object: str
    symbol: str
    instructions: int
    calls: int
    stack_refs: int
    sdot: int
    round_even: int
    umax: int
    fmaxnm: int
    fminnm: int


W8_SHAPES = (
    (1024, 1024, 64),
    (1024, 2048, 64),
    (2048, 1024, 64),
    (1024, 3072, 64),
    (3072, 1024, 64),
    (1024, 4096, 64),
    (1024, 152064, 32),
)


def selected_kernels() -> list[KernelSpec]:
    specs: list[KernelSpec] = []
    for k, n, block_n in W8_SHAPES:
        directory = f"k{k}-n{n}-bn{block_n}"
        specs.extend(
            (
                KernelSpec(
                    f"w8-k{k}-n{n}",
                    directory,
                    "_quantize_bf16_w8_kernel",
                    expected_round=4,
                    expected_umax=3,
                    expected_fmaxnm=2,
                    expected_fminnm=0,
                ),
                KernelSpec(
                    f"w8-k{k}-n{n}",
                    directory,
                    "_w8a8_wide_gemv_kernel",
                    expected_sdot=block_n // 2,
                ),
            )
        )

    mlp_dir = "mlp-k1024-n3072-bn64"
    specs.extend(
        (
            KernelSpec(
                "mlp",
                mlp_dir,
                "_quantize_bf16_w8_kernel",
                expected_round=4,
                expected_umax=3,
                expected_fmaxnm=2,
                expected_fminnm=0,
            ),
            KernelSpec(
                "mlp", mlp_dir, "_w8a8_wide_gemv_kernel", expected_sdot=32
            ),
            KernelSpec(
                "mlp",
                mlp_dir,
                "_bf16_swiglu_inline_exp_kernel",
                expected_round=4,
                max_stack=8,
            ),
        )
    )
    specs.extend(
        KernelSpec("rmsnorm", f"rms-m{m}-n{n}-e1e-6", "_rms_norm_aot_kernel")
        for m, n in ((1, 1024), (16, 128), (8, 128))
    )
    specs.extend(
        (
            KernelSpec(
                "fused-add-rmsnorm",
                "fused-rms-m1-n1024-e1e-6",
                "_fused_add_rms_aot_kernel",
            ),
            KernelSpec(
                "rope",
                "rope-hq16-hkv8-d128",
                "_rope_qk_aot_kernel",
            ),
        )
    )
    return specs


INSTRUCTION = re.compile(
    r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+([a-zA-Z0-9.]+)", re.MULTILINE
)


def audit_one(bundle: Path, spec: KernelSpec) -> AuditResult:
    object_path = bundle / spec.relative_dir / f"{spec.symbol}.so"
    if not object_path.is_file():
        raise RuntimeError(f"missing selected object: {object_path}")
    disassembly = subprocess.run(
        ["objdump", "-d", f"--disassemble={spec.symbol}", str(object_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.lower()
    mnemonics = INSTRUCTION.findall(disassembly)
    calls = sum(opcode in {"bl", "blr"} for opcode in mnemonics)
    stack_refs = sum("[sp" in line for line in disassembly.splitlines())
    sdot = mnemonics.count("sdot")
    round_even = sum(opcode in {"fcvtns", "frintn"} for opcode in mnemonics)
    umax = mnemonics.count("umax")
    fmaxnm = mnemonics.count("fmaxnm")
    fminnm = mnemonics.count("fminnm")

    failures = []
    if not mnemonics:
        failures.append("exported function has no instructions")
    if calls:
        failures.append(f"external/indirect calls={calls}")
    if stack_refs > spec.max_stack:
        failures.append(f"stack refs={stack_refs}, maximum={spec.max_stack}")
    if sdot != spec.expected_sdot:
        failures.append(f"SDOT={sdot}, expected={spec.expected_sdot}")
    if round_even != spec.expected_round:
        failures.append(
            f"round-even instructions={round_even}, expected={spec.expected_round}"
        )
    if spec.expected_umax >= 0 and umax != spec.expected_umax:
        failures.append(f"UMAX={umax}, expected={spec.expected_umax}")
    if spec.expected_fmaxnm >= 0 and fmaxnm != spec.expected_fmaxnm:
        failures.append(
            f"FMAXNM={fmaxnm}, expected={spec.expected_fmaxnm}"
        )
    if spec.expected_fminnm >= 0 and fminnm != spec.expected_fminnm:
        failures.append(
            f"FMINNM={fminnm}, expected={spec.expected_fminnm}"
        )
    if failures:
        raise RuntimeError(f"{object_path}: " + "; ".join(failures))

    return AuditResult(
        group=spec.group,
        object=str(object_path.resolve()),
        symbol=spec.symbol,
        instructions=len(mnemonics),
        calls=calls,
        stack_refs=stack_refs,
        sdot=sdot,
        round_even=round_even,
        umax=umax,
        fmaxnm=fmaxnm,
        fminnm=fminnm,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = [audit_one(args.bundle.resolve(), spec) for spec in selected_kernels()]
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return

    for result in results:
        print(
            f"{result.group:24} {result.symbol:38} "
            f"insns={result.instructions:3} calls={result.calls} "
            f"stack={result.stack_refs} sdot={result.sdot:2} "
            f"round={result.round_even} umax={result.umax} "
            f"fmaxnm={result.fmaxnm} fminnm={result.fminnm}"
        )
    print(
        f"PASS selected_objects={len(results)} calls=0 "
        "unbounded_stack=0"
    )


if __name__ == "__main__":
    main()
