#!/usr/bin/env python3
"""Run textual LLVM IR through Triton 3.7.2's LLVM 23 O3 and Arm codegen."""

from __future__ import annotations

import argparse
from pathlib import Path

from triton._C.libtriton import llvm
from triton.backends.cpu.compiler import CPUBackend
from triton.runtime.driver import driver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    llvm.init_targets()
    context = llvm.context()
    module = llvm.parse_ir(args.input.read_text(), context)
    llvm.set_host_target(module)

    # Target choice is supplied once, below, to both frontends.  Remove
    # Clang-only target and O0 attributes before the common O3 pipeline.
    for function in module.get_functions():
        for attribute in (
            "noinline",
            "optnone",
            "optsize",
            "minsize",
            "target-cpu",
            "target-features",
            "tune-cpu",
            "frame-pointer",
            "min-legal-vector-width",
        ):
            function.remove_fn_attr(attribute)

    llvm.optimize_module(module, llvm.OPTIMIZE_O3)
    optimized = str(module)
    target = driver.active.get_current_target()
    backend = CPUBackend(target)
    assembly = llvm.translate_to_host_asm(
        optimized,
        True,
        True,
        backend.llvm_target_features(),
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".optimized.ll").write_text(optimized)
    args.output_prefix.with_suffix(".s").write_text(str(assembly))


if __name__ == "__main__":
    main()
