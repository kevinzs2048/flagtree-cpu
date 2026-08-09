# Arm Triton Mac handoff source revisions

The Mac/Qwen3.5 handoff is split across the parent compiler repository and
three source submodules. Generated artifacts, model weights, Linux shared
objects, JIT caches, build directories, and benchmark output are not part of
the source handoff.

| component | branch | revision |
| --- | --- | --- |
| Triton-CPU 3.7.2 | `codex/arm-codegen-port-3.7.2-20260810` | `03f96ac0477f1c994879b0ce07896c335bae50d2` |
| FlagGems Arm | `codex/arm-q4-q8-qwen35-20260810` | `e89cad9aea825dae8ed02efbd67cb762cde0e3ee` |
| libtriton_jit CPU | `codex/cpu-q4-q8-20260810` | `02179a1b005648bd473ffc930824b728ac18d89d` |

Clone the parent handoff branch recursively, or initialize after pulling:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Do not copy CIX build products to Darwin. Build every submodule locally on the
Mac and confirm that generated JIT libraries are Mach-O arm64 before running
the validation scripts in `MAC_QWEN35_9B_HANDOFF.md`.
