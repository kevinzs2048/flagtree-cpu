# Q4 compiler-visible fused decode

## Result

The production M=1--3 Q4 route now launches activation quantization and SDOT
GEMV through one Triton entry.  It remains ordinary Triton codegen:

```
Triton frontend
  -> TTIR (BF16 absmax/RNE quantization + packed Q4 dot algebra)
  -> TTCIR
  -> LLVM IR
  -> AArch64 FCVTNS/SDOT/ADDP
```

There is no TLE_raw op, external compute symbol, or hand-written C compute
runtime.  The legacy two-launch route remains available with
`FLAGGEMS_ARM_Q4_FUSED_DECODE=0`; fusion is the default.

The CPU launcher has no cross-program shared-memory barrier.  Each output
partition therefore quantizes the small activation row into private workspace
before entering the existing SDOT body.  These copies execute concurrently.
This trades a small amount of redundant activation work for removal of one
Python/native launch per matrix.  A single UINT8 allocation contains scratch
and returned BF16 output, and the kernel receives only one workspace pointer.

For Qwen3-1.7B decode this changes 113 Q4 matrices from 226 kernel launches to
113.  QKV and gate/up are already physically packed as fused matrices.

## Same-process microbenchmarks

CIX performance cores `0,1,6-11`, eight PyTorch/OpenMP threads, complete direct
pipeline including activation packing and output allocation:

| Qwen matrix | Legacy us | Fused us | Paired improvement |
| --- | ---: | ---: | ---: |
| QKV, K=2048 N=4096 | 191.86 | 174.25 | 10.37% |
| O, K=2048 N=2048 | 145.91 | 132.96 | 9.93% |
| Gate/up, K=2048 N=12288 | 266.51 | 253.07 | 4.88% |
| Down, K=6144 N=2048 | 199.94 | 183.47 | 7.53% |

The QKV shape also improved with 1/2/4/8 threads by
2.99%/5.05%/7.48%/10.37%.  At eight threads M=2 and M=3 improved by 10.80%
and 7.11%.  A small K=1024 N=64, M=1 case improved by 10.56%.

## Qwen3-1.7B end-to-end decode

The durable paired benchmark loads and quantizes one model once, alternates the
legacy and fused routes for 15 pairs, and builds KV input outside each timed
decode step:

- legacy median: 67.851 ms
- fused median: 64.650 ms
- paired-ratio median: 0.95281, or 4.72% faster
- complete vocabulary logits: bit-exact
- greedy token: identical

Artifact:
`artifacts/qwen3-1.7b-q4-fused-decode-paired-pcores.json`.

A second same-process test keeps two copies of the model and alternates fused
Triton with the controlled ATen/KleidiAI Q4 backend:

- fused Triton median: 58.684 ms
- ATen/KleidiAI median: 55.436 ms
- paired Triton delta: +5.79%
- greedy token: identical in all pairs

The arithmetic is not bit-identical between these backends: ATen uses its own
dynamic activation quantization contract.  This comparison is a performance
baseline, not an equality test.  Artifact:
`artifacts/qwen3-1.7b-q4-fused-triton-vs-aten-paired-pcores.json`.

## Codegen and correctness gates

For K=1024, unroll=1, the fused function contains exactly eight FCVTNS and
eight SDOT instructions, three UMAX reductions and an ADDP.  It has no external
compute calls and only two function-level stack references.  The gate also
checks fused versus legacy output bit-for-bit for M=1,2,3, ordinary reference
accuracy for all router shapes, dynamic `torch.compile` shapes, KAI RHS packing,
and M4/M8/M12/M16 I8MM codegen.

Passing commands:

```
/home/cix/venv-fep-e2e/bin/ninja -C ports/triton-cpu-3.7.2/build-port
taskset -c 0,1,6-11 env OMP_NUM_THREADS=8 \
  /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/test_flaggems_q4_production_router.py --k 1024 --n 64
taskset -c 0,1,6-11 env OMP_NUM_THREADS=8 \
  /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/test_qwen3_q4_full_codegen.py
```

Both pass.  The full-model gate confirms five fused decode calls in the tiny
Qwen model and preserves the established attention tolerance.

## Remaining gap

The direct SDOT compute body is already close to the corresponding KleidiAI
microkernel.  Follow-on work therefore attacked model-local launch boundaries
without adding a C runtime matrix implementation.

## Qwen model-local fusion follow-on

Two more default decode routes now remain entirely compiler-visible:

1. `input RMSNorm -> joined QKV` becomes one Triton entry.  It does not
   materialize the normalized K=2048 row.  The generated function contains
   `FSQRT`, exact BF16 rounding, `FCVTNS`, and `SDOT` in the same LLVM body.
2. `residual add + post-attention RMSNorm -> joined gate/up` becomes one
   Triton entry.  Each output partition uses private BF16 summed-residual
   scratch.  Partition zero's partition-major slice is returned as the new
   residual, avoiding both a second allocation and any cross-program write.

The generic fused Q4 pack was also changed to keep four native K8 BF16 slices
live across absmax and quantization.  The resulting assembly has no hot-loop
spill, so the activation is loaded only once per K32 group.

### Microbenchmarks

For Qwen3-1.7B QKV (M=1, K=2048, N=4096, eight threads), standalone RMSNorm
plus Q4 took 231.04 us and the single entry took 156.95 us, a 31.66% reduction.
The paired improvements at 1/2/4/8 threads were
15.61%/21.60%/28.05%/31.66%.  M=2 and M=3 improved by 27.09% and 23.06%.

For post-attention add/RMSNorm plus gate/up (M=1, K=2048, N=12288), the paired
microbenchmark improved by 3.71%/6.24%/9.19%/11.03% at 1/2/4/8 threads.

### End-to-end and native baseline

On fixed CIX performance cores, the input-RMSNorm/QKV fusion improved a
15-pair Qwen3-1.7B decode A/B by 3.26%.  The race-free post-attention fusion
added another independently measured 1.58%.  Every complete vocabulary logit
was bit-exact in both A/B tests.

With both fusions enabled, a same-process two-model 15-pair comparison measured:

- ordinary-Triton Q4: 61.472 ms
- ATen/KleidiAI Q4: 61.271 ms
- paired Triton delta: +0.49%

This is within the observed board variation and closes the previous measured
5.79% decode gap.  Both models selected token 8741 in every pair.  Their Q4
activation arithmetic remains independently defined, so token equality is not
presented as backend bit equality.

Artifacts:

- `artifacts/qwen3-1.7b-q4-rmsnorm-qkv-paired-pcores.json`
- `artifacts/qwen3-1.7b-q4-add-rmsnorm-gateup-racefree-paired-pcores.json`
- `artifacts/qwen3-1.7b-q4-double-fused-triton-vs-aten-paired-pcores.json`
- `artifacts/qwen3-1.7b-q4-double-fused-decode-profile-pcores.json`

### Correctness and codegen audit

An early post-attention prototype contained an intermittent cross-program
race: partition zero could overwrite the public residual while a delayed
partition was still reading it.  A 200-iteration stress test reproduced the
failure at iteration 126.  The production design never publishes into shared
input storage; after the fix, 500 large-shape iterations and all 15 full-model
pairs were bit-exact.

Both new kernels contain exactly eight `FCVTNS`, eight `SDOT`, one `FSQRT`,
three `UMAX`, and an `ADDP` for K=1024/unroll=1.  They contain no external
compute call and no `triton_cpu.dot`.  The add/RMSNorm kernel's six stack
references are incoming scalar ABI loads at function entry; there are no
K-loop stack accesses.

The next measured target is the remaining joined Q/K head RMSNorm launch and
framework view/allocation overhead.  Any further fusion must preserve the same
race-free private ownership rule and keep compute phases in generated LLVM.
