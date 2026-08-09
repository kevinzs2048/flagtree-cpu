# Qwen3-1.7B Q4 ordinary-Triton coverage on CIX

## Scope

This report covers Hugging Face `Qwen3ForCausalLM` with the local
Qwen3-1.7B configuration:

- hidden size 2048, intermediate size 6144;
- 28 decoder layers;
- 16 query heads, 8 KV heads, head dimension 128;
- vocabulary size 151936 and tied input/output embedding weights.

The production Q4 path uses ordinary Triton operations and the Arm
Triton-CPU compiler.  Its matrix kernels contain no TLE_raw operation and no
external C/C++ compute runtime.

## Model execution coverage

| Model operation | Production implementation | Status and reason |
|---|---|---|
| Token embedding | optimized ATen embedding | An ordinary-Triton row-copy kernel is implemented and exact, but is not routed: on CIX it is 2.3--2.8x slower for 1--128 tokens. |
| Input/final RMSNorm | rolled ordinary Triton | Routed for BF16. |
| Q/K/V projections | one joined Q4A8 matrix | 3 logical projections share one activation pack and matrix launch. |
| Q/K RMSNorm | one fused ordinary-Triton launch | Uses the adjacent joined-QKV decode layout. |
| RoPE frequency table | ordinary Triton | Fuses matmul, repeated-half construction, cos and sin.  It is bit-exact for the tested Qwen3 positions and is routed by the Q4 optimizer. |
| Apply RoPE to Q/K | fused ordinary Triton | Decode and prefill are covered; prefill also materializes the contiguous attention layout. |
| Dynamic KV-cache growth | ATen `cat` | A Triton copy implementation was measured 4.3--6.5x slower.  The useful future route is a preallocated/static cache, not another launcher around a copy. |
| Decode attention | ordinary Triton | Compiler-visible online or staged QK/softmax/PV; the QK reduction lowers to BFDOT. |
| Prefill attention T=2--9 | ATen SDPA | Below the measured CIX crossover. |
| Prefill attention T=10--31 | two-stage ordinary Triton | Faster than ATen from T=10; register-bounded QK/softmax and PV kernels. |
| Prefill attention T>=32 | ordinary-Triton Flash Attention | Existing tiled path. |
| Attention output projection | Q4A8 ordinary Triton | Routed. |
| Attention residual + post RMSNorm | fused ordinary Triton | Routed for BF16 decode. |
| Gate/up projections | one joined Q4A8 matrix | 2 logical projections share one activation pack and matrix launch. |
| SwiGLU | rolled ordinary Triton | Preserves the BF16 SiLU intermediate rounding. |
| Down projection | Q4A8 ordinary Triton | K=6144 decode uses four-way K-loop unrolling. |
| MLP residual add | optimized ATen add | A rolled Triton implementation was exact but measured 16.27 us versus 5.31 us for ATen, so it was rejected. |
| Vocabulary projection | Q4A8 ordinary Triton | The tied 151936x2048 weight is packed in bounded row chunks. |
| Greedy vocabulary argmax | rolled ordinary Triton | Routed for the finite model-logit path. |

Views, reshapes, slices, transposes, allocation, cache ownership and causal
mask bookkeeping remain framework operations.  They are not arithmetic
kernels and should not be relabeled as Triton merely to inflate coverage.

## Linear consolidation

Each decoder layer has seven logical linear operations.  Q/K/V and gate/up
fusion reduce these to four physical Q4 matrices per layer:

- 196 logical layer linears -> 112 physical layer matrices;
- plus one Q4 vocabulary matrix;
- 197 logical linears -> 113 physical Q4 matrix calls per decode token.

The Q4 decode kernel consumes the KAI qsi8d32p/qsi4c32p physical layout, but
its computation is expressed with `tl.load`, integer arithmetic, reductions
and `tl.dot`-recognizable layout algebra.  LLVM output for the K=6144/N=2048
down projection contains 32 SDOT and 4 ADDP instructions, no residual
`triton_cpu.dot`, no external compute call and no hot-loop stack access.
Prefill M4/M8/M12/M16 specializations contain 16/32/48/64 SMMLA instructions.

## Measured results

Environment: CIX AArch64 board, cores 4--11 pinned, eight PyTorch/OpenMP
threads, BF16 activations, Q4 K32 weights, Triton-CPU 3.7.2 port.

- Full Qwen3-1.7B production Q4 route, 12-token prompt: 303.05 ms median
  prefill (three samples) and 104.13 ms median warmed one-token decode (five
  samples).
- The matching BF16 eager baseline measured 1485.77 ms prefill and 1179.23 ms
  decode: the current Q4 route is 4.90x and 11.32x faster respectively.  This
  is a quantized-model comparison, not an assertion that compiler changes
  alone provide the full speedup.
- Profiler-visible Triton self-CPU coverage: 91.27%.
- The largest remaining non-Triton entries were allocation/view bookkeeping,
  dynamic KV-cache `cat` (1.28%) and residual add (0.44%).
- Q4 K=6144/N=2048 decode output partitioning: 622 us with one partition and
  421 us with eight partitions, a 1.48x CIX multi-core gain.
- Direct single-core, same-packed-blob microbenchmarks versus KleidiAI v1.24
  were at parity for every Qwen3-1.7B matrix: joined QKV 1.0058x, O projection
  1.0042x, joined gate/up 1.0046x, and down projection 0.9912x
  (Triton/native; lower is better).  All outputs were exact.
- RoPE frequency generation was bit-exact and measured 64.25 us versus
  126.54 us for ATen at T=1, and 84.51 us versus 167.19 us at T=12.  An
  exhaustive check of all 40960 configured positions compared 10485760 BF16
  cos/sin values with zero mismatches.
- Short causal attention B=1/Hq=16/Hkv=8/D=128: Triton is faster than ATen
  from T=10; at T=12 it measured 191 us versus 277 us, and at T=31 it measured
  517 us versus 1593 us.  Maximum observed BF16 absolute error was 0.015625.
- The tiled prefill path remained ahead at ordinary prompt sizes: T=32 was
  1660 us versus 1698 us, T=64 was 4965 us versus 6269 us, T=128 was 18401 us
  versus 24590 us, and T=256 was 52547 us versus 97033 us.  Maximum absolute
  error in this sweep was also 0.015625.

Artifacts and reproducible gates:

- `artifacts/qwen3-1.7b-q4-full-codegen-production.json`
- `artifacts/operator-native-gate/20260808-162743/results.json`
- `benchmarks/test_qwen3_q4_full_codegen.py`
- `benchmarks/test_flaggems_q4_production_router.py`
- `benchmarks/audit_arm_q4_q8_codegen.py`

## Honest boundary

“All expensive model arithmetic is covered” is supportable.  “Every ATen
operator was replaced” is neither true nor desirable.  The measured rejected
embedding, residual-add and dynamic-cache kernels show
the sweet spot: Triton replaces sufficiently dense, fused or reduction-heavy
work; mature ATen primitives remain preferable for tiny copies and adds until
launch overhead or cross-operator fusion is removed.

The kernel correctness gate compares against the exact grouped-Q4A8 reference
and allows only sparse BF16 accumulation-order differences.  Q4 inference is
not bit-equivalent to the original BF16 model: for the documented prompt the
first generated token matched BF16 (`264`), while the second token differed
(`8741` versus `18512`).  Model-quality evaluation on representative tasks is
still required before treating live Q4 quantization as a deployable model
conversion rather than a code-generation benchmark.
