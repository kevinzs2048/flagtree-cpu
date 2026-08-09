# CIX Q4/Q8 production codegen status

Date: 2026-08-03

## Active paths

The active Q4 and Q8 routes are compiler generated. Their matrix loops do not
tail-call a C/KleidiAI runtime implementation.

| Route | Triton-level expression | CIX lowering |
| --- | --- | --- |
| Q4 M1-M3 decode | ordinary BF16 bit operations, `libdevice.rint`, loads, `tl.dot`, and stores over KAI blobs | UMAX/FCVTNS compact pack + 8 SDOT + ADDP, no external call |
| Q4 M4/M8/M12/M16 | ordinary M4 scale-vectorized `libdevice.rint` pack + `tl.dot` over KAI qsi8d32p/qsi4c32p panels | FRINTN pack + 16/32/48/64 fixed I8MM instructions |
| Q8 M1 single core | compiler op expanded to rolled vector IR | 16 SDOT sites, no external call |
| Q8 M1 multicore | ordinary `libdevice.rint` quantizer + compiler-expanded N64 grid | 4 FRINTN + 16 SDOT sites, both kernels spill-free |
| Q8 M2-M12 | ordinary KAI M4/K8 pack + shape-specialized `tl.dot` | 16/32/48 SVE2 SMMLA sites for padded M4/M8/M12 |
| Q8 M>=13 | ordinary KAI M4/K8 pack + ordinary M16 `tl.dot` | 64 SVE2 SMMLA sites, no external call |
| Fused W8 QKV | whole SDOT or shared-quant N64 grid | same spill-free decode microtiles |
| Fused W8 gate/up+SwiGLU | compiler-expanded SDOT microtiles and BF16 epilogue | 16 SDOT sites; inline SLEEF-u10 polynomial, no external exp call |

Q8 prefill uses the same N4/K8 physical layout principle as KleidiAI, but the
matrix calculation remains visible to Triton/LLVM. The compiler selects SVE2
I8MM from `tl.dot`; there is no `TLE_raw` leaf or runtime GEMM symbol.

## CIX microbenchmarks

All numbers below are process medians with the process pinned to the stated
Cortex-A720 CPU set. Python launch and dynamic LHS preparation are included
unless a row says otherwise.

| Test | Before | Current | Result |
| --- | ---: | ---: | ---: |
| Q4 decode LHS pack, M1 K1024, CPU11, paired direct call | 0.808 us | 0.565 us | 1.43x, bit-exact blob |
| Q4 decode pack+matrix, M1 N=K=1024, CPU11, paired direct call | 46.717 us | 46.467 us | 1.005x, bit-exact output |
| Q4 decode pack+matrix, M3 N=K=1024, CPU11, paired direct call | 140.525 us | 139.869 us | 1.005x, bit-exact output |
| Q4 prefill LHS pack, M16 K1024, CPU11, direct call | 9.78 us scalar BF16 reduction plus saturation | 6.66 us BF16 lane-max pack | 1.47x; 0.881x KleidiAI; bit-exact blob |
| Q8 decode quantizer, K1024, CPU11, direct call | 1.512 us manual RNE | 0.349 us native RNE + BF16 lane max | 4.33x, bit-exact |
| Q8 decode absmax only, K1024, CPU11, paired direct call | 0.443 us FP32 reduction | 0.349 us BF16 lane max | 1.271x, bit-exact |
| Q8 whole decode, K=N=1024, CPU11, paired direct call | 27.071 us FP32 absmax lowering | 26.373 us BF16 lane-max lowering | 1.026x, bit-exact |
| Q8 whole decode, K1024 N3072, CPU11, paired direct call | 79.031 us FP32 absmax lowering | 78.386 us BF16 lane-max lowering | 1.008x, bit-exact |
| Fused W8 gate/up+SwiGLU, K1024 N3072, CPU11, paired direct call | 175.918 us FP32 absmax lowering | 171.470 us BF16 lane-max lowering | 1.026x, bit-exact |
| Q8 production AOT quantizer, K1024, CPU11, paired direct call | 0.500 us old bundle object | 0.349 us BF16 lane-max object | 1.433x, bit-exact |
| Q8 production AOT quant+GEMV, K=N=1024 BN64, CPU11, paired direct call | 27.063 us old bundle object | 26.769 us rebuilt bundle object | 1.011x, bit-exact |
| Q8 production AOT gate/up+SwiGLU, K1024 N3072, CPU11, paired C API | 169.973 us old bundle | 166.260 us rebuilt bundle | 1.022x, bit-exact |
| exact-KAI-layout W8 decode, K1024 N3072, CPU11, direct call | KleidiAI 79.64 us | Triton 78.17 us | 0.982x; max abs 2.39e-7 |
| exact-KAI-layout W8 decode, K9728 N2560, CPU11, direct call | KleidiAI 1006.51 us | Triton 953.92 us | 0.948x; max abs 2.39e-7 |
| exact-KAI W8 BF16 pipeline, K1024 N3072, CPU11, direct call | KleidiAI 81.15-81.26 us | Triton 80.57-80.98 us | 0.992-0.998x; bit-exact |
| exact-KAI W8 BF16 pipeline, K2560 N9728, CPU11, direct call | KleidiAI 1017.98-1021.92 us | Triton 983.13-990.08 us | 0.962-0.973x; bit-exact |
| exact-KAI W8 BF16 pipeline, K9728 N2560, CPU11, direct call | KleidiAI 1010.82-1012.32 us | Triton 982.75-985.20 us | 0.972-0.973x; bit-exact |
| exact-KAI W8 M16 BF16 pipeline, K1024 N3072, CPU11, direct call | KleidiAI 361.90 us | Triton 357.60 us | 0.988x; bit-exact |
| exact-KAI W8 M16 BF16 pipeline, K4096 N2560, CPU11, direct call | KleidiAI 1365.70 us | Triton 1343.25 us | 0.984x; bit-exact |
| vLLM custom-op W8 M16, K1024 N3072, CPU11 | KleidiAI 485.37 us | Triton 443.01 us | 0.913x; bit-exact |
| vLLM custom-op W8 M16, K9728 N2560, CPU11 | KleidiAI 3183.70 us | Triton 3118.57 us | 0.980x; bit-exact |
| Q8 row quantizer, M16 K1024, CPU11 | 32.72 us | 31.63 us | 1.03x, bit-exact |
| Q8 KAI LHS pack, M16 K1024, CPU11, paired direct call | 11.315 us FP reduction + clamps | 8.423 us BF16 lane max | 1.343x; 0.681x ACLE; bit-exact blob |
| Q8 KAI pack+matrix, M16 N=K=1024, CPU11, paired direct call | 124.984 us | 122.198 us | 1.023x, bit-exact output |
| Q8 prefill, M16 K1024 N1024, CPU11 | 606.94 us | 219.21 us | 2.77x, bit-exact |
| Q8 prefill, M64 K1024 N1024, CPU11 | 2094.76 us | 667.64 us | 3.14x, bit-exact |
| Q8 decode, K1024 N1024, CPUs6-11 | 47.857 us fused N512 | 42.947 us shared quant + N64 grid | 1.114x, bit-exact |
| FlagGems W8 `Linear.forward`, K=N=1024, CPU11 | 87.070 us whole-TLE | 47.840 us production AOT | 1.820x, bit-exact |
| FlagGems W8 `Linear.forward`, K1024 N3072, CPU11 | 139.676 us whole-TLE | 103.799 us production AOT | 1.346x, bit-exact |
| FlagGems fused W8 QKV, K1024 N2048+1024+1024, CPU11 | 175.117 us whole-TLE | 142.067 us production AOT | 1.233x, bit-exact |
| Triton-CPU 3.7 wide W8 `tl.dot`, K1024 N152064 BN128, CPU11, direct call | 22620.6 us | 7155.8 us | 3.16x, bit-exact |

The Q4 matrix comparison was re-audited against KleidiAI v1.24 using the
production BF16 output contract. Both functions receive the same official
KAI LHS/RHS packed blobs; KleidiAI's FP32 output is rounded to BF16 before
comparison. Packing and Python are excluded and timed order alternates.

| M, N=K=1024 | KleidiAI v1.24 | Triton production object | Triton / KAI |
| ---: | ---: | ---: | ---: |
| 4 | 73.599 us | 73.660 us | 1.001x |
| 8 | 147.129 us | 128.139 us | 0.871x |
| 12 | 220.847 us | 183.140 us | 0.829x |
| 16 | 239.375 us | 238.925 us | 0.998x |

All four comparisons have zero BF16 ULP difference. Shape-specialized M4,
M8 and M12 beat KAI's shared tail path, while M16 is now within 0.3% of KAI.
The compiler used to materialize both M4 row pairs before the first SMMLA;
that made LLVM rotate sixteen FP32 accumulators and spill one Q register in
every K32 group. For the four-panel M16 loop it now loads and consumes one
row pair at a time. The object contracts from 482 to 442 lines and the hot
spill/reload plus 32 accumulator moves disappear. Two M8 programs are now
5.6% slower in the paired CIX test, so the default M16 route is again the
measured best schedule. The old split remains only as an A/B diagnostic.

The three-panel M12 tail has a different LLVM register-allocation optimum.
All six legal M4-panel visit orders were compiled under one compiler build
and compared from fresh caches. Visiting panels in 1,0,2 order reduces the
object from 263 to 258 instructions and total stack references from 14 to 11.
Against the former 2,0,1 order, paired direct matrix calls improve
189.665 to 187.504 us at N=K=1024, 568.674 to 561.232 us at N3072/K1024,
and 565.975 to 558.707 us at N1024/K3072 (1.012-1.013x). Outputs are
bit-exact. The official 3.7.2/LLVM 20 port keeps its previous order: its
lower-stack alternatives measured 0.9993-0.9997x and were correctly rejected.
The only alternate M8 order was also rejected at 0.9986x.

Including the generated BF16 activation pack gives the following direct
pipeline comparison. The KleidiAI column is its measured pack plus matrix
component sum; the Triton column calls both generated symbols back-to-back.

| M, N=K=1024 | KleidiAI components | Triton pipeline | Triton / KAI |
| ---: | ---: | ---: | ---: |
| 4 | 75.50 us | 75.27 us | 0.997x |
| 8 | 150.91 us | 131.23 us | 0.870x |
| 12 | 226.51 us | 188.10 us | 0.830x |
| 16 | 246.93 us | 245.79 us | 0.995x |

The M16 schedule also generalizes beyond the square audit case. With the same
official KAI blobs, alternating direct calls, and BF16 comparison, Triton/KAI
is 1.003x at N3072/K1024, 1.005x at N1024/K3072, and 1.012x at N4096/K4096;
every case has zero BF16 ULP difference. M16 is consistently about 5.6%
faster than issuing two generated M8 programs at these shapes.

The same Q4-prefill lowering is now implemented and tested in the separate
official Triton-CPU 3.7.2 port, not only in the older development compiler.
Before the port, this ordinary `tl.dot` graph generated no SMMLA instructions,
about 1040 assembly lines and hot stack traffic; the M16 Python-path latency
was about 2511 us at N=K=1024. The current object has 64 SMMLA instructions,
no residual dot or external call, and no hot-loop spill. The corresponding
Python-path latency is about 273 us, a 9.2x compiler-only improvement with the
Triton kernel unchanged. Direct symbol comparisons against KleidiAI are
1.000x at N=K=1024, 0.998x at N3072/K1024, 1.004x at N1024/K3072 and 1.004x
at N=K=4096. All four have zero BF16 ULP difference. The port regression test
constructs its own official-layout blobs and requires the full ordinary
Triton graph to become 64 SMMLA calls, so this result does not rely on
FlagGems importing a hidden runtime implementation. The 3.7 generated M16
activation pack is 6.66 us by direct symbol call versus KleidiAI's 7.56 us
from the same BF16-equivalent input, with byte-identical blobs. A C++ wrapper
that dispatches only the two generated symbols measures the complete pipeline
at 245.79 us; timing the symbols separately gives 245.65 us. KleidiAI's
pack-plus-matrix sum is 246.93 us, or 0.995x for the complete native path.
Python launcher timing is not used for the direct native comparison.

The active M4 LHS pack uses a BF16-specific reduction that remains ordinary
Triton. Clearing the sign bit gives a monotonic encoding for finite BF16
magnitudes. Three lane-wise integer maxima combine the four K8 slices before
one horizontal reduction. Because the inverse scale comes from the same K32
absmax, quantization can go directly to RNE conversion without redundant
clamps. The final object has 12 UMAX and 32 FCVTNS instructions, no FMAXNM or
FMINNM clamp, external call, or spill. Full M4 panels are specialized without
dynamic row predicates; the masked final partial panel retains the original
floating-point path. M4/M8/M12/M16 comparisons at K=256, 1024, 3072, and 4096
take 0.876-0.883x the KleidiAI pack latency, with byte-identical blobs.

A contract-equivalent pack comparison first rounds source values to BF16,
then passes the BF16 bits to Triton and the exactly representable FP32 values
to KleidiAI. The resulting KAI blobs are byte-identical for both the timed
input and a suite containing zeros, subnormals, normal bounds, and maximum
finite BF16 values. At M16/K1024 the active generated pack is 6.66 us versus
KleidiAI v1.24's 7.56 us (0.881x).

The M1-M3 compact decode pack now uses the same provable finite-BF16
magnitude ordering without changing its 34-byte-per-K32 ABI. Exact-grid
launches remove the dynamic row predicate, four K8 slices collapse through
three lane-wise integer maxima before one horizontal reduction, and RNE no
longer carries redundant saturation. The direct function body contracts from
107 to 75 instructions: 16 `FMAXNM` plus 8 `FMINNM` operations become three
`UMAX`, with eight `FCVTNS`, no external call, and no stack access. At M1/K1024
the pack improves from 0.808 to 0.565 us (1.43x). M1-M3 blobs are byte-identical
to the independent float-absmax/clamped implementation, including finite BF16
edge patterns. A paired generated-symbol pipeline comparison improves by
about 0.5% at M1-M3/N=K=1024; the smaller whole-operator gain is expected because
the SDOT matrix loop accounts for almost all decode latency.

Q8 short prefill previously padded every M2-M15 input to M16. The compiler
now recognizes the KAI physical graph for M4 and M8 directly, and can fuse an
M8 plus M4 pair through one multi-result K loop for M12. At K1024,N1024,
alternating-order A/B measurements show 1.38-1.40x speedup for M2-M4,
1.18-1.20x for M6-M8, and 1.12-1.14x for M9-M12 versus the padded-M16 route.
M13-M15 intentionally keep M16 because a smaller legal tile is unavailable.
The generated M4/M8/M12 objects contain 16/32/48 SVE2 SMMLA instructions and
163/253/371 assembly lines; stack references are 0/2/0 and all three have no
external call or residual dot operation.

The Q8 LHS pack has a full-row specialization for exact M4/M8/M12/M16 inputs.
It retains one row per Triton program and removes the dynamic predicate. For
finite BF16, four K8 slices are compared in their monotonic magnitude encoding
before one horizontal reduction. The row scale still uses the existing
`max(absmax, 1e-8) / 127` contract; because it comes from the same row, the
per-K8 saturation is redundant. On both the development compiler and 3.7.2,
the object has three UMAX, two FMAXNM, no FMINNM, spill, or external call.
Paired direct calls improve by 1.34x at M4/M8/M12/M16 and take only
0.681-0.686x the latency of the same-contract ACLE Neon packer. Every normal
and finite-edge blob matches the independent floating-point/clamped path, and
an exhaustive scalar check covers every finite BF16 magnitude and sign.
Non-full rows deliberately retain the masked floating-point path.

The generated Q8 pack also generalizes over K. At M16, direct Triton/ACLE
ratios are 0.756x for K256, 0.681x for K1024, 0.679x for K3072, and 0.679x for
K4096; all blobs and the following matrix outputs are bit-exact. LLVM fully
unrolls the eight K32 reduction iterations at K256, producing 24 UMAX sites;
larger shapes keep the intended rolled loop with three static UMAX sites.
Both variants remain spill- and call-free, and the codegen audit accepts only
these two proven forms.

This also moves end-to-end prefill, not just the isolated matrix. On
Qwen3-0.6B W8 with an eight-token prompt and one thread pinned to CPU 11, two
off/on process pairs measured 130.76 -> 100.24 ms and 130.50 -> 97.75 ms:
23.3-25.1% lower prefill latency. Each process used 7-9 warmed prefill repeats,
and generated token IDs were identical. `FLAGGEMS_ARM_W8_SHORT_PREFILL=0`
restores the padded-M16 schedule for A/B or emergency fallback.

The Q8 M16 comparison includes both activation preparation and the matrix
launch. The matrix object itself changed from 2399 assembly lines and
343/353 folded spill/reload references to 448 lines and zero spills/reloads.
It retains 64 SVE SMMLA instructions.

The W8 matrix core was also compared directly with KleidiAI v1.24 under its
official asymmetric-LHS N4/K8 packed ABI, excluding both packers and Python.
For M16, Triton/KAI is 1.013x at N=K=1024, 1.015x at N3072/K1024, 0.999x at
N1024/K3072, and 1.013x at N=K=4096. The maximum FP32 absolute difference in
all four runs is 2.39e-7. This puts the generated SMMLA loop at native-library
level across both N-heavy and K-heavy shapes, rather than only ahead of the
retired row-major implementation.

That W8 physical-graph lowering is now also present in the independent
official Triton-CPU 3.7.2 port.  With the lowering disabled, the unchanged
M16/N=K=1024 Triton kernel takes 1944.55 us by direct symbol call and produces
1441 assembly lines, no SMMLA, and 24 folded stack references.  The enabled
port takes 115.45 us, emits 64 SVE SMMLA instructions in 491 lines, and has no
spill/reload, residual dot, or external call: a 16.84x compiler-only gain.
KleidiAI takes 113.41 us in the same alternating harness (Triton/KAI 1.018x).
The port ratios remain 1.017x at N3072/K1024, 1.003x at N1024/K3072, and
1.014x at N=K=4096; maximum FP32 absolute error is 2.38e-7 throughout.  A
port-level regression builds the ordinary graph without importing FlagGems,
requires 64 SMMLA and zero hot stack traffic for K1024, and proves that a
different RHS permutation is rejected by the layout matcher.

The 3.7 production BF16 pipeline has also been measured through a C++ wrapper
that calls only the two generated symbols and reuses the same buffers for the
old/new pair. At N=K=1024, M4/M8/M12/M16 improve from
46.623/74.665/95.769/124.984 us to 46.006/74.077/93.477/122.198 us. The new
pack takes 2.120/4.246/6.318/8.421 us, versus the ACLE packer's
3.098/6.190/9.282/12.372 us. All generated and reference blobs and outputs are
bit-exact. The roughly 246-us Python production measurement is therefore
dominated by allocation/dispatch outside the generated kernels; it is not the
LLVM matrix latency.

The same production JITFunctions now run through the CPU backend in
`libtriton_jit`; a thin Python shim only imports the FlagGems definitions and
does not duplicate a kernel. With the new pack, warmed M4/M8/M12/M16 raw C++
pipelines are 42.69/70.51/92.16/122.04 us and typed pipelines are
45.55/73.41/95.04/124.77 us. A reusable two-pointer/two-i32 CPU ABI fast path
removes libffi from every pack program. A standalone ABI kernel tests this
path, and the focused launch-hooks, CPU-backend, and CPU-JIT tests pass 3/3.
Every Q8 shape is also checked against an independent scalar reference and is
BF16 bit-exact.

Q4 now uses the same `libtriton_jit` integration benchmark.  At
M16/N=K=1024, the typed C++ pipeline is 249.20 us and the raw pipeline is
246.35 us, compared with 245.79 us for direct generated-symbol dispatch.
Generic and raw outputs are BF16-bit-identical; the independent Q4 compiler
regression and KleidiAI harness provide the numerical reference.  Reusable
M4/M8/M12 raw C++ pipelines are 76.82/132.76/189.46 us, while their typed API
latencies are 79.46/135.58/192.36 us; all typed/raw pairs are bit-identical.
The reusable three-pointer/two-i32 and five-pointer CPU ABI fast paths reduce
the raw result
from 264.78 us by avoiding libffi for both grids.  None of these launch paths
contains a Q4 compute implementation.

For an ordinary decode `tl.dot`, Triton represents the packed W8 panel as a
flat load, `[N,4]` reshape, and logical `[4,N]` transpose. Both the development
compiler and the actual Triton-CPU 3.7.2 production port now recognize that
algebraic graph and feed each adjacent sixteen-byte physical slice directly
to SDOT. They no longer materialize the complete transposed weight tile in a
compiler-owned stack buffer. On 3.7.2, the active BN64 production schedule was
already optimized by LLVM: paired GEMV calls remain 26.6745 versus 26.6634 us,
while the object contracts from 326 to 301 lines. This matcher is instead a
large safety improvement for a wide schedule: K1024,N152064,BN128 improves
22620.6 to 7155.8 us (3.16x), and stack load/store sites fall from 215 to 113.
Production deliberately retains spill-free BN32 for this vocabulary shape.

A separate decode kernel now consumes KleidiAI's exact asymmetric-LHS N4/K8
W8 ABI rather than translating it into the FlagGems row pack. This is still an
ordinary Triton graph: loads, reshape/join/permute, `tl.dot`, and the numeric
epilogue remain visible to the compiler. Preserving the K8 activation as a
2x4 value makes LLVM use `LD1R` broadcasts instead of a paired 64-bit load and
two lane-insertion moves. The final pointer-induction/unroll-two object has 16
static SDOT instructions, one ADDP, no stack access, residual dot, or external
compute call. Strict generated-object A/B gives about 1.2% over the flattened
source; direct calls are 78.17 versus 79.64 us at K1024/N3072 and 953.92 versus
1006.51 us at K9728/N2560 against KleidiAI v1.24. Intermediate shapes
K4096/N2560 and K2560/N9728 are 0.991x and 0.946x KleidiAI, respectively.
Packing and Python are excluded, and maximum FP32 absolute error is 2.39e-7.

The corresponding model contract has also been measured without prepacking
the activation. An ordinary Triton BF16-to-qai8dxp pack produces a byte-exact
KAI blob, then the generated pointer kernel stores BF16 directly. Against the
official BF16 pack plus KAI matrix plus FP32-to-BF16 conversion, complete
generated/KAI ratios are 0.992-0.998x at K1024/N3072, 0.962-0.973x at
K2560/N9728, and 0.972-0.973x at K9728/N2560.
All final BF16 values are bit-exact. The generated pack alone is slower, so
only the complete-pipeline numbers support replacement.

A cache-keyed vLLM AOT bundle covers 18 Qwen3 shapes, including the TP=1
fused projection shapes for Qwen3-0.6B, 1.7B and 4B. M1 objects must retain
`16 SDOT + 1 ADDP + 10 LD1R` with no lane insert, stack access, or call. M16
objects must retain 64 SMMLA with no K-loop stack access or call; their MR4
pack must retain nine FCVTNS and zero stack access. The opt-in plugin uses
generated M1/M16 only at the measured one-thread cutoff and installs the
original KleidiAI closure at higher thread counts. Same-layer
custom-op microbenchmarks improve by 2-9% at M16 because the generated route
also avoids the previous OpenMP-region and separate output-conversion
overhead; this is wrapper integration gain, not matrix-only codegen gain.

Real eager vLLM Qwen3-0.6B decode confirms the policy. Generated versus KAI
is 13.838/13.315 tok/s at one thread, but loses at two, four and eight threads
(17.867/18.319, 19.143/19.346 and 19.324/19.900 tok/s). With the default safe
router, the eight-thread `triton_codegen` backend directly installs the KAI
closure and measures 19.815 versus 19.810 tok/s in the same phase. This is a
no-regression fallback, not a codegen speedup.

Direct schedule A/B confirms the production choice. At K1024,N152064, BN32
is 7155-7172 us; BN16 is 7211 us and BN8 is 7172 us. With BN32 fixed, K-loop
unroll 1/4/8 are respectively 1.6%/3.8%/7.9% slower than the active unroll-2
kernel. Every candidate is bit-exact and spill-free, so the rejection is based
on measured bandwidth/loop efficiency rather than an IR-size guess.

The earlier 4.59x BN128 result came from the main development tree, whose old
M1 lowering lacked the 3.7 port's direct-output optimization. It is useful as
a compiler regression case but is not reported as a current production
speedup. An automatic BN128-to-BN32 loop interchange was also tested and
rejected: paired direct calls regressed because it repeated the full K
traversal, despite reducing register pressure. It is not in the active path.

The standard CPU `libdevice.rint` path lowers through MLIR `math.roundeven`
and LLVM's round-even intrinsic. On the Triton-CPU 3.7.2 AArch64 path LLVM
combines round-even and integer conversion into vector `FCVTNS`, with no
SLEEF or runtime call. At K1024 this first change takes the Q8 decode
quantizer from 1.512 to 0.443 us. The active finite-BF16 absmax reduction then
uses three lane-wise integer `UMAX` instructions and two `FMAXNM` operations
instead of seven `FMAXNM` reductions. It takes the same quantizer to 0.349 us,
a further 1.271x gain. The final 211-line port object has four `FCVTNS`, no
external call, and no stack traffic. Q4 uses the same compiler round-even
path. The retired one-row and active M4-panel pack
objects are 477/854 compiled assembly lines on the 3.7.2 port; the latter is
larger because four row bodies are explicit, but both are spill-free. The active
schedule is 2.83x faster than
the original manual-RNE pack and 1.283x faster than the one-row LLVM pack.
The compact decode pack is now 388 compiled assembly lines and its 75-instruction
function body contains three UMAX, eight FCVTNS, no saturation clamp, call, or
stack access. Packed Q4 blobs and Q8 values/scales remain bit-exact.

The same standard operations are implemented in the Triton-CPU 3.7.2 port
used by the production AOT bundle. The optimized K32 loop is used for the
full prefix while a masked floating-point reduction preserves arbitrary
K%16 tails; K1008 and K1024 both pass finite-BF16 edge-pattern and independent
reference checks. A production-bundle audit found that its exported
`_quantize_bf16_w8_kernel` was still compiled from the older FP32-absmax
source even though the runtime FlagGems quantizer had already moved to lane
max. The AOT source now uses the same ordinary Triton K32 graph without
changing its symbol or ABI. Direct latency is 0.500 to 0.349 us; calling the
same BN64 GEMV immediately afterwards gives 27.063 to 26.769 us at
K=N=1024. The smaller complete-pipeline percentage is expected because GEMV
dominates. The production source itself now tests signed zero, subnormals,
the normal boundary, and maximum finite BF16 values; K1008 also exercises the
masked non-K32 suffix.

The bundle cache key now includes the two imported kernel-definition files,
not only their top-level benchmark drivers. This prevents an imported Triton
source edit from silently selecting stale AOT objects. The rebuilt bundle's
22 selected objects pass disassembly gates; every one of its eight quantizer
objects must contain exactly three `UMAX`, two `FMAXNM`, four round-even
conversions, no `FMINNM`, call, or stack reference. Real FlagGems routing was
then checked separately: complete K=N=1024 and K1024/N3072 `Linear.forward`
calls are 1.82x and 1.35x faster than whole-TLE, while fused QKV is 1.23x
faster. These are Python-level route measurements and all outputs are
bit-exact; they are not substituted for the direct-symbol compiler numbers.

The main compiler's single-program Q8 decode lowering now uses the same
finite-BF16 magnitude property before it creates the shared quantized stack
buffer. Its absmax loop changes from five static `FMAXNM` instructions to
three `UMAX`, one horizontal `UMAXV`, and two `FMAXNM`; the 16-SDOT matrix
loop and output contract are unchanged. A same-process alternating-order A/B
improves K=N=1024 by 2.65% and K1024/N3072 by 0.82%. K1008 exercises the
static non-K32 suffix and remains bit-exact on the finite-BF16 edge suite.
`TRITON_CPU_DISABLE_BF16_LANE_MAX=1` retains the old compiler lowering for a
fresh-cache A/B.

The production 3.7 AOT bundle now also selects the separate inline-exp
SwiGLU epilogue. On CPU11 the isolated activation improves 20.014 to 16.392
us (1.221x), and an alternating-order complete MLP comparison improves
171.860 to 168.113 us (1.022x). Four external exp calls become zero. Its eight
stack references are the bounded save/restore of callee-saved vector
registers, not loop-body spills; output remains BF16 bit-exact.

The vLLM Q4 slot no longer routes eager execution through an opaque
`torch.library.custom_op`. It calls the same production router directly and
uses the custom op only while Dynamo captures a graph. At M1/M16,
K=N=1024 a fresh CPU11 run measures custom-op versus eager-slot latency of
206.86 -> 154.01 us and 477.81 -> 430.89 us. The eager slot is only 3.31/2.57
us above direct routing. A
`torch.compile(..., fullgraph=True)` regression confirms that compiled
execution still captures and runs bit-exactly.

The multicore decode comparison is the retired fused-N512 grid versus shared
quantization plus an N64 grid. In the main compiler tree the new
quantizer/matrix objects are 221/275 lines and spill-free; the same quantizer
is 211 lines on the 3.7.2 port. The old fused object is 2003 lines with
151/206 folded spill/reload references.

The fused gate/up+SwiGLU lowering now emits SLEEF's u10 `expf` algorithm as
LLVM arithmetic instead of eight calls to `Sleef_expf4_u10`. At K=1024,
N=3072 it improves 208.23 us to 205.95 us (1.011x); at K=2048,N=6144 it
improves 1070.64 us to 1058.25 us (1.012x). The object grows from 465 to 687
assembly lines, but folded stack references fall from 24 stores/31 loads to
5/5 and external vector-exp calls fall from eight to zero. Twelve numerical
cases covering K=256/512/1024, N=512/1024 and varied activation/scale ranges
remain BF16 bit-exact. `TRITON_CPU_INLINE_EXP_U10=0` restores the external
SLEEF call for comparison.

The same finite-BF16 lane-max helper now feeds this fused lowering. At
K1024/N3072, six N512 programs each avoid the old five-instruction FP32
absmax chain. The strict paired direct-symbol result is 175.918 to 171.470 us
(1.026x), with identical complete SwiGLU BF16 output for normal and finite
edge-pattern activations. The active object keeps 16 static SDOT sites and
zero vector-exp calls; `TRITON_CPU_DISABLE_BF16_LANE_MAX=1` restores only the
old absmax lowering for A/B.

Two end-to-end A/B pairs on Qwen3-0.6B produced identical token IDs. Their
measured improvement ranged from 0.12% to 2.1%, so the stable claim is the
roughly 1.1% fused-MLP microbenchmark result, not the noisier model maximum.

## Weight memory

Q8 keeps two target layouts: an SDOT decode pack and an I8MM prefill pack.
The former third row-major INT8 copy is released after construction. If a
user explicitly caps the KAI prefill route and reaches the compatibility
path, row-major weights are reconstructed lazily from the KAI pack.

The measured KAI prefill path remains faster than the old row-major path at
M64 and above, so the default cap now covers practical sequence lengths.
`FLAGGEMS_ARM_FUSED_PREFILL_MAX_M` can still enforce a smaller cap.

## Reproducible checks

```bash
taskset -c 11 python3 -S benchmarks/audit_arm_q4_q8_codegen.py
taskset -c 11 python3 -S benchmarks/test_flaggems_q4_production_router.py
taskset -c 11 python3 -S benchmarks/flaggems_e2e/validate_w8a8.py
taskset -c 11 python3 -S benchmarks/bench_q8_prefill_production_codegen.py
taskset -c 11 python3 -S benchmarks/bench_q8_lhs_pack_codegen.py
benchmarks/cpp_wrapper/build.sh
taskset -c 11 env TRITON_CACHE_DIR=artifacts/cache-roundeven-repro \
  python3 -S benchmarks/bench_q8_roundeven_codegen.py
taskset -c 6-11 env OMP_NUM_THREADS=6 python3 -S \
  benchmarks/bench_q8_decode_multicore_codegen.py --threads 6
taskset -c 11 env TRITON_CACHE_DIR=artifacts/cache-wide-packed-repro \
  python3 -S benchmarks/bench_bf16_w8a8_wide_split.py \
  --k 1024 --n 1024 --block-n 128 --unroll 2
taskset -c 11 env TRITON_CACHE_DIR=artifacts/cache-kai-w8-repro \
  python3 -S benchmarks/bench_w8_kleidiai_layout_codegen.py \
  --k 9728 --n 2560 --unroll 2 --mode auto --compile-only
/home/cix/venv-fep-e2e/bin/python \
  integrations/pytorch/audit_qwen3_w8_bundle.py \
  artifacts/pytorch-triton-backend/qwen3-0.6b-bf16-w8
TRITON_BACKENDS_IN_TREE=1 \
PYTHONPATH=ports/triton-cpu-3.7.2/python \
taskset -c 11 /home/kevin/venv-int8-clean/bin/python -m pytest -s --tb=short \
  ports/triton-cpu-3.7.2/python/test/unit/cpu/test_sve2_i8mm.py
```

`audit_arm_q4_q8_codegen.py` gates every active Q4/Q8 production
specialization on instruction selection, residual dot removal, stack
pressure, and external operation calls, including fused QKV and gate/up
SwiGLU. The fused MLP gate additionally requires zero SLEEF vector calls,
at most five folded stack stores/loads, and fewer than 750 assembly lines.
The audit also records the two retired high-spill objects so they cannot be
confused with the production route. The fresh result is
`artifacts/q4-q8-codegen-audit-lanemax-helper.json`.

The production-port audit is separate: `audit_qwen3_w8_bundle.py` checks the
22 Triton-CPU 3.7 objects actually selected by the Qwen3 bundle. It requires
zero calls in every active function, exact SDOT counts for every W8 shape,
four FCVTNS/FRINTN operations in each quantizer and inline activation, and
bounded stack use. Quantizers additionally require exactly three UMAX, two
FMAXNM, and zero FMINNM instructions. The bundle build runs this audit
automatically.

For a full-chip latency option, the C++ AOT dispatcher can partition only the
large vocabulary GEMV across persistent workers. The same generated BN32
object improves 7198.62 to 4043.79 us with two Cortex-A720 cores; four and six
cores reach 3872.29/3860.11 us, showing bandwidth saturation. A 32-token Qwen3
run improves 2.03495 to 1.96991 s (15.73 to 16.24 token/s) with identical
tokens. This requires `FLAGGEMS_ARM_W8_AOT_VOCAB_THREADS=2` and a two-core CPU
mask; it is disabled by default and is not counted as a single-core compiler
speedup.
