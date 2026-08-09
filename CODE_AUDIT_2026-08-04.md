# Arm Q4/Q8 code audit — 2026-08-04

## Scope and source of truth

This audit covers the active Triton-CPU 3.7.2 tree at
`ports/triton-cpu-3.7.2`, the Arm Q4/Q8 kernels in
`third_party/FlagGems`, and the vLLM W8 integration in `integrations/vllm`.
The older Triton tree at the workspace root is not used for the results below.

## Correctness and deployment defects fixed

1. The Q4 dot-pair matcher checked the operation graph and types but did not
   prove the nibble mask, shift, or zero points. It could therefore replace a
   lookalike expression with hard-coded Q4 semantics. The matcher now requires
   mask 15, shift 4, and both zero points 8. Wrong-mask, wrong-shift, and
   wrong-zero-point MLIR tests retain generic semantics.
2. Packed Q4 and W8 loop rewrites rebuilt the dot result and erased the old
   loop without rejecting unrelated stores, atomics, calls, or unknown
   effects. Whole-loop replacement now accepts only recursively read-only or
   effect-free loops. A W8 loop containing a store preserves that store and
   uses only the safe single-dot fallback.
3. The vLLM W8 Python closure could specialize on the warm-up M while Dynamo
   traced it. Compiled graphs now always enter one opaque custom op, whose
   runtime body selects generated M1/M8/M12/M16 or KleidiAI from the live M.
   One fullgraph dynamic compile has been reused across
   M=1/4/8/12/16/3/1 with bit-identical output. M4 deliberately remains KAI.
4. Decode and prefill shared one thread cutoff. They now have independent
   `FL_CPU_INT8_TRITON_M1_MAX_THREADS` and
   `FL_CPU_INT8_TRITON_PREFILL_MAX_THREADS` policies.
5. Shapes satisfying KleidiAI's K%8 contract but not the generated K%32/N%4
   contract previously fell through to the original BF16 loader. They now
   retain W8 KleidiAI; generated code is only an additional specialization.
6. The old AOT bundle used machine-local absolute symlinks, had an incomplete
   cache key, and did not validate target ISA, SVE vector length, dispatcher
   ABI, KAI packing ABI, manifest contents, or object integrity. The new
   builder publishes copied, read-only objects in an atomic target-specific,
   content-addressed directory. Production loading checks all of those
   contracts and SHA-256 before `dlopen`.
7. Kernel compilation and bundle packaging now use separate keys. Changes to
   packaging reuse compiled kernels, while changes to Triton kernels, target
   detection, lowering passes, compiler mode, or `libtriton.so` invalidate the
   compile cache. Reusing an existing release revalidates all 126 objects.
8. The Q4 public router now rejects zero/invalid dimensions, scalar or empty
   activations, non-floating activations, non-UINT8/non-contiguous RHS blobs,
   wrong byte counts, non-finite weights/scales, and values outside signed
   Q4 range before launching a kernel.
9. Relevant Q4/Q8 tests now default to the active 3.7.2 port. This prevents an
   apparently successful run from silently importing the older root tree.

## Verification completed on CIX

- Full active 3.7.2 build completed.
- MLIR FileCheck gates passed for Q4 constants, SVE/fixed dot lowering, W8
  unroll chains, and side-effect preservation.
- Targeted CPU pytest: 13 passed.
- The production Q4 router passed M=1,2,3,4,7,8,12,15,16,17,20,24,28,31,32,47,48,
  its input-negative suite, and one dynamic graph across M=20/1/7/32/2/12/47/3.
- The combined active codegen audit reports Q4 M4/M8/M12/M16 =
  16/32/48/64 SMMLA, Q4 decode = 8 SDOT + 1 ADDP, Q8 M16 = 64 SMMLA,
  Q8 M4/M8/M12 = 16/32/48 SMMLA, Q8 decode = 32 SDOT, no external compute
  calls, and no residual `triton_cpu.dot`.
- Comprehensive W8/FlagGems validation passed eight decode shapes, thirteen
  prefill shapes, all 65,280 finite BF16 quantizer encodings, fused QKV, fused
  MLP/SwiGLU, RMSNorm, RoPE, argmax, and the default-disabled attention
  tolerance check.
- Final CIX W8 ABI-v2 release contains 18 shapes. All 126 object hashes and exported
  symbols loaded; cross-OS metadata and wrong-digest negative gates rejected.
- C API M1/M4/M8/M12/M16 at K=1024/N=3072 are BF16 bit-exact against the
  reference. Production routing uses generated M1/M8/M12/M16 and KAI M4.
- The plugin-layer M16 K=N=1024 check remained bit-exact and measured
  262.43 us generated versus 274.66 us KleidiAI in the short post-audit run.
- Real vLLM Qwen3-0.6B eager smoke produced identical token IDs for generated
  and KleidiAI backends. The generated run recorded 112 W8 linears, four
  generated shapes, and 784 generated M1 calls for seven decode steps; it was
  not a backend-name-only pass.
- The full eager 32-token runner completed one warm-up and three measured
  rounds at 1/2/4/8 threads with exact backend-to-backend token digests. Above
  one thread, zero generated shapes were loaded and the direct KleidiAI
  closure was installed, as the production cutoff requires.
- With the ABI-v2 bundle, the default 12-token prompt exercises M12 prefill and
  M1 decode. Eager TTFT measured 115.50 versus 130.44 ms (-11.5%) and full
  request wall time 2.487 versus 2.518 s (-1.2%). The generated run recorded
  448 M12 calls, 13,888 M1 calls, and 112 engine warm-up KAI calls.
- A real non-eager/compiled vLLM engine completed the same protocol. TTFT
  measured 80.42 versus 92.17 ms (-12.7%) and full request wall time 1.732
  versus 1.750 s (-1.0%), with identical 32-token output. The
  `triton_kernels.matmul_ogs` log line is an unrelated optional MoE-package
  namespace warning; it does not prevent vLLM compilation or W8 execution.
- A real exact-16-token prompt recorded explicit M16 calls. Generated eager
  TTFT improved by 4.0% and compiled TTFT by 5.5%, with identical output.
- Paired plugin microbenchmarks over the four Qwen3-0.6B production shapes
  measured 699.70 us generated versus 774.76 us KleidiAI per layer, with every
  output bit-identical. A new in-process generated-matrix A/B harness rejected
  BN8 and unroll-1 candidates and retained the spill-free BN4/unroll-2
  production schedule.
- Exact-KAI M4/M8/M12 short-prefill objects generate 16/32/48 SMMLA with zero
  stack traffic and zero external calls. Across the four Qwen3-0.6B
  projections, generated M4 lost 13-17%, M8 won 4.4-7.0%, and M12 won 18-19%.
  The router therefore enables only M8/M12 (plus the existing M16) and has an
  explicit test proving M4 still uses KAI even though its object is audited.

## Eager P0 codegen follow-up

- The first fused add + RMSNorm candidate was rejected because it normalized
  a BF16-rounded residual. Current vLLM stores a BF16 residual but computes
  variance from the unrounded FP32 sum. The replacement is now a two-pass
  ordinary Triton kernel that reproduces those two outputs independently.
- The first RoPE candidate was also rejected. It rounded each product to BF16
  before the add/sub, whereas current vLLM evaluates the complete expression
  in FP32 and rounds only the result. Random native-op comparison exposed
  hundreds of differing elements. The corrected kernel is bit-exact and now
  performs position/cache-row selection inside generated code, avoiding
  Python `item()` and slicing overhead.
- Calling these small kernels through `torch.library.custom_op` was itself
  costly in eager mode. The production eager patch calls the validated AOT
  function pointer directly. The C++ launcher contains only `dlopen`, symbol
  lookup, grid iteration, and error handling; it has no operator arithmetic.
- Compiled mode explicitly keeps the original graph expressions. End-to-end
  testing showed that opaque P0 calls disrupt profitable Inductor fusion. The
  compiled validation records zero P0 calls and retains its original token
  digest.
- The P0 release now contains six ordinary-Triton objects: three RMSNorm
  shapes, fused RMSNorm, RoPE, and SwiGLU. RMSNorm/fused/RoPE have no stack
  access or external calls. SwiGLU has no external calls and only callee-save
  stack traffic outside the hot loop.
- Five-round eight-thread eager testing produced identical tokens and improved
  median decode from 18.176 to 23.236 tok/s versus the same direct-KleidiAI W8
  route. Three profiled requests recorded 15,717 generated P0 calls, zero
  fallback, and a 5.390% full-operator share of total inference time.
- At eight threads W8 remains on the measured KleidiAI cutoff. This is
  intentional: forcing generated W8 raises Triton coverage but lowers total
  throughput. Coverage and performance claims now state this distinction.

## Remaining work and risk

1. Convert the dirty nested Triton and FlagGems worktrees into reviewable
   commits or a patch series. Generated artifacts, backup dylibs, caches, and
   historical reports must not enter an upstream change.
2. Exercise the compiled production router with Qwen3-1.7B and 4B if CIX
   memory and checkpoint availability permit. Their codegen objects are in the
   bundle, but the current real-engine result covers Qwen3-0.6B.
3. Extend real-engine prefill coverage beyond the exact M8/M12/M16 cases. The
   current router conservatively keeps all other M on KAI; larger prompts and
   request batching need separate measurements before adding more generated
   shapes.
4. The new target-specific Apple M4 bundle is intentionally deferred at the
   user's request. Earlier native Mach-O validation remains separate from the
   current CIX release result.
5. Keep attention disabled by default. It passes the current tolerance but is
   not bit-exact and is unrelated to the Q4/Q8 deployment requirement.

The active CIX Q4/Q8 paths are now protected against the identified
miscompiles and deployment mismatches. Upstream readiness is primarily a
change-isolation and cross-target validation task, not an unresolved CIX
kernel-codegen defect.
