# vLLM exact-KAI W8 M1/M8/M12/M16 codegen backend

This opt-in backend provides ordinary Triton-CPU 3.7.2 M1/M8/M12/M16 kernels
while retaining KleidiAI's packed ABI and a measured fallback router:

```text
BF16 activation
  -> M1: generated asymmetric MR1 pack + N4/K8 SDOT
  -> M8/M12/M16: generated asymmetric MR4 pack + N4/K8 I8MM
  -> generated BF16 store
```

The weight blob is the existing `qsi8cxp4x8` blob produced at model load.
Decode and prefill use independent conservative one-thread defaults. Real
Qwen3 end-to-end measurements favor KleidiAI above one thread, so
the default router installs the original KleidiAI closure when both generated
routes are above their cutoffs. Override decode and prefill independently with
`FL_CPU_INT8_TRITON_M1_MAX_THREADS` and
`FL_CPU_INT8_TRITON_PREFILL_MAX_THREADS`. The former
`FL_CPU_INT8_TRITON_M16_MAX_THREADS` remains a compatibility alias. Raising
the prefill cutoff does not implicitly enable generated M1. M4 remains on KAI
because it lost on every measured production shape; batch sizes other than
1/8/12/16 also remain on KAI. The C++ dispatcher only loads and calls generated symbols;
it contains no quantization, matrix, or output-conversion arithmetic.

## Build and install

```bash
cd /home/kevin/triton-opt-cpu
build_result="$(bash integrations/vllm/build_qwen3_w8_kai_codegen_bundle.sh)"
printf '%s\n' "${build_result}"
bundle="$(printf '%s\n' "${build_result}" | awk -F= '$1 == "bundle" {print $2}')"
bash integrations/vllm/build_w8_codegen_backend.sh
bash integrations/vllm/apply_vllm_plugin_fl_w8_codegen.sh \
  /path/to/vllm-plugin-FL
```

Select it before starting vLLM:

```bash
export FL_CPU_INT8=1
export FL_CPU_INT8_BACKEND=triton_codegen
export FL_CPU_INT8_TRITON_BUNDLE="${bundle}"
```

The install helper copies the dispatch library into `vllm_fl/ops`. Set
`FL_CPU_INT8_TRITON_LIBRARY` only when loading it from another path. Shapes
absent from the AOT bundle retain the KleidiAI implementation.

The builder publishes a non-overwriting, content-addressed release below an
OS/architecture/lowering/vector-length directory. It copies real object
files, marks them read-only, and does not create machine-local symlinks. At
import and first use,
the production router checks the bundle format, dispatcher ABI, KleidiAI
packing ABI, OS, architecture, required CPU features, SVE vector length,
shape manifest, and SHA-256 digest of every object it loads. Bundle format 2
and dispatcher ABI 2 publish seven objects per shape (126 objects total).
Therefore the
target-specific directory printed by the builder is required; the parent
`qwen3-w8-kai-codegen` directory is not itself a loadable bundle.

## Codegen and correctness gates

The bundle currently covers 18 Qwen3 projection shapes, including every
fused `qkv`, `o`, fused `gate_up`, and `down` shape used at TP=1 by
Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B. It also retains the earlier individual
projection shapes used by the microbenchmarks. Every M1 matrix
object must contain 16 SDOT, one ADDP, ten LD1R, zero lane-insertion moves,
zero stack access, and zero calls. M4/M8/M12/M16 matrix objects must contain
16/32/48/64 SMMLA respectively, no stack access in the K loop, and no calls.
M1/MR4 activation packs must contain 3/9 FCVTNS respectively and no stack
access or calls. The bundle
key includes all Triton sources, the dot lowering, and the exact
`libtriton.so`.

The TP=1 production projection coverage is:

| model | qkv KxN | o KxN | gate_up KxN | down KxN |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1024x4096 | 2048x1024 | 1024x6144 | 3072x1024 |
| Qwen3-1.7B | 2048x4096 | 2048x2048 | 2048x12288 | 6144x2048 |
| Qwen3-4B | 2560x6144 | 4096x2560 | 2560x19456 | 9728x2560 |

The two retained audit shapes K1024/N1024 and K1024/N2048 do not pass the M1
direct native gate (latest generated/KAI 1.070x and 1.034x). The router therefore
keeps their decode calls on KAI while still allowing their independently
measured M8/M12/M16 generated prefill objects. They are not TP=1 production
projection shapes in the table above.

Generated activation blobs are byte-identical to KleidiAI's BF16 qai8dxp
packer. Complete generated and KleidiAI pipelines also produce bit-identical
BF16 output. Direct C++ pipeline measurements on one CIX Cortex-A720 are:

| K x N | generated pack+matrix+BF16 | KleidiAI pack+matrix+BF16 | ratio |
| --- | ---: | ---: | ---: |
| 1024 x 3072 | 80.57-80.98 us | 81.15-81.26 us | 0.992-0.998x |
| 2560 x 9728 | 983.13-990.08 us | 1017.98-1021.92 us | 0.962-0.973x |
| 9728 x 2560 | 982.75-985.20 us | 1010.82-1012.32 us | 0.972-0.973x |

With one PyTorch/OpenMP thread, repeated paired M=1 microbenchmarks through
the same Python custom-op layer used by the plugin give 134.4-135.3 versus
164.0-165.2 us at K1024/N3072,
1085.63 versus 1165.89 us at K2560/N9728, and 1074.74 versus 1148.72 us at
K9728/N2560. The extra gain over the direct-symbol table comes from avoiding
the existing wrapper's OpenMP-region and FP32-output-conversion overhead; it
is not counted as a compiler-only matrix speedup.

A post-audit paired run over all four Qwen3-0.6B production decode shapes
measured the complete plugin operation, including activation packing and the
BF16 store:

| projection K x N | generated us | KleidiAI us | ratio |
| --- | ---: | ---: | ---: |
| fused qkv 1024 x 4096 | 181.81 | 204.14 | 0.891x |
| o 2048 x 1024 | 108.38 | 131.81 | 0.822x |
| fused gate_up 1024 x 6144 | 259.96 | 263.70 | 0.986x |
| down 3072 x 1024 | 149.54 | 175.11 | 0.854x |

The four operations total 699.70 us generated versus 774.76 us KleidiAI per
layer in this microbenchmark. BN8 and alternate unroll schedules were also
tested for the weakest 1024x6144 shape. BN8 was 8.1-19.5% slower than KAI.
The in-process generated-to-generated A/B gate found unroll 1 slower than the
production BN4/unroll-2 schedule by 0.61% at N=4096 and 0.88% at N=6144, so
the production schedule was not changed. The reproducible A/B executable is
`artifacts/bench_w8_kai_matrix_codegen_ab`, built from
`benchmarks/cpp_wrapper/bench_w8_kai_matrix_codegen_ab.cpp`.

The generated M16 pack carries eight vector extrema through K and performs
the horizontal reduction once. Quantized values narrow to int16 before the
zero-point add and clamp. This produces byte-identical KAI activation blobs
for all 65,280 finite BF16 encodings, with zero stack access. Direct C++
M16 measurements on the same CIX core are:

| K x N | generated us | KleidiAI us | ratio |
| --- | ---: | ---: | ---: |
| 1024 x 1024 | 124.88 | 126.79 | 0.985x |
| 1024 x 2048 | 245.77 | 249.53 | 0.985x |
| 1024 x 3072 | 357.60 | 361.90 | 0.988x |
| 1024 x 4096 | 468.63 | 477.88 | 0.981x |
| 2560 x 1024 | 294.87 | 296.29 | 0.995x |
| 2560 x 4096 | 1149.24 | 1146.56 | 1.002x |
| 2560 x 9728 | 2642.87 | 2651.65 | 0.997x |
| 4096 x 2560 | 1343.25 | 1365.70 | 0.984x |
| 9728 x 2560 | 3011.92 | 2988.35 | 1.008x |

At the actual plugin custom-op layer, generated/KleidiAI ratios are 0.913x
for 1024x3072, 0.957x for 2560x4096, 0.959x for 2560x9728, and 0.980x for
9728x2560. The generated route wins there because it also avoids the current
KAI wrapper's OpenMP-region and separate FP32-to-BF16 overhead. All compared
outputs are bit-identical.

The exact-KAI short-prefill extension was evaluated separately before routing.
Across the four Qwen3-0.6B production projections, generated M4 was 13-17%
slower than KAI and is therefore never selected. Generated M8 was 4.4-7.0%
faster and generated M12 was 18-19% faster, with byte-identical activation
packs and BF16 outputs. At plugin level for K1024/N6144, the production router
measured generated/KAI ratios of 0.942x for M8, 0.828x for M12, and 0.919x for
M16; M4 took the direct KAI fallback.

## End-to-end router result

The real vLLM engine was run in eager mode on Qwen3-0.6B with prefix caching
disabled, greedy 32-token generations, one warmup, and three measured rounds.
The generated route is genuinely faster only in the one-thread case:

| threads | KleidiAI median tok/s | forced generated median tok/s | result |
| ---: | ---: | ---: | --- |
| 1 | 13.315 | 13.838 | generated +3.9% |
| 2 | 18.319 | 17.867 | generated -2.5% |
| 4 | 19.346 | 19.143 | generated -1.1% |
| 8 | 19.900 | 19.324 | generated -2.9% |

With the default measured cutoff, the eight-thread `triton_codegen` backend
installs the original KAI closure and gives 19.815 tok/s versus a same-phase
KAI rerun at 19.810 tok/s. This is a safety fallback, not a codegen speedup.
The runner is `benchmarks/bench_vllm_w8_codegen_e2e.py`.
It now records complete token IDs and a deterministic SHA-256 digest, accepts
an expected digest from the comparison backend, and reports runtime route
counters. A post-audit CIX smoke generated the same eight token IDs through
both backends; the codegen run recorded 784 generated M1 calls over seven
decode steps, rather than merely enabling a backend with no generated work.

The ABI-v2 default-router rerun used the same 32-token/one-warmup/three-round
protocol and required the complete token digest from the comparison backend.
The default prompt contains 12 tokens, so each request exercises generated
M12 prefill followed by generated M1 decode. In eager mode, TTFT measured
115.50 ms generated versus 130.44 ms KAI (-11.5%); full request wall time was
2.487 versus 2.518 s (-1.2%), and decode was 13.080 versus 13.001 tok/s. The
run recorded 448 M12 calls, 13,888 M1 calls, and only 112 engine-internal KAI
warm-up calls. Both backends produced digest
`6ed177097af805f0f879510e8eaf8694efe093dc6cf1560e158d2a72989343db`.

The complete non-eager vLLM engine is also validated. Its TTFT measured 80.42
ms generated versus 92.17 ms KAI (-12.7%); full request wall time was 1.732
versus 1.750 s (-1.0%). Both produced digest
`ba930545728f4520f2f21b72726afb468bef7e13c650cbcadd559f70b6ca0204`.
An exact 16-token prompt also exercised M16 in the real engine: eager TTFT
improved by 4.0% and compiled TTFT by 5.5%, with identical tokens and explicit
M16 counters. Eager and compiled
vLLM modes have different deterministic greedy streams in this environment,
so digest equality is required between backends within the same execution
mode rather than across modes.

Run the standalone C API checks with:

```bash
for m in 1 4 8 12 16; do
  /home/kevin/venv-int8-clean/bin/python \
    benchmarks/test_vllm_w8_codegen_backend.py \
    --bundle "${FL_CPU_INT8_TRITON_BUNDLE}" \
    --m "${m}" --k 1024 --n 3072
done
```

Run all bundle deployment gates (18/18 shapes, 126 object hashes/symbol loads,
cross-target rejection, digest rejection, and unsupported-shape KleidiAI
fallback) with:

```bash
/home/kevin/venv-int8-clean/bin/python \
  benchmarks/test_vllm_w8_codegen_bundle.py \
  --bundle "${FL_CPU_INT8_TRITON_BUNDLE}"
```

The compiled route now keeps the M-dependent decision inside the opaque
custom op instead of specializing the Python closure on the warm-up M.
The same dynamic compiled graph has been checked across M=1/4/8/12/16/3/1 on CIX,
with bit-identical output against KleidiAI on every route. Native Apple M4
Pro validation also confirms bit-exact W8 decode/prefill and 16/32/48/64
fixed-width SMMLA instructions for M4/M8/M12/M16. Generated multi-thread
routing remains disabled by the conservative measured cutoffs until an
end-to-end multi-thread win is demonstrated on each target.
