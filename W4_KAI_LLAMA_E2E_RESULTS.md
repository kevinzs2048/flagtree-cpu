# KAI-compatible Triton W4 in Qwen3-4B

## Outcome

The physical-microtile W4 experiment is now connected to the real llama.cpp
Qwen3-4B decode path for both Q4_0 x Q8_0 and Q4_1 x Q8_1.  This is ordinary
Triton/LLVM code generation, not a TLE_raw wrapper around KleidiAI or a
hand-written runtime GEMV.  Both generated inner loops contain eight SDOTs,
one ADDP, fixed-point `SCVTF #4`, no stack spill, and no external compute call.

The first table is a single-core diagnostic used to correlate whole-model
changes with direct operator measurements.  It is not the throughput of the
full CIX P1 processor.  Qwen3-4B Q4_0 decode is pinned to Cortex-A720 CPU 0:

| Backend | 16-token latency | Throughput |
| --- | ---: | ---: |
| Native llama.cpp Arm repack | 3423.68 ms | 4.673 tok/s |
| Previous Triton W4 layout | 3319.65 ms | 4.820 tok/s |
| KleidiAI v1.24 | 2890.31-2891.08 ms | 5.534-5.536 tok/s |
| KAI-style Triton W4 | 2805.53 ms | 5.703 tok/s |

These figures are means of five 16-token decode repetitions; model loading is
excluded by `llama-bench`.  The final row includes the new Q4_1 microtile;
its previous Triton implementation was still the older dot-ready layout.

## Multi-core device throughput

The CIX P1 has eight Cortex-A720 cores and four slower Cortex-A520 cores.
Using only selected A720 cores gives the following 32-token decode results;
each entry is the mean of three repetitions with strict affinity:

| A720 workers | CPU mask | Native llama.cpp | KleidiAI | Triton default |
| ---: | --- | ---: | ---: | ---: |
| 4 | 0,1,10,11 | 14.290 tok/s | 13.979 tok/s | 14.859 tok/s |
| 6 | 0,1,6,7,10,11 | 15.273 tok/s | 15.054 tok/s | 16.045 tok/s |
| 8 | 0,1,6-11 | 15.573 tok/s | 15.039 tok/s | 15.599 tok/s |

The generated path is 3.98%, 5.05%, and 0.17% faster than native at 4, 6,
and 8 A720 workers in these runs.  Against the independently built KleidiAI
backend it is 6.29%, 6.58%, and 3.73% faster.  These are now default-routing
results, not forced experiments.  The earlier 8-core loss was recovered by
removing the per-projection global cache mutex after first use and using
fine-grained dynamic N-range scheduling at eight workers.

KleidiAI with all 12 cores, including four A520s, reaches only 10.887 tok/s.
The heterogeneous slow-core tail makes it worse than the eight-A720 result.
The appropriate whole-device result for this model is therefore about
15-16 tok/s, while the earlier 5.5-5.7 tok/s numbers intentionally measure
one core.

## Production data path

The loader converts each canonical ggml Q4_0 block directly into KAI's final
`qsi4c32p4x8` physical layout.  One K32 group for four output channels is 72
bytes: four FP16 scales followed by four 16-byte signed-nibble vectors.  The
nibbles are transformed with XOR `0x88` during model loading, not decode.

ggml's canonical Q8_0 activation already has the required per-K32 layout: one
FP16 scale followed by 32 signed int8 values.  The new decode path therefore
passes those blocks directly to the generated function.  The old path copied
activation values and expanded scales before every projection.

The previous Triton weight cache used 80 bytes per K32 by four outputs.  The
new cache uses 72 bytes, a 10% reduction.  Observed peak RSS fell from
6,940,808 KiB to 6,724,984 KiB, or about 210.8 MiB, for the end-to-end test.
The native tensors remain resident for fallback in both measurements.

Q4_1 uses a project-specific extension of that microtile: four FP16 scales,
four FP16 minimums, and the same 64 nibble bytes, or 80 bytes per K32/N4.
It is deliberately called KAI-style rather than a KleidiAI format because
KleidiAI does not define this Q4_1 ABI.  Canonical ggml Q8_1 blocks are passed
directly, so decode no longer expands activation bytes and scales.

Set `GGML_TRITON_W4_LEGACY_LAYOUT=1` to select the previous layout for a
controlled A/B comparison.  If the KAI-layout generated symbol is absent from
a shape bundle, the adapter also falls back to the previous kernel.

## Same-blob microbenchmark

Both symbols below receive the same 64-byte-aligned blobs produced by the
official KAI packers.  Packing, Python, and launch-framework overhead are
excluded.  Outputs are bit-exact.

| K x N | Triton | KleidiAI v1.24 | Triton / KAI |
| --- | ---: | ---: | ---: |
| 2560 x 1024 | 91.382 us | 90.801 us | 1.006x |
| 2560 x 4096 | 364.403 us | 361.781 us | 1.007x |
| 4096 x 2560 | 357.412 us | 359.230 us | 0.995x |
| 9728 x 2560 | 848.742 us | 850.576 us | 0.998x |

The result is parity across every production Qwen shape: the largest measured
gap is 0.8%, and neither implementation wins consistently.

## Production adapter microbenchmark

This benchmark starts from canonical ggml Q4_0 weights and Q8_0 activations,
then calls the same cached adapter used by llama.cpp.  The activation prepare
operation in the new path is a pointer assignment; the old path performs an
unpack.  Checksums match exactly between layouts.

| K x N | Previous launch | New launch | Previous prepare | New prepare |
| --- | ---: | ---: | ---: | ---: |
| 2560 x 1024 | 115.016 us | 91.142 us | 0.097 us | 0.017 us |
| 2560 x 4096 | 465.341 us | 369.397 us | 0.097 us | 0.017 us |
| 2560 x 9728 | 1089.510 us | 864.232 us | 0.097 us | 0.017 us |
| 4096 x 2560 | 458.764 us | 360.845 us | 0.142 us | 0.017 us |
| 9728 x 2560 | 1081.440 us | 849.835 us | 0.313 us | 0.017 us |

## End-to-end operator profile

The following is one 16-token decode run with
`GGML_TRITON_W4_PROFILE=1`.  The call counts cover all Q4_0/Q4_1 projections
routed through Triton.

| Shape | Calls | Previous mean | New mean | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 2560 x 1024 | 1224 | 117.570 us | 95.471 us | 18.8% |
| Q4_0 2560 x 4096 | 612 | 461.227 us | 370.979 us | 19.6% |
| Q4_0 2560 x 9728 | 1224 | 1088.381 us | 871.921 us | 19.9% |
| Q4_0 4096 x 2560 | 612 | 459.124 us | 365.953 us | 20.3% |
| Q4_0 9728 x 2560 | 544 | 1079.376 us | 854.058 us | 20.9% |
| Q4_1 9728 x 2560 | 68 | 1271.453 us | 1016.959 us | 20.0% |

Total profiled Q4_0 time is about 2098.72 ms, while the Q4_1 total falls from
about 86.46 ms to 69.15 ms.  The counters include one warmup token in addition
to the 16 timed tokens.  After multiplying the 17-token operator total by
16/17, all profiled W4 projections account for approximately 72.7% of the
measured 16-token wall time.  This is an operator-timer estimate, not a
whole-program sampling profile.

Native and generated paths produce the same deterministic eight-token greedy
text.  Standalone adapter tests report `max_abs=0` for Q4_0 K=2560,N=1024 and
`max_abs=1.73897e-05` for Q4_1 K=9728,N=2560.

## Reproduction

Build the generated shape bundle, adapter, and llama.cpp benchmark:

```bash
bash integrations/llama.cpp/build_qwen3_4b_w4_bundle.sh
bash integrations/llama.cpp/build.sh
PATH=/home/cix/venv-fep-e2e/bin:$PATH cmake --build \
  third_party/llama.cpp-w4/build-triton-w4 -j2 --target llama-bench
```

Run the end-to-end benchmark:

```bash
GGML_TRITON_W4_BUNDLE="$PWD/artifacts/llama-triton-backend/qwen3-4b-w4" \
taskset -c 0 third_party/llama.cpp-w4/build-triton-w4/bin/llama-bench \
  -m /home/cix/models/Qwen3-4B/Qwen3-4B-Q4_0.gguf \
  -p 0 -n 16 -r 5 -t 1 -C 0x1 --cpu-strict 1 -o json
```

The compiler/code-generation experiment is in
`benchmarks/bench_w4_kleidiai_layout_codegen.py`; the strict same-blob harness
is `benchmarks/cpp_wrapper/bench_w4_kleidiai_layout.cpp`; and the production
adapter is `integrations/llama.cpp/triton_w4_backend.cpp`.
