# ARM Triton CPU codegen 优化迭代：1 / 2 / 3

日期：2026-07-31
平台：CIX P1/CD8180，AArch64，Cortex-A720，大核固定到 CPU 11，
SVE2+i8mm+BF16，SVE VL=128。AOT 数据为单线程、预热后的中位数。

## 结论

本轮完成了三个方向：

1. 通用、可配置的 vector loop strip-mine 编译 pass；
2. ordinary Triton attention 的 BF16 dot 到 AArch64 BFDOT lowering；
3. W8 SDOT 显式预取 A/B，以及更有效的 Q/K/V projection fusion。

所有新增高性能路径都保留在编译器可见 IR 中。attention 没有 TLE/raw
leaf；QKV 复用 typed CPU op，但该 op 会展开成 SCF/vector/SDOT，
不会 tail-call 外部 GEMV。

最重要的结果：

| 项目 | 优化前 | 优化后 | 结果 |
|---|---:|---:|---:|
| Attention direct AOT，N=128 | 140.45 us | 112.14 us | **-20.2%** |
| Attention 手写 C runtime | 191.06 us | 112.14 us | Triton **1.70x** |
| QKV synthetic Python，K=1024，N=1024+512+512 | 277.16 us | 158.58 us | **1.75x** |
| Qwen3 QKV direct AOT，K=1024，N=2048+1024+1024 | 132.47 us | 125.40 us | **-5.34%** |
| Qwen3-0.6B，8-token decode | 0.7142 s | 0.6811 s | **+4.85% tok/s** |

Qwen 的 8 个 greedy token 完全一致。pack 所有权转移后，融合模式 RSS
为 3637.5 MiB，未融合为 3647.6 MiB，没有保留第二份 QKV 权重 pack。

## 1. 通用 vector loop strip-mine

新增 pass：

```text
triton-cpu-strip-mine-vector-loops
```

它在 TTCIR 的 SCF/vector 层工作：

- 根据元素位宽、目标 vector bits 和 unroll factor 缩窄 rank-1 vector；
- 同步缩窄 `vector`、`memref`、常量、splat、broadcast；
- 重建循环 step，保持地址步长和循环上界一致；
- 默认拒绝改变浮点 reduction 的结合顺序；
- 只有显式允许 reassociation 时才 strip-mine FP reduction；
- 遇到嵌套 region、动态/不兼容 shape 或无法重建的外部值时安全跳过。

开关：

```bash
TRITON_CPU_STRIP_MINE_VECTOR_LOOPS=1
TRITON_CPU_NATIVE_VECTOR_BITS=128
TRITON_CPU_VECTOR_UNROLL_FACTOR=1
TRITON_CPU_STRIP_MINE_FP_REDUCTIONS=0
```

该 pass 默认关闭，便于逐 kernel allowlist。真实 3584 元素 square kernel
把 `vector<256xf32>` 自动改成 native `vector<4xf32>`，结果 bit-exact，
汇编 84 行、零栈访存。Python 直调从 16.635 us 降到 16.279 us，
约 **2.1%**。

回归测试覆盖：

- elementwise loop 自动缩窄；
- 默认模式保留 `vector<128xf32>` reduction；
- reassociation 模式把 reduction 缩窄到 `vector<4xf32>`；
- 修复了 loop-invariant vector 克隆到新循环之后造成的 dominance bug。

## 2. Attention BF16 dot -> BFDOT

原 ordinary Triton attention 的 QK dot IR 是：

```text
bf16 load
  -> extf fp32
  -> shape_cast / broadcast
  -> mulf vector<1x128xf32>
  -> vector.multi_reduction
```

旧 matcher 只识别：

```text
mulf(bf16, bf16) -> extf -> reduction
```

因此 attention 生成 48 组 `shll/shll2` 和 81 条 FMLA，没有 BFDOT。
新 matcher 可以穿过 `vector.shape_cast` / `vector.broadcast`，识别
“先 extf、后 mulf”的 ordinary Triton 形式，并重建 BF16 输入：

```text
16 x llvm.aarch64.neon.bfdot.v4f32.v8bf16
1  x llvm.aarch64.neon.faddv.f32.v4f32
```

该 lowering 只在 CPU feature 同时包含 NEON、FP 和 BF16 时启用；
没有 FEAT_BF16 的 ARM CPU 不会生成非法指令。可用
`TRITON_CPU_DISABLE_BF16_DOT=1` 做回退/A-B。

汇编变化：

| 指标 | FP32 FMLA lowering | BF16 BFDOT lowering |
|---|---:|---:|
| 汇编行 | 1194 | 951 |
| BFDOT | 0 | 16 |
| FMLA | 81 | 65 |
| 栈帧 | 1136 B | 880 B |
| 栈 load/store | 173 | 113 |

固定大核 direct AOT，三轮中位：

| 路径 | 延迟 |
|---|---:|
| 禁用 BFDOT | 140.45 us |
| 启用 BFDOT | **112.14 us** |
| libtriton-jit wrapper | 112.29 us |
| 现有手写 C runtime | 约 191.06 us |

新 BFDOT 路径比旧 codegen 降低 **20.2%** 延迟，比手写 C runtime
降低约 **41.3%**。输出与本项目原 codegen/C online-softmax 路径保持
一致；相对 ATen 的误差仍来自 online-softmax reduction 顺序，不是
BFDOT 新增误差。

## 3a. W8 显式预取：实现但默认关闭

SDOT lowering 新增了有边界保护的 `memref.prefetch`：

```bash
TRITON_CPU_SDOT_PREFETCH_DISTANCE=<K-step distance>
TRITON_CPU_SDOT_PREFETCH_GROUP_STRIDE=<group stride>
```

编译 hash 包含这两个参数，避免不同预取配置误用同一 cache。默认
distance 为 0，即完全不生成 prefetch。

K=1024、N=1024、BLOCK_N=512 的 direct AOT：

| 配置 | `prfm`/循环体 | 延迟 | 相对默认 |
|---|---:|---:|---:|
| 默认，无 prefetch | 0 | **33.24 us** | 1.000x |
| distance=1, stride=4 | 32 | 36.91 us | 1.110x |
| distance=2, stride=4 | 32 | 36.62 us | 1.102x |
| distance=2, stride=8 | 16 | 35.46 us | 1.067x |
| distance=2, stride=16 | 8 | 35.33 us | 1.063x |

该 block-major 布局已经形成连续权重流，硬件 prefetcher 足够有效；
显式 `prfm` 只增加前端/地址计算开销。因此不把“生成了 prefetch 指令”
包装成性能收益，默认保持关闭。

## 3b. W8 Q/K/V projection fusion

Qwen attention 对同一 activation 顺序执行：

```text
q_proj(x) -> k_proj(x) -> v_proj(x)
```

原来会执行三次 absmax、三次动态 INT8 quantization 和三次 Triton
launch。新 patch 将三个现有 block-major compiler pack 沿 block 维拼接：

```text
one Triton launch
  -> one activation absmax/quantization
  -> Q blocks + K blocks + V blocks
  -> three output views
```

这不是数值近似：每个 output channel 的 SDOT 和 dequant 顺序不变。
decode 输出 bit-exact；M>1/non-BF16 走原 prefill；异常 K/V 调用顺序也会
安全重新执行一次融合投影。

内存处理：

- 融合 pack 创建后释放三个 source decode pack；
- prefill 继续使用原 `_w_int8_kn`；
- unpatch 时从 row-major W8 按需重建三个 pack。

真实 Qwen3-0.6B 形状，direct AOT：

| 投影 | 延迟 |
|---|---:|
| Q: K=1024, N=2048 | 65.85 us |
| K: K=1024, N=1024 | 33.31 us |
| V: K=1024, N=1024 | 33.31 us |
| 三次合计 | 132.47 us |
| 融合 K=1024, N=4096 | **125.40 us** |
| 融合 libtriton-jit wrapper | 125.53 us |

真实模型固定条件：

| 模式 | 8-token 时间 | tok/s | token |
|---|---:|---:|---|
| BFDOT attention + 原 Q/K/V | 0.71417 s | 11.2019 | reference |
| BFDOT attention + fused QKV | **0.68112 s** | **11.7453** | identical |

端到端吞吐提升 **4.85%**。

## 最新 decode profile

QKV+BFDOT 配置的一次同步 decode step：

- total self CPU：65.450 ms
- Triton range：57.384 ms
- 时间覆盖率：87.68%

| Triton range | Calls | Self time | Total share |
|---|---:|---:|---:|
| W8 decode SDOT | 57 | 21.735 ms | 33.21% |
| Fused W8 MLP | 28 | 12.992 ms | 19.85% |
| Fused W8 QKV | 28 | 9.099 ms | 13.90% |
| RMSNorm | 85 | 6.744 ms | 10.30% |
| Decode attention | 28 | 2.357 ms | 3.60% |
| RoPE Q/K | 28 | 2.140 ms | 3.27% |
| Fused add + RMSNorm | 28 | 2.134 ms | 3.26% |
| Vocabulary argmax | 1 | 0.183 ms | 0.28% |

W8-family launch 从 141 个普通 W8 + 28 个 fused MLP，变成 57 个普通
W8 + 28 个 QKV + 28 个 fused MLP，少 **56 次 launch/decode**。

覆盖率百分比从约 88% 小幅下降不是覆盖倒退，而是被覆盖的重复 Q/K/V
工作被删除：profile total 同时降到了 65.45 ms。这里应同时报告时间、
调用数和语义算子覆盖，不能单独最大化“耗时覆盖率”。

## 验证

通过：

- `libtriton.so` 全量增量构建；
- strip-mine MLIR parse/verifier，两种 reduction 模式；
- BFDOT MLIR regression：16 BFDOT + 1 FADDV，无 multi_reduction；
- FlagGems W8A8 focused validation：
  - prefill SVE2 i8mm；
  - fused MLP bit-exact；
  - QKV decode bit-exact、prefill fallback、pack transfer/unpatch；
  - RMSNorm / fused norm / RoPE / argmax；
  - attention 误差界限；
- Qwen3-0.6B 真实 generate，greedy token 一致。

## ARM 泛化边界

- strip-mine 是 target-vector-bits 驱动，不绑定 CIX；当前实现只处理
  固定长度 rank-1 vector，复杂多维/动态/scalable vector 安全跳过。
- BFDOT 是 128-bit NEON BF16 lowering，可用于报告 FEAT_BF16 的
  AArch64（Linux `/proc/cpuinfo` 或 macOS `sysctl`）；它不要求 SVE2。
- SVE2 i8mm 仍只在检测到 SVE2+i8mm 且当前实现支持的 VL 上启用。
- QKV fusion 泛化到相同 K、相同 compiler block size、Q→K→V 调用顺序
  的 W8 attention 模块；其他模型自动不 patch。
- 显式 SDOT prefetch 保留为实验 knob，不作为默认优化。
