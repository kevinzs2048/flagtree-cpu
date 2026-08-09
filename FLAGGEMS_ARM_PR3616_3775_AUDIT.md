# FlagGems Arm PR #3616 / #3775 代码与性能审计

## 结论

这两个 PR 不是两套独立实现。`cf5a8bcaa`（#3616）和
`cd5ab198d`（#3775）之间，Arm 算子主体没有变化；后者主要是面向
5.3 分支重新接入，并调整 runtime 初始化与少量注册配置。#3775 共增加
55 个文件、约 10,325 行，因此两者应作为同一批 Arm 实现审计。

原实现不能因为“写成 Triton”就直接替换 ATen。新 Triton 3.7 CPU
编译器本身也不会自动修复 GPU 风格的超宽 tile、两阶段 reduction、每元素
mask 和过深静态展开。干净 PR 基线中，`topk/cumsum/min/index_select/gather`
等算子比 ATen 慢 2.8 倍至数千倍；`sort/isin` 甚至无法在实用时间内完成
编译。默认全量注册这些实现是不安全的。

有效方法是同时改变 kernel schedule 和 CPU lowering：使用受寄存器预算约束
的 rolled loop、将无 mask 主循环和尾部拆开、让 reduction 只携带标量状态，
并针对量化 GEMV 声明 packed layout。按这一原则重写后，一批普通 Triton
算子已经稳定快于 ATen；Qwen3.5 的 RMSNorm、gated RMSNorm 和 causal conv
也已去掉粗粒度 TLE。仍慢于手写 C 的 gated-delta 保留 TLE 自动快路，不能
为了“纯 codegen”强行制造性能回退。

## 审计方法

- 硬件：CIX P1 CD8180，测试进程固定在 Cortex-A720 大核。
- 编译器：新的 Triton/FlagTree CPU 3.7 路径；旧 TLE 对照使用原 3.3
  前端和同一 `libTritonCPURuntime.so`。
- 粒度：直接调用算子函数的 microbenchmark，独立 ATen reference，预热后
  多批次中位数。
- 线程：重点报告单核，并用 8 核和 1M/4M 元素验证 schedule 泛化。
- 代码生成审计：检查 LLIR 外部调用、AArch64 汇编、静态 SDOT/ADDP、栈访问
  和相同 packed blob 的输出。

## 原 PR 基线

以下是干净 #3775 checkout 在新 3.7 编译器上的单核结果。比值小于 1 才是
FlagGems 更快。

| 算子/形状 | ATen | 原 PR | PR/ATen | 判断 |
| --- | ---: | ---: | ---: | --- |
| argmax, BF16 vocab 151936 | 823.45 us | 374.52 us | 0.455x | 可优化后保留 |
| masked_fill, BF16 131072 | 505.93 us | 229.00 us | 0.453x | 可优化后保留 |
| sub, BF16 131072 | 233.93 us | 452.13 us | 1.933x | 回退 |
| div scalar, BF16 131072 | 130.28 us | 228.10 us | 1.751x | 回退 |
| pow square, BF16 131072 | 105.19 us | 208.17 us | 1.979x | 回退 |
| lt, BF16 131072 | 87.42 us | 363.47 us | 4.158x | 回退 |
| topk, vocab, k=50 | 269.60 us | 1527.46 us | 5.666x | 禁止默认替换 |
| cumsum, vocab | 142.03 us | 593.26 us | 4.177x | 禁止默认替换 |
| min, BF16 vocab | 30.48 us | 2639.62 us | 86.59x | 禁止默认替换 |
| index_select, BF16 | 3.39 us | 12371.50 us | 3647x | 禁止默认替换 |
| gather, BF16 | 258.75 us | 732.92 us | 2.833x | 禁止默认替换 |
| scatter, F32 | 63.70 us | 304.60 us | 4.782x | 禁止默认替换 |
| bmm, BF16 1x32x128x128 | 161.98 us | 115.04 us | 0.710x | 仅该形状获益 |
| mm, BF16 M=1, 2560x4096 | 26844 us | 16440 us | 0.612x | decode 路由可保留 |
| addmm, 同一 M=1 形状 | 26691 us | 14454 us | 0.542x | 仅该形状获益 |

`where` 在 3.7 下还因旧 element-type API 不能编译。`sort(16x1024)` 超过
180 秒未完成，`isin(131072, 512)` 超过 80 秒未产出结果。新版编译器没有
自动改变这些结论。

## 逐文件覆盖

55 个变更文件已按实际职责全部归类，不能把 wiring 文件也算成“新增高性能
算子”：

| 文件组 | PR 内容 | 当前结论 |
| --- | --- | --- |
| `config.py`、runtime/common/libentry、Arm `__init__`、`tune_configs.yaml` | backend 注册、设备与配置 | 基础设施，不计性能覆盖 |
| `ops/addmm,bmm,cumsum,exponential_,full,gather,index_select,isin,lt,min,multinomial,quantile,scatter,sort,topk` | 普通 Triton | 无跨形状胜率依据或已实测回退，默认禁用 |
| `ops/all,any,argmax,div,masked_fill,pow,sub,where` | 原 GPU 风格普通 Triton | 已有 register-bounded 重写；只在大 tensor/指定 vocab 形状直接调用，不做全局 ATen override |
| `ops/mm,attention` | BF16 `tl.dot`/Flash Attention | `mm` 仅 M=1 显式 opt-in；attention 有 ATen/C/codegen crossover router |
| `ops/int_mm,quantized_linear_dynamic` | Q8 动态量化 GEMM/GEMV | 保留 Q8，decode 走普通 quantizer+SDOT，prefill 走 KAI-layout I8MM；旧 TLE 只作显式诊断 |
| `ops/rms_norm,rope,silu_and_mul` | FlagGems fused API 适配 | RMS/RoPE 使用普通 codegen；standalone SwiGLU 仅在大于实测阈值时选择普通 codegen |
| `fused/fused_add_rms_norm.py`、Qwen3 RMS/RoPE/layernorm patch | 模型融合 | 普通 Triton/AOT codegen；patch 文件本身只负责接线 |
| Qwen3 MLP patch | 量化 gate/up/down 接线 | 无 AOT 时也走普通 quantizer、joined SDOT、inline-SwiGLU；旧 `fused_mlp` TLE 不在 active router |
| Qwen3.5 RMS、gated RMS、conv patch | 原 TLE runtime | 已换成普通 Triton codegen |
| Qwen3.5 attention、gated-delta patch | 原 TLE runtime | 按实测长度/形状选择 ordinary codegen 或 C runtime，不伪装成纯 codegen |
| `fused/__init__`、`patch_llama_arch.py`、`int8/__init__,replace,quantize_live` | patch/量化替换管理 | wiring，不计 kernel 数量；原 PR 只有 W8，没有 Q4 |

删除的旧 `ops/add.py` 和 `ops/gelu.py` 也已检查：PR 没有用新的 Arm
codegen 替代它们，而是退回公共 FlagGems 路径。因此不能把这两项记成 Arm
新增覆盖。当前 W8 Linear、QKV 和 MLP 的 active router 已不依赖 TLE；遗留
`_tle_*` 入口只供旧开发树显式诊断。仍会按实测使用 runtime 的是
gated-delta、attention crossover 与少数兼容路径，不能把它们计入纯
codegen 覆盖。

语义审计还发现原 PR 的 `mm` 小形状和 `pow` 小 tensor 路径通过
`detach().numpy()` 计算，再 `from_numpy()` 返回。这会丢失 autograd 关系，
也不是可泛化的 Triton codegen；`min(dim=...)` 同样依赖 NumPy。当前默认
策略不会注册这些路径，后续若要恢复必须使用保留语义的 dispatcher fallback，
不能把 NumPy 绕路算成 Arm 编译器优化。

## 重写后的普通 Triton 算子

131072 元素 BF16/BOOL，最终单核结果：

| 算子 | ATen | 新 codegen | codegen/ATen |
| --- | ---: | ---: | ---: |
| argmax vocab | 823.38 us | 113.82 us | 0.138x |
| masked_fill | 502.46 us | 52.32 us | 0.104x |
| where tensor/tensor | 438.44 us | 137.34 us | 0.313x |
| where scalar/tensor | 438.12 us | 170.92 us | 0.390x |
| sub | 233.79 us | 97.37 us | 0.416x |
| div scalar | 130.63 us | 83.59 us | 0.640x |
| pow square | 105.06 us | 68.02 us | 0.647x |
| all | 78.17 us | 44.98 us | 0.575x |
| any | 82.11 us | 34.57 us | 0.421x |
| lt | 87.28 us | 84.17 us | 0.964x |

8 核下 `masked_fill/where/sub/div/pow/all/any` 的比值分别为
0.225/0.736/0.588/0.615/0.940/0.512/0.423。`lt` 变为 1.187x，说明它
只有噪声范围内的单核优势，因此已从默认注册中移除。

在 1,048,576 和 4,194,304 元素上也验证了多 program strided 路径；上述
保留算子没有随着数据量增加反转。核心代码生成特征是：

- 主循环不带 mask，只在尾部处理 mask；
- logical tile 被限制在目标向量宽度和寄存器预算内；
- `all/any` 每个 tile 先归约成一个标量，再作为 loop-carried state；
- `all/any` 无栈访问、无外部调用；elementwise 热循环没有寄存器 spill。

但“大形状快”不等于可以全局接管 ATen。补充的单核 size sweep 显示固定
launcher/分配成本在小 tensor 上占主导：

| 元素数 | masked_fill | where | where-scalar | sub | div-scalar | pow-square | all | any | lt |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 9.22x | 49.10x | 52.54x | 18.90x | 6.62x | 14.19x | 15.34x | 14.49x | 27.21x |
| 4096 | 2.94x | 15.70x | 21.78x | 3.28x | 3.10x | 5.09x | 5.45x | 5.10x | 10.21x |
| 65536 | 0.159x | 0.509x | 0.725x | 0.527x | 0.790x | 0.901x | 0.794x | 0.623x | 2.977x |
| 131072 | 0.113x | 0.268x | 0.374x | 0.451x | 0.650x | 0.579x | 0.512x | 0.391x | 2.583x |

这里比值仍是 codegen/ATen，小于 1 才更快。结论是这些 kernel 的循环
codegen 已经有效，但 FlagGems 当前 registrar 缺少不递归的 per-shape ATen
fallback，不能直接全局注册。它们现在保留为显式调用/AOT 融合路径，并全部
加入默认 denylist；后续若加入 dispatcher-level threshold，可从约 64K 的
实测 crossover 开始，而不能拍脑袋设阈值。

## 默认注册策略

无参数注册现在只保留内部有可靠 fallback 的 fused add+RMSNorm、RoPE API
兼容和 attention router。全局 `aten::argmax`、`aten::mm`、standalone
SwiGLU 以及两个 Q8 namespace hook 都改成显式 opt-in。原因不是放弃 Q8，
而是原 PR 的 `quantized_linear_dynamic` 假定 N%64，`_int_mm` 的 M=1 fallback
还假定 K%4；全局拦截任意合法 ATen 形状会越界或产生非等价量化语义。模型
Q8 替换器验证 packed ABI/K/N 后仍会显式启用它们，并优先使用本次验证的
KAI-layout codegen。模型接线也可直接调用 vocab argmax 和 M=1 mm，不把
decode 局部胜利扩成全局 ATen 拦截。

以下实现仍保留源码用于研究，但加入 `CUSTOMIZED_UNUSED_OPS`，不会随
FlagGems 默认替换 ATen：`addmm/addmm_out/bmm/cumsum/exponential_/full/`
`gather/index_select/isin/lt/min/multinomial/quantile/scatter/sort/topk`，
以及上表中只有大 tensor 获益的 elementwise/reduction overload。
其中 `bmm/addmm` 在个别 decode 形状获益，但当前 wrapper 没有足够严格的
形状 fallback，不能把局部 benchmark 胜利扩大成全局注册许可。
standalone `silu_and_mul` 仍不在无参数 override 的默认集合中。显式调用时
采用实测 `auto` 路由：N<8192 保留 ATen，N>=8192 才选择内联 SLEEF-u10 的
普通 Triton kernel。完整调用（含分配和 dispatch）在 N8192 为
38.03 us，对比 ATen 47.24 us；融合 MLP 则直接复用该 epilogue。

## Qwen3.5 粗粒度 TLE 复审

旧 TLE 总时间包含 Triton 3.3 launcher 和 C runtime；“C direct”只通过
`ctypes` 调用手写函数，刻意排除 launcher。普通 codegen 使用新 3.7
launcher。所有结果均为单核 microbenchmark。

| 路径/形状 | ATen/参考 | 普通 codegen | 旧 TLE 总时间 | C direct | 决策 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5 RMSNorm, D=2560 | 53.00 us | 11.72 us | 15.97 us | 2.05 us | 用 codegen |
| gated RMSNorm, 16x128 | 77.48 us | 19.03 us | 34.40 us | 18.31 us | 用 codegen |
| causal conv, C=8192, K=4 | 1948.66 us | 98.80 us | 124.39 us | 108.41 us | 用 codegen |
| gated-delta, 16x128x128 | 505.82 us | 336.39 us | 175.51 us | 182.20 us | 有 TLE 时保留 TLE |

RMSNorm codegen 保留 FP32 `(1 + weight)`，而旧 runtime wrapper 先把该权重
舍入到 BF16，最大绝对误差为 0.015625。gated RMSNorm 显式保留 HF 在
normalized value 上的 BF16 中间舍入。causal conv 审计还发现旧注释把 state
写成 `kernel_size-1`，实际 HF cache 和 C runtime 均为 4 个元素；新 kernel
按 `[old1, old2, old3, hidden]` 更新，和 ATen/C bit-exact。

gated-delta 的 rolled codegen 虽比 ATen 快约 1.5 倍，但仍比手写状态循环慢
约 2 倍。曾测试把 `128x4` state tile 一次性放入 SSA：汇编膨胀到 6189 行、
1104 个栈访问，性能 499.57 us，更差。因此生产路径采用 `auto`：存在 TLE
时走已验证的 C 快路，stock Triton-CPU 没有 TLE 时才用普通 codegen fallback。

## Attention crossover

M=1、Hq=16、Hkv=8、D=128：

旧表把完整 Python codegen router 与 C direct 混比，不能作为 kernel
crossover。重新采用相同预分配、direct-compute timed region 后，单核结果为：

| KV 长度 | online 普通 Triton | staged BFDOT Triton | C direct | staged/C |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 159.53 us | 84.41 us | 153.06 us | 0.551x |
| 512 | 562.25 us | 259.88 us | 592.72 us | 0.438x |
| 1024 | 1100.96 us | 463.15 us | 1182.00 us | 0.392x |
| 2048 | 2162.00 us | 866.36 us | 2346.98 us | 0.369x |

新的 staged schedule 仍只使用普通 Triton：第一段 QK/softmax 写 FP32
scratch，第二段按 D64 做 PV。编译器把 BF16 mul+FP32 sum 识别为 16 个
BFDOT；PV 热循环无 spill。它减少一半 exp，并去掉 online 路径每个 token
对完整 D128 accumulator 的重缩放。8 核下 N=128 的第二次 OpenMP launch
尚不能摊薄，因此生产 `auto` 从 N>=512 使用 staged；可选 C runtime 存在时
仍可服务更短上下文。stock Triton-CPU 没有该 runtime 时，N<512 使用原
online ordinary codegen。Qwen3-0.6B 的 28 次真实 attention 调用由 5.538
降到 4.361 ms，单步 decode 中位数由 71.893 降到 71.338 ms，token 完全一致。
详细数据见 `ATTENTION_STAGED_BFDOT_CODEGEN.md`。

旧 PR attention TLE 还把动态 `seq_len` 作为 i32 传给要求 i64 的 op，原写法
会在旧编译器报 verifier error；对照测试把它修为 `tl.constexpr` 后，N=128
的 TLE 总时间为 167.85 us。

## Q4/Q8 与 KleidiAI 路线

后续模型量化只保留 Q4 和 Q8：

- Q8 decode：KleidiAI 同 blob ABI 的普通 Triton kernel 已生成 SDOT/ADDP。
  在重新构建的 Triton 3.7 上，1024x3072、4096x3072、8192x3072 和
  11008x4096 的 Triton/KAI 比值为 1.0001/0.9929/0.9473/0.9499。
- Q8 自动策略：`K<8192` 使用 split-index U2，`K>=8192` 使用 pointer-induction
  U2；策略只依赖 K，packed ABI 不变。所有模型形状均为 16 个静态 SDOT、
  一个 ADDP、零栈访问、零外部调用。
- Q4 decode：Q4_0 和 Q4_1 已接入相同的 ordinary-Triton/KAI-layout 路线；
  Q4_0 已通过相同 packed blob 和 llama.cpp 端到端验证。
- Q8 prefill：相同 KAI packed blob 的 M16/N4 ordinary-Triton 路径已经融合
  外层 K loop，直接携带 16 个 SMMLA 累加器。M16/N1024、K=512/1024/
  2048/4096 的 Triton/KAI 比值为 1.024/1.014/1.009/1.025；M32 和 N4096
  也分别为 1.012/1.012。热循环零栈访问、零外部调用，已达到 KAI parity。
- Q4 prefill：四个普通 `tl.dot([4,32],[32,4])` 现在会和 K32 scale
  epilogue 融合；编译器直接消费物理 LHS/RHS panel，生成 64 个 NEON
  SMMLA 和 16 个 `SCVTF #4`。进一步采用 KAI 原生的 `[K8,row,8]` LHS
  与 `[K8-segment,column,packed-K8]` RHS panel 后，后端以 16-byte load
  加 `SHL/AND` 直接送入 SMMLA，消除了旧布局 nibble 展开的 ZIP1。严格
  使用 KAI 官方 packer 产生的同一份 scale+data blob 时，16x1024x1024
  为 230.44 us，
  KAI 为 230.28 us（1.001x）；K512-4096、M32 和 N4096 的比值为
  1.000/1.006/1.015/1.003/1.001。输出 bit-exact，packing 与 launcher
  排除，热 K loop 无 spill、无外部调用。因此现在可结论为 kernel-level
  parity，而不是先前各自 packed buffer 下仍慢 24-27% 的研究结果。相同
  exact ABI 下，M4/M8/M12 shape-specialized tail 分别为 KAI 的
  0.979/0.867/0.866x；编译器按一到三个 M4 panel 生成静态循环，避免 KAI
  通用 M<16 分支与 partial-store 开销。当前 tail contract 要求 M%4=0。
- 非 SVE Arm：Q4 decode 继续只要求 DotProd；Q4 prefill 的 M4 融合新增
  fixed-NEON-i8mm 模式，Apple M 系列不会误发 SVE 指令。该模式已有
  64-SMMLA/零残留 dot 的 M16 pass 回归和 16-SMMLA 的 M4 pass 回归，但
  在 M4 主机上重新实测前不宣称与 CIX 数字相同。

详细 Q4/Q8 数据分别见 `W4_KLEIDIAI_CODEGEN_STUDY.md`、
`W4_KAI_LLAMA_E2E_RESULTS.md` 和 `W8_KLEIDIAI_CODEGEN_STUDY.md`。

## 可复现入口

- `benchmarks/bench_flaggems_arm_pr_ops.py`
- `benchmarks/test_flaggems_arm_rolled_ops.py`
- `benchmarks/bench_qwen35_rms_codegen.py`
- `benchmarks/bench_qwen35_conv1d_codegen.py`
- `benchmarks/bench_qwen35_gated_delta_codegen.py`
- `benchmarks/bench_qwen35_legacy_tle_launch.py`
- `benchmarks/bench_attention_decode_codegen.py`
- `benchmarks/bench_w8_kleidiai_layout_codegen.py`
- `benchmarks/bench_w8_prefill_kai_layout_codegen.py`
- `benchmarks/bench_w4_prefill_i8mm_direct_codegen.py`

`py_compile`、`git diff --check`、rolled-op correctness、W8A8 builder/codegen
validation 和上述 microbench 均已通过。缺失的正确版本 LLVM 包已经恢复，
核心 `triton` 和独立 `triton-opt` 目标均可重新构建。CPU-only optimizer
不再无条件包含/链接 AMD、NVIDIA dialect 和 test pass；rolled SVE2 i8mm
以及 KAI-layout SDOT、replicated-load、ADDP、SMMLA loop fusion、fixed-point
SCVTF 的 FileCheck 回归测试均已实际运行通过。性能结论另有 fresh-cache
JIT/AOT 生成物、汇编审计、数值测试和端到端 W8 smoke test 支撑。
