# Triton W4 llama.cpp 路径代码审计

审计日期：2026-08-01
审计对象：CIX P1 上 Qwen3-4B Q4_0 decode 的 KAI-layout Triton W4 路径

## 结论

这条路径不是把 KleidiAI、手写 C 或 TLE_raw 包在 Triton 外面。Q4_0 的
GEMV 循环确实来自普通 Triton 表达式，经我们增加的 AArch64 定向匹配和
LLVM O3 生成机器码。现有五个生产 shape 的函数主体均包含 8 个 NEON
`SDOT`、1 个 `ADDP` 和 fixed-point `SCVTF #4`，函数内无栈访存、无调用；
生成的 `.so` 没有 `DT_NEEDED` 依赖。与 KleidiAI v1.24 使用同一份官方
packed blob 的 microbenchmark 差距为 -0.5% 到 +0.8%，因此“CIX P1
单核、M=1、这些 shape 达到 KleidiAI 性能档位”有证据支持。

审计最初发现的两个高优先级集成问题已经修复：模型 buffer 销毁时会按地址
范围释放对应 cache 和 `dlopen` 句柄；AOT cache 目录由 kernel、编译器和
`libtriton.so` 内容 hash 决定，打包要求目标唯一，并把真实 `.so` 和相对
checksum manifest 复制进 bundle，不再选择任意旧文件或创建绝对软链接。

因此本次审计判断是：**核心 codegen 有真实依据，性能结论在明确边界内
成立；集成层仍是实验原型，不能把当前结果扩写成“任意 Arm、任意模型、
任意线程数均优于原生”。生命周期和构建复现门槛已处理，主要剩余风险是
双份权重内存、跨 Arm 目标验证和完整 logits 回归。**

## 1. 覆盖度：三个不同口径

模型文件为 `/home/cix/models/Qwen3-4B/Qwen3-4B-Q4_0.gguf`。审计脚本
`benchmarks/audit_qwen_w4_coverage.py` 直接读取 GGUF tensor 表，并使用
`repack.cpp` 中的实际 shape allowlist。

| 指标 | 覆盖结果 | 含义 |
| --- | ---: | --- |
| Tensor | 252 / 398 | 可路由的量化线性投影 tensor |
| 参数量 | 3,633,315,840 / 4,022,468,096 = 90.3255% | 静态模型参数覆盖 |
| GGUF tensor 字节 | 2,049,966,080 / 2,369,816,064 = 86.5032% | 文件中 tensor 数据覆盖 |
| 单核 decode CPU 时间 | 约 72.7% | 生成路径中 W4 投影计时 / 16-token wall time |

252 个 tensor 的完整构成为：

| 类型和 shape `[K,N]` | Tensor 数 | 每个 token 的调用数 | 模型功能 |
| --- | ---: | ---: | --- |
| Q4_0 2560x1024 | 72 | 72 | 36 层 K/V 投影 |
| Q4_0 2560x4096 | 36 | 36 | Q 投影 |
| Q4_0 4096x2560 | 36 | 36 | attention 输出投影 |
| Q4_0 2560x9728 | 72 | 72 | gate/up 投影 |
| Q4_0 9728x2560 | 32 | 32 | FFN down 投影 |
| Q4_1 9728x2560 | 4 | 4 | FFN down 投影 |

未覆盖的是 embedding/output head（本模型的大 tensor 是 Q6_K
`[2560,151936]`），以及 RMSNorm、RoPE、attention、激活和 sampling 等
非投影算子。90.33% 是参数覆盖，不等于执行时间覆盖。

时间覆盖此前报告成 77.4%，这个数字口径有误：operator counter 记录了
1 个 warmup 加 16 个 timed token，wall time 只含 16 个 timed token。
将 17-token operator total 乘以 `16/17` 后，早期 Q4_1 实现约为 72.9%。
Q4_1 新 microtile 的最新复测约为 72.7%；变化来自算子本身缩短，不代表
覆盖 tensor 变少。

线程数还会改变实际路由：

| ggml workers | 默认 Triton 路由 |
| ---: | --- |
| 1–2 | 全部 248 个 Q4_0 和 4 个 Q4_1 投影 |
| 3–8 | 全部 248 个 Q4_0 和 4 个 Q4_1 投影 |
| 9 及以上 | 默认全部回原生（尚未验证） |
| 强制实验开关 | 超出默认范围时仍可强制所有 252 个投影走 range kernel |

最新 4/6/8 A720 数据均使用默认路由，不再依赖强制开关；仍没有多线程
operator-time 覆盖率数据，因此 72.7% 时间覆盖只用于单核口径。

## 2. 数据 ABI 不是猜的

ggml 的 canonical contract：

- `block_q4_0` 是 FP16 scale 加 16 个 nibble byte，共 18 字节；低 nibble
  对应 K[0:16]，高 nibble 对应 K[16:32]，数值为 nibble - 8。
- `block_q8_0` 是 FP16 scale 加 32 个 signed int8，共 34 字节。

生产 `pack_q4_0_kai()` 按 KleidiAI RHS packer 的物理顺序组织四个输出：
4 个 FP16 scale 后接 4 个 16-byte segment，并对 nibble byte 执行
`xor 0x88`。新增的 `audit_w4_kai_abi.cpp` 没有复制生产 packer，而是直接
包含生产实现，同时调用 KleidiAI v1.24 官方 LHS/RHS packer。测试结果：

```text
PASS W4 KAI ABI audit rhs_bytes=648 lhs_bytes=102 rhs_exact=true lhs_exact=true
```

这证明测试 case 中生产 RHS pack 与官方 KAI pack 逐字节一致，也证明
ggml canonical Q8_0 已量化块和 KAI M=1 LHS 的 scale+32-byte 结构一致。

边界：KAI 的 F32 LHS packer 用 `roundf`，ggml Arm Q8_0 路径使用
round-to-nearest-even 指令。在精确 half-tie 上两套“F32 到 Q8”结果可能
不同。生产 Triton 路径消费 ggml 已经生成的 Q8_0，因而保持 ggml 语义；
不能据此宣称 KAI 与 ggml 的 F32 动态量化对所有输入 bit-exact。

## 3. Triton 到机器码的真实链路

新 Q4_0 kernel 位于 `benchmarks/bench_w4_kleidiai_layout_codegen.py`。它没有
`tl.dot`，也没有 TLE/raw op，使用的是：

```text
tl.load -> nibble shift/mask -> tl.cat/tl.sum -> int32-to-float
        -> scale/FMA -> tl.store
```

每个循环处理 K32 x N4 物理 tile：低 nibble 左移四位，高 nibble 保留
`0xf0`，四段 Q8 activation 各复制为 16-byte operand，形成 8 次 dot。
整数累加值放大了 16 倍，随后用 `SCVTF #4` 在转换时除以 16。

编译器的 `ConvertDotProduct.cpp` 增加了严格 shape 匹配：

- `4x4 i8` multiply + row reduction + carried add -> NEON SDOT；
- 首个无 carried add 的 reduction -> NEON SDOT；
- `4x2 i32` pair reduction -> NEON ADDP；
- `sitofp * 1/16` -> fixed-point `SCVTF #4`；
- 识别 lowered `tl.cat(x,x)` 的准确 IR 链，恢复 replicated 8-byte load。

这些 pattern 只有在 AArch64 同时检测到 NEON 与 `dotprod` 时才注册。CPU
backend hash 包含 CPU name、全部 feature、SVE vector length 和 SVE2-i8mm
开关，避免不同目标错误复用 Triton cache。它属于**面向 KAI physical
microtile 的定向编译器优化**，不是任意 `tl.sum` 都会自动变成高性能
SDOT，也不是上游 Triton-CPU 3.7 已有行为。

五个生产 shape 的实际 AOT 函数主体一致：8 `sdot`、1 `addp`、0 栈
load/store、0 call。以 K=2560,N=1024 为例，导出函数为 264 字节；ELF
无 `DT_NEEDED`，只有四个链接器生成的 weak undefined hook，没有外部计算
符号。反汇编核心为：

```text
ldur q... / ld1r ...
shl v*.16b, #4
and z*.b, #0xf0
sdot x8
addp
scvtf v4.4s, v4.4s, #4
fmla
```

因此这里“生成代码逼近原生”的准确含义是：Triton 描述数据流，定向 MLIR
rewrite 产生 LLVM AArch64 intrinsic，LLVM 完成寄存器分配、地址生成和
调度；不是在运行时跳到 KAI/C GEMV。

Q4_1 此后也从旧 dot-ready `tl.dot` 路径迁移到同一 N4/K32 microtile。
每组从 canonical Q4_1 保存 4 个 FP16 `d`、4 个 FP16 `m` 和 64 个变换后
nibble byte，共 80 字节；kernel 直接消费 36-byte canonical Q8_1 block，
计算 `(q-8)` 的 SDOT 后补上 `(m+8d)*s_x`。这是本项目基于 KAI Q4_0
microtile 的扩展，不是 KleidiAI 定义的 Q4_1 ABI。它同样生成 8 SDOT、
1 ADDP、`SCVTF #4`、0 栈访存和 0 外部调用；9728x2560 microbenchmark
从旧 Triton 的 1303.878 us 降至 1046.535 us，模型内 operator mean 从
1271.453 us 降至 1016.959 us。

## 4. 架构泛化边界

这条 W4 decode kernel 的主算术是 NEON DotProd，不是 I8MM。KleidiAI 在
M=1 decode 时同样选择 NEON DotProd fallback；I8MM 主要服务更大的 M。

当前 CIX AOT 对象包含 SVE `and z*.b`，因为 LLVM 针对本机 SVE2 目标做了
指令选择。因此：

- Triton source 和 SDOT rewrite 可按目标重新编译；
- 当前 `.so` 只能视为 CIX/同 ISA 目标对象，不是通用 Armv8.2 bundle；
- DotProd-only、不同 SVE VL、Apple M 系列和非 Linux 目标都需要重新编译
  并跑 correctness/codegen/performance tests；
- 当前 microtile 只支持 M=1、K%32、N4 输出 tile，生产 adapter 又保守限制
  N%32 和固定 Qwen shape allowlist；不覆盖 prefill/M>1。

## 5. llama.cpp 路由与并发

模型加载时，Q4_0 trait 先把 canonical 数据放入 Triton KAI-layout cache，
再继续生成 llama.cpp 自己的 native repack，确保未命中 Triton 时仍可回退。
decode 时 ggml 将单行 FP32 activation 量化到 `params->wdata`，barrier 后每个
worker 调用 prepare，再以互不重叠的 N4 range 调用同一个 AOT symbol；末尾
barrier 后返回。

并发正确性方面：activation 状态是 `thread_local`；range 以 4 输出对齐；
动态 chunk id 使用 threadpool atomic；所有 worker 在量化后和返回前都有
barrier。每个线程首次查到 weight 后保存 thread-local handle，后续投影不再
获取全局 mutex；cache 插入或释放会增加 generation，使下一次查找清空本地
索引。模型 buffer 的 free hook 在 ggml 已停止 inference worker 后释放地址
范围内的 Q4_0/Q4_1 cache。该“先停 worker、再销毁 buffer”的前提与 ggml
自身生命周期一致；不支持销毁模型与推理并发进行。

## 6. 审计发现与优先级

### 已关闭（原高）：缓存生命周期

新增 `triton_w4_cache_release_range()`，并从 repack backend buffer 的 free
hook 调用。真实模型 verbose 测试记录到一次释放 252 个 cached weights；
Q4_0/Q4_1 单测在释放后再次 prepare 均按预期失败。缓存仍以 tensor data
地址为查找 key，但 owner buffer 销毁时会同步移除，不再跨模型生命周期遗留。

### 已关闭（原高）：AOT bundle 旧产物选择

bundle cache 目录现在由 kernel source、对应 compiler.py 和实际
`libtriton.so` 的 SHA256 内容共同确定；目标必须恰好一个，否则构建失败。
bundle 内是独立 `.so` 文件而不是绝对 symlink，每个 shape 有记录 codegen
key 和相对文件 checksum 的 manifest。compile-only 仍执行 SDOT/ADDP、
`SCVTF #4`、无 widening multiply、无 stack traffic、无外部 call 的 guard。

### 已关闭（原中）：decode 热路径全局 mutex

每线程使用带 generation 的本地 weight lookup。第一次命中后，后续 token
不再锁全局 map；cache 插入和释放会使 generation 失效。单核 prepare 从约
17 ns 降至 15.2 ns，主要价值是移除多 worker 竞争。配合 8-worker 细粒度
range scheduling，默认 4/6/8 A720 吞吐分别达到 14.859/16.045/15.599 tok/s；
同场 native 为 14.290/15.273/15.573 tok/s。

### 中：内存占用约增加一份投影权重

为同时保留 native fallback 与 Triton layout，模型内存中存在 native repack
和约 2.05 GB Triton packed cache。测得总 peak RSS 仍约 6.7 million KiB。
实验方便，端侧产品不可接受。长期应让 backend-owned packed buffer 成为
可选择的 tensor storage，或按线程策略只保留一种 layout。

### 中：Q4_1 的 prefill/fallback 策略仍不完整

Q4_1 decode 已迁移到高质量 microtile，但 cache trait 仍保留 canonical
bytes，让非 decode 路径走 generic 实现，没有 optimal native repack；可能
影响 prompt/multi-token 性能。Q4_1 的旧 Triton kernel可作为新 symbol 缺失
时的 layout fallback，但 bundle 整体缺失仍需更清晰的 native capability
query 和错误策略。

### 中：正确性证据仍不是完整模型 logits 对比

已有证据包括同 blob W4 microbenchmark bit-exact、Q4_0 adapter `max_abs=0`、
Q4_1 adapter `max_abs=1.73897e-05`，以及三条 compiler tests 通过。端到端只
比较了 deterministic greedy text，没有逐 token 全 vocab logits diff，也未
覆盖 NaN/Inf、极值和模型热加载。发布前必须补这些测试。

### 低：集成和 bundle 不可移植

实验 llama.cpp CMake 通过目录层级推导 `triton-opt-cpu` integration，始终
编译该 adapter。bundle 已没有绝对 symlink，但 AOT 对象仍按 CIX 本机 feature
生成。它适合当前 worktree 验证，不是跨机器分发包；仍需显式 CMake option、
目标 ISA manifest 和运行时 dispatch。

## 7. 已执行验证

```text
ABI byte audit:
  PASS W4 KAI ABI audit rhs_bytes=648 lhs_bytes=102
  rhs_exact=true lhs_exact=true

Compiler tests:
  3 passed, 2 deselected

Adapter correctness:
  Q4_0 K=2560 N=1024 max_abs=0 released=1
  Q4_1 K=9728 N=2560 max_abs=1.73897e-05 released=1

Production Q4_0/Q4_1 U1 compile guards:
  asm_sdot=8 asm_addp=1 stack_load_store=0 external_calls=0 (each)

Q4_1 microbenchmark, K=9728 N=2560:
  old dot-ready Triton=1303.878 us
  new KAI-style Triton=1046.535 us

Qwen3-4B default decode, selected A720 cores:
  workers=4 Triton=14.859 native=14.290 KleidiAI=13.979 tok/s
  workers=6 Triton=16.045 native=15.273 KleidiAI=15.054 tok/s
  workers=8 Triton=15.599 native=15.573 KleidiAI=15.039 tok/s

GGUF coverage:
  routed_tensors=252
  routed_param_percent=90.325535
  routed_byte_percent=86.503173
```

对应命令：

```bash
./artifacts/audit_w4_kai_abi

/home/cix/venv-fep-e2e/bin/python \
  benchmarks/audit_qwen_w4_coverage.py \
  /home/cix/models/Qwen3-4B/Qwen3-4B-Q4_0.gguf

cd ports/triton-cpu-3.7.2
TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 PYTHONPATH=python \
  /home/kevin/venv-int8-clean/bin/python -m pytest -q -s --tb=short \
  python/test/unit/cpu/test_sve2_i8mm.py \
  -k 'packed_w8a8_honors_loop_unroll or kai_w8_physical_layout or kai_w4_physical_layout'
```

## 8. 下一步修复顺序

1. 解决 native/Triton 双份权重，让 backend-owned packed storage 可独占模型
   tensor；这是当前端侧部署的最大实际障碍。
2. 补逐 token 全 vocab logits、极值、model reload/multi-model 压测。
3. 增加 DotProd-only、不同 SVE VL 和 Apple M 系列重新编译/正确性 CI，并把
   ISA 要求写入可机器读取的 bundle manifest。
4. 补 Q4_1 prefill/M>1 的 optimal fallback，再研究 W4 prefill 的 I8MM/SME
   compiler path。
