# ARM W8 LLVM codegen：量化复用与寄存器微块

## 结论

本轮新增的路径仍然是 Triton frontend → TritonCPU op →
SCF/vector IR → LLVM AArch64 intrinsic。生成物不调用外部 GEMV，也没有把
手写 C 塞进 `TLE_raw`。

两个已落地优化：

1. `sdot_gemv_whole`：activation 只量化一次，以 64-output 微块遍历完整
   W8 projection。
2. fused MLP 高优先级 lowering：gate/up 使用 32-output 双累加器微块，
   并配套 32-output K-contiguous packed layout。

在 Qwen3-0.6B、单 Cortex-A720 大核、单线程、8-token greedy decode 上，
同版本旧 grid 路径为 0.678352 s / 11.7933 tok/s；新路径为
0.655065 s / 12.2125 tok/s。吞吐提升 3.55%，生成 token 完全相同。

## Whole W8 projection

旧 `BLOCK_N=512` lowering 在一个 dot loop 中同时携带 128 个
`vector<4xi32>` 累加器，超过 AArch64 32 个向量寄存器，产生大量 spill。
新 lowering：

- 先做一次 BF16 absmax 和 INT8 quantization；
- 将 K 个 INT8 activation 放入 64-byte aligned stack scratch；
- 外层 rolled loop 遍历 64-output packed microtile；
- dot loop 只携带 16 个向量累加器；
- 直接生成 `llvm.aarch64.neon.sdot.v4i32.v16i8`。

K=1024,N=4096：

| 指标 | 旧 grid BLOCK_N=512 | whole TILE_N=64 |
|---|---:|---:|
| Python kernel | 152.027 us | 130.702 us |
| C++ AOT / wrapper | 125.040 us | 105.246 us |
| 手写 C runtime | 126-127 us | 126-127 us |
| 汇编行数 | 2003 | 357 |
| 静态 SDOT | 128 | 16 |
| `[sp]` 静态引用 | 357 | 2 |
| 外部 GEMV call | 0 | 0 |

新 AOT wrapper 比旧 wrapper 延迟低 15.8%，比当前手写 C runtime 低约
17%。1072-byte frame 中约 1024 bytes 是主动管理的量化 scratch，不是
向量 spill。

真实 projection 形状的 Python launch 结果：

| K,N | 旧 grid | whole | 延迟下降 |
|---|---:|---:|---:|
| 2048,1024 | 89.762 us | 75.356 us | 16.0% |
| 3072,1024 | 120.616 us | 105.314 us | 12.7% |
| 1024,3072 | 120.672 us | 102.343 us | 15.2% |
| 1024,约 152K vocab | 8163.372 us | 7104.178 us | 13.0% |

普通 W8 Linear 在单线程时自动使用 whole lowering；第一次 decode 将
已有 512-output pack 原地替换为 64-output pack，不保留重复权重。多线程
保留原 grid 路径，避免丢失 output-block 并行。

## Fused gate/up/SwiGLU

仅仅把循环切成 32-output 微块会回退：如果权重仍按 512-output block
排列，每个微块会以 512-byte stride 扫 K，破坏连续流式读取。最终实现
同时把 gate/up 权重改成 `[N/32,K/4,8,16]`，每个微块拥有连续 K stream。

K=1024,N=3072：

| 指标 | 旧 lowering | microtile lowering |
|---|---:|---:|
| Python kernel | 约 245.5 us | 212.880 us |
| C++ direct AOT | 192.927 us | 176.287 us |
| 手写 C runtime（本次） | 228.250 us | 228.250 us |
| 汇编行数 | 5568 | 488 |
| 静态 SDOT | 256 | 16 |
| 静态 stack 引用 | 1480 | 57 |
| 静态 `Sleef_expf4` call | 128 | 8 |
| 外部 fused-MLP runtime call | 0 | 0 |

新 AOT 比旧 AOT 延迟低 8.6%，比当前手写 C runtime 低 22.7%。
BF16 gate/up 和 SiLU 中间舍入边界保持不变，focused validation 和真实
Qwen greedy token 均通过。

## 端到端 A/B

固定条件：Qwen3-0.6B、prompt 6 tokens、生成 8 tokens、单线程、固定到
CPU 11、7 次中位数。

| 路径 | 8-token 中位数 | tok/s |
|---|---:|---:|
| 同版本旧 grid W8 + 旧 fused MLP | 0.678352 s | 11.7933 |
| whole 普通 W8 | 0.665902 s | 12.0138 |
| whole W8 + fused MLP microtile | 0.655065 s | 12.2125 |

最终相对旧 grid：延迟下降 3.43%，吞吐提升 3.55%。RSS 约 3.64 GiB，
没有因 retile 保留第二份 packed weights。

## 下一步优先级

1. **Whole fused MLP**：当前 6 个 512-output program 仍各自重复一次
   activation 量化。增加一个整投影 op，以 32-output 双 bank 微块在单个
   program 中遍历 N，可继续去掉 5 次量化和 program dispatch。
2. **AOT/PyTorch C++ 接线**：C++ wrapper 与 direct AOT 只差约 1 us，
   Python launch 在 whole GEMV 上约增加 25 us。将稳定 shape 的 kernel
   作为 `.so` 后端注册到 C++ dispatcher，收益可能高于再调一条指令。
3. **多核 two-phase lowering**：先共享量化 activation，再并行调度
   output microtiles。当前 whole 路径只在单线程自动启用。
4. **SwiGLU + down projection producer/consumer fusion**：消除 3072 个
   BF16 intermediate 的物化和一次独立 launch；需要在编译器中表达
   nonlinear producer 与 W8 consumer，而不是 runtime 大算子。
5. **VL>128 的 SVE2 SDOT**：当前 CIX 的 SVE VL=128，NEON SDOT 已匹配
   硬件宽度；在 256/512-bit SVE2 机器上可增加 scalable-vector dot
   lowering，并按 VL 选择微块。

## 产物

- End-to-end：
  `artifacts/qwen3-whole-all-fused-microtile-e2e.json`
- Profile：
  `artifacts/qwen3-whole-all-fused-microtile-profile.json`
- Whole GEMV AOT cache：
  `artifacts/cache-whole-gemv-k1024-n4096/`
- Fused MLP microtile AOT cache：
  `artifacts/cache-fused-mlp-microtile-packed-k1024-n3072/`
