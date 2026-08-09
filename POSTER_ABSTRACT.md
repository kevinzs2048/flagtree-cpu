# Triton Developer Conference Poster

## Title

**Making Triton CPU Hardware-Affine on Arm: From `tl.dot` to SME, SVE2/i8mm, and High-Performance Quantized Inference**

## Abstract / Summary

Upstream Triton-CPU provides generic CPU lowering and selected microkernel/ISA fast paths, but its Arm backend still lacks matrix lowering for SME and SVE2/i8mm. An unmatched `tl.dot` falls back to `vector.contract`. In a clean Triton-CPU 3.7.2 build on CIX P1, our tested FP32, BF16, and INT8 tiles failed to use the available SVE2/i8mm capabilities; 32x32 tiles expanded to 443–474 KB of LLVM IR and 5,355–17,217 static assembly lines marked by LLVM as spills or reloads. Other performance-critical paths may hide computation behind raw ops or external C kernels, limiting compiler optimization, while Python launch overhead can dominate short CPU kernels.

We build a hardware-affine Triton CPU compiler and runtime path that keeps computation compiler-visible. It proves layouts from standard Triton pointer programs, selects rolled SME or SVE2/i8mm, SDOT, and BFDOT lowerings, and coordinates packing reuse, vector-length and hardware-resource constraints, heterogeneous-core dispatch, measured fallback, and native C++ AOT launch.

On Apple M4 Pro, compiler-generated SME kernels reach the performance of direct ACLE SME C. On CIX P1, generated SVE2-i8mm blocks are 5–9% faster than optimized ACLE C, while reuse-aware full-matrix execution reaches C parity. C++ launch overhead is about 47 ns versus 13.2 us from Python. For Qwen3-0.6B, code-generated kernels improve decode throughput by 17.86% and cover 88.02% of profiled CPU time. For Qwen3-4B Q4, generated W4 kernels cover 77.69% of single-thread decode time and improve throughput by 3.12%.

Attendees will see why instruction selection alone is insufficient: competitive Triton CPU requires joint compiler ownership of layout, rolled IR, packing reuse, runtime scheduling, and hardware resources.

## 中文对照

### 标题

**让 Triton CPU 亲和 Arm 硬件：从 `tl.dot` 到 SME、SVE2/i8mm 与高性能量化推理**

### 摘要

上游 Triton-CPU 已提供通用 CPU lowering 和部分 microkernel/ISA fast path，但 Arm 后端仍缺少面向 SME 和 SVE2/i8mm 的矩阵 lowering；未命中专用模式的 `tl.dot` 会回退到 `vector.contract`。我们在 CIX P1 上对干净 Triton-CPU 3.7.2 构建的实测表明，典型 FP32、BF16 和 INT8 tile 未使用硬件提供的 SVE2/i8mm 能力；32×32 tile 膨胀到 443–474 KB LLVM IR，并产生 5,355–17,217 条被 LLVM 标记为 spill 或 reload 的静态汇编。另一些性能关键路径可能通过 raw op 或外部 C kernel 隐藏计算过程，限制编译器优化；Python 启动开销也可能主导短 CPU kernel 的执行时间。

我们构建了一条面向 Arm 硬件能力、并保持计算过程对编译器可见的 Triton CPU 编译与运行时路径。它从标准 Triton 指针程序中证明数据布局，选择循环化的 SME 或 SVE2/i8mm、SDOT、BFDOT lowering，并协同处理 packing 复用、向量长度和硬件资源约束、异构核心调度、基于实测的回退，以及原生 C++ AOT 启动。

在 Apple M4 Pro 上，编译器生成的 SME kernel 达到直接编写 ACLE SME C 的性能。在 CIX P1 上，生成的 SVE2-i8mm 计算块比优化后的 ACLE C 快 5%–9%，感知复用的完整矩阵计算达到 C 实现性能。C++ 启动开销约为 47 ns，而 Python 路径为 13.2 us。在 Qwen3-0.6B 上，codegen kernel 将 decode 吞吐提高 17.86%，覆盖 88.02% 的 profile CPU 时间；在 Qwen3-4B Q4 上，生成的 W4 kernel 覆盖 77.69% 的单线程 decode 时间，并将吞吐提高 3.12%。

参会者将看到为什么仅选择正确指令仍不足以获得高性能：具有竞争力的 Triton CPU 需要编译器共同掌握数据布局、循环化 IR、packing 复用、运行时调度和硬件资源。
