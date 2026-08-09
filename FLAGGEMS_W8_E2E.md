# FlagGems + Triton CPU SVE2/i8mm：Qwen3 W8A8 端到端结果

日期：2026-07-30

## 结论

本轮按端侧场景只优化和测试 W8A8，不再把 BF16 作为目标方案。

在单个 Cortex-A720 大核、SVE2+i8mm、SVE VL=128、Qwen3-0.6B、
6-token prompt、greedy decode 16 tokens 的固定条件下：

| 路径 | 生成中位耗时 | 吞吐 | 相对 W8 基线 | 16 个生成 token |
|---|---:|---:|---:|---|
| W8A8 Linear 基线 | 1.5362 s | 10.4153 tok/s | — | 参考 |
| 默认精确组合 | 1.4096 s | 11.3509 tok/s | **+8.98%** | 与参考逐 token 相同 |
| 加在线-softmax attention | 1.2819 s | 12.4811 tok/s | **+19.83%** | 可能分叉 |

默认精确组合包括：

- 197 个 `nn.Linear` 替换为 `TLEInt8Linear`；
- decode 的 W8A8 SDOT GEMV；
- 短 prefill 的 Triton `tl.dot` → SVE2 i8mm `smmla`；
- bit-exact RoPE；
- bit-exact RMSNorm 和 residual-add + RMSNorm；
- bit-exact fused W8 MLP；
- bit-exact Triton CPU vocabulary argmax。

在线-softmax attention 改变浮点归约顺序。单算子验证在
`[1,16,1,128] x [1,8,128,128]` 上相对 L2 误差为
`0.002356`，最大绝对误差为 `0.001953125`。它有明确端到端收益，
但不是 bit-exact，因此正式入口默认关闭，必须显式
`enable_attention=True` 才启用。

## 为什么没有直接使用 `flag_gems.enable()`

参考了 flagos-ai/community PR #15 的 PyTorch/FlagGems 推理方式，但
`flag_gems.enable()` 会注册完整通用算子表。该路径在本机 Qwen3 warmup
超过两分钟仍未完成，并且 profiler 证明其中一些通用 CPU kernel 对当前
shape 是负收益。

本实现增加的是 profiler 驱动的白名单入口：

```python
from flag_gems.runtime.backend._arm.int8 import optimize_qwen3_w8a8

setup = optimize_qwen3_w8a8(model)
```

默认参数选择数值可复现组合。若接受 online-softmax 与 ATen 不同的归约
顺序，可使用：

```python
setup = optimize_qwen3_w8a8(model, enable_attention=True)
```

入口也接受预量化 W8 state dict；未提供时对模型权重执行一次内存内
per-output-channel W8 量化。

## SVE2/i8mm prefill 路径

短 prefill 不再拆成：

```text
BF16 -> FP32 -> abs -> amax -> div -> clamp -> INT8
     -> torch._int_mm -> FP32 -> 两次 scale -> BF16
```

而是两个 Triton launch：

```text
Triton row quantization
    -> Triton tl.dot(INT8, INT8)
    -> fused row/channel dequant
    -> BF16 store
```

`tl.dot` 保留到 CPU dot lowering，最终经过
`ConvertDotToSVE2I8MM.cpp` 生成 SVE2 i8mm。自动验证在生成物中找到：

- 64 条 `smmla` 汇编指令；
- `llvm.aarch64.sve.smmla` LLVM intrinsic。

算子微基准（单核，输出与旧拆分路径 bit-exact）：

| M x K x N | 旧拆分路径 | 新融合路径 | 加速 |
|---|---:|---:|---:|
| 6 x 1024 x 1024 | 451.97 us | 336.41 us | **1.343x** |
| 6 x 1024 x 3072 | 937.18 us | 831.77 us | **1.127x** |
| 6 x 3072 x 1024 | 931.05 us | 823.36 us | **1.131x** |
| 32 x 1024 x 3072 | 3125.59 us | 2999.16 us | **1.042x** |

M=64 时收益降到约 0.8%，端到端无收益。默认只对 `M<=32` 使用新融合，
更长 prefill 自动回到已有 `_int_mm` 路径，避免为了覆盖更多 shape 引入
回退。阈值可用 `FLAGGEMS_ARM_FUSED_PREFILL_MAX_M` 调整，也可用
`FLAGGEMS_ARM_FUSED_PREFILL=0` 完全关闭。

## 数值问题及修正

在第一次端到端测试中，性能组合会在若干 decode token 后改变 greedy
输出。逐算子检查找到了三个实现问题：

1. RoPE 在核内一直使用 FP32，跳过 PyTorch BF16 乘法后的舍入；
2. fused residual+RMSNorm 使用未舍入的 FP32 residual 计算方差；
3. fused MLP 使用二阶 exp 近似，并跳过 projection、SiLU、mul 之间的
   BF16 可观察边界。

修正后，RoPE（包括真实非连续 transpose 输入）、fused norm、fused MLP
及短 prefill 都与对应 W8 参考逐元素完全一致。默认精确组合的 16 个
greedy token 也与 W8 基线一致。

attention runtime 的输出转换另行从截断修正为 BF16 round-to-nearest，
相对 L2 误差从约 0.42% 降到约 0.24%；剩余差异来自 online softmax
归约顺序，因此没有把它包装成“精确”路径。

## 可复现命令

所有命令从 `/home/kevin/triton-opt-cpu` 执行。

正确性、codegen 和路由验证：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  FLAGGEMS_VENDOR_NAME=arm \
  taskset -c 0 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/validate_w8a8.py
```

W8 基线：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  FLAGGEMS_VENDOR_NAME=arm \
  taskset -c 0 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/run_qwen3.py \
  --mode int8 --threads 1 --warmup-tokens 2 \
  --new-tokens 16 --repeats 3 --prefill-repeats 5
```

默认精确组合：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  FLAGGEMS_VENDOR_NAME=arm \
  taskset -c 0 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/run_qwen3.py \
  --mode int8-qwen3-argmax --threads 1 --warmup-tokens 2 \
  --new-tokens 16 --repeats 3 --prefill-repeats 5
```

显式启用高性能 attention：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  FLAGGEMS_VENDOR_NAME=arm \
  taskset -c 0 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/run_qwen3.py \
  --mode int8-qwen3-attn-argmax --threads 1 --warmup-tokens 2 \
  --new-tokens 16 --repeats 3 --prefill-repeats 5
```

使用 `python -S` 是有意的：测试虚拟环境里存在指向其他 FlagGems 和
FlagTree checkout 的 editable-install finder；显式 `sys.path` 保证测到的
就是 `/home/kevin/triton-opt-cpu` 内的代码。

## 当前边界

- 本轮是 W8A8，不是 W4A8；W4 已有另一套依赖 KleidiAI/预打包权重的
  plugin 路径，不应在没有同一模型、量化格式和质量基准的情况下混测。
- live quantization 的 setup 时间约 5.16 s，不计入稳态生成吞吐。实际部署
  应直接加载预量化 W8 权重。
- 当前数字是单核 Qwen3-0.6B 的稳定中位数，不外推为所有核数、所有
  prompt 长度或所有 ARM CPU 的结果。
- attention 高性能档仍需要在更大语料上补 perplexity/任务精度评估，
  因此默认关闭。
