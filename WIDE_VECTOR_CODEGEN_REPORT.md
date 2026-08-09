# ARM Triton CPU 普通算子宽向量 codegen 修复

日期：2026-07-31

平台：CIX P1/CD8180，AArch64，SVE2+i8mm，进程 SVE VL=128。性能测试固定
在一个性能核（`taskset -c 0`），OpenMP/BLAS 线程数为 1。

## 结论

RMSNorm、fused add+RMSNorm、Qwen3 RoPE 和 vocabulary argmax 都不是
`TLE_raw`。它们原先是普通 `@triton.jit`，但 GPU 风格的大 block 被
triton-cpu 直接变成 `<64>/<128>/<512>` 固定 LLVM vector，导致 AArch64
后端产生几千行展开代码、2 KB 级栈帧和大量 spill。

这次没有把计算替换为 C runtime call，也没有增加 raw/extern leaf。修复后的链路是：

```text
ordinary @triton.jit
  -> 小向量 lane state + runtime rolled scf.for
  -> vector.load / arith / vector.reduction
  -> LLVM AArch64 NEON（目标支持时也可使用 SVE）
  -> 独立生成的 kernel .so
```

直接 AOT 计算主体已经达到手写 NEON C 水平：三个 BF16 算子等于或快于
C，vocabulary argmax 比充分展开到 16 lanes 的 NEON C 慢 4.7%。

## 原始问题

`tle.program_id()` 只是对 `tl.program_id()` 做 i64 转换，不是 raw 调用。
四个 kernel 的 LLIR 都没有非 LLVM 外部声明，汇编也没有跳到
`libTritonCPURuntime.so`。

真正的问题是固定向量形状：

| 算子 | 原始 LLVM 形状 | 原始实现特征 |
|---|---:|---|
| RMSNorm | `<128 x f32/bf16/i32>` | 每个 tile 横向 reduction |
| fused add+RMS | `<128 x ...>` | 两遍宽 load/store + reduction |
| RoPE | `<64 x ...>` | 四组 BF16/FP32 乘法与转换全部展开 |
| argmax stage 1/2 | `<512 x f32/i32/i64>` | value/index reduce 展开为 shuffle tree |

triton-cpu 的 `ConvertReductionOp.cpp::lower1DInput` 对多结果 reduction 使用
butterfly `vector.shuffle`。argmax 同时归约 value 和 index，因此 512 lanes
会形成两个巨大的静态 shuffle 网络。LLVM 后端无法重新恢复成 C 风格的
rolled loop。

MLIR 的通用 `vector unroll` 也不是答案：它把宽 vector 静态切成更多小
vector 副本，不会创建运行时滚动循环，代码体积问题依然存在。

## 修改

### 目标感知的 ARM vector schedule

`runtime/backend/_arm/vector_config.py` 在 Linux 读取进程 SVE VL；Apple
M 系列和没有 SVE 的 AArch64 使用 128-bit Advanced SIMD。也可以通过
`FLAGGEMS_ARM_VECTOR_BITS` 显式覆盖。

- elementwise tile：一个 FP32 原生向量；
- reduction tile：四个 FP32 原生向量，提供足够独立累加链；
- CIX VL=128 和 Apple M4 均得到 elementwise=4、reduction=16；
- 更宽 SVE 机器会随 VL 扩展，而不是写死某一款 CPU。

### RMSNorm / fused add+RMSNorm / RoPE

这三个 kernel 保留普通 Triton 表达，只把逻辑 block 改为目标 tile，
由已有 `for range(0, N, BLOCK)` 生成 rolled `scf.for`。

### Vocabulary argmax

GPU 两阶段 reduction 被替换为一个普通 Triton CPU kernel：

1. 循环内保持 16 个 FP32 lane winner 和 i32 index；
2. 扫描完整个 vocabulary 后只做一次横向 max 和 min-index reduction；
3. 最后将标量 index 扩展为 aten ABI 需要的 i64；
4. 通用 aten 路径保留“第一个 NaN 胜出”语义；
5. Qwen3 优化入口选择有限 logits 分支，避免在正常 decode 中维护
   无意义的 NaN 状态。

有限 logits 是模型级契约；普通 `flag_gems.argmax` 默认仍走 NaN-correct
版本。

## 生成代码

| 算子 | 原始 ASM 行数 | 新 ASM 行数 | 原始栈帧 | 新栈帧 |
|---|---:|---:|---:|---:|
| RMSNorm N=1024 | 5,752 | 248 | 352 B | 0 B |
| fused add+RMS N=1024 | 8,149 | 288 | 416 B | 0 B |
| RoPE Hq/Hkv=16/8 D=128 | 4,104 | 209 | 816 B | 0 B |
| vocab argmax N=151936 | 9,432 + 7,957 | 262 | 2,160 + 2,144 B | 0 B |

新的四个 `.asm` 中都没有以 `sp` 为基址的数据 load/store。新的 argmax
LLIR 仅 88 行，不再有 512-lane value/index shuffle tree。

## 直接 AOT 对手写 NEON C

测试预先分配输入输出，由 C++ 直接调用生成 `.so`，不包含 Python
dispatcher。C reference 使用 BF16/NEON intrinsic；argmax C 版本同样展开
到 16 lanes，不是未优化标量循环。

| 算子与形状 | Triton AOT | 手写 NEON C | Triton/C | 结果 |
|---|---:|---:|---:|---|
| RMSNorm BF16 N=1024 | 0.662 us | 0.758 us | **0.874x** | bit-exact |
| fused add+RMS BF16 N=1024 | 0.939 us | 1.195 us | **0.786x** | 已通过模型语义验证 |
| RoPE BF16 Hq/Hkv=16/8 D=128 | 1.890 us | 1.932 us | **0.978x** | bit-exact |
| argmax BF16 N=151936 | 38.785 us | 37.040 us | 1.047x | tie exact |

argmax 通过 `libtriton-jit` C++ wrapper 为 38.949 us，只比直接函数调用多
0.164 us；因此这里的 4.7% 是计算主体差距，不是 launch 假象。

## Python 逐算子 tile sweep

这一组包含相同的 Triton Python launcher，主要用于比较 codegen 形状。

| 算子 | 宽向量 | rolled tile | 延迟变化 |
|---|---:|---:|---:|
| RMSNorm N=1024 | tile 128: 18.97 us | tile 16: 17.00 us | -10.4% |
| fused add+RMS | tile 128: 27.01 us | tile 16: 23.52 us | -12.9% |
| RoPE Q+K | tile 64: 29.65 us | tile 4: 25.95 us | -12.5% |
| argmax 旧两阶段 | block 512: 320.2 us | finite rolled: 55.3 us | -82.7% |

短算子的 Python 数字主要受约 15-20 us 调度影响，所以 AOT/C++ 数字才是
判断 LLVM 计算主体质量的依据。

## Qwen3 W8 端到端

Qwen3-0.6B，6-token prompt，greedy 生成 16 tokens，单核：

| 版本 | 16-token 中位时间 | 吞吐 | token |
|---|---:|---:|---|
| 修复前 codegen attention 配置 | 1.2660 s | 12.6385 tok/s | reference |
| 当前 rolled/native-vector 配置 | 1.2602 s | 12.6962 tok/s | identical |

单 decode profiler 总 self CPU 从 72.366 ms 降到 70.217 ms（-2.97%）。
vocabulary argmax range 从 0.546 ms 降到 0.179 ms（-67.2%，profiler 包含
range/dispatcher 开销）。Triton CPU 时间覆盖率从 88.02% 变为 88.19%；
算子更快后覆盖率没有被人为放大。

## 验证与复现

完整正确性与 codegen 守卫：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  FLAGGEMS_VENDOR_NAME=arm FLAGGEMS_ARM_ATTN_DECODE_IMPL=codegen \
  taskset -c 0 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/validate_w8a8.py
```

tile sweep：

```bash
env FLAGGEMS_VENDOR_NAME=arm OMP_NUM_THREADS=1 \
  taskset -c 0 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/bench_wide_vector_tiles.py
```

直接 AOT/C：

```bash
bash benchmarks/cpp_wrapper/build.sh

rms_so=$(find artifacts/wide-vector-rolled -name _rms_norm_kernel.so | head -1)
fused_so=$(find artifacts/wide-vector-rolled \
  -name _fused_add_rms_norm_kernel.so | head -1)
rope_so=$(find artifacts/wide-vector-rolled \
  -name _rope_qk_bf16_kernel.so | head -1)
taskset -c 0 artifacts/bench_norm_rope_aot \
  "$rms_so" "$fused_so" "$rope_so"

argmax_so=$(find artifacts/argmax-rolled-finite-aot \
  -name argmax_vocab_rolled_kernel.so | head -1)
taskset -c 0 artifacts/bench_argmax_aot "$argmax_so" 3000 11
```

## 边界

- 本次修复说明“普通 Triton 可以生成 C 级 LLVM”，但也说明 GPU kernel
  的 block 形状不能不经调度直接照搬到 CPU。
- 当前通用 triton-cpu pipeline 仍不会自动把任意 `<512>` tensor 重新
  strip-mine 成 runtime loop；本次把目标感知 schedule 放在 ARM kernel
  层，计算仍完全 compiler-visible。
- argmax finite 快路径只由 Qwen3 模型优化入口启用；通用 API 保留 NaN
  语义。
- 这些算子主要使用 BF16 load/store 和 FP32/NEON 算术；SVE2 i8mm 的
  核心收益仍在 W8 `tl.dot`/SDOT/SMMLA 路径，不能把所有 elementwise
  加速都归因于 i8mm。
