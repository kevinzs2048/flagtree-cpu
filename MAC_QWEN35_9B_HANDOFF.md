# Mac M4 Pro 复现与 Qwen3.5-9B Triton 适配交接

## 1. 当前工作做到了哪里

当前有效开发树不是仓库根目录下的旧 Triton，而是：

- Triton-CPU 3.7.2：`ports/triton-cpu-3.7.2`
- Arm FlagGems 算子和 router：`third_party/FlagGems`
- libtriton_jit 接入：`third_party/libtriton_jit`
- vLLM 绑定和 AOT bundle：`integrations/vllm`

已完成的主线是 Q4/Q8 Arm CPU 量化线性层：

- Q4：主干使用 signed Q4 G128 `qsi4c32p x qai8dxp`，词表头可使用 G32。
- Q8：权重是 per-channel signed W8 `qsi8cxp`，激活是 BF16 动态 per-row asymmetric `qai8dxp`。
- Triton kernel 仍然使用 `tl.load` / reshape / permute / `tl.dot` 等普通 Triton 语义，通过 Triton-CPU/MLIR/LLVM lowering 生成 Arm 指令。
- Q4/Q8 计算路径没有 TLE_raw，没有 tail-call 到手写 C GEMV/GEMM runtime。`libtriton_jit` 负责缓存和低开销启动，不替代计算主体。
- CIX 和 M4 已经在生成物中确认 SDOT/SMMLA，并审计了外部 compute call、残留 `triton_cpu.dot` 和热循环 spill。
- Q8 pack 已对全部 65,280 个有限 BF16 编码做过 KAI 逐字节对比；Q4/Q8 direct output 已覆盖 decode/prefill 的主要 M tail。

已有详细结果见：

- `APPLE_M4_PRO_CODEGEN_RESULTS.md`
- `CIX_Q4_Q8_PRODUCTION_CODEGEN.md`
- `VLLM_MINICPM5_LIBTRITON_JIT_RESULTS.md`
- `ACTIVE_SOURCE.md`

MiniCPM5/vLLM 结果证明了这条路径可以做到端到端替换，但这些数据不能直接当作 Qwen3.5-9B 的性能数据。

## 2. Mac 上必须保持的条件

Apple M4 Pro 没有 SVE2，正确目标是 Neon DotProd/I8MM：

```bash
export TRITON_CPU_FIXED_I8MM=1
unset TRITON_CPU_DISABLE_SVE2_I8MM
export FLAGGEMS_VENDOR_NAME=arm
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

必须在 Mac 本机重新编译，不能携带 CIX 的 ELF `.so` 或 JIT cache。Mac 结果应当是 Mach-O arm64，并加载本地 `libTritonCPURuntime.dylib`。

建议使用一个干净 Python 3.11 venv，先安装当前 Qwen3.5 可用的 Transformers，再编译 Triton-CPU：

```bash
cd /Users/kevin/triton-opt-cpu/ports/triton-cpu-3.7.2
python3 -m venv /Users/kevin/venvs/triton-qwen35
source /Users/kevin/venvs/triton-qwen35/bin/activate
python -m pip install -U pip setuptools wheel ninja cmake
python -m pip install torch==2.13.0 transformers==5.13.0 safetensors sentencepiece
export TRITON_PLUGIN_DIRS="$PWD/third_party/cpu"
make dev-install PYTHON=python
make PYTHON=python
```

`TRITON_PLUGIN_DIRS` 不能省略。CPU backend 在 Triton 3.7.2 中以外部 plugin 形式接入；如果干净 venv 安装时没有该变量，`triton.runtime` 会报告 `0 active drivers`。

如果 Transformers/PyTorch 的 Mac wheel 版本与上述不同，要在结果中记录实际版本，不要将不同版本的 eager/compiled 结果混在一起。

先跑已有 M4 回归：

```bash
cd /Users/kevin/triton-opt-cpu
export TRITON_TEST_PYTHON=/Users/kevin/venvs/triton-qwen35/bin/python
bash benchmarks/run_q4_router_m4pro.sh
```

该脚本会验证 Q4 router、Q4/Q8 正确性、W8/QKV/MLP 生成物审计以及 Q4/Q8 microbenchmark。旧的 M4 报告早于后续 exact-KAI M16 变更，因此必须全量重跑，不能只使用旧数据。

## 3. Qwen3.5-9B 不是普通 Qwen3

官方 `Qwen/Qwen3.5-9B` 配置的文本主干是：

- hidden size 4096，intermediate size 12288，32 层。
- 24 层 Gated DeltaNet linear attention + 8 层 full attention，排列为 3:1。
- full attention：16 Q heads，4 KV heads，head dim 256。
- Gated DeltaNet：16 key heads，32 value heads，K/V head dim 128，causal-conv kernel 4。
- vocabulary 248,320，RoPE 只旋转 head dim 的 25%，即 64 个元素，并且是 multimodal RoPE。

官方配置：<https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json>

官方 Transformers 文档：<https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md>

第一阶段只做 `Qwen3_5ForCausalLM` text-only。等文本 prefill/decode 稳定后，再处理视觉塔和真正的图文 mRoPE。

## 4. Qwen3.5-9B 的精确 Linear shape

Text-only 主干共有 249 个 `nn.Linear`（未将融合后的 packed 权重当成额外 Linear）。该数字已用 Transformers `Qwen3_5ForCausalLM` 在 `meta` device 上实例化核对：唯一 `(K,N)` 的计数为 `(4096,32)x48`、`(4096,1024)x16`、`(4096,4096)x56`、`(4096,8192)x32`、`(4096,12288)x64`、`(12288,4096)x32` 和 `(4096,248320)x1`。

| 区域 | 次数 | K | N | 建议 |
| --- | ---: | ---: | ---: | --- |
| GDN `in_proj_qkv` | 24 | 4096 | 8192 | 保留模型已有的合并投影 |
| GDN `in_proj_z` | 24 | 4096 | 4096 | Q8/Q4 GEMV/GEMM |
| GDN `in_proj_a` | 24 | 4096 | 32 | 小 N，先做 ATen/Triton microbench；优先研究 a+b 融合 |
| GDN `in_proj_b` | 24 | 4096 | 32 | 同上 |
| GDN `out_proj` | 24 | 4096 | 4096 | Q8/Q4 GEMV/GEMM |
| full-attn `q_proj` | 8 | 4096 | 8192 | 输出同时含 query 和 gate |
| full-attn `k_proj` | 8 | 4096 | 1024 | 可与 Q/V 打包 |
| full-attn `v_proj` | 8 | 4096 | 1024 | 可与 Q/K 打包 |
| full-attn `o_proj` | 8 | 4096 | 4096 | Q8/Q4 GEMV/GEMM |
| MLP `gate_proj` | 32 | 4096 | 12288 | 与 up 打包为 N=24576 |
| MLP `up_proj` | 32 | 4096 | 12288 | 与 gate 打包 |
| MLP `down_proj` | 32 | 12288 | 4096 | Q8/Q4 GEMV/GEMM |
| `lm_head` | 1 | 4096 | 248320 | 单独做 vocab BN32/BN64 A/B |

full-attention Q/K/V 可以打包为一次 K=4096、N=10240 的投影，但必须保持 Q 的 query/gate 拆分语义。MLP 可复用已有 joined gate/up + 独立 SwiGLU + down-proj 路径。

## 5. 已有 Qwen3.5 算子与 CIX 9B-shape 实测

现有普通 Triton 补丁：

- `patch_qwen3_5_conv1d.py`
- `patch_qwen3_5_gated_delta.py`
- `patch_qwen3_5_rmsnorm.py`
- `patch_qwen3_5_rmsnorm_gated.py`
- `patch_qwen3_5_attention.py`

CIX 单核 microbenchmark，使用 9B 实际 decode shape：

| 算子 | PyTorch reference | 普通 Triton | 手写 runtime | 结论 |
| --- | ---: | ---: | ---: | --- |
| causal conv, C=8192, K=4 | 475.903 us | 200.994 us | 136.926 us | Triton 比 reference 快 2.37x，但仍落后 runtime 1.47x |
| recurrent gated delta, H=32,K=128,V=128 | 889.851 us | 906.375 us | 374.148 us | 普通 Triton 尚不适合生产路由 |
| Qwen3.5 RMSNorm, D=4096 | 73.880 us | 16.382 us | 3.271 us | Triton 生成函数无 spill/call；需用 libtriton_jit 分离 launch 开销 |
| gated RMSNorm, M=32,D=128 | 128.335 us | 33.831 us | 42.600 us | 普通 Triton 比 reference 快 3.79x，也比 runtime 快 20.6% |

注意：这是 CIX microbenchmark，不是 M4 数据，也不是 Qwen3.5-9B 端到端吞吐。

生成物审计显示：

- causal conv 函数仍过大，有 384 个 stack reference 和 16 个 call，还有明确的 codegen 缩减空间。
- gated delta 无外部 call，但大状态更新的寄存和内存调度仍不如分块手写实现。
- D=4096 RMSNorm 生成函数本身是 spill-free/call-free，Python launch 占比不能用来否定生成代码，需使用 libtriton_jit C++ wrapper/AOT 重测。

## 6. Mac 复现顺序

### A. 先复现已有 Q4/Q8 编译器路径

1. 跑 `benchmarks/run_q4_router_m4pro.sh`。
2. 检查 JIT 产物全部是 Mach-O arm64。
3. 检查 M4 上为 SDOT/SMMLA，不能出现 SVE z/p 寄存。
4. 重跑 exact pack/output 检查，再跑 microbenchmark。正确性失败时不允许报性能数据。

### B. 在 M4 上重测 Qwen3.5-9B 非 Linear 算子

使用本机 `libTritonCPURuntime.dylib` 作为对照组。一键入口是：

```bash
cd /Users/kevin/triton-opt-cpu
export TRITON_TEST_PYTHON=/Users/kevin/venvs/triton-qwen35/bin/python
bash benchmarks/run_qwen35_9b_m4pro.sh
```

等价的分项命令是：

```bash
cd /Users/kevin/triton-opt-cpu
export PYTHONPATH="$PWD/ports/triton-cpu-3.7.2/python:$PWD/third_party/FlagGems/src"
runtime="$PWD/ports/triton-cpu-3.7.2/python/triton/_C/libTritonCPURuntime.dylib"

python benchmarks/bench_qwen35_conv1d_codegen.py \
  --channels 8192 --block-size 64 --tle-runtime "$runtime"

FLAGGEMS_ARM_GATED_DELTA_IMPL=triton \
python benchmarks/bench_qwen35_gated_delta_codegen.py \
  --heads 32 --k-dim 128 --v-dim 128 --tle-runtime "$runtime"

python benchmarks/bench_qwen35_rms_codegen.py \
  --d 4096 --gated-m 32 --gated-d 128 --tle-runtime "$runtime"
```

如果 Mac runtime 没有对照符号，先修复 Darwin runtime build；不要用 Linux `.so`。

### C. 生成 Qwen3.5-9B Q8 bundle

1. 从原模型权重派生 `qsi8cxp`，不允许运行时重量化权重。
2. 先支持 M=1 decode，再支持 M=4/8/12/16 和长 prefill。
3. 对上表每个唯一 K/N 做 direct-op exact 测试和 ATen/KAI A/B。
4. N=32 的 a/b 投影单独定策：如果量化+启动开销超过 BF16 ATen，保留 ATen 或与相邻投影融合。
5. N=248320 lm_head 在 M4 重做 BN32/BN64 交替顺序 A/B，不复用 CIX schedule 结论。

### D. 再生成 Q4 bundle

Q4 必须继续使用当前 exact G128/G32 format，不引入第二种私有 nibble layout。先从 M=1 decode 最大权重开始，再扩展 prefill。Q4 是更符合 Mac 内存压力的最终部署格式，但实现顺序仍建议先 Q8，因为 Q8 更容易定位是调度问题还是 Q4 unpack/dequant 问题。

### E. 最后才做端到端路由

顺序是：

1. eager BF16 原模型生成固定 token digest。
2. 只打开一类 Triton 算子，逐类比对 logits/token。
3. 启用全部非 Linear Triton 补丁。
4. 分别接 Q8 和 Q4 Linear router。
5. 用相同 prompt、prefill 长度和 decode token 数报 TTFT、tok/s、端到端时间和 Triton CPU 时间覆盖率。

macOS 第一阶段使用 Hugging Face PyTorch runner，不依赖 vLLM。vLLM/libtriton_jit 生产接入仍在 CIX/Linux 上做；Mac 首先负责 Darwin 生成物、精度和 microbenchmark 验证。

## 7. 当前未完成的事

- 还没有 Qwen3.5-9B 全模型一键 patch/runner。
- 还没有按 9B 全部 shape 生成的 Q4/Q8 Darwin AOT bundle。
- 还没有 Qwen3.5-9B 的 Mac 端到端 eager/compiled 对比和算子时间覆盖率。
- recurrent gated delta 的普通 Triton codegen 尚未达到可替代高性能 runtime 的水平。
- full-attention 需重新验证 head-dim 256、4 KV heads、partial mRoPE=64 和长上下文，不能直接用旧 Qwen3 attention 结论。
- 视觉塔尚未适配，应在 text-only 稳定之后单独 profile。

## 8. 验收标准

一个 Qwen3.5-9B Triton 路由只有同时满足以下条件才能默认开启：

- direct-op 精度通过，量化 pack 与参考格式 exact。
- 生成物是 Mach-O arm64，指令类型与 M4 一致，无 Linux object/cache 污染。
- 无 TLE_raw/外部 C compute call，除非该算子被明确标成 runtime baseline，而不是 Triton codegen 结果。
- 逐算子 microbenchmark 不回归；如果单算子回归，必须有可解释的融合端到端收益。
- 端到端 greedy token/logit 通过指定的精度门限，并同时报告 TTFT、tok/s、端到端延迟和 Triton 时间覆盖率。
