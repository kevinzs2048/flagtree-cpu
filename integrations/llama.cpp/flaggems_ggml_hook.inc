// Reviewable hook for flaggems_ggml.cpp's decode MUL_MAT branch.
//
// Weight metadata owns one triton_w8_kernel created after W8 packing:
//
//   triton = triton_w8_kernel_create_from_bundle(
//       std::getenv("GGML_TRITON_W8_BUNDLE"), K, N, BLOCK_N);
//
// The ggml threadpool can then replace its SDOT range call with:
//
//   int block;
//   while ((block = __atomic_fetch_add(counter, 1, __ATOMIC_RELAXED)) <
//          n_blocks) {
//       if (triton_w8_launch_f32_range(
//               triton, static_cast<const float *>(in->data),
//               packed.data(), scale.data(), static_cast<float *>(dst->data),
//               block, 1) != 0) {
//           return false;
//       }
//   }
//
// The AOT kernel owns activation quantization for each block. This avoids any
// new shared scratch ABI and lets heterogeneous ggml workers steal work.
//
// A second, single-quantization variant is available for shapes where it wins:
//
//   split = triton_w8_split_kernel_create_from_bundle(
//       std::getenv("GGML_TRITON_W8_BUNDLE"), K, N, BLOCK_N);
//
// Thread 0 runs triton_w8_split_quantize_f32(split, in->data), followed by the
// existing ggml barrier. Workers then steal output blocks and call:
//
//   triton_w8_split_launch_prequant_f32_range(
//       split, packed.data(), scale.data(), dst->data, block, 1);
//
// Selection must use a measured (K,N,thread-count) allowlist. On the measured
// CIX P1, split is strongest on one core; fused is usually stronger when
// multiple workers can quantize their independent output blocks in parallel.

// Q4_0 -> grouped Q8 path
// -----------------------
//
// The W4 backend is compiler-generated and has no hand-written GEMV runtime.
// At model load, canonical row-major block_q4_0 weights are repacked once:
//
//   packed.resize(triton_w4_packed_size(K, N));
//   scales.resize(triton_w4_scale_count(K, N));
//   triton_w4_pack_q4_0(
//       weight->data, packed.data(), scales.data(), K, N);
//   w4 = triton_w4_kernel_create_from_bundle(
//       std::getenv("GGML_TRITON_W4_BUNDLE"), K, N);
//
// During decode, ggml already provides a block_q8_0 activation. Split its
// interleaved blocks into the compiler ABI's contiguous Q8 values and fp32
// group scales, then call the one-program whole-projection AOT kernel:
//
//   triton_w4_unpack_q8_0(
//       src1->data, x_q.data(), x_scales.data(), K);
//   triton_w4_launch_q8(
//       w4, x_q.data(), x_scales.data(), packed.data(), scales.data(),
//       static_cast<float *>(dst->data));
//
// This initial adapter is intentionally single-owner: exactly one ggml worker
// performs the whole projection, while the remaining workers wait at the
// normal graph barrier. A range ABI should be added before enabling it for
// thread counts where ggml's row partitioning is faster.
