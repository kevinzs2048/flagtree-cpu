typedef unsigned int u32;
typedef long long i64;
typedef signed char i8;
typedef unsigned short bf16;

static __attribute__((always_inline)) int load_i32(const void *pointer) {
  int value;
  __builtin_memcpy(&value, pointer, sizeof(value));
  return value;
}

static __attribute__((always_inline)) float load_f32(const void *pointer) {
  float value;
  __builtin_memcpy(&value, pointer, sizeof(value));
  return value;
}

static __attribute__((always_inline)) float bf16_to_f32(bf16 value) {
  const u32 bits = (u32)value << 16;
  float result;
  __builtin_memcpy(&result, &bits, sizeof(result));
  return result;
}

static __attribute__((always_inline)) bf16 f32_to_bf16(float value) {
  u32 bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return (bf16)(bits >> 16U);
}

__attribute__((visibility("default"))) void _rms_same_backend_c(
    const bf16 *__restrict input, const bf16 *__restrict weight,
    bf16 *__restrict output, u32 pid_x, u32 pid_y, u32 pid_z, u32 grid_x,
    u32 grid_y, u32 grid_z) {
  (void)pid_y;
  (void)pid_z;
  (void)pid_x;
  (void)grid_x;
  (void)grid_y;
  (void)grid_z;

  const int size = 1024;
  const float eps = 1.0e-6f;
  float square_sum = 0.0f;
  for (int offset = 0; offset < size; ++offset) {
    const float value = bf16_to_f32(input[offset]);
    square_sum += value * value;
  }
  const float rrms = 1.0f / __builtin_sqrtf(square_sum / size + eps);
  for (int offset = 0; offset < size; ++offset) {
    const bf16 normalized =
        f32_to_bf16(bf16_to_f32(input[offset]) * rrms);
    output[offset] = f32_to_bf16(
        bf16_to_f32(normalized) * bf16_to_f32(weight[offset]));
  }
}

__attribute__((visibility("default"))) void _rope_same_backend_c(
    bf16 *__restrict q, bf16 *__restrict k,
    const i64 *__restrict positions, const bf16 *__restrict cos_sin_cache,
    u32 pid_x, u32 pid_y, u32 pid_z, u32 grid_x, u32 grid_y, u32 grid_z) {
  (void)pid_y;
  (void)pid_z;
  (void)grid_x;
  (void)grid_y;
  (void)grid_z;
  const int q_heads = 16;
  const int head_dim = 128;
  const int half = 64;
  bf16 *row = pid_x < (u32)q_heads
                  ? q + pid_x * head_dim
                  : k + (pid_x - q_heads) * head_dim;
  const bf16 *cache = cos_sin_cache + positions[0] * head_dim;
  for (int offset = 0; offset < half; ++offset) {
    const float first = bf16_to_f32(row[offset]);
    const float second = bf16_to_f32(row[half + offset]);
    const float cosine = bf16_to_f32(cache[offset]);
    const float sine = bf16_to_f32(cache[half + offset]);
    row[offset] = f32_to_bf16(first * cosine - second * sine);
    row[half + offset] = f32_to_bf16(first * sine + second * cosine);
  }
}

// Portable scalar C over the exact KAI qai8dxp/qsi8cxp physical layout.
// K=1024 and N=3072 are intentionally shape-specialized to match Triton.
__attribute__((visibility("default"))) void _w8_same_backend_c(
    const i8 *__restrict lhs_packed, const i8 *__restrict rhs_packed,
    const float *__restrict clamp, float *__restrict output, int range_begin,
    int range_end, u32 pid_x, u32 pid_y, u32 pid_z, u32 grid_x, u32 grid_y,
    u32 grid_z) {
  (void)pid_x;
  (void)pid_y;
  (void)pid_z;
  (void)grid_x;
  (void)grid_y;
  (void)grid_z;
  const int k = 1024;
  const int rhs_stride = 4 * k + 48;
  const int lhs_offset = load_i32(lhs_packed + k);
  const float lhs_scale = load_f32(lhs_packed + k + 4);
  const float clamp_min = clamp[0];
  const float clamp_max = clamp[1];

  for (int block = range_begin; block < range_end; ++block) {
    const i8 *rhs = rhs_packed + block * rhs_stride;
    int accumulator0 = 0;
    int accumulator1 = 0;
    int accumulator2 = 0;
    int accumulator3 = 0;
    for (int group = 0; group < k / 8; ++group) {
      const i8 *x = lhs_packed + group * 8;
      const i8 *w = rhs + group * 32;
      for (int lane = 0; lane < 8; ++lane) {
        const int value = (int)x[lane];
        accumulator0 += value * (int)w[lane];
        accumulator1 += value * (int)w[8 + lane];
        accumulator2 += value * (int)w[16 + lane];
        accumulator3 += value * (int)w[24 + lane];
      }
    }

    const int accumulators[4] = {accumulator0, accumulator1, accumulator2,
                                 accumulator3};
    const i8 *epilogue = rhs + 4 * k;
    for (int lane = 0; lane < 4; ++lane) {
      const int rhs_sum = load_i32(epilogue + lane * 4);
      const float rhs_scale = load_f32(epilogue + 16 + lane * 4);
      const float bias = load_f32(epilogue + 32 + lane * 4);
      const int corrected = accumulators[lane] + rhs_sum * lhs_offset;
      float result = (float)corrected * lhs_scale * rhs_scale + bias;
      result = result < clamp_min ? clamp_min : result;
      result = result > clamp_max ? clamp_max : result;
      output[block * 4 + lane] = result;
    }
  }
}
