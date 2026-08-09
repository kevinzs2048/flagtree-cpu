// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s
// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm="w4-only=true fixed-i8mm=true" | FileCheck %s --check-prefix=FIXED
// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm="w4-only=true fixed-i8mm=false" | FileCheck %s --check-prefix=DOTONLY

// Protect the M4 tail form of the native-KAI Q4 loop.  This is the same
// ordinary dot/scale graph as the M16 path, but carries one output panel and
// must remain available in fixed-NEON mode for non-SVE Arm CPUs.

// CHECK-LABEL: @kai_q4_native_m4_loop
// CHECK-COUNT-1: scf.for
// CHECK-COUNT-16: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
// CHECK-NOT: triton_cpu.dot
// FIXED-LABEL: @kai_q4_native_m4_loop
// FIXED-COUNT-16: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
// FIXED-NOT: triton_cpu.dot
// DOTONLY-LABEL: @kai_q4_native_m4_loop
// DOTONLY-NOT: llvm.aarch64.neon.smmla
// DOTONLY-COUNT-1: triton_cpu.dot

module {
  tt.func public @kai_q4_native_m4_loop(
      %lhs: memref<?xi8>, %lhs_scale: memref<?xf16>,
      %rhs: memref<?xi8>, %rhs_scale: memref<?xf16>,
      %out: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %shift = arith.constant dense<4> : vector<16x4xi8>
    %mask = arith.constant dense<-16> : vector<16x4xi8>
    %fixed = arith.constant dense<6.250000e-02> : vector<4x4xf32>
    %zero_i32 = arith.constant dense<0> : vector<4x4xi32>
    %zero_f32 = arith.constant dense<0.000000e+00> : vector<4x4xf32>

    %result = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%carried = %zero_f32) -> vector<4x4xf32> {
      %rhs_raw = vector.load %rhs[%iv] : memref<?xi8>, vector<64xi8>
      %rhs_physical = vector.shape_cast %rhs_raw :
        vector<64xi8> to vector<2x4x8xi8>
      %rhs_transpose = vector.transpose %rhs_physical, [0, 2, 1] :
        vector<2x4x8xi8> to vector<2x8x4xi8>
      %rhs_packed = vector.shape_cast %rhs_transpose :
        vector<2x8x4xi8> to vector<16x4xi8>
      %rhs_low = arith.shli %rhs_packed, %shift : vector<16x4xi8>
      %rhs_high = arith.andi %rhs_packed, %mask : vector<16x4xi8>
      %rhs_join = vector.interleave %rhs_low, %rhs_high :
        vector<16x4xi8> -> vector<16x8xi8>
      %rhs_join_shape = vector.shape_cast %rhs_join :
        vector<16x8xi8> to vector<16x4x2xi8>
      %rhs_logical_transpose = vector.transpose %rhs_join_shape, [0, 2, 1] :
        vector<16x4x2xi8> to vector<16x2x4xi8>
      %rhs_logical = vector.shape_cast %rhs_logical_transpose :
        vector<16x2x4xi8> to vector<32x4xi8>

      %lhs_raw = vector.load %lhs[%iv] : memref<?xi8>, vector<128xi8>
      %lhs_physical = vector.shape_cast %lhs_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs_k_order = vector.transpose %lhs_physical, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs_halves = vector.shape_cast %lhs_k_order :
        vector<4x4x8xi8> to vector<4x2x16xi8>
      %lhs_interleave = vector.transpose %lhs_halves, [0, 2, 1] :
        vector<4x2x16xi8> to vector<4x16x2xi8>
      %lhs_logical = vector.shape_cast %lhs_interleave :
        vector<4x16x2xi8> to vector<4x32xi8>

      %ls_f16 = vector.load %lhs_scale[%iv] :
        memref<?xf16>, vector<4xf16>
      %ls = arith.extf %ls_f16 : vector<4xf16> to vector<4xf32>
      %ls_shape = vector.shape_cast %ls :
        vector<4xf32> to vector<4x1xf32>
      %ls_broadcast = vector.broadcast %ls_shape :
        vector<4x1xf32> to vector<4x4xf32>
      %rs_f16 = vector.load %rhs_scale[%iv] :
        memref<?xf16>, vector<4xf16>
      %rs = arith.extf %rs_f16 : vector<4xf16> to vector<4xf32>
      %rs_shape = vector.shape_cast %rs :
        vector<4xf32> to vector<1x4xf32>
      %rs_broadcast = vector.broadcast %rs_shape :
        vector<1x4xf32> to vector<4x4xf32>

      %dot = triton_cpu.dot %lhs_logical, %rhs_logical, %zero_i32,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      %fp = arith.sitofp %dot : vector<4x4xi32> to vector<4x4xf32>
      %scaled = arith.mulf %fp, %fixed : vector<4x4xf32>
      %lhs_scaled = arith.mulf %scaled, %ls_broadcast : vector<4x4xf32>
      %group_scaled = arith.mulf %lhs_scaled, %rs_broadcast : vector<4x4xf32>
      %next = arith.addf %carried, %group_scaled : vector<4x4xf32>
      scf.yield %next : vector<4x4xf32>
    }
    %flat = vector.shape_cast %result : vector<4x4xf32> to vector<16xf32>
    vector.store %flat, %out[%c0] : memref<?xf32>, vector<16xf32>
    tt.return
  }
}
