// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s
// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s --check-prefix=CVT
// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm="w4-only=true fixed-i8mm=true" | FileCheck %s --check-prefix=FIXED
// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm="w4-only=true fixed-i8mm=false" | FileCheck %s --check-prefix=DOTONLY

// Model the ordinary Triton graph used by the KAI-layout Q4 x Q8 prefill
// kernel.  Four M4 dots share one packed Q4 panel and carry four FP32 output
// panels through the K32 loop.  The target rewrite must fuse dot,
// fixed-point conversion, group scales and accumulation without a runtime
// call or an expanded-weight temporary.

// CHECK-LABEL: @kai_q4_prefill_loop
// CHECK-COUNT-1: scf.for
// CHECK-SAME: iter_args(
// CHECK-COUNT-64: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
// CHECK-NOT: triton_cpu.dot
// CVT-LABEL: @kai_q4_prefill_loop
// CVT-COUNT-16: llvm.call_intrinsic "llvm.aarch64.neon.vcvtfxs2fp"
// FIXED-LABEL: @kai_q4_prefill_loop
// FIXED-COUNT-64: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
// FIXED-NOT: triton_cpu.dot
// DOTONLY-LABEL: @kai_q4_prefill_loop
// DOTONLY-NOT: llvm.aarch64.neon.smmla
// DOTONLY-COUNT-4: triton_cpu.dot

module {
  tt.func public @kai_q4_prefill_loop(
      %lhs0: memref<?xi8>, %lhs1: memref<?xi8>,
      %lhs2: memref<?xi8>, %lhs3: memref<?xi8>,
      %lhs_scale0: memref<?xf16>, %lhs_scale1: memref<?xf16>,
      %lhs_scale2: memref<?xf16>, %lhs_scale3: memref<?xf16>,
      %rhs: memref<?xi8>, %rhs_scale: memref<?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %shift = arith.constant dense<4> : vector<16x4xi8>
    %mask = arith.constant dense<-16> : vector<16x4xi8>
    %fixed = arith.constant dense<6.250000e-02> : vector<4x4xf32>
    %zero_i32 = arith.constant dense<0> : vector<4x4xi32>
    %zero_f32 = arith.constant dense<0.000000e+00> : vector<4x4xf32>

    %result:4 = scf.for %iv = %c0 to %c32 step %c1
        iter_args(%out0 = %zero_f32, %out1 = %zero_f32,
                  %out2 = %zero_f32, %out3 = %zero_f32) ->
        (vector<4x4xf32>, vector<4x4xf32>,
         vector<4x4xf32>, vector<4x4xf32>) {
      %rhs_raw = vector.load %rhs[%iv] : memref<?xi8>, vector<64xi8>
      %rhs_physical = vector.shape_cast %rhs_raw :
        vector<64xi8> to vector<4x4x4xi8>
      %rhs_transpose = vector.transpose %rhs_physical, [0, 2, 1] :
        vector<4x4x4xi8> to vector<4x4x4xi8>
      %rhs_packed = vector.shape_cast %rhs_transpose :
        vector<4x4x4xi8> to vector<16x4xi8>
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

      %rs_f16 = vector.load %rhs_scale[%iv] :
        memref<?xf16>, vector<4xf16>
      %rs = arith.extf %rs_f16 : vector<4xf16> to vector<4xf32>
      %rs_shape = vector.shape_cast %rs :
        vector<4xf32> to vector<1x4xf32>
      %rs_broadcast = vector.broadcast %rs_shape :
        vector<1x4xf32> to vector<4x4xf32>

      %lhs0_raw = vector.load %lhs0[%iv] : memref<?xi8>, vector<128xi8>
      %lhs0_physical = vector.shape_cast %lhs0_raw :
        vector<128xi8> to vector<4x2x2x8xi8>
      %lhs0_transpose = vector.transpose %lhs0_physical, [1, 2, 0, 3] :
        vector<4x2x2x8xi8> to vector<2x2x4x8xi8>
      %lhs0_logical = vector.shape_cast %lhs0_transpose :
        vector<2x2x4x8xi8> to vector<4x32xi8>
      %ls0_f16 = vector.load %lhs_scale0[%iv] :
        memref<?xf16>, vector<4xf16>
      %ls0 = arith.extf %ls0_f16 : vector<4xf16> to vector<4xf32>
      %ls0_shape = vector.shape_cast %ls0 :
        vector<4xf32> to vector<4x1xf32>
      %ls0_broadcast = vector.broadcast %ls0_shape :
        vector<4x1xf32> to vector<4x4xf32>
      %dot0 = triton_cpu.dot %lhs0_logical, %rhs_logical, %zero_i32,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      %fp0 = arith.sitofp %dot0 : vector<4x4xi32> to vector<4x4xf32>
      %scaled0 = arith.mulf %fp0, %fixed : vector<4x4xf32>
      %lhs_scaled0 = arith.mulf %scaled0, %ls0_broadcast : vector<4x4xf32>
      %group_scaled0 = arith.mulf %lhs_scaled0, %rs_broadcast : vector<4x4xf32>
      %next0 = arith.addf %out0, %group_scaled0 : vector<4x4xf32>

      %lhs1_raw = vector.load %lhs1[%iv] : memref<?xi8>, vector<128xi8>
      %lhs1_physical = vector.shape_cast %lhs1_raw :
        vector<128xi8> to vector<4x2x2x8xi8>
      %lhs1_transpose = vector.transpose %lhs1_physical, [1, 2, 0, 3] :
        vector<4x2x2x8xi8> to vector<2x2x4x8xi8>
      %lhs1_logical = vector.shape_cast %lhs1_transpose :
        vector<2x2x4x8xi8> to vector<4x32xi8>
      %ls1_f16 = vector.load %lhs_scale1[%iv] :
        memref<?xf16>, vector<4xf16>
      %ls1 = arith.extf %ls1_f16 : vector<4xf16> to vector<4xf32>
      %ls1_shape = vector.shape_cast %ls1 :
        vector<4xf32> to vector<4x1xf32>
      %ls1_broadcast = vector.broadcast %ls1_shape :
        vector<4x1xf32> to vector<4x4xf32>
      %dot1 = triton_cpu.dot %lhs1_logical, %rhs_logical, %zero_i32,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      %fp1 = arith.sitofp %dot1 : vector<4x4xi32> to vector<4x4xf32>
      %scaled1 = arith.mulf %fp1, %fixed : vector<4x4xf32>
      %lhs_scaled1 = arith.mulf %scaled1, %ls1_broadcast : vector<4x4xf32>
      %group_scaled1 = arith.mulf %lhs_scaled1, %rs_broadcast : vector<4x4xf32>
      %next1 = arith.addf %out1, %group_scaled1 : vector<4x4xf32>

      %lhs2_raw = vector.load %lhs2[%iv] : memref<?xi8>, vector<128xi8>
      %lhs2_physical = vector.shape_cast %lhs2_raw :
        vector<128xi8> to vector<4x2x2x8xi8>
      %lhs2_transpose = vector.transpose %lhs2_physical, [1, 2, 0, 3] :
        vector<4x2x2x8xi8> to vector<2x2x4x8xi8>
      %lhs2_logical = vector.shape_cast %lhs2_transpose :
        vector<2x2x4x8xi8> to vector<4x32xi8>
      %ls2_f16 = vector.load %lhs_scale2[%iv] :
        memref<?xf16>, vector<4xf16>
      %ls2 = arith.extf %ls2_f16 : vector<4xf16> to vector<4xf32>
      %ls2_shape = vector.shape_cast %ls2 :
        vector<4xf32> to vector<4x1xf32>
      %ls2_broadcast = vector.broadcast %ls2_shape :
        vector<4x1xf32> to vector<4x4xf32>
      %dot2 = triton_cpu.dot %lhs2_logical, %rhs_logical, %zero_i32,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      %fp2 = arith.sitofp %dot2 : vector<4x4xi32> to vector<4x4xf32>
      %scaled2 = arith.mulf %fp2, %fixed : vector<4x4xf32>
      %lhs_scaled2 = arith.mulf %scaled2, %ls2_broadcast : vector<4x4xf32>
      %group_scaled2 = arith.mulf %lhs_scaled2, %rs_broadcast : vector<4x4xf32>
      %next2 = arith.addf %out2, %group_scaled2 : vector<4x4xf32>

      %lhs3_raw = vector.load %lhs3[%iv] : memref<?xi8>, vector<128xi8>
      %lhs3_physical = vector.shape_cast %lhs3_raw :
        vector<128xi8> to vector<4x2x2x8xi8>
      %lhs3_transpose = vector.transpose %lhs3_physical, [1, 2, 0, 3] :
        vector<4x2x2x8xi8> to vector<2x2x4x8xi8>
      %lhs3_logical = vector.shape_cast %lhs3_transpose :
        vector<2x2x4x8xi8> to vector<4x32xi8>
      %ls3_f16 = vector.load %lhs_scale3[%iv] :
        memref<?xf16>, vector<4xf16>
      %ls3 = arith.extf %ls3_f16 : vector<4xf16> to vector<4xf32>
      %ls3_shape = vector.shape_cast %ls3 :
        vector<4xf32> to vector<4x1xf32>
      %ls3_broadcast = vector.broadcast %ls3_shape :
        vector<4x1xf32> to vector<4x4xf32>
      %dot3 = triton_cpu.dot %lhs3_logical, %rhs_logical, %zero_i32,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      %fp3 = arith.sitofp %dot3 : vector<4x4xi32> to vector<4x4xf32>
      %scaled3 = arith.mulf %fp3, %fixed : vector<4x4xf32>
      %lhs_scaled3 = arith.mulf %scaled3, %ls3_broadcast : vector<4x4xf32>
      %group_scaled3 = arith.mulf %lhs_scaled3, %rs_broadcast : vector<4x4xf32>
      %next3 = arith.addf %out3, %group_scaled3 : vector<4x4xf32>

      scf.yield %next0, %next1, %next2, %next3 :
        vector<4x4xf32>, vector<4x4xf32>,
        vector<4x4xf32>, vector<4x4xf32>
    }
    tt.return
  }
}
