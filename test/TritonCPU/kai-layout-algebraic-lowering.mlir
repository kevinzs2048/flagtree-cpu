// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s

// These forms are produced by ordinary Triton 3.7 tl.join/reshape/reduction
// source.  Keep the test at the algebraic boundary so frontend graph changes
// cannot silently turn a KAI-layout Q4/Q8 kernel into scalar multiply/reduce.

module {
  // CHECK-LABEL: @ordinary_sdot_replicated
  // CHECK:       llvm.bitcast {{.*}} : vector<8xi8> to i64
  // CHECK:       vector.broadcast {{.*}} : i64 to vector<2xi64>
  // CHECK:       llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // CHECK-NOT:   vector.multi_reduction
  tt.func public @ordinary_sdot_replicated(
      %weights: vector<4x4xi8>, %activation: vector<8xi8>,
      %acc: vector<4xi32>) {
    %joined = vector.interleave %activation, %activation :
      vector<8xi8> -> vector<16xi8>
    %join_shape = vector.shape_cast %joined :
      vector<16xi8> to vector<8x2xi8>
    %transposed = vector.transpose %join_shape, [1, 0] :
      vector<8x2xi8> to vector<2x8xi8>
    %repeated = vector.shape_cast %transposed :
      vector<2x8xi8> to vector<4x4xi8>
    %weights_i32 = arith.extsi %weights :
      vector<4x4xi8> to vector<4x4xi32>
    %activation_i32 = arith.extsi %repeated :
      vector<4x4xi8> to vector<4x4xi32>
    %product = arith.muli %weights_i32, %activation_i32 :
      vector<4x4xi32>
    %dot = vector.multi_reduction <add>, %product, %acc [1] :
      vector<4x4xi32> to vector<4xi32>
    tt.return
  }

  // CHECK-LABEL: @ordinary_addp_fixed_scale
  // CHECK:       llvm.call_intrinsic "llvm.aarch64.neon.addp.v4i32"
  // CHECK:       %[[BITS:.*]] = arith.constant 4 : i32
  // CHECK:       llvm.call_intrinsic "llvm.aarch64.neon.vcvtfxs2fp"({{.*}}, %[[BITS]])
  // CHECK-NOT:   vector.multi_reduction
  // CHECK-NOT:   arith.mulf
  tt.func public @ordinary_addp_fixed_scale(
      %low: vector<4xi32>, %high: vector<4xi32>) {
    %zero = arith.constant dense<0> : vector<4xi32>
    %scale = arith.constant dense<6.250000e-02> : vector<4xf32>
    %joined = vector.interleave %low, %high :
      vector<4xi32> -> vector<8xi32>
    %join_shape = vector.shape_cast %joined :
      vector<8xi32> to vector<4x2xi32>
    %transposed = vector.transpose %join_shape, [1, 0] :
      vector<4x2xi32> to vector<2x4xi32>
    %pairs = vector.shape_cast %transposed :
      vector<2x4xi32> to vector<4x2xi32>
    %sum = vector.multi_reduction <add>, %pairs, %zero [1] :
      vector<4x2xi32> to vector<4xi32>
    %converted = arith.sitofp %sum : vector<4xi32> to vector<4xf32>
    %scaled = arith.mulf %converted, %scale : vector<4xf32>
    tt.return
  }
}
