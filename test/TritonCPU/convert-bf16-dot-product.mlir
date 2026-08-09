// RUN: triton-opt %s -triton-cpu-convert-dot-product | FileCheck %s

// Regression test for the ordinary-Triton attention form.  Extending bf16
// before a shape cast and multiply must still become native BFDOT operations.

// CHECK-LABEL: @attention_dot
// CHECK-NOT:   arith.extf
// CHECK-COUNT-16: llvm.call_intrinsic "llvm.aarch64.neon.bfdot.v4f32.v8bf16"
// CHECK:       llvm.call_intrinsic "llvm.aarch64.neon.faddv.f32.v4f32"
// CHECK-NOT:   vector.multi_reduction

module {
  tt.func public @attention_dot(
      %query: vector<128xbf16>,
      %key: vector<1x128xbf16>,
      %acc: vector<1xf32>) -> vector<1xf32> {
    %query_fp32 = arith.extf %query : vector<128xbf16> to vector<128xf32>
    %query_row = vector.shape_cast %query_fp32 : vector<128xf32> to vector<1x128xf32>
    %key_fp32 = arith.extf %key : vector<1x128xbf16> to vector<1x128xf32>
    %product = arith.mulf %key_fp32, %query_row : vector<1x128xf32>
    %sum = vector.multi_reduction <add>, %product, %acc [1] :
        vector<1x128xf32> to vector<1xf32>
    tt.return %sum : vector<1xf32>
  }
}
