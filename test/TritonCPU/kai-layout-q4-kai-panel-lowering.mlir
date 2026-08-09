// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s
// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm="w4-only=true fixed-i8mm=false" | FileCheck %s --check-prefix=DOTONLY

// Model an ordinary tl.dot whose operands reconstruct logical K32 values
// from KAI's physical panels.  The RHS is [K8-segment, N4, packed-K8] and
// the LHS is [K8, M4, 8].  The fixed-NEON rewrite must consume the original
// 16-byte panels directly instead of expanding nibbles through ZIP.

// CHECK-LABEL: @kai_q4_native_panel
// CHECK-COUNT-16: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
// CHECK-NOT: triton_cpu.dot
// DOTONLY-LABEL: @kai_q4_native_panel
// DOTONLY-NOT: llvm.aarch64.neon.smmla
// DOTONLY-COUNT-1: triton_cpu.dot

module {
  tt.func public @kai_q4_native_panel(
      %lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    %shift = arith.constant dense<4> : vector<16x4xi8>
    %mask = arith.constant dense<-16> : vector<16x4xi8>
    %zero = arith.constant dense<0> : vector<4x4xi32>

    %rhs_raw = vector.load %rhs[%c0] : memref<?xi8>, vector<64xi8>
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

    %lhs_raw = vector.load %lhs[%c0] : memref<?xi8>, vector<128xi8>
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

    %dot = triton_cpu.dot %lhs_logical, %rhs_logical, %zero,
      inputPrecision = ieee :
      vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
    %flat = vector.shape_cast %dot : vector<4x4xi32> to vector<16xi32>
    vector.store %flat, %out[%c0] : memref<?xi32>, vector<16xi32>
    tt.return
  }
}
