// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s
// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm="w4-only=true fixed-i8mm=true" | FileCheck %s --check-prefix=FIXEDONLY

// Model the ordinary Triton 3.7 join/permute graph for KAI's qai8dxp and
// qsi8cxp prefill panels.  The target-aware rewrite must remove the logical
// matrix reconstruction and carry sixteen native 2x2 SMMLA accumulators
// directly through the K loop.

// CHECK-LABEL: @kai_w8_prefill_loop
// CHECK-COUNT-1: scf.for
// CHECK-SAME: iter_args(
// CHECK-SAME: !llvm.vec<? x 4 x  i32>
// CHECK-COUNT-5: vector.load
// CHECK-NOT: vector.interleave
// CHECK-NOT: vector.transpose
// CHECK-COUNT-64: llvm.call_intrinsic "llvm.aarch64.sve.smmla.nxv4i32"
// CHECK-NOT: triton_cpu.dot
// FIXEDONLY-LABEL: @kai_w8_prefill_loop
// FIXEDONLY-NOT: llvm.aarch64.sve.smmla
// FIXEDONLY-COUNT-1: triton_cpu.dot

// CHECK-LABEL: @kai_w8_prefill_m4_loop
// CHECK-COUNT-1: scf.for
// CHECK-SAME: iter_args(
// CHECK-COUNT-2: vector.load
// CHECK-COUNT-16: llvm.call_intrinsic "llvm.aarch64.sve.smmla.nxv4i32"
// CHECK-NOT: triton_cpu.dot
// FIXEDONLY-LABEL: @kai_w8_prefill_m4_loop
// FIXEDONLY-NOT: llvm.aarch64.sve.smmla
// FIXEDONLY-COUNT-1: triton_cpu.dot

// CHECK-LABEL: @kai_w8_prefill_m8_loop
// CHECK-COUNT-1: scf.for
// CHECK-SAME: iter_args(
// CHECK-COUNT-3: vector.load
// CHECK-NOT: vector.interleave
// CHECK-NOT: vector.transpose
// CHECK-COUNT-32: llvm.call_intrinsic "llvm.aarch64.sve.smmla.nxv4i32"
// CHECK-NOT: triton_cpu.dot
// FIXEDONLY-LABEL: @kai_w8_prefill_m8_loop
// FIXEDONLY-NOT: llvm.aarch64.sve.smmla
// FIXEDONLY-COUNT-1: triton_cpu.dot

// CHECK-LABEL: @kai_w8_prefill_m12_loop
// CHECK-COUNT-1: scf.for
// CHECK-SAME: iter_args(
// CHECK-NOT: vector.interleave
// CHECK-NOT: vector.transpose
// CHECK-COUNT-48: llvm.call_intrinsic "llvm.aarch64.sve.smmla.nxv4i32"
// CHECK-NOT: triton_cpu.dot
// FIXEDONLY-LABEL: @kai_w8_prefill_m12_loop
// FIXEDONLY-NOT: llvm.aarch64.sve.smmla
// FIXEDONLY-COUNT-2: triton_cpu.dot

module {
  tt.func public @kai_w8_prefill_loop(
      %lhs0: memref<?xi8>, %lhs1: memref<?xi8>,
      %lhs2: memref<?xi8>, %lhs3: memref<?xi8>,
      %rhs: memref<?xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant dense<0> : vector<16x4xi32>
    %result = scf.for %iv = %c0 to %c32 step %c1
        iter_args(%acc = %zero) -> (vector<16x4xi32>) {
      %lhs0_raw = vector.load %lhs0[%iv] : memref<?xi8>, vector<128xi8>
      %lhs0_panel = vector.shape_cast %lhs0_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs0_transpose = vector.transpose %lhs0_panel, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs0_logical = vector.shape_cast %lhs0_transpose :
        vector<4x4x8xi8> to vector<4x32xi8>

      %lhs1_raw = vector.load %lhs1[%iv] : memref<?xi8>, vector<128xi8>
      %lhs1_panel = vector.shape_cast %lhs1_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs1_transpose = vector.transpose %lhs1_panel, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs1_logical = vector.shape_cast %lhs1_transpose :
        vector<4x4x8xi8> to vector<4x32xi8>

      %lhs2_raw = vector.load %lhs2[%iv] : memref<?xi8>, vector<128xi8>
      %lhs2_panel = vector.shape_cast %lhs2_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs2_transpose = vector.transpose %lhs2_panel, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs2_logical = vector.shape_cast %lhs2_transpose :
        vector<4x4x8xi8> to vector<4x32xi8>

      %lhs3_raw = vector.load %lhs3[%iv] : memref<?xi8>, vector<128xi8>
      %lhs3_panel = vector.shape_cast %lhs3_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs3_transpose = vector.transpose %lhs3_panel, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs3_logical = vector.shape_cast %lhs3_transpose :
        vector<4x4x8xi8> to vector<4x32xi8>

      %lhs01_join = vector.interleave %lhs0_logical, %lhs1_logical :
        vector<4x32xi8> -> vector<4x64xi8>
      %lhs01_join_shape = vector.shape_cast %lhs01_join :
        vector<4x64xi8> to vector<4x32x2xi8>
      %lhs01_transpose = vector.transpose %lhs01_join_shape, [2, 0, 1] :
        vector<4x32x2xi8> to vector<2x4x32xi8>
      %lhs01 = vector.shape_cast %lhs01_transpose :
        vector<2x4x32xi8> to vector<8x32xi8>

      %lhs23_join = vector.interleave %lhs2_logical, %lhs3_logical :
        vector<4x32xi8> -> vector<4x64xi8>
      %lhs23_join_shape = vector.shape_cast %lhs23_join :
        vector<4x64xi8> to vector<4x32x2xi8>
      %lhs23_transpose = vector.transpose %lhs23_join_shape, [2, 0, 1] :
        vector<4x32x2xi8> to vector<2x4x32xi8>
      %lhs23 = vector.shape_cast %lhs23_transpose :
        vector<2x4x32xi8> to vector<8x32xi8>

      %lhs_join = vector.interleave %lhs01, %lhs23 :
        vector<8x32xi8> -> vector<8x64xi8>
      %lhs_join_shape = vector.shape_cast %lhs_join :
        vector<8x64xi8> to vector<8x32x2xi8>
      %lhs_transpose = vector.transpose %lhs_join_shape, [2, 0, 1] :
        vector<8x32x2xi8> to vector<2x8x32xi8>
      %lhs = vector.shape_cast %lhs_transpose :
        vector<2x8x32xi8> to vector<16x32xi8>

      %rhs_raw = vector.load %rhs[%iv] : memref<?xi8>, vector<128xi8>
      %rhs_panel = vector.shape_cast %rhs_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %rhs_transpose = vector.transpose %rhs_panel, [0, 2, 1] :
        vector<4x4x8xi8> to vector<4x8x4xi8>
      %rhs_logical = vector.shape_cast %rhs_transpose :
        vector<4x8x4xi8> to vector<32x4xi8>
      %next = triton_cpu.dot %lhs, %rhs_logical, %acc,
        inputPrecision = ieee :
        vector<16x32xi8> * vector<32x4xi8> -> vector<16x4xi32>
      scf.yield %next : vector<16x4xi32>
    }
    tt.return
  }

  tt.func public @kai_w8_prefill_m4_loop(
      %lhs: memref<?xi8>, %rhs: memref<?xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant dense<0> : vector<4x4xi32>
    %result = scf.for %iv = %c0 to %c32 step %c1
        iter_args(%acc = %zero) -> (vector<4x4xi32>) {
      %lhs_raw = vector.load %lhs[%iv] : memref<?xi8>, vector<128xi8>
      %lhs_panel = vector.shape_cast %lhs_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs_transpose = vector.transpose %lhs_panel, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs_logical = vector.shape_cast %lhs_transpose :
        vector<4x4x8xi8> to vector<4x32xi8>

      %rhs_raw = vector.load %rhs[%iv] : memref<?xi8>, vector<128xi8>
      %rhs_panel = vector.shape_cast %rhs_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %rhs_transpose = vector.transpose %rhs_panel, [0, 2, 1] :
        vector<4x4x8xi8> to vector<4x8x4xi8>
      %rhs_logical = vector.shape_cast %rhs_transpose :
        vector<4x8x4xi8> to vector<32x4xi8>
      %next = triton_cpu.dot %lhs_logical, %rhs_logical, %acc,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      scf.yield %next : vector<4x4xi32>
    }
    tt.return
  }

  tt.func public @kai_w8_prefill_m8_loop(
      %lhs0: memref<?xi8>, %lhs1: memref<?xi8>, %rhs: memref<?xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant dense<0> : vector<8x4xi32>
    %result = scf.for %iv = %c0 to %c32 step %c1
        iter_args(%acc = %zero) -> (vector<8x4xi32>) {
      %lhs0_raw = vector.load %lhs0[%iv] : memref<?xi8>, vector<128xi8>
      %lhs0_panel = vector.shape_cast %lhs0_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs0_transpose = vector.transpose %lhs0_panel, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs0_logical = vector.shape_cast %lhs0_transpose :
        vector<4x4x8xi8> to vector<4x32xi8>

      %lhs1_raw = vector.load %lhs1[%iv] : memref<?xi8>, vector<128xi8>
      %lhs1_panel = vector.shape_cast %lhs1_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs1_transpose = vector.transpose %lhs1_panel, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs1_logical = vector.shape_cast %lhs1_transpose :
        vector<4x4x8xi8> to vector<4x32xi8>
      %lhs_join = vector.interleave %lhs0_logical, %lhs1_logical :
        vector<4x32xi8> -> vector<4x64xi8>
      %lhs_join_shape = vector.shape_cast %lhs_join :
        vector<4x64xi8> to vector<4x32x2xi8>
      %lhs_transpose = vector.transpose %lhs_join_shape, [2, 0, 1] :
        vector<4x32x2xi8> to vector<2x4x32xi8>
      %lhs_logical = vector.shape_cast %lhs_transpose :
        vector<2x4x32xi8> to vector<8x32xi8>

      %rhs_raw = vector.load %rhs[%iv] : memref<?xi8>, vector<128xi8>
      %rhs_panel = vector.shape_cast %rhs_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %rhs_transpose = vector.transpose %rhs_panel, [0, 2, 1] :
        vector<4x4x8xi8> to vector<4x8x4xi8>
      %rhs_logical = vector.shape_cast %rhs_transpose :
        vector<4x8x4xi8> to vector<32x4xi8>
      %next = triton_cpu.dot %lhs_logical, %rhs_logical, %acc,
        inputPrecision = ieee :
        vector<8x32xi8> * vector<32x4xi8> -> vector<8x4xi32>
      scf.yield %next : vector<8x4xi32>
    }
    tt.return
  }

  tt.func public @kai_w8_prefill_m12_loop(
      %lhs0_raw: vector<128xi8>, %lhs1_raw: vector<128xi8>,
      %lhs2_raw: vector<128xi8>, %rhs_raw: vector<128xi8>) {
    %lhs0_panel = vector.shape_cast %lhs0_raw :
      vector<128xi8> to vector<4x4x8xi8>
    %lhs0_transpose = vector.transpose %lhs0_panel, [1, 0, 2] :
      vector<4x4x8xi8> to vector<4x4x8xi8>
    %lhs0_logical = vector.shape_cast %lhs0_transpose :
      vector<4x4x8xi8> to vector<4x32xi8>
    %lhs1_panel = vector.shape_cast %lhs1_raw :
      vector<128xi8> to vector<4x4x8xi8>
    %lhs1_transpose = vector.transpose %lhs1_panel, [1, 0, 2] :
      vector<4x4x8xi8> to vector<4x4x8xi8>
    %lhs1_logical = vector.shape_cast %lhs1_transpose :
      vector<4x4x8xi8> to vector<4x32xi8>
    %lhs2_panel = vector.shape_cast %lhs2_raw :
      vector<128xi8> to vector<4x4x8xi8>
    %lhs2_transpose = vector.transpose %lhs2_panel, [1, 0, 2] :
      vector<4x4x8xi8> to vector<4x4x8xi8>
    %lhs2_logical = vector.shape_cast %lhs2_transpose :
      vector<4x4x8xi8> to vector<4x32xi8>
    %lhs01_join = vector.interleave %lhs0_logical, %lhs1_logical :
      vector<4x32xi8> -> vector<4x64xi8>
    %lhs01_shape = vector.shape_cast %lhs01_join :
      vector<4x64xi8> to vector<4x32x2xi8>
    %lhs01_transpose = vector.transpose %lhs01_shape, [2, 0, 1] :
      vector<4x32x2xi8> to vector<2x4x32xi8>
    %lhs8 = vector.shape_cast %lhs01_transpose :
      vector<2x4x32xi8> to vector<8x32xi8>
    %rhs_panel = vector.shape_cast %rhs_raw :
      vector<128xi8> to vector<4x4x8xi8>
    %rhs_transpose = vector.transpose %rhs_panel, [0, 2, 1] :
      vector<4x4x8xi8> to vector<4x8x4xi8>
    %rhs = vector.shape_cast %rhs_transpose :
      vector<4x8x4xi8> to vector<32x4xi8>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %zero8 = arith.constant dense<0> : vector<8x4xi32>
    %zero4 = arith.constant dense<0> : vector<4x4xi32>
    %result:2 = scf.for %iv = %c0 to %c32 step %c1
        iter_args(%acc8 = %zero8, %acc4 = %zero4) ->
        (vector<8x4xi32>, vector<4x4xi32>) {
      %next8 = triton_cpu.dot %lhs8, %rhs, %acc8,
        inputPrecision = ieee :
        vector<8x32xi8> * vector<32x4xi8> -> vector<8x4xi32>
      %next4 = triton_cpu.dot %lhs2_logical, %rhs, %acc4,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      scf.yield %next8, %next4 : vector<8x4xi32>, vector<4x4xi32>
    }
    tt.return
  }
}
