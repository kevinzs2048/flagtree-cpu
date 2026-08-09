// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s

// The SVE2 path must remain a rolled 8x8x8 microkernel.  In particular, do
// not expand the complete Triton tile into hundreds of simultaneously-live
// fixed-vector SSA values.
//
// CHECK-LABEL: @s8_gemm_16x16x32
// CHECK-NOT:       vector.transfer_write
// CHECK:       memref.alloca() {alignment = 64 : i64} : memref<4x4x16xi8>
// CHECK:       scf.for {{.*}} step {{.*}} {
// CHECK:         scf.for {{.*}} step {{.*}} {
// CHECK-COUNT-8:   vector.load {{.*}} vector<8xi8>
// CHECK:           vector.shuffle
// CHECK:           vector.bitcast
// CHECK-NOT:       vector.transpose
// CHECK:         scf.for {{.*}} step {{.*}} {
// CHECK:           llvm.mlir.zero
// CHECK:           scf.for {{.*}} iter_args(
// CHECK-COUNT-16:    llvm.call_intrinsic "llvm.aarch64.sve.smmla.nxv4i32"
// CHECK-NOT:     triton_cpu.dot

#loc = loc(unknown)
module {
  tt.func public @s8_gemm_16x16x32(
      %lhs: memref<16x32xi8, strided<[?, 1]>>,
      %rhs: memref<32x16xi8, strided<[?, 1]>>) {
    %c0 = arith.constant 0 : index
    %zero_i8 = arith.constant 0 : i8
    %zero = arith.constant dense<0> : vector<16x16xi32>
    %a = vector.transfer_read %lhs[%c0, %c0], %zero_i8
      {in_bounds = [true, true]} :
      memref<16x32xi8, strided<[?, 1]>>, vector<16x32xi8>
    %b = vector.transfer_read %rhs[%c0, %c0], %zero_i8
      {in_bounds = [true, true]} :
      memref<32x16xi8, strided<[?, 1]>>, vector<32x16xi8>
    %result = triton_cpu.dot %a, %b, %zero, inputPrecision = ieee :
      vector<16x32xi8> * vector<32x16xi8> -> vector<16x16xi32>
    tt.return
  }

  // Q4 prefill uses a four-column output panel to keep the surrounding
  // groupwise FP32 accumulator below the register-pressure cliff.
  // CHECK-LABEL: @s8_gemm_16x4x32
  // CHECK:       scf.for {{.*}} step {{.*}} {
  // CHECK:         scf.for {{.*}} step {{.*}} {
  // CHECK-COUNT-8:   vector.load {{.*}} vector<4xi8>
  // CHECK:         scf.for {{.*}} step {{.*}} {
  // CHECK-COUNT-8:    llvm.call_intrinsic "llvm.aarch64.sve.smmla.nxv4i32"
  // CHECK-NOT:     triton_cpu.dot
  tt.func public @s8_gemm_16x4x32(
      %lhs: memref<16x32xi8, strided<[?, 1]>>,
      %rhs: memref<32x4xi8, strided<[?, 1]>>) {
    %c0 = arith.constant 0 : index
    %zero_i8 = arith.constant 0 : i8
    %zero = arith.constant dense<0> : vector<16x4xi32>
    %a = vector.transfer_read %lhs[%c0, %c0], %zero_i8
      {in_bounds = [true, true]} :
      memref<16x32xi8, strided<[?, 1]>>, vector<16x32xi8>
    %b = vector.transfer_read %rhs[%c0, %c0], %zero_i8
      {in_bounds = [true, true]} :
      memref<32x4xi8, strided<[?, 1]>>, vector<32x4xi8>
    %result = triton_cpu.dot %a, %b, %zero, inputPrecision = ieee :
      vector<16x32xi8> * vector<32x4xi8> -> vector<16x4xi32>
    tt.return
  }

  // A normal Triton W8 decode kernel loads an output-major [N,4] physical
  // panel, reshapes it and transposes it to the logical [4,N] dot operand.
  // The lowering must consume the physical buffer directly: each adjacent
  // 4x4 byte slice is already in SDOT lane order.  Materializing the full
  // transpose in a compiler-owned stack buffer is both redundant and
  // catastrophic for wide output tiles.
  // CHECK-LABEL: @s8_gemv_packed_n4_128
  // CHECK-NOT:   memref.alloca() {{.*}} memref<4x128xi8>
  // CHECK:       scf.for {{.*}} step {{.*}} {
  // CHECK:         arith.muli {{.*}} : index
  // CHECK:         vector.load %{{.*}}[{{.*}}] : memref<512xi8>, vector<16xi8>
  // CHECK:         llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // CHECK-NOT:   vector.transfer_write {{.*}} memref<4x128xi8>
  // CHECK-NOT:   triton_cpu.dot
  tt.func public @s8_gemv_packed_n4_128(
      %lhs: memref<1x4xi8, strided<[?, 1]>>,
      %rhs: memref<512xi8>) {
    %c0 = arith.constant 0 : index
    %zero_i8 = arith.constant 0 : i8
    %zero = arith.constant dense<0> : vector<1x128xi32>
    %a = vector.transfer_read %lhs[%c0, %c0], %zero_i8
      {in_bounds = [true, true]} :
      memref<1x4xi8, strided<[?, 1]>>, vector<1x4xi8>
    %flat = vector.load %rhs[%c0] : memref<512xi8>, vector<512xi8>
    %physical = vector.shape_cast %flat :
      vector<512xi8> to vector<128x4xi8>
    %logical = vector.transpose %physical, [1, 0] :
      vector<128x4xi8> to vector<4x128xi8>
    %result = triton_cpu.dot %a, %logical, %zero, inputPrecision = ieee :
      vector<1x4xi8> * vector<4x128xi8> -> vector<1x128xi32>
    tt.return
  }
}
