// RUN: triton-opt %s -triton-cpu-strip-mine-vector-loops="native-vector-bits=128 unroll-factor=1" | FileCheck %s --check-prefix=SAFE
// RUN: triton-opt %s -triton-cpu-strip-mine-vector-loops="native-vector-bits=128 unroll-factor=1 allow-fp-reduction-reassociation=true" | FileCheck %s --check-prefix=REASSOC

// The default mode narrows elementwise loops to native SIMD width while
// preserving floating-point reduction order.

// SAFE-LABEL: @elementwise
// SAFE:       step %c4_i32
// SAFE:       memref<4xf32>
// SAFE:       vector<4xf32>
// SAFE:       vector.store

// SAFE-LABEL: @fp_reduction
// SAFE:       step %c128_i32
// SAFE:       vector<128xf32>
// SAFE:       vector.reduction

// The opt-in reassociation mode also strip-mines reduction loops.

// REASSOC-LABEL: @elementwise
// REASSOC:       step %c4_i32
// REASSOC:       vector<4xf32>

// REASSOC-LABEL: @fp_reduction
// REASSOC:       step %c4_i32
// REASSOC:       vector<4xf32>
// REASSOC:       vector.reduction

module {
  tt.func public @elementwise(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    scf.for %iv = %c0_i32 to %arg2 step %c128_i32 : i32 {
      %src = tt.addptr %arg0, %iv : !tt.ptr<f32>, i32
      %src_memref = triton_cpu.ptr_to_memref %src : <f32> -> memref<128xf32>
      %value = vector.load %src_memref[%c0] : memref<128xf32>, vector<128xf32>
      %square = arith.mulf %value, %value : vector<128xf32>
      %dst = tt.addptr %arg1, %iv : !tt.ptr<f32>, i32
      %dst_memref = triton_cpu.ptr_to_memref %dst : <f32> -> memref<128xf32>
      vector.store %square, %dst_memref[%c0] : memref<128xf32>, vector<128xf32>
    }
    tt.return
  }

  tt.func public @fp_reduction(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    scf.for %iv = %c0_i32 to %arg1 step %c128_i32 : i32 {
      %src = tt.addptr %arg0, %iv : !tt.ptr<f32>, i32
      %src_memref = triton_cpu.ptr_to_memref %src : <f32> -> memref<128xf32>
      %value = vector.load %src_memref[%c0] : memref<128xf32>, vector<128xf32>
      %sum = vector.reduction <add>, %value, %cst : vector<128xf32> into f32
    }
    tt.return
  }
}
