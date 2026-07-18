// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// RUN: triton-opt --split-input-file %s | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_create_single
  // CHECK: nvws.aref.create
  tt.func @aref_create_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    tt.return
  }

}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_get_single
  // CHECK: nvws.aref.get
  tt.func @aref_get_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %1 = nvws.aref.get %0 as (%b0 : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
      nvws.aref.return %b0 : !ttg.memdesc<1x64x16xf16, #shared0, #smem>
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>) -> !ttg.memdesc<1x64x16xf16, #shared0, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put_single
  // CHECK: nvws.aref.put
  tt.func @aref_put_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    nvws.aref.put %0 as (%b0 : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
      nvws.aref.return
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>) -> ()
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_create_batch
  // CHECK: nvws.aref.create
  tt.func @aref_create_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    tt.return
  }

}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_get_batch
  // CHECK: nvws.aref.get
  tt.func @aref_get_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    %1 = nvws.aref.get %0[%c0_i32] as (%b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<16x32xf16, #shared0, #smem>) {
      nvws.aref.return %b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>, i32) -> !ttg.memdesc<64x16xf16, #shared0, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put_batch
  // CHECK: nvws.aref.put
  tt.func @aref_put_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    nvws.aref.put %0[%c0_i32] as (%b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<16x32xf16, #shared0, #smem>) {
      nvws.aref.return
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>, i32) -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put_tensor
  // CHECK: nvws.aref.put
  tt.func @aref_put_tensor(%d : tensor<1x64x16xf16>, %e : tensor<1x16x32xf16>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>
    nvws.aref.put %0[%c0_i32] as (%b0 : tensor<64x16xf16>, %b1 : tensor<16x32xf16>) {
      %1 = math.exp %b0 : tensor<64x16xf16>
      %2 = math.cos %b1 : tensor<16x32xf16>
      nvws.aref.return %1, %2 : tensor<64x16xf16>, tensor<16x32xf16>
    } : (!nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>, i32) -> ()
    tt.return
  }
}

// -----


// CHECK-LABEL: @warp_group_nothing
tt.func @warp_group_nothing() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  tt.return
}

// CHECK-LABEL: @warp_1_partition
tt.func @warp_1_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT:  num_warps(4) {
  partition0  num_warps(4) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}

// CHECK-LABEL: @warp_2_partition
tt.func @warp_2_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT: partition0  num_warps(8) {
  partition0  num_warps(8) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition1 num_warps(4) {
  partition1 num_warps(4) {
  // CHECK-NEXT:   nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}
