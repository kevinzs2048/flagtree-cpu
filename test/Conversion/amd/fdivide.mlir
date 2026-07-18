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

// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fdiv_f32(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_fdiv_f32
    // CHECK: llvm.amdgcn.div.scale.f32
    // CHECK: llvm.amdgcn.div.scale.f32
    // CHECK: llvm.amdgcn.rcp.f32
    // CHECK: llvm.fmul
    // CHECK: llvm.amdgcn.div.fmas.f32
    // CHECK: llvm.amdgcn.div.fixup.f32
    // CHECK-NOT: llvm.fdiv
    %0 = arith.divf %arg0, %arg1 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fdiv_f64(%arg0: tensor<64xf64, #blocked>, %arg1: tensor<64xf64, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_fdiv_f64
    // CHECK: llvm.fdiv
    %0 = arith.divf %arg0, %arg1 : tensor<64xf64, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_div_rn(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_div_rn
    // CHECK: llvm.fdiv
    %0 = tt.precise_divf %arg0, %arg1 : tensor<64xf32, #blocked>
    tt.return
  }
}
