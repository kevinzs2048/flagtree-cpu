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

// RUN: triton-opt -split-input-file %s --convert-triton-amdgpu-to-llvm='arch=gfx942' -verify-diagnostics

// Invalid size
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_size_input(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{sizes [256, 2] must be a multiple of shapePerCTATile [256, 16]}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x2xi32, #blocked1>
  tt.return
}

// -----

// Invalid zero source dimension
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_size_input(%arg0: tensor<256x0xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{source tensor dimension size zero at dimension 1}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x0xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}

// -----

// Invalid zero result dimension
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_size_input(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result tensor dimension size zero at dimension 1}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x0xi32, #blocked1>
  tt.return
}

// -----

// Invalid offset, not multiple of shapePerTile
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_offset_input(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{offset [0, 5] must be a multiple of shapePerCTATile [256, 16]}}
  %1 = amdgpu.extract_slice %arg0 [0,5] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}

// -----

// Invalid offset, out of bounds for dimension
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_offset_input(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{invalid offset 128 at dimension 1}}
  %1 = amdgpu.extract_slice %arg0 [0,128] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}

// -----

// Invalid result layout
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_layout(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result layout must match source layout}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked2>
  tt.return
}

// -----

// Invalid result element type
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_element_type(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result element type must match source element type}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x16xi64, #blocked1>
  tt.return
}

// -----

// Invalid result rank
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_rank(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result rank must be equal to source rank}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x16x2xi32, #blocked1>
  tt.return
}

// -----

// Invalid result shape
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_rank(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result shape cannot be larger than input shape at dimension 1}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x256xi32, #blocked1>
  tt.return
}

// -----

// Invalid rank
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_rank(%arg0: tensor<256x128x2xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{currently only 2D tensors are supported}}
  %1 = amdgpu.extract_slice %arg0 [0,0,0] : tensor<256x128x2xi32, #blocked1> to tensor<256x16x2xi32, #blocked1>
  tt.return
}

// -----

// Invalid non static offset
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_non_static_offset(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}, %arg1: i32) {
  // expected-error @+2 {{expected ']'}}
  // expected-error @+1 {{expected integer value}}
  %2 = amdgpu.extract_slice %arg0 [%arg1, 0] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}
