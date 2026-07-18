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

// RUN: triton-opt %s -split-input-file -triton-cpu-convert-reduction -canonicalize

// Regression test: Check that we handle consecutive calls to tt.reduce with
// different types & number of arguments.

module {
  tt.func public @triton_(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xi32>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      tt.reduce.return %arg3 : f32
    }) : (tensor<1x4xf32>) -> tensor<1xf32>
    %1:2 = "tt.reduce"(%arg0, %arg1) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: i32, %arg5: f32, %arg6: i32):
      tt.reduce.return %arg3, %arg4 : f32, i32
    }) : (tensor<1x4xf32>, tensor<1x4xi32>) -> (tensor<1xf32>, tensor<1xi32>)
    tt.return
  }
}
