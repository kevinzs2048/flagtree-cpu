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

#ifndef MLIR_SCALARIZEINTERFACE
#define MLIR_SCALARIZEINTERFACE

include "mlir/IR/OpBase.td"

def ScalarizeInterface : OpInterface<"ScalarizeInterface"> {
  let description = [{
    Interface for allowing operations to expose information needed to
    scalarize them or in simpler terms inserts SCF loops to reduce amount of
    generated ir. Similar with checking operands of specific operations for
    constancy - to understand is it possible to put it inside of loop's body.
  }];
  let cppNamespace = "mlir::triton::cpu";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Checks operand and is ScalarizeInterface registered for this operation.
        }],
        /*retType=*/"bool",
        /*methodName=*/"canComputeScalarValue",
        /*args=*/(ins
          "mlir::Value ":$vals)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns value that can be  put inside of generated cycle and creates required constants.
          Can go throught operands to check type of passed values. Implementation for static indeces.
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"computeScalarValue",
        /*args=*/(ins
          "mlir::Value ":$vals,
          "mlir::ArrayRef<int64_t>  ":$indices,
          "mlir::PatternRewriter &":$rewriter)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns value that can be  put inside of generated cycle and creates required constants.
          Can go throught operands to check type of passed values. Implementation for dynamic indices
          which is in common used in loops to iterate with Inductional Variable.
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"computeScalarValueForLoop",
        /*args=*/(ins
          "mlir::Value ":$vals,
          "mlir::ValueRange  ":$indices,
          "mlir::PatternRewriter &":$rewriter)
      >
  ];
}

#endif // MLIR_SCALARIZEINTERFACE
