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

#ifndef TRITON_OP_INTERFACES
#define TRITON_OP_INTERFACES

include "mlir/IR/OpBase.td"


def TransposeOpInterface : OpInterface<"TransposeOpInterface"> {
  let description = [{
    This interface is implemented by operations that perform a transpose.
    It provides methods to access common properties such as the order attribute
    and the source operand.
  }];

  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the source operand of the transposition.",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getSrc",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Get the order of the transposition.",
      /*retType=*/"::mlir::ArrayRef<int32_t>",
      /*methodName=*/"getOrder",
      /*args=*/(ins)>
  ];

  let verify = [{
    return ::mlir::triton::impl::verifyTransposeOpInterface($_op);
  }];
}

def DotOpInterface : OpInterface<"DotOpInterface"> {
  let description = [{
    This interface is implemented by operations that perform a dot product.
  }];

  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the LHS A tensor",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getA",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Get the RHS B tensor",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getB",
      /*args=*/(ins)>,
  InterfaceMethod<
      /*desc=*/"Verify the dimensions of the A and B DotOp operands.",
      /*retType=*/"bool",
      /*methodName=*/"verifyDims",
      /*args=*/(ins)>
  ];

  let verify = [{ return ::mlir::triton::impl::verifyDotOpInterface($_op); }];
}


#endif // TRITON_OP_INTERFACES
