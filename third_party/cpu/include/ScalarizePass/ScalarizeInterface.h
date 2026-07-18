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

#ifndef MLIR_INTERFACES_SCALARIZE_INTERFACE_H_
#define MLIR_INTERFACES_SCALARIZE_INTERFACE_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "mlir/IR/OpDefinition.h"

/// Include the ODS generated interface header files.
#include "cpu/include/ScalarizePass/ScalarizeInterface.h.inc"

namespace mlir {
namespace triton {
namespace cpu {

mlir::Value computeScalarValue(mlir::Operation *scalarizationOp,
                               mlir::Value vals,
                               mlir::ArrayRef<int64_t> indices,
                               mlir::PatternRewriter &rewriter);

mlir::Value computeScalarValue(mlir::Operation *scalarizationOp,
                               mlir::Value vals, mlir::ValueRange indices,
                               mlir::PatternRewriter &rewriter);

bool canComputeScalarValue(mlir::Value vals);
} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // MLIR_INTERFACES_SCALARIZE_INTERFACE_H_
