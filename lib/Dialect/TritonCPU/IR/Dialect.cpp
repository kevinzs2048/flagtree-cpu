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

#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonCPU/IR/TritonCPUAttrDefs.cpp.inc"

void ExtractMemRefOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {}

void ExtractIndicesOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {}

/// Parse an attribute registered to this dialect.
::mlir::Attribute
TritonCPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                 ::mlir::Type type) const {
  llvm_unreachable("parse stub called");
}

/// Print an attribute registered to this dialect.
void TritonCPUDialect::printAttribute(::mlir::Attribute attr,
                                      ::mlir::DialectAsmPrinter &os) const {
  llvm_unreachable("print stub called");
}

void ExtractIndicesOp::build(::mlir::OpBuilder &builder,
                             ::mlir::OperationState &state, Value src) {
  assert(triton::isTensorPointerType(src.getType()) &&
         "Unexecpeted source type");
  auto tensorTy = dyn_cast<RankedTensorType>(
      dyn_cast<PointerType>(src.getType()).getPointeeType());
  SmallVector<Type> resTypes(tensorTy.getRank(), builder.getIndexType());
  build(builder, state, resTypes, src);
}

void TritonCPUDialect::initialize() {
  registerTypes();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonCPU/IR/TritonCPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonCPU/IR/Ops.cpp.inc"
#include "triton/Dialect/TritonCPU/IR/OpsEnums.cpp.inc"
      >();
}

// verify TritonCPU ops
LogicalResult TritonCPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
