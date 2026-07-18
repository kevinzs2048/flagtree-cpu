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

#include "cpu/include/TritonCPUTransforms/OptCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include <iostream>
#include <utility>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTGENERIC
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class DotConversionTarget : public ConversionTarget {
public:
  explicit DotConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addIllegalOp<cpu::DotOp>();
  }
};

struct DotOpConversion : public OpConversionPattern<cpu::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    Value a = op.getA();
    Value b = op.getB();
    Value c = op.getC();
    VectorType aType = cast<VectorType>(a.getType());
    VectorType bType = cast<VectorType>(b.getType());
    VectorType cType = cast<VectorType>(c.getType());

    uint32_t rank = aType.getRank();
    if (rank == 2) {
      auto aMap = AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx);
      auto bMap = AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx);
      auto cMap = AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx);
      auto iteratorTypes = rewriter.getArrayAttr(
          {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
           vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
           vector::IteratorTypeAttr::get(ctx,
                                         vector::IteratorType::reduction)});
      rewriter.replaceOpWithNewOp<vector::ContractionOp>(
          op, a, b, c, rewriter.getAffineMapArrayAttr({aMap, bMap, cMap}),
          iteratorTypes);
      return success();
    } else if (rank == 3) {
      auto aMap = AffineMap::getMultiDimMapWithTargets(4, {0, 1, 3}, ctx);
      auto bMap = AffineMap::getMultiDimMapWithTargets(4, {0, 3, 2}, ctx);
      auto cMap = AffineMap::getMultiDimMapWithTargets(4, {0, 1, 2}, ctx);
      auto iteratorTypes = rewriter.getArrayAttr(
          {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
           vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
           vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
           vector::IteratorTypeAttr::get(ctx,
                                         vector::IteratorType::reduction)});
      rewriter.replaceOpWithNewOp<vector::ContractionOp>(
          op, a, b, c, rewriter.getAffineMapArrayAttr({aMap, bMap, cMap}),
          iteratorTypes);
      return success();
    }

    return failure();
  }

  SmallVector<Value> deinterleave(Location loc, ArrayRef<Value> vals,
                                  ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> res;
    for (auto &val : vals) {
      auto op = rewriter.create<vector::DeinterleaveOp>(loc, val);
      res.push_back(op.getResult(0));
      res.push_back(op.getResult(1));
    }
    return res;
  }
};

struct ConvertDotGeneric
    : public triton::cpu::impl::ConvertDotGenericBase<ConvertDotGeneric> {
  using ConvertDotGenericBase::ConvertDotGenericBase;

  ConvertDotGeneric() : ConvertDotGenericBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    DotConversionTarget convTarget(*context);
    RewritePatternSet patterns(context);
    patterns.add<DotOpConversion>(context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotGeneric() {
  return std::make_unique<ConvertDotGeneric>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
