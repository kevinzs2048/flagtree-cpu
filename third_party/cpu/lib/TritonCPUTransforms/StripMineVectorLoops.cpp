#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_STRIPMINEVECTORLOOPS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "triton-cpu-strip-mine-vector-loops"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

static bool isWideType(Type type, int64_t width) {
  if (auto vectorType = dyn_cast<VectorType>(type))
    return vectorType.getRank() == 1 && !vectorType.isScalable() &&
           vectorType.getDimSize(0) == width;
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return memrefType.getRank() == 1 && memrefType.hasStaticShape() &&
           memrefType.getDimSize(0) == width;
  return false;
}

static FailureOr<Type> narrowType(Type type, int64_t oldWidth,
                                  int64_t newWidth) {
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    if (vectorType.getRank() != 1 || vectorType.isScalable())
      return failure();
    if (vectorType.getDimSize(0) == oldWidth)
      return VectorType::get({newWidth}, vectorType.getElementType());
    if (vectorType.getDimSize(0) != 1)
      return failure();
    return type;
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    if (memrefType.getRank() != 1 || !memrefType.hasStaticShape())
      return failure();
    if (memrefType.getDimSize(0) == oldWidth) {
      return MemRefType::get({newWidth}, memrefType.getElementType(),
                             memrefType.getLayout(),
                             memrefType.getMemorySpace());
    }
    if (memrefType.getDimSize(0) != 1)
      return failure();
  }
  return type;
}

static int64_t elementBitWidth(Type type) {
  Type elementType;
  if (auto vectorType = dyn_cast<VectorType>(type))
    elementType = vectorType.getElementType();
  else if (auto memrefType = dyn_cast<MemRefType>(type))
    elementType = memrefType.getElementType();
  else
    return 0;

  if (auto intType = dyn_cast<IntegerType>(elementType))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return floatType.getWidth();
  return 0;
}

static FailureOr<DenseElementsAttr> narrowDenseAttr(DenseElementsAttr attr,
                                                    ShapedType newType) {
  if (attr.isSplat())
    return DenseElementsAttr::get(newType, attr.getSplatValue<Attribute>());

  SmallVector<Attribute> elements;
  elements.reserve(newType.getNumElements());
  for (Attribute value : attr.getValues<Attribute>()) {
    if (elements.size() == static_cast<size_t>(newType.getNumElements()))
      break;
    elements.push_back(value);
  }
  if (elements.size() != static_cast<size_t>(newType.getNumElements()))
    return failure();
  return DenseElementsAttr::get(newType, elements);
}

class LoopNarrower {
public:
  LoopNarrower(IRRewriter &rewriter, int64_t oldWidth, int64_t newWidth)
      : rewriter(rewriter), oldWidth(oldWidth), newWidth(newWidth) {}

  FailureOr<Value> getExternal(Value value, Operation *anchor) {
    if (!isWideType(value.getType(), oldWidth))
      return value;
    if (auto mapped = externalValues.lookup(value))
      return mapped;

    Operation *def = value.getDefiningOp();
    if (!def)
      return failure();
    if (!isa<arith::ConstantOp, vector::SplatOp, vector::BroadcastOp>(def))
      return failure();

    rewriter.setInsertionPoint(anchor);
    SmallVector<Value> operands;
    for (Value operand : def->getOperands()) {
      auto narrowed = getExternal(operand, anchor);
      if (failed(narrowed))
        return failure();
      operands.push_back(*narrowed);
    }

    auto cloned = cloneOperation(def, operands);
    if (failed(cloned))
      return failure();
    for (auto [oldResult, newResult] :
         llvm::zip(def->getResults(), (*cloned)->getResults()))
      externalValues[oldResult] = newResult;
    return externalValues.lookup(value);
  }

  FailureOr<Operation *> cloneOperation(Operation *op, ValueRange operands) {
    if (op->getNumRegions() != 0)
      return failure();

    SmallVector<Type> resultTypes;
    for (Type resultType : op->getResultTypes()) {
      auto converted = narrowType(resultType, oldWidth, newWidth);
      if (failed(converted))
        return failure();
      resultTypes.push_back(*converted);
    }

    NamedAttrList attributes(op->getAttrs());
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto oldAttr = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (oldAttr && !resultTypes.empty() &&
          resultTypes.front() != op->getResult(0).getType()) {
        auto newShapedType = dyn_cast<ShapedType>(resultTypes.front());
        if (!newShapedType)
          return failure();
        auto newAttr = narrowDenseAttr(oldAttr, newShapedType);
        if (failed(newAttr))
          return failure();
        attributes.set("value", *newAttr);
      }
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(attributes);
    return rewriter.create(state);
  }

private:
  IRRewriter &rewriter;
  int64_t oldWidth;
  int64_t newWidth;
  llvm::MapVector<Value, Value> externalValues;
};

static bool hasUnsupportedTypes(scf::ForOp loop, int64_t width) {
  bool unsupported = false;
  loop.getBody()->walk([&](Operation *op) {
    for (Type type :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      if (auto vectorType = dyn_cast<VectorType>(type)) {
        if (vectorType.getRank() != 1 || vectorType.isScalable() ||
            (vectorType.getDimSize(0) != 1 &&
             vectorType.getDimSize(0) != width))
          unsupported = true;
      }
      if (auto memrefType = dyn_cast<MemRefType>(type)) {
        if (memrefType.getRank() != 1 || !memrefType.hasStaticShape() ||
            (memrefType.getDimSize(0) != 1 &&
             memrefType.getDimSize(0) != width))
          unsupported = true;
      }
    }
    if (op->getNumRegions() != 0 && op != loop.getOperation())
      unsupported = true;
  });
  return unsupported;
}

static bool containsFpReduction(scf::ForOp loop) {
  bool found = false;
  loop.getBody()->walk([&](vector::ReductionOp reduction) {
    auto vectorType = cast<VectorType>(reduction.getVector().getType());
    if (isa<FloatType>(vectorType.getElementType()))
      found = true;
  });
  return found;
}

struct StripMineVectorLoops
    : public triton::cpu::impl::StripMineVectorLoopsBase<StripMineVectorLoops> {
  using Base::Base;

  StripMineVectorLoops(int64_t bits, int64_t unroll, bool allowReassociation) {
    nativeVectorBits = bits;
    unrollFactor = unroll;
    allowFpReductionReassociation = allowReassociation;
  }

  void runOnOperation() override {
    if (nativeVectorBits <= 0 || unrollFactor <= 0) {
      getOperation().emitError(
          "vector width and unroll factor must be positive");
      return signalPassFailure();
    }

    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) { loops.push_back(loop); });

    IRRewriter rewriter(&getContext());
    for (scf::ForOp loop : loops) {
      if (!loop || loop->getParentOfType<scf::ForOp>())
        continue;
      auto step = getConstantIntValue(loop.getStep());
      if (!step || *step <= 1 || !llvm::isPowerOf2_64(*step))
        continue;
      int64_t oldWidth = *step;
      if (hasUnsupportedTypes(loop, oldWidth))
        continue;
      if (!allowFpReductionReassociation && containsFpReduction(loop))
        continue;

      int64_t maxElementBits = 0;
      loop.getBody()->walk([&](Operation *op) {
        for (Type type :
             llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()))
          maxElementBits = std::max(maxElementBits, elementBitWidth(type));
      });
      if (maxElementBits <= 0)
        continue;
      int64_t nativeLanes = nativeVectorBits / maxElementBits;
      int64_t newWidth =
          std::min(oldWidth, nativeLanes * static_cast<int64_t>(unrollFactor));
      if (newWidth <= 0 || !llvm::isPowerOf2_64(newWidth) ||
          oldWidth % newWidth != 0 || newWidth == oldWidth)
        continue;

      LoopNarrower narrower(rewriter, oldWidth, newWidth);
      SmallVector<Value> initArgs(loop.getInitArgs());
      rewriter.setInsertionPoint(loop);
      Value newStep = rewriter.create<arith::ConstantIntOp>(
          loop.getLoc(), newWidth, loop.getStep().getType());
      auto newLoop =
          rewriter.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                                      loop.getUpperBound(), newStep, initArgs);

      Block *oldBody = loop.getBody();
      Block *newBody = newLoop.getBody();
      rewriter.eraseOp(newBody->getTerminator());
      IRMapping mapping;
      mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
      for (auto [oldArg, newArg] :
           llvm::zip(oldBody->getArguments().drop_front(),
                     newBody->getArguments().drop_front()))
        mapping.map(oldArg, newArg);

      bool failedToClone = false;
      rewriter.setInsertionPointToEnd(newBody);
      for (Operation &op : oldBody->getOperations()) {
        if (auto yield = dyn_cast<scf::YieldOp>(op)) {
          SmallVector<Value> yielded;
          for (Value operand : yield.getOperands())
            yielded.push_back(mapping.lookupOrDefault(operand));
          rewriter.create<scf::YieldOp>(yield.getLoc(), yielded);
          continue;
        }

        SmallVector<Value> operands;
        for (Value operand : op.getOperands()) {
          if (Value mapped = mapping.lookupOrNull(operand)) {
            operands.push_back(mapped);
            continue;
          }
          // Materialize narrowed loop-invariant vectors before the replacement
          // loop.  Inserting them before the old loop would put them after
          // newLoop and violate dominance for uses in its body.
          auto external = narrower.getExternal(operand, newLoop);
          if (failed(external)) {
            failedToClone = true;
            break;
          }
          operands.push_back(*external);
        }
        if (failedToClone)
          break;
        rewriter.setInsertionPointToEnd(newBody);
        auto cloned = narrower.cloneOperation(&op, operands);
        if (failed(cloned)) {
          failedToClone = true;
          break;
        }
        for (auto [oldResult, newResult] :
             llvm::zip(op.getResults(), (*cloned)->getResults()))
          mapping.map(oldResult, newResult);
      }

      if (failedToClone) {
        rewriter.eraseOp(newLoop);
        continue;
      }
      rewriter.replaceOp(loop, newLoop.getResults());
      LLVM_DEBUG(llvm::dbgs() << "strip-mined vector loop " << oldWidth
                              << " -> " << newWidth << "\n");
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::cpu::createStripMineVectorLoops() {
  return std::make_unique<StripMineVectorLoops>(128, 4, false);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::cpu::createStripMineVectorLoops(int64_t nativeVectorBits,
                                              int64_t unrollFactor,
                                              bool allowReassociation) {
  return std::make_unique<StripMineVectorLoops>(nativeVectorBits, unrollFactor,
                                                allowReassociation);
}
