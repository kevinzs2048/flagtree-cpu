// Lower int8 tt.DotOp to NEON i8mm (SMMLA) on ARM cores that have NEON i8mm but
// NOT SVE2 (e.g. Apple M-series). Fixed-width analog of ConvertDotToSVE2I8MM:
// emits llvm.aarch64.neon.smmla.v4i32.v16i8 on fixed v16i8/v4i32 vectors (no
// scalable nxv wrapping). Gated on "i8mm" in cpu_features (TRITON_CPU_NEON_I8MM=1),
// inserted before ConvertDotGeneric.
//
// SMMLA semantics (vmmlaq_s32): acc[2x2] += A[2x8] @ B[2x8]^T over K=8.
//   a (v16i8) = row-major 2x8 of A[m:m+2, k:k+8]
//   b (v16i8) = row-major 2x8 of B[k:k+8, n:n+2]^T  (each of the 2 N-cols, 8 deep)
//   acc (v4i32) = [c00, c01, c10, c11]
//
// Scope: M%2==0, N%2==0, K%8==0 int8 GEMM (the common TLE-Lite case). Other shapes
// fall through to ConvertDotGeneric (notifyMatchFailure).
#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTTONEONI8MM
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// 2x8 i8 tile -> v16i8 (row-major flatten).
static Value flat2x8(Location loc, Value tile, Type v16i8Ty,
                     PatternRewriter &rewriter) {
  return rewriter.create<vector::ShapeCastOp>(loc, v16i8Ty, tile);
}
// 2x2 i32 tile -> v4i32.
static Value flat2x2(Location loc, Value tile, Type v4i32Ty,
                     PatternRewriter &rewriter) {
  return rewriter.create<vector::ShapeCastOp>(loc, v4i32Ty, tile);
}
// v4i32 -> 2x2 i32 tile.
static Value to2x2(Location loc, Value vec, Type tile2x2Ty,
                   PatternRewriter &rewriter) {
  return rewriter.create<vector::ShapeCastOp>(loc, tile2x2Ty, vec);
}

static LogicalResult convertNeonI8MM(cpu::DotOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto aTy = dyn_cast<VectorType>(op.getA().getType());
  auto bTy = dyn_cast<VectorType>(op.getB().getType());
  auto cTy = dyn_cast<VectorType>(op.getC().getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
      cTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "rank != 2");
  if (!aTy.getElementType().isInteger(8) || !bTy.getElementType().isInteger(8) ||
      !cTy.getElementType().isInteger(32))
    return rewriter.notifyMatchFailure(op, "not int8->int32");

  int64_t M = aTy.getDimSize(0), K = aTy.getDimSize(1), N = bTy.getDimSize(1);
  if (bTy.getDimSize(0) != K || cTy.getDimSize(0) != M || cTy.getDimSize(1) != N)
    return rewriter.notifyMatchFailure(op, "shape mismatch");
  if (M % 2 != 0 || N % 2 != 0 || K % 8 != 0)
    return rewriter.notifyMatchFailure(op, "need M%2==0, N%2==0, K%8==0");

  Type i8Ty = aTy.getElementType();
  Type i32Ty = cTy.getElementType();
  Type v16i8Ty = VectorType::get({16}, i8Ty);
  Type v4i32Ty = VectorType::get({4}, i32Ty);
  Type tile2x2Ty = VectorType::get({2, 2}, i32Ty);
  StringAttr smmla = StringAttr::get(op.getContext(),
                                     "llvm.aarch64.neon.smmla.v4i32.v16i8");

  Value A = op.getA(), B = op.getB(), res = op.getC();
  int64_t numK = K / 8;

  for (int64_t m = 0; m < M; m += 2) {
    // Pre-pack A[m:m+2, :] for each K step (reused across all N).
    SmallVector<Value> aPk(numK);
    for (int64_t ki = 0, k = 0; k < K; k += 8, ++ki) {
      Value aTile = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, A, ArrayRef<int64_t>{m, k}, ArrayRef<int64_t>{2, 8},
          ArrayRef<int64_t>{1, 1});
      aPk[ki] = flat2x8(loc, aTile, v16i8Ty, rewriter);
    }
    for (int64_t n = 0; n < N; n += 2) {
      // Pre-pack B[:, n:n+2] transposed for each K step.
      SmallVector<Value> bPk(numK);
      for (int64_t ki = 0, k = 0; k < K; k += 8, ++ki) {
        Value bTile = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, B, ArrayRef<int64_t>{k, n}, ArrayRef<int64_t>{8, 2},
            ArrayRef<int64_t>{1, 1});  // 8x2
        Value bT = rewriter.create<vector::TransposeOp>(
            loc, bTile, ArrayRef<int64_t>{1, 0});  // 2x8
        bPk[ki] = flat2x8(loc, bT, v16i8Ty, rewriter);
      }
      // Load 2x2 accumulator, flatten to v4i32.
      Value accTile = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, res, ArrayRef<int64_t>{m, n}, ArrayRef<int64_t>{2, 2},
          ArrayRef<int64_t>{1, 1});
      Value acc = flat2x2(loc, accTile, v4i32Ty, rewriter);
      // Accumulate over K with SMMLA.
      for (int64_t ki = 0; ki < numK; ++ki)
        acc = rewriter
                  .create<LLVM::CallIntrinsicOp>(
                      loc, v4i32Ty, smmla, ValueRange{acc, aPk[ki], bPk[ki]})
                  .getResult(0);
      // Store back.
      Value tile = to2x2(loc, acc, tile2x2Ty, rewriter);
      res = rewriter.create<vector::InsertStridedSliceOp>(
          loc, tile, res, ArrayRef<int64_t>{m, n}, ArrayRef<int64_t>{1, 1});
    }
  }
  rewriter.replaceOp(op, res);
  return success();
}

struct ConvertDotToNeonI8MM
    : public triton::cpu::impl::ConvertDotToNeonI8MMBase<ConvertDotToNeonI8MM> {
  ConvertDotToNeonI8MM() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    SmallVector<cpu::DotOp> dots;
    mod.walk([&dots](cpu::DotOp op) { dots.push_back(op); return WalkResult::advance(); });
    for (auto op : dots) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(op);
      (void) convertNeonI8MM(op, rewriter);  // failure -> left for ConvertDotGeneric
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToNeonI8MM() {
  return std::make_unique<ConvertDotToNeonI8MM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
