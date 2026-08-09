#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <string>
#include <utility>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTTOSVE2I8MM
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// This lowering intentionally follows the shape of a conventional ACLE
// microkernel:
//
//   for n in 0..N step 8:
//     pack B[K, n:n+8]
//     for m in 0..M step 8:
//       acc[4][4] = load C[m:m+8, n:n+8]
//       for k in 0..K step 8:
//         load/pack four 2x8 A row pairs
//         load four packed 2x8 B column pairs
//         acc[4][4] = SVE2 SMMLA(acc, A, B)
//       store acc
//
// Keeping the loops in IR is important.  Expanding a complete Triton tile
// into SSA values makes LLVM keep hundreds of pack values live and produces
// a spill-heavy function even though SMMLA itself is selected.
struct SVE2I8MMCandidate {
  cpu::DotOp op;
  MemBuffer lhsBuf;
  MemBuffer rhsBuf;
  // Ordinary Triton expresses an output-major SDOT panel loaded as [N, 4]
  // and then transposed to the logical dot operand [4, N].  Preserve the
  // physical flat buffer when that exact graph is present: four adjacent
  // output rows are already the sixteen bytes consumed by one SDOT.
  MemBuffer rhsPackedN4Buf;
  bool keepAccInBuffer = false;
};

struct W4A8DotPair {
  cpu::DotOp lowDot;
  cpu::DotOp highDot;
  Value packed;
};

struct PackedQ4PrefillCandidate {
  cpu::DotOp op;
  MemBuffer lhsBuf;
  MemBuffer directLhsBuf;
  Value directPacked;
  MemBuffer directPackedBuf;
  bool kaiLayout = false;
};

struct PackedQ4M4Epilogue {
  PackedQ4PrefillCandidate dot;
  Value lhsScale;
  Value rhsScale;
  arith::AddFOp add;
};

struct PackedQ4M4LoopCandidate {
  scf::ForOp loop;
  SmallVector<PackedQ4M4Epilogue, 4> panels;
};

struct PackedW8PrefillCandidate {
  cpu::DotOp op;
  SmallVector<Value, 4> lhsPanels;
  Value rhsPanel;
};

static bool canClonePackedLoopValue(Value value, scf::ForOp loop);
static Value clonePackedLoopValue(Value value, scf::ForOp oldLoop,
                                  IRMapping &mapping,
                                  PatternRewriter &rewriter);

static Value addIndex(Location loc, Value base, Value offset,
                      PatternRewriter &rewriter) {
  return rewriter.create<arith::AddIOp>(loc, base, offset);
}

static Value addIndex(Location loc, Value base, int64_t offset,
                      PatternRewriter &rewriter) {
  if (offset == 0)
    return base;
  Value c = rewriter.create<arith::ConstantIndexOp>(loc, offset);
  return addIndex(loc, base, c, rewriter);
}

static MemBuffer findFlatVectorLoadBuffer(Value physical) {
  auto shape = physical.getDefiningOp<vector::ShapeCastOp>();
  if (!shape)
    return {};
  auto load = shape.getSource().getDefiningOp<vector::LoadOp>();
  if (!load)
    return {};
  return {load.getBase(), SmallVector<Value>(load.getIndices())};
}

static MemBuffer findPackedN4DotBuffer(Value logical, int64_t kSize,
                                       int64_t nSize) {
  auto transpose = logical.getDefiningOp<vector::TransposeOp>();
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({1, 0}))
    return {};
  auto physicalTy = dyn_cast<VectorType>(transpose.getVector().getType());
  if (!physicalTy ||
      physicalTy.getShape() != ArrayRef<int64_t>({nSize, kSize}) ||
      kSize != 4 || !physicalTy.getElementType().isInteger(8))
    return {};
  MemBuffer buffer = findFlatVectorLoadBuffer(transpose.getVector());
  if (buffer.empty())
    return {};
  auto memRefTy = dyn_cast<MemRefType>(buffer.memRef.getType());
  if (!memRefTy || memRefTy.getRank() != 1 ||
      memRefTy.getDimSize(0) != nSize * kSize ||
      !memRefTy.getElementType().isInteger(8))
    return {};
  return buffer;
}

static SmallVector<Value> getFlatIndices(Location loc, const MemBuffer &buf,
                                         int64_t offset,
                                         PatternRewriter &rewriter) {
  assert(!buf.indices.empty() && "expected a flat input buffer");
  SmallVector<Value> indices(buf.indices);
  indices.back() = shiftIndex(loc, indices.back(), offset, rewriter);
  return indices;
}

static SmallVector<Value> get2DIndices(Location loc, const MemBuffer &buf,
                                       Value row, Value col,
                                       PatternRewriter &rewriter) {
  assert(buf.indices.size() >= 2 && "expected a rank-2 input buffer");
  SmallVector<Value> indices(buf.indices);
  size_t rowIdx = indices.size() - 2;
  size_t colIdx = indices.size() - 1;
  indices[rowIdx] = addIndex(loc, indices[rowIdx], row, rewriter);
  indices[colIdx] = addIndex(loc, indices[colIdx], col, rewriter);
  return indices;
}

static SmallVector<Value> get2DIndices(Location loc, const MemBuffer &buf,
                                       int64_t row, int64_t col,
                                       PatternRewriter &rewriter) {
  assert(buf.indices.size() >= 2 && "expected a rank-2 input buffer");
  SmallVector<Value> indices(buf.indices);
  size_t rowIdx = indices.size() - 2;
  size_t colIdx = indices.size() - 1;
  indices[rowIdx] = shiftIndex(loc, indices[rowIdx], row, rewriter);
  indices[colIdx] = shiftIndex(loc, indices[colIdx], col, rewriter);
  return indices;
}

static SmallVector<Value> get2DIndices(Location loc, const MemBuffer &buf,
                                       Value row, int64_t col,
                                       PatternRewriter &rewriter) {
  assert(buf.indices.size() >= 2 && "expected a rank-2 input buffer");
  SmallVector<Value> indices(buf.indices);
  size_t rowIdx = indices.size() - 2;
  size_t colIdx = indices.size() - 1;
  indices[rowIdx] = addIndex(loc, indices[rowIdx], row, rewriter);
  indices[colIdx] = shiftIndex(loc, indices[colIdx], col, rewriter);
  return indices;
}

static SmallVector<Value> get2DIndices(Location loc, const MemBuffer &buf,
                                       int64_t row, Value col,
                                       PatternRewriter &rewriter) {
  assert(buf.indices.size() >= 2 && "expected a rank-2 input buffer");
  SmallVector<Value> indices(buf.indices);
  size_t rowIdx = indices.size() - 2;
  size_t colIdx = indices.size() - 1;
  indices[rowIdx] = shiftIndex(loc, indices[rowIdx], row, rewriter);
  indices[colIdx] = addIndex(loc, indices[colIdx], col, rewriter);
  return indices;
}

static Value insertFixedInScalable(Location loc, Value fixed, Type scalableTy,
                                   PatternRewriter &rewriter) {
  Value undef = rewriter.create<LLVM::UndefOp>(loc, scalableTy);
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
                                                 rewriter.getI64IntegerAttr(0));
  auto fixedTy = cast<VectorType>(fixed.getType());
  auto scalableVecTy = cast<LLVM::LLVMScalableVectorType>(scalableTy);
  std::string intrinsic =
      "llvm.vector.insert.nxv" +
      std::to_string(scalableVecTy.getMinNumElements()) + "i" +
      std::to_string(fixedTy.getElementType().getIntOrFloatBitWidth()) + ".v" +
      std::to_string(fixedTy.getNumElements()) + "i" +
      std::to_string(fixedTy.getElementType().getIntOrFloatBitWidth());
  return rewriter
      .create<LLVM::CallIntrinsicOp>(
          loc, scalableTy, StringAttr::get(rewriter.getContext(), intrinsic),
          ValueRange{undef, fixed, zero})
      .getResult(0);
}

static Value extractFixedFromScalable(Location loc, Value scalable,
                                      VectorType fixedTy,
                                      PatternRewriter &rewriter) {
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
                                                 rewriter.getI64IntegerAttr(0));
  auto scalableTy = cast<LLVM::LLVMScalableVectorType>(scalable.getType());
  std::string intrinsic =
      "llvm.vector.extract.v" + std::to_string(fixedTy.getNumElements()) + "i" +
      std::to_string(fixedTy.getElementType().getIntOrFloatBitWidth()) +
      ".nxv" + std::to_string(scalableTy.getMinNumElements()) + "i" +
      std::to_string(fixedTy.getElementType().getIntOrFloatBitWidth());
  return rewriter
      .create<LLVM::CallIntrinsicOp>(
          loc, fixedTy, StringAttr::get(rewriter.getContext(), intrinsic),
          ValueRange{scalable, zero})
      .getResult(0);
}

static Value packRows2x8(Location loc, Value row0, Value row1,
                         VectorType v16i8Ty, Type nxv16i8Ty,
                         PatternRewriter &rewriter) {
  Type i8Ty = v16i8Ty.getElementType();
  auto tileTy = VectorType::get({2, 8}, i8Ty);
  Value tile = rewriter.create<arith::ConstantOp>(loc, tileTy,
                                                  rewriter.getZeroAttr(tileTy));
  tile = rewriter.create<vector::InsertOp>(loc, row0, tile, 0LL);
  tile = rewriter.create<vector::InsertOp>(loc, row1, tile, 1LL);
  Value flat = rewriter.create<vector::ShapeCastOp>(loc, v16i8Ty, tile);
  return insertFixedInScalable(loc, flat, nxv16i8Ty, rewriter);
}

static std::pair<Value, Value> transposePair(Location loc, Value lhs, Value rhs,
                                             ArrayRef<int64_t> lowMask,
                                             ArrayRef<int64_t> highMask,
                                             PatternRewriter &rewriter) {
  Value low = rewriter.create<vector::ShuffleOp>(loc, lhs, rhs, lowMask);
  Value high = rewriter.create<vector::ShuffleOp>(loc, lhs, rhs, highMask);
  return {low, high};
}

static Value loadAcc2x2(Location loc, const MemBuffer &accBuf, Value row,
                        Value col, VectorType v2i32Ty, VectorType v4i32Ty,
                        Type nxv4i32Ty, PatternRewriter &rewriter) {
  Value row1 = addIndex(loc, row, 1, rewriter);
  Value r0 = rewriter.create<vector::LoadOp>(
      loc, v2i32Ty, accBuf.memRef,
      get2DIndices(loc, accBuf, row, col, rewriter));
  Value r1 = rewriter.create<vector::LoadOp>(
      loc, v2i32Ty, accBuf.memRef,
      get2DIndices(loc, accBuf, row1, col, rewriter));
  auto tileTy = VectorType::get({2, 2}, v2i32Ty.getElementType());
  Value tile = rewriter.create<arith::ConstantOp>(loc, tileTy,
                                                  rewriter.getZeroAttr(tileTy));
  tile = rewriter.create<vector::InsertOp>(loc, r0, tile, 0LL);
  tile = rewriter.create<vector::InsertOp>(loc, r1, tile, 1LL);
  Value flat = rewriter.create<vector::ShapeCastOp>(loc, v4i32Ty, tile);
  return insertFixedInScalable(loc, flat, nxv4i32Ty, rewriter);
}

static void storeAcc2x2(Location loc, const MemBuffer &accBuf, Value row,
                        Value col, Value scalable, VectorType v2i32Ty,
                        VectorType v4i32Ty, PatternRewriter &rewriter) {
  Value flat = extractFixedFromScalable(loc, scalable, v4i32Ty, rewriter);
  auto tileTy = VectorType::get({2, 2}, v2i32Ty.getElementType());
  Value tile = rewriter.create<vector::ShapeCastOp>(loc, tileTy, flat);
  Value r0 = rewriter.create<vector::ExtractOp>(loc, tile, 0LL);
  Value r1 = rewriter.create<vector::ExtractOp>(loc, tile, 1LL);
  Value row1 = addIndex(loc, row, 1, rewriter);
  rewriter.create<vector::StoreOp>(
      loc, r0, accBuf.memRef, get2DIndices(loc, accBuf, row, col, rewriter));
  rewriter.create<vector::StoreOp>(
      loc, r1, accBuf.memRef, get2DIndices(loc, accBuf, row1, col, rewriter));
}

static bool hasUnitMinorStride(const MemBuffer &buf) {
  auto memRefTy = dyn_cast<MemRefType>(buf.memRef.getType());
  if (!memRefTy || memRefTy.getRank() < 2)
    return false;
  auto [strides, offset] = memRefTy.getStridesAndOffset();
  (void)offset;
  return strides.back() == 1;
}

static bool isCandidate(cpu::DotOp op, SVE2I8MMCandidate &candidate) {
  auto lhsTy = dyn_cast<VectorType>(op.getA().getType());
  auto rhsTy = dyn_cast<VectorType>(op.getB().getType());
  auto accTy = dyn_cast<VectorType>(op.getC().getType());
  if (!lhsTy || !rhsTy || !accTy || lhsTy.getRank() != 2 ||
      rhsTy.getRank() != 2 || accTy.getRank() != 2)
    return false;
  if (!lhsTy.getElementType().isInteger(8) ||
      !rhsTy.getElementType().isInteger(8) ||
      !accTy.getElementType().isInteger(32))
    return false;

  int64_t m = lhsTy.getDimSize(0);
  int64_t k = lhsTy.getDimSize(1);
  int64_t n = rhsTy.getDimSize(1);
  if (rhsTy.getDimSize(0) != k || accTy.getDimSize(0) != m ||
      accTy.getDimSize(1) != n)
    return false;

  // Decode uses a rolled 1x4x4 NEON SDOT microkernel.  Prefill uses a rolled
  // 8x4x8 or 8x8x8 SVE2 SMMLA macro-tile below.  Four columns are native:
  // SMMLA itself produces a 2x2 accumulator, and the narrower panel keeps
  // half as many outer FP32 accumulators live for Q4 groupwise dequant.
  // Tails remain on the generic path until predicated packing is implemented.
  bool isM1Sdot = m == 1 && n >= 4 && k >= 4 && n % 4 == 0 && k % 4 == 0;
  bool isSVE2I8MM =
      m >= 8 && n >= 4 && k >= 8 && m % 8 == 0 && n % 4 == 0 && k % 8 == 0;
  if (!isM1Sdot && !isSVE2I8MM)
    return false;

  candidate.op = op;
  candidate.lhsBuf = findInputBuffer(op.getA());
  candidate.rhsBuf = findInputBuffer(op.getB());
  if (m == 1) {
    candidate.rhsPackedN4Buf = findPackedN4DotBuffer(op.getB(), k, n);
  }
  // Packed low-bit kernels naturally put cheap SSA transforms between a
  // load and dot (shape_cast for A; and/shr/sub for an unpacked Q4 B tile).
  // Do not reject those ordinary Triton programs merely because the input is
  // no longer a direct transfer_read.  convertCandidate() materializes such
  // values into compiler-owned stack buffers, after which the same compact
  // rolled SDOT lowering applies.  Direct load inputs retain the zero-copy
  // path.
  if ((!candidate.lhsBuf.empty() && !hasUnitMinorStride(candidate.lhsBuf)) ||
      (!candidate.rhsBuf.empty() && !hasUnitMinorStride(candidate.rhsBuf)))
    return false;

  candidate.keepAccInBuffer = isLoopCarriedAcc(op.getC());
  return true;
}

// Match the canonical ordinary-Triton Q4 expression:
//
//   lo = dot(x[0:4],  (packed & 15) - 8, acc)
//   hi = dot(x[16:20], (packed >> 4) - 8, lo)
//
// The exact constants are already type-checked by Triton.  The important
// legality condition for this peephole is that both transforms consume the
// same packed byte vector and that the final dot is the loop-carried update.
static bool matchW4A8DotPair(cpu::DotOp lowDot, W4A8DotPair &pair) {
  if (!lowDot->hasOneUse())
    return false;
  auto highDot = dyn_cast<cpu::DotOp>(*lowDot->getUsers().begin());
  if (!highDot || highDot.getC() != lowDot.getResult())
    return false;

  auto lhsTy = dyn_cast<VectorType>(lowDot.getA().getType());
  auto rhsTy = dyn_cast<VectorType>(lowDot.getB().getType());
  auto accTy = dyn_cast<VectorType>(lowDot.getC().getType());
  if (!lhsTy || !rhsTy || !accTy || lhsTy.getRank() != 2 ||
      lhsTy.getDimSize(0) != 1 || lhsTy.getDimSize(1) != 4 ||
      rhsTy.getRank() != 2 || rhsTy.getDimSize(0) != 4 ||
      rhsTy.getDimSize(1) < 4 || rhsTy.getDimSize(1) % 4 != 0 ||
      !lhsTy.getElementType().isInteger(8) ||
      !rhsTy.getElementType().isInteger(8) ||
      !accTy.getElementType().isInteger(32) ||
      highDot.getA().getType() != lowDot.getA().getType() ||
      highDot.getB().getType() != lowDot.getB().getType())
    return false;

  auto lowSub = lowDot.getB().getDefiningOp<arith::SubIOp>();
  auto highSub = highDot.getB().getDefiningOp<arith::SubIOp>();
  if (!lowSub || !highSub)
    return false;
  auto lowAnd = lowSub.getLhs().getDefiningOp<arith::AndIOp>();
  auto highShift = highSub.getLhs().getDefiningOp<arith::ShRUIOp>();
  if (!lowAnd || !highShift || lowAnd.getLhs() != highShift.getLhs())
    return false;

  auto accArg = dyn_cast<BlockArgument>(lowDot.getC());
  if (!accArg || lowDot->getParentOp() != highDot->getParentOp())
    return false;
  auto outerLoop = dyn_cast<scf::ForOp>(lowDot->getParentOp());
  if (!outerLoop || !highDot->hasOneUse())
    return false;
  auto yield = dyn_cast<scf::YieldOp>(*highDot->getUsers().begin());
  int64_t iterArg = accArg.getArgNumber() - outerLoop.getNumInductionVars();
  if (!yield || iterArg < 0 ||
      static_cast<unsigned>(iterArg) >= yield.getNumOperands() ||
      yield.getOperand(iterArg) != highDot.getResult())
    return false;

  pair = {lowDot, highDot, lowAnd.getLhs()};
  return true;
}

// Fuse the matched Q4 pair into one packed-byte transpose and two SDOTs.
// This keeps the source program in ordinary tl.load/bitwise/tl.dot form while
// producing the same dataflow as a tuned ACLE W4A8 microkernel.
static LogicalResult convertW4A8DotPair(W4A8DotPair &pair,
                                        PatternRewriter &rewriter) {
  cpu::DotOp lowDot = pair.lowDot;
  cpu::DotOp highDot = pair.highDot;
  Location loc = lowDot.getLoc();
  auto lhsTy = cast<VectorType>(lowDot.getA().getType());
  auto rhsTy = cast<VectorType>(lowDot.getB().getType());
  auto accTy = cast<VectorType>(lowDot.getC().getType());
  Type i8Ty = lhsTy.getElementType();
  Type i32Ty = accTy.getElementType();
  int64_t nSize = rhsTy.getDimSize(1);

  auto v4i8Ty = VectorType::get({4}, i8Ty);
  auto v8i8Ty = VectorType::get({8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  StringAttr sdot = StringAttr::get(lowDot.getContext(),
                                    "llvm.aarch64.neon.sdot.v4i32.v16i8");

  Operation *allocaPoint = lowDot;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  // Materializing the original packed value (rather than each decoded
  // nibble matrix) gives LLVM a small, alias-free buffer.  SROA forwards the
  // original loads into the microkernel and removes the alloca.
  MemBuffer aLow = storeToTmpBuffer(loc, lowDot.getA(), allocaPoint, rewriter);
  MemBuffer aHigh =
      storeToTmpBuffer(loc, highDot.getA(), allocaPoint, rewriter);
  MemBuffer packed = storeToTmpBuffer(loc, pair.packed, allocaPoint, rewriter);

  auto outerLoop = cast<scf::ForOp>(lowDot->getParentOp());
  Value initialAcc = getInitAccValue(lowDot.getC());
  MemBuffer accBuf;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(outerLoop);
    accBuf = allocateTmpBufferStack(loc, accTy, allocaPoint, rewriter);
    op_write(initialAcc, accBuf.memRef, accBuf.indices);
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
  Value nLimit = rewriter.create<arith::ConstantIndexOp>(loc, nSize);
  Value zero8 = rewriter.create<arith::ConstantOp>(
      loc, v8i8Ty, rewriter.getZeroAttr(v8i8Ty));
  Value zero16 = rewriter.create<arith::ConstantOp>(
      loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
  Value maskScalar = rewriter.create<arith::ConstantIntOp>(loc, 15, 8);
  Value zeroPointScalar = rewriter.create<arith::ConstantIntOp>(loc, 8, 8);
  Value shiftScalar = rewriter.create<arith::ConstantIntOp>(loc, 4, 8);
  Value mask = rewriter.create<vector::BroadcastOp>(loc, v16i8Ty, maskScalar);
  Value zeroPoint =
      rewriter.create<vector::BroadcastOp>(loc, v16i8Ty, zeroPointScalar);
  Value shift = rewriter.create<vector::BroadcastOp>(loc, v16i8Ty, shiftScalar);

  auto broadcastA = [&](const MemBuffer &buf) {
    Value a4 = rewriter.create<vector::LoadOp>(
        loc, v4i8Ty, buf.memRef, get2DIndices(loc, buf, 0, 0, rewriter));
    // Bitcast directly to a scalar word before broadcasting.  Using
    // vector.bitcast through vector<1xi32> made the vector-to-LLVM lowering
    // reconstruct the word byte-by-byte (USHLL + MOV + UZP1).  The scalar
    // LLVM bitcast folds with the four-byte load and selects a single DUP.
    Value scalar = rewriter.create<LLVM::BitcastOp>(loc, i32Ty, a4);
    Value words = rewriter.create<vector::BroadcastOp>(loc, v4i32Ty, scalar);
    return rewriter.create<LLVM::BitcastOp>(loc, v16i8Ty, words).getResult();
  };
  Value aLowBroadcast = broadcastA(aLow);
  Value aHighBroadcast = broadcastA(aHigh);

  constexpr int64_t transposePacked[] = {0, 4, 16, 20, 1, 5, 17, 21,
                                         2, 6, 18, 22, 3, 7, 19, 23};

  auto nLoop = rewriter.create<scf::ForOp>(loc, c0, nLimit, c4);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nBase = nLoop.getInductionVar();
    Value acc = rewriter.create<vector::LoadOp>(
        loc, v4i32Ty, accBuf.memRef,
        get2DIndices(loc, accBuf, 0, nBase, rewriter));

    SmallVector<Value> rows(4);
    for (int64_t r = 0; r < 4; ++r) {
      rows[r] = rewriter.create<vector::LoadOp>(
          loc, v4i8Ty, packed.memRef,
          get2DIndices(loc, packed, r, nBase, rewriter));
    }
    Value rows01 = rewriter.create<vector::InsertStridedSliceOp>(
        loc, rows[0], zero8, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
    rows01 = rewriter.create<vector::InsertStridedSliceOp>(
        loc, rows[1], rows01, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{1});
    Value rows23 = rewriter.create<vector::InsertStridedSliceOp>(
        loc, rows[2], zero8, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
    rows23 = rewriter.create<vector::InsertStridedSliceOp>(
        loc, rows[3], rows23, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{1});
    Value rows01Wide = rewriter.create<vector::InsertStridedSliceOp>(
        loc, rows01, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
    Value rows23Wide = rewriter.create<vector::InsertStridedSliceOp>(
        loc, rows23, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
    Value transposed = rewriter.create<vector::ShuffleOp>(
        loc, rows01Wide, rows23Wide, transposePacked);
    Value weightLow = rewriter.create<arith::AndIOp>(loc, transposed, mask);
    weightLow = rewriter.create<arith::SubIOp>(loc, weightLow, zeroPoint);
    Value weightHigh = rewriter.create<arith::ShRUIOp>(loc, transposed, shift);
    weightHigh = rewriter.create<arith::SubIOp>(loc, weightHigh, zeroPoint);

    Value lowAcc =
        rewriter
            .create<LLVM::CallIntrinsicOp>(
                loc, v4i32Ty, sdot, ValueRange{acc, aLowBroadcast, weightLow})
            .getResult(0);
    Value highAcc = rewriter
                        .create<LLVM::CallIntrinsicOp>(
                            loc, v4i32Ty, sdot,
                            ValueRange{lowAcc, aHighBroadcast, weightHigh})
                        .getResult(0);
    rewriter.create<vector::StoreOp>(
        loc, highAcc, accBuf.memRef,
        get2DIndices(loc, accBuf, 0, nBase, rewriter));
  }

  rewriter.setInsertionPointAfter(nLoop);
  Value loopResult =
      outerLoop.getTiedLoopResult(cast<BlockArgument>(lowDot.getC()));
  rewriter.replaceOp(highDot, lowDot.getC());
  rewriter.eraseOp(lowDot);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(outerLoop);
  Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
  rewriter.replaceAllUsesWith(loopResult, result);
  return success();
}

// Recognize the ordinary Triton expression used for a groupwise Q4 x Q8
// prefill tile:
//
//   packed: vector<16xNxi8>
//   lo = packed << 4
//   hi = packed & 0xf0
//   weight = reshape(transpose(reshape(interleave(lo, hi))))
//   dot(lhs: vector<Mx32xi8>, weight: vector<32x8xi8>)
//
// Two physical layouts are accepted after the semantic graph is proven:
// [4,N,4] expands each eight-byte column pair with an interleave, while KAI's
// [2,N,8] layout keeps K8 segments intact so a 16-byte shift/mask pair feeds
// SMMLA without a nibble-expansion ZIP1.  The generic i8mm lowering can
// consume either expression,
// but doing so first
// materializes the expanded 32x8 matrix and then transposes it into SMMLA
// panels.  Keeping the original packed bytes lets the lowering expand one
// 2-column x 8-K panel immediately before each SMMLA instead.
static bool matchPackedQ4Prefill(cpu::DotOp op,
                                 PackedQ4PrefillCandidate &candidate) {
  auto lhsTy = dyn_cast<VectorType>(op.getA().getType());
  auto rhsTy = dyn_cast<VectorType>(op.getB().getType());
  auto accTy = dyn_cast<VectorType>(op.getC().getType());
  if (!lhsTy || !rhsTy || !accTy || lhsTy.getRank() != 2 ||
      rhsTy.getRank() != 2 || accTy.getRank() != 2 || lhsTy.getDimSize(0) < 4 ||
      lhsTy.getDimSize(0) % 4 != 0 || lhsTy.getDimSize(1) != 32 ||
      rhsTy.getDimSize(0) != 32 ||
      (rhsTy.getDimSize(1) != 4 && rhsTy.getDimSize(1) != 8) ||
      accTy.getDimSize(0) != lhsTy.getDimSize(0) ||
      accTy.getDimSize(1) != rhsTy.getDimSize(1) ||
      !lhsTy.getElementType().isInteger(8) ||
      !rhsTy.getElementType().isInteger(8) ||
      !accTy.getElementType().isInteger(32) ||
      !matchPattern(op.getC(), m_Zero()))
    return false;

  auto finalShape = op.getB().getDefiningOp<vector::ShapeCastOp>();
  if (!finalShape)
    return false;
  auto transpose = finalShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return false;
  auto innerShape = transpose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!innerShape)
    return false;
  auto interleave =
      innerShape.getSource().getDefiningOp<vector::InterleaveOp>();
  if (!interleave)
    return false;
  auto lowShift = interleave.getLhs().getDefiningOp<arith::ShLIOp>();
  auto highMask = interleave.getRhs().getDefiningOp<arith::AndIOp>();
  if (!lowShift || !highMask || lowShift.getLhs() != highMask.getLhs())
    return false;

  Value packed = lowShift.getLhs();
  auto packedTy = dyn_cast<VectorType>(packed.getType());
  int64_t nSize = rhsTy.getDimSize(1);
  if ((nSize == 4 && lhsTy.getDimSize(0) != 4 &&
       lhsTy.getDimSize(0) % 16 != 0) ||
      (nSize == 8 && lhsTy.getDimSize(0) % 8 != 0))
    return false;
  if (!packedTy || packedTy.getShape() != ArrayRef<int64_t>({16, nSize}) ||
      !packedTy.getElementType().isInteger(8))
    return false;

  // Require either compiler-oriented [K8-step, N, K4] or native KAI
  // [K8-segment, N, packed-K8].  This reshape/transpose is semantic Triton IR,
  // not a private runtime marker.  Other row-major packings remain on the
  // proven generic path.
  auto packedShape = packed.getDefiningOp<vector::ShapeCastOp>();
  if (!packedShape)
    return false;
  auto packedTranspose =
      packedShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!packedTranspose ||
      packedTranspose.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return false;
  Value directPacked = packedTranspose.getVector();
  auto directTy = dyn_cast<VectorType>(directPacked.getType());
  if (!directTy || !directTy.getElementType().isInteger(8))
    return false;
  bool kaiLayout = false;
  if (directTy.getShape() == ArrayRef<int64_t>({4, nSize, 4})) {
    kaiLayout = false;
  } else if (nSize == 4 &&
             directTy.getShape() == ArrayRef<int64_t>({2, nSize, 8})) {
    kaiLayout = true;
  } else {
    return false;
  }

  MemBuffer directLhsBuf;
  auto lhsLogicalShape = op.getA().getDefiningOp<vector::ShapeCastOp>();
  if (lhsLogicalShape && !kaiLayout) {
    auto lhsTranspose =
        lhsLogicalShape.getSource().getDefiningOp<vector::TransposeOp>();
    if (lhsTranspose &&
        lhsTranspose.getPermutation() == ArrayRef<int64_t>({1, 2, 0, 3})) {
      Value directLhsPacked = lhsTranspose.getVector();
      auto directLhsTy = dyn_cast<VectorType>(directLhsPacked.getType());
      if (!directLhsTy ||
          directLhsTy.getShape() != ArrayRef<int64_t>({4, 2, 2, 8}) ||
          !directLhsTy.getElementType().isInteger(8))
        directLhsPacked = {};
      if (directLhsPacked)
        directLhsBuf = findFlatVectorLoadBuffer(directLhsPacked);
    }
  } else if (lhsLogicalShape && kaiLayout) {
    // KAI keeps K[0:16] and K[16:32] in separate i8mm panels.  The ordinary
    // Triton expression first reconstructs rows in sequential-K order and
    // then interleaves the two halves to agree with tl.join(low, high).
    auto interleaveTranspose =
        lhsLogicalShape.getSource().getDefiningOp<vector::TransposeOp>();
    if (interleaveTranspose &&
        interleaveTranspose.getPermutation() == ArrayRef<int64_t>({0, 2, 1})) {
      auto halvesShape =
          interleaveTranspose.getVector().getDefiningOp<vector::ShapeCastOp>();
      vector::TransposeOp physicalTranspose;
      if (halvesShape) {
        physicalTranspose =
            halvesShape.getSource().getDefiningOp<vector::TransposeOp>();
        if (!physicalTranspose) {
          auto sequentialShape =
              halvesShape.getSource().getDefiningOp<vector::ShapeCastOp>();
          if (sequentialShape)
            physicalTranspose = sequentialShape.getSource()
                                    .getDefiningOp<vector::TransposeOp>();
        }
      }
      if (physicalTranspose &&
          physicalTranspose.getPermutation() == ArrayRef<int64_t>({1, 0, 2})) {
        Value directLhsPacked = physicalTranspose.getVector();
        auto directLhsTy = dyn_cast<VectorType>(directLhsPacked.getType());
        if (directLhsTy &&
            directLhsTy.getShape() == ArrayRef<int64_t>({4, 4, 8}) &&
            directLhsTy.getElementType().isInteger(8))
          directLhsBuf = findFlatVectorLoadBuffer(directLhsPacked);
      }
    }
  }
  MemBuffer lhsBuf;
  if (directLhsBuf.empty()) {
    if (kaiLayout)
      return false;
    lhsBuf = findInputBuffer(op.getA());
    if (lhsBuf.empty() || !hasUnitMinorStride(lhsBuf))
      return false;
  }
  MemBuffer directPackedBuf = findFlatVectorLoadBuffer(directPacked);
  if (directPackedBuf.empty() || directPackedBuf.indices.empty())
    return false;
  candidate = {op,           lhsBuf,          directLhsBuf,
               directPacked, directPackedBuf, kaiLayout};
  return true;
}

static Value getOtherMulOperand(arith::MulFOp op, Value known) {
  if (op.getLhs() == known)
    return op.getRhs();
  if (op.getRhs() == known)
    return op.getLhs();
  return {};
}

static bool matchQ4ScaleBroadcast(Value value, bool rowScale, Value &scale) {
  auto broadcast = value.getDefiningOp<vector::BroadcastOp>();
  if (!broadcast)
    return false;
  auto shape = broadcast.getSource().getDefiningOp<vector::ShapeCastOp>();
  if (!shape)
    return false;
  auto shapedTy = dyn_cast<VectorType>(shape.getType());
  auto scaleTy = dyn_cast<VectorType>(shape.getSource().getType());
  SmallVector<int64_t, 2> expected =
      rowScale ? SmallVector<int64_t, 2>{4, 1} : SmallVector<int64_t, 2>{1, 4};
  if (!shapedTy || shapedTy.getShape() != ArrayRef<int64_t>(expected) ||
      !scaleTy || scaleTy.getShape() != ArrayRef<int64_t>({4}) ||
      !scaleTy.getElementType().isF32())
    return false;
  scale = shape.getSource();
  return true;
}

static bool matchPackedQ4M4Epilogue(PackedQ4PrefillCandidate dot,
                                    scf::ForOp loop,
                                    PackedQ4M4Epilogue &epilogue,
                                    int64_t &iterIndex) {
  auto lhsTy = cast<VectorType>(dot.op.getA().getType());
  auto accTy = cast<VectorType>(dot.op.getC().getType());
  if (lhsTy.getShape() != ArrayRef<int64_t>({4, 32}) ||
      accTy.getShape() != ArrayRef<int64_t>({4, 4}) || !dot.op->hasOneUse())
    return false;
  auto conversion = dyn_cast<arith::SIToFPOp>(*dot.op->getUsers().begin());
  if (!conversion || !conversion->hasOneUse())
    return false;
  auto fixedScale = dyn_cast<arith::MulFOp>(*conversion->getUsers().begin());
  if (!fixedScale || !fixedScale->hasOneUse())
    return false;
  Value fixedScaleValue = getOtherMulOperand(fixedScale, conversion.getOut());
  auto constant = fixedScaleValue.getDefiningOp<arith::ConstantOp>();
  auto scaleAttr =
      constant ? dyn_cast<DenseFPElementsAttr>(constant.getValue()) : nullptr;
  if (!scaleAttr || !scaleAttr.isSplat() ||
      scaleAttr.getSplatValue<APFloat>().convertToDouble() != 0.0625)
    return false;

  auto lhsMul = dyn_cast<arith::MulFOp>(*fixedScale->getUsers().begin());
  if (!lhsMul || !lhsMul->hasOneUse())
    return false;
  Value lhsFactor = getOtherMulOperand(lhsMul, fixedScale.getResult());
  Value lhsScale;
  if (!lhsFactor || !matchQ4ScaleBroadcast(lhsFactor, true, lhsScale))
    return false;

  auto rhsMul = dyn_cast<arith::MulFOp>(*lhsMul->getUsers().begin());
  if (!rhsMul || !rhsMul->hasOneUse())
    return false;
  Value rhsFactor = getOtherMulOperand(rhsMul, lhsMul.getResult());
  Value rhsScale;
  if (!rhsFactor || !matchQ4ScaleBroadcast(rhsFactor, false, rhsScale))
    return false;

  auto add = dyn_cast<arith::AddFOp>(*rhsMul->getUsers().begin());
  if (!add || !add->hasOneUse())
    return false;
  Value carried;
  if (add.getLhs() == rhsMul.getResult())
    carried = add.getRhs();
  else if (add.getRhs() == rhsMul.getResult())
    carried = add.getLhs();
  else
    return false;
  auto blockArg = dyn_cast<BlockArgument>(carried);
  if (!blockArg || blockArg.getOwner() != loop.getBody())
    return false;
  iterIndex = blockArg.getArgNumber() - loop.getNumInductionVars();
  auto yield = dyn_cast<scf::YieldOp>(*add->getUsers().begin());
  if (!yield || iterIndex < 0 ||
      static_cast<unsigned>(iterIndex) >= yield.getNumOperands() ||
      yield.getOperand(iterIndex) != add.getResult())
    return false;
  epilogue = {dot, lhsScale, rhsScale, add};
  return true;
}

static bool matchPackedQ4M4Loop(scf::ForOp loop,
                                PackedQ4M4LoopCandidate &candidate) {
  unsigned panelCount = loop.getNumRegionIterArgs();
  if (panelCount < 1 || panelCount > 4 || loop.getNumResults() != panelCount)
    return false;
  SmallVector<std::pair<int64_t, PackedQ4M4Epilogue>, 4> matches;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    auto dotOp = dyn_cast<cpu::DotOp>(operation);
    if (!dotOp)
      continue;
    PackedQ4PrefillCandidate dot;
    PackedQ4M4Epilogue epilogue;
    int64_t iterIndex = -1;
    if (!matchPackedQ4Prefill(dotOp, dot))
      return false;
    if (!matchPackedQ4M4Epilogue(dot, loop, epilogue, iterIndex))
      return false;
    matches.emplace_back(iterIndex, epilogue);
  }
  if (matches.size() != panelCount)
    return false;
  llvm::sort(matches, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (int64_t index = 0; index < panelCount; ++index) {
    if (matches[index].first != index)
      return false;
    candidate.panels.push_back(matches[index].second);
  }
  candidate.loop = loop;
  return true;
}

// Lower the packed Q4 pattern to a rolled fixed-width SMMLA kernel.  The
// [4,N,4] layout interleaves shifted/masked eight-byte halves.  KAI's
// [2,N,8] layout instead loads two columns as one 16-byte panel and applies
// the low-nibble shift to K[0:16] and high-nibble mask to K[16:32], avoiding
// ZIP1 in nibble expansion.  Neither path materializes an expanded-weight
// temporary.
static SmallVector<Value, 4>
emitPackedQ4M4Dot(Location loc, const MemBuffer &lhsBuf,
                  const MemBuffer &directLhsBuf,
                  const MemBuffer &directPackedBuf, bool kaiLayout,
                  bool oneRowPairAtATime, PatternRewriter &rewriter) {
  Type i8Ty = rewriter.getI8Type();
  Type i32Ty = rewriter.getI32Type();
  auto v8i8Ty = VectorType::get({8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  auto tile2x8Ty = VectorType::get({2, 8}, i8Ty);
  StringAttr smmla = StringAttr::get(rewriter.getContext(),
                                     "llvm.aarch64.neon.smmla.v4i32.v16i8");
  Value shift4 = rewriter.create<arith::ConstantOp>(
      loc, v8i8Ty,
      DenseElementsAttr::get(v8i8Ty, rewriter.getI8IntegerAttr(4)));
  Value highMask = rewriter.create<arith::ConstantOp>(
      loc, v8i8Ty,
      DenseElementsAttr::get(v8i8Ty, rewriter.getI8IntegerAttr(-16)));
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, v4i32Ty, rewriter.getZeroAttr(v4i32Ty));
  SmallVector<Value, 4> acc(4, zero);

  if (kaiLayout) {
    Value shift4x16 = rewriter.create<arith::ConstantOp>(
        loc, v16i8Ty,
        DenseElementsAttr::get(v16i8Ty, rewriter.getI8IntegerAttr(4)));
    Value highMaskx16 = rewriter.create<arith::ConstantOp>(
        loc, v16i8Ty,
        DenseElementsAttr::get(v16i8Ty, rewriter.getI8IntegerAttr(-16)));
    for (int64_t segment = 0; segment < 2; ++segment) {
      SmallVector<Value, 2> lhsLowPairs;
      SmallVector<Value, 2> lhsHighPairs;
      if (!oneRowPairAtATime) {
        for (int64_t rowPair = 0; rowPair < 2; ++rowPair) {
          lhsLowPairs.push_back(rewriter.create<vector::LoadOp>(
              loc, v16i8Ty, directLhsBuf.memRef,
              getFlatIndices(loc, directLhsBuf, segment * 32 + rowPair * 16,
                             rewriter)));
          lhsHighPairs.push_back(rewriter.create<vector::LoadOp>(
              loc, v16i8Ty, directLhsBuf.memRef,
              getFlatIndices(loc, directLhsBuf,
                             (segment + 2) * 32 + rowPair * 16, rewriter)));
        }
      }
      SmallVector<Value, 2> rhsLow;
      SmallVector<Value, 2> rhsHigh;
      for (int64_t colPair = 0; colPair < 2; ++colPair) {
        Value packed = rewriter.create<vector::LoadOp>(
            loc, v16i8Ty, directPackedBuf.memRef,
            getFlatIndices(loc, directPackedBuf, segment * 32 + colPair * 16,
                           rewriter));
        rhsLow.push_back(
            rewriter.create<arith::ShLIOp>(loc, packed, shift4x16));
        rhsHigh.push_back(
            rewriter.create<arith::AndIOp>(loc, packed, highMaskx16));
      }
      for (int64_t rowPair = 0; rowPair < 2; ++rowPair) {
        // Keep only one row pair live at a time.  M16 already carries sixteen
        // FP32 result vectors through the group loop; materializing both row
        // pairs before the first SMMLA leaves LLVM no scratch Q register and
        // triggers an accumulator rotation plus a hot-loop spill/reload.
        Value lhsLow =
            oneRowPairAtATime
                ? rewriter.create<vector::LoadOp>(
                      loc, v16i8Ty, directLhsBuf.memRef,
                      getFlatIndices(loc, directLhsBuf,
                                     segment * 32 + rowPair * 16, rewriter))
                : lhsLowPairs[rowPair];
        Value lhsHigh =
            oneRowPairAtATime
                ? rewriter.create<vector::LoadOp>(
                      loc, v16i8Ty, directLhsBuf.memRef,
                      getFlatIndices(loc, directLhsBuf,
                                     (segment + 2) * 32 + rowPair * 16,
                                     rewriter))
                : lhsHighPairs[rowPair];
        for (int64_t colPair = 0; colPair < 2; ++colPair) {
          int64_t index = rowPair * 2 + colPair;
          acc[index] = rewriter
                           .create<LLVM::CallIntrinsicOp>(
                               loc, v4i32Ty, smmla,
                               ValueRange{acc[index], lhsLow, rhsLow[colPair]})
                           .getResult(0);
          acc[index] =
              rewriter
                  .create<LLVM::CallIntrinsicOp>(
                      loc, v4i32Ty, smmla,
                      ValueRange{acc[index], lhsHigh, rhsHigh[colPair]})
                  .getResult(0);
        }
      }
    }
    return acc;
  }

  for (int64_t ki = 0; ki < 4; ++ki) {
    SmallVector<Value, 2> lhsPacked;
    for (int64_t rowPair = 0; rowPair < 2; ++rowPair) {
      if (!directLhsBuf.empty()) {
        lhsPacked.push_back(rewriter.create<vector::LoadOp>(
            loc, v16i8Ty, directLhsBuf.memRef,
            getFlatIndices(loc, directLhsBuf, ki * 32 + rowPair * 16,
                           rewriter)));
        continue;
      }
      Value a0 = rewriter.create<vector::LoadOp>(
          loc, v8i8Ty, lhsBuf.memRef,
          get2DIndices(loc, lhsBuf, rowPair * 2, ki * 8, rewriter));
      Value a1 = rewriter.create<vector::LoadOp>(
          loc, v8i8Ty, lhsBuf.memRef,
          get2DIndices(loc, lhsBuf, rowPair * 2 + 1, ki * 8, rewriter));
      Value tile = rewriter.create<arith::ConstantOp>(
          loc, tile2x8Ty, rewriter.getZeroAttr(tile2x8Ty));
      tile = rewriter.create<vector::InsertOp>(loc, a0, tile, 0LL);
      tile = rewriter.create<vector::InsertOp>(loc, a1, tile, 1LL);
      lhsPacked.push_back(
          rewriter.create<vector::ShapeCastOp>(loc, v16i8Ty, tile));
    }
    SmallVector<Value, 2> rhsPacked;
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value packed = rewriter.create<vector::LoadOp>(
          loc, v8i8Ty, directPackedBuf.memRef,
          getFlatIndices(loc, directPackedBuf, ki * 16 + colPair * 8,
                         rewriter));
      Value low = rewriter.create<arith::ShLIOp>(loc, packed, shift4);
      Value high = rewriter.create<arith::AndIOp>(loc, packed, highMask);
      rhsPacked.push_back(
          rewriter.create<vector::InterleaveOp>(loc, low, high));
    }
    for (int64_t rowPair = 0; rowPair < 2; ++rowPair) {
      for (int64_t colPair = 0; colPair < 2; ++colPair) {
        int64_t index = rowPair * 2 + colPair;
        acc[index] = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, v4i32Ty, smmla,
                             ValueRange{acc[index], lhsPacked[rowPair],
                                        rhsPacked[colPair]})
                         .getResult(0);
      }
    }
  }
  return acc;
}

static SmallVector<Value, 4> formPackedQ4M4Rows(Location loc,
                                                ValueRange tileAccumulators,
                                                PatternRewriter &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  auto tile2x2Ty = VectorType::get({2, 2}, i32Ty);
  SmallVector<Value, 4> rows;
  for (int64_t rowPair = 0; rowPair < 2; ++rowPair) {
    Value left = rewriter.create<vector::ShapeCastOp>(
        loc, tile2x2Ty, tileAccumulators[rowPair * 2]);
    Value right = rewriter.create<vector::ShapeCastOp>(
        loc, tile2x2Ty, tileAccumulators[rowPair * 2 + 1]);
    for (int64_t row = 0; row < 2; ++row) {
      Value leftRow = rewriter.create<vector::ExtractOp>(loc, left, row);
      Value rightRow = rewriter.create<vector::ExtractOp>(loc, right, row);
      Value combined = rewriter.create<arith::ConstantOp>(
          loc, v4i32Ty, rewriter.getZeroAttr(v4i32Ty));
      combined = rewriter.create<vector::InsertStridedSliceOp>(
          loc, leftRow, combined, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
      combined = rewriter.create<vector::InsertStridedSliceOp>(
          loc, rightRow, combined, ArrayRef<int64_t>{2}, ArrayRef<int64_t>{1});
      rows.push_back(combined);
    }
  }
  return rows;
}

// Fuse the four M4 dots and their groupwise dequantization into one backend
// loop.  KAI keeps sixteen FP32 output rows live, evaluates one M4 panel at a
// time with four temporary SMMLA accumulators, and immediately folds that
// panel into the FP32 state.  Preserving the same lifetimes avoids LLVM
// scheduling all sixteen integer accumulators concurrently and spilling both
// integer and floating-point state.
static LogicalResult convertPackedQ4M4Loop(PackedQ4M4LoopCandidate &candidate,
                                           PatternRewriter &rewriter) {
  scf::ForOp oldLoop = candidate.loop;
  for (const PackedQ4M4Epilogue &panel : candidate.panels) {
    if (!canClonePackedLoopValue(panel.dot.directPackedBuf.memRef, oldLoop) ||
        !canClonePackedLoopValue(panel.lhsScale, oldLoop) ||
        !canClonePackedLoopValue(panel.rhsScale, oldLoop))
      return failure();
    for (Value index : panel.dot.directPackedBuf.indices)
      if (!canClonePackedLoopValue(index, oldLoop))
        return failure();
    if (!panel.dot.directLhsBuf.empty()) {
      if (!canClonePackedLoopValue(panel.dot.directLhsBuf.memRef, oldLoop))
        return failure();
      for (Value index : panel.dot.directLhsBuf.indices)
        if (!canClonePackedLoopValue(index, oldLoop))
          return failure();
    } else {
      if (!canClonePackedLoopValue(panel.dot.lhsBuf.memRef, oldLoop))
        return failure();
      for (Value index : panel.dot.lhsBuf.indices)
        if (!canClonePackedLoopValue(index, oldLoop))
          return failure();
    }
  }

  Location loc = oldLoop.getLoc();
  Type f32Ty = rewriter.getF32Type();
  auto v4f32Ty = VectorType::get({4}, f32Ty);
  rewriter.setInsertionPoint(oldLoop);
  auto newLoop = rewriter.create<scf::ForOp>(
      loc, oldLoop.getLowerBound(), oldLoop.getUpperBound(), oldLoop.getStep(),
      oldLoop.getInitArgs());

  IRMapping mapping;
  mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
  rewriter.setInsertionPointToEnd(newLoop.getBody());
  SmallVector<Value, 4> outputPanels(newLoop.getRegionIterArgs().begin(),
                                     newLoop.getRegionIterArgs().end());
  Value fractionalBits = rewriter.create<arith::ConstantIntOp>(loc, 4, 32);
  StringAttr scvtf =
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.neon.vcvtfxs2fp");

  int64_t panelCount = candidate.panels.size();
  for (int64_t panelOrder = 0; panelOrder < panelCount; ++panelOrder) {
    // LLVM's AArch64 allocator gives the three-panel tail fewer live-range
    // collisions in 1,0,2 order. It uses fewer instructions and stack
    // references than all five other permutations on the tested Arm target.
    // Four panels retain the established reverse order;
    // their hot-loop pressure is controlled separately by consuming one LHS
    // row pair at a time. One and two panels are naturally reversed.
    static constexpr int64_t m12PanelOrder[] = {1, 0, 2};
    int64_t panelIndex = panelCount == 3 ? m12PanelOrder[panelOrder]
                                         : panelCount - 1 - panelOrder;
    const PackedQ4M4Epilogue &panel = candidate.panels[panelIndex];
    MemBuffer lhsBuf = panel.dot.lhsBuf;
    MemBuffer directLhsBuf = panel.dot.directLhsBuf;
    if (!directLhsBuf.empty()) {
      directLhsBuf.memRef =
          clonePackedLoopValue(directLhsBuf.memRef, oldLoop, mapping, rewriter);
      for (Value &index : directLhsBuf.indices)
        index = clonePackedLoopValue(index, oldLoop, mapping, rewriter);
    } else {
      lhsBuf.memRef =
          clonePackedLoopValue(lhsBuf.memRef, oldLoop, mapping, rewriter);
      for (Value &index : lhsBuf.indices)
        index = clonePackedLoopValue(index, oldLoop, mapping, rewriter);
    }
    MemBuffer directPackedBuf = panel.dot.directPackedBuf;
    directPackedBuf.memRef = clonePackedLoopValue(directPackedBuf.memRef,
                                                  oldLoop, mapping, rewriter);
    for (Value &index : directPackedBuf.indices)
      index = clonePackedLoopValue(index, oldLoop, mapping, rewriter);
    Value lhsScale =
        clonePackedLoopValue(panel.lhsScale, oldLoop, mapping, rewriter);
    Value rhsScale =
        clonePackedLoopValue(panel.rhsScale, oldLoop, mapping, rewriter);

    SmallVector<Value, 4> tileAcc =
        emitPackedQ4M4Dot(loc, lhsBuf, directLhsBuf, directPackedBuf,
                          panel.dot.kaiLayout, panelCount == 4, rewriter);
    SmallVector<Value, 4> dotRows = formPackedQ4M4Rows(loc, tileAcc, rewriter);
    for (int64_t row = 0; row < 4; ++row) {
      Value converted =
          rewriter
              .create<LLVM::CallIntrinsicOp>(
                  loc, v4f32Ty, scvtf, ValueRange{dotRows[row], fractionalBits})
              .getResult(0);
      Value lhsScalar = rewriter.create<vector::ExtractOp>(loc, lhsScale, row);
      Value lhsBroadcast =
          rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, lhsScalar);
      Value scale = rewriter.create<arith::MulFOp>(loc, lhsBroadcast, rhsScale);
      Value contribution =
          rewriter.create<arith::MulFOp>(loc, converted, scale);
      Value oldRow = rewriter.create<vector::ExtractOp>(
          loc, outputPanels[panelIndex], row);
      Value newRow = rewriter.create<arith::AddFOp>(loc, oldRow, contribution);
      outputPanels[panelIndex] = rewriter.create<vector::InsertOp>(
          loc, newRow, outputPanels[panelIndex], row);
    }
  }
  rewriter.create<scf::YieldOp>(loc, outputPanels);

  rewriter.setInsertionPointAfter(newLoop);
  for (int64_t panel = 0; panel < panelCount; ++panel)
    rewriter.replaceAllUsesWith(oldLoop.getResult(panel),
                                newLoop.getResult(panel));
  rewriter.eraseOp(oldLoop);
  return success();
}

static LogicalResult
convertPackedQ4PrefillM4(PackedQ4PrefillCandidate &candidate,
                         PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  auto accTy = cast<VectorType>(op.getC().getType());
  Type i32Ty = accTy.getElementType();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  auto tile2x2Ty = VectorType::get({2, 2}, i32Ty);
  SmallVector<Value, 4> acc = emitPackedQ4M4Dot(
      loc, candidate.lhsBuf, candidate.directLhsBuf, candidate.directPackedBuf,
      candidate.kaiLayout, false, rewriter);

  Value result = rewriter.create<arith::ConstantOp>(
      loc, accTy, rewriter.getZeroAttr(accTy));
  for (int64_t rowPair = 0; rowPair < 2; ++rowPair) {
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value tile = rewriter.create<vector::ShapeCastOp>(
          loc, tile2x2Ty, acc[rowPair * 2 + colPair]);
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, tile, result, ArrayRef<int64_t>{rowPair * 2, colPair * 2},
          ArrayRef<int64_t>{1, 1});
    }
  }
  rewriter.replaceOp(op, result);
  return success();
}

static LogicalResult convertPackedQ4Prefill(PackedQ4PrefillCandidate &candidate,
                                            PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  auto lhsTy = cast<VectorType>(op.getA().getType());
  auto accTy = cast<VectorType>(op.getC().getType());
  Type i8Ty = lhsTy.getElementType();
  Type i32Ty = accTy.getElementType();
  int64_t mSize = lhsTy.getDimSize(0);
  int64_t nSize = accTy.getDimSize(1);
  if (mSize == 4 && nSize == 4)
    return convertPackedQ4PrefillM4(candidate, rewriter);
  int64_t nPairs = nSize / 2;
  int64_t mBlock = nSize == 4 ? std::min<int64_t>(16, mSize) : 8;
  int64_t mPairs = mBlock / 2;
  if (mSize % mBlock != 0)
    return failure();

  auto v8i8Ty = VectorType::get({8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v2i32Ty = VectorType::get({2}, i32Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  Type nxv16i8Ty = LLVM::LLVMScalableVectorType::get(i8Ty, 16);
  Type nxv4i32Ty = LLVM::LLVMScalableVectorType::get(i32Ty, 4);
  StringAttr smmla =
      StringAttr::get(op.getContext(), "llvm.aarch64.sve.smmla.nxv4i32");

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();
  MemBuffer accBuf = allocateTmpBufferStack(loc, accTy, allocaPoint, rewriter);

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value mStep = rewriter.create<arith::ConstantIndexOp>(loc, mBlock);
  Value mLimit = rewriter.create<arith::ConstantIndexOp>(loc, mSize);
  Value shift4 = rewriter.create<arith::ConstantOp>(
      loc, v8i8Ty,
      DenseElementsAttr::get(v8i8Ty, rewriter.getI8IntegerAttr(4)));
  Value highMask = rewriter.create<arith::ConstantOp>(
      loc, v8i8Ty,
      DenseElementsAttr::get(v8i8Ty, rewriter.getI8IntegerAttr(-16)));
  MemBuffer directBuf;
  if (nSize == 8) {
    auto flatTy = VectorType::get({4 * nSize * 4}, i8Ty);
    Value flat = rewriter.create<vector::ShapeCastOp>(loc, flatTy,
                                                      candidate.directPacked);
    {
      OpBuilder::InsertionGuard allocaGuard(rewriter);
      rewriter.setInsertionPoint(allocaPoint);
      auto memRefTy = MemRefType::get(flatTy.getShape(), i8Ty);
      Value memRef = rewriter.create<memref::AllocaOp>(
          loc, memRefTy, rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
      Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      directBuf = {memRef, SmallVector<Value>{zeroIndex}};
    }
    op_write(flat, directBuf.memRef, directBuf.indices);
  }
  auto mLoop = rewriter.create<scf::ForOp>(loc, c0, mLimit, mStep);
  {
    OpBuilder::InsertionGuard mGuard(rewriter);
    rewriter.setInsertionPointToStart(mLoop.getBody());
    Value mBase = mLoop.getInductionVar();
    Value zero = rewriter.create<LLVM::ZeroOp>(loc, nxv4i32Ty);
    SmallVector<Value> acc(mPairs * nPairs, zero);

    auto emitKStep = [&](Value lhsCol,
                         llvm::function_ref<Value(int64_t)> loadPackedPanel) {
      SmallVector<Value> aPacked(mPairs);
      for (int64_t mr = 0; mr < mPairs; ++mr) {
        Value row0 = addIndex(loc, mBase, mr * 2, rewriter);
        Value row1 = addIndex(loc, row0, 1, rewriter);
        Value a0 = rewriter.create<vector::LoadOp>(
            loc, v8i8Ty, candidate.lhsBuf.memRef,
            get2DIndices(loc, candidate.lhsBuf, row0, lhsCol, rewriter));
        Value a1 = rewriter.create<vector::LoadOp>(
            loc, v8i8Ty, candidate.lhsBuf.memRef,
            get2DIndices(loc, candidate.lhsBuf, row1, lhsCol, rewriter));
        aPacked[mr] = packRows2x8(loc, a0, a1, v16i8Ty, nxv16i8Ty, rewriter);
      }

      SmallVector<Value> bPacked(nPairs);
      for (int64_t nr = 0; nr < nPairs; ++nr) {
        Value gathered = loadPackedPanel(nr);
        Value low = rewriter.create<arith::ShLIOp>(loc, gathered, shift4);
        Value high = rewriter.create<arith::AndIOp>(loc, gathered, highMask);
        Value expanded = rewriter.create<vector::InterleaveOp>(loc, low, high);
        bPacked[nr] = insertFixedInScalable(loc, expanded, nxv16i8Ty, rewriter);
      }

      for (int64_t mr = 0; mr < mPairs; ++mr) {
        for (int64_t nr = 0; nr < nPairs; ++nr) {
          int64_t index = mr * nPairs + nr;
          acc[index] = rewriter
                           .create<LLVM::CallIntrinsicOp>(
                               loc, nxv4i32Ty, smmla,
                               ValueRange{acc[index], aPacked[mr], bPacked[nr]})
                           .getResult(0);
        }
      }
    };

    if (nSize == 8) {
      // With two M8 iterations, LLVM otherwise hoists all sixteen expanded
      // RHS panels and spills them around the M loop.  A tiny rolled K4 loop
      // keeps only four panels live.  The 128-byte physical Q4 tile is still
      // half the size of the logical int8 matrix and needs no transpose.
      Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
      Value c8 = rewriter.create<arith::ConstantIndexOp>(loc, 8);
      Value c32 = rewriter.create<arith::ConstantIndexOp>(loc, 32);
      auto kLoop = rewriter.create<scf::ForOp>(loc, c0, c4, c1, acc);
      {
        OpBuilder::InsertionGuard kGuard(rewriter);
        rewriter.setInsertionPointToStart(kLoop.getBody());
        Value ki = kLoop.getInductionVar();
        Value lhsCol = rewriter.create<arith::MulIOp>(loc, ki, c8);
        Value panelBase = rewriter.create<arith::MulIOp>(loc, ki, c32);
        acc.assign(kLoop.getRegionIterArgs().begin(),
                   kLoop.getRegionIterArgs().end());
        emitKStep(lhsCol, [&](int64_t nr) {
          Value offset = addIndex(loc, panelBase, nr * 8, rewriter);
          SmallVector<Value> indices(directBuf.indices);
          indices.back() = addIndex(loc, indices.back(), offset, rewriter);
          return rewriter.create<vector::LoadOp>(loc, v8i8Ty, directBuf.memRef,
                                                 indices);
        });
        rewriter.setInsertionPointToEnd(kLoop.getBody());
        rewriter.create<scf::YieldOp>(loc, acc);
      }
      rewriter.setInsertionPointAfter(kLoop);
      acc.assign(kLoop.getResults().begin(), kLoop.getResults().end());
    } else {
      // N4 x M16 matches KAI's register balance.  The single M iteration
      // allows constant slices to stay in registers without LICM spills.
      for (int64_t ki = 0; ki < 4; ++ki) {
        Value lhsCol = rewriter.create<arith::ConstantIndexOp>(loc, ki * 8);
        emitKStep(lhsCol, [&](int64_t nr) {
          Value panel = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, candidate.directPacked, ArrayRef<int64_t>{ki, nr * 2, 0},
              ArrayRef<int64_t>{1, 2, 4}, ArrayRef<int64_t>{1, 1, 1});
          return rewriter.create<vector::ShapeCastOp>(loc, v8i8Ty, panel);
        });
      }
    }

    for (int64_t mr = 0; mr < mPairs; ++mr) {
      Value row = addIndex(loc, mBase, mr * 2, rewriter);
      for (int64_t nr = 0; nr < nPairs; ++nr) {
        Value col = rewriter.create<arith::ConstantIndexOp>(loc, nr * 2);
        storeAcc2x2(loc, accBuf, row, col, acc[mr * nPairs + nr], v2i32Ty,
                    v4i32Ty, rewriter);
      }
    }
  }

  rewriter.setInsertionPointAfter(mLoop);
  Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
  rewriter.replaceOp(op, result);
  return success();
}

// Match the exact ordinary Triton 3.7 join/permute graph used to describe
// KleidiAI's prefill packing.  Each LHS leaf is one [K8, M4, 8-byte] physical
// panel; the RHS is one [K8, N4, 8-byte] panel.  The logical source remains a
// portable tl.dot([M, 32], [32, 4]) for M4, M8, or M16.
static bool matchKaiW8LhsLeaf(Value value, Value &physical) {
  auto logicalShape = value.getDefiningOp<vector::ShapeCastOp>();
  if (!logicalShape)
    return false;
  auto transpose =
      logicalShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({1, 0, 2}))
    return false;
  auto physicalShape =
      transpose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!physicalShape)
    return false;
  physical = physicalShape.getSource();
  auto physicalTy = dyn_cast<VectorType>(physical.getType());
  return physicalTy && physicalTy.getShape() == ArrayRef<int64_t>({128}) &&
         physicalTy.getElementType().isInteger(8);
}

static bool matchKaiW8LhsPair(Value value, Value &low, Value &high) {
  auto finalShape = value.getDefiningOp<vector::ShapeCastOp>();
  if (!finalShape)
    return false;
  auto transpose = finalShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({2, 0, 1}))
    return false;
  auto joinShape = transpose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!joinShape)
    return false;
  auto join = joinShape.getSource().getDefiningOp<vector::InterleaveOp>();
  if (!join)
    return false;
  low = join.getLhs();
  high = join.getRhs();
  return true;
}

static bool matchPackedW8Prefill(cpu::DotOp op,
                                 PackedW8PrefillCandidate &candidate) {
  auto lhsTy = dyn_cast<VectorType>(op.getA().getType());
  auto rhsTy = dyn_cast<VectorType>(op.getB().getType());
  auto accTy = dyn_cast<VectorType>(op.getC().getType());
  if (!lhsTy || !rhsTy || !accTy || lhsTy.getRank() != 2 ||
      (lhsTy.getDimSize(0) != 4 && lhsTy.getDimSize(0) != 8 &&
       lhsTy.getDimSize(0) != 16) ||
      lhsTy.getDimSize(1) != 32 ||
      rhsTy.getShape() != ArrayRef<int64_t>({32, 4}) || accTy.getRank() != 2 ||
      accTy.getDimSize(0) != lhsTy.getDimSize(0) || accTy.getDimSize(1) != 4 ||
      !lhsTy.getElementType().isInteger(8) ||
      !rhsTy.getElementType().isInteger(8) ||
      !accTy.getElementType().isInteger(32))
    return false;

  SmallVector<Value, 4> logicalPanels;
  int64_t mSize = lhsTy.getDimSize(0);
  if (mSize == 4) {
    logicalPanels.push_back(op.getA());
  } else if (mSize == 8) {
    Value lhs0;
    Value lhs1;
    if (!matchKaiW8LhsPair(op.getA(), lhs0, lhs1))
      return false;
    logicalPanels.push_back(lhs0);
    logicalPanels.push_back(lhs1);
  } else {
    Value lhs01;
    Value lhs23;
    if (!matchKaiW8LhsPair(op.getA(), lhs01, lhs23))
      return false;
    Value lhs0;
    Value lhs1;
    Value lhs2;
    Value lhs3;
    if (!matchKaiW8LhsPair(lhs01, lhs0, lhs1) ||
        !matchKaiW8LhsPair(lhs23, lhs2, lhs3))
      return false;
    logicalPanels.push_back(lhs0);
    logicalPanels.push_back(lhs1);
    logicalPanels.push_back(lhs2);
    logicalPanels.push_back(lhs3);
  }
  SmallVector<Value, 4> panels;
  panels.reserve(logicalPanels.size());
  for (Value logical : logicalPanels) {
    Value physical;
    if (!matchKaiW8LhsLeaf(logical, physical))
      return false;
    panels.push_back(physical);
  }

  auto rhsShape = op.getB().getDefiningOp<vector::ShapeCastOp>();
  if (!rhsShape)
    return false;
  auto rhsTranspose = rhsShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!rhsTranspose ||
      rhsTranspose.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return false;
  auto rhsPhysicalShape =
      rhsTranspose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!rhsPhysicalShape)
    return false;
  Value rhsPanel = rhsPhysicalShape.getSource();
  auto rhsPanelTy = dyn_cast<VectorType>(rhsPanel.getType());
  if (!rhsPanelTy || rhsPanelTy.getShape() != ArrayRef<int64_t>({128}) ||
      !rhsPanelTy.getElementType().isInteger(8))
    return false;

  candidate = {op, panels, rhsPanel};
  return true;
}

static SmallVector<Value, 16>
emitPackedW8PrefillSmmla(Location loc, ArrayRef<Value> lhsPanels,
                         Value rhsPanel, ValueRange initialAccumulators,
                         PatternRewriter &rewriter) {
  Type i8Ty = rewriter.getI8Type();
  Type i32Ty = rewriter.getI32Type();
  auto panelTy = VectorType::get({4, 4, 8}, i8Ty);
  auto packed2x8Ty = VectorType::get({2, 8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  Type nxv16i8Ty = LLVM::LLVMScalableVectorType::get(i8Ty, 16);
  Type nxv4i32Ty = LLVM::LLVMScalableVectorType::get(i32Ty, 4);
  StringAttr smmla =
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.sve.smmla.nxv4i32");

  SmallVector<Value, 4> lhsPhysical;
  int64_t panelCount = lhsPanels.size();
  lhsPhysical.resize(panelCount);
  for (int64_t panel = 0; panel < panelCount; ++panel)
    lhsPhysical[panel] =
        rewriter.create<vector::ShapeCastOp>(loc, panelTy, lhsPanels[panel]);
  Value rhsPhysical =
      rewriter.create<vector::ShapeCastOp>(loc, panelTy, rhsPanel);
  SmallVector<Value, 16> accumulators(initialAccumulators.begin(),
                                      initialAccumulators.end());

  int64_t rowPairs = panelCount * 2;
  assert(initialAccumulators.size() == static_cast<size_t>(rowPairs * 2) &&
         "packed W8 accumulator count does not match LHS panels");
  for (int64_t kStep = 0; kStep < 4; ++kStep) {
    SmallVector<Value, 8> lhsPacked;
    for (int64_t rowPair = 0; rowPair < rowPairs; ++rowPair) {
      int64_t panel = rowPair / 2;
      int64_t firstRow = (rowPair % 2) * 2;
      Value rows = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, lhsPhysical[panel], ArrayRef<int64_t>{kStep, firstRow, 0},
          ArrayRef<int64_t>{1, 2, 8}, ArrayRef<int64_t>{1, 1, 1});
      Value packed =
          rewriter.create<vector::ShapeCastOp>(loc, packed2x8Ty, rows);
      Value flat = rewriter.create<vector::ShapeCastOp>(loc, v16i8Ty, packed);
      lhsPacked.push_back(
          insertFixedInScalable(loc, flat, nxv16i8Ty, rewriter));
    }

    SmallVector<Value, 2> rhsPacked;
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value columns = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, rhsPhysical, ArrayRef<int64_t>{kStep, colPair * 2, 0},
          ArrayRef<int64_t>{1, 2, 8}, ArrayRef<int64_t>{1, 1, 1});
      Value packed =
          rewriter.create<vector::ShapeCastOp>(loc, packed2x8Ty, columns);
      Value flat = rewriter.create<vector::ShapeCastOp>(loc, v16i8Ty, packed);
      rhsPacked.push_back(
          insertFixedInScalable(loc, flat, nxv16i8Ty, rewriter));
    }

    for (int64_t rowPair = 0; rowPair < rowPairs; ++rowPair) {
      for (int64_t colPair = 0; colPair < 2; ++colPair) {
        int64_t index = rowPair * 2 + colPair;
        accumulators[index] =
            rewriter
                .create<LLVM::CallIntrinsicOp>(loc, nxv4i32Ty, smmla,
                                               ValueRange{accumulators[index],
                                                          lhsPacked[rowPair],
                                                          rhsPacked[colPair]})
                .getResult(0);
      }
    }
  }
  return accumulators;
}

static SmallVector<Value, 16>
splitPackedW8Accumulator(Location loc, Value accumulator,
                         PatternRewriter &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  auto accTy = cast<VectorType>(accumulator.getType());
  int64_t mSize = accTy.getDimSize(0);
  assert(accTy.getShape().back() == 4 && mSize % 2 == 0 &&
         "expected packed W8 Mx4 accumulator");
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  Type nxv4i32Ty = LLVM::LLVMScalableVectorType::get(i32Ty, 4);
  SmallVector<Value, 16> tiles;
  tiles.reserve(mSize);
  for (int64_t rowPair = 0; rowPair < mSize / 2; ++rowPair) {
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value tile = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, accumulator, ArrayRef<int64_t>{rowPair * 2, colPair * 2},
          ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
      Value flat = rewriter.create<vector::ShapeCastOp>(loc, v4i32Ty, tile);
      tiles.push_back(insertFixedInScalable(loc, flat, nxv4i32Ty, rewriter));
    }
  }
  return tiles;
}

static Value joinPackedW8Accumulator(Location loc, VectorType accTy,
                                     ValueRange accumulators,
                                     PatternRewriter &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  int64_t mSize = accTy.getDimSize(0);
  assert(accTy.getDimSize(1) == 4 && mSize % 2 == 0 &&
         accumulators.size() == static_cast<size_t>(mSize) &&
         "expected one packed W8 accumulator per output row");
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  auto tile2x2Ty = VectorType::get({2, 2}, i32Ty);
  Value result = rewriter.create<arith::ConstantOp>(
      loc, accTy, rewriter.getZeroAttr(accTy));
  for (int64_t rowPair = 0; rowPair < mSize / 2; ++rowPair) {
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value flat = extractFixedFromScalable(
          loc, accumulators[rowPair * 2 + colPair], v4i32Ty, rewriter);
      Value tile = rewriter.create<vector::ShapeCastOp>(loc, tile2x2Ty, flat);
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, tile, result, ArrayRef<int64_t>{rowPair * 2, colPair * 2},
          ArrayRef<int64_t>{1, 1});
    }
  }
  return result;
}

static bool canClonePackedLoopValue(Value value, scf::ForOp loop) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() != loop.getBody())
      return true;
    return arg == loop.getInductionVar();
  }
  Operation *def = value.getDefiningOp();
  if (!def || !loop->isProperAncestor(def))
    return true;
  if (def->getBlock() != loop.getBody() || def->getNumRegions() != 0)
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return canClonePackedLoopValue(operand, loop);
  });
}

static Value clonePackedLoopValue(Value value, scf::ForOp oldLoop,
                                  IRMapping &mapping,
                                  PatternRewriter &rewriter) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    assert(arg.getOwner() != oldLoop.getBody() &&
           "unmapped W8 prefill loop block argument");
    return value;
  }
  Operation *def = value.getDefiningOp();
  if (!def || !oldLoop->isProperAncestor(def))
    return value;
  for (Value operand : def->getOperands())
    (void)clonePackedLoopValue(operand, oldLoop, mapping, rewriter);
  rewriter.clone(*def, mapping);
  return mapping.lookup(value);
}

// Logical Mx4 loop arguments otherwise get split and rebuilt on every K32
// iteration. Carry every native SMMLA accumulator through the loop and join
// only once after the reduction is complete. Supporting all region iter args
// together is important for an M12 tail expressed as one M8 and one M4 dot.
static LogicalResult
convertPackedW8PrefillLoop(PackedW8PrefillCandidate &candidate,
                           PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  auto oldLoop = dyn_cast<scf::ForOp>(op->getParentOp());
  if (!oldLoop || oldLoop.getNumRegionIterArgs() == 0 ||
      oldLoop.getNumResults() != oldLoop.getNumRegionIterArgs() ||
      !op->hasOneUse())
    return failure();
  auto yield = dyn_cast<scf::YieldOp>(*op->getUsers().begin());
  if (!yield || yield->getParentOp() != oldLoop ||
      yield.getNumOperands() != oldLoop.getNumRegionIterArgs())
    return failure();

  SmallVector<PackedW8PrefillCandidate, 4> candidates;
  candidates.reserve(oldLoop.getNumRegionIterArgs());
  for (auto [index, accArg] : llvm::enumerate(oldLoop.getRegionIterArgs())) {
    auto dot = yield.getOperand(index).getDefiningOp<cpu::DotOp>();
    if (!dot || dot.getC() != accArg || dot->getParentOp() != oldLoop ||
        !dot->hasOneUse())
      return failure();
    PackedW8PrefillCandidate sibling;
    if (!matchPackedW8Prefill(dot, sibling))
      return failure();
    for (Value panel : sibling.lhsPanels)
      if (!canClonePackedLoopValue(panel, oldLoop))
        return failure();
    if (!canClonePackedLoopValue(sibling.rhsPanel, oldLoop))
      return failure();
    candidates.push_back(std::move(sibling));
  }

  Location loc = op.getLoc();
  rewriter.setInsertionPoint(oldLoop);
  SmallVector<Value, 32> initial;
  SmallVector<size_t, 4> offsets;
  for (Value init : oldLoop.getInitArgs()) {
    offsets.push_back(initial.size());
    SmallVector<Value, 16> tiles =
        splitPackedW8Accumulator(loc, init, rewriter);
    initial.append(tiles.begin(), tiles.end());
  }
  auto newLoop = rewriter.create<scf::ForOp>(loc, oldLoop.getLowerBound(),
                                             oldLoop.getUpperBound(),
                                             oldLoop.getStep(), initial);

  IRMapping mapping;
  mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
  rewriter.setInsertionPointToEnd(newLoop.getBody());
  SmallVector<Value, 32> next;
  for (auto [index, packed] : llvm::enumerate(candidates)) {
    SmallVector<Value, 4> lhsPanels;
    for (Value panel : packed.lhsPanels)
      lhsPanels.push_back(
          clonePackedLoopValue(panel, oldLoop, mapping, rewriter));
    Value rhsPanel =
        clonePackedLoopValue(packed.rhsPanel, oldLoop, mapping, rewriter);
    size_t count = packed.lhsPanels.size() * 4;
    ValueRange current =
        newLoop.getRegionIterArgs().slice(offsets[index], count);
    SmallVector<Value, 16> accumulators =
        emitPackedW8PrefillSmmla(loc, lhsPanels, rhsPanel, current, rewriter);
    next.append(accumulators.begin(), accumulators.end());
  }
  rewriter.create<scf::YieldOp>(loc, next);

  rewriter.setInsertionPointAfter(newLoop);
  for (auto [index, packed] : llvm::enumerate(candidates)) {
    auto accTy = cast<VectorType>(packed.op.getC().getType());
    size_t count = packed.lhsPanels.size() * 4;
    ValueRange nativeResults =
        newLoop.getResults().slice(offsets[index], count);
    Value result = joinPackedW8Accumulator(loc, accTy, nativeResults, rewriter);
    rewriter.replaceAllUsesWith(oldLoop.getResult(index), result);
  }
  rewriter.eraseOp(oldLoop);
  return success();
}

// Feed KAI's already-packed panels directly to SMMLA.  The original generic
// path first rebuilt two logical matrices and packed them again.  This rewrite
// retains the Triton loop-carried vector accumulator but removes that duplicate
// data movement; LLVM can keep the sixteen 2x2 accumulators in registers.
static LogicalResult convertPackedW8Prefill(PackedW8PrefillCandidate &candidate,
                                            PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  if (succeeded(convertPackedW8PrefillLoop(candidate, rewriter)))
    return success();
  Location loc = op.getLoc();
  auto accTy = cast<VectorType>(op.getC().getType());
  int64_t mSize = accTy.getDimSize(0);
  Type i32Ty = rewriter.getI32Type();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  Type nxv4i32Ty = LLVM::LLVMScalableVectorType::get(i32Ty, 4);

  SmallVector<Value, 16> accumulators;
  accumulators.reserve(mSize);
  for (int64_t rowPair = 0; rowPair < mSize / 2; ++rowPair) {
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value tile = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, op.getC(), ArrayRef<int64_t>{rowPair * 2, colPair * 2},
          ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
      Value flat = rewriter.create<vector::ShapeCastOp>(loc, v4i32Ty, tile);
      accumulators.push_back(
          insertFixedInScalable(loc, flat, nxv4i32Ty, rewriter));
    }
  }
  accumulators = emitPackedW8PrefillSmmla(
      loc, candidate.lhsPanels, candidate.rhsPanel, accumulators, rewriter);
  Value result = joinPackedW8Accumulator(loc, accTy, accumulators, rewriter);
  rewriter.replaceOp(op, result);
  return success();
}

// Fold the ordinary signed-i8 multiply/reduce form emitted by Triton 3.7
// into one architectural SDOT.  This is deliberately an algebraic compiler
// pattern rather than a frontend intrinsic: kernels remain portable
// tl.load/cast/multiply/tl.sum programs, while AArch64 gets the instruction
// that implements four independent four-byte reductions.
static bool isNeonSdotReduction(vector::MultiDimReductionOp op) {
  if (op.getKind() != vector::CombiningKind::ADD ||
      op.getReductionDims() != ArrayRef<int64_t>({1}))
    return false;
  auto sourceTy = dyn_cast<VectorType>(op.getSource().getType());
  auto accTy = dyn_cast<VectorType>(op.getAcc().getType());
  if (!sourceTy || sourceTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
      !sourceTy.getElementType().isInteger(32) || !accTy ||
      accTy.getShape() != ArrayRef<int64_t>({4}) ||
      !accTy.getElementType().isInteger(32))
    return false;
  auto multiply = op.getSource().getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;
  auto lhsExt = multiply.getLhs().getDefiningOp<arith::ExtSIOp>();
  auto rhsExt = multiply.getRhs().getDefiningOp<arith::ExtSIOp>();
  if (!lhsExt || !rhsExt)
    return false;
  auto lhsTy = dyn_cast<VectorType>(lhsExt.getIn().getType());
  auto rhsTy = dyn_cast<VectorType>(rhsExt.getIn().getType());
  return lhsTy && rhsTy && lhsTy.getShape() == ArrayRef<int64_t>({4, 4}) &&
         rhsTy.getShape() == ArrayRef<int64_t>({4, 4}) &&
         lhsTy.getElementType().isInteger(8) &&
         rhsTy.getElementType().isInteger(8);
}

static Value findRepeated8ByteInput(Value value) {
  auto outerShape = value.getDefiningOp<vector::ShapeCastOp>();
  if (!outerShape)
    return {};
  auto transpose = outerShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({1, 0}))
    return {};
  auto innerShape = transpose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!innerShape)
    return {};
  auto interleave =
      innerShape.getSource().getDefiningOp<vector::InterleaveOp>();
  if (!interleave || interleave.getLhs() != interleave.getRhs())
    return {};
  Value input = interleave.getLhs();
  auto inputTy = dyn_cast<VectorType>(input.getType());
  if (!inputTy || inputTy.getShape() != ArrayRef<int64_t>({8}) ||
      !inputTy.getElementType().isInteger(8))
    return {};
  return input;
}

static void convertNeonSdotReduction(vector::MultiDimReductionOp op,
                                     PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto multiply = cast<arith::MulIOp>(op.getSource().getDefiningOp());
  auto lhsExt = cast<arith::ExtSIOp>(multiply.getLhs().getDefiningOp());
  auto rhsExt = cast<arith::ExtSIOp>(multiply.getRhs().getDefiningOp());
  auto v16i8Ty = VectorType::get({16}, rewriter.getI8Type());
  auto v4i32Ty = VectorType::get({4}, rewriter.getI32Type());
  auto makeOperand = [&](Value input) -> Value {
    if (Value repeated = findRepeated8ByteInput(input)) {
      Value scalar = rewriter.create<LLVM::BitcastOp>(
          loc, rewriter.getI64Type(), repeated);
      auto v2i64Ty = VectorType::get({2}, rewriter.getI64Type());
      Value broadcast =
          rewriter.create<vector::BroadcastOp>(loc, v2i64Ty, scalar);
      return rewriter.create<LLVM::BitcastOp>(loc, v16i8Ty, broadcast);
    }
    return rewriter.create<vector::ShapeCastOp>(loc, v16i8Ty, input);
  };
  Value lhs = makeOperand(lhsExt.getIn());
  Value rhs = makeOperand(rhsExt.getIn());
  StringAttr sdot =
      StringAttr::get(op.getContext(), "llvm.aarch64.neon.sdot.v4i32.v16i8");
  Value result = rewriter
                     .create<LLVM::CallIntrinsicOp>(
                         loc, v4i32Ty, sdot, ValueRange{op.getAcc(), lhs, rhs})
                     .getResult(0);
  rewriter.replaceOp(op, result);
}

static bool matchI32ToF32Scale16(arith::MulFOp op,
                                 arith::SIToFPOp &conversion) {
  arith::ConstantOp constant;
  if ((conversion = op.getLhs().getDefiningOp<arith::SIToFPOp>()) &&
      (constant = op.getRhs().getDefiningOp<arith::ConstantOp>())) {
  } else if ((conversion = op.getRhs().getDefiningOp<arith::SIToFPOp>()) &&
             (constant = op.getLhs().getDefiningOp<arith::ConstantOp>())) {
  } else {
    return false;
  }
  auto resultTy = dyn_cast<VectorType>(op.getType());
  auto inputTy = dyn_cast<VectorType>(conversion.getIn().getType());
  auto scale = dyn_cast<DenseFPElementsAttr>(constant.getValue());
  return resultTy && inputTy && resultTy.getShape() == ArrayRef<int64_t>({4}) &&
         resultTy.getElementType().isF32() &&
         inputTy.getShape() == ArrayRef<int64_t>({4}) &&
         inputTy.getElementType().isInteger(32) && scale && scale.isSplat() &&
         scale.getSplatValue<APFloat>().convertToDouble() == 0.0625;
}

static void convertI32ToF32Scale16(arith::MulFOp op, arith::SIToFPOp conversion,
                                   PatternRewriter &rewriter) {
  Value fractionalBits =
      rewriter.create<arith::ConstantIntOp>(op.getLoc(), 4, 32);
  StringAttr scvtf =
      StringAttr::get(op.getContext(), "llvm.aarch64.neon.vcvtfxs2fp");
  Value result = rewriter
                     .create<LLVM::CallIntrinsicOp>(
                         op.getLoc(), op.getType(), scvtf,
                         ValueRange{conversion.getIn(), fractionalBits})
                     .getResult(0);
  rewriter.replaceOp(op, result);
}

// Triton 3.7 expresses concatenation through interleave/reshape/transpose.
// Recognize the exact two-vector pair reduction used by the KAI-layout W8
// epilogue and select one ADDP instead of four scalar vector reductions.
static bool matchNeonAddpReduction(vector::MultiDimReductionOp op, Value &lhs,
                                   Value &rhs) {
  if (op.getKind() != vector::CombiningKind::ADD ||
      op.getReductionDims() != ArrayRef<int64_t>({1}) ||
      !matchPattern(op.getAcc(), m_Zero()))
    return false;
  auto sourceTy = dyn_cast<VectorType>(op.getSource().getType());
  if (!sourceTy || sourceTy.getShape() != ArrayRef<int64_t>({4, 2}) ||
      !sourceTy.getElementType().isInteger(32))
    return false;
  auto finalShape = op.getSource().getDefiningOp<vector::ShapeCastOp>();
  if (!finalShape)
    return false;
  auto transpose = finalShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({1, 0}))
    return false;
  auto innerShape = transpose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!innerShape)
    return false;
  auto interleave =
      innerShape.getSource().getDefiningOp<vector::InterleaveOp>();
  if (!interleave)
    return false;
  auto lhsTy = dyn_cast<VectorType>(interleave.getLhs().getType());
  auto rhsTy = dyn_cast<VectorType>(interleave.getRhs().getType());
  if (!lhsTy || !rhsTy || lhsTy.getShape() != ArrayRef<int64_t>({4}) ||
      rhsTy != lhsTy || !lhsTy.getElementType().isInteger(32))
    return false;
  lhs = interleave.getLhs();
  rhs = interleave.getRhs();
  return true;
}

static void convertNeonAddpReduction(vector::MultiDimReductionOp op, Value lhs,
                                     Value rhs, PatternRewriter &rewriter) {
  auto v4i32Ty = VectorType::get({4}, rewriter.getI32Type());
  StringAttr addp =
      StringAttr::get(op.getContext(), "llvm.aarch64.neon.addp.v4i32");
  Value result = rewriter
                     .create<LLVM::CallIntrinsicOp>(op.getLoc(), v4i32Ty, addp,
                                                    ValueRange{lhs, rhs})
                     .getResult(0);
  rewriter.replaceOp(op, result);
}

// Lower an ordinary Triton M=1 int8 tl.dot to a rolled SDOT GEMV.
//
// This deliberately consumes the buffers discovered behind cpu::DotOp
// instead of expanding the complete [1,K] x [K,N] tile into SSA values.
// Consequently the generated LLVM IR retains compact N/K loops even for
// model-sized dimensions.  Four row-major B rows are transposed in registers
// into SDOT's four groups of four K bytes:
//
//   [b(k+0,n+0), b(k+1,n+0), b(k+2,n+0), b(k+3,n+0), ... n+3]
//
// A prepacked variant can later replace those four loads and the shuffle with
// one 16-byte load without changing the accumulator structure.
static LogicalResult convertCandidateM1(SVE2I8MMCandidate &candidate,
                                        PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  auto lhsTy = cast<VectorType>(op.getA().getType());
  auto rhsTy = cast<VectorType>(op.getB().getType());
  auto accTy = cast<VectorType>(op.getC().getType());
  Type i8Ty = lhsTy.getElementType();
  Type i32Ty = accTy.getElementType();
  int64_t kSize = lhsTy.getDimSize(1);
  int64_t nSize = rhsTy.getDimSize(1);

  auto v4i8Ty = VectorType::get({4}, i8Ty);
  auto v8i8Ty = VectorType::get({8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  StringAttr sdot =
      StringAttr::get(op.getContext(), "llvm.aarch64.neon.sdot.v4i32.v16i8");

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  scf::ForOp outerKLoop;
  Value initialAcc = op.getC();
  if (candidate.keepAccInBuffer) {
    outerKLoop = cast<scf::ForOp>(op->getParentOp());
    initialAcc = getInitAccValue(op.getC());
  }
  bool zeroInitialAcc =
      !candidate.keepAccInBuffer && matchPattern(initialAcc, m_Zero());

  MemBuffer accBuf;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    if (candidate.keepAccInBuffer)
      rewriter.setInsertionPoint(outerKLoop);
    accBuf = allocateTmpBufferStack(loc, accTy, allocaPoint, rewriter);
    if (!zeroInitialAcc)
      op_write(initialAcc, accBuf.memRef, accBuf.indices);
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
  Value nLimit = rewriter.create<arith::ConstantIndexOp>(loc, nSize);
  Value kLimit = rewriter.create<arith::ConstantIndexOp>(loc, kSize);
  Value zero8 = rewriter.create<arith::ConstantOp>(
      loc, v8i8Ty, rewriter.getZeroAttr(v8i8Ty));
  Value zero16 = rewriter.create<arith::ConstantOp>(
      loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
  Value zeroAcc = rewriter.create<arith::ConstantOp>(
      loc, v4i32Ty, rewriter.getZeroAttr(v4i32Ty));

  constexpr int64_t transposeB[] = {0, 4, 16, 20, 1, 5, 17, 21,
                                    2, 6, 18, 22, 3, 7, 19, 23};

  auto nLoop = rewriter.create<scf::ForOp>(loc, c0, nLimit, c4);
  {
    OpBuilder::InsertionGuard nGuard(rewriter);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nBase = nLoop.getInductionVar();
    Value initAcc = zeroAcc;
    if (!zeroInitialAcc) {
      initAcc = rewriter.create<vector::LoadOp>(
          loc, v4i32Ty, accBuf.memRef,
          get2DIndices(loc, accBuf, 0, nBase, rewriter));
    }

    auto kLoop =
        rewriter.create<scf::ForOp>(loc, c0, kLimit, c4, ValueRange{initAcc});
    {
      OpBuilder::InsertionGuard kGuard(rewriter);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kBase = kLoop.getInductionVar();

      Value a4 = rewriter.create<vector::LoadOp>(
          loc, v4i8Ty, candidate.lhsBuf.memRef,
          get2DIndices(loc, candidate.lhsBuf, 0, kBase, rewriter));
      // SDOT needs the same four activation bytes in every 32-bit lane.
      // Express that as a word broadcast rather than an i8 shuffle padded
      // through two zero vectors.  LLVM then selects LD1R/DUP instead of an
      // extend/insert/shuffle sequence.
      Value aScalar = rewriter.create<LLVM::BitcastOp>(loc, i32Ty, a4);
      Value aWords =
          rewriter.create<vector::BroadcastOp>(loc, v4i32Ty, aScalar);
      Value aBroadcast = rewriter.create<LLVM::BitcastOp>(loc, v16i8Ty, aWords);

      Value bPacked;
      if (!candidate.rhsPackedN4Buf.empty()) {
        // Physical [N,4] storage is exactly
        //   [n0.k0..k3, n1.k0..k3, n2.k0..k3, n3.k0..k3].
        // It is already in SDOT lane order; avoid loading/transposing the
        // complete [4,N] logical vector and round-tripping it through stack.
        Value packedOffset = rewriter.create<arith::MulIOp>(loc, nBase, c4);
        SmallVector<Value> indices(candidate.rhsPackedN4Buf.indices);
        indices.back() = addIndex(loc, indices.back(), packedOffset, rewriter);
        bPacked = rewriter.create<vector::LoadOp>(
            loc, v16i8Ty, candidate.rhsPackedN4Buf.memRef, indices);
      } else {
        SmallVector<Value> bRows(4);
        for (int64_t r = 0; r < 4; ++r) {
          Value row = addIndex(loc, kBase, r, rewriter);
          bRows[r] = rewriter.create<vector::LoadOp>(
              loc, v4i8Ty, candidate.rhsBuf.memRef,
              get2DIndices(loc, candidate.rhsBuf, row, nBase, rewriter));
        }
        Value b01 = rewriter.create<vector::InsertStridedSliceOp>(
            loc, bRows[0], zero8, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        b01 = rewriter.create<vector::InsertStridedSliceOp>(
            loc, bRows[1], b01, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{1});
        Value b23 = rewriter.create<vector::InsertStridedSliceOp>(
            loc, bRows[2], zero8, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        b23 = rewriter.create<vector::InsertStridedSliceOp>(
            loc, bRows[3], b23, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{1});
        Value b01Wide = rewriter.create<vector::InsertStridedSliceOp>(
            loc, b01, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        Value b23Wide = rewriter.create<vector::InsertStridedSliceOp>(
            loc, b23, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        bPacked = rewriter.create<vector::ShuffleOp>(loc, b01Wide, b23Wide,
                                                     transposeB);
      }

      Value next =
          rewriter
              .create<LLVM::CallIntrinsicOp>(
                  loc, v4i32Ty, sdot,
                  ValueRange{kLoop.getRegionIterArgs()[0], aBroadcast, bPacked})
              .getResult(0);
      rewriter.setInsertionPointToEnd(kLoop.getBody());
      rewriter.create<scf::YieldOp>(loc, next);
    }

    rewriter.setInsertionPointAfter(kLoop);
    rewriter.create<vector::StoreOp>(
        loc, kLoop.getResult(0), accBuf.memRef,
        get2DIndices(loc, accBuf, 0, nBase, rewriter));
  }

  rewriter.setInsertionPointAfter(nLoop);
  if (candidate.keepAccInBuffer) {
    Value loopResult =
        outerKLoop.getTiedLoopResult(cast<BlockArgument>(op.getC()));
    rewriter.replaceOp(op, op.getC());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(outerKLoop);
    Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
    rewriter.replaceAllUsesWith(loopResult, result);
  } else {
    Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
    rewriter.replaceOp(op, result);
  }
  return success();
}

static LogicalResult convertCandidate(SVE2I8MMCandidate &candidate,
                                      PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  auto lhsTy = cast<VectorType>(op.getA().getType());
  auto rhsTy = cast<VectorType>(op.getB().getType());
  auto accTy = cast<VectorType>(op.getC().getType());
  Type i8Ty = lhsTy.getElementType();
  Type i32Ty = accTy.getElementType();
  int64_t mSize = lhsTy.getDimSize(0);
  int64_t kSize = lhsTy.getDimSize(1);
  int64_t nSize = rhsTy.getDimSize(1);
  int64_t nBlock = nSize % 8 == 0 ? 8 : 4;
  int64_t nPairs = nBlock / 2;

  // See isCandidate(): preserve the fast direct-buffer case, but provide a
  // general compiler path for computed vectors such as Q4 nibble unpack.
  // The allocas live at function scope while the stores remain immediately
  // before the replacement microkernel, so loop-varying values are correct.
  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();
  if (candidate.lhsBuf.empty())
    candidate.lhsBuf = storeToTmpBuffer(loc, op.getA(), allocaPoint, rewriter);
  if (candidate.rhsBuf.empty() && candidate.rhsPackedN4Buf.empty())
    candidate.rhsBuf = storeToTmpBuffer(loc, op.getB(), allocaPoint, rewriter);

  if (mSize == 1)
    return convertCandidateM1(candidate, rewriter);

  auto v8i8Ty = VectorType::get({8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v2i32Ty = VectorType::get({2}, i32Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  Type nxv16i8Ty = LLVM::LLVMScalableVectorType::get(i8Ty, 16);
  Type nxv4i32Ty = LLVM::LLVMScalableVectorType::get(i32Ty, 4);
  StringAttr smmla =
      StringAttr::get(op.getContext(), "llvm.aarch64.sve.smmla.nxv4i32");

  scf::ForOp outerKLoop;
  Value initialAcc = op.getC();
  if (candidate.keepAccInBuffer) {
    outerKLoop = cast<scf::ForOp>(op->getParentOp());
    initialAcc = getInitAccValue(op.getC());
  }
  bool zeroInitialAcc =
      !candidate.keepAccInBuffer && matchPattern(initialAcc, m_Zero());

  MemBuffer accBuf;
  Value bPack;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    if (candidate.keepAccInBuffer)
      rewriter.setInsertionPoint(outerKLoop);

    accBuf = allocateTmpBufferStack(loc, accTy, allocaPoint, rewriter);
    if (!zeroInitialAcc)
      op_write(initialAcc, accBuf.memRef, accBuf.indices);

    auto bPackTy = MemRefType::get({kSize / 8, nPairs, 16}, i8Ty);
    {
      OpBuilder::InsertionGuard allocaGuard(rewriter);
      rewriter.setInsertionPoint(allocaPoint);
      bPack = rewriter.create<memref::AllocaOp>(
          loc, bPackTy, rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
    }
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value c8 = rewriter.create<arith::ConstantIndexOp>(loc, 8);
  Value nStep = rewriter.create<arith::ConstantIndexOp>(loc, nBlock);
  Value mLimit = rewriter.create<arith::ConstantIndexOp>(loc, mSize);
  Value nLimit = rewriter.create<arith::ConstantIndexOp>(loc, nSize);
  Value kSteps = rewriter.create<arith::ConstantIndexOp>(loc, kSize / 8);

  // N is outermost so each packed B panel is reused by every M block.
  auto nLoop = rewriter.create<scf::ForOp>(loc, c0, nLimit, nStep);
  {
    OpBuilder::InsertionGuard nGuard(rewriter);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nBase = nLoop.getInductionVar();

    // Pack one Kx8 B panel.  Loading an 8x8 block by rows keeps source memory
    // contiguous; the local transpose creates four SMMLA 2x8 column pairs.
    auto packLoop = rewriter.create<scf::ForOp>(loc, c0, kSteps, c1);
    {
      OpBuilder::InsertionGuard packGuard(rewriter);
      rewriter.setInsertionPointToStart(packLoop.getBody());
      Value ki = packLoop.getInductionVar();
      Value kBase = rewriter.create<arith::MulIOp>(loc, ki, c8);
      SmallVector<Value> rows(8);
      for (int64_t r = 0; r < 8; ++r) {
        Value row = addIndex(loc, kBase, r, rewriter);
        if (nBlock == 8) {
          rows[r] = rewriter.create<vector::LoadOp>(
              loc, v8i8Ty, candidate.rhsBuf.memRef,
              get2DIndices(loc, candidate.rhsBuf, row, nBase, rewriter));
        } else {
          auto v4i8Ty = VectorType::get({4}, i8Ty);
          Value row4 = rewriter.create<vector::LoadOp>(
              loc, v4i8Ty, candidate.rhsBuf.memRef,
              get2DIndices(loc, candidate.rhsBuf, row, nBase, rewriter));
          Value zero8 = rewriter.create<arith::ConstantOp>(
              loc, v8i8Ty, rewriter.getZeroAttr(v8i8Ty));
          rows[r] = rewriter.create<vector::InsertStridedSliceOp>(
              loc, row4, zero8, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        }
      }

      // Spell out an 8x8 byte transpose as the same three-level TRN network
      // used by an optimized NEON C implementation.  A generic
      // vector.transpose was scalarized into dozens of lane moves by LLVM;
      // these shuffle masks select trn1/trn2 (and zip1/zip2 for 32-bit lanes).
      constexpr int64_t trn1b[] = {0, 8, 2, 10, 4, 12, 6, 14};
      constexpr int64_t trn2b[] = {1, 9, 3, 11, 5, 13, 7, 15};
      constexpr int64_t trn1h[] = {0, 4, 2, 6};
      constexpr int64_t trn2h[] = {1, 5, 3, 7};
      constexpr int64_t trn1s[] = {0, 2};
      constexpr int64_t trn2s[] = {1, 3};
      constexpr int64_t concat[] = {0, 1, 2,  3,  4,  5,  6,  7,
                                    8, 9, 10, 11, 12, 13, 14, 15};

      SmallVector<Value> bytes(8);
      for (int64_t pair = 0; pair < 4; ++pair) {
        auto [lo, hi] = transposePair(loc, rows[pair * 2], rows[pair * 2 + 1],
                                      trn1b, trn2b, rewriter);
        bytes[pair * 2] = lo;
        bytes[pair * 2 + 1] = hi;
      }

      auto v4i16Ty = VectorType::get({4}, rewriter.getI16Type());
      for (Value &value : bytes)
        value = rewriter.create<vector::BitCastOp>(loc, v4i16Ty, value);
      SmallVector<Value> halfwords(8);
      constexpr std::pair<int64_t, int64_t> halfwordPairs[] = {
          {0, 2}, {1, 3}, {4, 6}, {5, 7}};
      for (auto [idx, halfwordPair] : llvm::enumerate(halfwordPairs)) {
        auto [lo, hi] =
            transposePair(loc, bytes[halfwordPair.first],
                          bytes[halfwordPair.second], trn1h, trn2h, rewriter);
        halfwords[idx * 2] = lo;
        halfwords[idx * 2 + 1] = hi;
      }

      auto v2i32Ty = VectorType::get({2}, rewriter.getI32Type());
      for (Value &value : halfwords)
        value = rewriter.create<vector::BitCastOp>(loc, v2i32Ty, value);
      constexpr std::pair<int64_t, int64_t> wordPairs[] = {
          {0, 4}, {2, 6}, {1, 5}, {3, 7}};
      SmallVector<Value> columns(8);
      for (auto [idx, wordPair] : llvm::enumerate(wordPairs)) {
        auto [lo, hi] =
            transposePair(loc, halfwords[wordPair.first],
                          halfwords[wordPair.second], trn1s, trn2s, rewriter);
        columns[idx] = lo;
        columns[idx + 4] = hi;
      }
      for (Value &value : columns)
        value = rewriter.create<vector::BitCastOp>(loc, v8i8Ty, value);

      for (int64_t pair = 0; pair < nPairs; ++pair) {
        Value packed = rewriter.create<vector::ShuffleOp>(
            loc, columns[pair * 2], columns[pair * 2 + 1], concat);
        Value pairIdx = rewriter.create<arith::ConstantIndexOp>(loc, pair);
        rewriter.create<vector::StoreOp>(loc, packed, bPack,
                                         ValueRange{ki, pairIdx, c0});
      }
    }

    auto mLoop = rewriter.create<scf::ForOp>(loc, c0, mLimit, c8);
    {
      OpBuilder::InsertionGuard mGuard(rewriter);
      rewriter.setInsertionPointToStart(mLoop.getBody());
      Value mBase = mLoop.getInductionVar();

      SmallVector<Value> initAcc(4 * nPairs);
      if (zeroInitialAcc) {
        Value zero = rewriter.create<LLVM::ZeroOp>(loc, nxv4i32Ty);
        std::fill(initAcc.begin(), initAcc.end(), zero);
      } else {
        for (int64_t mr = 0; mr < 4; ++mr) {
          Value row = addIndex(loc, mBase, mr * 2, rewriter);
          for (int64_t nr = 0; nr < nPairs; ++nr) {
            Value col = addIndex(loc, nBase, nr * 2, rewriter);
            initAcc[mr * nPairs + nr] = loadAcc2x2(
                loc, accBuf, row, col, v2i32Ty, v4i32Ty, nxv4i32Ty, rewriter);
          }
        }
      }

      auto computeLoop =
          rewriter.create<scf::ForOp>(loc, c0, kSteps, c1, initAcc);
      {
        OpBuilder::InsertionGuard computeGuard(rewriter);
        rewriter.setInsertionPointToStart(computeLoop.getBody());
        Value ki = computeLoop.getInductionVar();
        Value kBase = rewriter.create<arith::MulIOp>(loc, ki, c8);

        SmallVector<Value> aPacked(4);
        SmallVector<Value> bPacked(nPairs);
        for (int64_t pair = 0; pair < 4; ++pair) {
          Value row0 = addIndex(loc, mBase, pair * 2, rewriter);
          Value row1 = addIndex(loc, row0, 1, rewriter);
          Value a0 = rewriter.create<vector::LoadOp>(
              loc, v8i8Ty, candidate.lhsBuf.memRef,
              get2DIndices(loc, candidate.lhsBuf, row0, kBase, rewriter));
          Value a1 = rewriter.create<vector::LoadOp>(
              loc, v8i8Ty, candidate.lhsBuf.memRef,
              get2DIndices(loc, candidate.lhsBuf, row1, kBase, rewriter));
          aPacked[pair] =
              packRows2x8(loc, a0, a1, v16i8Ty, nxv16i8Ty, rewriter);

          if (pair < nPairs) {
            Value pairIdx = rewriter.create<arith::ConstantIndexOp>(loc, pair);
            Value bFixed = rewriter.create<vector::LoadOp>(
                loc, v16i8Ty, bPack, ValueRange{ki, pairIdx, c0});
            bPacked[pair] =
                insertFixedInScalable(loc, bFixed, nxv16i8Ty, rewriter);
          }
        }

        SmallVector<Value> nextAcc(4 * nPairs);
        ValueRange carried = computeLoop.getRegionIterArgs();
        for (int64_t mr = 0; mr < 4; ++mr) {
          for (int64_t nr = 0; nr < nPairs; ++nr) {
            int64_t idx = mr * nPairs + nr;
            nextAcc[idx] =
                rewriter
                    .create<LLVM::CallIntrinsicOp>(
                        loc, nxv4i32Ty, smmla,
                        ValueRange{carried[idx], aPacked[mr], bPacked[nr]})
                    .getResult(0);
          }
        }

        // The scf.for builder cannot synthesize a terminator for non-empty
        // iter_args because there is no identity yield to infer.  Finish the
        // newly-created body explicitly.
        rewriter.setInsertionPointToEnd(computeLoop.getBody());
        rewriter.create<scf::YieldOp>(loc, nextAcc);
      }

      rewriter.setInsertionPointAfter(computeLoop);
      ValueRange finalAcc = computeLoop.getResults();
      for (int64_t mr = 0; mr < 4; ++mr) {
        Value row = addIndex(loc, mBase, mr * 2, rewriter);
        for (int64_t nr = 0; nr < nPairs; ++nr) {
          Value col = addIndex(loc, nBase, nr * 2, rewriter);
          storeAcc2x2(loc, accBuf, row, col, finalAcc[mr * nPairs + nr],
                      v2i32Ty, v4i32Ty, rewriter);
        }
      }
    }
  }

  rewriter.setInsertionPointAfter(nLoop);
  if (candidate.keepAccInBuffer) {
    Value loopResult =
        outerKLoop.getTiedLoopResult(cast<BlockArgument>(op.getC()));
    rewriter.replaceOp(op, op.getC());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(outerKLoop);
    Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
    rewriter.replaceAllUsesWith(loopResult, result);
  } else {
    Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
    rewriter.replaceOp(op, result);
  }
  return success();
}

struct ConvertDotToSVE2I8MM
    : public triton::cpu::impl::ConvertDotToSVE2I8MMBase<ConvertDotToSVE2I8MM> {
  ConvertDotToSVE2I8MM() = default;
  explicit ConvertDotToSVE2I8MM(bool w4OnlyValue) {
    this->w4Only = w4OnlyValue;
  }
  ConvertDotToSVE2I8MM(bool w4OnlyValue, bool fixedI8MMValue) {
    this->w4Only = w4OnlyValue;
    this->fixedI8MM = fixedI8MMValue;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<vector::MultiDimReductionOp, 2> addpReductions;
    SmallVector<vector::MultiDimReductionOp, 4> sdotReductions;
    SmallVector<arith::MulFOp, 2> fixedScaleConversions;
    mod.walk([&](vector::MultiDimReductionOp op) {
      Value lhs;
      Value rhs;
      if (matchNeonAddpReduction(op, lhs, rhs))
        addpReductions.push_back(op);
      if (isNeonSdotReduction(op))
        sdotReductions.push_back(op);
    });
    mod.walk([&](arith::MulFOp op) {
      arith::SIToFPOp conversion;
      if (matchI32ToF32Scale16(op, conversion))
        fixedScaleConversions.push_back(op);
    });
    for (auto op : addpReductions) {
      Value lhs;
      Value rhs;
      if (!matchNeonAddpReduction(op, lhs, rhs))
        continue;
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(op);
      convertNeonAddpReduction(op, lhs, rhs, rewriter);
    }
    for (auto op : sdotReductions) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(op);
      convertNeonSdotReduction(op, rewriter);
    }
    for (auto op : fixedScaleConversions) {
      arith::SIToFPOp conversion;
      if (!matchI32ToF32Scale16(op, conversion))
        continue;
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(op);
      convertI32ToF32Scale16(op, conversion, rewriter);
    }

    SmallVector<W4A8DotPair, 1> w4Pairs;
    llvm::SmallPtrSet<Operation *, 4> pairedDots;
    mod.walk([&](cpu::DotOp op) {
      if (pairedDots.contains(op))
        return WalkResult::advance();
      W4A8DotPair pair;
      if (matchW4A8DotPair(op, pair)) {
        w4Pairs.push_back(pair);
        pairedDots.insert(pair.lowDot);
        pairedDots.insert(pair.highDot);
      }
      return WalkResult::advance();
    });

    SmallVector<PackedQ4M4LoopCandidate, 1> packedQ4M4Loops;
    llvm::SmallPtrSet<Operation *, 4> packedQ4M4Dots;
    SmallVector<PackedQ4PrefillCandidate, 1> packedQ4Prefill;
    llvm::SmallPtrSet<Operation *, 4> packedQ4Dots;
    SmallVector<PackedW8PrefillCandidate, 1> packedW8Prefill;
    llvm::SmallPtrSet<Operation *, 4> packedW8Dots;
    llvm::SmallPtrSet<Operation *, 2> packedW8Loops;
    SmallVector<SVE2I8MMCandidate, 1> candidates;
    if (!this->w4Only || this->fixedI8MM) {
      mod.walk([&](scf::ForOp loop) {
        PackedQ4M4LoopCandidate candidate;
        if (matchPackedQ4M4Loop(loop, candidate)) {
          packedQ4M4Loops.push_back(candidate);
          for (const PackedQ4M4Epilogue &panel : candidate.panels)
            packedQ4M4Dots.insert(panel.dot.op);
        }
        return WalkResult::advance();
      });
    }
    if (!this->w4Only) {
      mod.walk([&](cpu::DotOp op) {
        if (pairedDots.contains(op) || packedQ4M4Dots.contains(op))
          return WalkResult::advance();
        PackedW8PrefillCandidate candidate;
        if (matchPackedW8Prefill(op, candidate)) {
          packedW8Dots.insert(op);
          Operation *parent = op->getParentOp();
          if (!isa_and_nonnull<scf::ForOp>(parent) ||
              packedW8Loops.insert(parent).second)
            packedW8Prefill.push_back(candidate);
        }
        return WalkResult::advance();
      });
      mod.walk([&](cpu::DotOp op) {
        if (pairedDots.contains(op) || packedQ4M4Dots.contains(op) ||
            packedW8Dots.contains(op))
          return WalkResult::advance();
        PackedQ4PrefillCandidate candidate;
        if (matchPackedQ4Prefill(op, candidate)) {
          packedQ4Prefill.push_back(candidate);
          packedQ4Dots.insert(op);
        }
        return WalkResult::advance();
      });
      mod.walk([&](cpu::DotOp op) {
        if (pairedDots.contains(op) || packedQ4M4Dots.contains(op) ||
            packedW8Dots.contains(op) || packedQ4Dots.contains(op))
          return WalkResult::advance();
        SVE2I8MMCandidate candidate;
        if (isCandidate(op, candidate))
          candidates.push_back(candidate);
        return WalkResult::advance();
      });
    }

    for (auto &pair : w4Pairs) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(pair.lowDot);
      if (failed(convertW4A8DotPair(pair, rewriter)))
        pair.lowDot.emitRemark("packed W4A8 SDOT fusion skipped");
    }

    for (auto &candidate : packedQ4M4Loops) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.loop);
      if (failed(convertPackedQ4M4Loop(candidate, rewriter)))
        candidate.loop.emitRemark("packed Q4 M4 loop fusion skipped");
    }

    for (auto &candidate : packedW8Prefill) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (failed(convertPackedW8Prefill(candidate, rewriter)))
        candidate.op.emitRemark("packed W8 prefill i8mm lowering skipped");
    }

    for (auto &candidate : packedQ4Prefill) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (failed(convertPackedQ4Prefill(candidate, rewriter)))
        candidate.op.emitRemark("packed Q4 prefill i8mm lowering skipped");
    }

    for (auto &candidate : candidates) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (failed(convertCandidate(candidate, rewriter)))
        candidate.op.emitRemark("SVE2 i8mm rolled lowering skipped");
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToSVE2I8MM() {
  return std::make_unique<ConvertDotToSVE2I8MM>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToSVE2I8MM(bool w4Only) {
  return std::make_unique<ConvertDotToSVE2I8MM>(w4Only);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToSVE2I8MM(bool w4Only, bool fixedI8MM) {
  return std::make_unique<ConvertDotToSVE2I8MM>(w4Only, fixedI8MM);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
