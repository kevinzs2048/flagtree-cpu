/*
 * Lower NeonSdotOp → LLVM SDOT intrinsic
 * Lower CpuSdotGemvOp → runtime call to sdot_gemv_m1_prepacked()
 */

#include "cpu/include/TritonCPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdlib>
#include <limits>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

static int64_t getNonNegativeEnv(StringRef name, int64_t fallback,
                                 int64_t maximum) {
  const char *text = std::getenv(name.str().c_str());
  if (!text || !*text)
    return fallback;
  char *end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (!end || *end != '\0' || value < 0)
    return fallback;
  return std::min<int64_t>(value, maximum);
}

// Emit the u10 single-precision exp approximation used by SLEEF.  Keeping the
// expression in IR lets LLVM schedule it together with the fused MLP epilogue
// instead of preserving every live dot accumulator across an external vector
// exp call.  It remains possible to select the vector-library path with
// TRITON_CPU_INLINE_EXP_U10=0 for architecture-specific comparisons.
static Value emitInlineSleefExpfU10(Location loc, Value input,
                                    StringAttr roundEven,
                                    PatternRewriter &rewriter) {
  auto vectorTy = cast<VectorType>(input.getType());
  auto i32VectorTy =
      VectorType::get(vectorTy.getShape(), rewriter.getI32Type());

  auto splatF32 = [&](float value) -> Value {
    Value scalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(value));
    return rewriter.create<vector::BroadcastOp>(loc, vectorTy, scalar);
  };
  auto splatI32 = [&](int32_t value) -> Value {
    Value scalar = rewriter.create<arith::ConstantIntOp>(loc, value, 32);
    return rewriter.create<vector::BroadcastOp>(loc, i32VectorTy, scalar);
  };
  auto fma = [&](Value a, Value b, Value c) -> Value {
    return rewriter.create<math::FmaOp>(loc, a, b, c);
  };

  Value scaled =
      rewriter.create<arith::MulFOp>(loc, input, splatF32(1.4426950408889634f));
  Value rounded = rewriter
                      .create<LLVM::CallIntrinsicOp>(loc, vectorTy, roundEven,
                                                     ValueRange{scaled})
                      .getResult(0);
  Value q = rewriter.create<arith::FPToSIOp>(loc, i32VectorTy, rounded);
  Value qf = rewriter.create<arith::SIToFPOp>(loc, vectorTy, q);

  Value reduced = fma(qf, splatF32(-0.693145751953125f), input);
  reduced = fma(qf, splatF32(-1.428606765330187e-6f), reduced);

  Value polynomial = splatF32(0.00019852761761285365f);
  polynomial = fma(polynomial, reduced, splatF32(0.0013930435525253415f));
  polynomial = fma(polynomial, reduced, splatF32(0.008333360776305199f));
  polynomial = fma(polynomial, reduced, splatF32(0.041666485369205475f));
  polynomial = fma(polynomial, reduced, splatF32(0.1666666716337204f));
  polynomial = fma(polynomial, reduced, splatF32(0.5f));
  Value reducedSquared = rewriter.create<arith::MulFOp>(loc, reduced, reduced);
  Value result = fma(reducedSquared, polynomial, reduced);
  result = rewriter.create<arith::AddFOp>(loc, splatF32(1.0f), result);

  Value halfQ = rewriter.create<arith::ShRSIOp>(loc, q, splatI32(1));
  Value pow0Exponent =
      rewriter.create<arith::AddIOp>(loc, halfQ, splatI32(127));
  Value pow1Exponent = rewriter.create<arith::AddIOp>(
      loc, rewriter.create<arith::SubIOp>(loc, q, halfQ), splatI32(127));
  Value pow0Bits =
      rewriter.create<arith::ShLIOp>(loc, pow0Exponent, splatI32(23));
  Value pow1Bits =
      rewriter.create<arith::ShLIOp>(loc, pow1Exponent, splatI32(23));
  Value pow0 = rewriter.create<arith::BitcastOp>(loc, vectorTy, pow0Bits);
  Value pow1 = rewriter.create<arith::BitcastOp>(loc, vectorTy, pow1Bits);
  result = rewriter.create<arith::MulFOp>(loc, result, pow0);
  result = rewriter.create<arith::MulFOp>(loc, result, pow1);

  Value belowRange = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, input, splatF32(-104.0f));
  result =
      rewriter.create<arith::SelectOp>(loc, belowRange, splatF32(0.0f), result);
  Value aboveRange = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, input, splatF32(100.0f));
  return rewriter.create<arith::SelectOp>(
      loc, aboveRange, splatF32(std::numeric_limits<float>::infinity()),
      result);
}

// The blocked W8 layout makes one K-step a sequential BLOCK_N*4-byte stream.
// Prefetch one cache line out of every `groupStride` groups for a future
// K-step. A single bounds check protects all hints in the iteration.
static void maybePrefetchPackedWeights(Location loc, ValueRange packedMemrefs,
                                       Value nBlock, Value kb, Value k4Limit,
                                       Value c0, int64_t groups,
                                       PatternRewriter &rewriter) {
  int64_t distance =
      getNonNegativeEnv("TRITON_CPU_SDOT_PREFETCH_DISTANCE", 0, 32);
  if (distance == 0 || packedMemrefs.empty())
    return;
  int64_t groupStride = std::max<int64_t>(
      1, getNonNegativeEnv("TRITON_CPU_SDOT_PREFETCH_GROUP_STRIDE", 4, 64));

  Value distanceValue = rewriter.create<arith::ConstantIndexOp>(loc, distance);
  Value futureKb = rewriter.create<arith::AddIOp>(loc, kb, distanceValue);
  Value inBounds = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, futureKb, k4Limit);
  auto prefetchIf = rewriter.create<scf::IfOp>(loc, inBounds, false);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&prefetchIf.getThenRegion().front());
  for (Value packedMemref : packedMemrefs) {
    for (int64_t group = 0; group < groups; group += groupStride) {
      Value groupOffset = rewriter.create<arith::ConstantIndexOp>(loc, group);
      rewriter.create<memref::PrefetchOp>(
          loc, packedMemref, ValueRange{nBlock, futureKb, groupOffset, c0},
          /*isWrite=*/false, /*localityHint=*/3, /*isDataCache=*/true);
    }
  }
}

// Return the exact absolute maximum for a finite BF16 vector. Clearing the
// sign bit leaves a monotonic magnitude encoding, so four K8 slices can be
// reduced with integer UMAX before the loop-carried scalar FP32 maximum. The
// static suffix keeps the operation valid for every K divisible by four.
static Value emitFiniteBf16Absmax(Location loc, Value xMem, int64_t kSize,
                                  PatternRewriter &rewriter) {
  Type bf16Ty = rewriter.getBF16Type();
  Type f32Ty = rewriter.getF32Type();
  Type i16Ty = rewriter.getI16Type();
  auto v4bf16Ty = VectorType::get({4}, bf16Ty);
  auto v8bf16Ty = VectorType::get({8}, bf16Ty);
  auto v4f32Ty = VectorType::get({4}, f32Ty);
  auto v8i16Ty = VectorType::get({8}, i16Ty);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
  Value c32 = rewriter.create<arith::ConstantIndexOp>(loc, 32);
  Value zeroF = rewriter.create<arith::ConstantOp>(
      loc, f32Ty, rewriter.getF32FloatAttr(0.0));
  int64_t fullK = (kSize / 32) * 32;
  Value fullLimit = rewriter.create<arith::ConstantIndexOp>(loc, fullK);
  Value kLimit = rewriter.create<arith::ConstantIndexOp>(loc, kSize);
  Value magnitudeMaskScalar =
      rewriter.create<arith::ConstantIntOp>(loc, 0x7fff, 16);
  Value magnitudeMask =
      rewriter.create<vector::BroadcastOp>(loc, v8i16Ty, magnitudeMaskScalar);
  auto maxLoop =
      rewriter.create<scf::ForOp>(loc, c0, fullLimit, c32, ValueRange{zeroF});
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(maxLoop.getBody());
    Value offset = maxLoop.getInductionVar();
    auto loadMagnitude = [&](int64_t delta) -> Value {
      Value loadOffset = offset;
      if (delta != 0) {
        Value deltaValue = rewriter.create<arith::ConstantIndexOp>(loc, delta);
        loadOffset = rewriter.create<arith::AddIOp>(loc, offset, deltaValue);
      }
      Value values = rewriter.create<vector::LoadOp>(loc, v8bf16Ty, xMem,
                                                     ValueRange{loadOffset});
      Value bits = rewriter.create<arith::BitcastOp>(loc, v8i16Ty, values);
      return rewriter.create<arith::AndIOp>(loc, bits, magnitudeMask);
    };
    Value bits0 = loadMagnitude(0);
    Value bits1 = loadMagnitude(8);
    Value bits2 = loadMagnitude(16);
    Value bits3 = loadMagnitude(24);
    Value max01 = rewriter.create<arith::MaxUIOp>(loc, bits0, bits1);
    Value max23 = rewriter.create<arith::MaxUIOp>(loc, bits2, bits3);
    Value laneMax = rewriter.create<arith::MaxUIOp>(loc, max01, max23);
    Value blockBits = rewriter.create<vector::ReductionOp>(
        loc, vector::CombiningKind::MAXUI, laneMax);
    Value blockBf16 = rewriter.create<arith::BitcastOp>(loc, bf16Ty, blockBits);
    Value blockAbsmax = rewriter.create<arith::ExtFOp>(loc, f32Ty, blockBf16);
    Value nextMax = rewriter.create<arith::MaxNumFOp>(
        loc, maxLoop.getRegionIterArgs()[0], blockAbsmax);
    rewriter.setInsertionPointToEnd(maxLoop.getBody());
    rewriter.create<scf::YieldOp>(loc, nextMax);
  }
  rewriter.setInsertionPointAfter(maxLoop);
  Value rawMax = maxLoop.getResult(0);

  if (fullK != kSize) {
    auto tailLoop = rewriter.create<scf::ForOp>(loc, fullLimit, kLimit, c4,
                                                ValueRange{rawMax});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      Value offset = tailLoop.getInductionVar();
      Value xb = rewriter.create<vector::LoadOp>(loc, v4bf16Ty, xMem,
                                                 ValueRange{offset});
      Value xf = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, xb);
      Value abs = rewriter.create<math::AbsFOp>(loc, xf);
      Value nextMax = tailLoop.getRegionIterArgs()[0];
      for (int64_t lane = 0; lane < 4; ++lane) {
        Value value = rewriter.create<vector::ExtractOp>(loc, abs, lane);
        nextMax = rewriter.create<arith::MaxNumFOp>(loc, nextMax, value);
      }
      rewriter.setInsertionPointToEnd(tailLoop.getBody());
      rewriter.create<scf::YieldOp>(loc, nextMax);
    }
    rewriter.setInsertionPointAfter(tailLoop);
    rawMax = tailLoop.getResult(0);
  }
  return rawMax;
}

} // namespace

// ---------- NeonSdotOp → LLVM intrinsic ----------

struct NeonSdotOpLowering : public OpRewritePattern<NeonSdotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NeonSdotOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto v4i32Ty = VectorType::get({4}, rewriter.getI32Type());
    auto sdotName = StringAttr::get(ctx, "llvm.aarch64.neon.sdot.v4i32.v16i8");
    auto result = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, v4i32Ty, sdotName, ValueRange{op.getAcc(), op.getA(), op.getB()},
        LLVM::FastmathFlagsAttr());
    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

// ---------- CpuSdotGemvOp → runtime function call ----------

struct SdotGemvOpLowering : public OpRewritePattern<triton::cpu::SdotGemvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SdotGemvOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();

    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    // Get or declare the runtime function
    auto funcName = "sdot_gemv_m1_prepacked";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    // Convert tt.ptr to llvm.ptr via unrealized_conversion_cast
    auto castToLLVMPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    auto aPtr = castToLLVMPtr(op.getAPtr());
    auto bPtr = castToLLVMPtr(op.getBPackedPtr());
    auto cPtr = castToLLVMPtr(op.getCPtr());

    auto four = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(4));
    auto N4 = rewriter.create<LLVM::SDivOp>(loc, i64Ty, op.getN(), four);

    rewriter.create<LLVM::CallOp>(
        loc, funcOp, ValueRange{aPtr, bPtr, cPtr, op.getK(), op.getN(), N4});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuFusedDecodeStepOp → runtime call (returns i64) ----------

struct FusedDecodeStepOpLowering
    : public OpRewritePattern<triton::cpu::FusedDecodeStepOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::FusedDecodeStepOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "fused_decode_step";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      // 2 i64 + 9 ptr + 8 i64 + 1 f32 = 20 args, returns i64
      SmallVector<Type, 20> argTypes;
      argTypes.push_back(i64Ty); // token_id
      argTypes.push_back(i64Ty); // pos
      for (int i = 0; i < 9; i++)
        argTypes.push_back(ptrTy); // 9 ptrs
      for (int i = 0; i < 8; i++)
        argTypes.push_back(i64Ty); // 8 dims
      argTypes.push_back(f32Ty);   // rms_eps
      auto funcType = LLVM::LLVMFunctionType::get(i64Ty, argTypes, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };
    auto toI64 = [&](Value v) -> Value {
      if (v.getType() == i64Ty)
        return v;
      return rewriter.create<LLVM::SExtOp>(loc, i64Ty, v);
    };

    auto epsVal =
        rewriter.create<LLVM::ConstantOp>(loc, f32Ty, op.getRmsEpsAttr());

    SmallVector<Value, 20> args;
    args.push_back(toI64(op.getTokenId()));
    args.push_back(toI64(op.getPos()));
    args.push_back(castPtr(op.getEmbedTable()));
    args.push_back(castPtr(op.getLayerPtrs()));
    args.push_back(castPtr(op.getKCache()));
    args.push_back(castPtr(op.getVCache()));
    args.push_back(castPtr(op.getRopeCos()));
    args.push_back(castPtr(op.getRopeSin()));
    args.push_back(castPtr(op.getFinalNormW()));
    args.push_back(castPtr(op.getLmHeadPacked()));
    args.push_back(castPtr(op.getLmHeadScale()));
    args.push_back(toI64(op.getHiddenDim()));
    args.push_back(toI64(op.getHeadDim()));
    args.push_back(toI64(op.getNHeads()));
    args.push_back(toI64(op.getNKvHeads()));
    args.push_back(toI64(op.getIntermediate()));
    args.push_back(toI64(op.getVocabSize()));
    args.push_back(toI64(op.getNLayers()));
    args.push_back(toI64(op.getMaxSeq()));
    args.push_back(epsVal);

    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

// ---------- CpuFusedTransformerLayerOp → runtime call ----------

struct FusedTransformerLayerOpLowering
    : public OpRewritePattern<triton::cpu::FusedTransformerLayerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::FusedTransformerLayerOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "fused_transformer_decode_layer";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      // 26 ptr + 5 i64 + 1 f32 = 32 args
      SmallVector<Type, 32> argTypes;
      // hidden_states
      argTypes.push_back(ptrTy);
      // wq wk wv wo wq_s wk_s wv_s wo_s (8 ptrs)
      for (int i = 0; i < 8; i++)
        argTypes.push_back(ptrTy);
      // q_norm k_norm cos sin (4 ptrs)
      for (int i = 0; i < 4; i++)
        argTypes.push_back(ptrTy);
      // k_cache v_cache (2 ptrs)
      argTypes.push_back(ptrTy);
      argTypes.push_back(ptrTy);
      // cache_pos max_seq (2 i64)
      argTypes.push_back(i64Ty);
      argTypes.push_back(i64Ty);
      // gate up down gate_s up_s down_s (6 ptrs)
      for (int i = 0; i < 6; i++)
        argTypes.push_back(ptrTy);
      // input_norm post_norm (2 ptrs)
      argTypes.push_back(ptrTy);
      argTypes.push_back(ptrTy);
      // hidden head_dim n_heads n_kv_heads intermediate (5 i64)
      for (int i = 0; i < 5; i++)
        argTypes.push_back(i64Ty);
      // rms_eps (1 f32)
      argTypes.push_back(f32Ty);

      auto funcType = LLVM::LLVMFunctionType::get(voidTy, argTypes, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    auto epsVal =
        rewriter.create<LLVM::ConstantOp>(loc, f32Ty, op.getRmsEpsAttr());

    SmallVector<Value, 32> args;
    args.push_back(castPtr(op.getHiddenStates()));
    args.push_back(castPtr(op.getWq()));
    args.push_back(castPtr(op.getWk()));
    args.push_back(castPtr(op.getWv()));
    args.push_back(castPtr(op.getWo()));
    args.push_back(castPtr(op.getWqS()));
    args.push_back(castPtr(op.getWkS()));
    args.push_back(castPtr(op.getWvS()));
    args.push_back(castPtr(op.getWoS()));
    args.push_back(castPtr(op.getQNormW()));
    args.push_back(castPtr(op.getKNormW()));
    args.push_back(castPtr(op.getCosEmb()));
    args.push_back(castPtr(op.getSinEmb()));
    args.push_back(castPtr(op.getKCache()));
    args.push_back(castPtr(op.getVCache()));
    args.push_back(op.getCachePos());
    args.push_back(op.getMaxSeqLen());
    args.push_back(castPtr(op.getGateW()));
    args.push_back(castPtr(op.getUpW()));
    args.push_back(castPtr(op.getDownW()));
    args.push_back(castPtr(op.getGateS()));
    args.push_back(castPtr(op.getUpS()));
    args.push_back(castPtr(op.getDownS()));
    args.push_back(castPtr(op.getInputNormW()));
    args.push_back(castPtr(op.getPostNormW()));
    args.push_back(op.getHiddenDim());
    args.push_back(op.getHeadDim());
    args.push_back(op.getNHeads());
    args.push_back(op.getNKvHeads());
    args.push_back(op.getIntermediate());
    args.push_back(epsVal);

    rewriter.create<LLVM::CallOp>(loc, funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSdotGemvFusedBf16Op → inline packed SDOT codegen ----------

struct SdotGemvFusedBf16OpLowering
    : public OpRewritePattern<triton::cpu::SdotGemvFusedBf16Op> {
  using OpRewritePattern::OpRewritePattern;

  static LogicalResult lowerRuntimeRange(triton::cpu::SdotGemvFusedBf16Op op,
                                         PatternRewriter &rewriter) {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto funcName = "sdot_gemv_m1_fused_bf16_blk_range";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType =
          LLVM::LLVMFunctionType::get(voidTy,
                                      {ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty,
                                       i64Ty, i64Ty, i64Ty, i64Ty},
                                      false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }
    auto castPtr = [&](Value value) -> Value {
      if (isa<LLVM::LLVMPointerType>(value.getType()))
        return value;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, value)
          .getResult(0);
    };
    Value four = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(4));
    Value n4 = rewriter.create<LLVM::SDivOp>(loc, i64Ty, op.getN(), four);
    Value bn4 = rewriter.create<LLVM::SDivOp>(loc, i64Ty, op.getNCount(), four);
    Value blockStart = rewriter.create<LLVM::SDivOp>(loc, i64Ty, op.getNStart(),
                                                     op.getNCount());
    Value blockCount = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(1));
    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getXPtr()), castPtr(op.getBPackedPtr()),
                   castPtr(op.getWScalePtr()), castPtr(op.getOutPtr()),
                   op.getK(), op.getN(), n4, bn4, blockStart, blockCount});
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult matchAndRewrite(triton::cpu::SdotGemvFusedBf16Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    APInt kValue, nValue, blockNValue;
    bool staticShape =
        matchPattern(op.getK(), m_ConstantInt(&kValue)) &&
        matchPattern(op.getN(), m_ConstantInt(&nValue)) &&
        matchPattern(op.getNCount(), m_ConstantInt(&blockNValue));
    int64_t kSize = staticShape ? kValue.getSExtValue() : 0;
    int64_t nSize = staticShape ? nValue.getSExtValue() : 0;
    int64_t blockN = staticShape ? blockNValue.getSExtValue() : 0;
    auto xPtrType = dyn_cast<triton::PointerType>(op.getXPtr().getType());
    auto outPtrType = dyn_cast<triton::PointerType>(op.getOutPtr().getType());
    Type xElement = xPtrType ? xPtrType.getPointeeType() : Type();
    Type outElement = outPtrType ? outPtrType.getPointeeType() : Type();
    bool f32IO = xElement.isF32() && outElement.isF32();
    bool bf16IO = xElement.isBF16() && outElement.isBF16();
    bool validShape = staticShape && kSize >= 4 && kSize % 4 == 0 &&
                      nSize >= 4 && nSize % 4 == 0 && blockN >= 4 &&
                      blockN % 4 == 0 && blockN <= nSize && (f32IO || bf16IO);
    const char *forceRuntime = std::getenv("TRITON_CPU_SDOT_FUSED_RUNTIME");
    if (!validShape)
      return rewriter.notifyMatchFailure(
          op,
          "inline fused GEMV requires static shape and matching BF16/F32 IO");
    if (bf16IO && forceRuntime && StringRef(forceRuntime) == "1")
      return lowerRuntimeRange(op, rewriter);

    Type bf16Ty = rewriter.getBF16Type();
    Type f32Ty = rewriter.getF32Type();
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    auto v4bf16Ty = VectorType::get({4}, bf16Ty);
    auto v4f32Ty = VectorType::get({4}, f32Ty);
    Type ioTy = f32IO ? f32Ty : bf16Ty;
    auto v4ioTy = VectorType::get({4}, ioTy);
    auto v4i8Ty = VectorType::get({4}, i8Ty);
    auto v16i8Ty = VectorType::get({16}, i8Ty);
    auto v4i32Ty = VectorType::get({4}, i32Ty);

    Value xMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({kSize}, ioTy), op.getXPtr());
    Value bMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize / blockN, kSize / 4, blockN / 4, 16}, i8Ty),
        op.getBPackedPtr());
    Value scaleMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, f32Ty), op.getWScalePtr());
    Value outMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, ioTy), op.getOutPtr());

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value kLimit = rewriter.create<arith::ConstantIndexOp>(loc, kSize);
    Value k4Limit = rewriter.create<arith::ConstantIndexOp>(loc, kSize / 4);
    Value zeroF = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(0.0));

    // First pass: vector BF16 loads with a scalar loop-carried absmax.  Keeping
    // this in generated IR allows LLVM to schedule the reduction with the
    // target's native BF16 conversion instructions.
    auto maxLoop =
        rewriter.create<scf::ForOp>(loc, c0, kLimit, c4, ValueRange{zeroF});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(maxLoop.getBody());
      Value offset = maxLoop.getInductionVar();
      Value xb = rewriter.create<vector::LoadOp>(loc, v4ioTy, xMem,
                                                 ValueRange{offset});
      Value xf = f32IO ? xb : rewriter.create<arith::ExtFOp>(loc, v4f32Ty, xb);
      Value abs = rewriter.create<math::AbsFOp>(loc, xf);
      Value nextMax = maxLoop.getRegionIterArgs()[0];
      for (int64_t lane = 0; lane < 4; ++lane) {
        Value value = rewriter.create<vector::ExtractOp>(loc, abs, lane);
        nextMax = rewriter.create<arith::MaxNumFOp>(loc, nextMax, value);
      }
      rewriter.setInsertionPointToEnd(maxLoop.getBody());
      rewriter.create<scf::YieldOp>(loc, nextMax);
    }
    rewriter.setInsertionPointAfter(maxLoop);
    Value epsilon = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(1.0e-8));
    Value maxAbs =
        rewriter.create<arith::MaxNumFOp>(loc, maxLoop.getResult(0), epsilon);
    Value c127 = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(127.0));
    Value invScale = rewriter.create<arith::DivFOp>(loc, c127, maxAbs);
    Value xScale = rewriter.create<arith::DivFOp>(loc, maxAbs, c127);
    Value invScaleVec =
        rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, invScale);

    Value nStart = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), op.getNStart());
    Value blockNIndex = rewriter.create<arith::ConstantIndexOp>(loc, blockN);
    Value nBlock = rewriter.create<arith::DivUIOp>(loc, nStart, blockNIndex);

    int64_t groups = blockN / 4;
    Value zeroAcc = rewriter.create<arith::ConstantOp>(
        loc, v4i32Ty, rewriter.getZeroAttr(v4i32Ty));
    SmallVector<Value> initialAcc(groups, zeroAcc);
    Value zero16 = rewriter.create<arith::ConstantOp>(
        loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
    constexpr int64_t broadcastA[] = {0, 1, 2, 3, 0, 1, 2, 3,
                                      0, 1, 2, 3, 0, 1, 2, 3};
    StringAttr roundEven = StringAttr::get(ctx, "llvm.roundeven.v4f32");
    StringAttr sdot =
        StringAttr::get(ctx, "llvm.aarch64.neon.sdot.v4i32.v16i8");

    // K is the only rolled compute loop.  N_count/4 accumulators are explicit
    // loop-carried SSA values, so BLOCK_N=64 maps to 16 vector accumulators
    // and stays within the AArch64 register file.
    auto dotLoop =
        rewriter.create<scf::ForOp>(loc, c0, k4Limit, c1, initialAcc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(dotLoop.getBody());
      Value kb = dotLoop.getInductionVar();
      Value xOffset = rewriter.create<arith::MulIOp>(loc, kb, c4);
      Value xb = rewriter.create<vector::LoadOp>(loc, v4ioTy, xMem,
                                                 ValueRange{xOffset});
      Value xf = f32IO ? xb : rewriter.create<arith::ExtFOp>(loc, v4f32Ty, xb);
      Value scaled = rewriter.create<arith::MulFOp>(loc, xf, invScaleVec);
      Value rounded = rewriter
                          .create<LLVM::CallIntrinsicOp>(
                              loc, v4f32Ty, roundEven, ValueRange{scaled})
                          .getResult(0);
      Value quant32 = rewriter.create<arith::FPToSIOp>(loc, v4i32Ty, rounded);
      Value quant8 = rewriter.create<arith::TruncIOp>(loc, v4i8Ty, quant32);
      Value a16 = rewriter.create<vector::InsertStridedSliceOp>(
          loc, quant8, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
      Value aBroadcast =
          rewriter.create<vector::ShuffleOp>(loc, a16, zero16, broadcastA);

      maybePrefetchPackedWeights(loc, ValueRange{bMem}, nBlock, kb, k4Limit, c0,
                                 groups, rewriter);

      SmallVector<Value> nextAcc;
      nextAcc.reserve(groups);
      ValueRange carried = dotLoop.getRegionIterArgs();
      for (int64_t group = 0; group < groups; ++group) {
        Value groupOffset = rewriter.create<arith::ConstantIndexOp>(loc, group);
        Value packed = rewriter.create<vector::LoadOp>(
            loc, v16i8Ty, bMem, ValueRange{nBlock, kb, groupOffset, c0});
        nextAcc.push_back(
            rewriter
                .create<LLVM::CallIntrinsicOp>(
                    loc, v4i32Ty, sdot,
                    ValueRange{carried[group], aBroadcast, packed})
                .getResult(0));
      }
      rewriter.setInsertionPointToEnd(dotLoop.getBody());
      rewriter.create<scf::YieldOp>(loc, nextAcc);
    }

    rewriter.setInsertionPointAfter(dotLoop);
    Value xScaleVec =
        rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, xScale);
    for (int64_t group = 0; group < groups; ++group) {
      Value groupOffset =
          rewriter.create<arith::ConstantIndexOp>(loc, group * 4);
      Value outOffset =
          rewriter.create<arith::AddIOp>(loc, nStart, groupOffset);
      Value accF32 = rewriter.create<arith::SIToFPOp>(loc, v4f32Ty,
                                                      dotLoop.getResult(group));
      Value weightScale = rewriter.create<vector::LoadOp>(
          loc, v4f32Ty, scaleMem, ValueRange{outOffset});
      Value dequant = rewriter.create<arith::MulFOp>(loc, accF32, xScaleVec);
      dequant = rewriter.create<arith::MulFOp>(loc, dequant, weightScale);
      Value output =
          f32IO ? dequant
                : rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, dequant);
      rewriter.create<vector::StoreOp>(loc, output, outMem,
                                       ValueRange{outOffset});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSdotGemvWholeOp → quantize once + rolled SDOT microtiles
// ----------

struct SdotGemvWholeOpLowering
    : public OpRewritePattern<triton::cpu::SdotGemvWholeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SdotGemvWholeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    APInt kValue, nValue, tileNValue;
    bool staticShape = matchPattern(op.getK(), m_ConstantInt(&kValue)) &&
                       matchPattern(op.getN(), m_ConstantInt(&nValue)) &&
                       matchPattern(op.getTileN(), m_ConstantInt(&tileNValue));
    int64_t kSize = staticShape ? kValue.getSExtValue() : 0;
    int64_t nSize = staticShape ? nValue.getSExtValue() : 0;
    int64_t tileN = staticShape ? tileNValue.getSExtValue() : 0;
    auto xPtrType = dyn_cast<triton::PointerType>(op.getXPtr().getType());
    auto outPtrType = dyn_cast<triton::PointerType>(op.getOutPtr().getType());
    Type xElement = xPtrType ? xPtrType.getPointeeType() : Type();
    Type outElement = outPtrType ? outPtrType.getPointeeType() : Type();
    bool f32IO = xElement.isF32() && outElement.isF32();
    bool bf16IO = xElement.isBF16() && outElement.isBF16();
    bool validShape = staticShape && kSize >= 4 && kSize % 4 == 0 &&
                      nSize >= 4 && tileN >= 4 && tileN <= 64 &&
                      tileN % 4 == 0 && nSize % tileN == 0 && (f32IO || bf16IO);
    if (!validShape)
      return rewriter.notifyMatchFailure(
          op, "whole GEMV requires static K/N, TILE_N<=64, divisibility, "
              "and matching BF16/F32 IO");

    Type bf16Ty = rewriter.getBF16Type();
    Type f32Ty = rewriter.getF32Type();
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    Type ioTy = f32IO ? f32Ty : bf16Ty;
    auto v4bf16Ty = VectorType::get({4}, bf16Ty);
    auto v4f32Ty = VectorType::get({4}, f32Ty);
    auto v4ioTy = VectorType::get({4}, ioTy);
    auto v4i8Ty = VectorType::get({4}, i8Ty);
    auto v16i8Ty = VectorType::get({16}, i8Ty);
    auto v4i32Ty = VectorType::get({4}, i32Ty);

    Value xMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({kSize}, ioTy), op.getXPtr());
    Value bMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize / tileN, kSize / 4, tileN / 4, 16}, i8Ty),
        op.getBPackedPtr());
    Value scaleMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, f32Ty), op.getWScalePtr());
    Value outMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, ioTy), op.getOutPtr());
    Value quantMem = rewriter.create<memref::AllocaOp>(
        loc, MemRefType::get({kSize}, i8Ty),
        rewriter.getIntegerAttr(rewriter.getI64Type(), 64));

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value kLimit = rewriter.create<arith::ConstantIndexOp>(loc, kSize);
    Value k4Limit = rewriter.create<arith::ConstantIndexOp>(loc, kSize / 4);
    Value tileLimit =
        rewriter.create<arith::ConstantIndexOp>(loc, nSize / tileN);
    Value tileNIndex = rewriter.create<arith::ConstantIndexOp>(loc, tileN);
    Value zeroF = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(0.0));

    // Pass one: activation absolute maximum. For finite BF16, clearing the
    // sign bit produces an unsigned magnitude encoding with the same order as
    // the represented value. Grouping four K8 slices shortens the loop-carried
    // reduction chain while preserving the exact floating-point scale. The
    // environment switch keeps a strict compiler A/B available.
    bool useBf16LaneMax =
        bf16IO && kSize >= 32 &&
        getNonNegativeEnv("TRITON_CPU_DISABLE_BF16_LANE_MAX", 0, 1) == 0;
    Value rawMax;
    if (useBf16LaneMax) {
      rawMax = emitFiniteBf16Absmax(loc, xMem, kSize, rewriter);
    } else {
      auto maxLoop =
          rewriter.create<scf::ForOp>(loc, c0, kLimit, c4, ValueRange{zeroF});
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(maxLoop.getBody());
        Value offset = maxLoop.getInductionVar();
        Value xb = rewriter.create<vector::LoadOp>(loc, v4ioTy, xMem,
                                                   ValueRange{offset});
        Value xf =
            f32IO ? xb : rewriter.create<arith::ExtFOp>(loc, v4f32Ty, xb);
        Value abs = rewriter.create<math::AbsFOp>(loc, xf);
        Value nextMax = maxLoop.getRegionIterArgs()[0];
        for (int64_t lane = 0; lane < 4; ++lane) {
          Value value = rewriter.create<vector::ExtractOp>(loc, abs, lane);
          nextMax = rewriter.create<arith::MaxNumFOp>(loc, nextMax, value);
        }
        rewriter.setInsertionPointToEnd(maxLoop.getBody());
        rewriter.create<scf::YieldOp>(loc, nextMax);
      }
      rewriter.setInsertionPointAfter(maxLoop);
      rawMax = maxLoop.getResult(0);
    }
    Value epsilon = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(1.0e-8));
    Value maxAbs = rewriter.create<arith::MaxNumFOp>(loc, rawMax, epsilon);
    Value c127 = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(127.0));
    Value invScale = rewriter.create<arith::DivFOp>(loc, c127, maxAbs);
    Value xScale = rewriter.create<arith::DivFOp>(loc, maxAbs, c127);
    Value invScaleVec =
        rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, invScale);
    Value xScaleVec =
        rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, xScale);

    // Pass two: quantize once. Every output microtile reuses this 1-4 KiB
    // stack buffer instead of repeating BF16 conversion and round-to-even.
    StringAttr roundEven = StringAttr::get(ctx, "llvm.roundeven.v4f32");
    auto quantLoop = rewriter.create<scf::ForOp>(loc, c0, kLimit, c4);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(quantLoop.getBody());
      Value offset = quantLoop.getInductionVar();
      Value xb = rewriter.create<vector::LoadOp>(loc, v4ioTy, xMem,
                                                 ValueRange{offset});
      Value xf = f32IO ? xb : rewriter.create<arith::ExtFOp>(loc, v4f32Ty, xb);
      Value scaled = rewriter.create<arith::MulFOp>(loc, xf, invScaleVec);
      Value rounded = rewriter
                          .create<LLVM::CallIntrinsicOp>(
                              loc, v4f32Ty, roundEven, ValueRange{scaled})
                          .getResult(0);
      Value quant32 = rewriter.create<arith::FPToSIOp>(loc, v4i32Ty, rounded);
      Value quant8 = rewriter.create<arith::TruncIOp>(loc, v4i8Ty, quant32);
      rewriter.create<vector::StoreOp>(loc, quant8, quantMem,
                                       ValueRange{offset});
    }
    rewriter.setInsertionPointAfter(quantLoop);

    int64_t groups = tileN / 4;
    Value zeroAcc = rewriter.create<arith::ConstantOp>(
        loc, v4i32Ty, rewriter.getZeroAttr(v4i32Ty));
    SmallVector<Value> initialAcc(groups, zeroAcc);
    Value zero16 = rewriter.create<arith::ConstantOp>(
        loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
    constexpr int64_t broadcastA[] = {0, 1, 2, 3, 0, 1, 2, 3,
                                      0, 1, 2, 3, 0, 1, 2, 3};
    StringAttr sdot =
        StringAttr::get(ctx, "llvm.aarch64.neon.sdot.v4i32.v16i8");

    // The outer loop is part of one generated function. TILE_N=64 carries
    // only 16 vector accumulators, which fits the AArch64 register file.
    auto tileLoop = rewriter.create<scf::ForOp>(loc, c0, tileLimit, c1);
    {
      OpBuilder::InsertionGuard tileGuard(rewriter);
      rewriter.setInsertionPointToStart(tileLoop.getBody());
      Value tile = tileLoop.getInductionVar();
      auto dotLoop =
          rewriter.create<scf::ForOp>(loc, c0, k4Limit, c1, initialAcc);
      {
        OpBuilder::InsertionGuard dotGuard(rewriter);
        rewriter.setInsertionPointToStart(dotLoop.getBody());
        Value kb = dotLoop.getInductionVar();
        Value xOffset = rewriter.create<arith::MulIOp>(loc, kb, c4);
        Value quant8 = rewriter.create<vector::LoadOp>(loc, v4i8Ty, quantMem,
                                                       ValueRange{xOffset});
        Value a16 = rewriter.create<vector::InsertStridedSliceOp>(
            loc, quant8, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        Value aBroadcast =
            rewriter.create<vector::ShuffleOp>(loc, a16, zero16, broadcastA);

        maybePrefetchPackedWeights(loc, ValueRange{bMem}, tile, kb, k4Limit, c0,
                                   groups, rewriter);

        SmallVector<Value> nextAcc;
        nextAcc.reserve(groups);
        ValueRange carried = dotLoop.getRegionIterArgs();
        for (int64_t group = 0; group < groups; ++group) {
          Value groupOffset =
              rewriter.create<arith::ConstantIndexOp>(loc, group);
          Value packed = rewriter.create<vector::LoadOp>(
              loc, v16i8Ty, bMem, ValueRange{tile, kb, groupOffset, c0});
          nextAcc.push_back(
              rewriter
                  .create<LLVM::CallIntrinsicOp>(
                      loc, v4i32Ty, sdot,
                      ValueRange{carried[group], aBroadcast, packed})
                  .getResult(0));
        }
        rewriter.setInsertionPointToEnd(dotLoop.getBody());
        rewriter.create<scf::YieldOp>(loc, nextAcc);
      }

      rewriter.setInsertionPointAfter(dotLoop);
      Value tileBase = rewriter.create<arith::MulIOp>(loc, tile, tileNIndex);
      for (int64_t group = 0; group < groups; ++group) {
        Value groupOffset =
            rewriter.create<arith::ConstantIndexOp>(loc, group * 4);
        Value outOffset =
            rewriter.create<arith::AddIOp>(loc, tileBase, groupOffset);
        Value accF32 = rewriter.create<arith::SIToFPOp>(
            loc, v4f32Ty, dotLoop.getResult(group));
        Value weightScale = rewriter.create<vector::LoadOp>(
            loc, v4f32Ty, scaleMem, ValueRange{outOffset});
        Value dequant = rewriter.create<arith::MulFOp>(loc, accF32, xScaleVec);
        dequant = rewriter.create<arith::MulFOp>(loc, dequant, weightScale);
        Value output =
            f32IO ? dequant
                  : rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, dequant);
        rewriter.create<vector::StoreOp>(loc, output, outMem,
                                         ValueRange{outOffset});
      }
    }

    rewriter.setInsertionPointAfter(tileLoop);
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSdotGemvPrequantOp → inline packed SDOT codegen ----------

struct SdotGemvPrequantOpLowering
    : public OpRewritePattern<triton::cpu::SdotGemvPrequantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SdotGemvPrequantOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    APInt kValue, nValue, blockNValue;
    bool staticShape =
        matchPattern(op.getK(), m_ConstantInt(&kValue)) &&
        matchPattern(op.getN(), m_ConstantInt(&nValue)) &&
        matchPattern(op.getNCount(), m_ConstantInt(&blockNValue));
    int64_t kSize = staticShape ? kValue.getSExtValue() : 0;
    int64_t nSize = staticShape ? nValue.getSExtValue() : 0;
    int64_t blockN = staticShape ? blockNValue.getSExtValue() : 0;
    auto xPtrType = dyn_cast<triton::PointerType>(op.getXQPtr().getType());
    auto xScalePtrType =
        dyn_cast<triton::PointerType>(op.getXScalePtr().getType());
    auto outPtrType = dyn_cast<triton::PointerType>(op.getOutPtr().getType());
    Type xElement = xPtrType ? xPtrType.getPointeeType() : Type();
    Type xScaleElement =
        xScalePtrType ? xScalePtrType.getPointeeType() : Type();
    Type outElement = outPtrType ? outPtrType.getPointeeType() : Type();
    bool validOutput = outElement.isF32() || outElement.isBF16();
    bool validShape =
        staticShape && kSize >= 4 && kSize % 4 == 0 && nSize >= 4 &&
        nSize % 4 == 0 && blockN >= 4 && blockN % 4 == 0 && blockN <= nSize &&
        xElement.isInteger(8) && xScaleElement.isF32() && validOutput;
    if (!validShape)
      return rewriter.notifyMatchFailure(
          op, "prequant GEMV requires static shape, i8 x, f32 scale, and "
              "BF16/F32 output");

    Type bf16Ty = rewriter.getBF16Type();
    Type f32Ty = rewriter.getF32Type();
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    auto v4bf16Ty = VectorType::get({4}, bf16Ty);
    auto v4f32Ty = VectorType::get({4}, f32Ty);
    auto v4i8Ty = VectorType::get({4}, i8Ty);
    auto v16i8Ty = VectorType::get({16}, i8Ty);
    auto v4i32Ty = VectorType::get({4}, i32Ty);

    Value xMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({kSize}, i8Ty), op.getXQPtr());
    Value xScaleMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({1}, f32Ty), op.getXScalePtr());
    Value bMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize / blockN, kSize / 4, blockN / 4, 16}, i8Ty),
        op.getBPackedPtr());
    Value scaleMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, f32Ty), op.getWScalePtr());
    Value outMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, outElement), op.getOutPtr());

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value k4Limit = rewriter.create<arith::ConstantIndexOp>(loc, kSize / 4);
    Value nStart = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), op.getNStart());
    Value blockNIndex = rewriter.create<arith::ConstantIndexOp>(loc, blockN);
    Value nBlock = rewriter.create<arith::DivUIOp>(loc, nStart, blockNIndex);

    int64_t groups = blockN / 4;
    Value zeroAcc = rewriter.create<arith::ConstantOp>(
        loc, v4i32Ty, rewriter.getZeroAttr(v4i32Ty));
    SmallVector<Value> initialAcc(groups, zeroAcc);
    Value zero16 = rewriter.create<arith::ConstantOp>(
        loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
    constexpr int64_t broadcastA[] = {0, 1, 2, 3, 0, 1, 2, 3,
                                      0, 1, 2, 3, 0, 1, 2, 3};
    StringAttr sdot =
        StringAttr::get(ctx, "llvm.aarch64.neon.sdot.v4i32.v16i8");

    auto dotLoop =
        rewriter.create<scf::ForOp>(loc, c0, k4Limit, c1, initialAcc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(dotLoop.getBody());
      Value kb = dotLoop.getInductionVar();
      Value xOffset = rewriter.create<arith::MulIOp>(loc, kb, c4);
      Value quant8 = rewriter.create<vector::LoadOp>(loc, v4i8Ty, xMem,
                                                     ValueRange{xOffset});
      Value a16 = rewriter.create<vector::InsertStridedSliceOp>(
          loc, quant8, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
      Value aBroadcast =
          rewriter.create<vector::ShuffleOp>(loc, a16, zero16, broadcastA);

      maybePrefetchPackedWeights(loc, ValueRange{bMem}, nBlock, kb, k4Limit, c0,
                                 groups, rewriter);

      SmallVector<Value> nextAcc;
      nextAcc.reserve(groups);
      ValueRange carried = dotLoop.getRegionIterArgs();
      for (int64_t group = 0; group < groups; ++group) {
        Value groupOffset = rewriter.create<arith::ConstantIndexOp>(loc, group);
        Value packed = rewriter.create<vector::LoadOp>(
            loc, v16i8Ty, bMem, ValueRange{nBlock, kb, groupOffset, c0});
        nextAcc.push_back(
            rewriter
                .create<LLVM::CallIntrinsicOp>(
                    loc, v4i32Ty, sdot,
                    ValueRange{carried[group], aBroadcast, packed})
                .getResult(0));
      }
      rewriter.setInsertionPointToEnd(dotLoop.getBody());
      rewriter.create<scf::YieldOp>(loc, nextAcc);
    }

    rewriter.setInsertionPointAfter(dotLoop);
    Value xScale =
        rewriter.create<memref::LoadOp>(loc, xScaleMem, ValueRange{c0});
    Value xScaleVec =
        rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, xScale);
    for (int64_t group = 0; group < groups; ++group) {
      Value groupOffset =
          rewriter.create<arith::ConstantIndexOp>(loc, group * 4);
      Value outOffset =
          rewriter.create<arith::AddIOp>(loc, nStart, groupOffset);
      Value accF32 = rewriter.create<arith::SIToFPOp>(loc, v4f32Ty,
                                                      dotLoop.getResult(group));
      Value weightScale = rewriter.create<vector::LoadOp>(
          loc, v4f32Ty, scaleMem, ValueRange{outOffset});
      Value dequant = rewriter.create<arith::MulFOp>(loc, accF32, xScaleVec);
      dequant = rewriter.create<arith::MulFOp>(loc, dequant, weightScale);
      Value output =
          outElement.isF32()
              ? dequant
              : rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, dequant)
                    .getResult();
      rewriter.create<vector::StoreOp>(loc, output, outMem,
                                       ValueRange{outOffset});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSdotPackWeightsOp → runtime call ----------

struct SdotPackWeightsOpLowering
    : public OpRewritePattern<triton::cpu::SdotPackWeightsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SdotPackWeightsOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "sdot_pack_weights";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(loc, funcOp,
                                  ValueRange{castPtr(op.getBPtr()),
                                             castPtr(op.getBPackedPtr()),
                                             op.getK(), op.getN()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuFusedMlpOp → spill-free gate/up SDOT microtiles ----------

struct FusedMlpMicrotileOpLowering
    : public OpRewritePattern<triton::cpu::FusedMlpOp> {
  explicit FusedMlpMicrotileOpLowering(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(triton::cpu::FusedMlpOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    APInt kValue, nValue, blockNValue;
    bool staticShape =
        matchPattern(op.getK(), m_ConstantInt(&kValue)) &&
        matchPattern(op.getN(), m_ConstantInt(&nValue)) &&
        matchPattern(op.getNCount(), m_ConstantInt(&blockNValue));
    int64_t kSize = staticShape ? kValue.getSExtValue() : 0;
    int64_t nSize = staticShape ? nValue.getSExtValue() : 0;
    int64_t blockN = staticShape ? blockNValue.getSExtValue() : 0;
    constexpr int64_t microN = 32;
    if (!staticShape || kSize < 4 || kSize % 4 != 0 || nSize < blockN ||
        nSize % blockN != 0 || blockN < microN || blockN % microN != 0) {
      return rewriter.notifyMatchFailure(
          op, "microtile fused MLP requires static divisible K/N and "
              "BLOCK_N divisible by 32");
    }

    Type bf16Ty = rewriter.getBF16Type();
    Type f32Ty = rewriter.getF32Type();
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    auto v4bf16Ty = VectorType::get({4}, bf16Ty);
    auto v4f32Ty = VectorType::get({4}, f32Ty);
    auto v4i8Ty = VectorType::get({4}, i8Ty);
    auto v16i8Ty = VectorType::get({16}, i8Ty);
    auto v4i32Ty = VectorType::get({4}, i32Ty);
    auto packedTy =
        MemRefType::get({nSize / microN, kSize / 4, microN / 4, 16}, i8Ty);

    Value xMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({kSize}, bf16Ty), op.getXPtr());
    Value gateMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, packedTy, op.getGatePackedPtr());
    Value upMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, packedTy, op.getUpPackedPtr());
    Value gateScaleMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, f32Ty), op.getGateScalePtr());
    Value upScaleMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, f32Ty), op.getUpScalePtr());
    Value outMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, bf16Ty), op.getOutPtr());
    Value quantMem = rewriter.create<memref::AllocaOp>(
        loc, MemRefType::get({kSize}, i8Ty),
        rewriter.getIntegerAttr(rewriter.getI64Type(), 64));

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value kLimit = rewriter.create<arith::ConstantIndexOp>(loc, kSize);
    Value k4Limit = rewriter.create<arith::ConstantIndexOp>(loc, kSize / 4);
    Value microLimit =
        rewriter.create<arith::ConstantIndexOp>(loc, blockN / microN);
    Value microNIndex = rewriter.create<arith::ConstantIndexOp>(loc, microN);
    Value zeroF = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(0.0));

    bool useBf16LaneMax =
        kSize >= 32 &&
        getNonNegativeEnv("TRITON_CPU_DISABLE_BF16_LANE_MAX", 0, 1) == 0;
    Value rawMax;
    if (useBf16LaneMax) {
      rawMax = emitFiniteBf16Absmax(loc, xMem, kSize, rewriter);
    } else {
      auto maxLoop =
          rewriter.create<scf::ForOp>(loc, c0, kLimit, c4, ValueRange{zeroF});
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(maxLoop.getBody());
        Value offset = maxLoop.getInductionVar();
        Value xb = rewriter.create<vector::LoadOp>(loc, v4bf16Ty, xMem,
                                                   ValueRange{offset});
        Value xf = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, xb);
        Value abs = rewriter.create<math::AbsFOp>(loc, xf);
        Value nextMax = maxLoop.getRegionIterArgs()[0];
        for (int64_t lane = 0; lane < 4; ++lane) {
          Value value = rewriter.create<vector::ExtractOp>(loc, abs, lane);
          nextMax = rewriter.create<arith::MaxNumFOp>(loc, nextMax, value);
        }
        rewriter.setInsertionPointToEnd(maxLoop.getBody());
        rewriter.create<scf::YieldOp>(loc, nextMax);
      }
      rewriter.setInsertionPointAfter(maxLoop);
      rawMax = maxLoop.getResult(0);
    }
    Value epsilon = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(1.0e-8));
    Value maxAbs = rewriter.create<arith::MaxNumFOp>(loc, rawMax, epsilon);
    Value c127 = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(127.0));
    Value invScale = rewriter.create<arith::DivFOp>(loc, c127, maxAbs);
    Value xScale = rewriter.create<arith::DivFOp>(loc, maxAbs, c127);
    Value invScaleVec =
        rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, invScale);
    Value xScaleVec =
        rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, xScale);

    StringAttr roundEven = StringAttr::get(ctx, "llvm.roundeven.v4f32");
    auto quantLoop = rewriter.create<scf::ForOp>(loc, c0, kLimit, c4);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(quantLoop.getBody());
      Value offset = quantLoop.getInductionVar();
      Value xb = rewriter.create<vector::LoadOp>(loc, v4bf16Ty, xMem,
                                                 ValueRange{offset});
      Value xf = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, xb);
      Value scaled = rewriter.create<arith::MulFOp>(loc, xf, invScaleVec);
      Value rounded = rewriter
                          .create<LLVM::CallIntrinsicOp>(
                              loc, v4f32Ty, roundEven, ValueRange{scaled})
                          .getResult(0);
      Value quant32 = rewriter.create<arith::FPToSIOp>(loc, v4i32Ty, rounded);
      Value quant8 = rewriter.create<arith::TruncIOp>(loc, v4i8Ty, quant32);
      rewriter.create<vector::StoreOp>(loc, quant8, quantMem,
                                       ValueRange{offset});
    }
    rewriter.setInsertionPointAfter(quantLoop);

    Value nStart = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), op.getNStart());
    Value firstMicro =
        rewriter.create<arith::DivUIOp>(loc, nStart, microNIndex);
    constexpr int64_t groups = microN / 4;
    Value zeroAcc = rewriter.create<arith::ConstantOp>(
        loc, v4i32Ty, rewriter.getZeroAttr(v4i32Ty));
    SmallVector<Value> initialAcc(groups * 2, zeroAcc);
    Value zero16 = rewriter.create<arith::ConstantOp>(
        loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
    constexpr int64_t broadcastA[] = {0, 1, 2, 3, 0, 1, 2, 3,
                                      0, 1, 2, 3, 0, 1, 2, 3};
    StringAttr sdot =
        StringAttr::get(ctx, "llvm.aarch64.neon.sdot.v4i32.v16i8");
    Value oneF = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(1.0));
    Value oneVec = rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, oneF);

    auto tileLoop = rewriter.create<scf::ForOp>(loc, c0, microLimit, c1);
    {
      OpBuilder::InsertionGuard tileGuard(rewriter);
      rewriter.setInsertionPointToStart(tileLoop.getBody());
      Value tile = tileLoop.getInductionVar();
      Value packedTile = rewriter.create<arith::AddIOp>(loc, firstMicro, tile);
      auto dotLoop =
          rewriter.create<scf::ForOp>(loc, c0, k4Limit, c1, initialAcc);
      {
        OpBuilder::InsertionGuard dotGuard(rewriter);
        rewriter.setInsertionPointToStart(dotLoop.getBody());
        Value kb = dotLoop.getInductionVar();
        Value xOffset = rewriter.create<arith::MulIOp>(loc, kb, c4);
        Value quant8 = rewriter.create<vector::LoadOp>(loc, v4i8Ty, quantMem,
                                                       ValueRange{xOffset});
        Value a16 = rewriter.create<vector::InsertStridedSliceOp>(
            loc, quant8, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        Value aBroadcast =
            rewriter.create<vector::ShuffleOp>(loc, a16, zero16, broadcastA);

        ValueRange carried = dotLoop.getRegionIterArgs();
        SmallVector<Value> gateNext;
        SmallVector<Value> upNext;
        gateNext.reserve(groups);
        upNext.reserve(groups);
        for (int64_t group = 0; group < groups; ++group) {
          Value groupOffset =
              rewriter.create<arith::ConstantIndexOp>(loc, group);
          Value gatePacked = rewriter.create<vector::LoadOp>(
              loc, v16i8Ty, gateMem,
              ValueRange{packedTile, kb, groupOffset, c0});
          Value upPacked = rewriter.create<vector::LoadOp>(
              loc, v16i8Ty, upMem, ValueRange{packedTile, kb, groupOffset, c0});
          gateNext.push_back(
              rewriter
                  .create<LLVM::CallIntrinsicOp>(
                      loc, v4i32Ty, sdot,
                      ValueRange{carried[group], aBroadcast, gatePacked})
                  .getResult(0));
          upNext.push_back(
              rewriter
                  .create<LLVM::CallIntrinsicOp>(
                      loc, v4i32Ty, sdot,
                      ValueRange{carried[groups + group], aBroadcast, upPacked})
                  .getResult(0));
        }
        gateNext.append(upNext);
        rewriter.setInsertionPointToEnd(dotLoop.getBody());
        rewriter.create<scf::YieldOp>(loc, gateNext);
      }

      rewriter.setInsertionPointAfter(dotLoop);
      Value tileOffset = rewriter.create<arith::MulIOp>(loc, tile, microNIndex);
      Value tileStart = rewriter.create<arith::AddIOp>(loc, nStart, tileOffset);
      for (int64_t group = 0; group < groups; ++group) {
        Value groupOffset =
            rewriter.create<arith::ConstantIndexOp>(loc, group * 4);
        Value outOffset =
            rewriter.create<arith::AddIOp>(loc, tileStart, groupOffset);
        Value gateF = rewriter.create<arith::SIToFPOp>(
            loc, v4f32Ty, dotLoop.getResult(group));
        Value upF = rewriter.create<arith::SIToFPOp>(
            loc, v4f32Ty, dotLoop.getResult(groups + group));
        Value gateScale = rewriter.create<vector::LoadOp>(
            loc, v4f32Ty, gateScaleMem, ValueRange{outOffset});
        Value upScale = rewriter.create<vector::LoadOp>(
            loc, v4f32Ty, upScaleMem, ValueRange{outOffset});
        gateF = rewriter.create<arith::MulFOp>(loc, gateF, xScaleVec);
        gateF = rewriter.create<arith::MulFOp>(loc, gateF, gateScale);
        upF = rewriter.create<arith::MulFOp>(loc, upF, xScaleVec);
        upF = rewriter.create<arith::MulFOp>(loc, upF, upScale);

        Value gateBf16 = rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, gateF);
        gateF = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, gateBf16);
        Value upBf16 = rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, upF);
        upF = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, upBf16);
        Value negGate = rewriter.create<arith::NegFOp>(loc, gateF);
        Value expNeg;
        if (getNonNegativeEnv("TRITON_CPU_INLINE_EXP_U10", 1, 1))
          expNeg = emitInlineSleefExpfU10(loc, negGate, roundEven, rewriter);
        else
          expNeg = rewriter.create<math::ExpOp>(loc, negGate);
        Value denominator = rewriter.create<arith::AddFOp>(loc, oneVec, expNeg);
        Value silu = rewriter.create<arith::DivFOp>(loc, gateF, denominator);
        Value siluBf16 = rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, silu);
        silu = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, siluBf16);
        Value result = rewriter.create<arith::MulFOp>(loc, silu, upF);
        Value outBf16 = rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, result);
        rewriter.create<vector::StoreOp>(loc, outBf16, outMem,
                                         ValueRange{outOffset});
      }
    }

    rewriter.setInsertionPointAfter(tileLoop);
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuFusedMlpOp → inline gate/up packed SDOT + SwiGLU ----------

struct FusedMlpOpLowering : public OpRewritePattern<triton::cpu::FusedMlpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::FusedMlpOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    APInt kValue, nValue, blockNValue;
    bool staticShape =
        matchPattern(op.getK(), m_ConstantInt(&kValue)) &&
        matchPattern(op.getN(), m_ConstantInt(&nValue)) &&
        matchPattern(op.getNCount(), m_ConstantInt(&blockNValue));
    int64_t kSize = staticShape ? kValue.getSExtValue() : 0;
    int64_t nSize = staticShape ? nValue.getSExtValue() : 0;
    int64_t blockN = staticShape ? blockNValue.getSExtValue() : 0;
    if (!staticShape || kSize < 4 || kSize % 4 != 0 || nSize < 4 ||
        nSize % 4 != 0 || blockN < 4 || blockN % 4 != 0 || blockN > nSize) {
      return rewriter.notifyMatchFailure(
          op, "inline fused MLP requires static K/N/N_count multiples of 4");
    }

    Type bf16Ty = rewriter.getBF16Type();
    Type f32Ty = rewriter.getF32Type();
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    auto v4bf16Ty = VectorType::get({4}, bf16Ty);
    auto v4f32Ty = VectorType::get({4}, f32Ty);
    auto v4i8Ty = VectorType::get({4}, i8Ty);
    auto v16i8Ty = VectorType::get({16}, i8Ty);
    auto v4i32Ty = VectorType::get({4}, i32Ty);
    auto packedTy =
        MemRefType::get({nSize / blockN, kSize / 4, blockN / 4, 16}, i8Ty);

    Value xMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({kSize}, bf16Ty), op.getXPtr());
    Value gateMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, packedTy, op.getGatePackedPtr());
    Value upMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, packedTy, op.getUpPackedPtr());
    Value gateScaleMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, f32Ty), op.getGateScalePtr());
    Value upScaleMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, f32Ty), op.getUpScalePtr());
    Value outMem = rewriter.create<triton::cpu::PtrToMemRefOp>(
        loc, MemRefType::get({nSize}, bf16Ty), op.getOutPtr());

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value kLimit = rewriter.create<arith::ConstantIndexOp>(loc, kSize);
    Value k4Limit = rewriter.create<arith::ConstantIndexOp>(loc, kSize / 4);
    Value zeroF = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(0.0));

    auto maxLoop =
        rewriter.create<scf::ForOp>(loc, c0, kLimit, c4, ValueRange{zeroF});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(maxLoop.getBody());
      Value offset = maxLoop.getInductionVar();
      Value xb = rewriter.create<vector::LoadOp>(loc, v4bf16Ty, xMem,
                                                 ValueRange{offset});
      Value xf = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, xb);
      Value abs = rewriter.create<math::AbsFOp>(loc, xf);
      Value nextMax = maxLoop.getRegionIterArgs()[0];
      for (int64_t lane = 0; lane < 4; ++lane) {
        Value value = rewriter.create<vector::ExtractOp>(loc, abs, lane);
        nextMax = rewriter.create<arith::MaxNumFOp>(loc, nextMax, value);
      }
      rewriter.setInsertionPointToEnd(maxLoop.getBody());
      rewriter.create<scf::YieldOp>(loc, nextMax);
    }
    rewriter.setInsertionPointAfter(maxLoop);
    Value epsilon = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(1.0e-8));
    Value maxAbs =
        rewriter.create<arith::MaxNumFOp>(loc, maxLoop.getResult(0), epsilon);
    Value c127 = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(127.0));
    Value invScale = rewriter.create<arith::DivFOp>(loc, c127, maxAbs);
    Value xScale = rewriter.create<arith::DivFOp>(loc, maxAbs, c127);
    Value invScaleVec =
        rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, invScale);
    Value xScaleVec =
        rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, xScale);

    Value nStart = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), op.getNStart());
    Value blockNIndex = rewriter.create<arith::ConstantIndexOp>(loc, blockN);
    Value nBlock = rewriter.create<arith::DivUIOp>(loc, nStart, blockNIndex);

    int64_t groups = blockN / 4;
    Value zeroAcc = rewriter.create<arith::ConstantOp>(
        loc, v4i32Ty, rewriter.getZeroAttr(v4i32Ty));
    SmallVector<Value> initialAcc(groups * 2, zeroAcc);
    Value zero16 = rewriter.create<arith::ConstantOp>(
        loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
    constexpr int64_t broadcastA[] = {0, 1, 2, 3, 0, 1, 2, 3,
                                      0, 1, 2, 3, 0, 1, 2, 3};
    StringAttr roundEven = StringAttr::get(ctx, "llvm.roundeven.v4f32");
    StringAttr sdot =
        StringAttr::get(ctx, "llvm.aarch64.neon.sdot.v4i32.v16i8");

    auto dotLoop =
        rewriter.create<scf::ForOp>(loc, c0, k4Limit, c1, initialAcc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(dotLoop.getBody());
      Value kb = dotLoop.getInductionVar();
      Value xOffset = rewriter.create<arith::MulIOp>(loc, kb, c4);
      Value xb = rewriter.create<vector::LoadOp>(loc, v4bf16Ty, xMem,
                                                 ValueRange{xOffset});
      Value xf = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, xb);
      Value scaled = rewriter.create<arith::MulFOp>(loc, xf, invScaleVec);
      Value rounded = rewriter
                          .create<LLVM::CallIntrinsicOp>(
                              loc, v4f32Ty, roundEven, ValueRange{scaled})
                          .getResult(0);
      Value quant32 = rewriter.create<arith::FPToSIOp>(loc, v4i32Ty, rounded);
      Value quant8 = rewriter.create<arith::TruncIOp>(loc, v4i8Ty, quant32);
      Value a16 = rewriter.create<vector::InsertStridedSliceOp>(
          loc, quant8, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
      Value aBroadcast =
          rewriter.create<vector::ShuffleOp>(loc, a16, zero16, broadcastA);

      maybePrefetchPackedWeights(loc, ValueRange{gateMem, upMem}, nBlock, kb,
                                 k4Limit, c0, groups, rewriter);

      ValueRange carried = dotLoop.getRegionIterArgs();
      SmallVector<Value> gateNext;
      SmallVector<Value> upNext;
      gateNext.reserve(groups);
      upNext.reserve(groups);
      for (int64_t group = 0; group < groups; ++group) {
        Value groupOffset = rewriter.create<arith::ConstantIndexOp>(loc, group);
        Value gatePacked = rewriter.create<vector::LoadOp>(
            loc, v16i8Ty, gateMem, ValueRange{nBlock, kb, groupOffset, c0});
        Value upPacked = rewriter.create<vector::LoadOp>(
            loc, v16i8Ty, upMem, ValueRange{nBlock, kb, groupOffset, c0});
        gateNext.push_back(
            rewriter
                .create<LLVM::CallIntrinsicOp>(
                    loc, v4i32Ty, sdot,
                    ValueRange{carried[group], aBroadcast, gatePacked})
                .getResult(0));
        upNext.push_back(
            rewriter
                .create<LLVM::CallIntrinsicOp>(
                    loc, v4i32Ty, sdot,
                    ValueRange{carried[groups + group], aBroadcast, upPacked})
                .getResult(0));
      }
      gateNext.append(upNext);
      rewriter.setInsertionPointToEnd(dotLoop.getBody());
      rewriter.create<scf::YieldOp>(loc, gateNext);
    }

    rewriter.setInsertionPointAfter(dotLoop);
    Value oneF = rewriter.create<arith::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(1.0));
    Value oneVec = rewriter.create<vector::BroadcastOp>(loc, v4f32Ty, oneF);
    for (int64_t group = 0; group < groups; ++group) {
      Value groupOffset =
          rewriter.create<arith::ConstantIndexOp>(loc, group * 4);
      Value outOffset =
          rewriter.create<arith::AddIOp>(loc, nStart, groupOffset);
      Value gateF = rewriter.create<arith::SIToFPOp>(loc, v4f32Ty,
                                                     dotLoop.getResult(group));
      Value upF = rewriter.create<arith::SIToFPOp>(
          loc, v4f32Ty, dotLoop.getResult(groups + group));
      Value gateScale = rewriter.create<vector::LoadOp>(
          loc, v4f32Ty, gateScaleMem, ValueRange{outOffset});
      Value upScale = rewriter.create<vector::LoadOp>(loc, v4f32Ty, upScaleMem,
                                                      ValueRange{outOffset});
      gateF = rewriter.create<arith::MulFOp>(loc, gateF, xScaleVec);
      gateF = rewriter.create<arith::MulFOp>(loc, gateF, gateScale);
      upF = rewriter.create<arith::MulFOp>(loc, upF, xScaleVec);
      upF = rewriter.create<arith::MulFOp>(loc, upF, upScale);

      // Preserve the observable BF16 boundaries of gate_proj/up_proj and
      // silu before the final multiply.
      Value gateBf16 = rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, gateF);
      gateF = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, gateBf16);
      Value upBf16 = rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, upF);
      upF = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, upBf16);
      Value negGate = rewriter.create<arith::NegFOp>(loc, gateF);
      Value expNeg = rewriter.create<math::ExpOp>(loc, negGate);
      Value denominator = rewriter.create<arith::AddFOp>(loc, oneVec, expNeg);
      Value silu = rewriter.create<arith::DivFOp>(loc, gateF, denominator);
      Value siluBf16 = rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, silu);
      silu = rewriter.create<arith::ExtFOp>(loc, v4f32Ty, siluBf16);
      Value result = rewriter.create<arith::MulFOp>(loc, silu, upF);
      Value outBf16 = rewriter.create<arith::TruncFOp>(loc, v4bf16Ty, result);
      rewriter.create<vector::StoreOp>(loc, outBf16, outMem,
                                       ValueRange{outOffset});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSmeGemmOp → runtime call (ARM SME int8 GEMM) ----------

struct SmeGemmOpLowering : public OpRewritePattern<triton::cpu::SmeGemmOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SmeGemmOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "sme_gemm_int32";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getApPtr()), castPtr(op.getBpPtr()),
                   castPtr(op.getCPtr()), op.getMp(), op.getNp(), op.getK4()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSmmlaUkOp → runtime call (TLE-Struct micro-kernel) ----------

struct SmmlaUkOpLowering : public OpRewritePattern<triton::cpu::SmmlaUkOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SmmlaUkOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "smmla_uk";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType =
          LLVM::LLVMFunctionType::get(voidTy,
                                      {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty,
                                       i64Ty, i64Ty, i64Ty, i64Ty},
                                      false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getApPtr()), castPtr(op.getWpPtr()),
                   castPtr(op.getCPtr()), castPtr(op.getXsPtr()),
                   castPtr(op.getWsPtr()), op.getK8(), op.getMP(), op.getN(),
                   op.getMp0(), op.getNp0()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- More TLE-Struct micro-kernel ops → runtime calls ----------
namespace {
LLVM::LLVMFuncOp getOrDeclare(PatternRewriter &rewriter, ModuleOp module,
                              StringRef name, ArrayRef<Type> argTys) {
  auto ctx = rewriter.getContext();
  auto f = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (f)
    return f;
  auto ft =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argTys, false);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), name, ft);
}
} // namespace

#define UK_LOWERING(OpName, RtName)                                            \
  struct OpName##Lowering : public OpRewritePattern<triton::cpu::OpName> {     \
    using OpRewritePattern::OpRewritePattern;                                  \
    LogicalResult matchAndRewrite(triton::cpu::OpName op,                      \
                                  PatternRewriter &rewriter) const override {  \
      auto loc = op.getLoc();                                                  \
      auto ctx = rewriter.getContext();                                        \
      auto module = op->getParentOfType<ModuleOp>();                           \
      auto i64Ty = IntegerType::get(ctx, 64);                                  \
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);                            \
      SmallVector<Type> tys;                                                   \
      for (auto v : op->getOperands())                                         \
        tys.push_back(isa<IntegerType>(v.getType()) ? (Type)i64Ty              \
                                                    : (Type)ptrTy);            \
      auto fn = getOrDeclare(rewriter, module, RtName, tys);                   \
      auto castPtr = [&](Value v) -> Value {                                   \
        if (isa<LLVM::LLVMPointerType>(v.getType()))                           \
          return v;                                                            \
        if (isa<IntegerType>(v.getType()))                                     \
          return v;                                                            \
        return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)      \
            .getResult(0);                                                     \
      };                                                                       \
      SmallVector<Value> args;                                                 \
      for (auto v : op->getOperands())                                         \
        args.push_back(castPtr(v));                                            \
      rewriter.create<LLVM::CallOp>(loc, fn, args);                            \
      rewriter.eraseOp(op);                                                    \
      return success();                                                        \
    }                                                                          \
  };
UK_LOWERING(SmeUkOp, "sme_uk")
UK_LOWERING(SdotGemvUkOp, "sdot_gemv_uk")
UK_LOWERING(SwigluUkOp, "swiglu_uk")
UK_LOWERING(RmsnormUkOp, "rmsnorm_uk")
UK_LOWERING(ResidualUkOp, "residual_uk")

// ---------- CpuFlashAttnDecodeOp → runtime call ----------

struct FlashAttnDecodeOpLowering
    : public OpRewritePattern<triton::cpu::FlashAttnDecodeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::FlashAttnDecodeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "flash_attn_decode_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType =
          LLVM::LLVMFunctionType::get(voidTy,
                                      {ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty,
                                       f32Ty, i64Ty, i64Ty, i64Ty, i64Ty},
                                      false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    auto smScaleVal =
        rewriter.create<LLVM::ConstantOp>(loc, f32Ty, op.getSmScaleAttr());

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getQPtr()), castPtr(op.getKPtr()),
                   castPtr(op.getVPtr()), castPtr(op.getOutPtr()),
                   op.getSeqLen(), op.getHeadDim(), smScaleVal,
                   op.getNumHeads(), op.getNumKvHeads(), op.getStrideKn(),
                   op.getStrideVn()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSmeAttnPrefillOp → runtime call ----------

struct SmeAttnPrefillOpLowering
    : public OpRewritePattern<triton::cpu::SmeAttnPrefillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SmeAttnPrefillOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "sme_attn_prefill_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      // (q,k,v,out, M, Nkv, head_dim, sm_scale, num_heads, num_kv_heads,
      // causal)
      auto funcType =
          LLVM::LLVMFunctionType::get(voidTy,
                                      {ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty,
                                       i64Ty, f32Ty, i64Ty, i64Ty, i64Ty},
                                      false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    auto smScaleVal =
        rewriter.create<LLVM::ConstantOp>(loc, f32Ty, op.getSmScaleAttr());

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getQPtr()), castPtr(op.getKPtr()),
                   castPtr(op.getVPtr()), castPtr(op.getOutPtr()), op.getM(),
                   op.getNKv(), op.getHeadDim(), smScaleVal, op.getNumHeads(),
                   op.getNumKvHeads(), op.getCausal()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuRmsNormOp → runtime call ----------

struct RmsNormOpLowering : public OpRewritePattern<triton::cpu::RmsNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::RmsNormOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "standalone_rms_norm_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty, f32Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    // Extract eps from F32Attr
    auto epsVal =
        rewriter.create<LLVM::ConstantOp>(loc, f32Ty, op.getEpsAttr());

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getXPtr()), castPtr(op.getWeightPtr()),
                   castPtr(op.getOutPtr()), op.getD(), epsVal});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSwigluOp → runtime call ----------

struct SwigluOpLowering : public OpRewritePattern<triton::cpu::SwigluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SwigluOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "swiglu_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                 funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getGatePtr()), castPtr(op.getUpPtr()),
                   castPtr(op.getOutPtr()), op.getN()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- Pass ----------

namespace mlir::triton::cpu {

std::unique_ptr<Pass> createNeonSdotToLLVMPass() {
  struct NeonSdotToLLVMPass : public OperationPass<ModuleOp> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NeonSdotToLLVMPass)
    NeonSdotToLLVMPass()
        : OperationPass<ModuleOp>(TypeID::get<NeonSdotToLLVMPass>()) {}
    StringRef getName() const override { return "NeonSdotToLLVM"; }
    std::unique_ptr<Pass> clonePass() const override {
      return std::make_unique<NeonSdotToLLVMPass>();
    }
    StringRef getArgument() const override {
      return "convert-neon-sdot-to-llvm";
    }
    void runOnOperation() override {
      auto *ctx = &getContext();
      RewritePatternSet patterns(ctx);
      patterns.add<NeonSdotOpLowering>(ctx);
      patterns.add<SdotGemvOpLowering>(ctx);
      patterns.add<SdotGemvFusedBf16OpLowering>(ctx);
      patterns.add<SdotGemvWholeOpLowering>(ctx);
      patterns.add<SdotGemvPrequantOpLowering>(ctx);
      patterns.add<SdotPackWeightsOpLowering>(ctx);
      patterns.add<RmsNormOpLowering>(ctx);
      patterns.add<SwigluOpLowering>(ctx);
      patterns.add<FlashAttnDecodeOpLowering>(ctx);
      patterns.add<SmeAttnPrefillOpLowering>(ctx);
      patterns.add<FusedMlpMicrotileOpLowering>(ctx);
      patterns.add<FusedMlpOpLowering>(ctx);
      patterns.add<FusedTransformerLayerOpLowering>(ctx);
      patterns.add<FusedDecodeStepOpLowering>(ctx);
      patterns.add<SmeGemmOpLowering>(ctx);
      patterns.add<SmmlaUkOpLowering>(ctx);
      patterns.add<SmeUkOpLowering>(ctx);
      patterns.add<SdotGemvUkOpLowering>(ctx);
      patterns.add<SwigluUkOpLowering>(ctx);
      patterns.add<RmsnormUkOpLowering>(ctx);
      patterns.add<ResidualUkOpLowering>(ctx);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
    }
  };
  return std::make_unique<NeonSdotToLLVMPass>();
}

} // namespace mlir::triton::cpu
