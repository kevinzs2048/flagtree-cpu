// Lower int8 tt.DotOp to NEON i8mm (SMMLA) on ARM cores that have NEON i8mm but
// NOT SVE2 (e.g. Apple M-series). Analogous to ConvertDotToSVE2I8MM but emits the
// fixed-width NEON intrinsic llvm.aarch64.neon.smmla.v4i32.v16i8 instead of the
// scalable SVE smmla -> no nxv wrapping needed (simpler). Gated on "i8mm" in
// cpu_features, inserted before ConvertDotGeneric (the scalar fallback).
//
// STATUS: stub (plumbing/build de-risk first). runOnOperation is a no-op so dots
// fall through to ConvertDotGeneric unchanged. The lowering algorithm (ported from
// the SVE2 pass, fixed-width) is filled in next phase.
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

struct ConvertDotToNeonI8MM
    : public triton::cpu::impl::ConvertDotToNeonI8MMBase<ConvertDotToNeonI8MM> {
  ConvertDotToNeonI8MM() = default;

  void runOnOperation() override {
    // STUB: no-op. Dots fall through to ConvertDotGeneric. Algorithm next phase.
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
