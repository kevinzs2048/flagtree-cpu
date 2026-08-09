#pragma once
#ifdef TRITON_BUILD_AMD_BACKEND
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"
#endif
#ifdef TRITON_BUILD_NVIDIA_BACKEND
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/include/Dialect/NVWS/IR/Dialect.h"
#endif
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#ifdef TRITON_BUILD_NVIDIA_BACKEND
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#endif

// Below headers will allow registration to ROCm passes
#ifdef TRITON_BUILD_AMD_BACKEND
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"
#endif

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#ifdef TRITON_BUILD_NVIDIA_BACKEND
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#endif

#include "cpu/include/ScalarizePass/ScalarizeInterfaceImpl.h"
#include "cpu/include/TritonCPUToLLVM/Passes.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "cpu/include/TritonToTritonCPU/Passes.h"
#ifdef TRITON_BUILD_NVIDIA_BACKEND
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#endif
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/AMX/AMXDialect.h"
#ifdef TRITON_BUILD_NVIDIA_BACKEND
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#endif
#ifdef TRITON_BUILD_AMD_BACKEND
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#endif
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
#ifdef TRITON_BUILD_NVIDIA_BACKEND
void registerTestMembarPass();
#endif
#ifdef TRITON_BUILD_AMD_BACKEND
void registerTestTritonAMDGPURangeAnalysis();
#endif
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
#ifdef TRITON_BUILD_NVIDIA_BACKEND
  mlir::registerTritonNvidiaGPUPasses();
#endif
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
#ifdef TRITON_BUILD_NVIDIA_BACKEND
  mlir::test::registerTestMembarPass();
#endif
#ifdef TRITON_BUILD_AMD_BACKEND
  mlir::test::registerTestTritonAMDGPURangeAnalysis();
#endif
  mlir::triton::registerConvertTritonToTritonGPUPass();
#ifdef TRITON_BUILD_NVIDIA_BACKEND
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::triton::registerConvertWarpSpecializeToLLVM();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
#endif
  mlir::registerLLVMDIScope();

#ifdef TRITON_BUILD_AMD_BACKEND
  // TritonAMDGPUToLLVM passes
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerOptimizeAMDLDSUsage();

  // TritonAMDGPUTransforms passes
  mlir::registerTritonAMDGPUAccelerateMatmul();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  mlir::registerTritonAMDGPUHoistLayoutConversions();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUBlockPingpong();
  mlir::registerTritonAMDGPUStreamPipeline();
  mlir::registerTritonAMDGPUCanonicalizePointers();
  mlir::registerTritonAMDGPUConvertToBufferOps();
  mlir::registerTritonAMDGPUInThreadTranspose();
  mlir::registerTritonAMDGPUCoalesceAsyncCopy();
  mlir::triton::registerTritonAMDGPUInsertInstructionSchedHints();
  mlir::triton::registerTritonAMDGPULowerInstructionSchedHints();
  mlir::registerTritonAMDFoldTrueCmpI();
#endif

#ifdef TRITON_BUILD_NVIDIA_BACKEND
  // NVWS passes
  mlir::registerNVWSTransformsPasses();
#endif

  // CPU passes
  mlir::triton::cpu::registerTritonToTritonCPUPasses();
  mlir::triton::cpu::registerTritonCPUTransformsPasses();
  mlir::triton::cpu::registerTritonCPUToLLVMPasses();
  mlir::triton::cpu::registerTritonOpScalarizeExternalModels(registry);

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::cpu::TritonCPUDialect, mlir::triton::gpu::TritonGPUDialect,
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
      mlir::amx::AMXDialect, mlir::tensor::TensorDialect, mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect, mlir::triton::proton::ProtonDialect>();
#ifdef TRITON_BUILD_NVIDIA_BACKEND
  registry.insert<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                  mlir::NVVM::NVVMDialect, mlir::triton::nvgpu::NVGPUDialect,
                  mlir::triton::nvws::NVWSDialect>();
#endif
#ifdef TRITON_BUILD_AMD_BACKEND
  registry.insert<mlir::triton::amdgpu::TritonAMDGPUDialect,
                  mlir::ROCDL::ROCDLDialect>();
#endif
}
