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

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

std::unique_ptr<Pass>
createTritonAMDGPUStreamPipelinePass(int numStages = 2, int globalPrefetch = 0,
                                     int localPrefetch = 0,
                                     bool useAsyncCopy = false);

std::unique_ptr<Pass>
createTritonAMDGPUAccelerateMatmulPass(std::string archGenName = std::string(),
                                       int matrixInstructionSize = 0,
                                       int kpack = 1);

std::unique_ptr<Pass> createTritonAMDGPUCanonicalizeLoopsPass();

std::unique_ptr<Pass> createTritonAMDGPUReorderInstructionsPass();

std::unique_ptr<Pass> createTritonAMDGPUVerifier();

std::unique_ptr<Pass> createTritonAMDGPUOptimizeEpiloguePass();

std::unique_ptr<Pass> createTritonAMDGPUHoistLayoutConversionsPass();

std::unique_ptr<Pass> createTritonAMDGPUCanonicalizePointersPass();

std::unique_ptr<Pass> createTritonAMDGPUConvertToBufferOpsPass(
    std::string archGenName = std::string());

std::unique_ptr<Pass>
createTritonAMDGPUBlockPingpongPass(int32_t numStages = 2);

std::unique_ptr<Pass> createTritonAMDGPUInThreadTransposePass();

std::unique_ptr<Pass>
createTritonAMDGPUCoalesceAsyncCopyPass(std::string archGenName = {});

std::unique_ptr<Pass> createTritonAMDGPUFoldTrueCmpIPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUTransforms/Passes.h.inc"

} // namespace mlir
#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_
