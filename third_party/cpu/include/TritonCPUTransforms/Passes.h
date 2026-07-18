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

#ifndef TritonCPUTransforms_CONVERSION_PASSES_H
#define TritonCPUTransforms_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace cpu {

enum class Ukernels {
  OneDNN,
  XSMM,
};

#define GEN_PASS_DECL
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertUnsupportedOps();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertUnsupportedOps(bool promoteBf16ToFp32,
                            bool convertMixedPrecisionMatmul,
                            bool promoteLibMathToFp32);
std::unique_ptr<OperationPass<ModuleOp>> createDecomposeFpConversions();
std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeFpConversions(bool decomposeBf16Conversions,
                             bool decomposeFp8Conversions);
std::unique_ptr<OperationPass<ModuleOp>> createOptimizeMasks();

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotProduct();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotProduct(bool useHorizontalSum);

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToAMX();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToAMX(bool convertInt8, bool convertFp16, bool convertBf16);
std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToSVE2I8MM();
std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToFMA();
std::unique_ptr<OperationPass<ModuleOp>> createConvertDotGeneric();
std::unique_ptr<OperationPass<ModuleOp>> createCanonicalize();

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotOpToUkernelOps(
    Ukernels ukernels = mlir::triton::cpu::Ukernels::OneDNN);

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"

} // namespace cpu
} // namespace triton

} // namespace mlir

#endif
