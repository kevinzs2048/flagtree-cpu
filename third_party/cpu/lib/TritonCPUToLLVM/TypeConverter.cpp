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

#include "TypeConverter.h"

#include "mlir/Dialect/AMX/AMXDialect.h"

using namespace mlir;
using namespace mlir::triton;

TritonCPUToLLVMTypeConverter::TritonCPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([this](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  addConversion([&](amx::TileType type) {
    return LLVM::LLVMX86AMXType::get(type.getContext());
  });
}

Type TritonCPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  if (isa<RankedTensorType>(pointeeType)) {
    // struct {
    //   ptr base_ptr;
    //   array<rank x i64> offsets;
    //   array<rank x i64> shape;
    //   array<rank x i64> strides;
    // }
    auto tensorTy = cast<RankedTensorType>(pointeeType);
    auto rank = tensorTy.getShape().size();
    auto i64Ty = IntegerType::get(ctx, 64);
    SmallVector<Type, 4> types;
    types.push_back(LLVM::LLVMPointerType::get(ctx));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx);
}

Type TritonCPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  if (isa<PointerType>(type.getElementType()))
    return VectorType::get(type.getShape(),
                           IntegerType::get(type.getContext(), 64));
  llvm_unreachable("No tensor types are expected in TTCIR");
}
