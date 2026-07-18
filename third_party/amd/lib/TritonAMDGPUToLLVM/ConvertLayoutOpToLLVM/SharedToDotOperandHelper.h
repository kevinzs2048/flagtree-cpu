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

#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_CONVERTLAYOUTOPTOLLVM_SHAREDTODOTOPERANDHELPER_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_CONVERTLAYOUTOPTOLLVM_SHAREDTODOTOPERANDHELPER_H_

#include "Utility.h"

namespace mlir::triton::AMD {

// Get warpId inside block of warps.
Value getWarpIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value warpId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx,
                       const ArrayRef<unsigned int> &order);

bool isSwizzled(gpu::SwizzledSharedEncodingAttr layout);

/// Swizzling tensor element indexes according pattern encoded in
/// SwizzledSharedEncodingAttr
///
/// \param rewriter
/// \param loc
/// \param row row of target tensor element related to the start of smemObj
/// \param col col of target tensor element related to the start of smemObj
/// \param smemObj shared memory object, contains info about tensor in LDS
/// \param attr layout attribute, contains swizzling info
/// \returns swizzled row, col indexes in tensor notation
std::pair<mlir::Value, mlir::Value>
swizzleIndexes(ConversionPatternRewriter &rewriter, Location loc, Value row,
               Value col, SharedMemoryObject smemObj,
               gpu::SwizzledSharedEncodingAttr attr);

Value computeOffset(ConversionPatternRewriter &rewriter, Location loc,
                    Value row, Value col, SharedMemoryObject smemObj,
                    ArrayRef<Value> strides,
                    gpu::SwizzledSharedEncodingAttr srcLayout);

Value computeBasePtr(ConversionPatternRewriter &rewriter, Location loc,
                     const SharedMemoryObject &smemObj,
                     ArrayRef<Value> strides);

bool isKContig(llvm::ArrayRef<unsigned> order, int opIdx);

using computeTensorElemMappingInBlockT =
    std::function<llvm::SmallVector<llvm::SmallVector<Value>>(
        ConversionPatternRewriter &, Location, const ArrayRef<int64_t> &, Value,
        Value, int, ArrayRef<int64_t>, ArrayRef<Value>, int, unsigned,
        unsigned)>;

llvm::SmallVector<Value> computeOffsetsAType(
    ConversionPatternRewriter &rewriter, Location loc,
    computeTensorElemMappingInBlockT fn, const ArrayRef<int64_t> &elemsPerInstr,
    Value warpId, Value laneId, int warpsPerBlock, int numOfElems,
    ArrayRef<int64_t> reps, SharedMemoryObject smemObj, ArrayRef<Value> strides,
    gpu::SwizzledSharedEncodingAttr srcLayout, unsigned nonKDim, unsigned kDim);

llvm::SmallVector<Value> computeOffsetsBType(
    ConversionPatternRewriter &rewriter, Location loc,
    computeTensorElemMappingInBlockT fn, const ArrayRef<int64_t> &elemsPerInstr,
    Value warpId, Value laneId, int warpsPerBlock, int numOfElems,
    ArrayRef<int64_t> reps, SharedMemoryObject smemObj, ArrayRef<Value> strides,
    gpu::SwizzledSharedEncodingAttr srcLayout, unsigned nonKDim, unsigned kDim);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_CONVERTLAYOUTOPTOLLVM_SHAREDTODOTOPERANDHELPER_H_
