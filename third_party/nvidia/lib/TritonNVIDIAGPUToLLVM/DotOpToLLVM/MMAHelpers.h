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

#include "Utility.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace triton {
namespace NVIDIA {

// The descriptor format is described in the spec:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
// Unnamed fieids are not used
union SMEMDescriptor {
  uint64_t descriptor;
  struct {
    uint64_t baseAddress : 14;
    uint64_t : 2;
    uint64_t leadDimensionBaseOffset : 14;
    uint64_t : 2;
    uint64_t strideDimensionBaseOffset : 14;
    uint64_t : 3;
    uint64_t matrixBaseOffset : 3;
    uint64_t : 10;
    uint64_t swizzlingMode : 2;
  };
};

// Abstract class to calculate the address of a shared or tensor memory slice.
class DotOpMmaMemLoader {
public:
  virtual ~DotOpMmaMemLoader() = default;
  virtual Value memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                        Location loc) = 0;
};

// Helper class to load shared memory slices following MMAv3 layout.
class DotOpMmaV3SmemLoader : public DotOpMmaMemLoader {
public:
  DotOpMmaV3SmemLoader() {}
  DotOpMmaV3SmemLoader(Value tensor, Value base, SmallVector<int64_t> shape,
                       ArrayRef<int64_t> allocSwizzleShape, Value warpId,
                       unsigned int dimWpt, bool trans,
                       SmallVector<unsigned int> instrShape,
                       int64_t elementBitwidth,
                       ConversionPatternRewriter &rewriter, Location loc);
  // Return a descriptor pointing to the shared memory slice at coordinates (a,
  // b)
  Value smemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                 Location loc);

  Value memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                Location loc) override {
    return smemLoad(a, b, rewriter, loc);
  }

private:
  Value base;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> allocSwizzleShape;
  Value warpId;
  int dimWpt;
  bool trans;
  int fastMovingDim;
  Value elemsPerSwizzlingRowVal;
  SmallVector<unsigned int> instrShape;
  int elemsPerSwizzlingRow;
  int64_t elemBits;
  Value descriptor;
};

// Helper class to load tensor memory following MMAv5 layout.
class DotOpMmaV5TmemLoader : public DotOpMmaMemLoader {
public:
  DotOpMmaV5TmemLoader() {}
  DotOpMmaV5TmemLoader(Value tensor, Value base,
                       SmallVector<unsigned int> instrShape, bool interleaved,
                       bool trans);
  Value tmemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                 Location loc);

  Value memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                Location loc) override {
    return tmemLoad(a, b, rewriter, loc);
  }

private:
  Value base;
  bool trans;
  bool interleaved;
  bool unpacked;
  SmallVector<unsigned int> instrShape;
  int numElementsPer32b;
  int numRepM;
};

} // namespace NVIDIA
} // namespace triton
} // namespace mlir
