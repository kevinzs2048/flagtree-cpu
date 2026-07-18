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

#ifndef TRITONTOTRITONCPU_CONVERSION_PASSES
#define TRITONTOTRITONCPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMemoryOps : Pass<"triton-cpu-convert-memory-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory ops.";
    let description = [{

    }];

    let options = [
        Option<"useGatherScatter", "use-gather-scatter",
               "bool", /*default*/"false",
               "Use Gather or Scatter to lower memory ops.">,
    ];

    let constructor = "mlir::triton::cpu::createConvertMemoryOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertElementwiseOps : Pass<"triton-cpu-convert-elementwise-ops", "mlir::ModuleOp"> {
    let summary = "Convert elementwise ops.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertElementwiseOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertElemManipOps : Pass<"triton-cpu-convert-elem-manip-ops", "mlir::ModuleOp"> {
    let summary = "Convert elements manipulation ops (transpose, shuffle, etc.).";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertElemManipOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertPtrOps : Pass<"triton-cpu-convert-ptr-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton ops related to pointer arithmetics.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertPtrOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotOp : Pass<"triton-cpu-convert-dot-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton DotOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertDotOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertControlFlowOps : Pass<"triton-cpu-convert-control-flow-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton DotOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertControlFlowOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertHistogramOp : Pass<"triton-cpu-convert-histogram-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton HistogramOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertHistogramOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect"];
}

def ConvertReductionOp : Pass<"triton-cpu-convert-reduction", "mlir::ModuleOp"> {
    let summary = "Convert Triton ReduceOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertReductionOp()";

    let options = [
        Option<"useMultiDimReductionOp", "use-multidim-reduction-op",
               "bool", /*default*/"false",
               "Use vector::MultiDimReductionOp and its default lowering when possible.">,

        Option<"useReductionOp", "use-reduction-op",
               "bool", /*default*/"false",
                "Use vector::ReductionOp and its default lowering when possible.">,
    ];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertScanOp : Pass<"triton-cpu-convert-scan", "mlir::ModuleOp"> {
    let summary = "Convert Triton ScanOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertScanOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertAtomicOps : Pass<"triton-cpu-convert-atomic-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton atomic operations.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertAtomicOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDebugOps : Pass<"triton-cpu-convert-debug-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton debug operations.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertDebugOps()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ScalarizeUsingForOp : Pass<"triton-cpu-scalarize", "mlir::ModuleOp"> {
    let summary = "Insert Loops for ops, that are not vectorizable";
    let description = [{
        This pass is used to reduce compile time by generating loops for
        operations that cannot be handled as vectors, and simply increases
        the amount of IR without any further optimization.
    }];

    let options = [
        Option<"skipGatherScatter", "skip-gather-scatter",
               "bool", /*default*/"false",
               "Skip scalarizing gather/scatter ops.">,
    ];

    let constructor = "mlir::triton::cpu::createScalarizeUsingForOpPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}


#endif
