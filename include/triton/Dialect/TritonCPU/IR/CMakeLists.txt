# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonCPUOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=triton_cpu)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=triton_cpu)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=triton_cpu)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=triton_cpu)
add_mlir_doc(TritonCPUDialect TritonCPUDialect dialects/ -gen-dialect-doc)
add_mlir_doc(TritonCPUOps TritonCPUOps dialects/ -gen-op-doc)
add_public_tablegen_target(TritonCPUTableGen)

set(LLVM_TARGET_DEFINITIONS TritonCPUAttrDefs.td)
mlir_tablegen(TritonCPUAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TritonCPUAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(TritonCPUAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonCPUAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TritonCPUAttrDefsIncGen)
