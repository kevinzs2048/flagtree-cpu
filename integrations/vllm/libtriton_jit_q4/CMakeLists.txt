cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(LibTritonJitQ4Op LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

if(NOT DEFINED TRITON_JIT_ROOT)
  message(FATAL_ERROR "TRITON_JIT_ROOT is required")
endif()
if(NOT DEFINED TRITON_JIT_BUILD)
  message(FATAL_ERROR "TRITON_JIT_BUILD is required")
endif()

list(APPEND CMAKE_MODULE_PATH "${TRITON_JIT_ROOT}/cmake")
set(fmt_DIR "${TRITON_JIT_BUILD}/_deps/fmt-build")
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch MODULE REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(TritonJIT CONFIG REQUIRED
  PATHS "${TRITON_JIT_BUILD}" NO_DEFAULT_PATH)

add_library(triton_jit_q4_op SHARED q4_op.cpp w8_op.cpp)
target_link_libraries(triton_jit_q4_op PRIVATE
  Torch::Torch triton_jit)
target_compile_definitions(triton_jit_q4_op PRIVATE
  Q4_KERNEL_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/q4_kernels.py"
  W8_KERNEL_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/w8_kernels.py")
set_target_properties(triton_jit_q4_op PROPERTIES
  PREFIX "lib"
  BUILD_RPATH "${TRITON_JIT_BUILD}/src")
