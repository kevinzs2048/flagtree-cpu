set(triton_test_analysis_sources
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
)

# Membar analysis calls NVIDIA-specific helpers.  Do not pull that test pass
# into a CPU-only optimizer, where the backend library is intentionally absent.
if("nvidia" IN_LIST TRITON_CODEGEN_BACKENDS)
  list(APPEND triton_test_analysis_sources TestMembar.cpp)
endif()

add_mlir_library(TritonTestAnalysis
  ${triton_test_analysis_sources}
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
