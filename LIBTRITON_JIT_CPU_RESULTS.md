# libtriton-jit CPU backend results

Date: 2026-08-02

## Scope

`third_party/libtriton_jit` now has a real `BACKEND=CPU` build. Its C++ API
can compile a Triton-CPU 3.7 kernel through embedded Python, load the generated
shared object, and invoke the exported CPU ABI directly. The wrapper does not
contain a replacement compute loop.

The CPU ABI is:

```text
runtime arguments, pid_x, pid_y, pid_z, grid_x, grid_y, grid_z
```

Runtime tuples are flattened to scalar ABI arguments. CPU launches do not add
the two GPU global-scratch pointers. Common pointer signatures have typed
direct-call fast paths; mixed signatures use a cached libffi call interface.

## Launch overhead on CIX

A two-program, three-runtime-argument add kernel was pinned to CPU 11. The
kernel had already been compiled and loaded, and 20,000 launches were timed
per process. Five-process medians after the thread-safety changes are:

| Entry | Median overhead |
| --- | ---: |
| generic typed C++ JIT call | 1.356 us |
| raw-argument C++ JIT call | 0.468 us |
| Python JIT call | 55.906 us |

The generic C++ entry removes about 41x of the Python launch overhead; the raw
entry removes about 119x. The raw entry is intended for a framework dispatcher
that already owns a stable signature and argument array. The generic entry
retains tensor/signature processing and is still below 1.4 us.

These values measure wrapper/dispatch overhead with a deliberately tiny
kernel. They are not kernel throughput numbers and must not be subtracted from
unrelated operator measurements.

## Correctness and concurrency

The CPU test suite covers:

- direct shared-object loading;
- a mixed pointer/i32/f32 libffi signature;
- the three-pointer typed fast path;
- flattened runtime tuple layout;
- C++ JIT compilation through local Triton-CPU 3.7 and exact execution;
- eight threads sharing one kernel during first lazy load and 1,600 launches;
- eight threads sharing one compiled JIT function for 800 launches.

Kernel handle initialization uses `std::call_once`. The global function
registry is locked during lookup/insertion. The specialization cache uses a
shared lock on non-hot hits, an exclusive rechecked miss path, and a stable
atomic last-entry pointer for repeated launches. Compiled kernels therefore
run concurrently; only new specialization compilation is serialized.

Five of five CTest cases pass.

## Reproduction

```bash
cd third_party/libtriton_jit
/home/cix/venv-fep-e2e/bin/cmake -S . -B build-cpu -G Ninja \
  -DBACKEND=CPU -DTRITON_JIT_BUILD_OPERATORS=OFF \
  -DBUILD_TESTING=ON
/home/cix/venv-fep-e2e/bin/cmake --build build-cpu -j4

PYTHONPATH=/home/kevin/triton-opt-cpu/python:/home/cix/venv-fep-e2e/lib/python3.11/site-packages \
LD_LIBRARY_PATH=$PWD/build-cpu/src:/home/cix/venv-fep-e2e/lib/python3.11/site-packages/torch/lib:/home/kevin/triton-opt-cpu/python/triton/_C \
TRITON_CACHE_DIR=/home/kevin/triton-opt-cpu/artifacts/cache-libtriton-jit-cpu-test \
  /home/cix/venv-fep-e2e/bin/ctest --test-dir build-cpu --output-on-failure
```
