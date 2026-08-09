#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using Kernel = void (*)(void *, void *, void *, void *, void *, int32_t,
                        int32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                        uint32_t, uint32_t);

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void pinToCpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
    throw std::runtime_error("pthread_setaffinity_np failed");
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 7 || argc > 9) {
    std::cerr << "usage: " << argv[0]
              << " SO K N BLOCK_N THREADS FIRST_CPU [ITERS] [BATCHES]\n";
    return 2;
  }
  try {
    const int k = std::stoi(argv[2]);
    const int n = std::stoi(argv[3]);
    const int blockN = std::stoi(argv[4]);
    const int threads = std::stoi(argv[5]);
    const int firstCpu = std::stoi(argv[6]);
    const int iterations = argc >= 8 ? std::stoi(argv[7]) : 5;
    const int batches = argc >= 9 ? std::stoi(argv[8]) : 15;
    if (k <= 0 || k % 4 || n <= 0 || blockN <= 0 || blockN % 4 ||
        n % blockN || threads < 2 || iterations <= 0 || batches <= 0)
      throw std::runtime_error("invalid shape, thread, or sample count");

    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error(dlerror());
    dlerror();
    auto kernel = reinterpret_cast<Kernel>(
        dlsym(handle, "_w8a8_wide_gemv_kernel"));
    if (const char *error = dlerror())
      throw std::runtime_error(error);

    const int outputBlocks = n / blockN;
    std::vector<int8_t> x(k);
    std::vector<int8_t> packed(static_cast<size_t>(k) * n);
    std::vector<float> weightScale(n);
    std::vector<uint16_t> singleOutput(n);
    std::vector<uint16_t> parallelOutput(n);
    float xScale = 0.013f;
    for (int i = 0; i < k; ++i)
      x[i] = static_cast<int8_t>((i * 17 + 5) % 255 - 127);
    for (size_t i = 0; i < packed.size(); ++i)
      packed[i] = static_cast<int8_t>(i * 29 + 11);
    for (int i = 0; i < n; ++i)
      weightScale[i] = 0.001f + static_cast<float>(i % 31) * 0.00002f;

    std::barrier start(threads + 1);
    std::barrier finish(threads + 1);
    int activeThreads = 1;
    bool stop = false;
    std::vector<std::thread> workers;
    for (int worker = 0; worker < threads; ++worker) {
      workers.emplace_back([&, worker] {
        pinToCpu(firstCpu + worker);
        while (true) {
          start.arrive_and_wait();
          if (stop)
            break;
          if (activeThreads == 1) {
            if (worker == threads - 1)
              kernel(x.data(), &xScale, packed.data(), weightScale.data(),
                     singleOutput.data(), 0, outputBlocks, 0, 0, 0, 1, 1, 1);
          } else if (worker < activeThreads) {
            const int begin = outputBlocks * worker / activeThreads;
            const int end = outputBlocks * (worker + 1) / activeThreads;
            kernel(x.data(), &xScale, packed.data(), weightScale.data(),
                   parallelOutput.data(), begin, end, 0, 0, 0, 1, 1, 1);
          }
          finish.arrive_and_wait();
        }
      });
    }

    auto run = [&](int count, int repeats) {
      activeThreads = count;
      const auto begin = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < repeats; ++iteration) {
        start.arrive_and_wait();
        finish.arrive_and_wait();
      }
      const auto end = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::micro>(end - begin).count() /
             repeats;
    };

    run(1, 20);
    run(threads, 20);
    if (singleOutput != parallelOutput)
      throw std::runtime_error("parallel output differs from single core");

    std::vector<double> singleSamples;
    std::vector<double> parallelSamples;
    for (int batch = 0; batch < batches; ++batch) {
      if ((batch & 1) == 0) {
        singleSamples.push_back(run(1, iterations));
        parallelSamples.push_back(run(threads, iterations));
      } else {
        parallelSamples.push_back(run(threads, iterations));
        singleSamples.push_back(run(1, iterations));
      }
    }
    const double singleUs = median(std::move(singleSamples));
    const double parallelUs = median(std::move(parallelSamples));
    std::cout << "PASS W8 vocabulary persistent-worker A/B K=" << k
              << " N=" << n << " BN=" << blockN << '\n'
              << "threads=" << threads << '\n'
              << "single_us=" << singleUs << '\n'
              << "parallel_us=" << parallelUs << '\n'
              << "speedup=" << singleUs / parallelUs << "x\n"
              << "bit_exact=true\n";

    stop = true;
    start.arrive_and_wait();
    for (auto &worker : workers)
      worker.join();
    dlclose(handle);
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
