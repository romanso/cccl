//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// Source to shared library (api_scenario_recipes.md, section 3): one CUDA source
// (host + device) compiled at runtime into a shared library, then loaded and
// called; the exported host function launches the device kernel.
//
//   source (.cu string) -> [clang FE -> device codegen -> lld link] -> .so
//                        -> load -> dlsym("run") -> run(p, v) -> kernel writes v
//
// This is the core product scenario: a self-contained host+device module built
// with no system compiler or linker. Verified end-to-end (compile, link, load,
// launch, read back the device result).

#include <cstdio>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

namespace
{
const char* k_src = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

__global__ void k(int* p, int v) { *p = v; }

extern "C" _CCCL_VISIBILITY_EXPORT void run(int* p, int v)
{
  k<<<1, 1>>>(p, v);
}
)";
} // namespace

int main()
{
  std::printf("shared-library -- source -> shared library -> load -> run\n");

  auto config       = hostjit::detectDefaultConfig();
  config.enable_pch = false;

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(k_src))
  {
    std::fprintf(stderr, "  compile/link/load failed: %s\n", compiler.getLastError().c_str());
    return 1;
  }
  std::printf("  loaded module: %s\n", compiler.getLoadedModulePath().c_str());

  auto run = compiler.getFunction<void (*)(int*, int)>("run");
  if (!run)
  {
    std::fprintf(stderr, "  entry 'run' not found: %s\n", compiler.getLastError().c_str());
    return 1;
  }

  int* d = nullptr;
  if (cudaMalloc(&d, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "  cudaMalloc failed\n");
    return 1;
  }

  run(d, 42);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
  {
    std::fprintf(stderr, "  launch/sync error: %s\n", cudaGetErrorString(e));
    cudaFree(d);
    return 1;
  }

  int got = -1;
  cudaMemcpy(&got, d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d);

  const bool ok = (got == 42);
  std::printf("  run(42) -> %d (expected 42)\n", got);
  std::printf("shared-library: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
