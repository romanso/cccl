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
// Source plus external device code (api_scenario_recipes.md, section 4) -- the
// CCCL motivating case: one CUDA source plus external, pre-built device code (an
// operator already compiled to device IR, e.g. by NVRTC) device-linked into the
// module fatbin of a shared library.
//
//   source (.cu) whose kernel calls extern "C" __device__ int op(int)
//        +  external device code defining op   (device-linked into the fatbin)
//        -> .so -> load -> run(p, v) -> kernel calls op(v) -> result checked
//
// The external device code is supplied here as LLVM bitcode via
// config.device_bitcode_files (--device-bitcode). LTO-IR uses the same mechanism
// (--device-ltoir) but needs an NVRTC-produced LTO-IR input, not exercised here.
// In the product this external code is passed in memory (--ltoir-input <data>
// <size>); the prototype takes a file path -- functionally equivalent for the
// device link (see chj/cfe_wp/113/scenarios/test_coverage.md).

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

namespace
{
// External device function provided as LLVM IR: op(x) = x + 100.
const char* k_op_ir = R"(
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i32 @op(i32 %x) alwaysinline {
entry:
  %r = add i32 %x, 100
  ret i32 %r
}
)";

// Source whose kernel calls the external op; the exported host entry launches it.
const char* k_src = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

extern "C" __device__ int op(int); // provided by the external device code

__global__ void k(int* p, int v) { *p = op(v); }

extern "C" _CCCL_VISIBILITY_EXPORT void run(int* p, int v)
{
  k<<<1, 1>>>(p, v);
}
)";
} // namespace

int main()
{
  namespace fs = std::filesystem;
  std::printf("external-device-code -- source + external device code -> shared library -> run\n");

  const std::string ir = (fs::temp_directory_path() / "hostjit_ext_op.ll").string();
  {
    std::ofstream f(ir);
    if (!f)
    {
      std::fprintf(stderr, "  could not write external device IR\n");
      return 1;
    }
    f << k_op_ir;
  }

  auto config       = hostjit::detectDefaultConfig();
  config.enable_pch = false;
  config.device_bitcode_files.push_back(ir); // external device code, device-linked (REQ-3)

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(k_src))
  {
    std::fprintf(stderr, "  compile/link/load failed: %s\n", compiler.getLastError().c_str());
    return 1;
  }

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

  run(d, 5); // op(5) = 105
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

  const bool ok = (got == 105);
  std::printf("  run(5) -> %d (expected 105; external op adds 100)\n", got);
  std::printf("external-device-code: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
