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
// Unload behaviour that test_hostjit_unload does NOT cover (CFE-104). That test
// takes the straight path -- load, launch, unload, repeat, with CUDA work in
// between -- and passes. These two probes go after the edges found while reading
// the CUDART and driver sources, where the unload story is weaker than the
// straight path suggests. Both record what happens today, so a change in either
// direction shows up as a test result rather than as a surprise in the field.
//
// The first probe runs on Linux only; the reason is in main().
//
//   1. One module image, two handles. Loading the same library twice gives two
//      handles to ONE mapped image, and the fatbin-unregister callbacks live in
//      that image, shared. Unloading one handle therefore unregisters the fatbin
//      out from under the other handle, which is still open and still callable.
//
//   2. The window between unload and the driver actually letting go. CUDART's
//      module teardown only queues the unload; the driver call is issued from the
//      next CUDA runtime entry point, so a process that stops calling CUDA keeps
//      the device state resident even though the library is gone. DynamicLibrary
//      closes that window by making one runtime call after unregistering, and the
//      probe checks it: it watches free device memory through the DRIVER API,
//      which does not drain the queue and therefore shows the state as the driver
//      sees it right after the unload.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>
#include <hostjit/loader.hpp>

namespace
{
// 16 MiB of device state, so that a module still resident is visible in the
// free-memory numbers; a small module moves them by nothing at all.
const char* k_source = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

__device__ int ballast[4 * 1024 * 1024];

__global__ void device_kernel(int* ptr, int v)
{
  ballast[0] = v;
  *ptr       = ballast[0];
}

extern "C" _CCCL_VISIBILITY_EXPORT void host_entry(int* ptr, int v)
{
  device_kernel<<<1, 1>>>(ptr, v);
}
)";

#ifndef _WIN32
// Build the module and keep it on disk, so the probe can open it directly.
bool build_module(hostjit::CompilerConfig config, std::string& module_path)
{
  config.enable_pch      = false;
  config.keep_artifacts  = true;

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(k_source))
  {
    std::fprintf(stderr, "  compile failed: %s\n", compiler.getLastError().c_str());
    return false;
  }
  module_path = compiler.getLoadedModulePath();
  return !module_path.empty() && std::filesystem::exists(module_path);
}

// Launch through one handle and report the outcome instead of asserting it:
// these probes are about what the runtime does, not about a known-good result.
cudaError_t launch_through(hostjit::DynamicLibrary& lib, int* d_ptr, int v)
{
  auto host_entry = reinterpret_cast<void (*)(int*, int)>(lib.getSymbol("host_entry"));
  if (!host_entry)
  {
    return cudaErrorSymbolNotFound;
  }
  cudaGetLastError(); // clear anything pending
  host_entry(d_ptr, v);
  const cudaError_t launch = cudaGetLastError();
  const cudaError_t sync   = cudaDeviceSynchronize();
  return launch != cudaSuccess ? launch : sync;
}

// 1. Two handles on one image: unloading one pulls the fatbin out from under
//    the other.
bool probe_shared_image(const std::string& so_path, int* d_ptr)
{
  hostjit::DynamicLibrary first, second;
  if (!first.load(so_path) || !second.load(so_path))
  {
    std::fprintf(stderr, "  could not open the module twice\n");
    return false;
  }

  const cudaError_t before_first  = launch_through(first, d_ptr, 1);
  const cudaError_t before_second = launch_through(second, d_ptr, 2);
  std::printf("  both handles open: first=%s second=%s\n",
              cudaGetErrorName(before_first),
              cudaGetErrorName(before_second));
  if (before_first != cudaSuccess || before_second != cudaSuccess)
  {
    std::fprintf(stderr, "  a launch failed while both handles were open\n");
    return false;
  }

  first.unload();

  const cudaError_t after = launch_through(second, d_ptr, 3);
  std::printf("  after unloading the first handle, the second launches: %s\n", cudaGetErrorName(after));

  second.unload();
  cudaGetLastError();

  // Today the fatbin is gone while the second handle is still open, so the
  // launch fails. If this ever starts succeeding -- because registration became
  // reference-counted, or because each handle got its own image -- the probe
  // fails and this file (and the CFE-104 write-up) should be updated.
  const bool as_documented = after != cudaSuccess;
  if (!as_documented)
  {
    std::fprintf(stderr, "  the second handle still works: unload is no longer image-wide -- update the notes\n");
  }
  return as_documented;
}
#endif // !_WIN32

// 2. Free device memory as the driver sees it. The driver API does not go
//    through CUDART's pending-unload queue, so it shows the state before the
//    queue is drained.
using CuMemGetInfoFn = int (*)(size_t*, size_t*);

CuMemGetInfoFn load_driver_mem_get_info()
{
#ifdef _WIN32
  HMODULE libcuda = LoadLibraryA("nvcuda.dll");
  if (!libcuda)
  {
    return nullptr;
  }
  return reinterpret_cast<CuMemGetInfoFn>(GetProcAddress(libcuda, "cuMemGetInfo_v2"));
#else
  void* libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (!libcuda)
  {
    return nullptr;
  }
  return reinterpret_cast<CuMemGetInfoFn>(dlsym(libcuda, "cuMemGetInfo_v2"));
#endif
}

bool probe_unload_window(hostjit::CompilerConfig config, int* d_ptr)
{
  CuMemGetInfoFn cu_mem_get_info = load_driver_mem_get_info();
  if (!cu_mem_get_info)
  {
    std::printf("  driver API unavailable; probe skipped\n");
    return true;
  }

  size_t free_resident = 0, total = 0;
  size_t free_after_unload = 0;
  size_t free_after_runtime_call = 0;

  {
    config.enable_pch = false;
    hostjit::JITCompiler compiler(config);
    if (!compiler.compile(k_source))
    {
      std::fprintf(stderr, "  compile failed: %s\n", compiler.getLastError().c_str());
      return false;
    }
    auto host_entry = compiler.getFunction<void (*)(int*, int)>("host_entry");
    if (!host_entry)
    {
      std::fprintf(stderr, "  'host_entry' not found\n");
      return false;
    }
    host_entry(d_ptr, 7);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
      std::fprintf(stderr, "  launch failed\n");
      return false;
    }
    cu_mem_get_info(&free_resident, &total);
  }
  // The library is unloaded here. No CUDA runtime call has been made since.
  cu_mem_get_info(&free_after_unload, &total);

  size_t runtime_free = 0, runtime_total = 0;
  cudaMemGetInfo(&runtime_free, &runtime_total); // a runtime entry point: drains the queue
  cu_mem_get_info(&free_after_runtime_call, &total);

  const long long held_after_unload =
    static_cast<long long>(free_after_runtime_call) - static_cast<long long>(free_after_unload);
  std::printf("  resident: %zu, right after unload: %zu, after one runtime call: %zu\n",
              free_resident,
              free_after_unload,
              free_after_runtime_call);
  std::printf("  device memory released only once the runtime was called again: %lld byte(s)\n", held_after_unload);

  if (free_after_runtime_call < free_resident)
  {
    std::fprintf(stderr, "  memory was not released at all -- that is a leak, not a window\n");
    return false;
  }
  if (held_after_unload > 0)
  {
    std::fprintf(stderr,
                 "  the module was still resident after unload: the loader is not draining the runtime's "
                 "pending-unload queue\n");
    return false;
  }
  std::printf("  no window: unload gave the device state back without a further runtime call\n");
  return true;
}
} // namespace

int main()
{
  std::printf("unload-gaps -- unload edges around CFE-104\n");

  auto config = hostjit::detectDefaultConfig();

  int* d_ptr = nullptr;
  if (cudaMalloc(&d_ptr, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "unload-gaps: cudaMalloc failed\n");
    return 2;
  }

  bool ok = true;

#ifdef _WIN32
  // Windows runs the second probe only. The DLL has no CRT startup, so load()
  // runs the static initializers itself; loading the same path twice hands back
  // the same image and runs them again, registering the fatbin a second time and
  // overflowing the single-slot capture table. The probe would trap rather than
  // report, so on Windows the shared-image question stays where the CFE-104
  // write-up leaves it -- open.
  std::printf("[one image, two handles] skipped: a second load re-runs the module ctor on Windows\n");
#else
  std::string module_path;
  if (!build_module(config, module_path))
  {
    cudaFree(d_ptr);
    return 2;
  }

  std::printf("[one image, two handles]\n");
  ok &= probe_shared_image(module_path, d_ptr);
#endif

  std::printf("[window between unload and the driver letting go]\n");
  ok &= probe_unload_window(config, d_ptr);

  cudaFree(d_ptr);
  std::printf("unload-gaps: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
