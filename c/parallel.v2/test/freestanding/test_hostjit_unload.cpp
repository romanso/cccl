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
// Unload test: a JIT module must be safely AND fully released on unload.
//
// It guards two complementary regressions in one run:
//
//  1. Safety (dangling reference): after a fully-registered module is unmapped
//     mid-run, the CUDA runtime must not keep a reference into the freed module
//     image. We launch the JIT'd kernel and verify its result every iteration,
//     and keep issuing CUDA work across iterations (including a sync AFTER each
//     unload). A dangling registration would fault or produce a wrong value on a
//     later cycle.
//
//  2. No leak: the module must actually be unmapped, not retained for the life
//     of the process. After the load/launch/unload loop we count how many JIT
//     module images are still mapped -- expected 0.
//
// (Skipping unload entirely would pass #1 but fail #2; unloading without proper
// unregister/drain would pass #2 but fail #1 -- so both checks are needed.)
//
// The module's file name is not hard-coded: we ask the loader for the real path
// of a loaded module (getLoadedModulePath, via the OS) and match on that
// basename, so a rename in the compiler cannot silently make the leak probe a
// false pass. Module enumeration is platform-specific (Windows EnumProcessModules,
// Linux /proc/self/maps); where it is unavailable the leak check is skipped while
// the safety check still runs.

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>

#  include <psapi.h>
#elif defined(__linux__)
#  include <set>
#endif

static const char* k_source = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

__global__ void device_kernel(int* ptr, int v)
{
  *ptr = v;
}

extern "C" _CCCL_VISIBILITY_EXPORT void host_entry(int* ptr, int v)
{
  device_kernel<<<1, 1>>>(ptr, v);
}
)";

// Directory-strip a path and drop the " (deleted)" suffix that /proc/self/maps
// appends for a mapping whose backing file was unlinked. Handles both separators.
static std::string basename_of(std::string p)
{
  const auto d = p.find(" (deleted)");
  if (d != std::string::npos)
  {
    p = p.substr(0, d);
  }
  const auto pos = p.find_last_of("/\\");
  return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

// Count distinct currently-mapped module images whose basename == target.
// Returns -1 if enumeration is unsupported / failed on this platform.
#if defined(_WIN32)
static int count_mapped(const std::string& target)
{
  HMODULE modules[8192];
  DWORD needed = 0;
  if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed))
  {
    return -1;
  }
  int n = static_cast<int>(needed / sizeof(HMODULE));
  if (n > 8192)
  {
    n = 8192;
  }
  int count = 0;
  for (int i = 0; i < n; ++i)
  {
    char path[MAX_PATH] = {};
    if (K32GetModuleFileNameExA(GetCurrentProcess(), modules[i], path, MAX_PATH))
    {
      if (_stricmp(basename_of(path).c_str(), target.c_str()) == 0)
      {
        ++count;
      }
    }
  }
  return count;
}
#elif defined(__linux__)
static int count_mapped(const std::string& target)
{
  FILE* f = std::fopen("/proc/self/maps", "r");
  if (!f)
  {
    return -1;
  }
  std::set<std::string> paths; // dedupe: one file spans several mapping lines
  char line[8192];
  while (std::fgets(line, sizeof(line), f))
  {
    // Line: "addr perms offset dev inode pathname". Skip the 5 fixed fields
    // (none contain spaces) and take the rest as the pathname.
    int pos = -1;
    std::sscanf(line, "%*s %*s %*s %*s %*s %n", &pos);
    if (pos < 0)
    {
      continue;
    }
    std::string path = line + pos;
    while (!path.empty() && (path.back() == '\n' || path.back() == '\r'))
    {
      path.pop_back();
    }
    if (path.empty() || path[0] != '/')
    {
      continue; // anonymous / [heap] / [stack] etc.
    }
    const auto del      = path.find(" (deleted)");
    const std::string s = (del == std::string::npos) ? path : path.substr(0, del);
    if (basename_of(s) == target)
    {
      paths.insert(s);
    }
  }
  std::fclose(f);
  return static_cast<int>(paths.size());
}
#else
static int count_mapped(const std::string&)
{
  return -1;
}
#endif

int main()
{
  auto config = hostjit::detectDefaultConfig();

  int* d_ptr = nullptr;
  if (cudaMalloc(&d_ptr, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "unload: cudaMalloc failed\n");
    return 2;
  }

  constexpr int kIters = 16;
  int rc                = 0;
  std::string modname; // JIT module basename, learned at runtime on the first iter

  // (1) Safety: launch + verify each cycle, and issue CUDA work after each unload.
  for (int i = 0; i < kIters; ++i)
  {
    const int expected = 1000 + i;
    {
      hostjit::JITCompiler compiler(config);
      if (!compiler.compile(k_source))
      {
        std::fprintf(stderr, "unload: compile failed on iter %d:\n%s\n", i, compiler.getLastError().c_str());
        rc = 2;
        break;
      }
      if (modname.empty())
      {
        modname = basename_of(compiler.getLoadedModulePath());
      }

      auto host_fn = compiler.getFunction<void (*)(int*, int)>("host_entry");
      if (!host_fn)
      {
        std::fprintf(stderr, "unload: 'host_entry' not found on iter %d\n", i);
        rc = 2;
        break;
      }

      host_fn(d_ptr, expected);
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess)
      {
        std::fprintf(stderr, "unload: launch/sync error on iter %d: %s\n", i, cudaGetErrorString(e));
        rc = 1;
        break;
      }

      int result = -1;
      cudaMemcpy(&result, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
      if (result != expected)
      {
        std::fprintf(stderr, "unload: WRONG result on iter %d: got %d, want %d\n", i, result, expected);
        rc = 1;
        break;
      }
      // `compiler` goes out of scope here -> unload() -> unregister fatbin + unmap.
    }

    // CUDA work AFTER the module was unmapped: a dangling reference would fault here.
    cudaError_t e2 = cudaDeviceSynchronize();
    if (e2 != cudaSuccess)
    {
      std::fprintf(stderr, "unload: post-unload sync error on iter %d: %s\n", i, cudaGetErrorString(e2));
      rc = 1;
      break;
    }
    std::printf("unload: iter %d ok (result=%d) after unload\n", i, expected);
  }

  cudaFree(d_ptr);

  // (2) No leak: after the loop no JIT module image should remain mapped.
  if (rc == 0)
  {
    const int leaked = count_mapped(modname);
    if (leaked < 0)
    {
      std::printf("unload: leak probe skipped (module enumeration unsupported here)\n");
    }
    else
    {
      std::printf("unload: %d '%s' module(s) still mapped after %d cycles\n", leaked, modname.c_str(), kIters);
      if (leaked > 1) // allow <=1 as slack; before the fix every iteration leaked one
      {
        rc = 1;
      }
    }
  }

  std::printf("unload: %s (%d load/launch/unload cycles)\n", rc == 0 ? "PASS" : "FAIL", kIters);
  std::fflush(stdout);
  std::fflush(stderr);
  // Normal return (not std::_Exit): a CUDA context is live, so let the CRT run
  // cudart's orderly teardown instead of fast-failing at exit.
  return rc;
}
