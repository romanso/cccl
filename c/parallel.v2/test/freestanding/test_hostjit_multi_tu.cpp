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
// Multi-piece linking: several inputs combined into one shared library.
//
// Under RDC (relocatable device code) each host object is compiled with
// -fgpu-rdc: it carries offloading entries and a device-bitcode sidecar, but no
// per-object fatbin and no per-TU registration ctor. The final link device-links
// every object's device code into ONE fatbin and generates a SINGLE registration
// object (via llvm::offloading::wrapCudaBinary); the atexit-capture shim is weak,
// so the per-TU copies merge into one instance. Both cases below therefore link,
// run, and unload cleanly:
//
//   * Device code (external LLVM bitcode / LTO-IR) device-linked into one object.
//   * Several *host* objects linked together into one shared library.
//   * A device function defined in one object and called from another, which is
//     what separate compilation is for: neither object is complete on its own.
//
// All three are exercised through ONE routine (build_link_run) with different
// inputs. Expectation: every scenario LINKS and runs its entry points.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <hostjit/compiler.hpp> // cudacc.h + detail helpers (CudaccProgramGuard, option ptrs, log)
#include <hostjit/config.hpp>
#include <hostjit/loader.hpp>

namespace
{
enum class LinkOutcome
{
  Linked, // linked into a .so AND every entry ran with the expected result
  LinkFailed, // the shared-library link step rejected the objects
  SetupError // a step unrelated to the property under test failed
};

const char* to_str(LinkOutcome o)
{
  switch (o)
  {
    case LinkOutcome::Linked:
      return "LINKED";
    case LinkOutcome::LinkFailed:
      return "LINK-FAILED";
    default:
      return "SETUP-ERROR";
  }
}

struct HostUnit
{
  const char* name; // logical .cu name
  const char* src; // CUDA C++ source
  const char* entry; // exported host entry point
  int in; // argument passed to the entry
  int expect; // expected value written to the device int
};

// A device function provided as external LLVM IR, device-linked into the fatbin.
const char* k_device_add_ir = R"(
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i32 @dev_add(i32 %a, i32 %b) alwaysinline {
entry:
  %r = add i32 %a, %b
  ret i32 %r
}
)";

// Host unit whose kernel calls the external (bitcode-provided) dev_add.
const char* k_src_device = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

extern "C" __device__ int dev_add(int a, int b);
__global__ void kernel_dev(int* p, int v) { *p = dev_add(v, 1); }

extern "C" _CCCL_VISIBILITY_EXPORT void entry_dev(int* p, int v)
{
  kernel_dev<<<1, 1>>>(p, v);
}
)";

// Two independent host units (distinct kernels + entries).
const char* k_src_a = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

__global__ void kernel_a(int* p, int v) { *p = v; }
extern "C" _CCCL_VISIBILITY_EXPORT void entry_a(int* p, int v) { kernel_a<<<1, 1>>>(p, v); }
)";

const char* k_src_b = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

__global__ void kernel_b(int* p, int v) { *p = v * 2; }
extern "C" _CCCL_VISIBILITY_EXPORT void entry_b(int* p, int v) { kernel_b<<<1, 1>>>(p, v); }
)";

// A device function defined in one TU and called from another. Neither object
// links on its own; the device symbol is resolved at the final device link.
const char* k_src_provider = R"(
extern "C" __device__ int scale(int x) { return x * 3; }
)";

const char* k_src_consumer = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

extern "C" __device__ int scale(int);
__global__ void kernel_scaled(int* p, int v) { *p = scale(v); }
extern "C" _CCCL_VISIBILITY_EXPORT void entry_scaled(int* p, int v) { kernel_scaled<<<1, 1>>>(p, v); }
)";

bool write_file(const std::string& path, const char* text)
{
  std::ofstream f(path);
  if (!f)
  {
    return false;
  }
  f << text;
  return static_cast<bool>(f);
}

// Compile each host unit to an object (external device bitcode, if any, is
// device-linked into the fatbin), link all objects into one shared library, and
// -- if the link succeeds -- load it and run every entry, checking results.
LinkOutcome build_link_run(
  const char* scenario, const std::vector<HostUnit>& units, const std::vector<std::string>& device_bitcode_files)
{
  std::printf("--- scenario: %s (%zu host object(s), %zu device-bitcode input(s))\n",
              scenario,
              units.size(),
              device_bitcode_files.size());

  auto config       = hostjit::detectDefaultConfig();
  config.enable_pch = false;
  config.device_pch_path.clear();
  config.host_pch_path.clear();
  for (const auto& bc : device_bitcode_files)
  {
    config.device_bitcode_files.push_back(bc);
  }

  std::vector<std::string> options;
  config.appendCommandLineArguments(options);
  auto opt_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / ("hostjit_multitu_" + std::to_string(std::random_device{}()));
  fs::create_directories(dir);

  std::vector<std::string> objs;
  objs.reserve(units.size());
  for (size_t i = 0; i < units.size(); ++i)
  {
    const std::string obj = (dir / (std::string("u") + std::to_string(i) + ".o")).string();
    hostjit::detail::CudaccProgramGuard prog;
    if (cudaccCreateProgram(&prog.program, units[i].src, units[i].name) != CUDACC_SUCCESS)
    {
      std::fprintf(stderr, "  create program failed for %s\n", units[i].name);
      return LinkOutcome::SetupError;
    }
    auto r = cudaccCompileProgramToObject(
      prog.program,
      obj.c_str(),
      /*outputCubinPath*/ "",
      static_cast<int>(opt_ptrs.size()),
      opt_ptrs.empty() ? nullptr : opt_ptrs.data());
    if (r != CUDACC_SUCCESS)
    {
      std::fprintf(
        stderr, "  compile failed for %s:\n%s\n", units[i].name, hostjit::detail::get_cudacc_program_log(prog.program).c_str());
      return LinkOutcome::SetupError;
    }
    objs.push_back(obj);
  }

#ifdef _WIN32
  const std::string lib = (dir / "multitu.dll").string();
#else
  const std::string lib = (dir / "libmultitu.so").string();
#endif

  hostjit::detail::CudaccProgramGuard link_prog;
  if (cudaccCreateProgram(&link_prog.program, "", "multitu-link") != CUDACC_SUCCESS)
  {
    return LinkOutcome::SetupError;
  }
  std::vector<const char*> obj_ptrs;
  obj_ptrs.reserve(objs.size());
  for (const auto& o : objs)
  {
    obj_ptrs.push_back(o.c_str());
  }
  auto lr = cudaccLinkToSharedLibrary(
    link_prog.program,
    static_cast<int>(obj_ptrs.size()),
    obj_ptrs.data(),
    lib.c_str(),
    static_cast<int>(opt_ptrs.size()),
    opt_ptrs.empty() ? nullptr : opt_ptrs.data());

  if (lr != CUDACC_SUCCESS)
  {
    // Print a compact first line of the linker error for context.
    std::string log = hostjit::detail::get_cudacc_program_log(link_prog.program);
    std::printf("  link REJECTED: %.200s%s\n", log.c_str(), log.size() > 200 ? " ..." : "");
    return LinkOutcome::LinkFailed;
  }

  // Linked: load and run every entry.
  int* d_ptr = nullptr;
  if (cudaMalloc(&d_ptr, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "  cudaMalloc failed\n");
    return LinkOutcome::SetupError;
  }

  LinkOutcome outcome = LinkOutcome::Linked;
  {
    hostjit::DynamicLibrary mod;
    if (!mod.load(lib))
    {
      std::fprintf(stderr, "  load failed: %s\n", mod.getLastError().c_str());
      cudaFree(d_ptr);
      return LinkOutcome::SetupError;
    }
    for (const auto& u : units)
    {
      if (u.entry == nullptr)
      {
        continue; // a unit that only provides device code exports nothing
      }
      auto fn = mod.getFunction<void (*)(int*, int)>(u.entry);
      if (!fn)
      {
        std::fprintf(stderr, "  entry '%s' not found\n", u.entry);
        outcome = LinkOutcome::SetupError;
        break;
      }
      fn(d_ptr, u.in);
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess)
      {
        std::fprintf(stderr, "  launch/sync error on '%s': %s\n", u.entry, cudaGetErrorString(e));
        outcome = LinkOutcome::SetupError;
        break;
      }
      int result = -1;
      cudaMemcpy(&result, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
      if (result != u.expect)
      {
        std::fprintf(stderr, "  '%s': got %d, expected %d\n", u.entry, result, u.expect);
        outcome = LinkOutcome::SetupError;
        break;
      }
      std::printf("  '%s'(%d) -> %d ok\n", u.entry, u.in, result);
    }
    // mod unloads here: unregister every captured fatbin, then unmap.
  }
  cudaError_t e = cudaDeviceSynchronize(); // a dangling registration would fault here
  if (e != cudaSuccess)
  {
    std::fprintf(stderr, "  post-unload sync error: %s\n", cudaGetErrorString(e));
    outcome = LinkOutcome::SetupError;
  }
  cudaFree(d_ptr);
  return outcome;
}
} // namespace

int main()
{
  namespace fs = std::filesystem;

  // Scenario 1: device linking -- one host object with an external device-bitcode
  // function device-linked into its fatbin. Expected: LINKS and runs.
  const std::string ir = (fs::temp_directory_path() / "hostjit_dev_add.ll").string();
  if (!write_file(ir, k_device_add_ir))
  {
    std::fprintf(stderr, "could not write device IR\n");
    return 2;
  }
  const std::vector<HostUnit> device_units = {{"dev.cu", k_src_device, "entry_dev", 41, 42}};
  const LinkOutcome device_outcome         = build_link_run("device-linking (1 host obj + device bitcode)", device_units, {ir});

  // Scenario 2: two host objects linked into one shared library. Under RDC the
  // device code combines into one fatbin and a single registration object, so
  // this links and both entry points run.
  const std::vector<HostUnit> host_units = {
    {"a.cu", k_src_a, "entry_a", 21, 21},
    {"b.cu", k_src_b, "entry_b", 21, 42},
  };
  const LinkOutcome multi_host_outcome = build_link_run("multi-host (2 host objs)", host_units, {});

  // Scenario 3: one object defines a device function, another calls it. The
  // caller's object cannot be finalized on its own, so this only links if the
  // device code really is left relocatable until the final link.
  const std::vector<HostUnit> cross_units = {
    {"prov.cu", k_src_provider, nullptr, 0, 0},
    {"use.cu", k_src_consumer, "entry_scaled", 14, 42},
  };
  const LinkOutcome cross_outcome = build_link_run("cross-object device call (2 host objs)", cross_units, {});

  const bool device_ok     = (device_outcome == LinkOutcome::Linked);
  const bool multi_host_ok = (multi_host_outcome == LinkOutcome::Linked);
  const bool cross_ok      = (cross_outcome == LinkOutcome::Linked);

  std::printf("\nresult: device-linking=%s, multi-host=%s, cross-object=%s (want LINKED)\n",
              to_str(device_outcome),
              to_str(multi_host_outcome),
              to_str(cross_outcome));

  if (device_ok && multi_host_ok && cross_ok)
  {
    std::printf("multi-TU: PASS (device code and multiple host objects both link and run)\n");
    return 0;
  }
  std::printf("multi-TU: FAIL (an outcome did not match the expectation)\n");
  std::fflush(stdout);
  std::fflush(stderr);
  return 1;
}
