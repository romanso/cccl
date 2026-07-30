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
// CCCL motivating case: one CUDA source whose kernel calls an external,
// pre-built device operator that is device-linked into the module fatbin of a
// shared library.
//
//   source (.cu) whose kernel calls extern "C" __device__ int op(int)
//        +  external device code defining op   (device-linked into the fatbin)
//        -> .so -> load -> run(p, v) -> kernel calls op(v) -> result checked
//
// The external operator can arrive in two transports; both are exercised here
// because they take *different* paths through the compiler:
//
//   A. LLVM bitcode (--device-bitcode / config.device_bitcode_files): linked
//      into the kernel's LLVM module (Linker::linkModules) and INLINED before
//      device codegen.
//   B. NVRTC LTO-IR (--device-ltoir / config.device_ltoir_files): CCCL's real
//      transport (an operator compiled by NVRTC -dlto). Device-linked by
//      nvJitLink -lto at the RDC final link. The kernel is clang-emitted PTX, so
//      this is *partial* LTO: the extern op is RESOLVED but not inlined (full
//      inlining needs the kernel in LTO-IR too -- the libNVVM product path).
//   C. The same NVRTC LTO-IR operator, but with the kernel handed to nvJitLink as
//      device IR (config.device_nvvm_bypass) instead of PTX. That makes it *full*
//      LTO: nvJitLink's own NVVM inlines the operator into the kernel, so the
//      linked cubin ends up with the kernel alone and no call in it.
//
// All cases must build, load, run, and produce the same result. In the product
// the external code is passed in memory (--ltoir-input <data> <size>); the
// prototype takes a file path -- functionally equivalent for the device link
// (see chj/cfe_wp/113/scenarios/test_coverage.md).

#include <cstdio>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvrtc.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

namespace
{
// External device function provided as LLVM IR (transport A): op(x) = x + 100.
const char* k_op_ir = R"(
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i32 @op(i32 %x) alwaysinline {
entry:
  %r = add i32 %x, 100
  ret i32 %r
}
)";

// External device function as CUDA source (transport B): NVRTC compiles it to
// LTO-IR below. Same op(x) = x + 100.
const char* k_op_src = R"(
extern "C" __device__ int op(int x) { return x + 100; }
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

// Compile k_op_src to NVRTC LTO-IR for the requested arch; write it to out_path.
bool make_op_ltoir(int sm_version, const std::string& out_path)
{
  nvrtcProgram prog = nullptr;
  if (nvrtcCreateProgram(&prog, k_op_src, "op.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS)
  {
    return false;
  }
  const std::string arch = "--gpu-architecture=compute_" + std::to_string(sm_version);
  const char* opts[]     = {arch.c_str(), "--relocatable-device-code=true", "-dlto"};
  const nvrtcResult r    = nvrtcCompileProgram(prog, 3, opts);
  if (r != NVRTC_SUCCESS)
  {
    size_t ls = 0;
    nvrtcGetProgramLogSize(prog, &ls);
    std::string log(ls, '\0');
    nvrtcGetProgramLog(prog, log.data());
    std::fprintf(stderr, "  nvrtc compile failed: %s\n", log.c_str());
    nvrtcDestroyProgram(&prog);
    return false;
  }
  size_t sz = 0;
  nvrtcGetLTOIRSize(prog, &sz);
  std::vector<char> ltoir(sz);
  nvrtcGetLTOIR(prog, ltoir.data());
  nvrtcDestroyProgram(&prog);

  std::ofstream f(out_path, std::ios::binary);
  if (!f)
  {
    return false;
  }
  f.write(ltoir.data(), static_cast<std::streamsize>(ltoir.size()));
  return static_cast<bool>(f);
}

// Launch the entry point of an already-built module and check op(5) == 105.
bool run_check(const char* label, hostjit::JITCompiler& compiler)
{
  auto run = compiler.getFunction<void (*)(int*, int)>("run");
  if (!run)
  {
    std::fprintf(stderr, "  [%s] entry 'run' not found: %s\n", label, compiler.getLastError().c_str());
    return false;
  }

  int* d = nullptr;
  if (cudaMalloc(&d, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "  [%s] cudaMalloc failed\n", label);
    return false;
  }
  run(d, 5); // op(5) = 105
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
  {
    std::fprintf(stderr, "  [%s] launch/sync error: %s\n", label, cudaGetErrorString(e));
    cudaFree(d);
    return false;
  }
  int got = -1;
  cudaMemcpy(&got, d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d);

  const bool ok = (got == 105);
  std::printf("  [%s] run(5) -> %d (expected 105): %s\n", label, got, ok ? "ok" : "MISMATCH");
  return ok;
}

// Build k_src (with the operator already attached to config), load, run(5), and
// check op(5) == 105.
bool build_run_check(const char* label, hostjit::CompilerConfig config)
{
  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(k_src))
  {
    std::fprintf(stderr, "  [%s] compile/link/load failed: %s\n", label, compiler.getLastError().c_str());
    return false;
  }
  return run_check(label, compiler);
}

// The two things device IR needs before the NVVM link-time optimizer will take
// it. Upstream Clang emits neither, which is why the compiler adds them (and a
// Clang built with -fnvvm-compatible-device-ir emits them itself):
//
//  * llvm.used listing the kernels. Nothing inside the device module refers to a
//    kernel -- the host launches it by name through the runtime -- so without
//    this the optimizer considers every kernel unreachable and deletes it, and
//    the link then yields a cubin with no code in it.
//  * a data layout that does not mark address space 15 as non-integral, which
//    NVVM rejects outright.
//
// Checked before the cubin, because a failure here explains a failure there.
bool check_nvvm_ir_annotations(const std::string& ir_path)
{
  std::ifstream f(ir_path, std::ios::binary);
  const std::string ir((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (ir.empty())
  {
    std::fprintf(stderr, "  [full LTO] no device IR at %s\n", ir_path.c_str());
    return false;
  }
  const bool has_used     = ir.find("llvm.used") != std::string::npos;
  const bool has_integral = ir.find("-ni:15") == std::string::npos;
  std::printf("  [full LTO] device IR: kernels in llvm.used: %s, no non-integral address space: %s\n",
              has_used ? "yes" : "NO",
              has_integral ? "yes" : "NO");
  return has_used && has_integral;
}

// Count the kernel/function bodies in a linked cubin: one .text.<name> section
// with a non-zero size per function. Zero means the optimizer dropped
// everything; one means the operator was inlined into the kernel.
int count_cubin_functions(const std::vector<char>& cubin)
{
  if (cubin.size() < sizeof(Elf64_Ehdr))
  {
    return -1;
  }
  const auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(cubin.data());
  if (ehdr->e_shoff == 0 || ehdr->e_shstrndx == SHN_UNDEF)
  {
    return -1;
  }
  const auto* shdrs = reinterpret_cast<const Elf64_Shdr*>(cubin.data() + ehdr->e_shoff);
  if (ehdr->e_shoff + static_cast<size_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr) > cubin.size())
  {
    return -1;
  }
  const char* names = cubin.data() + shdrs[ehdr->e_shstrndx].sh_offset;

  int functions = 0;
  for (unsigned i = 0; i < ehdr->e_shnum; ++i)
  {
    const std::string name = names + shdrs[i].sh_name;
    if (name.rfind(".text.", 0) == 0 && shdrs[i].sh_size > 0)
    {
      ++functions;
    }
  }
  return functions;
}
} // namespace

int main()
{
  namespace fs = std::filesystem;
  std::printf("external-device-code -- source + external operator -> shared library -> run\n");

  // Transport A: operator as LLVM bitcode (linked + inlined at the IR level).
  bool a = false;
  {
    const std::string ir = (fs::temp_directory_path() / "hostjit_ext_op.ll").string();
    std::ofstream f(ir);
    if (!f)
    {
      std::fprintf(stderr, "  could not write external device IR\n");
      return 1;
    }
    f << k_op_ir;
    f.close();

    auto config       = hostjit::detectDefaultConfig();
    config.enable_pch = false;
    config.device_bitcode_files.push_back(ir); // external device code, device-linked (REQ-3)
    a = build_run_check("bitcode --device-bitcode (inlined)", config);
  }

  // Transport B: operator as NVRTC LTO-IR (CCCL's real transport; resolved via
  // nvJitLink -lto at the RDC final link).
  bool b = false;
  {
    auto config       = hostjit::detectDefaultConfig();
    config.enable_pch = false;

    const std::string ltoir = (fs::temp_directory_path() / "hostjit_ext_op.ltoir").string();
    if (!make_op_ltoir(config.sm_version, ltoir))
    {
      std::fprintf(stderr, "  could not produce operator LTO-IR via NVRTC\n");
      return 1;
    }
    config.device_ltoir_files.push_back(ltoir); // external device code, device-linked (REQ-3)
    b = build_run_check("LTO-IR --device-ltoir (resolved, not inlined)", config);
  }

  // Transport C: same operator LTO-IR, kernel handed over as device IR, so the
  // device link is a full LTO and the operator gets inlined into the kernel.
  bool c = false;
  {
    auto config       = hostjit::detectDefaultConfig();
    config.enable_pch = false;

    const std::string ltoir = (fs::temp_directory_path() / "hostjit_ext_op.ltoir").string();
    if (!make_op_ltoir(config.sm_version, ltoir))
    {
      std::fprintf(stderr, "  could not produce operator LTO-IR via NVRTC\n");
      return 1;
    }
    config.device_ltoir_files.push_back(ltoir);
    config.device_nvvm_bypass = true;
    config.device_nvvm_ir_out = (fs::temp_directory_path() / "hostjit_ext_device.nvvm.bc").string();
    fs::remove(config.device_nvvm_ir_out);

    const char* label = "full LTO";
    hostjit::JITCompiler compiler(config);
    if (!compiler.compile(k_src))
    {
      std::fprintf(stderr, "  [%s] compile/link/load failed: %s\n", label, compiler.getLastError().c_str());
      return 1;
    }

    c = check_nvvm_ir_annotations(config.device_nvvm_ir_out);
    if (c)
    {
      const std::vector<char>& cubin = compiler.getCubin();
      const int functions            = count_cubin_functions(cubin);
      // A cubin with no code in it is the failure this transport exists to catch:
      // that is what comes out when the kernel is not retained across the link.
      // One function means the operator ended up inlined into the kernel.
      std::printf("  [%s] linked cubin: %zu bytes, %d function(s)\n", label, cubin.size(), functions);
      c = functions == 1;
      if (!c)
      {
        std::fprintf(stderr,
                     functions <= 0 ? "  [full LTO] cubin carries no device code\n"
                                    : "  [full LTO] operator was not inlined into the kernel\n");
      }
    }
    c = c && run_check(label, compiler);
  }

  const bool ok = a && b && c;
  std::printf("external-device-code: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
