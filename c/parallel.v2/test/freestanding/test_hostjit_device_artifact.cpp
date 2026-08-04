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
// Source to device artifact (api_scenario_recipes.md, section 1). The product
// intent is "compile the device side only, stop at the requested output kind
// (PTX / cubin / fatbin / LTO-IR), and hand it back in memory
// (cudaccGetOutputData)"; nothing runs on the host and there is no device link.
//
// The prototype is file-based and does not select the output kind by option, so
// it can only reach part of this scenario. This test exercises what it can
// produce and leaves the rest documented as gaps
// (chj/cfe_wp/113/scenarios/test_coverage.md):
//
//   [reachable]
//     * device LLVM bitcode in memory   (cudaccCompile --bitcode)
//         -- device-only, no host, no link: matches the scenario's shape.
//     * LTO-IR in memory                 (cudaccCompile --ltoir)
//         -- device-only as well, and checked below by feeding it back in as an
//            external operator, which is the form the product hands out.
//     * cubin IN MEMORY                  (JITCompiler::getCubin)
//         -- but only as a side effect of a full source->.so compile, so this
//            path is host+link, not the device-only compile the recipe asks for.
//
//   [not reachable in the prototype -> see test_coverage.md]
//     * option-selected output kind (--ptx / --cubin / --fatbin / --ltoir);
//     * standalone PTX / fatbin emit;
//     * in-memory GetOutputData for a device-only compile.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <hostjit/compiler.hpp> // cudacc.h + detail helpers
#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

namespace
{
// Pure device source (kernel only) -- section 1 is "device only, no host".
const char* k_device_src = R"(
extern "C" __global__ void scale(float* x, float a, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    x[i] *= a;
  }
}
)";

// A standalone device operator: no kernel, one external device function. This is
// what a caller asks for as LTO-IR, to device-link into someone else's module
// later.
const char* k_op_src = R"(
extern "C" __device__ int op(int x) { return x + 100; }
)";

// A kernel calling that operator, used to check the LTO-IR artifact links.
const char* k_caller_src = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

extern "C" __device__ int op(int);

__global__ void k(int* p, int v) { *p = op(v); }

extern "C" _CCCL_VISIBILITY_EXPORT void run(int* p, int v)
{
  k<<<1, 1>>>(p, v);
}
)";

// A full host+device source, so a complete compile yields a cubin to read back.
const char* k_full_src = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

__global__ void k(int* p, int v) { *p = v; }

extern "C" _CCCL_VISIBILITY_EXPORT void run(int* p, int v)
{
  k<<<1, 1>>>(p, v);
}
)";

void write_bytes(const std::string& path, const std::vector<char>& bytes)
{
  std::ofstream f(path, std::ios::binary);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// LLVM bitcode ("BC" 0xC0DE) or textual LLVM IR (.ll) written by the compiler.
bool looks_like_llvm(const std::vector<unsigned char>& b)
{
  if (b.size() >= 4 && b[0] == 0x42 && b[1] == 0x43 && b[2] == 0xC0 && b[3] == 0xDE)
  {
    return true;
  }
  const std::string head(b.begin(), b.begin() + std::min<size_t>(b.size(), 128));
  return head.find("target datalayout") != std::string::npos || head.find("ModuleID") != std::string::npos
      || head.find("define") != std::string::npos;
}

bool is_elf(const std::vector<unsigned char>& b)
{
  return b.size() >= 4 && b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F';
}

// Device-only path: source -> device LLVM bitcode (a file).
bool device_bitcode_to_file()
{
  namespace fs = std::filesystem;
  const std::string bc = (fs::temp_directory_path() / "hostjit_device.bc").string();

  auto config = hostjit::detectDefaultConfig();
  std::vector<std::string> options;
  config.appendCommandLineArguments(options);
  options.push_back("--bitcode");
  options.push_back("scale.cu");
  auto opt_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  const std::string source        = k_device_src;
  const cudaccFile source_file    = hostjit::detail::make_cudacc_source("scale.cu", source);
  const cudaccFile* const input[] = {&source_file};

  hostjit::detail::CudaccOutput out;
  if (cudaccCompile(&out.output, 1, input, static_cast<int>(opt_ptrs.size()), opt_ptrs.data()) != CUDACC_SUCCESS)
  {
    std::fprintf(stderr, "  device-bitcode compile failed:\n%s\n", out.log().c_str());
    return false;
  }

  const auto data = out.data();
  const std::vector<unsigned char> bytes(data.begin(), data.end());
  write_bytes(bc, data);
  std::printf("  device bitcode: %zu bytes\n", bytes.size());
  if (bytes.empty() || !looks_like_llvm(bytes))
  {
    std::fprintf(stderr, "  device bitcode missing or not LLVM bitcode/IR\n");
    return false;
  }
  return true;
}

// Count the function bodies in a linked cubin: one .text.<name> section with a
// non-zero size per function. One means the operator was inlined into the kernel.
int count_cubin_functions(const std::vector<char>& cubin)
{
  if (cubin.size() < sizeof(Elf64_Ehdr))
  {
    return -1;
  }
  const auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(cubin.data());
  if (ehdr->e_shoff == 0 || ehdr->e_shstrndx == SHN_UNDEF
      || ehdr->e_shoff + static_cast<size_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr) > cubin.size())
  {
    return -1;
  }
  const auto* shdrs = reinterpret_cast<const Elf64_Shdr*>(cubin.data() + ehdr->e_shoff);
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

// Device-only path: operator source -> LTO-IR (a file), then the same artifact
// device-linked into a kernel that calls it. Producing bytes is not enough --
// the point of the scenario is an artifact someone else can link -- so the
// check is that it links, inlines, and runs.
bool device_ltoir_to_file()
{
  namespace fs = std::filesystem;
  const std::string ltoir = (fs::temp_directory_path() / "hostjit_device_op.ltoir").string();
  fs::remove(ltoir);

  auto config = hostjit::detectDefaultConfig();
  std::vector<std::string> options;
  config.appendCommandLineArguments(options);
  options.push_back("--ltoir");
  options.push_back("op.cu");
  auto opt_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  const std::string source        = k_op_src;
  const cudaccFile source_file    = hostjit::detail::make_cudacc_source("op.cu", source);
  const cudaccFile* const input[] = {&source_file};

  hostjit::detail::CudaccOutput out;
  if (cudaccCompile(&out.output, 1, input, static_cast<int>(opt_ptrs.size()), opt_ptrs.data()) != CUDACC_SUCCESS)
  {
    std::fprintf(stderr, "  LTO-IR compile failed:\n%s\n", out.log().c_str());
    return false;
  }

  // The artifact comes back in memory; the link step below takes a path, so it
  // is put on disk here rather than by the compiler.
  const auto data = out.data();
  const std::vector<unsigned char> bytes(data.begin(), data.end());
  write_bytes(ltoir, data);
  std::printf("  device LTO-IR: %zu bytes\n", bytes.size());
  if (bytes.empty())
  {
    std::fprintf(stderr, "  LTO-IR output missing\n");
    return false;
  }

  auto link_config              = hostjit::detectDefaultConfig();
  link_config.enable_pch        = false;
  link_config.device_ltoir_files.push_back(ltoir);
  link_config.device_nvvm_bypass = true; // kernel as IR too, so the link is a full LTO

  hostjit::JITCompiler compiler(link_config);
  if (!compiler.compile(k_caller_src))
  {
    std::fprintf(stderr, "  linking the LTO-IR artifact failed: %s\n", compiler.getLastError().c_str());
    return false;
  }

  const int functions = count_cubin_functions(compiler.getCubin());
  std::printf("  linked against a caller: %d function(s) in the cubin\n", functions);
  if (functions != 1)
  {
    std::fprintf(stderr,
                 functions <= 0 ? "  cubin carries no device code\n"
                                : "  operator was not inlined into the kernel\n");
    return false;
  }

  auto run = compiler.getFunction<void (*)(int*, int)>("run");
  if (!run)
  {
    std::fprintf(stderr, "  entry 'run' not found: %s\n", compiler.getLastError().c_str());
    return false;
  }
  int* d = nullptr;
  if (cudaMalloc(&d, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "  cudaMalloc failed\n");
    return false;
  }
  run(d, 5); // op(5) = 105
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
  {
    std::fprintf(stderr, "  launch/sync error: %s\n", cudaGetErrorString(e));
    cudaFree(d);
    return false;
  }
  int got = -1;
  cudaMemcpy(&got, d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d);

  const bool ok = (got == 105);
  std::printf("  run(5) -> %d (expected 105): %s\n", got, ok ? "ok" : "MISMATCH");
  return ok;
}

// The only in-memory device artifact the prototype exposes: the cubin extracted
// during a full source->.so compile.
bool cubin_in_memory()
{
  auto config       = hostjit::detectDefaultConfig();
  config.enable_pch = false;

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(k_full_src))
  {
    std::fprintf(stderr, "  compile failed: %s\n", compiler.getLastError().c_str());
    return false;
  }

  const std::vector<char>& cubin = compiler.getCubin();
  const std::vector<unsigned char> b(cubin.begin(), cubin.end());
  std::printf("  cubin in memory: %zu bytes\n", b.size());
  if (b.empty() || !is_elf(b))
  {
    std::fprintf(stderr, "  cubin missing or not an ELF/cubin image\n");
    return false;
  }
  return true;
}
} // namespace

int main()
{
  std::printf("device-artifact -- source -> device artifact\n");
  bool ok = true;

  std::printf("[device bitcode -> file]\n");
  ok &= device_bitcode_to_file();

  std::printf("[device LTO-IR -> file, then device-linked]\n");
  ok &= device_ltoir_to_file();

  std::printf("[cubin -> memory]\n");
  ok &= cubin_in_memory();

  std::printf("device-artifact: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
