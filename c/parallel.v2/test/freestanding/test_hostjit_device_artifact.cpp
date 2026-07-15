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
//     * device LLVM bitcode to a FILE   (libnvccCompileProgramToDeviceBitcode)
//         -- device-only, no host, no link: matches the scenario's shape.
//     * cubin IN MEMORY                  (JITCompiler::getCubin)
//         -- but only as a side effect of a full source->.so compile, so this
//            path is host+link, not the device-only compile the recipe asks for.
//
//   [not reachable in the prototype -> see test_coverage.md]
//     * option-selected output kind (--ptx / --cubin / --fatbin / --ltoir);
//     * standalone PTX / fatbin / LTO-IR emit;
//     * in-memory GetOutputData for a device-only compile.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <hostjit/compiler.hpp> // libnvcc.h + detail helpers
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

std::vector<unsigned char> read_bytes(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
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
  auto opt_ptrs = hostjit::detail::make_libnvcc_option_ptrs(options);

  hostjit::detail::LibnvccProgramGuard prog;
  if (libnvccCreateProgram(&prog.program, k_device_src, "scale.cu") != LIBNVCC_SUCCESS)
  {
    std::fprintf(stderr, "  createProgram failed\n");
    return false;
  }
  auto r = libnvccCompileProgramToDeviceBitcode(
    prog.program, bc.c_str(), static_cast<int>(opt_ptrs.size()), opt_ptrs.empty() ? nullptr : opt_ptrs.data());
  if (r != LIBNVCC_SUCCESS)
  {
    std::fprintf(
      stderr, "  device-bitcode compile failed:\n%s\n", hostjit::detail::get_libnvcc_program_log(prog.program).c_str());
    return false;
  }

  const auto bytes = read_bytes(bc);
  std::printf("  device bitcode: %zu bytes\n", bytes.size());
  if (bytes.empty() || !looks_like_llvm(bytes))
  {
    std::fprintf(stderr, "  device bitcode missing or not LLVM bitcode/IR\n");
    return false;
  }
  return true;
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

  std::printf("[cubin -> memory]\n");
  ok &= cubin_in_memory();

  std::printf("device-artifact: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
