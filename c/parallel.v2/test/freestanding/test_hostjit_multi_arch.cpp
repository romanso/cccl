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
// Multiple architectures in one compile (api_scenario_recipes.md, section 5:
// -gencode -> one fatbin covering several archs).
//
// Support probe. The prototype carries a single sm_version and the option parser
// accepts only --gpu-architecture=sm_NN; there is no -gencode. So this scenario
// cannot be expressed yet. The test encodes exactly that:
//   * a single-architecture compile succeeds (baseline), and
//   * a -gencode multi-arch request is REJECTED by the option parser.
// It documents the gap empirically and will begin to fail -- prompting an update
// -- once multi-arch codegen and -gencode parsing land (CFE-113/114).

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <hostjit/compiler.hpp> // cudacc.h + detail helpers
#include <hostjit/config.hpp>

namespace
{
const char* k_src = R"(
extern "C" __global__ void k(int* p, int v) { *p = v; }
)";
} // namespace

int main()
{
  namespace fs = std::filesystem;
  std::printf("multi-arch -- -gencode support probe\n");

  auto config       = hostjit::detectDefaultConfig();
  config.enable_pch = false;
  std::vector<std::string> base;
  config.appendCommandLineArguments(base); // includes a single --gpu-architecture=sm_NN

  const std::string obj = (fs::temp_directory_path() / "hostjit_multiarch.o").string();

  // Baseline: a single architecture compiles. Establishes that the only delta in
  // the multi-arch attempt below is the -gencode options.
  {
    auto opts = hostjit::detail::make_cudacc_option_ptrs(base);
    hostjit::detail::CudaccProgramGuard prog;
    if (cudaccCreateProgram(&prog.program, k_src, "k.cu") != CUDACC_SUCCESS)
    {
      std::fprintf(stderr, "  createProgram failed\n");
      return 1;
    }
    auto r = cudaccCompileProgramToObject(
      prog.program, obj.c_str(), "", static_cast<int>(opts.size()), opts.empty() ? nullptr : opts.data());
    if (r != CUDACC_SUCCESS)
    {
      std::fprintf(stderr,
                   "  baseline single-arch compile failed (unexpected):\n%s\n",
                   hostjit::detail::get_cudacc_program_log(prog.program).c_str());
      return 1;
    }
    std::printf("  single-arch (--gpu-architecture): OK\n");
  }

  // Multi-arch attempt: add -gencode. Expected today: REJECTED (unknown option).
  bool gencode_rejected = false;
  {
    std::vector<std::string> multi = base;
    multi.emplace_back("-gencode");
    multi.emplace_back("arch=compute_90,code=sm_90");
    multi.emplace_back("-gencode");
    multi.emplace_back("arch=compute_120,code=sm_120");
    auto opts = hostjit::detail::make_cudacc_option_ptrs(multi);
    hostjit::detail::CudaccProgramGuard prog;
    if (cudaccCreateProgram(&prog.program, k_src, "k.cu") != CUDACC_SUCCESS)
    {
      std::fprintf(stderr, "  createProgram failed\n");
      return 1;
    }
    auto r = cudaccCompileProgramToObject(
      prog.program, obj.c_str(), "", static_cast<int>(opts.size()), opts.empty() ? nullptr : opts.data());
    gencode_rejected = (r != CUDACC_SUCCESS);
    std::printf(
      "  -gencode multi-arch: %s (%s)\n", gencode_rejected ? "REJECTED" : "ACCEPTED", cudaccGetErrorString(r));
    if (!gencode_rejected)
    {
      std::fprintf(stderr, "  -gencode was accepted -- multi-arch may now be supported; update this probe\n");
    }
  }

  const bool ok = gencode_rejected; // current expectation: no multi-arch support
  std::printf("multi-arch: %s (want -gencode REJECTED until multi-arch lands)\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
