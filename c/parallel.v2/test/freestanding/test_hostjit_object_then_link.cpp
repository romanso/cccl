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
// Separate compilation (api_scenario_recipes.md, section 2): compile a source to
// a host object as a distinct step, then link that object into a shared library,
// load it, and run.
//
//   source (.cu) --[compile -c]--> host object (.o) --[link]--> .so --> load --> run
//
// What this proves and what it does NOT: it drives compile-to-object and link as
// separate API calls for ONE translation unit, which works today. The recipe
// also wants the object to hold *relocatable* device code so several objects can
// be device-linked at the final step; the prototype instead embeds a complete
// fatbin per object, so multiple host objects collide at link. That multi-object
// consequence is covered by test_hostjit_multi_tu.cpp; here we exercise the
// single-TU separate-compilation path end to end.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <hostjit/compiler.hpp> // cudacc.h + detail helpers
#include <hostjit/config.hpp>
#include <hostjit/loader.hpp>

namespace
{
const char* k_src = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

__global__ void k(int* p, int v) { *p = v + 1; }

extern "C" _CCCL_VISIBILITY_EXPORT void run(int* p, int v)
{
  k<<<1, 1>>>(p, v);
}
)";

bool is_relocatable_object(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  unsigned char m[4] = {};
  f.read(reinterpret_cast<char*>(m), 4);
#ifdef _WIN32
  // COFF relocatable object: first two bytes are the machine type (little-endian).
  // amd64 == IMAGE_FILE_MACHINE_AMD64 (0x8664).
  return f.gcount() >= 2 && m[0] == 0x64 && m[1] == 0x86;
#else
  return f.gcount() == 4 && m[0] == 0x7f && m[1] == 'E' && m[2] == 'L' && m[3] == 'F';
#endif
}
} // namespace

int main()
{
  namespace fs = std::filesystem;
  std::printf("object-then-link -- source -> host object -> link -> run\n");

  auto config       = hostjit::detectDefaultConfig();
  config.enable_pch = false;
  std::vector<std::string> base_options;
  config.appendCommandLineArguments(base_options);

  const fs::path dir = fs::temp_directory_path() / ("hostjit_obj_" + std::to_string(std::random_device{}()));
  fs::create_directories(dir);
  const std::string obj = (dir / "u.o").string();
#ifdef _WIN32
  const std::string lib = (dir / "u.dll").string();
#else
  const std::string lib = (dir / "libu.so").string();
#endif

  // Step 1: source -> host object (-c), a separate compilation step.
  std::vector<std::string> compile_options = base_options;
  compile_options.emplace_back("-c");
  compile_options.emplace_back("-o");
  compile_options.push_back(obj);
  compile_options.emplace_back("u.cu");
  auto compile_opts = hostjit::detail::make_cudacc_option_ptrs(compile_options);

  const std::string source        = k_src;
  const cudaccFile source_file    = hostjit::detail::make_cudacc_source("u.cu", source);
  const cudaccFile* const input[] = {&source_file};

  hostjit::detail::CudaccOutput out;
  if (cudaccCompile(&out.output, 1, input, static_cast<int>(compile_opts.size()), compile_opts.data())
      != CUDACC_SUCCESS)
  {
    std::fprintf(stderr, "  compile-to-object failed:\n%s\n", out.log().c_str());
    return 1;
  }
  if (!fs::exists(obj) || fs::file_size(obj) == 0 || !is_relocatable_object(obj))
  {
    std::fprintf(stderr, "  host object missing or not a relocatable object\n");
    return 1;
  }
  std::printf("  host object: %s (%ju bytes)\n", obj.c_str(), static_cast<std::uintmax_t>(fs::file_size(obj)));

  // Step 2: link the single object into a shared library (deferred link step).
  std::vector<std::string> link_options = base_options;
  link_options.emplace_back("--shared");
  link_options.emplace_back("-o");
  link_options.push_back(lib);
  link_options.push_back(obj);
  auto link_opts = hostjit::detail::make_cudacc_option_ptrs(link_options);

  hostjit::detail::CudaccOutput link_out;
  if (cudaccCompile(&link_out.output, 0, nullptr, static_cast<int>(link_opts.size()), link_opts.data())
      != CUDACC_SUCCESS)
  {
    std::fprintf(stderr, "  link failed:\n%s\n", link_out.log().c_str());
    return 1;
  }

  // Step 3: load the shared library and run the entry.
  int* d = nullptr;
  if (cudaMalloc(&d, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "  cudaMalloc failed\n");
    return 1;
  }

  int got  = -1;
  bool ok  = false;
  bool err = false;
  {
    hostjit::DynamicLibrary mod;
    if (!mod.load(lib))
    {
      std::fprintf(stderr, "  load failed: %s\n", mod.getLastError().c_str());
      err = true;
    }
    else
    {
      auto run = mod.getFunction<void (*)(int*, int)>("run");
      if (!run)
      {
        std::fprintf(stderr, "  entry 'run' not found\n");
        err = true;
      }
      else
      {
        run(d, 41);
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
          std::fprintf(stderr, "  launch/sync error\n");
          err = true;
        }
        else
        {
          cudaMemcpy(&got, d, sizeof(int), cudaMemcpyDeviceToHost);
          ok = (got == 42);
        }
      }
    }
    // mod unloads here: unregister the captured fatbin, then unmap.
  }
  cudaDeviceSynchronize();
  cudaFree(d);

  if (err)
  {
    return 1;
  }
  std::printf("  run(41) -> %d (expected 42)\n", got);
  std::printf("object-then-link: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
