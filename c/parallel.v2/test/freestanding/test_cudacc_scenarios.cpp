//===----------------------------------------------------------------------===//
//
// The four product scenarios, run through the library API exactly as the
// requirements document spells them out: one cudaccCompile call per step, all
// inputs named on the command line, and the ones that live in memory passed as
// cudaccFile entries rather than written out by the caller.
//
// The point is the documented code itself -- that it is well formed, that the
// library accepts those options, and that the result runs. Where the prototype
// still needs a flag the product would not, the flag is added here and left out
// of the document; each such place says so.
//
//   1. source                          -> shared library
//   2. source + external operator      -> shared library (operator in memory)
//   3. two sources -> two objects      -> one shared library
//   4. source                          -> device artifact returned in memory
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <hostjit/compiler.hpp>
#include <hostjit/config.hpp>
#include <hostjit/loader.hpp>

namespace
{
// A kernel with a host entry point that launches it. Every scenario that ends
// in a shared library exports 'run'.
const char* k_lib_src = R"cu(
#include <cuda_runtime.h>
__global__ void k(int* p, int v) { *p = v + 1; }
extern "C" __attribute__((visibility("default"))) void run(int* p, int v) { k<<<1, 1>>>(p, v); }
)cu";

// Scenario 2: an operator compiled on its own, standing in for one NVRTC hands
// over as LTO-IR.
const char* k_op_src = R"cu(
extern "C" __device__ int op(int x) { return x * x; }
)cu";

const char* k_caller_src = R"cu(
#include <cuda_runtime.h>
extern "C" __device__ int op(int);   // resolved by the external LTO-IR
__global__ void k(int* p, int v) { *p = op(v); }
extern "C" __attribute__((visibility("default"))) void run(int* p, int v) { k<<<1, 1>>>(p, v); }
)cu";

// Scenario 3: two sources, the second calling a device function from the first.
const char* k_a_src = R"cu(
extern "C" __device__ int scale(int x) { return x * 3; }
)cu";

const char* k_b_src = R"cu(
#include <cuda_runtime.h>
extern "C" __device__ int scale(int);
__global__ void k(int* p, int v) { *p = scale(v); }
extern "C" __attribute__((visibility("default"))) void run(int* p, int v) { k<<<1, 1>>>(p, v); }
)cu";

std::vector<std::string> base_options()
{
  auto config       = hostjit::detectDefaultConfig();
  config.enable_pch = false;
  std::vector<std::string> options;
  config.appendCommandLineArguments(options);
  return options;
}

// Load the library, call run(v), and check what the kernel wrote.
bool load_and_run(const std::string& library, int in, int expect)
{
  int* d = nullptr;
  if (cudaMalloc(&d, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "  cudaMalloc failed\n");
    return false;
  }

  bool ok = false;
  {
    hostjit::DynamicLibrary module;
    if (!module.load(library))
    {
      std::fprintf(stderr, "  load failed: %s\n", module.getLastError().c_str());
      cudaFree(d);
      return false;
    }
    auto run = module.getFunction<void (*)(int*, int)>("run");
    if (!run)
    {
      std::fprintf(stderr, "  entry 'run' not found\n");
      cudaFree(d);
      return false;
    }
    run(d, in);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
      std::fprintf(stderr, "  launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
      cudaFree(d);
      return false;
    }
    int result = -1;
    cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
    ok = result == expect;
    std::printf("  run(%d) -> %d (expected %d)%s\n", in, result, expect, ok ? "" : "  <-- wrong");
  }
  cudaFree(d);
  return ok;
}

// 1 -- source -> shared library.
bool scenario_source_to_shared_library(const std::filesystem::path& dir)
{
  std::printf("[1] source -> shared library\n");
  const std::string library = (dir / "s1.so").string();
  const std::string source  = k_lib_src;

  auto options = base_options();
  options.emplace_back("--shared");
  options.emplace_back("-o");
  options.push_back(library);
  options.emplace_back("lib.cu");
  auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  const cudaccFile lib_cu         = hostjit::detail::make_cudacc_source("lib.cu", source);
  const cudaccFile* const files[] = {&lib_cu};

  hostjit::detail::CudaccOutput out;
  if (cudaccCompile(&out.output, 1, files, static_cast<int>(option_ptrs.size()), option_ptrs.data()) != CUDACC_SUCCESS)
  {
    std::fprintf(stderr, "  compile failed:\n%s\n", out.log().c_str());
    return false;
  }
  if (out.output.output_data != nullptr)
  {
    std::fprintf(stderr, "  a file output must leave output_data null\n");
    return false;
  }
  return load_and_run(library, 41, 42);
}

// 2 -- source + an external operator that only ever exists in memory.
bool scenario_external_operator(const std::filesystem::path& dir)
{
  std::printf("[2] source + external operator (in memory) -> shared library\n");

  // The operator, compiled on its own to LTO-IR. Nothing writes it out.
  std::vector<char> operator_ltoir;
  {
    const std::string source = k_op_src;
    auto options             = base_options();
    options.emplace_back("--ltoir");
    options.emplace_back("op.cu");
    auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

    const cudaccFile op_cu          = hostjit::detail::make_cudacc_source("op.cu", source);
    const cudaccFile* const files[] = {&op_cu};

    hostjit::detail::CudaccOutput out;
    if (cudaccCompile(&out.output, 1, files, static_cast<int>(option_ptrs.size()), option_ptrs.data())
        != CUDACC_SUCCESS)
    {
      std::fprintf(stderr, "  operator compile failed:\n%s\n", out.log().c_str());
      return false;
    }
    operator_ltoir = out.data();
    std::printf("  operator LTO-IR: %zu bytes, never on disk\n", operator_ltoir.size());
    if (operator_ltoir.empty())
    {
      return false;
    }
  }

  const std::string library = (dir / "s2.so").string();
  const std::string source  = k_caller_src;

  auto options = base_options();
  // Sends the kernel to nvJitLink as IR rather than PTX, which is what makes
  // the link a full LTO and gets the operator inlined. The documented option
  // list does not carry this flag: in the product that path is the default, so
  // the scenario's own options stay as the document has them.
  options.emplace_back("--device-nvvm-bypass");
  options.emplace_back("--shared");
  options.emplace_back("-o");
  options.push_back(library);
  options.emplace_back("--ltoir-input");
  options.emplace_back("op.ltoir");
  options.emplace_back("lib.cu");
  auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  const cudaccFile lib_cu         = hostjit::detail::make_cudacc_source("lib.cu", source);
  const cudaccFile op_input       = {"op.ltoir", operator_ltoir.size(), operator_ltoir.data()};
  const cudaccFile* const files[] = {&lib_cu, &op_input};

  hostjit::detail::CudaccOutput out;
  if (cudaccCompile(&out.output, 2, files, static_cast<int>(option_ptrs.size()), option_ptrs.data()) != CUDACC_SUCCESS)
  {
    std::fprintf(stderr, "  compile failed:\n%s\n", out.log().c_str());
    return false;
  }
  return load_and_run(library, 7, 49);
}

// 3 -- two sources, each compiled to an object, linked together.
bool scenario_multi_object(const std::filesystem::path& dir)
{
  std::printf("[3] two sources -> two objects -> one shared library\n");

  struct Unit
  {
    const char* name;
    const char* source;
    const char* object;
  };
  const Unit units[] = {{"a.cu", k_a_src, "a.o"}, {"b.cu", k_b_src, "b.o"}};

  std::vector<std::string> objects;
  for (const auto& unit : units)
  {
    const std::string object = (dir / unit.object).string();
    const std::string source = unit.source;

    auto options = base_options();
    options.emplace_back("-c");
    options.emplace_back("-o");
    options.push_back(object);
    options.emplace_back(unit.name);
    auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

    const cudaccFile source_file    = hostjit::detail::make_cudacc_source(unit.name, source);
    const cudaccFile* const files[] = {&source_file};

    hostjit::detail::CudaccOutput out;
    if (cudaccCompile(&out.output, 1, files, static_cast<int>(option_ptrs.size()), option_ptrs.data())
        != CUDACC_SUCCESS)
    {
      std::fprintf(stderr, "  compiling %s failed:\n%s\n", unit.name, out.log().c_str());
      return false;
    }
    objects.push_back(object);
  }

  const std::string library = (dir / "s3.so").string();

  auto options = base_options();
  options.emplace_back("--shared");
  options.emplace_back("-o");
  options.push_back(library);
  for (const auto& object : objects)
  {
    options.push_back(object);
  }
  auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  // No source in this call, so no files: the objects are read from disk.
  hostjit::detail::CudaccOutput out;
  if (cudaccCompile(&out.output, 0, nullptr, static_cast<int>(option_ptrs.size()), option_ptrs.data())
      != CUDACC_SUCCESS)
  {
    std::fprintf(stderr, "  link failed:\n%s\n", out.log().c_str());
    return false;
  }
  return load_and_run(library, 14, 42);
}

// 4 -- source -> one device artifact, handed back in memory.
bool scenario_device_artifact()
{
  std::printf("[4] source -> device artifact, in memory\n");

  struct Request
  {
    const char* option;
    const char* label;
  };
  const Request requests[] = {{"--ltoir", "LTO-IR"}, {"--cubin", "cubin"}, {"--bitcode", "bitcode"}};

  for (const auto& request : requests)
  {
    const std::string source = k_lib_src;
    auto options             = base_options();
    options.emplace_back(request.option);
    options.emplace_back("lib.cu");
    auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

    const cudaccFile lib_cu         = hostjit::detail::make_cudacc_source("lib.cu", source);
    const cudaccFile* const files[] = {&lib_cu};

    hostjit::detail::CudaccOutput out;
    if (cudaccCompile(&out.output, 1, files, static_cast<int>(option_ptrs.size()), option_ptrs.data())
        != CUDACC_SUCCESS)
    {
      std::fprintf(stderr, "  %s compile failed:\n%s\n", request.label, out.log().c_str());
      return false;
    }
    if (out.output.output_data == nullptr || out.output.output_size == 0)
    {
      std::fprintf(stderr, "  %s came back empty\n", request.label);
      return false;
    }
    std::printf("  %s: %zu bytes\n", request.label, out.output.output_size);
  }
  return true;
}

// The diagnostics live on the output, so a failed compilation still explains
// itself and a successful one can still carry warnings.
bool log_survives_failure()
{
  std::printf("[log] a rejected compilation still reports why\n");
  const std::string source = "__global__ void k(int* p) { *p = nosuchthing; }";

  auto options = base_options();
  options.emplace_back("--ltoir");
  options.emplace_back("bad.cu");
  auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  const cudaccFile bad_cu         = hostjit::detail::make_cudacc_source("bad.cu", source);
  const cudaccFile* const files[] = {&bad_cu};

  hostjit::detail::CudaccOutput out;
  const auto result = cudaccCompile(&out.output, 1, files, static_cast<int>(option_ptrs.size()), option_ptrs.data());
  if (result == CUDACC_SUCCESS)
  {
    std::fprintf(stderr, "  a source with an undeclared identifier compiled\n");
    return false;
  }
  const std::string log = out.log();
  const bool explained  = log.find("nosuchthing") != std::string::npos;
  std::printf("  %s (%s)\n", cudaccGetErrorString(result), explained ? "log names the identifier" : "log unhelpful");
  return explained;
}
} // namespace

int main()
{
  namespace fs = std::filesystem;
  std::printf("cudacc-scenarios -- the product scenarios through the library API\n");

  const fs::path dir = fs::temp_directory_path() / ("cudacc_scenarios_" + std::to_string(std::random_device{}()));
  fs::create_directories(dir);

  const bool ok = scenario_source_to_shared_library(dir) && scenario_external_operator(dir)
               && scenario_multi_object(dir) && scenario_device_artifact() && log_survives_failure();

  fs::remove_all(dir);
  std::printf("cudacc-scenarios: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
