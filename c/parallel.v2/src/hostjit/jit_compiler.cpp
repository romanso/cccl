#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <hostjit/jit_compiler.hpp>

#ifdef _WIN32
#  include <process.h>
#else
#  include <unistd.h>
#endif

namespace
{
static constexpr const char* pch_preamble_source =
  "#include <cuda_runtime.h>\n"
  "#include <cuda/std/iterator>\n"
  "#include <cuda/std/functional>\n"
  "#include <cuda/functional>\n"
  "#include <cub/device/device_adjacent_difference.cuh>\n"
  "#include <cub/device/device_copy.cuh>\n"
  "#include <cub/device/device_find.cuh>\n"
  "#include <cub/device/device_for.cuh>\n"
  "#include <cub/device/device_histogram.cuh>\n"
  "#include <cub/device/device_merge.cuh>\n"
  "#include <cub/device/device_merge_sort.cuh>\n"
  "#include <cub/device/device_partition.cuh>\n"
  "#include <cub/device/device_radix_sort.cuh>\n"
  "#include <cub/device/device_reduce.cuh>\n"
  "#include <cub/device/device_scan.cuh>\n"
  "#include <cub/device/device_segmented_radix_sort.cuh>\n"
  "#include <cub/device/device_segmented_scan.cuh>\n"
  "#include <cub/device/device_segmented_sort.cuh>\n"
  "#include <cub/device/device_select.cuh>\n"
  "#include <cub/device/device_transform.cuh>\n";

std::filesystem::path get_pch_cache_dir()
{
  auto dir = std::filesystem::temp_directory_path() / "hostjit_pch";
  std::filesystem::create_directories(dir);
  return dir;
}

std::string get_pch_path(const std::string& kind, int sm_version)
{
  return (get_pch_cache_dir() / (kind + "_sm" + std::to_string(sm_version) + ".pch")).string();
}

std::string get_pch_source_path(const std::string& kind, int sm_version)
{
  return (get_pch_cache_dir() / (kind + "_sm" + std::to_string(sm_version) + "_preamble.cu")).string();
}

bool create_pch_if_needed(
  hostjit::CompilerConfig config, const std::string& kind_name, std::string& diagnostics, std::string& pch_path)
{
  pch_path = get_pch_path(kind_name, config.sm_version);
  if (std::filesystem::exists(pch_path))
  {
    return true;
  }

  config.enable_pch = false;
  config.device_pch_path.clear();
  config.host_pch_path.clear();

  // Clang records the preamble's path inside the PCH, so the source is named by
  // its cache-stable path rather than by a per-build temporary one.
  const std::string source_path = get_pch_source_path(kind_name, config.sm_version);
  const std::string preamble    = pch_preamble_source;

  std::vector<std::string> options;
  config.appendCommandLineArguments(options);
  options.push_back("--gen-pch=" + kind_name);
  options.push_back("-o");
  options.push_back(pch_path);
  options.push_back(source_path);
  auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  const cudaccFile source_file    = hostjit::detail::make_cudacc_source(source_path.c_str(), preamble);
  const cudaccFile* const input[] = {&source_file};

  hostjit::detail::CudaccOutput out;
  auto pch_result = cudaccCompile(&out.output, 1, input, static_cast<int>(option_ptrs.size()), option_ptrs.data());
  if (pch_result != CUDACC_SUCCESS)
  {
    diagnostics += kind_name + " PCH generation failed: " + out.log();
    diagnostics += "\n";
    pch_path.clear();
    return false;
  }
  return true;
}

hostjit::CompilerConfig prepare_pch_config(const hostjit::CompilerConfig& config, std::string& diagnostics)
{
  hostjit::CompilerConfig prepared = config;
  prepared.device_pch_path.clear();
  prepared.host_pch_path.clear();

  if (!prepared.enable_pch)
  {
    return prepared;
  }

  std::string device_pch_path;
  if (create_pch_if_needed(prepared, "device", diagnostics, device_pch_path))
  {
    prepared.device_pch_path = std::move(device_pch_path);
  }

  std::string host_pch_path;
  if (create_pch_if_needed(prepared, "host", diagnostics, host_pch_path))
  {
    prepared.host_pch_path = std::move(host_pch_path);
  }

  return prepared;
}

bool read_file(const std::string& path, std::vector<char>& out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f)
  {
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}
} // anonymous namespace

namespace hostjit
{
JITCompiler::JITCompiler()
    : config_(detectDefaultConfig())
{}

JITCompiler::JITCompiler(const CompilerConfig& config)
    : config_(config)
{}

JITCompiler::~JITCompiler()
{
  cleanup();
}

bool JITCompiler::compile(const std::string& source_code)
{
  std::string config_error;
  if (!validateConfig(config_, &config_error))
  {
    last_error_ = "Configuration error: " + config_error;
    return false;
  }

  cleanup();

  temp_dir_ = createTempDirectory();
  if (temp_dir_.empty())
  {
    last_error_ = "Failed to create temporary directory";
    return false;
  }

  std::string pch_diagnostics;
  CompilerConfig cudacc_config = prepare_pch_config(config_, pch_diagnostics);
  if (config_.verbose && !pch_diagnostics.empty())
  {
    std::cout << pch_diagnostics;
  }

#ifdef _WIN32
  std::string lib_path = temp_dir_ + "/cuda_code.dll";
#else
  std::string lib_path = temp_dir_ + "/libcuda_code.so";
#endif
  std::string cubin_path = temp_dir_ + "/device.cubin";

  std::vector<std::string> options;
  cudacc_config.appendCommandLineArguments(options);
  options.push_back("--shared");
  options.push_back("-o");
  options.push_back(lib_path);
  // The cubin is a side artifact of the device link; the tests inspect it.
  options.push_back("--cubin-output=" + cubin_path);
  options.push_back("input.cu");
  auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  const cudaccFile source_file    = hostjit::detail::make_cudacc_source("input.cu", source_code);
  const cudaccFile* const input[] = {&source_file};

  hostjit::detail::CudaccOutput out;
  auto compile_result = cudaccCompile(&out.output, 1, input, static_cast<int>(option_ptrs.size()), option_ptrs.data());
  auto compile_log    = out.log();

  if (compile_result != CUDACC_SUCCESS)
  {
    last_error_ = "Compilation failed:\n" + compile_log;
    removeTempDirectory();
    return false;
  }

  cubin_.clear();
  if (!read_file(cubin_path, cubin_))
  {
    last_error_ = "Compilation failed: generated cubin could not be read";
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Compilation diagnostics:\n" << compile_log << "\n";
  }

  if (!library_.load(lib_path))
  {
    last_error_ = "Failed to load library: " + library_.getLastError();
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Successfully loaded library: " << lib_path << "\n";
  }

  last_error_.clear();
  return true;
}

void JITCompiler::cleanup()
{
  library_.unload();

  if (!config_.keep_artifacts)
  {
    removeTempDirectory();
  }

  last_error_.clear();
}

std::string JITCompiler::createTempDirectory()
{
  std::filesystem::path base_tmp_dir;

#ifdef _WIN32
  const char* tmp_dir = std::getenv("TEMP");
  if (!tmp_dir)
  {
    tmp_dir = std::getenv("TMP");
  }
  if (tmp_dir)
  {
    base_tmp_dir = tmp_dir;
  }
  else
  {
    base_tmp_dir = std::filesystem::temp_directory_path();
  }
#else
  const char* tmp_dir = std::getenv("TMPDIR");
  if (tmp_dir)
  {
    base_tmp_dir = tmp_dir;
  }
  else
  {
    base_tmp_dir = "/tmp";
  }
#endif

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 999999);

#ifdef _WIN32
  int pid = _getpid();
#else
  int pid = getpid();
#endif

  for (int attempt = 0; attempt < 10; ++attempt)
  {
    std::string dir_name            = "hostjit_" + std::to_string(pid) + "_" + std::to_string(dis(gen));
    std::filesystem::path full_path = base_tmp_dir / dir_name;

    std::error_code ec;
    if (std::filesystem::create_directories(full_path, ec) && !ec)
    {
      return full_path.string();
    }
  }

  return "";
}

void JITCompiler::removeTempDirectory()
{
  if (temp_dir_.empty())
  {
    return;
  }

  try
  {
    if (std::filesystem::exists(temp_dir_))
    {
      std::filesystem::remove_all(temp_dir_);
    }
  }
  catch (const std::filesystem::filesystem_error& e)
  {
    if (config_.verbose)
    {
      std::cerr << "Warning: Failed to remove temporary directory: " << e.what() << "\n";
    }
  }

  temp_dir_.clear();
}
} // namespace hostjit
