#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace hostjit
{
struct CompilerConfig
{
  std::string cuda_toolkit_path;
  std::string hostjit_include_path; // Path to hostjit include directory (for minimal CUDA runtime)
  std::string clang_headers_path; // Path to Clang's built-in CUDA headers (overrides CLANG_HEADERS_DIR)
  std::string cccl_include_path; // Path to CCCL headers (overrides CCCL_SOURCE_DIR); contains cub/, thrust/, cuda/
  std::string entry_point_name; // Name of the exported entry point function (used for post-link optimization)
  std::string device_pch_path; // Existing device PCH file to load during device compilation
  std::string host_pch_path; // Existing host PCH file to load during host compilation
  std::vector<std::string> include_paths;
  std::vector<std::string> library_paths;
  std::vector<std::string> device_bitcode_files; // Raw LLVM bitcode (magic "BC") linked via LLVM's Linker
  std::vector<std::string> device_ltoir_files; // NVRTC LTOIR; linked at the nvJitLink stage with -lto
  std::unordered_map<std::string, std::string> macro_definitions; // key=macro name, value=macro value (empty for flag
                                                                  // macros)
  std::vector<std::string> extra_clang_args; // Arguments passed directly to Clang via libcudacc's -XClang option
  int sm_version         = 75;
  int optimization_level = 2;
  bool debug             = false;
  bool verbose           = false;
  bool trace_includes    = false; // Show all included headers during compilation (for debugging header search)
  bool keep_artifacts    = false; // Keep compiled artifacts for inspection (PTX, object files, etc.)
  bool enable_pch        = false; // Let CCCL create/load cached PCH files before invoking cudacc
  // Hand the device LLVM IR to nvJitLink directly instead of going through PTX,
  // so external LTO-IR operators are inlined into the kernel by nvJitLink's own
  // NVVM instead of being called across a PTX boundary. Goes in through an entry
  // point of nvJitLink that is internal but supported for libraries inside the
  // toolkit; the version-skew and IR-shape caveats are tracked in CFE-113.
  bool device_nvvm_bypass = false;
  // Where to keep the device IR handed to nvJitLink in bypass mode; empty means
  // a temporary that goes away with the rest of the build directory.
  std::string device_nvvm_ir_out;

  void appendCommandLineArguments(std::vector<std::string>& args) const;
};

// Auto-detect CUDA toolkit and create default configuration
CompilerConfig detectDefaultConfig();

// Validate that the configuration is usable
bool validateConfig(const CompilerConfig& config, std::string* error_message = nullptr);
} // namespace hostjit
