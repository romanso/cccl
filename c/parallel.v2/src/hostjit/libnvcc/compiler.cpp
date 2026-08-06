#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <libnvcc/libnvcc.h>
#include <lld/Common/Driver.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Comdat.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/thread.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Frontend/Offloading/OffloadWrapper.h>
#include <llvm/Frontend/Offloading/Utility.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

// Selective target initialization (X86 for host, NVPTX for device)
extern "C" {
void LLVMInitializeX86TargetInfo();
void LLVMInitializeX86Target();
void LLVMInitializeX86TargetMC();
void LLVMInitializeX86AsmPrinter();
void LLVMInitializeX86AsmParser();
void LLVMInitializeNVPTXTargetInfo();
void LLVMInitializeNVPTXTarget();
void LLVMInitializeNVPTXTargetMC();
void LLVMInitializeNVPTXAsmPrinter();
}

#ifdef _WIN32
LLD_HAS_DRIVER(coff)
#else
LLD_HAS_DRIVER(elf)
#endif

#ifdef _WIN32
#  include <llvm/Object/COFFImportFile.h>
#endif

#include <atomic>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <nvFatbin.h>
#include <nvJitLink.h>

// nvJitLink can take NVVM IR directly, through the same entry point NVRTC uses.
// It is not part of the public header, so it is declared here. Going in this way
// keeps the kernel in IR form, which lets nvJitLink's own NVVM inline external
// LTO-IR operators into it instead of calling them across a PTX boundary.
extern "C" void* __nvJitLinkAPI(int api_kind);

namespace libnvcc
{
namespace
{
constexpr int nvjitlink_api_add_nvvm = 0xc0fd;
using nvJitLinkAddNvvmIRFn = nvJitLinkResult (*)(nvJitLinkHandle, const void*, size_t, const char*);
} // namespace

static std::once_flag llvm_init_flag;

static void initialize_llvm()
{
  std::call_once(llvm_init_flag, [] {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

// Embedding clang as a library bypasses the clang driver's
// runWithSufficientStackSpace guard, so the frontend runs on the caller's stack.
// On Windows the default main-thread stack is only 1 MB, which the deep
// (recursive-descent / template-instantiation) frontend overflows on heavier
// kernels such as radix_sort / segmented_reduce; Linux's 8 MB default hides it.
// Run the frontend on a worker thread sized to match clang's own
// DesiredStackSize (8 MB), which is the proven-sufficient value on Linux.
inline constexpr unsigned kFrontendStackSize = 8u << 20;

template <class Fn>
static bool runWithLargeStack(Fn&& fn)
{
  bool result = false;
  llvm::thread worker(std::optional<unsigned>(kFrontendStackSize), [&] {
    result = fn();
  });
  worker.join();
  return result;
}

struct CompilerOptions
{
  std::string cuda_toolkit_path;
  std::string hostjit_include_path;
  std::string clang_headers_path;
  std::string device_pch_path;
  std::string host_pch_path;
  std::string entry_point_name;
  std::vector<std::string> system_include_paths;
  std::vector<std::string> include_paths;
  std::vector<std::string> library_paths;
  std::vector<std::string> device_bitcode_files;
  std::vector<std::string> device_ltoir_files;
  std::unordered_map<std::string, std::string> macro_definitions;
  std::vector<std::string> extra_clang_args;
  int sm_version         = 75;
  int optimization_level = 2;
  bool debug             = false;
  bool verbose           = false;
  bool trace_includes    = false;
  bool keep_artifacts    = false;
  bool device_nvvm_bypass = false;
  std::string device_nvvm_ir_out;
};

struct CompilationResult
{
  bool success = false;
  std::string object_file_path;
  std::string diagnostics;
};

struct BitcodeResult
{
  bool success = false;
  std::string diagnostics;
};

struct LinkResult
{
  bool success = false;
  std::string library_path;
  std::string diagnostics;
};

static bool pathExists(const std::filesystem::path& path);

static void addDefaultCudaLibraryPath(CompilerOptions& options)
{
  if (!options.cuda_toolkit_path.empty())
  {
    std::filesystem::path lib64_path = std::filesystem::path(options.cuda_toolkit_path) / "lib64";
    std::filesystem::path lib_path   = std::filesystem::path(options.cuda_toolkit_path) / "lib";

    if (pathExists(lib64_path))
    {
      options.library_paths.push_back(lib64_path.string());
    }
    else if (pathExists(lib_path))
    {
      options.library_paths.push_back(lib_path.string());
    }
  }
}

static void setDefaultOptions(CompilerOptions& options)
{
  if (const char* env = std::getenv("CUDA_PATH"))
  {
    options.cuda_toolkit_path = env;
  }
  else if (const char* env = std::getenv("CUDA_HOME"))
  {
    options.cuda_toolkit_path = env;
  }
#ifdef CUDA_TOOLKIT_PATH
  else
  {
    options.cuda_toolkit_path = CUDA_TOOLKIT_PATH;
  }
#endif

  if (const char* env = std::getenv("HOSTJIT_INCLUDE_PATH"))
  {
    options.hostjit_include_path = env;
  }
#ifdef HOSTJIT_INCLUDE_DIR
  else
  {
    options.hostjit_include_path = HOSTJIT_INCLUDE_DIR;
  }
#endif

  if (const char* env = std::getenv("HOSTJIT_CLANG_PATH"))
  {
    options.clang_headers_path = env;
  }
#ifdef CLANG_HEADERS_DIR
  else
  {
    options.clang_headers_path = CLANG_HEADERS_DIR;
  }
#endif
}

static bool pathExists(const std::filesystem::path& path)
{
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

static std::filesystem::path tempDirectoryPath()
{
  std::error_code ec;
  auto path = std::filesystem::temp_directory_path(ec);
  if (!ec)
  {
    return path;
  }
#ifdef _WIN32
  if (const char* env = std::getenv("TEMP"))
  {
    return env;
  }
  if (const char* env = std::getenv("TMP"))
  {
    return env;
  }
#endif
  if (const char* env = std::getenv("TMPDIR"))
  {
    return env;
  }
  return ".";
}

static bool createDirectories(const std::filesystem::path& path, std::string& diagnostics)
{
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec)
  {
    diagnostics += "Failed to create directory " + path.string() + ": " + ec.message() + "\n";
    return false;
  }
  return true;
}

static void removeAll(const std::filesystem::path& path)
{
  std::error_code ec;
  std::filesystem::remove_all(path, ec);
}

template <typename Fn>
static void forEachDirectoryEntry(const std::filesystem::path& dir, Fn&& fn)
{
  std::error_code ec;
  for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
  {
    fn(*it);
  }
}

static bool parseInt(const std::string& value, int& out)
{
  if (value.empty())
  {
    return false;
  }

  int parsed        = 0;
  const char* begin = value.data();
  const char* end   = begin + value.size();
  auto [ptr, ec]    = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end)
  {
    return false;
  }
  out = parsed;
  return true;
}

static bool parseGpuArchitecture(const std::string& value, int& sm)
{
  std::string arch = value;
  if (arch.starts_with("sm_"))
  {
    arch.erase(0, 3);
  }
  return parseInt(arch, sm);
}

static bool parseMacroDefinition(const std::string& value, CompilerOptions& options)
{
  if (value.empty())
  {
    return false;
  }
  auto eq = value.find('=');
  if (eq == std::string::npos)
  {
    options.macro_definitions[value] = "";
  }
  else if (eq == 0)
  {
    return false;
  }
  else
  {
    options.macro_definitions[value.substr(0, eq)] = value.substr(eq + 1);
  }
  return true;
}

static bool parseOptions(int num_options, const char* const* raw_options, CompilerOptions& options, std::string& error)
{
  if (num_options < 0)
  {
    error = "Option count must be non-negative";
    return false;
  }
  if (num_options > 0 && raw_options == nullptr)
  {
    error = "Options array is null";
    return false;
  }

  setDefaultOptions(options);

  auto value_after_equals = [](std::string_view option, std::string_view prefix) -> std::string {
    return std::string(option.substr(prefix.size()));
  };

  for (int i = 0; i < num_options; ++i)
  {
    if (raw_options[i] == nullptr)
    {
      error = "Option string is null";
      return false;
    }

    std::string_view option(raw_options[i]);
    if (option.starts_with("--cuda-path="))
    {
      options.cuda_toolkit_path = value_after_equals(option, "--cuda-path=");
    }
    else if (option.starts_with("--hostjit-include-path="))
    {
      options.hostjit_include_path = value_after_equals(option, "--hostjit-include-path=");
    }
    else if (option.starts_with("--clang-headers-path="))
    {
      options.clang_headers_path = value_after_equals(option, "--clang-headers-path=");
    }
    else if (option.starts_with("--system-include-path="))
    {
      options.system_include_paths.push_back(value_after_equals(option, "--system-include-path="));
    }
    else if (option.starts_with("-isystem") && option.size() > 8)
    {
      options.system_include_paths.emplace_back(option.substr(8));
    }
    else if (option == "-isystem")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-isystem requires an argument";
        return false;
      }
      options.system_include_paths.emplace_back(raw_options[i]);
    }
    else if (option.starts_with("--include-path="))
    {
      options.include_paths.push_back(value_after_equals(option, "--include-path="));
    }
    else if (option.starts_with("-I") && option.size() > 2)
    {
      options.include_paths.emplace_back(option.substr(2));
    }
    else if (option == "-I")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-I requires an argument";
        return false;
      }
      options.include_paths.emplace_back(raw_options[i]);
    }
    else if (option.starts_with("--library-path="))
    {
      options.library_paths.push_back(value_after_equals(option, "--library-path="));
    }
    else if (option.starts_with("-L") && option.size() > 2)
    {
      options.library_paths.emplace_back(option.substr(2));
    }
    else if (option == "-L")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-L requires an argument";
        return false;
      }
      options.library_paths.emplace_back(raw_options[i]);
    }
    else if (option.starts_with("--device-bitcode="))
    {
      options.device_bitcode_files.push_back(value_after_equals(option, "--device-bitcode="));
    }
    else if (option.starts_with("--device-ltoir="))
    {
      options.device_ltoir_files.push_back(value_after_equals(option, "--device-ltoir="));
    }
    else if (option == "--device-nvvm-bypass")
    {
      options.device_nvvm_bypass = true;
    }
    else if (option.starts_with("--device-nvvm-ir-out="))
    {
      options.device_nvvm_ir_out = value_after_equals(option, "--device-nvvm-ir-out=");
    }
    else if (option.starts_with("--define-macro="))
    {
      if (!parseMacroDefinition(value_after_equals(option, "--define-macro="), options))
      {
        error = "Invalid macro definition: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("-D") && option.size() > 2)
    {
      if (!parseMacroDefinition(std::string(option.substr(2)), options))
      {
        error = "Invalid macro definition: " + std::string(option);
        return false;
      }
    }
    else if (option == "-D")
    {
      if (++i >= num_options || raw_options[i] == nullptr || !parseMacroDefinition(raw_options[i], options))
      {
        error = "-D requires a macro definition";
        return false;
      }
    }
    else if (option.starts_with("--gpu-architecture="))
    {
      if (!parseGpuArchitecture(value_after_equals(option, "--gpu-architecture="), options.sm_version))
      {
        error = "Invalid GPU architecture: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("--optimization-level="))
    {
      if (!parseInt(value_after_equals(option, "--optimization-level="), options.optimization_level))
      {
        error = "Invalid optimization level: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("-O") && option.size() > 2)
    {
      if (!parseInt(std::string(option.substr(2)), options.optimization_level))
      {
        error = "Invalid optimization level: " + std::string(option);
        return false;
      }
    }
    else if (option == "--debug")
    {
      options.debug = true;
    }
    else if (option == "--verbose")
    {
      options.verbose = true;
    }
    else if (option == "--trace-includes")
    {
      options.trace_includes = true;
    }
    else if (option == "--keep-artifacts")
    {
      options.keep_artifacts = true;
    }
    else if (option.starts_with("--entry-point="))
    {
      options.entry_point_name = value_after_equals(option, "--entry-point=");
    }
    else if (option.starts_with("--device-pch="))
    {
      options.device_pch_path = value_after_equals(option, "--device-pch=");
    }
    else if (option.starts_with("--host-pch="))
    {
      options.host_pch_path = value_after_equals(option, "--host-pch=");
    }
    else if (option.starts_with("-XClang="))
    {
      options.extra_clang_args.emplace_back(option.substr(8));
    }
    else if (option == "-XClang")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-XClang requires an argument";
        return false;
      }
      options.extra_clang_args.emplace_back(raw_options[i]);
    }
    else
    {
      error = "Unknown option: " + std::string(option);
      return false;
    }
  }

  if (options.library_paths.empty())
  {
    addDefaultCudaLibraryPath(options);
  }

  return true;
}

static bool validateOptions(const CompilerOptions& options, std::string* error_message)
{
  if (options.cuda_toolkit_path.empty())
  {
    if (error_message)
    {
      *error_message = "CUDA toolkit path not found. Please pass --cuda-path or set CUDA_PATH/CUDA_HOME.";
    }
    return false;
  }

  if (!pathExists(options.cuda_toolkit_path))
  {
    if (error_message)
    {
      *error_message = "CUDA toolkit path does not exist: " + options.cuda_toolkit_path;
    }
    return false;
  }

  std::filesystem::path cuda_h = std::filesystem::path(options.cuda_toolkit_path) / "include" / "cuda.h";
  if (!pathExists(cuda_h))
  {
    if (error_message)
    {
      *error_message = "CUDA headers not found at: " + cuda_h.string();
    }
    return false;
  }

  for (const auto& include_path : options.include_paths)
  {
    if (!pathExists(include_path))
    {
      if (error_message)
      {
        *error_message = "Include path does not exist: " + include_path;
      }
      return false;
    }
  }

  for (const auto& include_path : options.system_include_paths)
  {
    if (!pathExists(include_path))
    {
      if (error_message)
      {
        *error_message = "System include path does not exist: " + include_path;
      }
      return false;
    }
  }

  for (const auto& library_path : options.library_paths)
  {
    if (!pathExists(library_path))
    {
      if (error_message)
      {
        *error_message = "Library path does not exist: " + library_path;
      }
      return false;
    }
  }

  for (const auto& bitcode_path : options.device_bitcode_files)
  {
    if (!pathExists(bitcode_path))
    {
      if (error_message)
      {
        *error_message = "Device bitcode path does not exist: " + bitcode_path;
      }
      return false;
    }
  }

  for (const auto& ltoir_path : options.device_ltoir_files)
  {
    if (!pathExists(ltoir_path))
    {
      if (error_message)
      {
        *error_message = "Device LTOIR path does not exist: " + ltoir_path;
      }
      return false;
    }
  }

  if (!options.device_pch_path.empty() && !pathExists(options.device_pch_path))
  {
    if (error_message)
    {
      *error_message = "Device PCH path does not exist: " + options.device_pch_path;
    }
    return false;
  }

  if (!options.host_pch_path.empty() && !pathExists(options.host_pch_path))
  {
    if (error_message)
    {
      *error_message = "Host PCH path does not exist: " + options.host_pch_path;
    }
    return false;
  }

  if (options.sm_version < 30 || options.sm_version > 150)
  {
    if (error_message)
    {
      *error_message = "Invalid SM version: " + std::to_string(options.sm_version) + " (must be between 30 and 150)";
    }
    return false;
  }

  if (options.optimization_level < 0 || options.optimization_level > 3)
  {
    if (error_message)
    {
      *error_message =
        "Invalid optimization level: " + std::to_string(options.optimization_level) + " (must be between 0 and 3)";
    }
    return false;
  }

  return true;
}

static void appendExtraClangArgs(std::vector<std::string>& args, const CompilerOptions& options)
{
  args.insert(args.end(), options.extra_clang_args.begin(), options.extra_clang_args.end());
}

static void appendSystemIncludePaths(std::vector<std::string>& args, const CompilerOptions& options)
{
  for (const auto& include_path : options.system_include_paths)
  {
    args.push_back("-internal-isystem");
    args.push_back(include_path);
  }
}

static void appendIncludePaths(std::vector<std::string>& args, const CompilerOptions& options)
{
  for (const auto& include_path : options.include_paths)
  {
    args.push_back("-I" + include_path);
  }
}

static void appendMacroDefinitions(std::vector<std::string>& args, const CompilerOptions& options)
{
  for (const auto& [macro_name, macro_value] : options.macro_definitions)
  {
    if (macro_value.empty())
    {
      args.push_back("-D" + macro_name);
    }
    else
    {
      args.push_back("-D" + macro_name + "=" + macro_value);
    }
  }
}

#ifdef _WIN32
// Generate a minimal COFF import library for a given DLL.
// This allows linking without requiring the Windows SDK or MSVC .lib files.
// Symbols can be "name" or "name=dllexport" for aliasing.
static bool generateImportLib(
  const std::string& dll_name,
  const std::vector<std::string>& symbols,
  const std::string& output_path,
  bool data_only = false)
{
  std::vector<llvm::object::COFFShortExport> exports;
  for (const auto& sym : symbols)
  {
    llvm::object::COFFShortExport exp;
    auto eq = sym.find('=');
    if (eq != std::string::npos)
    {
      // "atexit=_crt_atexit" means: linker sees "atexit", DLL exports "_crt_atexit"
      exp.Name       = sym.substr(0, eq); // symbol name the linker resolves
      exp.ImportName = sym.substr(eq + 1); // actual DLL export name
    }
    else
    {
      exp.Name = sym;
    }
    exp.Data = data_only;
    exports.push_back(exp);
  }
  auto err = llvm::object::writeImportLibrary(
    dll_name,
    output_path,
    exports,
    llvm::COFF::IMAGE_FILE_MACHINE_AMD64,
    /*MinGW=*/false);
  if (err)
  {
    llvm::consumeError(std::move(err));
    return false;
  }
  return true;
}

// Find the actual DLL filename for cudart (e.g. "cudart64_13.dll") by
// scanning the CUDA toolkit bin directory.
static std::string findCudartDllName(const std::string& cuda_toolkit_path)
{
  namespace fs = std::filesystem;
  for (const auto& subdir : {"bin/x64", "bin"})
  {
    fs::path dir = fs::path(cuda_toolkit_path) / subdir;
    if (!pathExists(dir))
    {
      continue;
    }
    std::string cudart_name;
    forEachDirectoryEntry(dir, [&](const std::filesystem::directory_entry& entry) {
      auto name = entry.path().filename().string();
      if (cudart_name.empty() && name.starts_with("cudart64_") && name.ends_with(".dll"))
      {
        cudart_name = name;
      }
    });
    if (!cudart_name.empty())
    {
      return cudart_name;
    }
  }
  return "cudart64_12.dll"; // fallback
}
#endif

class CompilerImpl
{
public:
  CompilerImpl() {}

  // Write preamble to a persistent file and generate a PCH from it.
  // arg_strings[0] will be replaced with the persistent preamble path.
  //
  // Concurrent builds — other threads, or other processes sharing the
  // persistent cache directory — may generate the same artifacts at the same
  // time. All writes therefore go to a writer-unique temporary path followed
  // by an atomic rename, so readers only ever observe complete files, and
  // since the content is deterministic for a given path, whichever writer
  // lands last is correct. The preamble is additionally left untouched when
  // its content already matches: the PCH records the preamble file's
  // identity, so a needless rewrite would invalidate concurrently generated
  // PCHs.
  bool generatePCH(const std::string& pch_source,
                   const std::string& pch_source_path,
                   const std::string& pch_output_path,
                   std::vector<std::string> arg_strings,
                   std::string& diagnostics)
  {
    static std::atomic<unsigned long> temp_counter{0};
    const std::string temp_suffix =
      ".tmp." + std::to_string(llvm::sys::Process::getProcessId()) + "." + std::to_string(temp_counter++);

    const bool preamble_up_to_date = [&] {
      std::ifstream existing(pch_source_path, std::ios::binary);
      if (!existing)
      {
        return false;
      }
      std::stringstream contents;
      contents << existing.rdbuf();
      return contents.str() == pch_source;
    }();

    if (!preamble_up_to_date)
    {
      const std::string source_temp_path = pch_source_path + temp_suffix;
      {
        std::ofstream f(source_temp_path, std::ios::binary);
        if (!f)
        {
          diagnostics += "Failed to write PCH preamble to " + source_temp_path;
          return false;
        }
        f << pch_source;
      }
      std::error_code rename_error;
      std::filesystem::rename(source_temp_path, pch_source_path, rename_error);
      if (rename_error)
      {
        std::error_code ignored;
        std::filesystem::remove(source_temp_path, ignored);
        diagnostics += "Failed to move PCH preamble into place: " + rename_error.message();
        return false;
      }
    }

    // Replace the source file arg with the persistent path
    arg_strings[0] = pch_source_path;

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    std::string diag_output;
    llvm::raw_string_ostream diag_stream(diag_output);
    clang::DiagnosticOptions diag_opts;
    diag_opts.ShowColors = false;
    auto* diag_printer   = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diag_engine(diag_ids, diag_opts, diag_printer);

    clang::CompilerInstance compiler;
    auto& invocation = compiler.getInvocation();

    if (!clang::CompilerInvocation::CreateFromArgs(invocation, args, diag_engine))
    {
      diag_stream.flush();
      diagnostics += diag_output + "\nFailed to create PCH compiler invocation";
      return false;
    }

    compiler.createDiagnostics(diag_engine.getClient(), false);
    compiler.createFileManager();
    const std::string output_temp_path    = pch_output_path + temp_suffix;
    compiler.getFrontendOpts().OutputFile = output_temp_path;

    clang::GeneratePCHAction pch_action;
    const bool success = runWithLargeStack([&] {
      return compiler.ExecuteAction(pch_action);
    });

    diag_stream.flush();
    diagnostics += diag_output;

    if (!success)
    {
      std::error_code ignored;
      std::filesystem::remove(output_temp_path, ignored);
      return false;
    }

    std::error_code rename_error;
    std::filesystem::rename(output_temp_path, pch_output_path, rename_error);
    if (rename_error)
    {
      std::error_code ignored;
      std::filesystem::remove(output_temp_path, ignored);
      // A concurrent writer may have landed the (identical) PCH first; that
      // counts as success for this builder too.
      if (!std::filesystem::exists(pch_output_path))
      {
        diagnostics += "Failed to move PCH into place: " + rename_error.message();
        return false;
      }
    }
    return true;
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  createVFSWithSource(const std::string& source_code, const std::string& virtual_path)
  {
    auto mem_fs = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    mem_fs->addFile(virtual_path, 0, llvm::MemoryBuffer::getMemBuffer(source_code));

    auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    overlay->pushOverlay(mem_fs);
    return overlay;
  }

  // Write a copy of the device module in the shape the NVVM reader accepts.
  //
  // Two things differ from what Clang emits by default:
  //
  //  * the data layout must not mark address space 15 as non-integral, which
  //    NVVM refuses outright;
  //  * every kernel has to be listed in llvm.used. Nothing inside the device
  //    module refers to a kernel -- the host launches it by name through the
  //    runtime -- so NVVM's link-time optimizer treats kernels as unreachable
  //    and deletes them, and the link then yields a cubin with no code in it.
  //
  // Upstream Clang provides neither, hence the fixup here. A Clang carrying
  // -fnvvm-compatible-device-ir emits both directly and this becomes a no-op.
  // require_kernels is what the link path wants: a device module about to be
  // turned into an image has to contain at least one kernel, and an empty
  // llvm.used there means the cubin will come out empty. A device-only LTO-IR
  // artifact can legitimately hold device functions and no kernel at all.
  std::unique_ptr<llvm::Module>
  makeNvvmCompatibleClone(const llvm::Module& mod, std::string& diagnostics, bool require_kernels = true)
  {
    std::unique_ptr<llvm::Module> clone = llvm::CloneModule(mod);

    std::string layout          = clone->getDataLayoutStr();
    const std::string nonintegral = "-ni:15";
    if (size_t pos = layout.find(nonintegral); pos != std::string::npos)
    {
      layout.erase(pos, nonintegral.size());
      clone->setDataLayout(layout);
    }

    std::vector<llvm::GlobalValue*> kernels;
    for (llvm::Function& fn : *clone)
    {
      if (!fn.isDeclaration() && fn.getCallingConv() == llvm::CallingConv::PTX_Kernel)
      {
        kernels.push_back(&fn);
      }
    }
    if (kernels.empty())
    {
      if (require_kernels)
      {
        diagnostics += "No device kernels found while preparing NVVM IR\n";
        return nullptr;
      }
    }
    else
    {
      llvm::appendToUsed(*clone, kernels);
    }
    return clone;
  }

  bool writeNvvmCompatibleBitcode(const llvm::Module& mod, const std::string& path, std::string& diagnostics)
  {
    std::unique_ptr<llvm::Module> clone = makeNvvmCompatibleClone(mod, diagnostics);
    if (!clone)
    {
      diagnostics += "while preparing NVVM IR: " + path + "\n";
      return false;
    }

    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
    if (ec)
    {
      diagnostics += "Failed to write NVVM IR: " + path + " (" + ec.message() + ")\n";
      return false;
    }
    llvm::WriteBitcodeToFile(*clone, os);
    os.flush();
    return true;
  }

  bool compileDeviceToPTX(
    const std::string& source_code,
    const std::string& input_file,
    const std::string& output_ptx,
    const CompilerOptions& config,
    std::string& diagnostics,
    const std::string& rdc_bitcode_out = "",
    const std::string& nvvm_ir_out     = "")
  {
    std::string temp_dir    = std::filesystem::path(output_ptx).parent_path().string();
    std::string source_file = temp_dir + "/" + input_file;

    std::string resource_dir = CLANG_RESOURCE_DIR;

    // PTX version floor is 7.8. Some generated device code uses features
    // added in PTX 7.6 (e.g. `bmsk`), so older versions can fail to assemble
    // even on sm_75/sm_80.
    int ptx_version = 78;
    if (config.sm_version >= 120)
    {
      ptx_version = 87;
    }
    else if (config.sm_version >= 100)
    {
      ptx_version = 85;
    }
    else if (config.sm_version >= 90)
    {
      ptx_version = 80;
    }

    std::vector<std::string> arg_strings;
    arg_strings.push_back(source_file);
    arg_strings.push_back("-triple");
    arg_strings.push_back("nvptx64-nvidia-cuda");
    arg_strings.push_back("-aux-triple");
#ifdef _WIN32
    arg_strings.push_back("x86_64-pc-windows-msvc");
#else
    arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
    arg_strings.push_back("-S");
    arg_strings.push_back("-aux-target-cpu");
    arg_strings.push_back("x86-64");
    arg_strings.push_back("-fcuda-is-device");
    arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
    arg_strings.push_back("-fms-compatibility");
    arg_strings.push_back("-fms-compatibility-version=19.40");
#else
    arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
    arg_strings.push_back("-mlink-builtin-bitcode");
    arg_strings.push_back(config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc");
    arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
    arg_strings.push_back("-target-cpu");
    arg_strings.push_back("sm_" + std::to_string(config.sm_version));
    arg_strings.push_back("-target-feature");
    arg_strings.push_back("+ptx" + std::to_string(ptx_version));
    arg_strings.push_back("-resource-dir");
    arg_strings.push_back(resource_dir);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/stubs");
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(
      config.clang_headers_path.empty() ? std::string(CLANG_HEADERS_DIR) : config.clang_headers_path);
    appendSystemIncludePaths(arg_strings, config);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    appendIncludePaths(arg_strings, config);

    arg_strings.push_back("-D__HOSTJIT_DEVICE_COMPILATION__=1");
    arg_strings.push_back("-DNDEBUG");

    std::vector<std::string> bitcode_files_to_link = config.device_bitcode_files;

    appendMacroDefinitions(arg_strings, config);

    arg_strings.push_back("-fdeprecated-macro");
    arg_strings.push_back("--offload-new-driver");
    arg_strings.push_back("-fskip-odr-check-in-gmf");
    arg_strings.push_back("-fcxx-exceptions");
    arg_strings.push_back("-fexceptions");
    // RDC device codegen: emit relocatable device code so device symbols can be
    // resolved across translation units when the final link combines every TU's
    // device code into one fatbin.
    arg_strings.push_back("-fgpu-rdc");
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-std=c++17");

    if (config.trace_includes)
    {
      arg_strings.push_back("-H");
    }

    appendExtraClangArgs(arg_strings, config);
    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    std::string device_pch_path = config.device_pch_path;

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    if (config.verbose)
    {
      diagnostics += "Device args: ";
      for (const auto& arg : arg_strings)
      {
        diagnostics += arg + " ";
      }
      diagnostics += "\n";
    }

    std::string diag_output;
    llvm::raw_string_ostream diag_stream(diag_output);

    clang::DiagnosticOptions diag_opts;
    diag_opts.ShowColors                       = false;
    clang::TextDiagnosticPrinter* diag_printer = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diag_engine(diag_ids, diag_opts, diag_printer);

    clang::CompilerInstance compiler;
    auto& invocation = compiler.getInvocation();

    if (!clang::CompilerInvocation::CreateFromArgs(invocation, args, diag_engine))
    {
      diag_stream.flush();
      diagnostics += diag_output;
      diagnostics += "\nFailed to create device compiler invocation";
      return false;
    }

    // --- PCH: load cached device PCH ---
    if (!device_pch_path.empty() && pathExists(device_pch_path))
    {
      invocation.getPreprocessorOpts().ImplicitPCHInclude = device_pch_path;
    }

    auto vfs = createVFSWithSource(source_code, source_file);
    compiler.createDiagnostics(diag_engine.getClient(), false);
    compiler.setVirtualFileSystem(vfs);
    compiler.createFileManager();
    compiler.getFrontendOpts().OutputFile = output_ptx;

    if (config.trace_includes)
    {
      diagnostics += "\n=== Device Header Search Paths ===\n";
      const auto& hso = invocation.getHeaderSearchOpts();
      for (const auto& entry : hso.UserEntries)
      {
        diagnostics += "  " + entry.Path + "\n";
      }
      diagnostics += "=== End Header Search Paths ===\n\n";
    }

    llvm::LLVMContext llvm_context;

    clang::EmitLLVMOnlyAction emit_llvm_action(&llvm_context);
    bool success = runWithLargeStack([&] {
      return compiler.ExecuteAction(emit_llvm_action);
    });

    if (config.trace_includes && compiler.hasSourceManager())
    {
      diagnostics += "\n=== Device Included Files ===\n";
      auto& sm = compiler.getSourceManager();
      for (auto it = sm.fileinfo_begin(); it != sm.fileinfo_end(); ++it)
      {
        diagnostics += "  " + it->first.getName().str() + "\n";
      }
      diagnostics += "=== End Included Files ===\n\n";
    }

    if (success)
    {
      std::unique_ptr<llvm::Module> mod = emit_llvm_action.takeModule();
      if (mod)
      {
        for (const auto& bc_file : bitcode_files_to_link)
        {
          llvm::SMDiagnostic err;
          auto bc_mod = llvm::parseIRFile(bc_file, err, llvm_context);
          if (bc_mod)
          {
            if (llvm::Linker::linkModules(*mod, std::move(bc_mod)))
            {
              diagnostics += "Failed to link bitcode: " + bc_file + "\n";
              success = false;
              break;
            }
          }
          else
          {
            std::string err_msg;
            llvm::raw_string_ostream err_stream(err_msg);
            err.print("hostjit", err_stream);
            diagnostics += "Failed to parse bitcode: " + bc_file + "\n" + err_msg + "\n";
            success = false;
            break;
          }
        }

        // Re-link libdevice to resolve any new references (e.g. __nv_pow)
        // introduced by the extra bitcode modules.
        if (success && !bitcode_files_to_link.empty())
        {
          std::string libdevice_path = config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc";
          llvm::SMDiagnostic err;
          auto libdevice = llvm::parseIRFile(libdevice_path, err, llvm_context);
          if (libdevice)
          {
            // Use AppendToUsed to avoid internalization issues
            llvm::Linker::linkModules(*mod, std::move(libdevice), llvm::Linker::LinkOnlyNeeded);
          }
        }

        if (success && !rdc_bitcode_out.empty())
        {
          // RDC: persist this TU's device module so the final link can combine
          // every TU's device code into a single fatbin.
          std::error_code bec;
          llvm::raw_fd_ostream bos(rdc_bitcode_out, bec, llvm::sys::fs::OF_None);
          if (bec)
          {
            diagnostics += "Failed to write RDC device bitcode: " + rdc_bitcode_out + "\n";
          }
          else
          {
            llvm::WriteBitcodeToFile(*mod, bos);
            bos.flush();
          }
        }

        if (success && !nvvm_ir_out.empty())
        {
          success = writeNvvmCompatibleBitcode(*mod, nvvm_ir_out, diagnostics);
        }

        if (success)
        {
          std::string err_str;
          const llvm::Target* target = llvm::TargetRegistry::lookupTarget(mod->getTargetTriple(), err_str);
          if (target)
          {
            llvm::TargetOptions opt;
            auto tm = target->createTargetMachine(
              mod->getTargetTriple(),
              "sm_" + std::to_string(config.sm_version),
              "+ptx" + std::to_string(ptx_version),
              opt,
              llvm::Reloc::PIC_);
            if (tm)
            {
              mod->setDataLayout(tm->createDataLayout());

              // Run optimization passes after linking to inline user-provided
              // operations (from bitcode or embedded C++ source).
              if (!config.entry_point_name.empty())
              {
                // Internalize all functions except the entry point and
                // GPU kernels, so the optimizer can inline the linked
                // bitcode functions.
                for (auto& F : *mod)
                {
                  if (!F.isDeclaration() && F.getLinkage() == llvm::GlobalValue::ExternalLinkage
                      && F.getName() != config.entry_point_name && F.getCallingConv() != llvm::CallingConv::PTX_Kernel)
                  {
                    F.setLinkage(llvm::GlobalValue::InternalLinkage);
                    // Remove attributes that conflict with inlining
                    F.removeFnAttr(llvm::Attribute::NoInline);
                    F.removeFnAttr(llvm::Attribute::OptimizeNone);
                    F.addFnAttr(llvm::Attribute::AlwaysInline);
                  }
                }

                llvm::OptimizationLevel opt_level;
                switch (config.optimization_level)
                {
                  case 0:
                    opt_level = llvm::OptimizationLevel::O0;
                    break;
                  case 1:
                    opt_level = llvm::OptimizationLevel::O1;
                    break;
                  case 3:
                    opt_level = llvm::OptimizationLevel::O3;
                    break;
                  default:
                    opt_level = llvm::OptimizationLevel::O2;
                    break;
                }

                // Raise LLVM's loop-unroll thresholds (once) so small,
                // constant-trip-count loops in linked bitcode get fully
                // unrolled. Without full unroll the backing alloca keeps a
                // dynamic index, SROA can't promote it, and it lands in local
                // memory (a per-thread stack frame + LDL/STL traffic). ptxas
                // performs this promotion on the LTO path; the LLVM-NVPTX path
                // needs full-unroll-then-SROA at the IR level.
                static const bool unroll_tuned = [] {
                  auto& opts   = llvm::cl::getRegisteredOptions();
                  auto set_opt = [&](llvm::StringRef name, llvm::StringRef value) {
                    auto it = opts.find(name);
                    if (it != opts.end())
                    {
                      it->second->addOccurrence(0, name, value);
                    }
                  };
                  set_opt("unroll-threshold", "4000");
                  set_opt("unroll-full-max-count", "1024");
                  set_opt("unroll-max-upperbound", "1024");
                  return true;
                }();
                (void) unroll_tuned;

                llvm::LoopAnalysisManager LAM;
                llvm::FunctionAnalysisManager FAM;
                llvm::CGSCCAnalysisManager CGAM;
                llvm::ModuleAnalysisManager MAM;

                llvm::PassBuilder PB(tm);
                PB.registerModuleAnalyses(MAM);
                PB.registerCGSCCAnalyses(CGAM);
                PB.registerFunctionAnalyses(FAM);
                PB.registerLoopAnalyses(LAM);
                PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

                auto MPM = PB.buildPerModuleDefaultPipeline(opt_level);
                MPM.run(*mod, MAM);

                // Second optimization round with fresh analyses: now that the
                // op's loops are fully unrolled (constant indices), the early
                // SROA in the pipeline promotes the local arrays to registers.
                llvm::LoopAnalysisManager LAM2;
                llvm::FunctionAnalysisManager FAM2;
                llvm::CGSCCAnalysisManager CGAM2;
                llvm::ModuleAnalysisManager MAM2;

                llvm::PassBuilder PB2(tm);
                PB2.registerModuleAnalyses(MAM2);
                PB2.registerCGSCCAnalyses(CGAM2);
                PB2.registerFunctionAnalyses(FAM2);
                PB2.registerLoopAnalyses(LAM2);
                PB2.crossRegisterProxies(LAM2, FAM2, CGAM2, MAM2);

                auto MPM2 = PB2.buildPerModuleDefaultPipeline(opt_level);
                MPM2.run(*mod, MAM2);
              }

              std::error_code EC;
              llvm::raw_fd_ostream dest(output_ptx, EC);
              if (!EC)
              {
                llvm::legacy::PassManager pass;
                tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::AssemblyFile);
                pass.run(*mod);
                dest.flush();

                // Debug: when LIBNVCC_DUMP_DIR is set, dump the optimized IR
                // and the PTX fed to ptxas, keyed by entry point name. Lets us
                // inspect codegen (register pressure, launch bounds) post-inline.
                if (const char* dump_dir = std::getenv("LIBNVCC_DUMP_DIR"))
                {
                  std::error_code dec;
                  std::filesystem::create_directories(dump_dir, dec);
                  const std::string base =
                    config.entry_point_name.empty() ? std::string("kernel") : config.entry_point_name;
                  const std::string stem = (std::filesystem::path(dump_dir) / base).string();
                  llvm::raw_fd_ostream ll_os(stem + ".opt.ll", dec);
                  if (!dec)
                  {
                    mod->print(ll_os, nullptr);
                  }
                  std::error_code cec;
                  std::filesystem::copy_file(
                    output_ptx, stem + ".ptx", std::filesystem::copy_options::overwrite_existing, cec);
                  llvm::errs() << "[hostjit] dumped " << stem << ".opt.ll and " << stem << ".ptx\n";
                }
              }
              else
              {
                diagnostics += "Failed to open output file: " + output_ptx + "\n";
                success = false;
              }
            }
            else
            {
              diagnostics += "Failed to create target machine\n";
              success = false;
            }
          }
          else
          {
            diagnostics += "Failed to lookup target: " + err_str + "\n";
            success = false;
          }
        }
      }
    }

    diag_stream.flush();
    diagnostics += diag_output;

    return success;
  }

  // Run the device half of the compilation and hand back the module. Shared by
  // the device-only outputs (bitcode and LTO-IR), which differ only in what they
  // do with it; the object and shared-library paths go through
  // compileDeviceToPTX instead. The module belongs to llvm_context, so the
  // caller has to keep that alive for as long as it holds the module.
  std::unique_ptr<llvm::Module> emitDeviceModule(
    const std::string& source_code,
    const std::string& input_name,
    const CompilerOptions& config,
    llvm::LLVMContext& llvm_context,
    std::string& diagnostics)
  {
    std::string error_msg;
    if (!validateOptions(config, &error_msg))
    {
      diagnostics = "Configuration error: " + error_msg;
      return nullptr;
    }

    initialize_llvm();

    std::string temp_dir =
      (tempDirectoryPath() / ("hostjit_bc_" + std::to_string(reinterpret_cast<uintptr_t>(this)))).string();
    if (!createDirectories(temp_dir, diagnostics))
    {
      return nullptr;
    }

    std::string input_file   = input_name.empty() ? std::string("input.cu") : input_name;
    std::string source_file  = temp_dir + "/" + input_file;
    std::string resource_dir = CLANG_RESOURCE_DIR;

    // PTX version floor is 7.8. Some generated device code uses features
    // added in PTX 7.6 (e.g. `bmsk`), so older versions can fail to assemble
    // even on sm_75/sm_80.
    int ptx_version = 78;
    if (config.sm_version >= 120)
    {
      ptx_version = 87;
    }
    else if (config.sm_version >= 100)
    {
      ptx_version = 85;
    }
    else if (config.sm_version >= 90)
    {
      ptx_version = 80;
    }

    std::vector<std::string> arg_strings;
    arg_strings.push_back(source_file);
    arg_strings.push_back("-triple");
    arg_strings.push_back("nvptx64-nvidia-cuda");
    arg_strings.push_back("-aux-triple");
#ifdef _WIN32
    arg_strings.push_back("x86_64-pc-windows-msvc");
#else
    arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
    arg_strings.push_back("-S");
    arg_strings.push_back("-aux-target-cpu");
    arg_strings.push_back("x86-64");
    arg_strings.push_back("-fcuda-is-device");
    arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
    arg_strings.push_back("-fms-compatibility");
    arg_strings.push_back("-fms-compatibility-version=19.40");
#else
    arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
    arg_strings.push_back("-mlink-builtin-bitcode");
    arg_strings.push_back(config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc");
    arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
    arg_strings.push_back("-target-cpu");
    arg_strings.push_back("sm_" + std::to_string(config.sm_version));
    arg_strings.push_back("-target-feature");
    arg_strings.push_back("+ptx" + std::to_string(ptx_version));
    arg_strings.push_back("-resource-dir");
    arg_strings.push_back(resource_dir);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/stubs");
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(
      config.clang_headers_path.empty() ? std::string(CLANG_HEADERS_DIR) : config.clang_headers_path);
    appendSystemIncludePaths(arg_strings, config);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    appendIncludePaths(arg_strings, config);

    arg_strings.push_back("-D__HOSTJIT_DEVICE_COMPILATION__=1");
    arg_strings.push_back("-DNDEBUG");

    appendMacroDefinitions(arg_strings, config);

    arg_strings.push_back("-fdeprecated-macro");
    arg_strings.push_back("-fcxx-exceptions");
    arg_strings.push_back("-fexceptions");
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-Wno-c++11-narrowing");
    arg_strings.push_back("-std=c++17");
    appendExtraClangArgs(arg_strings, config);
    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    std::string diag_output;
    llvm::raw_string_ostream diag_stream(diag_output);

    clang::DiagnosticOptions diag_opts;
    diag_opts.ShowColors                       = false;
    clang::TextDiagnosticPrinter* diag_printer = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diag_engine(diag_ids, diag_opts, diag_printer);

    clang::CompilerInstance compiler;
    auto& invocation = compiler.getInvocation();

    if (!clang::CompilerInvocation::CreateFromArgs(invocation, args, diag_engine))
    {
      diag_stream.flush();
      diagnostics = diag_output + "\nFailed to create compiler invocation";
      removeAll(temp_dir);
      return nullptr;
    }

    if (!config.device_pch_path.empty())
    {
      invocation.getPreprocessorOpts().ImplicitPCHInclude = config.device_pch_path;
    }

    auto vfs = createVFSWithSource(source_code, source_file);
    compiler.createDiagnostics(diag_engine.getClient(), false);
    compiler.setVirtualFileSystem(vfs);
    compiler.createFileManager();

    clang::EmitLLVMOnlyAction emit_llvm_action(&llvm_context);
    bool success = runWithLargeStack([&] {
      return compiler.ExecuteAction(emit_llvm_action);
    });

    std::unique_ptr<llvm::Module> mod;
    if (success)
    {
      mod = emit_llvm_action.takeModule();
      if (!mod)
      {
        diagnostics = "Failed to get LLVM module";
      }
    }

    diag_stream.flush();
    diagnostics += diag_output;
    if (!config.keep_artifacts)
    {
      removeAll(temp_dir);
    }
    return mod;
  }

  BitcodeResult compileToDeviceBitcode(
    const std::string& source_code,
    const std::string& input_name,
    const std::string& output_bitcode_path,
    const CompilerOptions& config)
  {
    BitcodeResult result;
    result.success = false;

    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> mod = emitDeviceModule(source_code, input_name, config, llvm_context, result.diagnostics);
    if (!mod)
    {
      return result;
    }

    std::error_code ec;
    llvm::raw_fd_ostream os(output_bitcode_path, ec, llvm::sys::fs::OF_None);
    if (ec)
    {
      result.diagnostics = "Failed to open bitcode output file: " + output_bitcode_path + "\n" + result.diagnostics;
      return result;
    }
    llvm::WriteBitcodeToFile(*mod, os);
    os.flush();
    if (os.has_error())
    {
      result.diagnostics = "Failed to write bitcode output file: " + output_bitcode_path + "\n" + result.diagnostics;
      return result;
    }

    result.success = true;
    return result;
  }

  // Device-only compile whose output is LTO-IR: no host code, no fatbin, and no
  // final device link. The module goes to nvJitLink as NVVM IR through the same
  // entry point the bypass link path uses, but the link is relocatable, so
  // nvJitLink stops at the LTO-IR container instead of running on into ptxas
  // (which would fail on any unresolved external the caller means to link later).
  BitcodeResult compileToDeviceLTOIR(
    const std::string& source_code,
    const std::string& input_name,
    const std::string& output_ltoir_path,
    const CompilerOptions& config)
  {
    BitcodeResult result;
    result.success = false;

    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> mod = emitDeviceModule(source_code, input_name, config, llvm_context, result.diagnostics);
    if (!mod)
    {
      return result;
    }

    std::unique_ptr<llvm::Module> nvvm_mod = makeNvvmCompatibleClone(*mod, result.diagnostics, /*require_kernels=*/false);
    if (!nvvm_mod)
    {
      return result;
    }

    llvm::SmallVector<char, 0> nvvm_ir;
    {
      llvm::raw_svector_ostream os(nvvm_ir);
      llvm::WriteBitcodeToFile(*nvvm_mod, os);
    }

    auto add_nvvm_ir = reinterpret_cast<nvJitLinkAddNvvmIRFn>(__nvJitLinkAPI(nvjitlink_api_add_nvvm));
    if (!add_nvvm_ir)
    {
      result.diagnostics += "\nnvJitLink does not provide the NVVM IR entry point";
      return result;
    }

    std::string arch_opt  = "-arch=sm_" + std::to_string(config.sm_version);
    std::string opt_level = "-O" + std::to_string(config.optimization_level >= 1 ? 3 : 0);
    const char* jitlink_options[] = {arch_opt.c_str(), opt_level.c_str(), "-lto", "-r"};

    nvJitLinkHandle jitlink_handle = nullptr;
    nvJitLinkResult jlr            = nvJitLinkCreate(&jitlink_handle, 4, jitlink_options);
    if (jlr != NVJITLINK_SUCCESS)
    {
      result.diagnostics += "\nnvJitLinkCreate failed (error " + std::to_string(static_cast<int>(jlr)) + ")";
      return result;
    }

    auto fail = [&](const char* what) {
      size_t log_size = 0;
      nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
      if (log_size > 1)
      {
        std::string log(log_size, '\0');
        nvJitLinkGetErrorLog(jitlink_handle, log.data());
        result.diagnostics += "\n" + log;
      }
      result.diagnostics += std::string("\n") + what;
      nvJitLinkDestroy(&jitlink_handle);
      return result;
    };

    jlr = add_nvvm_ir(jitlink_handle, nvvm_ir.data(), nvvm_ir.size(), "device.nvvm.bc");
    if (jlr != NVJITLINK_SUCCESS)
    {
      return fail("nvJitLink NVVM IR input failed");
    }

    jlr = nvJitLinkComplete(jitlink_handle);
    if (jlr != NVJITLINK_SUCCESS)
    {
      return fail("nvJitLinkComplete failed");
    }

    size_t ltoir_size = 0;
    jlr               = nvJitLinkGetLinkedLTOIRSize(jitlink_handle, &ltoir_size);
    if (jlr != NVJITLINK_SUCCESS || ltoir_size == 0)
    {
      return fail("nvJitLinkGetLinkedLTOIRSize failed");
    }
    std::vector<char> ltoir(ltoir_size);
    jlr = nvJitLinkGetLinkedLTOIR(jitlink_handle, ltoir.data());
    if (jlr != NVJITLINK_SUCCESS)
    {
      return fail("nvJitLinkGetLinkedLTOIR failed");
    }
    nvJitLinkDestroy(&jitlink_handle);

    std::ofstream out(output_ltoir_path, std::ios::binary);
    out.write(ltoir.data(), static_cast<std::streamsize>(ltoir.size()));
    if (!out)
    {
      result.diagnostics += "\nFailed to write LTO-IR output file: " + output_ltoir_path;
      return result;
    }

    result.success = true;
    return result;
  }

  bool compileHostCode(
    const std::string& source_code,
    const std::string& input_file,
    const std::string& fatbin_path,
    const std::string& output_obj,
    const CompilerOptions& config,
    std::string& diagnostics)
  {
    std::string temp_dir    = std::filesystem::path(output_obj).parent_path().string();
    std::string source_file = temp_dir + "/host_" + input_file;

    std::string resource_dir = CLANG_RESOURCE_DIR;

    std::vector<std::string> arg_strings;
    arg_strings.push_back(source_file);
    arg_strings.push_back("-triple");
#ifdef _WIN32
    arg_strings.push_back("x86_64-pc-windows-msvc");
#else
    arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
    arg_strings.push_back("-aux-triple");
    arg_strings.push_back("nvptx64-nvidia-cuda");
    arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
    arg_strings.push_back("-emit-obj");
    arg_strings.push_back("-target-cpu");
    arg_strings.push_back("x86-64");
    arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
    arg_strings.push_back("-fms-compatibility");
    arg_strings.push_back("-fms-compatibility-version=19.40");
    // We do not have access to the windows CRT, so the guard support that
    // threadsafe statics need (_tls_index, _Init_thread_epoch, ...) is
    // unavailable and must be disabled. Generated code IS invoked from
    // multiple threads: first_call_gate (util/first_call_gate.h) serializes
    // the first call into each generated function so its function-local
    // statics initialize race-free despite this flag.
    arg_strings.push_back("-fno-threadsafe-statics");
#else
    arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
    arg_strings.push_back("-mrelocation-model");
    arg_strings.push_back("pic");
    arg_strings.push_back("-pic-level");
    arg_strings.push_back("2");
    arg_strings.push_back("-resource-dir");
    arg_strings.push_back(resource_dir);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/stubs");
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(
      config.clang_headers_path.empty() ? std::string(CLANG_HEADERS_DIR) : config.clang_headers_path);
    appendSystemIncludePaths(arg_strings, config);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    appendIncludePaths(arg_strings, config);

    arg_strings.push_back("-DNDEBUG");

    appendMacroDefinitions(arg_strings, config);

    arg_strings.push_back("-fdeprecated-macro");
    arg_strings.push_back("--offload-new-driver");
    arg_strings.push_back("-fskip-odr-check-in-gmf");
    // RDC host codegen: defer device registration to the final link (one fatbin,
    // one registration object) instead of baking a fatbin into every object.
    arg_strings.push_back("-fgpu-rdc");
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-std=c++17");

    if (config.trace_includes)
    {
      arg_strings.push_back("-H");
    }

    appendExtraClangArgs(arg_strings, config);
    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    std::string host_pch_path = config.host_pch_path;

    // RDC: the device image is not baked into each object. The final link
    // device-links every TU's device code into one fatbin and registers it via a
    // generated registration object, so no per-object -fcuda-include-gpubinary
    // here (that would produce a per-TU registration ctor that collides across
    // objects). fatbin_path is unused on this path.
    (void) fatbin_path;

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    if (config.verbose)
    {
      diagnostics += "Host args: ";
      for (const auto& arg : arg_strings)
      {
        diagnostics += arg + " ";
      }
      diagnostics += "\n";
    }

    std::string diag_output;
    llvm::raw_string_ostream diag_stream(diag_output);

    clang::DiagnosticOptions diag_opts;
    diag_opts.ShowColors                       = false;
    clang::TextDiagnosticPrinter* diag_printer = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diag_engine(diag_ids, diag_opts, diag_printer);

    clang::CompilerInstance compiler;
    auto& invocation = compiler.getInvocation();

    if (!clang::CompilerInvocation::CreateFromArgs(invocation, args, diag_engine))
    {
      diag_stream.flush();
      diagnostics += diag_output;
      diagnostics += "\nFailed to create host compiler invocation";
      return false;
    }

    // --- PCH: load cached host PCH ---
    if (!host_pch_path.empty() && pathExists(host_pch_path))
    {
      invocation.getPreprocessorOpts().ImplicitPCHInclude = host_pch_path;
    }

    auto vfs = createVFSWithSource(source_code, source_file);
    compiler.createDiagnostics(diag_engine.getClient(), false);
    compiler.setVirtualFileSystem(vfs);
    compiler.createFileManager();
    compiler.getFrontendOpts().OutputFile = output_obj;

    if (config.trace_includes)
    {
      diagnostics += "\n=== Host Header Search Paths ===\n";
      const auto& hso = invocation.getHeaderSearchOpts();
      for (const auto& entry : hso.UserEntries)
      {
        diagnostics += "  " + entry.Path + "\n";
      }
      diagnostics += "=== End Header Search Paths ===\n\n";
    }

#ifdef _WIN32
    // On Windows emit the host object ourselves (see below) so we can weaken
    // `_fltused` first; on other platforms keep clang's direct object emission.
    llvm::LLVMContext host_context;
    clang::EmitLLVMOnlyAction emit_action(&host_context);
#else
    clang::EmitObjAction emit_action;
#endif
    bool success = runWithLargeStack([&] { return compiler.ExecuteAction(emit_action); });

    if (config.trace_includes && compiler.hasSourceManager())
    {
      diagnostics += "\n=== Host Included Files ===\n";
      auto& sm = compiler.getSourceManager();
      for (auto it = sm.fileinfo_begin(); it != sm.fileinfo_end(); ++it)
      {
        diagnostics += "  " + it->first.getName().str() + "\n";
      }
      diagnostics += "=== End Included Files ===\n\n";
    }

#ifdef _WIN32
    // MSVC/clang emits `_fltused` (a benign FP-usage marker) as a strong external
    // definition in every host object. Linking several host objects into one
    // shared library (RDC multi-TU) then fails with "duplicate symbol: _fltused".
    // Emit it as a COMDAT weak symbol so the COFF linker folds the duplicates.
    // Manual object emission via addPassesToEmitFile mirrors
    // buildRdcRegistrationObject and preserves the llvm_offload_entries section.
    if (success)
    {
      std::unique_ptr<llvm::Module> mod = emit_action.takeModule();
      if (!mod)
      {
        diagnostics += "\nHost compile produced no module";
        success = false;
      }
      else
      {
        // Fold benign per-TU duplicate definitions across host objects. `_fltused`
        // (an FP-usage marker) is emitted as a strong external; the wrapper's
        // `atexit` shim is weak but may lack a COMDAT. Give each a weak COMDAT so
        // the COFF linker merges the copies instead of erroring with
        // "duplicate symbol" when several host objects link into one library.
        for (const char* sym : {"_fltused", "atexit"})
        {
          auto* go = llvm::dyn_cast_or_null<llvm::GlobalObject>(mod->getNamedValue(sym));
          if (!go || go->isDeclaration())
          {
            continue;
          }
          if (go->getLinkage() == llvm::GlobalValue::ExternalLinkage)
          {
            go->setLinkage(llvm::GlobalValue::WeakODRLinkage);
          }
          if (!go->getComdat())
          {
            llvm::Comdat* c = mod->getOrInsertComdat(sym);
            c->setSelectionKind(llvm::Comdat::Any);
            go->setComdat(c);
          }
        }

        std::string err_str;
        const llvm::Target* htarget = llvm::TargetRegistry::lookupTarget(mod->getTargetTriple(), err_str);
        if (!htarget)
        {
          diagnostics += "\nHost target not found: " + err_str;
          success = false;
        }
        else
        {
          llvm::TargetOptions hopt;
          std::unique_ptr<llvm::TargetMachine> htm(
            htarget->createTargetMachine(mod->getTargetTriple(), "x86-64", "", hopt, llvm::Reloc::PIC_));
          if (!htm)
          {
            diagnostics += "\nCould not create host target machine";
            success = false;
          }
          else
          {
            mod->setDataLayout(htm->createDataLayout());
            std::error_code ec;
            llvm::raw_fd_ostream ro(output_obj, ec, llvm::sys::fs::OF_None);
            if (ec)
            {
              diagnostics += "\nCannot open host object " + output_obj + ": " + ec.message();
              success = false;
            }
            else
            {
              llvm::legacy::PassManager pm;
              if (htm->addPassesToEmitFile(pm, ro, nullptr, llvm::CodeGenFileType::ObjectFile))
              {
                diagnostics += "\nHost backend cannot emit object";
                success = false;
              }
              else
              {
                pm.run(*mod);
              }
              ro.flush();
            }
          }
        }
      }
    }
#endif

    diag_stream.flush();
    diagnostics += diag_output;

    return success;
  }

  CompilationResult compileToObject(
    const std::string& source_code,
    const std::string& input_name,
    const std::string& output_path,
    const std::string& output_cubin_path,
    const CompilerOptions& config)
  {
    CompilationResult result;
    result.success          = false;
    result.object_file_path = output_path;

    std::string error_msg;
    if (!validateOptions(config, &error_msg))
    {
      result.diagnostics = "Configuration error: " + error_msg;
      return result;
    }

    initialize_llvm();

    std::string temp_dir =
      (tempDirectoryPath() / ("hostjit_" + std::to_string(reinterpret_cast<uintptr_t>(this)))).string();
    if (!createDirectories(temp_dir, result.diagnostics))
    {
      return result;
    }

    std::string input_file  = input_name.empty() ? std::string("input.cu") : input_name;
    std::string ptx_file    = temp_dir + "/device.ptx";
    std::string fatbin_file = temp_dir + "/device.fatbin";

    if (config.verbose)
    {
      result.diagnostics += "=== Device compilation ===\n";
    }

    // RDC: persist this TU's device module to a bitcode sidecar so the final
    // link can combine every TU's device code into a single fatbin.
    const std::string rdc_bitcode_out = output_path + ".dev.bc";

    // Bypass mode: keep the device code in IR form for nvJitLink as well.
    const std::string nvvm_ir_file =
      !config.device_nvvm_bypass
        ? std::string()
        : (config.device_nvvm_ir_out.empty() ? temp_dir + "/device.nvvm.bc" : config.device_nvvm_ir_out);

    if (!compileDeviceToPTX(
          source_code, input_file, ptx_file, config, result.diagnostics, rdc_bitcode_out, nvvm_ir_file))
    {
      result.diagnostics += "\nDevice compilation failed";
      removeAll(temp_dir);
      return result;
    }

    // Experimental RDC path: the device code was also written to a bitcode
    // sidecar (rdc_bitcode_out) so the final link can combine every TU's device
    // code into a single fatbin. Compilation otherwise proceeds normally (the
    // cubin is still produced for callers that read it); the only difference is
    // that the host object is built without an embedded fatbin -- see the RDC
    // gate in compileHostCode -- so it carries only offloading entries and no
    // per-TU registration ctor.

    if (config.verbose)
    {
      result.diagnostics += "\n=== nvJitLink + fatbinary ===\n";
    }

    // Separate compilation: this TU's device code is finalized at the link, out
    // of the sidecars, so the device link here is run only for a caller that
    // asked for the cubin. Running it unconditionally would also reject a TU
    // that calls a device function defined in another one, which is the whole
    // point of relocatable device code.
    if (!output_cubin_path.empty())
    {
      std::vector<char> ptx_data;
      {
        std::ifstream f(ptx_file, std::ios::binary);
        ptx_data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
      }
      if (ptx_data.empty())
      {
        result.diagnostics += "\nFailed to read ptx file";
        removeAll(temp_dir);
        return result;
      }
      if (ptx_data.back() != '\0')
      {
        ptx_data.push_back('\0');
      }

      std::vector<char> nvvm_ir_data;
      if (config.device_nvvm_bypass)
      {
        std::ifstream f(nvvm_ir_file, std::ios::binary);
        nvvm_ir_data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        if (nvvm_ir_data.empty())
        {
          result.diagnostics += "\nFailed to read device NVVM IR: " + nvvm_ir_file;
          removeAll(temp_dir);
          return result;
        }
      }

      std::string arch_opt  = "-arch=sm_" + std::to_string(config.sm_version);
      std::string opt_level = "-O" + std::to_string(config.optimization_level >= 1 ? 3 : 0);
      std::vector<std::string> jitlink_option_strs{arch_opt, opt_level};
      // LTOIR inputs require -lto. When present, both the PTX and the LTOIRs
      // get linked through the LTO codegen path. IR handed over directly always
      // goes through LTO codegen, so bypass mode asks for it unconditionally.
      const bool have_ltoir = !config.device_ltoir_files.empty();
      if (have_ltoir || config.device_nvvm_bypass)
      {
        jitlink_option_strs.emplace_back("-lto");
      }
      std::vector<const char*> jitlink_options;
      jitlink_options.reserve(jitlink_option_strs.size());
      for (const auto& s : jitlink_option_strs)
      {
        jitlink_options.push_back(s.c_str());
      }

      nvJitLinkHandle jitlink_handle = nullptr;
      nvJitLinkResult jlr =
        nvJitLinkCreate(&jitlink_handle, static_cast<uint32_t>(jitlink_options.size()), jitlink_options.data());
      if (jlr != NVJITLINK_SUCCESS)
      {
        result.diagnostics += "\nnvJitLinkCreate failed (error " + std::to_string(static_cast<int>(jlr)) + ")";
        result.diagnostics += "\nnvJitLink options:";
        for (const auto& option : jitlink_option_strs)
        {
          result.diagnostics += " " + option;
        }
        removeAll(temp_dir);
        return result;
      }

      if (config.device_nvvm_bypass)
      {
        auto add_nvvm_ir = reinterpret_cast<nvJitLinkAddNvvmIRFn>(__nvJitLinkAPI(nvjitlink_api_add_nvvm));
        if (!add_nvvm_ir)
        {
          result.diagnostics += "\nnvJitLink does not provide the NVVM IR entry point";
          nvJitLinkDestroy(&jitlink_handle);
          removeAll(temp_dir);
          return result;
        }
        jlr = add_nvvm_ir(jitlink_handle, nvvm_ir_data.data(), nvvm_ir_data.size(), "device.nvvm.bc");
      }
      else
      {
        jlr = nvJitLinkAddData(jitlink_handle, NVJITLINK_INPUT_PTX, ptx_data.data(), ptx_data.size(), "device.ptx");
      }
      if (jlr != NVJITLINK_SUCCESS)
      {
        size_t log_size = 0;
        nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
        if (log_size > 1)
        {
          std::string log(log_size, '\0');
          nvJitLinkGetErrorLog(jitlink_handle, log.data());
          result.diagnostics += "\n" + log;
        }
        result.diagnostics += config.device_nvvm_bypass ? "\nnvJitLink NVVM IR input failed"
                                                       : "\nnvJitLinkAddData failed";
        nvJitLinkDestroy(&jitlink_handle);
        removeAll(temp_dir);
        return result;
      }

      // Feed LTO-IR inputs to nvJitLink alongside the device PTX. This is the
      // path for callers with pre-built nvcc -dlto artifacts. LLVM bitcode
      // inputs travel through the path above and are already inlined into the
      // PTX by the time we get here. nvJitLink resolves any remaining extern
      // symbols from these modules.
      for (const auto& ltoir_path : config.device_ltoir_files)
      {
        std::ifstream f(ltoir_path, std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (buf.empty())
        {
          continue;
        }
        jlr = nvJitLinkAddData(jitlink_handle, NVJITLINK_INPUT_LTOIR, buf.data(), buf.size(), ltoir_path.c_str());
        if (jlr != NVJITLINK_SUCCESS)
        {
          size_t log_size = 0;
          nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
          if (log_size > 1)
          {
            std::string log(log_size, '\0');
            nvJitLinkGetErrorLog(jitlink_handle, log.data());
            result.diagnostics += "\n" + log;
          }
          result.diagnostics += "\nnvJitLinkAddData(LTOIR) failed for " + ltoir_path;
          nvJitLinkDestroy(&jitlink_handle);
          removeAll(temp_dir);
          return result;
        }
      }

      jlr = nvJitLinkComplete(jitlink_handle);
      if (jlr != NVJITLINK_SUCCESS)
      {
        size_t log_size = 0;
        nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
        if (log_size > 1)
        {
          std::string log(log_size, '\0');
          nvJitLinkGetErrorLog(jitlink_handle, log.data());
          result.diagnostics += "\n" + log;
        }
        result.diagnostics += "\nnvJitLinkComplete failed";
        nvJitLinkDestroy(&jitlink_handle);
        removeAll(temp_dir);
        return result;
      }

      size_t cubin_size = 0;
      nvJitLinkGetLinkedCubinSize(jitlink_handle, &cubin_size);
      std::vector<char> cubin_data(cubin_size);
      nvJitLinkGetLinkedCubin(jitlink_handle, cubin_data.data());
      nvJitLinkDestroy(&jitlink_handle);

      if (!output_cubin_path.empty())
      {
        std::ofstream cubin_out(output_cubin_path, std::ios::binary);
        cubin_out.write(cubin_data.data(), static_cast<std::streamsize>(cubin_data.size()));
        if (!cubin_out)
        {
          result.diagnostics += "\nFailed to write cubin file";
          removeAll(temp_dir);
          return result;
        }
      }

      std::string arch             = std::to_string(config.sm_version);
      const char* fatbin_options[] = {"-64", "-cuda"};
      nvFatbinHandle fatbin_handle = nullptr;
      nvFatbinResult fbr           = nvFatbinCreate(&fatbin_handle, fatbin_options, 2);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinCreate failed: ") + nvFatbinGetErrorString(fbr);
        removeAll(temp_dir);
        return result;
      }

      fbr = nvFatbinAddCubin(fatbin_handle, cubin_data.data(), cubin_data.size(), arch.c_str(), "device.cubin");
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddCubin failed: ") + nvFatbinGetErrorString(fbr);
        nvFatbinDestroy(&fatbin_handle);
        removeAll(temp_dir);
        return result;
      }

      // In bypass mode the PTX still holds unresolved calls into the external
      // device code, since it was emitted before the device link. Only the
      // linked cubin is complete, so the PTX is left out of the fatbin.
      if (!config.device_nvvm_bypass)
      {
        fbr = nvFatbinAddPTX(fatbin_handle, ptx_data.data(), ptx_data.size(), arch.c_str(), "device.ptx", nullptr);
        if (fbr != NVFATBIN_SUCCESS)
        {
          result.diagnostics += std::string("\nnvFatbinAddPTX failed: ") + nvFatbinGetErrorString(fbr);
          nvFatbinDestroy(&fatbin_handle);
          removeAll(temp_dir);
          return result;
        }
      }

      size_t fatbin_size = 0;
      fbr                = nvFatbinSize(fatbin_handle, &fatbin_size);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinSize failed: ") + nvFatbinGetErrorString(fbr);
        nvFatbinDestroy(&fatbin_handle);
        removeAll(temp_dir);
        return result;
      }

      std::vector<char> fatbin_data(fatbin_size);
      fbr = nvFatbinGet(fatbin_handle, fatbin_data.data());
      nvFatbinDestroy(&fatbin_handle);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinGet failed: ") + nvFatbinGetErrorString(fbr);
        removeAll(temp_dir);
        return result;
      }

      std::ofstream out(fatbin_file, std::ios::binary);
      out.write(fatbin_data.data(), static_cast<std::streamsize>(fatbin_data.size()));
      if (!out)
      {
        result.diagnostics += "\nFailed to write fatbin file";
        removeAll(temp_dir);
        return result;
      }
    }

    if (config.verbose)
    {
      result.diagnostics += "\n=== Host compilation ===\n";
    }

    if (!compileHostCode(source_code, input_file, fatbin_file, output_path, config, result.diagnostics))
    {
      result.diagnostics += "\nHost compilation failed";
      removeAll(temp_dir);
      return result;
    }

    if (!config.keep_artifacts)
    {
      removeAll(temp_dir);
    }
    result.success = true;
    return result;
  }

  bool createPCH(const std::string& source_code,
                 libnvccPCHKind kind,
                 const std::string& pch_source_path,
                 const std::string& pch_output_path,
                 const CompilerOptions& config,
                 std::string& diagnostics)
  {
    std::string error_msg;
    if (!validateOptions(config, &error_msg))
    {
      diagnostics = "Configuration error: " + error_msg;
      return false;
    }

    initialize_llvm();

    std::string resource_dir = CLANG_RESOURCE_DIR;
    std::vector<std::string> arg_strings;
    arg_strings.push_back(pch_source_path);

    if (kind == LIBNVCC_PCH_DEVICE)
    {
      int ptx_version = 78;
      if (config.sm_version >= 120)
      {
        ptx_version = 87;
      }
      else if (config.sm_version >= 100)
      {
        ptx_version = 85;
      }
      else if (config.sm_version >= 90)
      {
        ptx_version = 80;
      }

      arg_strings.push_back("-triple");
      arg_strings.push_back("nvptx64-nvidia-cuda");
      arg_strings.push_back("-aux-triple");
#ifdef _WIN32
      arg_strings.push_back("x86_64-pc-windows-msvc");
#else
      arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
      arg_strings.push_back("-S");
      arg_strings.push_back("-aux-target-cpu");
      arg_strings.push_back("x86-64");
      arg_strings.push_back("-fcuda-is-device");
      arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
      arg_strings.push_back("-fms-compatibility");
      arg_strings.push_back("-fms-compatibility-version=19.40");
#else
      arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
      arg_strings.push_back("-mlink-builtin-bitcode");
      arg_strings.push_back(config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc");
      arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
      arg_strings.push_back("-target-cpu");
      arg_strings.push_back("sm_" + std::to_string(config.sm_version));
      arg_strings.push_back("-target-feature");
      arg_strings.push_back("+ptx" + std::to_string(ptx_version));
    }
    else if (kind == LIBNVCC_PCH_HOST)
    {
      arg_strings.push_back("-triple");
#ifdef _WIN32
      arg_strings.push_back("x86_64-pc-windows-msvc");
#else
      arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
      arg_strings.push_back("-aux-triple");
      arg_strings.push_back("nvptx64-nvidia-cuda");
      arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
      arg_strings.push_back("-emit-obj");
      arg_strings.push_back("-target-cpu");
      arg_strings.push_back("x86-64");
      arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
      arg_strings.push_back("-fms-compatibility");
      arg_strings.push_back("-fms-compatibility-version=19.40");
#else
      arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
      arg_strings.push_back("-mrelocation-model");
      arg_strings.push_back("pic");
      arg_strings.push_back("-pic-level");
      arg_strings.push_back("2");
    }
    else
    {
      diagnostics = "Invalid PCH kind";
      return false;
    }

    arg_strings.push_back("-resource-dir");
    arg_strings.push_back(resource_dir);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/stubs");
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(
      config.clang_headers_path.empty() ? std::string(CLANG_HEADERS_DIR) : config.clang_headers_path);
    appendSystemIncludePaths(arg_strings, config);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    appendIncludePaths(arg_strings, config);

    if (kind == LIBNVCC_PCH_DEVICE)
    {
      arg_strings.push_back("-D__HOSTJIT_DEVICE_COMPILATION__=1");
    }
    arg_strings.push_back("-DNDEBUG");

    appendMacroDefinitions(arg_strings, config);

    arg_strings.push_back("-fdeprecated-macro");
    if (kind == LIBNVCC_PCH_DEVICE)
    {
      arg_strings.push_back("--offload-new-driver");
      arg_strings.push_back("-fskip-odr-check-in-gmf");
      arg_strings.push_back("-fcxx-exceptions");
      arg_strings.push_back("-fexceptions");
    }
    else
    {
      arg_strings.push_back("--offload-new-driver");
      arg_strings.push_back("-fskip-odr-check-in-gmf");
    }
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-std=c++17");

    if (config.trace_includes)
    {
      arg_strings.push_back("-H");
    }

    appendExtraClangArgs(arg_strings, config);
    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    return generatePCH(source_code, pch_source_path, pch_output_path, arg_strings, diagnostics);
  }

  // Experimental RDC final link: combine each input object's device-bitcode
  // sidecar (<obj>.dev.bc) into one fatbin, then synthesize a single host
  // registration object via llvm::offloading::wrapCudaBinary. The generated
  // ctor registers that one fatbin and iterates the offloading entries every
  // host object contributed to the llvm_offload_entries section -- so several
  // host TUs share exactly one fatbin registration (no per-TU collision).
  bool buildRdcRegistrationObject(
    const std::vector<std::string>& object_files,
    const std::string& output_path,
    const CompilerOptions& config,
    std::string& out_reg_obj,
    std::string& diagnostics)
  {
    initialize_llvm();
    llvm::LLVMContext ctx;

    // 1) Link every TU's device bitcode sidecar into one device module.
    std::unique_ptr<llvm::Module> device_mod;
    for (const auto& obj : object_files)
    {
      const std::string bc = obj + ".dev.bc";
      if (!pathExists(bc))
      {
        continue;
      }
      llvm::SMDiagnostic err;
      auto m = llvm::parseIRFile(bc, err, ctx);
      if (!m)
      {
        std::string es;
        llvm::raw_string_ostream os(es);
        err.print("hostjit", os);
        diagnostics += "RDC: failed to parse device bitcode " + bc + ": " + es + "\n";
        return false;
      }
      if (!device_mod)
      {
        device_mod = std::move(m);
      }
      else if (llvm::Linker::linkModules(*device_mod, std::move(m)))
      {
        diagnostics += "RDC: failed to device-link " + bc + "\n";
        return false;
      }
    }
    if (!device_mod)
    {
      // No object carries device code -> nothing to device-link or register; the
      // caller links only the host objects.
      out_reg_obj.clear();
      return true;
    }

    // libdevice for any intrinsics the combined module still references.
    {
      const std::string libdevice_path = config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc";
      llvm::SMDiagnostic err;
      if (auto ld = llvm::parseIRFile(libdevice_path, err, ctx))
      {
        llvm::Linker::linkModules(*device_mod, std::move(ld), llvm::Linker::LinkOnlyNeeded);
      }
    }

    int ptx_version = 78;
    if (config.sm_version >= 120)
    {
      ptx_version = 87;
    }
    else if (config.sm_version >= 100)
    {
      ptx_version = 85;
    }
    else if (config.sm_version >= 90)
    {
      ptx_version = 80;
    }

    // 2) Device module -> PTX via the NVPTX backend.
    std::string err_str;
    const llvm::Target* dtarget = llvm::TargetRegistry::lookupTarget(device_mod->getTargetTriple(), err_str);
    if (!dtarget)
    {
      diagnostics += "RDC: NVPTX target not found: " + err_str + "\n";
      return false;
    }
    llvm::TargetOptions dopt;
    std::unique_ptr<llvm::TargetMachine> dtm(dtarget->createTargetMachine(
      device_mod->getTargetTriple(),
      "sm_" + std::to_string(config.sm_version),
      "+ptx" + std::to_string(ptx_version),
      dopt,
      llvm::Reloc::PIC_));
    if (!dtm)
    {
      diagnostics += "RDC: could not create NVPTX target machine\n";
      return false;
    }
    device_mod->setDataLayout(dtm->createDataLayout());

    std::string ptx;
    {
      llvm::raw_string_ostream sos(ptx);
      llvm::buffer_ostream bos(sos);
      llvm::legacy::PassManager pm;
      if (dtm->addPassesToEmitFile(pm, bos, nullptr, llvm::CodeGenFileType::AssemblyFile))
      {
        diagnostics += "RDC: NVPTX backend cannot emit PTX\n";
        return false;
      }
      pm.run(*device_mod);
    }
    if (ptx.empty())
    {
      diagnostics += "RDC: empty PTX after device link\n";
      return false;
    }
    if (ptx.back() != '\0')
    {
      ptx.push_back('\0');
    }

    // 3) PTX -> cubin (nvJitLink) -> fatbin (nvFatbin).
    std::vector<char> fatbin;
    {
      const std::string arch_opt  = "-arch=sm_" + std::to_string(config.sm_version);
      const std::string opt_level = "-O" + std::to_string(config.optimization_level >= 1 ? 3 : 0);
      // External operators supplied as NVRTC LTO-IR (--device-ltoir) are linked
      // in here, at the final device link, alongside the combined PTX. LTO-IR
      // inputs require -lto; nvJitLink then resolves the kernel's extern device
      // symbol(s) from them. Because the kernel arrives as PTX (the clang NVPTX
      // backend does not emit LTO-IR), this is a *partial* LTO: the symbol
      // resolves but the operator is not inlined into the kernel. Full
      // cross-module inlining needs the kernel in LTO-IR too (product path via
      // libNVVM). Operators supplied as LLVM bitcode (--device-bitcode) are
      // instead linked + inlined at the IR level before device codegen.
      const bool have_ltoir = !config.device_ltoir_files.empty();
      std::vector<const char*> jl_opts{arch_opt.c_str(), opt_level.c_str()};
      if (have_ltoir)
      {
        jl_opts.push_back("-lto");
      }
      nvJitLinkHandle jl = nullptr;
      if (nvJitLinkCreate(&jl, static_cast<uint32_t>(jl_opts.size()), jl_opts.data()) != NVJITLINK_SUCCESS)
      {
        diagnostics += "RDC: nvJitLinkCreate failed\n";
        return false;
      }
      if (nvJitLinkAddData(jl, NVJITLINK_INPUT_PTX, ptx.data(), ptx.size(), "device.ptx") != NVJITLINK_SUCCESS)
      {
        size_t ls = 0;
        nvJitLinkGetErrorLogSize(jl, &ls);
        if (ls > 1)
        {
          std::string l(ls, '\0');
          nvJitLinkGetErrorLog(jl, l.data());
          diagnostics += "\n" + l;
        }
        diagnostics += "RDC: nvJitLinkAddData(PTX) failed\n";
        nvJitLinkDestroy(&jl);
        return false;
      }
      for (const auto& ltoir_path : config.device_ltoir_files)
      {
        std::ifstream f(ltoir_path, std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (buf.empty())
        {
          continue;
        }
        if (nvJitLinkAddData(jl, NVJITLINK_INPUT_LTOIR, buf.data(), buf.size(), ltoir_path.c_str())
            != NVJITLINK_SUCCESS)
        {
          size_t ls = 0;
          nvJitLinkGetErrorLogSize(jl, &ls);
          if (ls > 1)
          {
            std::string l(ls, '\0');
            nvJitLinkGetErrorLog(jl, l.data());
            diagnostics += "\n" + l;
          }
          diagnostics += "RDC: nvJitLinkAddData(LTOIR) failed for " + ltoir_path + "\n";
          nvJitLinkDestroy(&jl);
          return false;
        }
      }
      if (nvJitLinkComplete(jl) != NVJITLINK_SUCCESS)
      {
        size_t ls = 0;
        nvJitLinkGetErrorLogSize(jl, &ls);
        if (ls > 1)
        {
          std::string l(ls, '\0');
          nvJitLinkGetErrorLog(jl, l.data());
          diagnostics += "\n" + l;
        }
        diagnostics += "RDC: nvJitLinkComplete failed\n";
        nvJitLinkDestroy(&jl);
        return false;
      }
      size_t cubin_size = 0;
      nvJitLinkGetLinkedCubinSize(jl, &cubin_size);
      std::vector<char> cubin(cubin_size);
      nvJitLinkGetLinkedCubin(jl, cubin.data());
      nvJitLinkDestroy(&jl);

      const std::string arch       = std::to_string(config.sm_version);
      const char* fatbin_options[] = {"-64", "-cuda"};
      nvFatbinHandle fh            = nullptr;
      if (nvFatbinCreate(&fh, fatbin_options, 2) != NVFATBIN_SUCCESS)
      {
        diagnostics += "RDC: nvFatbinCreate failed\n";
        return false;
      }
      if (nvFatbinAddCubin(fh, cubin.data(), cubin.size(), arch.c_str(), "device.cubin") != NVFATBIN_SUCCESS
          || nvFatbinAddPTX(fh, ptx.data(), ptx.size(), arch.c_str(), "device.ptx", nullptr) != NVFATBIN_SUCCESS)
      {
        diagnostics += "RDC: nvFatbinAdd* failed\n";
        nvFatbinDestroy(&fh);
        return false;
      }
      size_t fsz = 0;
      if (nvFatbinSize(fh, &fsz) != NVFATBIN_SUCCESS)
      {
        diagnostics += "RDC: nvFatbinSize failed\n";
        nvFatbinDestroy(&fh);
        return false;
      }
      fatbin.resize(fsz);
      if (nvFatbinGet(fh, fatbin.data()) != NVFATBIN_SUCCESS)
      {
        diagnostics += "RDC: nvFatbinGet failed\n";
        nvFatbinDestroy(&fh);
        return false;
      }
      nvFatbinDestroy(&fh);
    }

    // 4) Synthesize the host registration module (one fatbin + all entries).
    llvm::Module regM("hostjit_rdc_registration", ctx);
#ifdef _WIN32
    regM.setTargetTriple(llvm::Triple("x86_64-pc-windows-msvc"));
#else
    regM.setTargetTriple(llvm::Triple("x86_64-pc-linux-gnu"));
#endif
    const llvm::Target* htarget = llvm::TargetRegistry::lookupTarget(regM.getTargetTriple(), err_str);
    if (!htarget)
    {
      diagnostics += "RDC: host target not found: " + err_str + "\n";
      return false;
    }
    llvm::TargetOptions hopt;
    // Emit constructors into .init_array (processed by the dynamic loader at
    // dlopen), not the legacy .ctors (which needs CRT startup code absent from
    // our freestanding .so) -- otherwise the fatbin-registration ctor never runs.
    hopt.UseInitArray = true;
    std::unique_ptr<llvm::TargetMachine> htm(
      htarget->createTargetMachine(regM.getTargetTriple(), "x86-64", "", hopt, llvm::Reloc::PIC_));
    if (!htm)
    {
      diagnostics += "RDC: could not create host target machine\n";
      return false;
    }
    regM.setDataLayout(htm->createDataLayout());

    auto entry_array = llvm::offloading::getOffloadEntryArray(regM, "llvm_offload_entries");
    llvm::ArrayRef<char> image(fatbin.data(), fatbin.size());
    if (auto e = llvm::offloading::wrapCudaBinary(regM, image, entry_array))
    {
      diagnostics += "RDC: wrapCudaBinary failed: " + llvm::toString(std::move(e)) + "\n";
      return false;
    }

#ifdef _WIN32
    // CUDA 13's cudart no longer exports __cudaRegisterSurface / __cudaRegisterTexture
    // (legacy texture/surface references). wrapCudaBinary still emits calls to them,
    // but JIT'd CUB kernels never register textures or surfaces, so those calls are
    // never reached at runtime. Give the declarations a local no-op body so the
    // Windows DLL links and loads -- otherwise the loader fails with
    // "The specified procedure could not be found" on the missing cudart imports.
    for (const char* sym : {"__cudaRegisterSurface", "__cudaRegisterTexture"})
    {
      if (auto* f = regM.getFunction(sym); f && f->isDeclaration())
      {
        f->setLinkage(llvm::GlobalValue::InternalLinkage);
        llvm::ReturnInst::Create(ctx, llvm::BasicBlock::Create(ctx, "entry", f));
      }
    }
#endif

    // 5) Emit the registration module to a host object for the final link.
    out_reg_obj = output_path + ".rdcreg.o";
    std::error_code ec;
    llvm::raw_fd_ostream ro(out_reg_obj, ec, llvm::sys::fs::OF_None);
    if (ec)
    {
      diagnostics += "RDC: cannot open registration object " + out_reg_obj + "\n";
      return false;
    }
    {
      llvm::legacy::PassManager pm;
      if (htm->addPassesToEmitFile(pm, ro, nullptr, llvm::CodeGenFileType::ObjectFile))
      {
        diagnostics += "RDC: host backend cannot emit registration object\n";
        return false;
      }
      pm.run(regM);
    }
    ro.flush();
    return true;
  }

  LinkResult linkToSharedLibrary(
    const std::vector<std::string>& object_files, const std::string& output_path, const CompilerOptions& config)
  {
    LinkResult result;
    result.success      = false;
    result.library_path = output_path;

    if (object_files.empty())
    {
      result.diagnostics = "No object files provided";
      return result;
    }

    // RDC: combine the input objects' device bitcode sidecars into one fatbin and
    // a single registration object, and link that in alongside the host objects.
    // If none of the objects carry device code, no registration object is
    // produced and only the host objects are linked.
    std::vector<std::string> link_objects = object_files;
    {
      std::string reg_obj;
      if (!buildRdcRegistrationObject(object_files, output_path, config, reg_obj, result.diagnostics))
      {
        result.diagnostics += "\nRDC registration build failed";
        return result;
      }
      if (!reg_obj.empty())
      {
        link_objects.push_back(reg_obj);
      }
    }

    std::vector<std::string> arg_strings;

#ifdef _WIN32
    arg_strings.push_back("lld-link");
    arg_strings.push_back("/DLL");
    // The DLL carries its own entry point (see __clang_cuda_runtime_wrapper.h):
    // it runs the static constructors on attach, which is what registers the
    // fatbin, and unregisters it on detach. A DLL linked /NOENTRY does neither,
    // and only works for a caller that knows to drive both by hand.
    arg_strings.push_back("/ENTRY:hostjit_dll_entry");
    arg_strings.push_back("/NODEFAULTLIB");
    arg_strings.push_back("/OUT:" + output_path);

    // Generate import libraries from DLLs present on the system,
    // so we don't require the Windows SDK or MSVC .lib files.
    std::string implib_dir = std::filesystem::path(output_path).parent_path().string();

    std::string cudart_dll = findCudartDllName(config.cuda_toolkit_path);
    generateImportLib(
      cudart_dll,
      {"cudaMalloc",
       "cudaFree",
       "cudaMemcpy",
       "cudaMemcpyAsync",
       "cudaMemset",
       "cudaMemsetAsync",
       "cudaDeviceSynchronize",
       "cudaFuncSetAttribute",
       "cudaGetDevice",
       "cudaGetDeviceProperties",
       "cudaGetLastError",
       "cudaPeekAtLastError",
       "cudaGetErrorString",
       "cudaStreamCreate",
       "cudaStreamDestroy",
       "cudaStreamSynchronize",
       "cudaEventCreate",
       "cudaEventDestroy",
       "cudaEventRecord",
       "cudaEventSynchronize",
       "cudaEventElapsedTime",
       "cudaMallocAsync",
       "cudaFreeAsync",
       "cudaDeviceGetAttribute",
       "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
       "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
       "cudaFuncGetAttributes",
       "cudaLaunchKernel",
       "cudaLaunchKernelExC",
       "__cudaRegisterFatBinary",
       "__cudaRegisterFatBinaryEnd",
       "__cudaUnregisterFatBinary",
       "__cudaRegisterFunction",
       "__cudaRegisterVar",
       "__cudaRegisterManagedVar",
       "__cudaPushCallConfiguration",
       "__cudaPopCallConfiguration"},
      implib_dir + "/cudart.lib");

    generateImportLib(
      "ucrtbase.dll",
      {"malloc",
       "free",
       "calloc",
       "realloc",
       "_callnewh",
       "_errno",
       "abort",
       "exit",
       "_exit",
       "_register_onexit_function",
       "_crt_atexit",
       "_initterm",
       "_initterm_e",
       "memcpy",
       "memset",
       "memmove",
       "memcmp",
       "strlen",
       "strcmp",
       "strncmp",
       "_initialize_onexit_table",
       "_execute_onexit_table",
       "_register_thread_local_exe_atexit_callback"},
      implib_dir + "/ucrt.lib");

    generateImportLib(
      "vcruntime140.dll",
      {"__std_exception_copy",
       "__std_exception_destroy",
       "__CxxFrameHandler3",
       "_CxxThrowException",
       "memcpy",
       "memset",
       "memmove",
       "memcmp",
       "__std_type_info_destroy_list",
       "_purecall"},
      implib_dir + "/vcruntime.lib");

    generateImportLib(
      "kernel32.dll",
      {"InitializeCriticalSection",
       "EnterCriticalSection",
       "LeaveCriticalSection",
       "DeleteCriticalSection",
       "InitOnceExecuteOnce",
       "LoadLibraryExA",
       "LoadLibraryExW",
       "GetProcAddress",
       "FreeLibrary",
       "GetModuleHandleA",
       "GetLastError",
       "SetLastError",
       "GetCurrentProcess",
       "GetCurrentThread",
       "GetCurrentThreadId",
       "VirtualProtect",
       "FlushInstructionCache",
       "QueryPerformanceCounter",
       "QueryPerformanceFrequency"},
      implib_dir + "/kernel32.lib");

    arg_strings.push_back("/LIBPATH:" + implib_dir);

    for (const auto& obj_file : link_objects)
    {
      arg_strings.push_back(obj_file);
    }

    arg_strings.push_back("cudart.lib");
    arg_strings.push_back("ucrt.lib");
    arg_strings.push_back("vcruntime.lib");
    arg_strings.push_back("kernel32.lib");
#else
    arg_strings.push_back("ld.lld");
    arg_strings.push_back("-shared");
    arg_strings.push_back("--build-id");
    arg_strings.push_back("--eh-frame-hdr");
    arg_strings.push_back("-m");
    arg_strings.push_back("elf_x86_64");
    // Allow unresolved symbols — they will be satisfied at dlopen() time
    // by libraries already loaded in the host process (libc, libstdc++,
    // cudart, etc.).  This removes the need for system CRT objects and
    // dev packages on the target machine.
    arg_strings.push_back("--allow-shlib-undefined");
    arg_strings.push_back("-o");
    arg_strings.push_back(output_path);

    for (const auto& lib_path : config.library_paths)
    {
      arg_strings.push_back("-L" + lib_path);
      // Embed the library path as RPATH so the dynamic linker can find
      // libcudart.so.XX at dlopen time without LD_LIBRARY_PATH.
      arg_strings.push_back("-rpath");
      arg_strings.push_back(lib_path);
    }

    for (const auto& obj_file : link_objects)
    {
      arg_strings.push_back(obj_file);
    }

    // pip packages ship libcudart.so.XX without an unversioned symlink,
    // so -lcudart won't work.  Find the actual .so by scanning library_paths.
    {
      bool found_cudart = false;
      for (const auto& lib_path : config.library_paths)
      {
        if (!pathExists(lib_path))
        {
          continue;
        }
        forEachDirectoryEntry(lib_path, [&](const std::filesystem::directory_entry& entry) {
          auto fname = entry.path().filename().string();
          if (!found_cudart && fname.starts_with("libcudart.so"))
          {
            arg_strings.push_back(entry.path().string());
            found_cudart = true;
          }
        });
        if (found_cudart)
        {
          break;
        }
      }
      if (!found_cudart)
      {
        arg_strings.push_back("-lcudart");
      }
    }
#endif

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    std::string stdout_str, stderr_str;
    llvm::raw_string_ostream stdout_os(stdout_str);
    llvm::raw_string_ostream stderr_os(stderr_str);

#ifdef _WIN32
    bool link_success = lld::coff::link(args, stdout_os, stderr_os, false, false);
#else
    bool link_success = lld::elf::link(args, stdout_os, stderr_os, false, false);
#endif

    stdout_os.flush();
    stderr_os.flush();

    if (!stdout_str.empty())
    {
      result.diagnostics += stdout_str;
    }
    if (!stderr_str.empty())
    {
      result.diagnostics += stderr_str;
    }

    if (!link_success)
    {
      result.diagnostics += "\nLinking failed";
      return result;
    }

    result.success = true;
    return result;
  }
};
} // namespace libnvcc

struct libnvccProgram_st
{
  std::string source;
  std::string name;
  std::string log;
  libnvcc::CompilerImpl compiler;
};

namespace
{
void setProgramLog(libnvccProgram prog, std::string log)
{
  if (prog)
  {
    prog->log = std::move(log);
  }
}

bool parseProgramOptions(
  libnvccProgram prog, int num_options, const char* const* raw_options, libnvcc::CompilerOptions& options)
{
  std::string error;
  if (!libnvcc::parseOptions(num_options, raw_options, options, error))
  {
    setProgramLog(prog, "Option error: " + error);
    return false;
  }
  return true;
}
} // anonymous namespace

extern "C" const char* libnvccGetErrorString(libnvccResult result)
{
  switch (result)
  {
    case LIBNVCC_SUCCESS:
      return "LIBNVCC_SUCCESS";
    case LIBNVCC_ERROR_OUT_OF_MEMORY:
      return "LIBNVCC_ERROR_OUT_OF_MEMORY";
    case LIBNVCC_ERROR_PROGRAM_CREATION_FAILURE:
      return "LIBNVCC_ERROR_PROGRAM_CREATION_FAILURE";
    case LIBNVCC_ERROR_INVALID_INPUT:
      return "LIBNVCC_ERROR_INVALID_INPUT";
    case LIBNVCC_ERROR_INVALID_PROGRAM:
      return "LIBNVCC_ERROR_INVALID_PROGRAM";
    case LIBNVCC_ERROR_INVALID_OPTION:
      return "LIBNVCC_ERROR_INVALID_OPTION";
    case LIBNVCC_ERROR_COMPILATION:
      return "LIBNVCC_ERROR_COMPILATION";
    case LIBNVCC_ERROR_LINKING:
      return "LIBNVCC_ERROR_LINKING";
    case LIBNVCC_ERROR_PCH_CREATE:
      return "LIBNVCC_ERROR_PCH_CREATE";
    case LIBNVCC_ERROR_INTERNAL_ERROR:
      return "LIBNVCC_ERROR_INTERNAL_ERROR";
  }
  return "LIBNVCC_ERROR_UNKNOWN";
}

// The compile stages are thread-safe: each build runs clang codegen with its own
// CompilerInstance and LLVMContext, so concurrent compiles do not race. The link
// stage is not. lld::elf::link() keeps its state in a process-global
// (CommonLinkerContext, a plain `static`, not `thread_local`), so concurrent links
// clobber that shared context and corrupt LLD's bump allocator. Serialize just
// the link step through one process-wide mutex.
static std::mutex g_link_mutex;

extern "C" libnvccResult libnvccCreateProgram(libnvccProgram* prog, const char* src, const char* name)
{
  if (!prog || !src)
  {
    return LIBNVCC_ERROR_INVALID_INPUT;
  }
  *prog = nullptr;

  auto* program   = new libnvccProgram_st;
  program->source = src;
  program->name   = (name && name[0]) ? name : "input.cu";
  *prog           = program;
  return LIBNVCC_SUCCESS;
}

extern "C" libnvccResult libnvccDestroyProgram(libnvccProgram* prog)
{
  if (!prog || !*prog)
  {
    return LIBNVCC_SUCCESS;
  }
  delete *prog;
  *prog = nullptr;
  return LIBNVCC_SUCCESS;
}

extern "C" libnvccResult libnvccCompileProgramToDeviceBitcode(
  libnvccProgram prog, const char* outputBitcodePath, int numOptions, const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!outputBitcodePath || outputBitcodePath[0] == '\0')
  {
    setProgramLog(prog, "outputBitcodePath must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  auto result = prog->compiler.compileToDeviceBitcode(prog->source, prog->name, outputBitcodePath, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_COMPILATION;
}

extern "C" libnvccResult libnvccCompileProgramToDeviceLTOIR(
  libnvccProgram prog, const char* outputLtoirPath, int numOptions, const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!outputLtoirPath || outputLtoirPath[0] == '\0')
  {
    setProgramLog(prog, "outputLtoirPath must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  auto result = prog->compiler.compileToDeviceLTOIR(prog->source, prog->name, outputLtoirPath, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_COMPILATION;
}

extern "C" libnvccResult libnvccCompileProgramToObject(
  libnvccProgram prog,
  const char* outputObjectPath,
  const char* outputCubinPath,
  int numOptions,
  const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!outputObjectPath || outputObjectPath[0] == '\0')
  {
    setProgramLog(prog, "outputObjectPath must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  const std::string cubin_path = outputCubinPath ? outputCubinPath : "";
  auto result = prog->compiler.compileToObject(prog->source, prog->name, outputObjectPath, cubin_path, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_COMPILATION;
}

extern "C" libnvccResult libnvccLinkToSharedLibrary(
  libnvccProgram prog,
  int numObjectFiles,
  const char* const* objectFiles,
  const char* outputLibraryPath,
  int numOptions,
  const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (numObjectFiles < 0 || (numObjectFiles > 0 && !objectFiles) || !outputLibraryPath || outputLibraryPath[0] == '\0')
  {
    setProgramLog(prog, "Invalid link input");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  std::vector<std::string> object_files;
  object_files.reserve(static_cast<size_t>(numObjectFiles));
  for (int i = 0; i < numObjectFiles; ++i)
  {
    if (!objectFiles[i] || objectFiles[i][0] == '\0')
    {
      setProgramLog(prog, "Object file path must be non-empty");
      return LIBNVCC_ERROR_INVALID_INPUT;
    }
    object_files.emplace_back(objectFiles[i]);
  }

  const std::lock_guard<std::mutex> lock(g_link_mutex);
  auto result = prog->compiler.linkToSharedLibrary(object_files, outputLibraryPath, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_LINKING;
}

extern "C" libnvccResult libnvccCreatePCH(
  libnvccProgram prog,
  libnvccPCHKind kind,
  const char* pchSourcePath,
  const char* pchOutputPath,
  int numOptions,
  const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!pchSourcePath || pchSourcePath[0] == '\0' || !pchOutputPath || pchOutputPath[0] == '\0')
  {
    setProgramLog(prog, "PCH source and output paths must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  std::string diagnostics;
  bool success =
    prog->compiler.createPCH(prog->source, kind, pchSourcePath, pchOutputPath, parsed_options, diagnostics);
  setProgramLog(prog, diagnostics);
  return success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_PCH_CREATE;
}

extern "C" libnvccResult libnvccGetProgramLogSize(libnvccProgram prog, size_t* logSizeRet)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!logSizeRet)
  {
    return LIBNVCC_ERROR_INVALID_INPUT;
  }
  *logSizeRet = prog->log.size() + 1;
  return LIBNVCC_SUCCESS;
}

extern "C" libnvccResult libnvccGetProgramLog(libnvccProgram prog, char* log)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!log)
  {
    return LIBNVCC_ERROR_INVALID_INPUT;
  }
  std::memcpy(log, prog->log.c_str(), prog->log.size() + 1);
  return LIBNVCC_SUCCESS;
}
