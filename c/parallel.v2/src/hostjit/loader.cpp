#ifndef _WIN32
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE // for dlinfo / RTLD_DI_LINKMAP
#  endif
#endif

#include <hostjit/loader.hpp>

#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <link.h>
#endif

namespace hostjit
{
#ifdef _WIN32
namespace
{
// GetProcAddress on a loaded DLL only searches that module's own export table,
// not the DLLs it imports. The JIT module imports cudaDeviceSynchronize from
// cudart, so resolve it from cudart (already loaded in-process) by scanning all
// loaded modules. This mirrors Linux dlsym(handle, ...), which follows the
// module's dependency graph. K32EnumProcessModules is resolved dynamically so
// no psapi link dependency is introduced.
void* resolveFromLoadedModules(const char* name)
{
  using EnumFn  = BOOL(WINAPI*)(HANDLE, HMODULE*, DWORD, LPDWORD);
  HMODULE k32   = GetModuleHandleA("kernel32.dll");
  auto enumMods =
    k32 ? reinterpret_cast<EnumFn>(reinterpret_cast<void*>(GetProcAddress(k32, "K32EnumProcessModules"))) : nullptr;
  if (!enumMods)
  {
    return nullptr;
  }
  HMODULE mods[1024];
  DWORD needed = 0;
  if (!enumMods(GetCurrentProcess(), mods, static_cast<DWORD>(sizeof(mods)), &needed))
  {
    return nullptr;
  }
  // needed reports what the full list would take, not what was written: the call
  // succeeds with a truncated list when the process has more modules than fit.
  const int capacity = static_cast<int>(sizeof(mods) / sizeof(mods[0]));
  int n              = static_cast<int>(needed / sizeof(HMODULE));
  if (n > capacity)
  {
    n = capacity;
  }
  for (int i = 0; i < n; ++i)
  {
    if (auto* s = reinterpret_cast<void*>(GetProcAddress(mods[i], name)))
    {
      return s;
    }
  }
  return nullptr;
}

std::string getWindowsError()
{
  DWORD error = GetLastError();
  if (error == 0)
  {
    return "";
  }

  LPSTR buffer = nullptr;
  DWORD size   = FormatMessageA(
    FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
    nullptr,
    error,
    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
    reinterpret_cast<LPSTR>(&buffer),
    0,
    nullptr);

  std::string message;
  if (size > 0 && buffer)
  {
    message = std::string(buffer, size);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
    {
      message.pop_back();
    }
    LocalFree(buffer);
  }
  else
  {
    message = "Unknown error (code: " + std::to_string(error) + ")";
  }

  return message;
}
} // anonymous namespace
#endif

DynamicLibrary::DynamicLibrary()
    : handle_(nullptr)
{}

DynamicLibrary::~DynamicLibrary()
{
  unload();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(other.handle_)
    , last_error_(std::move(other.last_error_))
{
  other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept
{
  if (this != &other)
  {
    unload();
    handle_       = other.handle_;
    last_error_   = std::move(other.last_error_);
    other.handle_ = nullptr;
  }
  return *this;
}

bool DynamicLibrary::load(const std::string& library_path)
{
  unload();

#ifdef _WIN32
  SetLastError(0);
  handle_ = static_cast<void*>(LoadLibraryA(library_path.c_str()));

  if (!handle_)
  {
    last_error_ = getWindowsError();
    if (last_error_.empty())
    {
      last_error_ = "Unknown LoadLibrary error";
    }
    return false;
  }
  // The DLL names its own entry point, so LoadLibrary has already run the static
  // constructors and the fatbin is registered. This used to be done here, for a
  // /NOENTRY DLL that had no entry point to run them.
#else
  dlerror();
  handle_ = dlopen(library_path.c_str(), RTLD_LAZY | RTLD_LOCAL);

  if (!handle_)
  {
    const char* error = dlerror();
    last_error_       = error ? error : "Unknown dlopen error";
    return false;
  }
#endif

  last_error_.clear();
  return true;
}

void* DynamicLibrary::getSymbol(const std::string& symbol_name)
{
  if (!handle_)
  {
    last_error_ = "Library not loaded";
    return nullptr;
  }

#ifdef _WIN32
  SetLastError(0);
  void* symbol = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), symbol_name.c_str()));

  if (!symbol)
  {
    last_error_ = getWindowsError();
    if (last_error_.empty())
    {
      last_error_ = "Symbol not found: " + symbol_name;
    }
    return nullptr;
  }
#else
  dlerror();
  void* symbol = dlsym(handle_, symbol_name.c_str());

  const char* error = dlerror();
  if (error)
  {
    last_error_ = error;
    return nullptr;
  }
#endif

  last_error_.clear();
  return symbol;
}

bool DynamicLibrary::isLoaded() const
{
  return handle_ != nullptr;
}

std::string DynamicLibrary::getLastError() const
{
  return last_error_;
}

void DynamicLibrary::unload()
{
  if (handle_)
  {
    // Kernel launches are asynchronous, so kernels from this module may still be
    // executing on the GPU when the caller unloads it, and the CUDA runtime keeps
    // a pointer into the module's embedded fatbin (modules are loaded lazily).
    // dlclose / FreeLibrary unmaps the module's memory immediately, so without a
    // barrier a later CUDA call would dereference freed memory and crash.
    // Synchronize first, so all GPU work referencing the module has finished
    // before its memory goes away.
    //
    // cudaDeviceSynchronize is looked up by symbol in the loaded module (which
    // links cudart) rather than called directly, so this file needs no cudart
    // link dependency. The module always links cudart, so the symbol must
    // resolve; if it does not, we cannot drain in-flight work and unmapping the
    // module anyway would risk a use-after-unmap crash -- fail loudly instead of
    // skipping the barrier silently.
    using sync_fn = int (*)();
    sync_fn sync = nullptr;
    {
#ifdef _WIN32
      // Resolve from cudart (imported by the JIT module, already in-process)
      // rather than the JIT module's own exports -- GetProcAddress on handle_
      // would not find an imported symbol.
      sync = reinterpret_cast<sync_fn>(resolveFromLoadedModules("cudaDeviceSynchronize"));
#else
      sync = reinterpret_cast<sync_fn>(dlsym(handle_, "cudaDeviceSynchronize"));
#endif
      if (!sync)
      {
        std::fprintf(stderr, "hostjit: cudaDeviceSynchronize not found in JIT module; cannot safely unload\n");
        std::abort();
      }
      sync();
    }

    // The module unregisters its fatbin on its own when the OS unloads it, but we
    // want that done while it is still mapped, so that the flush below can give the
    // device state back at unload time. The module exports __cudacc_module_fini()
    // for exactly this; it is one-shot, so the OS hook afterwards finds nothing to do.
    runModuleFini();

    // Unregistering only queues the module for unload; the runtime issues the
    // driver call from its next entry point, so without this the module would
    // stay resident on the device for as long as the process makes no CUDA call.
    // One more runtime call, made while the module is still mapped, drains that
    // queue and gives the device state back at unload time. Nothing public says
    // when the runtime issues that driver call or how to force it, so this rests
    // on measured behaviour; a CUDART entry point that flushed the queue would
    // make the sequence a supported one and save the second full-device sync.
    sync();

    // The fatbin is unregistered, so it is now safe to unmap the module.
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
    handle_ = nullptr;
  }
  last_error_.clear();
}

std::string DynamicLibrary::getLoadedModulePath() const
{
  if (!handle_)
  {
    return {};
  }
#ifdef _WIN32
  char path[MAX_PATH] = {};
  DWORD n = GetModuleFileNameA(static_cast<HMODULE>(handle_), path, static_cast<DWORD>(sizeof(path)));
  return (n > 0) ? std::string(path, n) : std::string{};
#else
  struct link_map* lm = nullptr;
  if (dlinfo(handle_, RTLD_DI_LINKMAP, &lm) == 0 && lm != nullptr && lm->l_name != nullptr)
  {
    return std::string(lm->l_name);
  }
  return {};
#endif
}

void DynamicLibrary::runModuleFini()
{
  if (!handle_)
  {
    return;
  }

#ifdef _WIN32
  auto proc = GetProcAddress(static_cast<HMODULE>(handle_), "__cudacc_module_fini");
  auto fini = reinterpret_cast<void(__cdecl*)(void)>(proc);
#else
  auto fini = reinterpret_cast<void (*)(void)>(dlsym(handle_, "__cudacc_module_fini"));
#endif

  if (fini)
  {
    fini();
  }
}
} // namespace hostjit
