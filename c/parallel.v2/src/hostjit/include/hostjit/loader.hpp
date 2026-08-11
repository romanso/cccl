#pragma once

#include <string>

namespace hostjit
{
// Holds one JIT-compiled CUDA module. Not a general-purpose dynamic-library
// wrapper: unload() drives the module's CUDA teardown and requires the library
// to link cudart, and it aborts rather than unmap a module it cannot drain.
class DynamicLibrary
{
public:
  DynamicLibrary();
  ~DynamicLibrary();

  // Disable copy
  DynamicLibrary(const DynamicLibrary&)            = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  // Enable move
  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

  // Load a shared library
  bool load(const std::string& library_path);

  // Get a symbol (function or variable) by name
  void* getSymbol(const std::string& symbol_name);

  // Template helper to get function pointers with type safety
  template <typename FuncType>
  FuncType getFunction(const std::string& name)
  {
    return reinterpret_cast<FuncType>(getSymbol(name));
  }

  // Check if library is loaded
  bool isLoaded() const;

  // Get the last error message
  std::string getLastError() const;

  // Unload the library
  void unload();

  // Absolute path of the currently-loaded module as the OS sees it (via
  // GetModuleFileName on Windows / dlinfo on Linux), or empty if nothing is
  // loaded. Lets callers identify the module by its real name at runtime rather
  // than hard-coding it.
  std::string getLoadedModulePath() const;

private:
  // Ask the module to tear itself down (this unregisters its fatbin), earlier than
  // the OS would do it on unmap. See unload().
  void runModuleFini();

  void* handle_;
  std::string last_error_;
};
} // namespace hostjit
