if (DEFINED LIBNVCC_CPM_CMAKE_PATH AND EXISTS "${LIBNVCC_CPM_CMAKE_PATH}")
  include("${LIBNVCC_CPM_CMAKE_PATH}")
else()
  find_file(
    LIBNVCC_CPM_CMAKE_PATH
    NAMES CPM.cmake
    PATHS ${CMAKE_MODULE_PATH}
    NO_DEFAULT_PATH
  )
  if (LIBNVCC_CPM_CMAKE_PATH)
    include("${LIBNVCC_CPM_CMAKE_PATH}")
  else()
    message(
      FATAL_ERROR
      "CPM.cmake not found. Set LIBNVCC_CPM_CMAKE_PATH or add it to CMAKE_MODULE_PATH."
    )
  endif()
endif()

if (MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(
    FATAL_ERROR
    "libnvcc does not support Debug builds on Windows. "
    "The statically-linked LLVM Debug build is too large and causes stack "
    "overflows at runtime. Use MinSizeRel, Release, or RelWithDebInfo instead."
  )
endif()

if (DEFINED HOSTJIT_LLVM_VERSION AND NOT DEFINED LIBNVCC_LLVM_VERSION)
  set(_libnvcc_llvm_version_default "${HOSTJIT_LLVM_VERSION}")
else()
  set(_libnvcc_llvm_version_default "llvmorg-22.1.1")
endif()
set(
  LIBNVCC_LLVM_VERSION
  "${_libnvcc_llvm_version_default}"
  CACHE STRING
  "LLVM git tag to fetch"
)

# Optionally use a prebuilt LLVM release instead of building LLVM/Clang/LLD from
# source. The release must match LIBNVCC_LLVM_VERSION's major version and ship
# the static dev libraries + CMake exports + headers (the official
# LLVM-<ver>-Linux-X64.tar.xz does).
if (DEFINED HOSTJIT_USE_PREBUILT_LLVM AND NOT DEFINED LIBNVCC_USE_PREBUILT_LLVM)
  set(_libnvcc_prebuilt_default "${HOSTJIT_USE_PREBUILT_LLVM}")
else()
  set(_libnvcc_prebuilt_default OFF)
endif()
option(
  LIBNVCC_USE_PREBUILT_LLVM
  "Use a prebuilt LLVM release instead of building from source via CPM"
  ${_libnvcc_prebuilt_default}
)
if (DEFINED HOSTJIT_PREBUILT_LLVM_DIR AND NOT DEFINED LIBNVCC_PREBUILT_LLVM_DIR)
  set(_libnvcc_prebuilt_dir_default "${HOSTJIT_PREBUILT_LLVM_DIR}")
else()
  set(_libnvcc_prebuilt_dir_default "")
endif()
set(
  LIBNVCC_PREBUILT_LLVM_DIR
  "${_libnvcc_prebuilt_dir_default}"
  CACHE PATH
  "Path to an extracted prebuilt LLVM release (root containing lib/cmake/llvm)"
)

if (LIBNVCC_USE_PREBUILT_LLVM)
  if (NOT EXISTS "${LIBNVCC_PREBUILT_LLVM_DIR}/lib/cmake/llvm")
    message(
      FATAL_ERROR
      "LIBNVCC_USE_PREBUILT_LLVM is ON but "
      "LIBNVCC_PREBUILT_LLVM_DIR='${LIBNVCC_PREBUILT_LLVM_DIR}' has no lib/cmake/llvm"
    )
  endif()
  # Official LLVM releases are built with zlib+zstd, so the exported LLVM/LLD
  # imported targets reference ZLIB::ZLIB and zstd::libzstd_{static,shared}.
  # Provide those targets from the system libraries before find_package(LLVM).
  find_package(ZLIB REQUIRED)
  if (NOT TARGET zstd::libzstd_shared OR NOT TARGET zstd::libzstd_static)
    find_library(
      _libnvcc_zstd_lib
      NAMES zstd libzstd zstd.1 libzstd.so.1
      PATHS
        /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
        /usr/lib
        /lib/${CMAKE_LIBRARY_ARCHITECTURE}
    )
    if (NOT _libnvcc_zstd_lib)
      set(_libnvcc_zstd_lib "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/libzstd.so.1")
    endif()
    foreach (_zt zstd::libzstd_shared zstd::libzstd_static)
      if (NOT TARGET ${_zt})
        add_library(${_zt} UNKNOWN IMPORTED)
        set_target_properties(
          ${_zt}
          PROPERTIES IMPORTED_LOCATION "${_libnvcc_zstd_lib}"
        )
      endif()
    endforeach()
  endif()
  find_package(
    LLVM REQUIRED CONFIG
    PATHS "${LIBNVCC_PREBUILT_LLVM_DIR}/lib/cmake/llvm"
    NO_DEFAULT_PATH
  )
  find_package(
    Clang REQUIRED CONFIG
    PATHS "${LIBNVCC_PREBUILT_LLVM_DIR}/lib/cmake/clang"
    NO_DEFAULT_PATH
  )
  find_package(
    LLD REQUIRED CONFIG
    PATHS "${LIBNVCC_PREBUILT_LLVM_DIR}/lib/cmake/lld"
    NO_DEFAULT_PATH
  )
  message(
    STATUS
    "libnvcc: using prebuilt LLVM ${LLVM_PACKAGE_VERSION} from ${LIBNVCC_PREBUILT_LLVM_DIR}"
  )
  # All llvm/clang/lld headers live under <prefix>/include in a release.
  set(_libnvcc_llvm_include_dirs "${LIBNVCC_PREBUILT_LLVM_DIR}/include")
  # Clang builtin + CUDA wrapper headers ship under lib/clang/<major>/include.
  set(
    _libnvcc_clang_resource_dir
    "${LIBNVCC_PREBUILT_LLVM_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
  )
  set(
    _libnvcc_clang_headers_dir
    "${LIBNVCC_PREBUILT_LLVM_DIR}/lib/clang/${LLVM_VERSION_MAJOR}/include"
  )
else()
  # List options must be set before CPMAddPackage
  set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "" FORCE)
  # Only the host backend is built, plus NVPTX for device code.  Building every
  # backend would roughly double the size of the shipped library.
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_libnvcc_host_backend "AArch64")
  else()
    set(_libnvcc_host_backend "X86")
  endif()
  set(LLVM_TARGETS_TO_BUILD "${_libnvcc_host_backend};NVPTX" CACHE STRING "" FORCE)

  CPMAddPackage(
    NAME llvm_project
    GIT_REPOSITORY https://github.com/llvm/llvm-project.git
    GIT_TAG ${LIBNVCC_LLVM_VERSION}
    GIT_SHALLOW ON
    SOURCE_SUBDIR llvm
    EXCLUDE_FROM_ALL YES
    OPTIONS
      "LLVM_BUILD_LLVM_C_DYLIB OFF"
      "LLVM_BUILD_TOOLS OFF"
      "LLVM_BUILD_UTILS OFF"
      "LLVM_BUILD_RUNTIME OFF"
      "LLVM_BUILD_RUNTIMES OFF"
      "LLVM_INCLUDE_BENCHMARKS OFF"
      "LLVM_INCLUDE_DOCS OFF"
      "LLVM_INCLUDE_EXAMPLES OFF"
      "LLVM_INCLUDE_RUNTIMES OFF"
      "LLVM_INCLUDE_TESTS OFF"
      "LLVM_INCLUDE_TOOLS ON"
      "LLVM_INCLUDE_UTILS OFF"
      "LLVM_ENABLE_ZLIB OFF"
      "LLVM_ENABLE_ZSTD OFF"
      "LLVM_ENABLE_TERMINFO OFF"
      "LLVM_ENABLE_BINDINGS OFF"
      "CLANG_BUILD_TOOLS OFF"
      "CLANG_ENABLE_ARCMT OFF"
      "CLANG_ENABLE_STATIC_ANALYZER OFF"
  )

  # Ensure the clang resource directory exists
  file(
    MAKE_DIRECTORY "${llvm_project_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
  )

  set(
    _libnvcc_llvm_include_dirs
    ${llvm_project_SOURCE_DIR}/llvm/include
    ${llvm_project_BINARY_DIR}/include
    ${llvm_project_SOURCE_DIR}/clang/include
    ${llvm_project_BINARY_DIR}/tools/clang/include
    ${llvm_project_SOURCE_DIR}/lld/include
    ${llvm_project_BINARY_DIR}/tools/lld/include
  )
  set(
    _libnvcc_clang_resource_dir
    "${llvm_project_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
  )
  set(
    _libnvcc_clang_headers_dir
    "${llvm_project_SOURCE_DIR}/clang/lib/Headers"
  )
endif()

add_library(libnvcc SHARED compiler.cpp)

target_include_directories(
  libnvcc
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${_libnvcc_llvm_include_dirs}
)

target_compile_definitions(
  libnvcc
  PRIVATE
    CLANG_RESOURCE_DIR="${_libnvcc_clang_resource_dir}"
    CLANG_HEADERS_DIR="${_libnvcc_clang_headers_dir}"
    HOSTJIT_INCLUDE_DIR="${_hostjit_include_dir}"
)

# A toolkit is required because we need to call into nvfatbin/nvjitlink.
# FIXME: Statically link against those libraries individually rather than
# requiring a toolkit.
if (NOT CUDAToolkit_FOUND)
  message(FATAL_ERROR "libnvcc requires a CUDA toolkit")
endif()

target_include_directories(libnvcc PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH CUDA_TOOLKIT_ROOT_FROM_CMAKE)
target_compile_definitions(
  libnvcc
  PRIVATE
    CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_ROOT_FROM_CMAKE}"
    CUDA_SDK_VERSION="${CUDAToolkit_VERSION_MAJOR}.0"
)

# Link against LLVM/Clang/LLD
target_link_libraries(
  libnvcc
  PRIVATE
    # LLVM
    LLVMCore
    LLVMSupport
    LLVMIRReader
    LLVMMC
    LLVMObject
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Desc
    LLVMX86Info
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    LLVMLinker
    LLVMPasses
    # Clang
    clangAST
    clangBasic
    clangCodeGen
    clangDriver
    clangFrontend
    clangFrontendTool
    clangLex
    clangParse
    clangSema
    clangEdit
    clangAnalysis
    clangRewrite
    clangSerialization
    # LLD
    $<IF:$<PLATFORM_ID:Windows>,lldCOFF,lldELF>
    lldCommon
)

if (CUDAToolkit_FOUND)
  target_link_libraries(libnvcc PRIVATE CUDA::cuda_driver CUDA::cudart)
  if (WIN32)
    # On Windows, static CUDA libs are built with /MT which conflicts with
    # the project's dynamic CRT (/MD). Use dynamic variants instead.
    target_link_libraries(libnvcc PRIVATE CUDA::nvJitLink CUDA::nvfatbin)
  else()
    # Prefer static CUDA libs on Linux for self-contained binaries. If the
    # toolchain (e.g. lite/pip CUDA installs or some Docker images) only ships
    # the dynamic variants, fall back to those rather than failing configure.
    foreach (_cudalib nvJitLink nvptxcompiler nvfatbin)
      if (TARGET "CUDA::${_cudalib}_static")
        target_link_libraries(libnvcc PRIVATE "CUDA::${_cudalib}_static")
      elseif (TARGET "CUDA::${_cudalib}")
        target_link_libraries(libnvcc PRIVATE "CUDA::${_cudalib}")
      else()
        message(
          FATAL_ERROR
          "libnvcc needs CUDA::${_cudalib}[_static] but neither variant was "
          "found by FindCUDAToolkit. Install the full CUDA toolkit "
          "(libnvjitlink-dev / libnvfatbin-dev or equivalent)."
        )
      endif()
    endforeach()
  endif()
endif()

if (NOT MSVC)
  target_compile_options(libnvcc PRIVATE -fno-rtti)
endif()

set_target_properties(
  libnvcc
  PROPERTIES
    CXX_STANDARD 20
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if (LIBNVCC_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(
    libnvcc
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${LIBNVCC_LIBRARY_OUTPUT_DIRECTORY}"
      ARCHIVE_OUTPUT_DIRECTORY "${LIBNVCC_LIBRARY_OUTPUT_DIRECTORY}"
      RUNTIME_OUTPUT_DIRECTORY "${LIBNVCC_LIBRARY_OUTPUT_DIRECTORY}"
  )
endif()

# On Windows with multi-config generators (Visual Studio), exclude libnvcc from
# Debug builds — the LLVM Debug build causes stack overflows.
if (MSVC)
  set_target_properties(
    libnvcc
    PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD_DEBUG TRUE
  )
endif()

if (NOT LIBNVCC_HEADER_INSTALL_DESTINATION)
  set(LIBNVCC_HEADER_INSTALL_DESTINATION "include/libnvcc")
endif()
if (NOT LIBNVCC_CLANG_HEADER_INSTALL_DESTINATION)
  set(LIBNVCC_CLANG_HEADER_INSTALL_DESTINATION "include/libnvcc/clang")
endif()

install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/libnvcc/libnvcc.h"
  DESTINATION "${LIBNVCC_HEADER_INSTALL_DESTINATION}"
)

# --------------------------------------------------------------------------
# Install the Clang headers required by libnvcc consumers.
# --------------------------------------------------------------------------
# Clang CUDA headers we still use from the LLVM source tree.
# We DON'T install device_functions, math, or libdevice_declares — our local
# copies in cuda_minimal/ replace them.
set(
  _clang_cuda_headers_needed
  "${_libnvcc_clang_headers_dir}/__clang_cuda_math_forward_declares.h"
  "${_libnvcc_clang_headers_dir}/__clang_cuda_builtin_vars.h"
  "${_libnvcc_clang_headers_dir}/__clang_cuda_cmath.h"
  "${_libnvcc_clang_headers_dir}/__clang_cuda_intrinsics.h"
  "${_libnvcc_clang_headers_dir}/__clang_cuda_complex_builtins.h"
  "${_libnvcc_clang_headers_dir}/__clang_cuda_texture_intrinsics.h"
)
install(
  FILES ${_clang_cuda_headers_needed}
  DESTINATION "${LIBNVCC_CLANG_HEADER_INSTALL_DESTINATION}"
)

# Clang builtin C headers needed by our stubs and CUDA toolkit headers.
file(
  GLOB _clang_stddef_headers
  "${_libnvcc_clang_headers_dir}/__stddef_*.h"
)
set(
  _clang_c_headers
  "${_libnvcc_clang_headers_dir}/limits.h"
  "${_libnvcc_clang_headers_dir}/stddef.h"
  "${_libnvcc_clang_headers_dir}/stdint.h"
  "${_libnvcc_clang_headers_dir}/__stddef_header_macro.h"
  "${_libnvcc_clang_headers_dir}/float.h"
  "${_libnvcc_clang_headers_dir}/__float_header_macro.h"
  "${_libnvcc_clang_headers_dir}/inttypes.h"
  ${_clang_stddef_headers}
)
install(
  FILES ${_clang_c_headers}
  DESTINATION "${LIBNVCC_CLANG_HEADER_INSTALL_DESTINATION}"
)
