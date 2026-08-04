#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Result codes returned by the libcudacc C API.
 *
 * Detailed compiler, linker, and tool diagnostics are returned in the
 * `program_log` member of cudaccOutput, independently of the result code:
 * warnings can be present on a successful compilation.
 */
typedef enum cudaccResult
{
  CUDACC_SUCCESS = 0,
  CUDACC_ERROR_OUT_OF_MEMORY,
  CUDACC_ERROR_INVALID_INPUT,
  CUDACC_ERROR_INVALID_OPTION,
  CUDACC_ERROR_COMPILATION,
  CUDACC_ERROR_LINKING,
  CUDACC_ERROR_PCH_CREATE,
  CUDACC_ERROR_INTERNAL_ERROR
} cudaccResult;

/**
 * \brief An input file held in memory.
 *
 * `file_name` is the name the compilation refers to the buffer by: whenever an
 * input path given on the command line matches this name, the buffer is used
 * instead of reading the path from disk. `data` points at `size` bytes owned by
 * the caller, which must stay alive for the duration of the cudaccCompile call.
 */
typedef struct cudaccFile
{
  const char* file_name;
  size_t size;
  const void* data;
} cudaccFile;

/**
 * \brief The result of one compilation.
 *
 * `output_data` holds `output_size` bytes of the compilation output and is not
 * guaranteed to be NUL-terminated. When the requested output is written to a
 * file (a host object, a shared library, or a PCH), `output_data` is NULL and
 * `output_size` is zero.
 *
 * `program_log` is a NUL-terminated string with the compiler diagnostics;
 * `program_log_size` is its length in bytes, excluding the terminator.
 *
 * The storage belongs to libcudacc: release it with cudaccDestroyOutput.
 */
typedef struct cudaccOutput
{
  const void* output_data;
  size_t output_size;
  const char* program_log;
  size_t program_log_size;
} cudaccOutput;

/**
 * \brief Return a static string describing a libcudacc result code.
 *
 * The returned pointer is owned by libcudacc and remains valid for the lifetime
 * of the process. Unknown result codes return `"CUDACC_ERROR_UNKNOWN"`.
 */
const char* cudaccGetErrorString(cudaccResult result);

/**
 * \brief Compile CUDA C++ sources, objects, and device code.
 *
 * \param output Receives the compilation output and the diagnostics log. Must
 * be released with cudaccDestroyOutput, including when the call fails: a failed
 * compilation still fills in the log.
 * \param numFiles Number of entries in `files`.
 * \param files In-memory files that shadow input paths of the same name. May be
 * NULL when `numFiles` is zero.
 * \param numOptions Number of entries in `options`.
 * \param options Command-line options, in the shape nvcc accepts them. Inputs
 * are given as paths, positionally; at most one of them may be a source file.
 *
 * Output selection (mutually exclusive; `--ptx`, `--cubin`, `--fatbin`, and
 * `--ltoir` return the artifact in memory, the rest write a file named by
 * `-o <path>`):
 * `--ptx`, `--cubin`, `--fatbin`, `--ltoir`, `-c`, `--shared`,
 * `--gen-pch=<device|host>`.
 *
 * Inputs and search paths:
 * `--ltoir-input <path>`, `--cubin-input <path>`, `--bitcode-input <path>`,
 * `-I <path>`, `-isystem <path>`, `-L <path>`, `-D<name>[=<value>]`,
 * `--use-pch <path>`, `--use-host-pch <path>`.
 *
 * Code generation and diagnostics:
 * `--gpu-architecture=sm_<NN>`, `-O<N>`, `--debug`, `--verbose`,
 * `--trace-includes`, `--keep-artifacts`, `-Xclang <arg>`.
 */
cudaccResult cudaccCompile(
  cudaccOutput* output, int numFiles, const cudaccFile* const* files, int numOptions, const char* const* options);

/**
 * \brief Release the storage held by a cudaccOutput.
 *
 * Accepts NULL and outputs that were never filled in. After the call the
 * members are zeroed, so a second call is harmless.
 */
void cudaccDestroyOutput(cudaccOutput* output);

#ifdef __cplusplus
}
#endif
