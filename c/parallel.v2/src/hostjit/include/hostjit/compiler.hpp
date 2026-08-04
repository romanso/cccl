#pragma once

#include <string>
#include <vector>

#include <cudacc/cudacc.h>

namespace hostjit::detail
{
// Owns one cudaccOutput. The output holds the diagnostics even when the
// compilation failed, so it is read first and released on the way out.
struct CudaccOutput
{
  cudaccOutput output{};

  CudaccOutput()                               = default;
  CudaccOutput(const CudaccOutput&)            = delete;
  CudaccOutput& operator=(const CudaccOutput&) = delete;

  ~CudaccOutput()
  {
    cudaccDestroyOutput(&output);
  }

  std::string log() const
  {
    return output.program_log ? std::string(output.program_log, output.program_log_size) : std::string();
  }

  std::vector<char> data() const
  {
    const char* bytes = static_cast<const char*>(output.output_data);
    return bytes ? std::vector<char>(bytes, bytes + output.output_size) : std::vector<char>();
  }
};

inline std::vector<const char*> make_cudacc_option_ptrs(const std::vector<std::string>& options)
{
  std::vector<const char*> ptrs;
  ptrs.reserve(options.size());
  for (const auto& option : options)
  {
    ptrs.push_back(option.c_str());
  }
  return ptrs;
}

// A source string handed to cudaccCompile as an in-memory file, named so the
// command line can refer to it.
inline cudaccFile make_cudacc_source(const char* name, const std::string& source)
{
  return cudaccFile{name, source.size(), source.data()};
}
} // namespace hostjit::detail
