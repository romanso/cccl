// Broad REQ-9 audit: JIT-compile a wide slice of CCCL through the real hostjit
// pipeline and confirm the produced shared library leaves no undefined symbol
// outside CUDART.
//
// The restrictions audit next door proves each forbidden construct is caught in
// isolation. This one asks the complementary question: whether ordinary CCCL
// code, with all the templates and headers it drags in, stays inside the
// contract. It reads the produced ELF directly rather than shelling out, so the
// same binary answers the question on x86_64 and on aarch64.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <elf.h>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

namespace {

// Several device algorithms with different value types and operators, so the
// instantiation covers block-level primitives, temp-storage sizing, occupancy
// queries and the launch stubs rather than a single code path.
const char* cuda_source = R"(
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

struct MaxOp {
    template <typename T>
    __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? b : a; }
};

struct IsEven {
    __device__ __forceinline__ bool operator()(int v) const { return (v & 1) == 0; }
};

extern "C" _CCCL_VISIBILITY_EXPORT
cudaError_t audit_reduce(void* tmp, size_t& bytes, int* in, int* out, int n) {
    return cub::DeviceReduce::Reduce(tmp, bytes, in, out, n, MaxOp{}, 0);
}

extern "C" _CCCL_VISIBILITY_EXPORT
cudaError_t audit_reduce_double(void* tmp, size_t& bytes, double* in, double* out, int n) {
    return cub::DeviceReduce::Sum(tmp, bytes, in, out, n);
}

extern "C" _CCCL_VISIBILITY_EXPORT
cudaError_t audit_scan(void* tmp, size_t& bytes, float* in, float* out, int n) {
    return cub::DeviceScan::InclusiveSum(tmp, bytes, in, out, n);
}

extern "C" _CCCL_VISIBILITY_EXPORT
cudaError_t audit_sort(void* tmp, size_t& bytes, unsigned long long* kin,
                       unsigned long long* kout, int n) {
    return cub::DeviceRadixSort::SortKeys(tmp, bytes, kin, kout, n);
}

extern "C" _CCCL_VISIBILITY_EXPORT
cudaError_t audit_select(void* tmp, size_t& bytes, int* in, int* out, int* num_out, int n) {
    return cub::DeviceSelect::If(tmp, bytes, in, out, num_out, n, IsEven{});
}
)";

// CUDART is the only external dependency REQ-9 allows. Entry points are either
// the public cuda* API or the __cuda* registration helpers the compiler emits.
bool is_cudart(const std::string& name)
{
  return name.rfind("cuda", 0) == 0 || name.rfind("__cuda", 0) == 0;
}

// Collect undefined entries of .dynsym. Both REQ-8 Linux targets are ELF64
// little-endian, so one reader covers them.
bool read_undefined_symbols(const std::string& path, std::vector<std::string>& out, std::string& error)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
  {
    error = "cannot open " + path;
    return false;
  }
  std::vector<char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (buf.size() < sizeof(Elf64_Ehdr) || std::memcmp(buf.data(), ELFMAG, SELFMAG) != 0)
  {
    error = "not an ELF file: " + path;
    return false;
  }

  const auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(buf.data());
  const auto* shdr = reinterpret_cast<const Elf64_Shdr*>(buf.data() + ehdr->e_shoff);

  for (unsigned i = 0; i < ehdr->e_shnum; ++i)
  {
    if (shdr[i].sh_type != SHT_DYNSYM)
    {
      continue;
    }
    const auto* sym    = reinterpret_cast<const Elf64_Sym*>(buf.data() + shdr[i].sh_offset);
    const char* strtab = buf.data() + shdr[shdr[i].sh_link].sh_offset;
    const size_t count = shdr[i].sh_size / sizeof(Elf64_Sym);

    for (size_t s = 0; s < count; ++s)
    {
      if (sym[s].st_shndx != SHN_UNDEF || sym[s].st_name == 0)
      {
        continue;
      }
      out.emplace_back(strtab + sym[s].st_name);
    }
    return true;
  }

  error = "no .dynsym in " + path;
  return false;
}

std::string find_shared_library(const std::string& dir)
{
  for (const char* name : {"/libcuda_code.so", "/cuda_code.so"})
  {
    std::ifstream probe(dir + name, std::ios::binary);
    if (probe)
    {
      return dir + name;
    }
  }
  return {};
}

} // namespace

int main()
{
  // The built-in source is a fixed sample. HOSTJIT_AUDIT_SOURCE points the same
  // check at arbitrary CCCL code without rebuilding, which is how a wider slice
  // gets audited as coverage grows.
  std::string source = cuda_source;
  if (const char* override_path = std::getenv("HOSTJIT_AUDIT_SOURCE"))
  {
    std::ifstream src(override_path);
    if (!src)
    {
      std::cout << "[FAIL] cannot read HOSTJIT_AUDIT_SOURCE=" << override_path << "\n";
      return 1;
    }
    source.assign((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
    std::cout << "source: " << override_path << "\n";
  }

  auto config           = hostjit::detectDefaultConfig();
  config.keep_artifacts = true;

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(source))
  {
    std::cout << "[FAIL] broad CCCL source did not build\n" << compiler.getLastError() << "\n";
    return 1;
  }

  const std::string artifacts = compiler.getArtifactsPath();
  const std::string so        = find_shared_library(artifacts);
  std::cout << "artifacts: " << artifacts << "\n";
  if (so.empty())
  {
    std::cout << "[FAIL] no shared library in the artifacts directory\n";
    return 1;
  }

  std::vector<std::string> undefined;
  std::string error;
  if (!read_undefined_symbols(so, undefined, error))
  {
    std::cout << "[FAIL] " << error << "\n";
    return 1;
  }

  std::vector<std::string> offenders;
  for (const auto& name : undefined)
  {
    if (!is_cudart(name))
    {
      offenders.push_back(name);
    }
  }

  std::cout << "undefined symbols: " << undefined.size() << " (" << undefined.size() - offenders.size()
            << " CUDART)\n";
  for (const auto& name : undefined)
  {
    std::cout << "    " << (is_cudart(name) ? "     " : "NOT  ") << name << "\n";
  }

  if (!offenders.empty())
  {
    std::cout << "\n[FAIL] " << offenders.size() << " undefined symbol(s) outside CUDART\n";
    return 1;
  }

  std::cout << "\n[ OK ] only CUDART left undefined — REQ-9 holds for this source\n";
  return 0;
}
