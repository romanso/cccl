// CFE-102 demonstration: a host construct that forces a compiler-rt builtin
// (_Complex float division -> __divsc3). Under the strict --no-undefined link
// it fails unless the compiler-rt builtins archive is supplied. Drive it through
// the real hostjit pipeline and report build result; with HOSTJIT_LINK_BUILTINS
// set, the archive is linked and the helper resolves in-image.
#include <cstdlib>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

int main()
{
  auto config           = hostjit::detectDefaultConfig();
  config.keep_artifacts = true;

  const char* src =
    "__global__ void k(){}\n"
    "extern \"C\" _Complex float f(_Complex float a, _Complex float b){ return a / b; }\n";

  hostjit::JITCompiler compiler(config);
  bool ok = compiler.compile(src);

  const char* want = std::getenv("HOSTJIT_LINK_BUILTINS");
  std::cout << "HOSTJIT_LINK_BUILTINS=" << (want ? want : "(unset)") << "\n";
  std::cout << "complex-division (emits __divsc3): " << (ok ? "BUILT" : "FAILED") << "\n";
  if (ok)
  {
    std::cout << "artifacts: " << compiler.getArtifactsPath() << "\n";
  }
  else
  {
    std::cout << "diag: " << compiler.getLastError().substr(0, 300) << "\n";
  }
  return ok ? 0 : 1;
}
