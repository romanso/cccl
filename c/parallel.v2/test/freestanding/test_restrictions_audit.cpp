// Audit: drive each CFE-101 host-code restriction through the real hostjit
// JIT pipeline (clang host compile + lld --no-undefined link) and confirm it
// fails as expected (compile error or link error). Not a correctness test in
// the usual sense — every case is *expected to fail to build*; the audit fails
// only if a forbidden construct unexpectedly builds, or an allowed one breaks.
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

namespace {

struct Case
{
  const char* id;
  const char* src;
  bool        expect_build_ok; // false = must fail (compile or link)
  const char* expect_marker;   // substring expected in the diagnostic (when failing)
};

// Each snippet puts the offending construct in host code and adds a trivial
// kernel so the device path is satisfied.
const Case cases[] = {
  {"R1 exceptions",
   "__global__ void k(){}\n"
   "extern \"C\" int f(){ throw 1; }\n",
   false, "exception"},

  {"R2 RTTI (typeid)",
   "struct Base { virtual ~Base() = default; };\n"
   "__global__ void k(){}\n"
   "extern \"C\" const void* f(Base* b){ return &typeid(*b); }\n",
   false, "typeid"},

  {"R3 stdlib header",
   "#include <vector>\n"
   "__global__ void k(){}\n",
   false, "file not found"},

  {"R4 heap new/delete",
   "__global__ void k(){}\n"
   "extern \"C\" int* f(){ return new int(5); }\n",
   false, "operator new"},

  {"R4 math (libm)",
   "extern \"C\" double sin(double);\n"
   "__global__ void k(){}\n"
   "extern \"C\" double f(double x){ return sin(x); }\n",
   false, "sin"},

  {"R4 non-trivial static",
   "struct C { C(); ~C(); };\n"
   "__global__ void k(){}\n"
   "extern \"C\" void f(){ static C c; (void)&c; }\n",
   false, "__cxa_atexit"},

  {"R4 thread_local",
   "struct C { C(); ~C(); };\n"
   "__global__ void k(){}\n"
   "extern \"C\" void f(){ thread_local C c; (void)&c; }\n",
   false, "tls"},

  {"R4 complex builtin",
   "__global__ void k(){}\n"
   "extern \"C\" _Complex float f(_Complex float a, _Complex float b){ return a / b; }\n",
   false, "__divsc3"},

  {"R4 wide atomic",
   "unsigned __int128 g;\n"
   "__global__ void k(){}\n"
   "extern \"C\" unsigned __int128 f(){ return __atomic_load_n(&g, __ATOMIC_SEQ_CST); }\n",
   false, "__atomic_load_16"},

  {"R4 pure-virtual key fn",
   "struct B { virtual void f() = 0; virtual ~B(); };\n"
   "B::~B() {}\n"
   "__global__ void k(){}\n"
   "extern \"C\" int use(){ return sizeof(B); }\n",
   false, "__cxa_pure_virtual"},

  // Allowed constructs — must build.
  {"allowed: placement new",
   "typedef __SIZE_TYPE__ size_t;\n"
   "inline void* operator new(size_t, void* p) noexcept { return p; }\n"
   "struct W { int x; W(int v):x(v){} };\n"
   "__global__ void k(){}\n"
   "extern \"C\" int f(){ alignas(W) unsigned char s[sizeof(W)]; W* w = new (s) W(7); return w->x; }\n",
   true, nullptr},

  {"allowed: atexit",
   "extern \"C\" int atexit(void(*)(void));\n"
   "static void on_unload(void){}\n"
   "__global__ void k(){}\n"
   "extern \"C\" int f(){ return atexit(&on_unload); }\n",
   true, nullptr},
};

std::string lower(std::string s)
{
  for (auto& c : s)
  {
    c = (char) ((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c);
  }
  return s;
}

} // namespace

int main()
{
  auto config = hostjit::detectDefaultConfig();

  int failures = 0;
  for (const auto& c : cases)
  {
    hostjit::JITCompiler compiler(config);
    bool ok          = compiler.compile(c.src);
    std::string diag = lower(compiler.getLastError());

    bool pass;
    std::string note;
    if (c.expect_build_ok)
    {
      pass = ok;
      note = ok ? "built (allowed)" : "UNEXPECTEDLY FAILED: " + compiler.getLastError().substr(0, 160);
    }
    else
    {
      bool marker_ok = (c.expect_marker == nullptr) || diag.find(lower(c.expect_marker)) != std::string::npos;
      pass           = (!ok) && marker_ok;
      if (ok)
      {
        note = "UNEXPECTEDLY BUILT (restriction not enforced!)";
      }
      else if (!marker_ok)
      {
        note = std::string("failed but marker '") + c.expect_marker
             + "' not in diagnostic; got: " + compiler.getLastError().substr(0, 200);
      }
      else
      {
        note = std::string("rejected via '") + c.expect_marker + "'";
      }
    }

    std::cout << (pass ? "[ OK ] " : "[FAIL] ") << c.id << "  — " << note << "\n";
    if (!pass)
    {
      ++failures;
    }
  }

  std::cout << "\n" << (failures == 0 ? "ALL RESTRICTIONS ENFORCED AS EXPECTED" : "SOME CASES UNEXPECTED")
            << " (" << failures << " unexpected)\n";
  return failures == 0 ? 0 : 1;
}
