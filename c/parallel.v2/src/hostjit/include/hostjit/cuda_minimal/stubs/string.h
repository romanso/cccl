#ifndef _HOSTJIT_STRING_H
#define _HOSTJIT_STRING_H

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
// REQ-9: clang may lower large copies/zero-fills to out-of-line memcpy/memset/
// memmove/memcmp libcalls. Those are libc functions, not compiler-rt builtins,
// so in the freestanding standalone environment the produced library must
// define them itself. Provide strong out-of-line byte-loop implementations.
// __attribute__((no_builtin)) stops clang's loop-idiom pass from rewriting the
// loops back into the very libcall they implement (which would recurse).
__attribute__((used, no_builtin)) inline void* memcpy(void* __s1, const void* __s2, size_t __n)
{
  unsigned char* __d       = (unsigned char*) __s1;
  const unsigned char* __s = (const unsigned char*) __s2;
  for (size_t __i = 0; __i < __n; ++__i)
  {
    __d[__i] = __s[__i];
  }
  return __s1;
}
__attribute__((used, no_builtin)) inline void* memset(void* __s, int __c, size_t __n)
{
  unsigned char* __d = (unsigned char*) __s;
  for (size_t __i = 0; __i < __n; ++__i)
  {
    __d[__i] = (unsigned char) __c;
  }
  return __s;
}
__attribute__((used, no_builtin)) inline void* memmove(void* __s1, const void* __s2, size_t __n)
{
  unsigned char* __d       = (unsigned char*) __s1;
  const unsigned char* __s = (const unsigned char*) __s2;
  if (__d < __s)
  {
    for (size_t __i = 0; __i < __n; ++__i)
    {
      __d[__i] = __s[__i];
    }
  }
  else if (__d > __s)
  {
    for (size_t __i = __n; __i > 0; --__i)
    {
      __d[__i - 1] = __s[__i - 1];
    }
  }
  return __s1;
}
__attribute__((used, no_builtin)) inline int memcmp(const void* __s1, const void* __s2, size_t __n)
{
  const unsigned char* __a = (const unsigned char*) __s1;
  const unsigned char* __b = (const unsigned char*) __s2;
  for (size_t __i = 0; __i < __n; ++__i)
  {
    if (__a[__i] != __b[__i])
    {
      return (int) __a[__i] - (int) __b[__i];
    }
  }
  return 0;
}
inline char* strchr(char* __s, int __c)
{
  return __builtin_strchr(__s, __c);
}
inline char* strpbrk(char* __s1, const char* __s2)
{
  return __builtin_strpbrk(__s1, __s2);
}
inline char* strrchr(char* __s, int __c)
{
  return __builtin_strrchr(__s, __c);
}
inline void* memchr(void* __s, int __c, size_t __n)
{
  return __builtin_memchr(__s, __c, __n);
}
inline char* strstr(char* __s1, const char* __s2)
{
  return __builtin_strstr(__s1, __s2);
}
inline char* strcpy(char* __s1, const char* __s2)
{
  return __builtin_strcpy(__s1, __s2);
}
inline char* strncpy(char* __s1, const char* __s2, size_t __n)
{
  return __builtin_strncpy(__s1, __s2, __n);
}
inline int strcmp(const char* __s1, const char* __s2)
{
  return __builtin_strcmp(__s1, __s2);
}
inline int strncmp(const char* __s1, const char* __s2, size_t __n)
{
  return __builtin_strncmp(__s1, __s2, __n);
}
inline size_t strlen(const char* __s)
{
  return __builtin_strlen(__s);
}
}
#else // ^^^ __cplusplus ^^^ / vvv !__cplusplus vvv
void* memcpy(void*, const void*, size_t);
void* memset(void*, int, size_t);
int memcmp(const void*, const void*, size_t);
void* memmove(void*, const void*, size_t);
size_t strlen(const char*);
#endif // !__cplusplus

#endif //_HOSTJIT_STRING_H
