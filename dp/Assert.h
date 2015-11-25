// Copyright NVIDIA Corporation 2012
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include <dp/Config.h>

namespace dp
{
  typedef int (*UserAssertCallback)( const char * message );

  DP_API UserAssertCallback setUserAssertCallback( UserAssertCallback callback );
  DP_API void assertionFailed(const char * expr, const char * file, unsigned int line); 
}

#undef DP_ASSERT

#if !defined(NDEBUG)
// Mind the ", 0" after assertionFailed() to get a boolean expression after the ||
#define DP_ASSERT_IMPL(expr) (void)( (!!(expr)) || (dp::assertionFailed(#expr, __FILE__, __LINE__), 0) );
#else
#define DP_ASSERT_IMPL(expr)
#endif

#if defined(DP_OS_WINDOWS) // Windows 32/64-bit
  #include <windows.h>
  #define DP_DBGBREAK() DebugBreak();
#elif defined(DP_OS_LINUX)
  #define DP_DBGBREAK() __builtin_trap();
#else
  #define DP_DBGBREAK()
#endif

#define DP_ASSERT(expr) DP_ASSERT_IMPL(expr)

#if !defined(NDEBUG)
#  define  DP_VERIFY(expr) DP_ASSERT(expr)
#else
#  define  DP_VERIFY(expr) (static_cast<void>(expr))
#endif

// compile time assert
#define DP_STATIC_ASSERT(exp)   static_assert(exp, #exp);
