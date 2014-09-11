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

#if defined(_MSC_VER)
#define DP_COMPILER_MSVC 1

#if defined(_WIN64)
#define DP_ARCH_X86_64
#else
#define DP_ARCH_X86
#endif

#endif

#if defined(__GNUC__)
#define DP_COMPILER_GCC 1

#if defined(_LP64)
#define DP_ARCH_X86_64
#elif defined(__ARMEL__)
#define DP_ARCH_ARM_32
#elif defined(__i386__)
#define DP_ARCH_X86
#endif

#endif

#if defined(_WIN32)
#define DP_OS_WINDOWS
#endif

#if defined(__MINGW32__)
#define DP_OS_WINDOWS
#define DP_POSIX
#endif

#if defined(__linux__)
#define DP_OS_LINUX
#define DP_POSIX

# define _stricmp strcasecmp
#endif


#ifdef DP_OS_WINDOWS
// microsoft specific storage-class defines
# ifdef DP_UTIL_EXPORTS
#  define DP_UTIL_API __declspec(dllexport)
# else
#  define DP_UTIL_API __declspec(dllimport)
# endif
#else
# define DP_UTIL_API
#endif

// controls whether the RCObject class contains a unique id per object, including a map id->RCObject*
#define DPUTIL_RCOBJECT_IDS 0

