// Copyright NVIDIA Corporation 2015
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

#include <dp/util/Config.h>

#if defined(DP_OS_WINDOWS)
// microsoft specific storage-class defines
# if defined(DP_CUDA_EXPORTS)
#  define DP_CUDA_API __declspec(dllexport)
# else
#  define DP_CUDA_API __declspec(dllimport)
# endif
#else
# define DP_CUDA_API
#endif

#include <sstream>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <driver_types.h>

inline void cudaVerify( cudaError_t err, char const* call, char const* file, unsigned int line )
{
  if ( err != cudaSuccess )
  {
    std::ostringstream oss;
    oss << "Error on executing " << call << " in file <" << file << "> line " << line << ": " << cudaGetErrorName( err ) << " : " << cudaGetErrorString( err );
    throw std::runtime_error( oss.str() );
  }
}

#define CUDA_VERIFY( fct )    cudaVerify( fct, #fct, __FILE__, __LINE__ )
