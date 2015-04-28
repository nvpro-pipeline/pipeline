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

#include <cuda_runtime_api.h>
#include <dp/cuda/Config.h>
#include <dp/cuda/Types.h>

namespace dp
{
  namespace cuda
  {

    class Stream
    {
      public:
        DP_CUDA_API static StreamSharedPtr create( bool blocking = true, int priority = 0 );
        DP_CUDA_API virtual ~Stream();

        DP_CUDA_API void          addCallback( cudaStreamCallback_t callback, void * userData );
        DP_CUDA_API int           getPriority() const;
        DP_CUDA_API cudaStream_t  getStream() const;
        DP_CUDA_API bool          isBlocking() const;
        DP_CUDA_API bool          isCompleted() const;
        DP_CUDA_API void          synchronize();
        DP_CUDA_API void          wait( EventSharedPtr const& event );

      protected:
        DP_CUDA_API Stream( bool blocking, int priority );

      private:
        bool          m_blocking;
        int           m_priority;
        cudaStream_t  m_stream;
    };

    // to be considered: cudaStreamAttachMemSync -> needs ManagedBuffer
  } // namespace cuda
} // namespace dp
