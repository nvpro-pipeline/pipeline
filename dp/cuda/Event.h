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

#include <driver_types.h>
#include <dp/cuda/Config.h>
#include <dp/cuda/Types.h>

namespace dp
{
  namespace cuda
  {

    class Event
    {
      friend class Stream;
      friend DP_CUDA_API float getElapsedTime( EventSharedPtr const& startEvent, EventSharedPtr const& stopEvent );

      public:
        DP_CUDA_API static EventSharedPtr create( unsigned int flags = cudaEventDefault );
        DP_CUDA_API virtual ~Event();

      public:
        DP_CUDA_API bool isBlocking() const;
        DP_CUDA_API bool isCompleted() const;
        DP_CUDA_API bool isTimingDisabled() const;
        DP_CUDA_API void record( StreamSharedPtr const& stream = StreamSharedPtr::null );
        DP_CUDA_API void synchronize();

      protected:
        DP_CUDA_API Event( unsigned int flags );
        DP_CUDA_API cudaEvent_t getEvent() const;

      private:
        cudaEvent_t   m_event;
        unsigned int  m_flags;
    };

    DP_CUDA_API float getElapsedTime( EventSharedPtr const& startEvent, EventSharedPtr const& stopEvent );

  } // namespace cuda
} // namespace dp
