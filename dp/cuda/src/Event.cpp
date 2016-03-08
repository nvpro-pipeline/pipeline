// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/Assert.h>
#include <dp/cuda/Config.h>
#include <dp/cuda/Event.h>
#include <dp/cuda/Stream.h>

namespace dp
{
  namespace cuda
  {
    EventSharedPtr Event::create( unsigned int flags )
    {
      return( std::shared_ptr<Event>( new Event( flags ) ) );
    }

    Event::Event( unsigned int flags )
      : m_flags( flags )
    {
      DP_ASSERT((m_flags & ~(cudaEventDefault | cudaEventBlockingSync | cudaEventDisableTiming | cudaEventInterprocess)) == 0);
      CUDA_VERIFY( cudaEventCreateWithFlags( &m_event, m_flags ) );
    }

    Event::~Event( )
    {
      CUDA_VERIFY( cudaEventDestroy( m_event ) );
    }

    bool Event::isBlocking() const
    {
      return( !!(m_flags & cudaEventBlockingSync) );
    }

    bool Event::isCompleted() const
    {
      cudaError_t err = cudaEventQuery( m_event );
      DP_ASSERT( ( err == cudaSuccess ) || ( err == cudaErrorNotReady ) );
      return( err == cudaSuccess );
    }

    bool Event::isTimingDisabled() const
    {
      return( !!(m_flags & cudaEventDisableTiming) );
    }

    void Event::record( StreamSharedPtr const& stream )
    {
      CUDA_VERIFY( cudaEventRecord( m_event, stream ? stream->getStream() : nullptr ) );
    }

    void Event::synchronize()
    {
      CUDA_VERIFY( cudaEventSynchronize( m_event ) );
    }

    cudaEvent_t Event::getEvent() const
    {
      return( m_event );
    }

    float getElapsedTime( EventSharedPtr const& startEvent, EventSharedPtr const& stopEvent )
    {
      float t;
      CUDA_VERIFY( cudaEventElapsedTime( &t, startEvent->getEvent(), stopEvent->getEvent() ) );
      return( t );
    }

  } // namespace cuda
} // namespace dp
