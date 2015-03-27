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


#include <dp/cuda/Event.h>
#include <dp/cuda/Stream.h>

namespace dp
{
  namespace cuda
  {
    StreamSharedPtr Stream::create( bool blocking, int priority )
    {
      return( std::shared_ptr<Stream>( new Stream( blocking, priority ) ) );
    }

    Stream::Stream( bool blocking, int priority )
      : m_blocking( blocking )
    {
      CUDA_VERIFY( cudaStreamCreateWithPriority( &m_stream, m_blocking ? cudaStreamDefault : cudaStreamNonBlocking, priority ) );
      CUDA_VERIFY( cudaStreamGetPriority( m_stream, &m_priority ) );
    }

    Stream::~Stream( )
    {
      CUDA_VERIFY( cudaStreamDestroy( m_stream ) );
    }

    void Stream::addCallback( cudaStreamCallback_t callback, void * userData )
    {
#if !defined(NDEBUG)
      int device;
      CUDA_VERIFY( cudaGetDevice( &device ) );
      int majorCap, minorCap;
      CUDA_VERIFY( cudaDeviceGetAttribute( &majorCap, cudaDevAttrComputeCapabilityMajor, device ) );
      CUDA_VERIFY( cudaDeviceGetAttribute( &minorCap, cudaDevAttrComputeCapabilityMinor, device ) );
      DP_ASSERT( ( 1 < majorCap ) || ( ( 1 == majorCap ) && ( 1 <= minorCap ) ) );
#endif
      CUDA_VERIFY( cudaStreamAddCallback( m_stream, callback, userData, 0 ) );
    }

    int Stream::getPriority() const
    {
      return( m_priority );
    }

    cudaStream_t Stream::getStream() const
    {
      return( m_stream );
    }

    bool Stream::isBlocking() const
    {
      return( m_blocking );
    }

    bool Stream::isCompleted() const
    {
      cudaError_t err = cudaStreamQuery( m_stream );
      DP_ASSERT( ( err == cudaSuccess ) || ( err == cudaErrorNotReady ) );
      return( err == cudaSuccess );
    }

    void Stream::synchronize()
    {
      CUDA_VERIFY( cudaStreamSynchronize( m_stream ) );
    }

    void Stream::wait( EventSharedPtr const& event )
    {
      CUDA_VERIFY( cudaStreamWaitEvent( m_stream, event->getEvent(), 0 ) );
    }

  } // namespace cuda
} // namespace dp
