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


#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <dp/Assert.h>
#include <dp/cuda/Config.h>
#include <dp/cuda/BufferHost.h>

namespace dp
{
  namespace cuda
  {
    BufferHostSharedPtr BufferHost::create( size_t size, unsigned int flags )
    {
      return( std::shared_ptr<BufferHost>( new BufferHost( size, flags ) ) );
    }

    BufferHost::BufferHost( size_t size, unsigned int flags )
      : m_isMapped( !!( flags & cudaHostAllocMapped ) )
#if !defined(NDEBUG)
      , m_isPortable( !!( flags & cudaHostAllocPortable ) )
#endif
      , m_size( size )
    {
      DP_ASSERT( ( flags & ( cudaHostAllocDefault | cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined ) ) == flags );
      CUDA_VERIFY( cudaHostAlloc( &m_pointer, m_size, flags ) );
#if !defined(NDEBUG)
      if ( m_isMapped && !m_isPortable )
      {
        CUDA_VERIFY( cudaGetDevice( &m_deviceID ) );
      }
#endif
    }

    BufferHost::~BufferHost( )
    {
      CUDA_VERIFY( cudaFreeHost( m_pointer ) );
    }

  } // namespace cuda
} // namespace dp
