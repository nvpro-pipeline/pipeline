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


#include <dp/cuda/Config.h>
#include <GL/glew.h>
#if defined( DP_OS_WINDOWS)
#include <GL/wglew.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <dp/cuda/GraphicsResource.h>


namespace dp
{
  namespace cuda
  {
    GraphicsResourceSharedPtr GraphicsResource::create( dp::gl::BufferSharedPtr const& buffer, unsigned int flags )
    {
      return( std::shared_ptr<GraphicsResource>( new GraphicsResource( buffer, flags ) ) );
    }

    GraphicsResource::GraphicsResource(  dp::gl::BufferSharedPtr const& buffer, unsigned int flags  )
      : m_buffer( buffer )
    {
      DP_ASSERT( ( flags & ~( cudaGraphicsMapFlagsNone | cudaGraphicsRegisterFlagsReadOnly | cudaGraphicsRegisterFlagsWriteDiscard ) ) == 0 )
      CUDA_VERIFY( cudaGraphicsGLRegisterBuffer( &m_resource, m_buffer->getGLId(), flags ) );
    }

    GraphicsResource::~GraphicsResource( )
    {
      cudaError_t err = cudaGraphicsUnregisterResource( m_resource );
      DP_ASSERT( err == cudaSuccess );
    }

    dp::gl::BufferSharedPtr const& GraphicsResource::getBuffer() const
    {
      return( m_buffer );
    }

    cudaGraphicsResource_t GraphicsResource::getResource()
    {
      return( m_resource );
    }


    MappedGraphicsResource::MappedGraphicsResource( GraphicsResourceSharedPtr const& resource, StreamSharedPtr const& stream )
      : m_resource( resource )
      , m_stream( stream )
    {
      cudaGraphicsResource_t res = m_resource->getResource();
      CUDA_VERIFY( cudaGraphicsMapResources( 1, &res, m_stream ? m_stream->getStream() : nullptr ) );

      size_t size;
      CUDA_VERIFY( cudaGraphicsResourceGetMappedPointer( &m_devicePointer, &size, res ) );

      DP_ASSERT( size == resource->getBuffer()->getSize() );
    }

    MappedGraphicsResource::~MappedGraphicsResource()
    {
      cudaGraphicsResource_t res = m_resource->getResource();
      CUDA_VERIFY( cudaGraphicsUnmapResources( 1, &res, m_stream ? m_stream->getStream() : nullptr ) );
    }

  } // namespace cuda
} // namespace dp
