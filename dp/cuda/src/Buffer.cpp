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


#include <dp/cuda/Buffer.h>

namespace dp
{
  namespace cuda
  {
    BufferSharedPtr Buffer::create( size_t size )
    {
      return( std::shared_ptr<Buffer>( new Buffer( size ) ) );
    }

    Buffer::Buffer( size_t size )
      : m_size( size )
    {
      CUDA_VERIFY( cudaMalloc( &m_devicePointer, m_size ) );
    }

    Buffer::~Buffer( )
    {
      CUDA_VERIFY( cudaFree( m_devicePointer ) );
    }

    void Buffer::getData( void * data, size_t size )
    {
      DP_ASSERT( data && ( size <= m_size ) );
      CUDA_VERIFY( cudaMemcpy( data, m_devicePointer, size, cudaMemcpyDeviceToHost ) );
    }

    void Buffer::getData( size_t bufferOffset, void * data, size_t size )
    {
      DP_ASSERT( data && ( bufferOffset + size <= m_size ) );
      CUDA_VERIFY( cudaMemcpy( data, static_cast<const char *>(m_devicePointer) + bufferOffset, size, cudaMemcpyDeviceToHost ) );
    }

    void Buffer::getData( dp::cuda::BufferHostSharedPtr const& data, dp::cuda::StreamSharedPtr const& stream )
    {
      DP_ASSERT( data && ( data->getSize() <= m_size ) );
      CUDA_VERIFY( cudaMemcpyAsync( data->getPointer<void>(), m_devicePointer, data->getSize(), cudaMemcpyDeviceToHost, stream ? stream->getStream() : nullptr ) );
    }

    void Buffer::getData( dp::cuda::BufferHostSharedPtr const& data, size_t offset, size_t size, dp::cuda::StreamSharedPtr const& stream )
    {
      DP_ASSERT( data && ( offset + size <= data->getSize() ) && ( size <= m_size ) );
      CUDA_VERIFY( cudaMemcpyAsync( data->getPointer<char>() + offset, m_devicePointer, size, cudaMemcpyDeviceToHost, stream ? stream->getStream() : nullptr ) );
    }

    void Buffer::getData( size_t bufferOffset, dp::cuda::BufferHostSharedPtr const& data, size_t offset, size_t size, dp::cuda::StreamSharedPtr const& stream )
    {
      DP_ASSERT( data && ( offset + size <= data->getSize() ) && ( bufferOffset + size <= m_size ) );
      CUDA_VERIFY( cudaMemcpyAsync( data->getPointer<char>() + offset, static_cast<const char *>(m_devicePointer) + bufferOffset, size, cudaMemcpyDeviceToHost, stream ? stream->getStream() : nullptr ) );
    }

    void Buffer::setData( void const* data, size_t size )
    {
      DP_ASSERT( data && ( size <= m_size ) );
      CUDA_VERIFY( cudaMemcpy( m_devicePointer, data, size, cudaMemcpyHostToDevice ) );
    }

    void Buffer::setData( size_t bufferOffset, void const* data, size_t size )
    {
      DP_ASSERT( data && ( bufferOffset + size <= m_size ) );
      CUDA_VERIFY( cudaMemcpy( static_cast<char *>(m_devicePointer) + bufferOffset, data, size, cudaMemcpyHostToDevice ) );
    }

    void Buffer::setData( dp::cuda::BufferHostSharedPtr const& data, dp::cuda::StreamSharedPtr const& stream )
    {
      DP_ASSERT( data && ( data->getSize() <= m_size ) );
      CUDA_VERIFY( cudaMemcpyAsync( m_devicePointer, data->getPointer<void>(), data->getSize(), cudaMemcpyHostToDevice, stream ? stream->getStream() : nullptr ) );
    }

    void Buffer::setData( dp::cuda::BufferHostSharedPtr const& data, size_t offset, size_t size, dp::cuda::StreamSharedPtr const& stream )
    {
      DP_ASSERT( data && ( offset + size <= data->getSize() ) && ( size <= m_size ) );
      CUDA_VERIFY( cudaMemcpyAsync( m_devicePointer, data->getPointer<char>() + offset, size, cudaMemcpyHostToDevice, stream ? stream->getStream() : nullptr ) );
    }

    void Buffer::setData( size_t bufferOffset, dp::cuda::BufferHostSharedPtr const& data, size_t offset, size_t size, dp::cuda::StreamSharedPtr const& stream )
    {
      DP_ASSERT( data && ( offset + size <= data->getSize() ) && ( bufferOffset + size <= m_size ) );
      CUDA_VERIFY( cudaMemcpyAsync( static_cast<char *>(m_devicePointer) + bufferOffset, data->getPointer<char>() + offset, size, cudaMemcpyHostToDevice, stream ? stream->getStream() : nullptr ) );
    }

    void Buffer::fill( int value, size_t count, size_t offset )
    {
      DP_ASSERT( count <= m_size );
      CUDA_VERIFY( cudaMemset( static_cast<char *>(m_devicePointer) + offset, value, count ) );
    }

  } // namespace cuda
} // namespace dp
