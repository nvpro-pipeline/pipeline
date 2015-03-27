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

#include <dp/cuda/Config.h>
#include <dp/cuda/BufferHost.h>
#include <dp/cuda/Stream.h>
#include <dp/cuda/Types.h>

namespace dp
{
  namespace cuda
  {

    class Buffer
    {
      public:
        DP_CUDA_API static BufferSharedPtr create( size_t size );
        DP_CUDA_API virtual ~Buffer();

      public:
        DP_CUDA_API void getData( void * data, size_t size );
        DP_CUDA_API void getData( size_t bufferOffset, void * data, size_t size );
        DP_CUDA_API void getData( dp::cuda::BufferHostSharedPtr const& data, dp::cuda::StreamSharedPtr const& stream = dp::cuda::StreamSharedPtr::null );
        DP_CUDA_API void getData( dp::cuda::BufferHostSharedPtr const& data, size_t offset, size_t size, dp::cuda::StreamSharedPtr const& stream = dp::cuda::StreamSharedPtr::null );
        DP_CUDA_API void getData( size_t bufferOffset, dp::cuda::BufferHostSharedPtr const& data, size_t offset, size_t size, dp::cuda::StreamSharedPtr const& stream = dp::cuda::StreamSharedPtr::null );

        DP_CUDA_API void setData( void const* data, size_t size );
        DP_CUDA_API void setData( size_t bufferOffset, void const* data, size_t size );
        DP_CUDA_API void setData( dp::cuda::BufferHostSharedPtr const& data, dp::cuda::StreamSharedPtr const& stream = dp::cuda::StreamSharedPtr::null );
        DP_CUDA_API void setData( dp::cuda::BufferHostSharedPtr const& data, size_t offset, size_t size, dp::cuda::StreamSharedPtr const& stream = dp::cuda::StreamSharedPtr::null );
        DP_CUDA_API void setData( size_t bufferOffset, dp::cuda::BufferHostSharedPtr const& data, size_t offset, size_t size, dp::cuda::StreamSharedPtr const& stream = dp::cuda::StreamSharedPtr::null );

        DP_CUDA_API void fill( int value, size_t count, size_t offset = 0 );

        template <typename T> T const* getDevicePointer() const;
        template <typename T> T * getDevicePointer();
        DP_CUDA_API size_t  getSize() const;

      protected:
        DP_CUDA_API Buffer( size_t size );

      private:
        void    * m_devicePointer;
        size_t    m_size;
    };

    template <typename T>
    inline T const* Buffer::getDevicePointer() const
    {
      return( static_cast<T const*>(m_devicePointer) );
    }

    template <typename T>
    inline T * Buffer::getDevicePointer()
    {
      return( static_cast<T *>(m_devicePointer) );
    }

    inline size_t Buffer::getSize() const
    {
      return( m_size );
    }

  } // namespace cuda
} // namespace dp
