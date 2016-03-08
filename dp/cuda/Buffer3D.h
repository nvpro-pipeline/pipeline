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


#pragma once

#include <dp/cuda/Config.h>
#include <dp/cuda/BufferHost.h>
#include <dp/cuda/Stream.h>
#include <dp/cuda/Types.h>
#include <dp/math/Vecnt.h>

namespace dp
{
  namespace cuda
  {

    class Buffer3D
    {
      public:
        DP_CUDA_API static Buffer3DSharedPtr create( size_t elementSize, dp::math::Vec3ui const& extent );
        DP_CUDA_API virtual ~Buffer3D();

      public:
        DP_CUDA_API void getData( void * dstData
                                , dp::math::Vec3ui const& dstOffset = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& dstStride = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& dstExtent = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& srcOffset = dp::math::Vec3ui( 0, 0, 0 ) );
        DP_CUDA_API void getData( dp::cuda::BufferHostSharedPtr const& dstBuffer
                                , dp::math::Vec3ui const& dstOffset = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& dstStride = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& dstExtent = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& srcOffset = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::cuda::StreamSharedPtr const& stream = dp::cuda::StreamSharedPtr() );

        DP_CUDA_API void setData( void const* srcData
                                , dp::math::Vec3ui const& srcOffset = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& srcStride = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& srcExtent = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& dstOffset = dp::math::Vec3ui( 0, 0, 0 ) );
        DP_CUDA_API void setData( dp::cuda::BufferHostSharedPtr const& srcBuffer
                                , dp::math::Vec3ui const& srcOffset = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& srcStride = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& srcExtent = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::math::Vec3ui const& dstOffset = dp::math::Vec3ui( 0, 0, 0 )
                                , dp::cuda::StreamSharedPtr const& stream = dp::cuda::StreamSharedPtr() );

        DP_CUDA_API void fill( int value
                             , dp::math::Vec3ui const& offset = dp::math::Vec3ui( 0, 0, 0 )
                             , dp::math::Vec3ui const& extent = dp::math::Vec3ui( 0, 0, 0 ) );

        template <typename T> T const*        getDevicePointer() const;
        template <typename T> T *             getDevicePointer();
        DP_CUDA_API cudaPitchedPtr const&     getPitchedPointer() const;
        DP_CUDA_API size_t                    getElementSize() const;
        DP_CUDA_API dp::math::Vec3ui const&   getExtent() const;

      protected:
        DP_CUDA_API Buffer3D( size_t elementSize, dp::math::Vec3ui const& extent );

      private:
        cudaPitchedPtr    m_pitchedPtr;
        size_t            m_elementSize;
        dp::math::Vec3ui  m_extent;
    };

    template <typename T>
    inline T const* Buffer3D::getDevicePointer() const
    {
      return( static_cast<T const*>(m_pitchedPtr.ptr) );
    }

    template <typename T>
    inline T * Buffer3D::getDevicePointer()
    {
      return( static_cast<T *>(m_pitchedPtr.ptr) );
    }

    inline size_t Buffer3D::getElementSize() const
    {
      return( m_elementSize );
    }

  } // namespace cuda
} // namespace dp
