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


#include <dp/cuda/Buffer3D.h>
#include <driver_functions.h>

namespace dp
{
  namespace cuda
  {
    Buffer3DSharedPtr Buffer3D::create( size_t elementSize, dp::math::Vec3ui const& extent )
    {
      return( std::shared_ptr<Buffer3D>( new Buffer3D( elementSize, extent ) ) );
    }

    Buffer3D::Buffer3D( size_t elementSize, dp::math::Vec3ui const& extent )
      : m_elementSize( elementSize )
      , m_extent( extent )
    {
      CUDA_VERIFY( cudaMalloc3D( &m_pitchedPtr, make_cudaExtent( m_elementSize * m_extent[0], m_extent[1], m_extent[2] ) ) );
    }

    Buffer3D::~Buffer3D( )
    {
      CUDA_VERIFY( cudaFree( m_pitchedPtr.ptr ) );
    }

    void Buffer3D::getData( void * dstData, dp::math::Vec3ui const& dstOffset, dp::math::Vec3ui const& dstStride, dp::math::Vec3ui const& dstExtent, dp::math::Vec3ui const& srcOffset )
    {
#if !defined(NDEBUG)
      for ( int i=0 ; i<3 ; i++ )
      {
        DP_ASSERT( dstOffset[i] + ( dstExtent[i] ? dstExtent[i] : m_extent[i] ) <= m_extent[i] );
        DP_ASSERT( srcOffset[i] + ( dstExtent[i] ? dstExtent[i] : m_extent[i] ) <= m_extent[i] );
      }
#endif
      cudaMemcpy3DParms parms = { 0 };
      parms.srcPos = make_cudaPos( m_elementSize * srcOffset[0], srcOffset[1], srcOffset[2] );
      parms.srcPtr = m_pitchedPtr;
      parms.dstPos = make_cudaPos( m_elementSize * dstOffset[0], dstOffset[1], dstOffset[2] );
      parms.dstPtr = make_cudaPitchedPtr( dstData, m_elementSize * ( dstStride[0] ? dstStride[0] : m_extent[0] ), dstStride[0] ? dstStride[0] : m_extent[0], dstStride[1] ? dstStride[1] : m_extent[1] );
      parms.extent = make_cudaExtent( m_elementSize * ( dstExtent[0] ? dstExtent[0] : m_extent[0] ), dstExtent[1] ? dstExtent[1] : m_extent[1], dstExtent[2] ? dstExtent[2] : m_extent[2] );
      parms.kind = cudaMemcpyDeviceToHost;
      CUDA_VERIFY( cudaMemcpy3D( &parms ) );
    }

    void Buffer3D::getData( dp::cuda::BufferHostSharedPtr const& dstBuffer, dp::math::Vec3ui const& dstOffset, dp::math::Vec3ui const& dstStride, dp::math::Vec3ui const& dstExtent
                          , dp::math::Vec3ui const& srcOffset, dp::cuda::StreamSharedPtr const& stream )
    {
#if !defined(NDEBUG)
      for ( int i=0 ; i<3 ; i++ )
      {
        DP_ASSERT( dstOffset[i] + ( dstExtent[i] ? dstExtent[i] : m_extent[i] ) <= m_extent[i] );
        DP_ASSERT( srcOffset[i] + ( dstExtent[i] ? dstExtent[i] : m_extent[i] ) <= m_extent[i] );
      }
#endif
      cudaMemcpy3DParms parms = { 0 };
      parms.srcPos = make_cudaPos( m_elementSize * srcOffset[0], srcOffset[1], srcOffset[2] );
      parms.srcPtr = m_pitchedPtr;
      parms.dstPos = make_cudaPos( m_elementSize * dstOffset[0], dstOffset[1], dstOffset[2] );
      parms.dstPtr = make_cudaPitchedPtr( dstBuffer->getPointer<void>(), m_elementSize * ( dstStride[0] ? dstStride[0] : m_extent[0] ), dstStride[0] ? dstStride[0] : m_extent[0], dstStride[1] ? dstStride[1] : m_extent[1] );
      parms.extent = make_cudaExtent( m_elementSize * ( dstExtent[0] ? dstExtent[0] : m_extent[0] ), dstExtent[1] ? dstExtent[1] : m_extent[1], dstExtent[2] ? dstExtent[2] : m_extent[2] );
      parms.kind = cudaMemcpyDeviceToHost;
      CUDA_VERIFY( cudaMemcpy3DAsync( &parms ) );
    }

    void Buffer3D::setData( void const* srcData, dp::math::Vec3ui const& srcOffset, dp::math::Vec3ui const& srcStride, dp::math::Vec3ui const& srcExtent, dp::math::Vec3ui const& dstOffset )
    {
#if !defined(NDEBUG)
      for ( int i=0 ; i<3 ; i++ )
      {
        DP_ASSERT( srcOffset[i] + ( srcExtent[i] ? srcExtent[i] : m_extent[i] ) <= m_extent[i] );
        DP_ASSERT( dstOffset[i] + ( srcExtent[i] ? srcExtent[i] : m_extent[i] ) <= m_extent[i] );
      }
#endif
      cudaMemcpy3DParms parms = { 0 };
      parms.srcPos = make_cudaPos( m_elementSize * srcOffset[0], srcOffset[1], srcOffset[2] );
      parms.srcPtr = make_cudaPitchedPtr( const_cast<void *>(srcData), m_elementSize * ( srcStride[0] ? srcStride[0] : m_extent[0] ), srcStride[0] ? srcStride[0] : m_extent[0], srcStride[1] ? srcStride[1] : m_extent[1] );
      parms.dstPos = make_cudaPos( m_elementSize * dstOffset[0], dstOffset[1], dstOffset[2] );
      parms.dstPtr = m_pitchedPtr;
      parms.extent = make_cudaExtent( m_elementSize * ( srcExtent[0] ? srcExtent[0] : m_extent[0] ), srcExtent[1] ? srcExtent[1] : m_extent[1], srcExtent[2] ? srcExtent[2] : m_extent[2] );
      parms.kind = cudaMemcpyHostToDevice;
      CUDA_VERIFY( cudaMemcpy3D( &parms ) );
    }

    void Buffer3D::setData( dp::cuda::BufferHostSharedPtr const& srcBuffer, dp::math::Vec3ui const& srcOffset, dp::math::Vec3ui const& srcStride, dp::math::Vec3ui const& srcExtent
                          , dp::math::Vec3ui const& dstOffset, dp::cuda::StreamSharedPtr const& stream )
    {
#if !defined(NDEBUG)
      for ( int i=0 ; i<3 ; i++ )
      {
        DP_ASSERT( srcOffset[i] + ( srcExtent[i] ? srcExtent[i] : m_extent[i] ) <= m_extent[i] );
        DP_ASSERT( dstOffset[i] + ( srcExtent[i] ? srcExtent[i] : m_extent[i] ) <= m_extent[i] );
      }
#endif
      cudaMemcpy3DParms parms = { 0 };
      parms.srcPos = make_cudaPos( m_elementSize * srcOffset[0], srcOffset[1], srcOffset[2] );
      parms.srcPtr = make_cudaPitchedPtr( srcBuffer->getPointer<void>(), m_elementSize * ( srcStride[0] ? srcStride[0] : m_extent[0] ), srcStride[0] ? srcStride[0] : m_extent[0], srcStride[1] ? srcStride[1] : m_extent[1] );
      parms.dstPos = make_cudaPos( m_elementSize * dstOffset[0], dstOffset[1], dstOffset[2] );
      parms.dstPtr = m_pitchedPtr;
      parms.extent = make_cudaExtent( m_elementSize * ( srcExtent[0] ? srcExtent[0] : m_extent[0] ), srcExtent[1] ? srcExtent[1] : m_extent[1], srcExtent[2] ? srcExtent[2] : m_extent[2] );
      parms.kind = cudaMemcpyHostToDevice;
      CUDA_VERIFY( cudaMemcpy3DAsync( &parms ) );
    }

    void Buffer3D::fill( int value, dp::math::Vec3ui const& offset, dp::math::Vec3ui const& extent )
    {
#if !defined(NDEBUG)
      for ( int i=0 ; i<3 ; i++ )
      {
        DP_ASSERT( offset[i] + ( extent[i] ? extent[i] : m_extent[i] ) <= m_extent[i] );
      }
#endif
      DP_ASSERT( ( offset[0] == 0 ) && ( offset[1] == 0 ) && ( offset[2] == 0 ) );
      CUDA_VERIFY( cudaMemset3D( m_pitchedPtr, value, make_cudaExtent( m_elementSize * ( extent[0] ? extent[0] : m_extent[0] ), extent[1] ? extent[1] : m_extent[1], extent[2] ? extent[2] : m_extent[2] ) ) );
    }

    cudaPitchedPtr const& Buffer3D::getPitchedPointer() const
    {
      return( m_pitchedPtr );
    }

    dp::math::Vec3ui const& Buffer3D::getExtent() const
    {
      return( m_extent );
    }

  } // namespace cuda
} // namespace dp
