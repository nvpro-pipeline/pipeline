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

#include <boost/noncopyable.hpp>
#include <dp/cuda/Config.h>
#include <dp/cuda/Stream.h>
#include <dp/cuda/Types.h>
#include <dp/gl/Buffer.h>

namespace dp
{
  namespace cuda
  {

    class GraphicsResource
    {
      public:
        DP_CUDA_API static GraphicsResourceSharedPtr create( dp::gl::BufferSharedPtr const& buffer, unsigned int flags );
        DP_CUDA_API virtual ~GraphicsResource();

      public:
        DP_CUDA_API dp::gl::BufferSharedPtr const&  getBuffer() const;
        DP_CUDA_API cudaGraphicsResource_t          getResource();

      protected:
        DP_CUDA_API GraphicsResource( dp::gl::BufferSharedPtr const& buffer, unsigned int flags );

      private:
        dp::gl::BufferSharedPtr m_buffer;
        cudaGraphicsResource_t  m_resource;
    };

    class MappedGraphicsResource : public boost::noncopyable
    {
      public:
        DP_CUDA_API MappedGraphicsResource( GraphicsResourceSharedPtr const& resource, StreamSharedPtr const& stream = StreamSharedPtr() );
        DP_CUDA_API ~MappedGraphicsResource();

      public:
        template <typename T> T * getDevicePointer();

      private:
        GraphicsResourceSharedPtr   m_resource;
        StreamSharedPtr             m_stream;
        void                      * m_devicePointer;
    };

    template <typename T>
    inline T * MappedGraphicsResource::getDevicePointer()
    {
      return( static_cast<T *>( m_devicePointer ) );
    }

  } // namespace cuda
} // namespace dp
