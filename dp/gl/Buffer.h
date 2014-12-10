// Copyright NVIDIA Corporation 2010
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

#include <dp/gl/Config.h>
#include <dp/gl/Object.h>

namespace dp
{
  namespace gl
  {
    class Buffer : public Object
    {
      public:
        DP_GL_API static BufferSharedPtr create();
        DP_GL_API static BufferSharedPtr create( GLenum target, size_t size, GLvoid const* data, GLenum usage );
        DP_GL_API virtual ~Buffer();

      public:
        DP_GL_API GLuint64EXT getAddress();
        DP_GL_API size_t getSize() const;
        DP_GL_API void setData( GLenum target, size_t size, GLvoid const* data, GLenum usage );
        DP_GL_API void setSubData( GLenum target, size_t offset, size_t size, GLvoid const* data );
        DP_GL_API void* map( GLenum target, GLenum access );
        DP_GL_API void* mapRange( GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access );
        DP_GL_API GLboolean unmap( GLenum target );

      protected:
        DP_GL_API Buffer();

      private:
        GLuint64EXT m_address; // 64-bit bindless address
        size_t      m_size;
    };


    DP_GL_API void bind( GLenum target, BufferSharedPtr const& buffer );
    DP_GL_API void copy( BufferSharedPtr const& src, BufferSharedPtr const& dst, size_t srcOffset, size_t dstOffset, size_t size );


    template <typename T>
    class MappedBuffer
    {
      public:
        MappedBuffer( BufferSharedPtr const& buffer, GLenum target, GLenum access );
        ~MappedBuffer();

      public:
        operator T*() const;

      private:
        BufferSharedPtr   m_buffer;
        GLenum            m_target;
        T               * m_ptr;
    };


    template <typename T>
    inline MappedBuffer<T>::MappedBuffer( BufferSharedPtr const& buffer, GLenum target, GLenum access )
      : m_buffer( buffer )
      , m_target( target )
    {
      m_ptr = reinterpret_cast<T*>(m_buffer->map( m_target, access ));
    }

    template <typename T>
    inline MappedBuffer<T>::~MappedBuffer()
    {
      DP_VERIFY( m_buffer->unmap( m_target ) );
    }

    template <typename T>
    inline MappedBuffer<T>::operator T*() const
    {
      return( m_ptr );
    }

  } // namespace gl
} // namespace dp
