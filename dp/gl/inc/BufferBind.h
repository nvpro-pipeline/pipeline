// Copyright NVIDIA Corporation 2014
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

#include <dp/gl/inc/IBuffer.h>

namespace dp
{
  namespace gl
  {
    class BufferBind : public IBuffer
    {
      public:
        virtual void copySubData( GLuint srcBuffer, GLuint dstBuffer, size_t srcOffset, size_t dstOffset, size_t size );
        virtual void setData( GLuint buffer, GLenum target, size_t size, GLvoid const* data, GLenum usage );
        virtual void setSubData( GLuint buffer, GLenum target, size_t offset, size_t size, GLvoid const* data );
        virtual void* map( GLuint buffer, GLenum target, GLenum access );
        virtual void* mapRange( GLuint buffer, GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access );
        virtual GLboolean unmap( GLuint buffer, GLenum target );
    };


    void BufferBind::copySubData( GLuint srcBuffer, GLuint dstBuffer, size_t srcOffset, size_t dstOffset, size_t size )
    {
      glBindBuffer( GL_COPY_READ_BUFFER, srcBuffer );
      glBindBuffer( GL_COPY_WRITE_BUFFER, dstBuffer );
      glCopyBufferSubData( GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, srcOffset, dstOffset, size );
    }

    void BufferBind::setData( GLuint buffer, GLenum target, size_t size, GLvoid const* data, GLenum usage )
    {
      glBindBuffer( target, buffer );
      glBufferData( target, size, data, usage );
    }

    void BufferBind::setSubData( GLuint buffer, GLenum target, size_t offset, size_t size, GLvoid const* data )
    {
      glBindBuffer( target, buffer );
      glBufferSubData( target, offset, size, data );
    }

    void * BufferBind::map( GLuint buffer, GLenum target, GLenum access )
    {
      glBindBuffer( target, buffer );
      return( glMapBuffer( target, access ) );
    }

    void * BufferBind::mapRange( GLuint buffer, GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access )
    {
      glBindBuffer( target, buffer );
      return( glMapBufferRange( target, offset, length, access ) );
    }

    GLboolean BufferBind::unmap( GLuint buffer, GLenum target )
    {
      glBindBuffer( target, buffer );
      return( glUnmapBuffer( target ) );
    }

  } // namespace gl
} // namespace dp
