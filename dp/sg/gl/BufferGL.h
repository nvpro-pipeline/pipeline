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

#include <GL/glew.h>
#include <dp/sg/gl/Config.h>
#include <dp/sg/gl/CoreTypes.h>
#include <dp/sg/core/Buffer.h>
#include <dp/gl/Buffer.h>
#include <dp/gl/RenderContext.h>
#include <dp/util/BitMask.h>

namespace dp
{
  namespace sg
  {
    namespace gl
    {

      /*! \brief This class uses an OpenGL buffer object as storage for the data of an dp::sg::core::Buffer. For
                 all operations on an nvgl::BufferGL object it is necessary to have a valid OpenGL context current.
      **/
      class BufferGL : public dp::sg::core::Buffer
      {
      public:
        /*! \brief Create a new nvgl::BufferGL object. It is required that an OpenGL context is current
                    when calling this function.
            \return An nvgl::BufferGLSharedPtr to the created object.
        !*/
        DP_SG_GL_API static BufferGLSharedPtr create();

        DP_SG_GL_API virtual dp::sg::core::HandledObjectSharedPtr clone() const;

        DP_SG_GL_API virtual ~BufferGL();

      public:
        enum BufferState{
          STATE_MANAGED                 = BIT0,
          STATE_CAPABILITY_RANGE        = BIT1,
          STATE_CAPABILITY_COPY         = BIT2,
          STATE_CAPABILITY_RANGEANDCOPY = STATE_CAPABILITY_RANGE | STATE_CAPABILITY_COPY,
        };
  
        using Buffer::setData;
        using Buffer::getData;

        /*! \brief Copy data from this buffer into the given buffer.
            \param srcOffset Offset in source buffer in bytes.
            \param length Number of bytes to copy from source buffer to destination buffer.
            \param dstBuffer Buffer which should receive the data.
            \param dstOffset Offset in destination buffer in bytes.
        !*/
        DP_SG_GL_API virtual void getData( size_t srcOffset, size_t length, const dp::sg::core::BufferSharedPtr &dstBuffer , size_t dstOffset) const;

        /*! \brief Copy data from a given buffer into this buffer.
            \param srcOffset Offset in this buffer in bytes.
            \param length Number of bytes to copy from source buffer into this buffer.
            \param srcBuffer Source for copy operation.
            \param srcOffset Offset in source buffer in bytes.
        !*/
        DP_SG_GL_API virtual void setData( size_t dstOffset, size_t length, const dp::sg::core::BufferSharedPtr &srcBuffer , size_t srcOffset);

        /*! \brief Resize the buffer. This function will resize the underlying storage. All data stored
                   in this buffer will be lost.
            \remarks Use resize( size_t ) to change the size without losing data in this buffer.
        !*/
        DP_SG_GL_API virtual void setSize(size_t size);

        /*! \brief Query the size of this buffer.
            \return Size of this buffer.
        !*/
        DP_SG_GL_API virtual size_t getSize() const;

        /*! \brief Sets the OpenGL target this buffer will be bound to.
         *  \param target OpenGL target (GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, ...)
        !*/
        DP_SG_GL_API void setTarget( GLenum target );

        /*! \brief Query the OpenGL target of this buffer.
            \return OpenGL target of this buffer.
        !*/
        DP_SG_GL_API GLenum getTarget() const;

        /*! \brief Sets the OpenGL access usage of this object. Becomes effective with the next resize of setSize operation.
         *  \param usage (GL_STATIC_DRAW, GL_DYNAMIC_READ, ...).
        !*/
        DP_SG_GL_API void setUsage( GLenum usage );

        /*! \brief Query the OpenGL access usage of this buffer.
            \return Usage of this buffer (GL_STATIC_DRAW, GL_DYNAMIC_READ, ...).
        **/
        DP_SG_GL_API GLenum getUsage() const;

        DP_SG_GL_API dp::gl::SharedBuffer const& getBuffer() const;

        /*! \brief Query the OpenGL buffer object name of this buffer.
            \return OpenGL buffer object name.
        **/
        DP_SG_GL_API GLuint getGLId() const;

      protected:
        /*! \brief Default constructor !*/
        DP_SG_GL_API BufferGL( );

        using Buffer::map;
        DP_SG_GL_API virtual void *map( MapMode mode, size_t offset, size_t length);
        DP_SG_GL_API virtual void unmap( );

        using Buffer::mapRead;
        DP_SG_GL_API virtual const void *mapRead( size_t offset, size_t length ) const;
        DP_SG_GL_API virtual void unmapRead( ) const;

        mutable Buffer::MapMode m_mapMode;

        GLenum          m_target;
        GLenum          m_usage;
        unsigned int    m_stateFlags;

      private:
        dp::gl::SharedBuffer m_buffer;
      };

      inline void BufferGL::setTarget( GLenum target )
      {
        DP_ASSERT( m_mapMode == MAP_NONE );
        m_target = target;
      }

      inline GLenum BufferGL::getTarget() const
      {
        return m_target;
      }

      inline void BufferGL::setUsage( GLenum usage )
      {
        DP_ASSERT( m_mapMode == MAP_NONE );
        m_usage = usage;
      }

      inline GLenum BufferGL::getUsage() const
      {
        return m_usage;
      }

    } // namespace gl
  } // namespace sg
} // namespace dp
