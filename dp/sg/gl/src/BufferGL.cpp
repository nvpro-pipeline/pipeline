// Copyright NVIDIA Corporation 2010-2012
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


#include <dp/sg/gl/BufferGL.h>

namespace dp
{
  namespace sg
  {
    namespace gl
    {
      BufferGLSharedPtr BufferGL::create()
      {
        return( std::shared_ptr<BufferGL>( new BufferGL() ) );
      }

      dp::sg::core::HandledObjectSharedPtr BufferGL::clone() const
      {
        return( std::shared_ptr<BufferGL>( new BufferGL( *this ) ) );
      }

      BufferGL::BufferGL( )
        : m_mapMode( MAP_NONE )
        , m_target( GL_ARRAY_BUFFER )
        , m_usage( GL_STATIC_DRAW )
        , m_stateFlags( STATE_MANAGED )
      {
        if ( !!GLEW_ARB_map_buffer_range )
        {
          m_stateFlags |= STATE_CAPABILITY_RANGE;
        }

        if ( !!GLEW_ARB_copy_buffer )
        {
          m_stateFlags |= STATE_CAPABILITY_COPY;
        }

        m_buffer = dp::gl::Buffer::create(dp::gl::Buffer::CORE, m_usage);
      }

    #if 0
      SmartBufferGL BufferGL::create( size_t sizeInBytes)
      {
        return new BufferGL( sizeInBytes );
      }
    #endif

      BufferGL::~BufferGL( )
      {
      }

      void *BufferGL::map( MapMode mapMode, size_t offset, size_t size)
      {
        DP_ASSERT( m_mapMode == MAP_NONE );
        DP_ASSERT( m_buffer->getGLId() );
        DP_ASSERT( (offset + size) >= offset && (offset + size) <= m_buffer->getSize() );

        m_mapMode = mapMode;

        GLbitfield access;
        switch (mapMode)
        {
        case core::Buffer::MAP_READ:
          access |= GL_MAP_READ_BIT;
          break;
        case core::Buffer::MAP_WRITE:
          access |= GL_MAP_WRITE_BIT;
            break;
        case core::Buffer::MAP_READWRITE:
          access = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
          break;
        case core::Buffer::MAP_NONE:
        default:
          return nullptr;
        }


        return m_buffer->map(access, offset, size );
      }

      const void *BufferGL::mapRead( size_t offset, size_t size ) const
      {
        DP_ASSERT( m_mapMode == MAP_NONE );
        DP_ASSERT( m_buffer->getGLId() );
        DP_ASSERT( (offset + size) >= offset && (offset + size) <= m_buffer->getSize() );

        m_mapMode = MAP_READ;

        return m_buffer->map(GL_MAP_READ_BIT, offset, size);
      }

      void BufferGL::unmap( )
      {
        if ( m_mapMode & MAP_WRITE )
        {
          notify( Event( this ) );
        }

        m_buffer->unmap();
        m_mapMode = MAP_NONE;
      }

      void BufferGL::unmapRead( ) const
      {
        DP_ASSERT(m_mapMode == MAP_READ);
        DP_VERIFY(m_buffer->unmap());
        m_mapMode = MAP_NONE;
      }

      void BufferGL::getData( size_t srcOffset, size_t size, const core::BufferSharedPtr &dstBuffer , size_t dstOffset) const
      {
        // check if we support GL buffer copy at all, and if other obj is a BufferGL
        if ( (m_stateFlags & STATE_CAPABILITY_COPY) && dstBuffer.isPtrTo<BufferGL>() )
        {
          copy( m_buffer, dstBuffer.staticCast<BufferGL>()->m_buffer, srcOffset, dstOffset, size );
        }
        else
        {
          Buffer::getData( srcOffset, size, dstBuffer, dstOffset );
        }
      }

      void BufferGL::setData( size_t dstOffset, size_t size, const core::BufferSharedPtr &srcBuffer , size_t srcOffset)
      {
        // check if we support GL buffer copy at all, and if other obj is a BufferGL
        if ( (m_stateFlags & STATE_CAPABILITY_COPY) && srcBuffer.isPtrTo<BufferGL>() )
        {
          copy( srcBuffer.staticCast<BufferGL>()->m_buffer, m_buffer, srcOffset, dstOffset, size );
        }
        else
        {
          Buffer::setData( dstOffset, size, srcBuffer, srcOffset );
        }
      }

      size_t BufferGL::getSize() const
      {
        DP_ASSERT( m_buffer->getGLId() );
        DP_ASSERT( m_stateFlags & STATE_MANAGED );

        return( m_buffer->getSize() );
      }

      void BufferGL::setSize( size_t size)
      {
        DP_ASSERT( m_buffer->getGLId() );
        DP_ASSERT( m_stateFlags & STATE_MANAGED );
        DP_ASSERT( m_mapMode == MAP_NONE );

        m_buffer->setSize(size);
      }

      dp::gl::BufferSharedPtr const& BufferGL::getBuffer() const
      {
        return( m_buffer );
      }

      GLuint BufferGL::getGLId() const
      {
        DP_ASSERT( m_buffer );
        return( m_buffer->getGLId() );
      }

    } // namespace gl
  } // namespace sg
} // namespace dp
