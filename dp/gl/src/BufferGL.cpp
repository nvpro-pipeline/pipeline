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


#include <dp/sg/gl/BufferGL.h>

using namespace nvgl;
using namespace nvutil;
using namespace nvsg;

namespace nvgl
{
  HANDLE_DEFINE_CLONE( BufferGL );

  class TmpBindingArrayBuffer
  {
  public:
    static const GLenum TARGET = GL_ARRAY_BUFFER;
    static const GLenum BINDING = GL_ARRAY_BUFFER_BINDING;

    TmpBindingArrayBuffer( GLuint glId )
    {
      glGetIntegerv( BINDING, &m_glId );
      glBindBuffer( TARGET, glId );
    }

    ~TmpBindingArrayBuffer()
    {
      glBindBuffer( TARGET, m_glId );
    }

  private:
    GLint m_glId;
  };

  BufferGLSharedPtr BufferGL::create()
  {
    return( new BufferGL() );
  }

  BufferGL::BufferGL( )
    : m_mapMode( MAP_NONE )
    , m_target( GL_ARRAY_BUFFER )
    , m_glId(0)
    , m_usage( GL_STATIC_DRAW )
    , m_sizeInBytes( 0 )
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

    glGenBuffers( 1, &m_glId );

    SmartRenderContextGL current = RenderContextGL::getCurrentRenderContextGL();
    if ( current )
    {
      m_shareGroupGL = current->getShareGroupGL();
    }
  }

#if 0
  SmartBufferGL BufferGL::create( size_t sizeInBytes)
  {
    return new BufferGL( sizeInBytes );
  }
#endif

  BufferGL::~BufferGL( )
  {
    class CleanupTask : public nvgl::ShareGroupGLTask
    {
    public:
      CleanupTask( GLuint id ) : m_id( id ) {}

      virtual void execute() { glDeleteBuffers( 1, &m_id ); }
    protected:
      GLuint m_id;
    };
    
    if ( m_glId && m_shareGroupGL )
    {
      const SmartRenderContextGL& currentRenderContextGL = RenderContextGL::getCurrentRenderContextGL();
      if ( currentRenderContextGL && m_shareGroupGL == currentRenderContextGL->getShareGroupGL() )
      {
        glDeleteBuffers( 1, &m_glId );
      }
      else
      {
        // Make destructor exception safe.
        try {
          ShareGroupGLLock(m_shareGroupGL)->executeTask( new CleanupTask( m_glId ) );
        } catch (...) {};
      }
    }
  }

  static inline GLvoid* BufferGLMapRange(GLuint glId, unsigned int stateFlags, nvsg::Buffer::MapMode mapMode, size_t offset, size_t size)
  {
    DP_ASSERT( mapMode != nvsg::Buffer::MAP_NONE );
    DP_ASSERT( glId );

    DP_STATIC_ASSERT( nvsg::Buffer::MAP_READ == GL_MAP_READ_BIT );
    DP_STATIC_ASSERT( nvsg::Buffer::MAP_WRITE == GL_MAP_WRITE_BIT );

    if ( (stateFlags & nvgl::BufferGL::STATE_CAPABILITY_RANGEANDCOPY) == nvgl::BufferGL::STATE_CAPABILITY_RANGEANDCOPY )
    {
      glBindBuffer( GL_COPY_READ_BUFFER, glId );
      void* ptr = glMapBufferRange( GL_COPY_READ_BUFFER, offset, size, mapMode );
      glBindBuffer( GL_COPY_READ_BUFFER, 0 );
      return ptr;
    }
    else if ( stateFlags & nvgl::BufferGL::STATE_CAPABILITY_RANGE )
    {
      TmpBindingArrayBuffer tmpBinding( glId );
      return glMapBufferRange( TmpBindingArrayBuffer::TARGET, offset, size, mapMode );
    }
    else
    {
      GLenum access;
      switch (mapMode)
      {
      case nvsg::Buffer::MAP_READ:
        access = GL_READ_ONLY;
        break;
      case nvsg::Buffer::MAP_WRITE:
        access = GL_WRITE_ONLY;
        break;
      case nvsg::Buffer::MAP_READWRITE:
        access = GL_READ_WRITE;
        break;
      case nvsg::Buffer::MAP_NONE:
      default:
        return 0;
      }

      TmpBindingArrayBuffer tmpBinding( glId );
      char* mappedbytes = reinterpret_cast<char*>(glMapBuffer( TmpBindingArrayBuffer::TARGET, access ));
      return reinterpret_cast<void*>( mappedbytes + offset );
    }
  }

  void *BufferGL::map( MapMode mapMode, size_t offset, size_t size)
  {
    DP_ASSERT( m_mapMode == MAP_NONE );
    DP_ASSERT( m_glId );
    DP_ASSERT( (offset + size) >= offset && (offset + size) <= m_sizeInBytes );

    m_mapMode = mapMode;

    return BufferGLMapRange( m_glId, m_stateFlags, mapMode, offset, size );
  }

  const void *BufferGL::mapRead( size_t offset, size_t size ) const
  {
    DP_ASSERT( m_mapMode == MAP_NONE );
    DP_ASSERT( m_glId );
    DP_ASSERT( (offset + size) >= offset && (offset + size) <= m_sizeInBytes );

    m_mapMode = MAP_READ;

    return BufferGLMapRange( m_glId, m_stateFlags, MAP_READ, offset, size );
  }

  void BufferGL::unmap( )
  {
    if ( m_mapMode & MAP_WRITE )
    {
      notify( Event( this ) );
    }

    TmpBindingArrayBuffer tmpBinding( m_glId );
    glUnmapBuffer( TmpBindingArrayBuffer::TARGET );
    m_mapMode = MAP_NONE;
  }

  void BufferGL::unmapRead( ) const
  {
    DP_ASSERT( m_mapMode == MAP_READ );

    TmpBindingArrayBuffer tmpBinding( m_glId );
    glUnmapBuffer( TmpBindingArrayBuffer::TARGET );
    m_mapMode = MAP_NONE;
  }

  void BufferGL::getData( size_t src_offset, size_t size, const BufferSharedPtr &dst_buffer , size_t dst_offset) const
  {
    // check if we support GL buffer copy at all, and if other obj is a BufferGL
    BufferGLWeakPtr dstGLBuffer = dynamic_cast<BufferGLWeakPtr>( dst_buffer.get() );

    if ( (m_stateFlags & STATE_CAPABILITY_COPY) && dstGLBuffer )
    {
      BufferGLLock tobufferGL( dstGLBuffer );

      DP_ASSERT( (src_offset + size) >= src_offset && (src_offset + size) <= m_sizeInBytes );
      DP_ASSERT( (dst_offset + size) >= dst_offset && (dst_offset + size) <= tobufferGL->m_sizeInBytes );

      glBindBuffer( GL_COPY_READ_BUFFER, m_glId );
      glBindBuffer( GL_COPY_WRITE_BUFFER, tobufferGL->m_glId );
      glCopyBufferSubData( GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, src_offset, dst_offset, size );
      glBindBuffer( GL_COPY_WRITE_BUFFER, 0 );
      glBindBuffer( GL_COPY_READ_BUFFER, 0 );
    }
    else
    {
      Buffer::getData( src_offset, size, dst_buffer, dst_offset );
    }
  }

  void BufferGL::setData( size_t dst_offset, size_t size, const BufferSharedPtr &src_buffer , size_t src_offset)
  {
    // check if we support GL buffer copy at all, and if other obj is a BufferGL
    BufferGLWeakPtr srcGLBuffer = dynamic_cast<BufferGLWeakPtr>( src_buffer.get() );

    if ( (m_stateFlags & STATE_CAPABILITY_COPY) && srcGLBuffer )
    {
      BufferGLLock frombufferGL( srcGLBuffer );

      DP_ASSERT( (src_offset + size) >= src_offset && (src_offset + size) <= frombufferGL->m_sizeInBytes );
      DP_ASSERT( (dst_offset + size) >= dst_offset && (dst_offset + size) <= m_sizeInBytes );

      glBindBuffer( GL_COPY_READ_BUFFER, frombufferGL->m_glId );
      glBindBuffer( GL_COPY_WRITE_BUFFER, m_glId );
      glCopyBufferSubData( GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, src_offset, dst_offset, size );
      glBindBuffer( GL_COPY_WRITE_BUFFER, 0 );
      glBindBuffer( GL_COPY_READ_BUFFER, 0 );
    }
    else
    {
      Buffer::setData( dst_offset, size, src_buffer, src_offset );
    }
  }

  size_t BufferGL::getSize() const
  {
    DP_ASSERT( m_glId );
    DP_ASSERT( m_stateFlags & STATE_MANAGED );

    return m_sizeInBytes;
  }

  void BufferGL::setSize( size_t size)
  {
    DP_ASSERT( m_glId );
    DP_ASSERT( m_stateFlags & STATE_MANAGED );
    DP_ASSERT( m_mapMode == MAP_NONE );

    if ( size != m_sizeInBytes )
    {
      m_sizeInBytes = size;
      TmpBindingArrayBuffer tmpBinding( m_glId );
      glBufferData( TmpBindingArrayBuffer::TARGET, m_sizeInBytes, 0, m_usage );
    }
  }

  void BufferGL::bind()
  {
    bind( m_target );
  }

  void BufferGL::bind( GLenum target )
  {
    glBindBuffer( target, m_glId );
  }

  void BufferGL::unbind()
  {
    unbind( m_target );
  }

  void BufferGL::unbind( GLenum target)
  {
    glBindBuffer( target, 0 );
  }

  GLuint BufferGL::getGLId() const
  {
    return m_glId;
  }

} // namespace nvutil
