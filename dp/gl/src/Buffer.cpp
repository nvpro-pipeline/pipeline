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


#include <dp/gl/Buffer.h>
#include <dp/gl/inc/BufferBind.h>
#include <dp/gl/inc/BufferDSA.h>

namespace dp
{
  namespace gl
  {
    namespace
    {
      inline IBuffer * getGLInterface()
      {
        static BufferBind  glInterfaceBind;
        static BufferDSA   glInterfaceDSA;
        static IBuffer * glInterface = GLEW_EXT_direct_state_access ? static_cast<IBuffer*>(&glInterfaceDSA) : static_cast<IBuffer*>(&glInterfaceBind);
        return( glInterface );
      }
    }

    SharedBuffer Buffer::create()
    {
      return( SharedBuffer( new Buffer ) );
    }

    SharedBuffer Buffer::create( GLenum target, size_t size, GLvoid const* data, GLenum usage )
    {
      SharedBuffer buffer = Buffer::create();
      buffer->setData( target, size, data, usage );
      return( buffer );
    }

    Buffer::Buffer()
      : m_address( 0 )
      , m_size(0)
    {
      GLuint id;
      glGenBuffers( 1, &id );
      setGLId( id );
    }

    Buffer::~Buffer( )
    {
      if ( getGLId() )
      {
        if ( getShareGroup() )
        {
          class CleanupTask : public ShareGroupTask
          {
            public:
              CleanupTask( GLuint id ) : m_id( id ) {}

              virtual void execute() { glDeleteBuffers( 1, &m_id ); }

            private:
              GLuint m_id;
          };

          // make destructor exception safe
          try
          {
            getShareGroup()->executeTask( SharedShareGroupTask( new CleanupTask( getGLId() ) ) );
          } catch (...) {}
        }
        else
        {
          GLuint id = getGLId();
          glDeleteBuffers( 1, &id );
        }
      }
    }

    GLuint64EXT Buffer::getAddress()
    {
      if ( !m_address )
      {
        glGetNamedBufferParameterui64vNV( getGLId(), GL_BUFFER_GPU_ADDRESS_NV, &m_address );
        glMakeNamedBufferResidentNV( getGLId(), GL_READ_ONLY ); // TODO how do writes affect the resident buffer?
      }
      return m_address;
    }

    size_t Buffer::getSize() const
    {
      return( m_size );
    }

    void Buffer::setData( GLenum target, size_t size, GLvoid const* data, GLenum usage )
    {
      if ( ( m_size != size ) && m_address )
      {
        glMakeNamedBufferNonResidentNV( getGLId() );
        m_address = 0;
      }
      m_size = size;

      getGLInterface()->setData( getGLId(), target, size, data, usage );
    }

    void Buffer::setSubData( GLenum target, size_t offset, size_t size, GLvoid const* data )
    {
      DP_ASSERT( size + offset <= m_size );
      getGLInterface()->setSubData( getGLId(), target, offset, size, data );
    }

    void * Buffer::map( GLenum target, GLenum access )
    {
      return( getGLInterface()->map( getGLId(), target, access ) );
    }

    void * Buffer::mapRange( GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access )
    {
      return( getGLInterface()->mapRange( getGLId(), target, offset, length, access ) );
    }

    GLboolean Buffer::unmap( GLenum target )
    {
      return( getGLInterface()->unmap( getGLId(), target ) );
    }

    void bind( GLenum target, SharedBuffer const& buffer )
    {
#if !defined(NDEBUG)
      DP_ASSERT( RenderContext::getCurrentRenderContext() );
      static std::map<std::pair<RenderContext*,GLenum>,Buffer*> boundBufferMap;
      if ( buffer )
      {
        boundBufferMap[std::make_pair( RenderContext::getCurrentRenderContext().get(), target )] = buffer.get();
      }
      else
      {
        std::map<std::pair<RenderContext*,GLenum>,Buffer*>::iterator it = boundBufferMap.find( std::make_pair( RenderContext::getCurrentRenderContext().get(), target ) );
        if ( it != boundBufferMap.end() )
        {
          boundBufferMap.erase( it );
        }
      }
#endif

      glBindBuffer( target, buffer ? buffer->getGLId() : 0 );
    }

    void copy( SharedBuffer const& srcBuffer, SharedBuffer const& dstBuffer, size_t srcOffset, size_t dstOffset, size_t size )
    {
      DP_ASSERT( ( srcOffset <= srcOffset + size ) && ( srcOffset + size <= srcBuffer->getSize() ) );
      DP_ASSERT( ( dstOffset <= dstOffset + size ) && ( dstOffset + size <= dstBuffer->getSize() ) );
      getGLInterface()->copySubData( srcBuffer->getGLId(), dstBuffer->getGLId(), srcOffset, dstOffset, size );
    }

  } // namespace gl
} // namespace dp
