// Copyright NVIDIA Corporation 2010-2015
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

    namespace
    {

      inline void checkAndUpdateRange(size_t offset, size_t &length, size_t maxLength)
      {
        length = (length == ~0) ? (maxLength - offset) : length;

        if (length > 0)
        {
          if (offset + length > maxLength)
          {
            throw std::runtime_error("offset + size out of range");
          }

          if (offset >= maxLength)
          {
            throw std::runtime_error("offset out of range");
          }
        }
      }


      class BufferCore : public Buffer 
      {
      public:
        BufferCore(GLenum usage, GLenum defaultTarget);
        virtual ~BufferCore();

        void update(void const *data, size_t offset, size_t size);
        void* map(GLbitfield access, size_t offset, size_t length);
        GLboolean unmap();
        void setSize(size_t size);
        void invalidate();
        void makeResident();
        void makeNonResident();

      private:
        GLenum m_target;
        GLenum m_usage;
      };

      BufferCore::BufferCore(GLenum usage, GLenum defaultTarget)
        : m_target(defaultTarget)
        , m_usage(usage)
      {
        glGenBuffers(1, &m_id);
      }

      BufferCore::~BufferCore()
      {
        if (m_address)
        {
          makeNonResident();
        }

        if (m_mappedAddress)
        {
          unmap();
        }

        if ( m_id )
        {
          if ( getShareGroup() )
          {
            DEFINE_PTR_TYPES( CleanupTask );
            class CleanupTask : public ShareGroupTask
            {
            public:
              static CleanupTaskSharedPtr create( GLuint id )
              {
                return( std::shared_ptr<CleanupTask>( new CleanupTask( id ) ) );
              }

              virtual void execute() { glDeleteBuffers( 1, &m_id ); }

            protected:
              CleanupTask( GLuint id ) : m_id( id ) {}

            private:
              GLuint m_id;
            };

            // make destructor exception safe
            try
            {
              getShareGroup()->executeTask( CleanupTask::create( m_id ) );
            } catch (...) {}
          }
          else
          {
            glDeleteBuffers( 1, &m_id );
          }
        }
      }

      void BufferCore::update(void const *data, size_t offset, size_t length)
      {
        checkAndUpdateRange(offset, length, m_size);

        if (length)
        {
          glBindBuffer(m_target, m_id);
          glBufferSubData(m_target, offset, length, data);
        }
      }

      void* BufferCore::map(GLbitfield access, size_t offset, size_t length)
      {
        checkAndUpdateRange(offset, length, m_size);
        if (m_mappedAddress)
        {
          throw std::runtime_error("Buffer is already mapped");
        }

        if (m_size)
        {
          glBindBuffer(m_target, m_id);
          m_mappedAddress = glMapBufferRange(m_target, offset, length, access);
        }
        else
        {
          m_mappedAddress = reinterpret_cast<void*>(~0);
        }

        return m_mappedAddress;
      }

      GLboolean BufferCore::unmap()
      {
        if (!m_mappedAddress)
        {
          throw std::runtime_error("Buffer is not mapped");
        }

        if (m_mappedAddress != reinterpret_cast<void*>(~0))
        {
          m_mappedAddress = 0;
          glBindBuffer(m_target, m_id);
          return glUnmapBuffer(m_target);
        }
        else
        {
          m_mappedAddress = 0;
          return GL_TRUE;
        }
      }

      void BufferCore::setSize(size_t size)
      {
        if (size != m_size)
        {
          glBindBuffer(m_target, m_id);
          glBufferData(m_target, size, nullptr, m_usage);
          m_size = size;
        }
      }

      void BufferCore::invalidate()
      {
        if (m_size)
        {
          glInvalidateBufferData(m_id);
        }
      }

      void BufferCore::makeResident()
      {
        if (m_address)
        {
          throw std::runtime_error("buffer is already resident");
        }

        if (m_size)
        {
          glBindBuffer(m_target, m_id);
          glGetBufferParameterui64vNV (m_target, GL_BUFFER_GPU_ADDRESS_NV, &m_address); 
          glMakeBufferResidentNV(m_target, GL_READ_ONLY);
        }
        else
        {
          // ~0 is specifed as pointer to buffers with size 0
          m_address = ~0;
        }
      }

      void BufferCore::makeNonResident()
      {
        if (!m_address)
        {
          throw std::runtime_error("buffer is not resident");
        }

        if (m_address != ~0)
        {
          glBindBuffer(m_target, m_id);
          glMakeBufferNonResidentNV(m_target);
        }
        m_address = 0;
      }

#if defined(GL_VERSION_4_5)
      /************************************************************************/
      /* BufferCoreDSA                                                        */
      /************************************************************************/
      class BufferCoreDSA : public Buffer 
      {
      public:
        BufferCoreDSA(GLenum usage);
        virtual ~BufferCoreDSA();

        void update(void const *data, size_t offset, size_t size);
        void* map(GLbitfield access, size_t offset, size_t length);
        GLboolean unmap();
        void setSize(size_t size);
        void invalidate();
        void makeResident();
        void makeNonResident();

      private:
        GLenum m_usage;
      };

      BufferCoreDSA::BufferCoreDSA(GLenum usage)
        : m_usage(usage)
      {
        glCreateBuffers(1, &m_id);
      }

      BufferCoreDSA::~BufferCoreDSA()
      {
        if (m_address)
        {
          makeNonResident();
        }

        if (m_mappedAddress)
        {
          unmap();
        }

        if ( m_id )
        {
          if ( getShareGroup() )
          {
            DEFINE_PTR_TYPES( CleanupTask );
            class CleanupTask : public ShareGroupTask
            {
            public:
              static CleanupTaskSharedPtr create( GLuint id )
              {
                return( std::shared_ptr<CleanupTask>( new CleanupTask( id ) ) );
              }

              virtual void execute() { glDeleteBuffers( 1, &m_id ); }

            protected:
              CleanupTask( GLuint id ) : m_id( id ) {}

            private:
              GLuint m_id;
            };

            // make destructor exception safe
            try
            {
              getShareGroup()->executeTask( CleanupTask::create( m_id ) );
            } catch (...) {}
          }
          else
          {
            glDeleteBuffers( 1, &m_id );
          }
        }
      }

      void BufferCoreDSA::update(void const *data, size_t offset, size_t length)
      {
        checkAndUpdateRange(offset, length, m_size);

        glNamedBufferSubData(m_id, offset, length, data);
      }

      void* BufferCoreDSA::map(GLbitfield access, size_t offset, size_t length)
      {
        checkAndUpdateRange(offset, length, m_size);
        if (m_mappedAddress)
        {
          throw std::runtime_error("Buffer is already mapped");
        }

        if (m_size)
        {
          m_mappedAddress = glMapNamedBufferRange(m_id, offset, length, access);
        }
        else
        {
          m_mappedAddress = reinterpret_cast<void*>(~0);
        }

        return m_mappedAddress;
      }

      GLboolean BufferCoreDSA::unmap()
      {
        if (!m_mappedAddress)
        {
          throw std::runtime_error("Buffer is not mapped");
        }

        if (m_mappedAddress != reinterpret_cast<void*>(~0))
        {
          m_mappedAddress = 0;
          return glUnmapNamedBuffer(m_id);
        }
        else
        {
          m_mappedAddress = 0;
          return GL_TRUE;
        }
      }

      void BufferCoreDSA::setSize(size_t size)
      {
        if (size != m_size)
        {
          glNamedBufferData(m_id, size, nullptr, m_usage);
          m_size = size;
        }
      }

      void BufferCoreDSA::invalidate()
      {
        if (m_size)
        {
          glInvalidateBufferData(m_id);
        }
      }

      void BufferCoreDSA::makeResident()
      {
        if (m_address)
        {
          throw std::runtime_error("buffer is already resident");
        }

        if (m_size)
        {
          glGetNamedBufferParameterui64vNV( m_id, GL_BUFFER_GPU_ADDRESS_NV, &m_address );
          glMakeNamedBufferResidentNV(m_id, GL_READ_ONLY);
        }
        else
        {
          // ~0 is specifed as pointer to buffers with size 0
          m_address = ~0;
        }
      }

      void BufferCoreDSA::makeNonResident()
      {
        if (!m_address)
        {
          throw std::runtime_error("buffer is not resident");
        }

        // ~0 is the pointer to a zero-sized buffer
        if (m_address != ~0)
        {
          glMakeNamedBufferNonResidentNV(m_id);
        }
        m_address = 0;
      }


      /************************************************************************/
      /* BufferPersistentDSA                                                        */
      /************************************************************************/
      class BufferPersistentDSA : public Buffer 
      {
      public:
        BufferPersistentDSA(GLenum usage);
        virtual ~BufferPersistentDSA();

        void update(void const *data, size_t offset, size_t size);
        void* map(GLbitfield access, size_t offset, size_t length);
        GLboolean unmap();
        void setSize(size_t size);
        void invalidate();
        void makeResident();
        void makeNonResident();

      private:
        GLenum m_usageBits;
      };

      BufferPersistentDSA::BufferPersistentDSA(GLbitfield usageBits)
        : m_usageBits(usageBits)
      {
        glGenBuffers(1, &m_id);
      }

      BufferPersistentDSA::~BufferPersistentDSA()
      {
        if (m_address)
        {
          makeNonResident();
        }

        if (m_mappedAddress)
        {
          unmap();
        }

        if ( m_id )
        {
          if ( getShareGroup() )
          {
            DEFINE_PTR_TYPES( CleanupTask );
            class CleanupTask : public ShareGroupTask
            {
            public:
              static CleanupTaskSharedPtr create(GLuint id)
              {
                return(std::shared_ptr<CleanupTask>(new CleanupTask(id)));
              }

              virtual void execute() { glDeleteBuffers( 1, &m_id ); }

            protected:
              CleanupTask( GLuint id ) : m_id( id ) {}

            private:
              GLuint m_id;
            };

            // make destructor exception safe
            try
            {
              getShareGroup()->executeTask( CleanupTask::create( m_id ) );
            } catch (...) {}
          }
          else
          {
            glDeleteBuffers( 1, &m_id );
          }
        }
      }

      void BufferPersistentDSA::update(void const *data, size_t offset, size_t length)
      {
        checkAndUpdateRange(offset, length, m_size);

        glNamedBufferSubData(m_id, offset, length, data);
      }

      void* BufferPersistentDSA::map(GLbitfield access, size_t offset, size_t length)
      {
        checkAndUpdateRange(offset, length, m_size);
        if (m_mappedAddress)
        {
          throw std::runtime_error("Buffer is already mapped");
        }

        if (m_size)
        {
          m_mappedAddress = glMapNamedBufferRange(m_id, offset, length, access);
        }
        else
        {
          m_mappedAddress = reinterpret_cast<void*>(~0);
        }

        return m_mappedAddress;
      }

      GLboolean BufferPersistentDSA::unmap()
      {
        if (!m_mappedAddress)
        {
          throw std::runtime_error("Buffer is not mapped");
        }

        if (m_mappedAddress != reinterpret_cast<void*>(~0))
        {
          m_mappedAddress = nullptr;
          return glUnmapNamedBuffer(m_id);
        }
        else
        {
          m_mappedAddress = nullptr;
          return GL_TRUE;
        }
      }

      void BufferPersistentDSA::setSize(size_t size)
      {
        if (m_address)
        {
          throw std::runtime_error("Buffer is resident while calling Buffer::setSize");
        }
        if (m_mappedAddress)
        {
          throw std::runtime_error("Buffer is mapped while calling Buffer::operation");
        }
        if (size != m_size)
        {
          // A persistent storage buffer cannot be resized. Create a new buffer object.
          glDeleteBuffers(1, &m_id);
          glCreateBuffers(1, &m_id);
          glNamedBufferStorage(m_id, size, nullptr, m_usageBits);
          m_size = size;
        }
      }

      void BufferPersistentDSA::invalidate()
      {
        if (m_size)
        {
          glInvalidateBufferData(m_id);
        }
      }

      void BufferPersistentDSA::makeResident()
      {
        if (m_address)
        {
          throw std::runtime_error("buffer is already resident");
        }

        if (m_size)
        {
          glGetNamedBufferParameterui64vNV(m_id, GL_BUFFER_GPU_ADDRESS_NV, &m_address);
          glMakeNamedBufferResidentNV(m_id, GL_READ_ONLY);
        }
        else
        {
          // ~0 is specifed as pointer to buffers with size 0
          m_address = ~0;
        }
      }

      void BufferPersistentDSA::makeNonResident()
      {
        if (!m_address)
        {
          throw std::runtime_error("buffer is not resident");
        }

        // ~0 is the pointer to a zero-sized buffer
        if (m_address != ~0)
        {
          glMakeNamedBufferNonResidentNV(m_id);
        }
        m_address = 0;
      }
#endif

    } // namespace anonymous

    BufferSharedPtr Buffer::create(Mode_Core, GLenum mode, GLenum defaultTarget)
    {
#if defined(GL_VERSION_4_5)
      if (GLEW_VERSION_4_5)
      {
        return(std::shared_ptr<Buffer>(new BufferCoreDSA(mode)));
      }
      else
      {
        return(std::shared_ptr<Buffer>(new BufferCore(mode, defaultTarget)));
      }
#else
      return(std::shared_ptr<Buffer>(new BufferCore(mode, defaultTarget)));
#endif
    }

    BufferSharedPtr Buffer::create(Mode_Persistent_Buffer, GLbitfield modeBits)
    {
#if defined(GL_VERSION_4_5)
      if (!GLEW_ARB_buffer_storage)
      {
        throw std::runtime_error("ARB_buffer_storage not available");
      }
      return(std::shared_ptr<Buffer>(new BufferPersistentDSA(modeBits)));
#else
      throw std::runtime_error("GL 4.5 support not enabled");
#endif
    }


    Buffer::Buffer()
      : m_address(0)
      , m_size(0)
      , m_mappedAddress(0)
    {
      m_id;
      GLuint id;
      glGenBuffers( 1, &id );
      setGLId( id );
    }

    Buffer::~Buffer( )
    {

    }

    void bind( GLenum target, BufferSharedPtr const& buffer )
    {
#if !defined(NDEBUG)
      DP_ASSERT( RenderContext::getCurrentRenderContext() );
      static std::map<std::pair<RenderContext*,GLenum>,Buffer*> boundBufferMap;
      if ( buffer )
      {
        boundBufferMap[std::make_pair( RenderContext::getCurrentRenderContext().getWeakPtr(), target )] = buffer.getWeakPtr();
      }
      else
      {
        std::map<std::pair<RenderContext*,GLenum>,Buffer*>::iterator it = boundBufferMap.find( std::make_pair( RenderContext::getCurrentRenderContext().getWeakPtr(), target ) );
        if ( it != boundBufferMap.end() )
        {
          boundBufferMap.erase( it );
        }
      }
#endif

      glBindBuffer( target, buffer ? buffer->getGLId() : 0 );
    }

    void copy( BufferSharedPtr const& srcBuffer, BufferSharedPtr const& dstBuffer, size_t srcOffset, size_t dstOffset, size_t size )
    {
      DP_ASSERT( ( srcOffset <= srcOffset + size ) && ( srcOffset + size <= srcBuffer->getSize() ) );
      DP_ASSERT( ( dstOffset <= dstOffset + size ) && ( dstOffset + size <= dstBuffer->getSize() ) );
      getGLInterface()->copySubData( srcBuffer->getGLId(), dstBuffer->getGLId(), srcOffset, dstOffset, size );
    }

  } // namespace gl
} // namespace dp
