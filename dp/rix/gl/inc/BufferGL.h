// Copyright NVIDIA Corporation 2011-2015
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
#include <dp/gl/Buffer.h>
#include <dp/rix/gl/RiXGL.h>

#include <dp/rix/gl/inc/TextureGL.h>
#include <dp/rix/gl/inc/SamplerStateGL.h>

#include <dp/util/Observer.h>

// bindless UBO defines until glew supports the extension
#define GL_UNIFORM_BUFFER_UNIFIED_NV                        0x936E
#define GL_UNIFORM_BUFFER_ADDRESS_NV                        0x936F

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      class BufferGL : public dp::rix::core::Buffer, public dp::util::Subject
      {
      public:
        class Event : public dp::util::Event
        {
        public:
          enum EventType
          {
            DATA_CHANGED = 0,
            DATA_AND_SIZE_CHANGED
          };

          Event( BufferGL *buffer, EventType type );

          BufferGL* m_buffer;
          EventType m_eventType;
        };

      public:
        static BufferGLHandle create( const dp::rix::core::BufferDescription& bufferDescription );

        dp::gl::BufferSharedPtr const&  getBuffer() const { return( m_buffer ); }

        void setSize( size_t width, size_t height = 0, size_t depth = 0 );
        void setElementSize( size_t elementSize );
        void setFormat( dp::rix::core::BufferFormat bufferFormat );
        void updateData( size_t offset, const void *data, size_t size );

        void* map( dp::rix::core::AccessType accessType );
        bool  unmap();

#if 0
        void setReference( size_t slot, const dp::rix::core::ContainerDataBuffer  &buffer,  BufferStoredReferenceGL &refstore );
        void setReference( size_t slot, const dp::rix::core::ContainerDataSampler &sampler, BufferStoredReferenceGL &refstore );
        void setReference( size_t slot, const dp::rix::core::ContainerData& data,           BufferStoredReferenceGL &refstore);

        void initReferences( const dp::rix::core::BufferReferences &refinfo );
        void initReferences( const dp::rix::core::BufferReferencesBuffer &refbufferinfo );
        void initReferences( const dp::rix::core::BufferReferencesSampler &refbufferinfo );

        void resetBufferReferences();
        void resetSamplerReferences();
#endif

      protected:
        BufferGL( const dp::rix::core::BufferDescription& bufferDescription );
        ~BufferGL();

#if 0
      private:
        struct SamplerReference {
          SamplerReference() {}
          void set( dp::rix::gl::TextureGLHandle texture, dp::rix::gl::SamplerStateGLHandle sampler, dp::rix::gl::TextureGL::BindlessReferenceHandle bindless )
          {
            m_bindlessHandle = bindless;
            m_samplerHandle  = sampler;
            m_textureHandle  = texture;
          }

        protected:
          dp::rix::gl::TextureGLSharedHandle                    m_textureHandle;
          dp::rix::gl::SamplerStateGLSharedHandle               m_samplerHandle;
          dp::rix::gl::TextureGL::SmartBindlessReferenceHandle m_bindlessHandle;

        private:
          SamplerReference(const SamplerReference& rhs);
          SamplerReference& operator=(SamplerReference &rhs);
        };

        struct BufferReference {
          BufferReference() {}

          dp::rix::gl::BufferGLSharedHandle m_bufferHandle;

        private:
          BufferReference(const BufferReference& rhs);
          BufferReference& operator=(BufferReference &rhs);
        };
#endif

      private:
        dp::gl::BufferSharedPtr     m_buffer;
        dp::rix::core::BufferFormat m_format;
        size_t                      m_elementSize;
        GLenum                      m_usageHint;
        size_t                      m_width;
        size_t                      m_height;
        size_t                      m_depth;
#if !defined(NDEBUG)
        dp::rix::core::AccessType   m_accessType;
        bool                        m_managesData;
#endif
#if 0
        // buffer reference system
        size_t                      m_numBufferSlots;
        size_t                      m_numSamplerSlots;
        BufferReference*            m_bufferReferences;
        SamplerReference*           m_samplerReferences;
#endif
      };

      void initBufferBinding();

      typedef void (*BindBufferRangePtr)( GLenum target,  GLuint index,  GLuint buffer,  GLintptr offset,  GLsizeiptr size );
      extern BindBufferRangePtr bindBufferRangeInternal;

      inline void bindBufferRange( GLenum target,  GLuint index,  GLuint buffer,  GLintptr offset,  GLsizeiptr size)
      {
        bindBufferRangeInternal( target, index, buffer, offset, size );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
