// Copyright NVIDIA Corporation 2011
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

#include <dp/gl/Texture.h>
#include <dp/rix/gl/RiXGL.h>
#include <GL/glew.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      class TextureGL : public dp::rix::core::Texture
      {
      public:
        class BindlessReference : public dp::rix::core::HandledObject
        {
        public:
          BindlessReference(GLuint tex, GLuint sampler=0);
          BindlessReference(GLuint tex, GLint level, GLboolean layered, GLint layer, GLint format, GLenum access);
          ~BindlessReference();

          inline GLuint64 getHandle() const
          {
            return m_handle;
          }
        private:
          BindlessReference(const BindlessReference& rhs);
          BindlessReference& operator=(BindlessReference &rhs);

          bool        m_isImage;
          GLuint64    m_handle;
        };

        typedef BindlessReference*                            BindlessReferenceHandle;
        typedef dp::rix::core::SmartHandle<BindlessReference> SmartBindlessReferenceHandle;

      protected:
        TextureGL( dp::gl::SmartTexture const& texture, bool mipMapped );

      public:
        ~TextureGL();

        static dp::rix::core::TextureHandle create( const dp::rix::core::TextureDescription& textureType );
        virtual bool setData( const dp::rix::core::TextureData& textureData );

        virtual void applySamplerState( SamplerStateGLHandle samplerState );
        virtual void applyDefaultSamplerState();
        virtual void setDefaultSamplerState( SamplerStateGLHandle samplerState );

        virtual BindlessReferenceHandle getBindlessTextureHandle( SamplerStateGLHandle samplerState );
        virtual BindlessReferenceHandle getBindlessTextureHandle( GLint level, GLboolean layered, GLint layer, GLint format, GLenum access );

        dp::gl::SmartTexture const& getTexture() const;

        // TODO: TextureDescriptionGL could be stored here
        bool    m_hasMipmaps;

  #if RIX_GL_SAMPLEROBJECT_SUPPORT
        // last sampler state doesnt make sense with sampler objects
        // default sampler state is just stored in texture via parameters
  #else
        SamplerStateGLSharedHandle m_lastSamplerStateHandle;
        SamplerStateGLSharedHandle m_defaultSamplerStateHandle;
  #endif

      protected:
        virtual void upload( unsigned int mipMapLevel, unsigned int layer, const void* ptr );

        static unsigned int getMipMapSize( unsigned int size, unsigned int level );

        void setTexture( dp::gl::SmartTexture const& texture );
        unsigned int getNumberOfMipMapLevels() const;

      private:
        dp::gl::SmartTexture  m_texture;
      };

    } // namespace gl
  } // namespace rix
} // namespace dp
