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

#include <dp/gl/Buffer.h>
#include <dp/gl/inc/ITexture.h>

namespace dp
{
  namespace gl
  {
    GLenum getBindingTargetFromTarget( GLenum target );

    class TextureBind : public ITexture
    {
      public:
        virtual void attachBuffer( GLuint texture, GLenum target, GLenum internalFormat, GLuint buffer );
        virtual void generateMipMap( GLuint texture, GLenum target );
        virtual void getCompressedImage( GLuint texture, GLenum target, GLint lod, GLvoid * img );
        virtual void getImage( GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, GLvoid * pixels );
        virtual void setCompareParameters( GLuint texture, GLenum target, GLenum mode, GLenum func );
        virtual void setCompressedImage1D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLint border, GLsizei imageSize, GLvoid const* data );
        virtual void setCompressedImage2D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, GLvoid const* data );
        virtual void setCompressedImage3D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, GLvoid const* data );
        virtual void setCompressedSubImage1D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLsizei width, GLenum format, GLsizei imageSize, GLvoid const* data );
        virtual void setCompressedSubImage2D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, GLvoid const* data );
        virtual void setCompressedSubImage3D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLint zOffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, GLvoid const* data );
        virtual void setFilterParameters( GLuint texture, GLenum target, GLenum minFilter, GLenum magFilter );
        virtual void setImage1D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLint border, GLenum format, GLenum type, GLvoid const* pixels );
        virtual void setImage2D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, GLvoid const* pixels );
        virtual void setImage3D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, GLvoid const* pixels );
        virtual void setLODParameters( GLuint texture, GLenum target, float minLOD, float maxLOD, float LODBias );
        virtual void setParameter( GLuint texture, GLenum target, GLenum pname, GLint param );
        virtual void setParameter( GLuint texture, GLenum target, GLenum pname, GLfloat param );
        virtual void setParameter( GLuint texture, GLenum target, GLenum pname, GLfloat const* param );
        virtual void setParameterUnmodified( GLuint texture, GLenum target, GLenum pname, GLint const* param );
        virtual void setParameterUnmodified( GLuint texture, GLenum target, GLenum pname, GLuint const* param );
        virtual void setSubImage1D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLsizei width, GLenum format, GLenum type, GLvoid const* pixels );
        virtual void setSubImage2D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid const* pixels );
        virtual void setSubImage3D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLint zOffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLvoid const* pixels );
        virtual void setWrapParameters( GLuint texture, GLenum target, GLenum wrapS, GLenum wrapT, GLenum wrapR );
    };

    class TextureBinding
    {
      public:
        TextureBinding( GLenum target, GLuint texture );
        ~TextureBinding();

      private:
        GLint  m_oldBinding;
        GLenum m_target;
    };

    inline GLenum getBindingTargetFromTarget( GLenum target )
    {
      switch( target )
      {
        case GL_TEXTURE_1D                    : return( GL_TEXTURE_BINDING_1D );
        case GL_TEXTURE_1D_ARRAY              : return( GL_TEXTURE_BINDING_1D_ARRAY );
        case GL_TEXTURE_2D                    : return( GL_TEXTURE_BINDING_2D );
        case GL_TEXTURE_2D_ARRAY              : return( GL_TEXTURE_BINDING_2D_ARRAY );
        case GL_TEXTURE_2D_MULTISAMPLE        : return( GL_TEXTURE_BINDING_2D_MULTISAMPLE );
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY  : return( GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY );
        case GL_TEXTURE_3D                    : return( GL_TEXTURE_BINDING_3D );
        case GL_TEXTURE_BUFFER                : return( GL_TEXTURE_BINDING_BUFFER );
        case GL_TEXTURE_CUBE_MAP              : return( GL_TEXTURE_BINDING_CUBE_MAP );
        case GL_TEXTURE_CUBE_MAP_ARRAY        : return( GL_TEXTURE_BINDING_CUBE_MAP_ARRAY );
        case GL_TEXTURE_RECTANGLE             : return( GL_TEXTURE_BINDING_RECTANGLE );
        default :
          DP_ASSERT( !"unknown texture target" );
          return( GL_INVALID_ENUM );
      }
    }

    inline void TextureBind::attachBuffer( GLuint texture, GLenum target, GLenum internalFormat, GLuint buffer )
    {
      TextureBinding tb( target, texture );
      glTexBuffer( target, internalFormat, buffer );
    }

    inline void TextureBind::generateMipMap( GLuint texture, GLenum target )
    {
      TextureBinding tb( target, texture );
      glTexParameteri( target, GL_TEXTURE_BASE_LEVEL, 0 );
      glGenerateMipmap( target );
    }

    inline void TextureBind::getCompressedImage( GLuint texture, GLenum target, GLint lod, GLvoid * img )
    {
      TextureBinding tb( target, texture );
      glGetCompressedTexImage( target, lod, img );
    }

    inline void TextureBind::getImage( GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, GLvoid * pixels )
    {
      TextureBinding tb( target, texture );
      glGetTexImage( target, level, format, type, pixels );
    }

    inline void TextureBind::setCompareParameters( GLuint texture, GLenum target, GLenum mode, GLenum func )
    {
      TextureBinding tb( target, texture );
      glTexParameteri( target, GL_TEXTURE_COMPARE_MODE, mode );
      glTexParameteri( target, GL_TEXTURE_COMPARE_FUNC, func );
    }

    inline void TextureBind::setCompressedImage1D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLint border, GLsizei imageSize, GLvoid const* data )
    {
      TextureBinding tb( target, texture );
      glCompressedTexImage1D( target, level, internalFormat, width, border, imageSize, data );
    }

    inline void TextureBind::setCompressedImage2D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, GLvoid const* data )
    {
      TextureBinding tb( target, texture );
      glCompressedTexImage2D( target, level, internalFormat, width, height, border, imageSize, data );
    }

    inline void TextureBind::setCompressedImage3D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, GLvoid const* data )
    {
      TextureBinding tb( target, texture );
      glCompressedTexImage3D( target, level, internalFormat, width, height, depth, border, imageSize, data );
    }

    inline void TextureBind::setCompressedSubImage1D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLsizei width, GLenum format, GLsizei imageSize, GLvoid const* data )
    {
      TextureBinding tb( target, texture );
      glCompressedTexSubImage1D( target, level, xOffset, width, format, imageSize, data );
    }

    inline void TextureBind::setCompressedSubImage2D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, GLvoid const* data )
    {
      TextureBinding tb( target, texture );
      glCompressedTexSubImage2D( target, level, xOffset, yOffset, width, height, format, imageSize, data );
    }

    inline void TextureBind::setCompressedSubImage3D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLint zOffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, GLvoid const* data )
    {
      TextureBinding tb( target, texture );
      glCompressedTexSubImage3D( target, level, xOffset, yOffset, zOffset, width, height, depth, format, imageSize, data );
    }

    inline void TextureBind::setFilterParameters( GLuint texture, GLenum target, GLenum minFilter, GLenum magFilter )
    {
      TextureBinding tb( target, texture );
      glTexParameteri( target, GL_TEXTURE_MIN_FILTER, minFilter );
      glTexParameteri( target, GL_TEXTURE_MAG_FILTER, magFilter );
    }

    inline void TextureBind::setImage1D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLint border, GLenum format, GLenum type, GLvoid const* pixels )
    {
      TextureBinding tb( target, texture );
      dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
      glTexImage1D( target, level, internalFormat, width, 0, format, type, pixels ); 
    }

    inline void TextureBind::setImage2D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, GLvoid const* pixels )
    {
      TextureBinding tb( target, texture );
      dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
      glTexImage2D( target, level, internalFormat, width, height, 0, format, type, pixels ); 
    }

    inline void TextureBind::setImage3D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, GLvoid const* pixels )
    {
      TextureBinding tb( target, texture );
      dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
      glTexImage3D( target, level, internalFormat, width, height, depth, 0, format, type, pixels ); 
    }

    inline void TextureBind::setLODParameters( GLuint texture, GLenum target, float minLOD, float maxLOD, float LODBias )
    {
      TextureBinding tb( target, texture );
      glTexParameterf( target, GL_TEXTURE_MIN_LOD, minLOD );
      glTexParameterf( target, GL_TEXTURE_MAX_LOD, maxLOD );
      glTexParameterf( target, GL_TEXTURE_LOD_BIAS, LODBias );
    }

    inline void TextureBind::setParameter( GLuint texture, GLenum target, GLenum pname, GLint param )
    {
      TextureBinding tb( target, texture );
      glTexParameteri( target, pname, param );
    }

    inline void TextureBind::setParameter( GLuint texture, GLenum target, GLenum pname, GLfloat param )
    {
      TextureBinding tb( target, texture );
      glTexParameterf( target, pname, param );
    }

    inline void TextureBind::setParameter( GLuint texture, GLenum target, GLenum pname, GLfloat const* param )
    {
      TextureBinding tb( target, texture );
      glTexParameterfv( target, pname, param );
    }

    inline void TextureBind::setParameterUnmodified( GLuint texture, GLenum target, GLenum pname, GLint const* param )
    {
      TextureBinding tb( target, texture );
      glTexParameterIiv( target, pname, param );
    }

    inline void TextureBind::setParameterUnmodified( GLuint texture, GLenum target, GLenum pname, GLuint const* param )
    {
      TextureBinding tb( target, texture );
      glTexParameterIuiv( target, pname, param );
    }

    inline void TextureBind::setSubImage1D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLsizei width, GLenum format, GLenum type, GLvoid const* pixels )
    {
      TextureBinding tb( target, texture );
      dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
      glTexSubImage1D( target, level, xOffset, width, format, type, pixels );
    }

    inline void TextureBind::setSubImage2D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid const* pixels )
    {
      TextureBinding tb( target, texture );
      dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
      glTexSubImage2D( target, level, xOffset, yOffset, width, height, format, type, pixels );
    }

    inline void TextureBind::setSubImage3D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLint zOffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLvoid const* pixels )
    {
      TextureBinding tb( target, texture );
      dp::gl::bind( GL_PIXEL_UNPACK_BUFFER, BufferSharedPtr::null );    // make sure, GL_PIXEL_UNPACK_BUFFER is unbound !
      glTexSubImage3D( target, level, xOffset, yOffset, zOffset, width, height, depth, format, type, pixels );
    }

    inline void TextureBind::setWrapParameters( GLuint texture, GLenum target, GLenum wrapS, GLenum wrapT, GLenum wrapR )
    {
      TextureBinding tb( target, texture );
      glTexParameteri( target, GL_TEXTURE_WRAP_S, wrapS );
      glTexParameteri( target, GL_TEXTURE_WRAP_T, wrapT );
      glTexParameteri( target, GL_TEXTURE_WRAP_R, wrapR );
    }

    inline TextureBinding::TextureBinding( GLenum target, GLuint texture )
      : m_target( target )
    {
      glGetIntegerv( getBindingTargetFromTarget( target ), &m_oldBinding );
      glBindTexture( m_target, texture );
    }

    inline TextureBinding::~TextureBinding()
    {
      glBindTexture( m_target, m_oldBinding );
    }

  } // namespace gl
} // namespace dp
