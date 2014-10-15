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

#include <dp/gl/inc/ITexture.h>

namespace dp
{
  namespace gl
  {
    class TextureDSA : public ITexture
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

    inline void TextureDSA::attachBuffer( GLuint texture, GLenum target, GLenum internalFormat, GLuint buffer )
    {
      glTextureBufferEXT( texture, target, internalFormat, buffer );
    }

    inline void TextureDSA::generateMipMap( GLuint texture, GLenum target )
    {
      glTextureParameteriEXT( texture, target, GL_TEXTURE_BASE_LEVEL, 0 );
      glGenerateTextureMipmapEXT( texture, target );
    }

    inline void TextureDSA::getCompressedImage( GLuint texture, GLenum target, GLint lod, GLvoid * img )
    {
      glGetCompressedTextureImageEXT( texture, target, lod, img );
    }

    inline void TextureDSA::getImage( GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, GLvoid * pixels )
    {
      glGetTextureImageEXT( texture, target, level, format, type, pixels );
    }

    inline void TextureDSA::setCompareParameters( GLuint texture, GLenum target, GLenum mode, GLenum func )
    {
      glTextureParameteriEXT( texture, target, GL_TEXTURE_COMPARE_MODE, mode );
      glTextureParameteriEXT( texture, target, GL_TEXTURE_COMPARE_FUNC, func );
    }

    inline void TextureDSA::setCompressedImage1D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLint border, GLsizei imageSize, GLvoid const* data )
    {
      glCompressedTextureImage1DEXT( texture, target, level, internalFormat, width, border, imageSize, data );
    }

    inline void TextureDSA::setCompressedImage2D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, GLvoid const* data )
    {
      glCompressedTextureImage2DEXT( texture, target, level, internalFormat, width, height, border, imageSize, data );
    }

    inline void TextureDSA::setCompressedImage3D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, GLvoid const* data )
    {
      glCompressedTextureImage3DEXT( texture, target, level, internalFormat, width, height, depth, border, imageSize, data );
    }

    inline void TextureDSA::setCompressedSubImage1D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLsizei width, GLenum format, GLsizei imageSize, GLvoid const* data )
    {
      glCompressedTextureSubImage1DEXT( texture, target, level, xOffset, width, format, imageSize, data );
    }

    inline void TextureDSA::setCompressedSubImage2D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, GLvoid const* data )
    {
      glCompressedTextureSubImage2DEXT( texture, target, level, xOffset, yOffset, width, height, format, imageSize, data );
    }

    inline void TextureDSA::setCompressedSubImage3D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLint zOffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, GLvoid const* data )
    {
      glCompressedTextureSubImage3DEXT( texture, target, level, xOffset, yOffset, zOffset, width, height, depth, format, imageSize, data );
    }

    inline void TextureDSA::setFilterParameters( GLuint texture, GLenum target, GLenum minFilter, GLenum magFilter )
    {
      glTextureParameteriEXT( texture, target, GL_TEXTURE_MIN_FILTER, minFilter );
      glTextureParameteriEXT( texture, target, GL_TEXTURE_MAG_FILTER, magFilter );
    }

    inline void TextureDSA::setImage1D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLint border, GLenum format, GLenum type, GLvoid const* pixels )
    {
      glPushClientAttribDefaultEXT( GL_CLIENT_PIXEL_STORE_BIT );
      glTextureImage1DEXT( texture, target, level, internalFormat, width, 0, format, type, pixels ); 
      glPopClientAttrib();
    }

    inline void TextureDSA::setImage2D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, GLvoid const* pixels )
    {
      glPushClientAttribDefaultEXT( GL_CLIENT_PIXEL_STORE_BIT );
      glTextureImage2DEXT( texture, target, level, internalFormat, width, height, 0, format, type, pixels ); 
      glPopClientAttrib();
    }

    inline void TextureDSA::setImage3D( GLuint texture, GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, GLvoid const* pixels )
    {
      glPushClientAttribDefaultEXT( GL_CLIENT_PIXEL_STORE_BIT );
      glTextureImage3DEXT( texture, target, level, internalFormat, width, height, depth, 0, format, type, pixels ); 
      glPopClientAttrib();
    }

    inline void TextureDSA::setLODParameters( GLuint texture, GLenum target, float minLOD, float maxLOD, float LODBias )
    {
      glTextureParameterfEXT( texture, target, GL_TEXTURE_MIN_LOD, minLOD );
      glTextureParameterfEXT( texture, target, GL_TEXTURE_MAX_LOD, maxLOD );
      glTextureParameterfEXT( texture, target, GL_TEXTURE_LOD_BIAS, LODBias );
    }

    inline void TextureDSA::setParameter( GLuint texture, GLenum target, GLenum pname, GLint param )
    {
      glTextureParameteriEXT( texture, target, pname, param );
    }

    inline void TextureDSA::setParameter( GLuint texture, GLenum target, GLenum pname, GLfloat param )
    {
      glTextureParameterfEXT( texture, target, pname, param );
    }

    inline void TextureDSA::setParameter( GLuint texture, GLenum target, GLenum pname, GLfloat const* param )
    {
      glTextureParameterfvEXT( texture, target, pname, param );
    }

    inline void TextureDSA::setParameterUnmodified( GLuint texture, GLenum target, GLenum pname, GLint const* param )
    {
      glTextureParameterIivEXT( texture, target, pname, param );
    }

    inline void TextureDSA::setParameterUnmodified( GLuint texture, GLenum target, GLenum pname, GLuint const* param )
    {
      glTextureParameterIuivEXT( texture, target, pname, param );
    }

    inline void TextureDSA::setSubImage1D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLsizei width, GLenum format, GLenum type, GLvoid const* pixels )
    {
      glTextureSubImage1DEXT( texture, target, level, xOffset, width, format, type, pixels );
    }

    inline void TextureDSA::setSubImage2D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid const* pixels )
    {
      glTextureSubImage2DEXT( texture, target, level, xOffset, yOffset, width, height, format, type, pixels );
    }

    inline void TextureDSA::setSubImage3D( GLuint texture, GLenum target, GLint level, GLint xOffset, GLint yOffset, GLint zOffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLvoid const* pixels )
    {
      glTextureSubImage3DEXT( texture, target, level, xOffset, yOffset, zOffset, width, height, depth, format, type, pixels );
    }

    inline void TextureDSA::setWrapParameters( GLuint texture, GLenum target, GLenum wrapS, GLenum wrapT, GLenum wrapR )
    {
      glTextureParameteriEXT( texture, target, GL_TEXTURE_WRAP_S, wrapS );
      glTextureParameteriEXT( texture, target, GL_TEXTURE_WRAP_T, wrapT );
      glTextureParameteriEXT( texture, target, GL_TEXTURE_WRAP_R, wrapR );
    }
  } // namespace gl
} // namespace dp
