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

#include <dp/sg/gl/Config.h>
#include <dp/sg/gl/BufferGL.h>
#include <dp/sg/core/Texture.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/gl/Texture.h>


namespace dp
{
  namespace sg
  {
    namespace gl
    {
    #define NVSG_TIF_INVALID 0    //!< Invalid texture format

      //! GLTexImageFmt specifies the format and data type of a texture image.
      struct GLTexImageFmt
      {
        GLint intFmt;       //!< The OpenGL internal format of a texture.
        GLint usrFmt;       //!< The OpenGL user format of a texture
        GLint type;         //!< The OpenGL type of a texture
        GLint uploadHint;   //!< An upload hint for the texture: 0=glTexImage, 1=PBO
      };

      //! Internal data structure used to describe formats
      struct NVSGTexImageFmt
      {
        GLint fixedPtFmt;     //!< The OpenGL internal format for fixed point 
        GLint floatFmt;       //!< The OpenGL internal format for floating point
        GLint integerFmt;     //!< The OpenGL internal format integer textures
        GLint compressedFmt;  //!< The OpenGL internal format for compressed textures
        GLint nonLinearFmt;   //!< The OpenGL internal format SRGB textures
        GLint usrFmt;         //!< The 'typical' OpenGL user format of a texture (integer textures excluded)
        GLint type;           //!< The OpenGL type of a texture
        GLint uploadHint;     //!< An upload hint for the texture: 0=glTexImage, 1=PBO
      };

      class TextureGL : public dp::sg::core::Texture
      {
      public:
        DP_SG_GL_API static TextureGLSharedPtr create( const dp::gl::TextureSharedPtr& textureGL );
        DP_SG_GL_API virtual dp::sg::core::HandledObjectSharedPtr clone() const;

        DP_SG_GL_API const dp::gl::TextureSharedPtr& getTexture() const;
        DP_SG_GL_API static bool getTexImageFmt( GLTexImageFmt & tfmt, dp::sg::core::Image::PixelFormat fmt, dp::sg::core::Image::PixelDataType type, dp::sg::core::TextureHost::TextureGPUFormat gpufmt );
      protected:
        DP_SG_GL_API TextureGL( const dp::gl::TextureSharedPtr& texture );
      private:
        dp::gl::TextureSharedPtr m_texture;
      };

    } // namespace gl
  } // namespace sg
} // namespace dp
