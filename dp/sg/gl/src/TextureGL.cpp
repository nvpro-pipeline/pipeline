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


#include <dp/sg/gl/TextureGL.h>

using namespace dp::sg::core;

namespace dp
{
  namespace sg
  {
    namespace gl
    {

      // glRecentTexImageFmts represents formats corresponding to recent hardware features.
      // Note: this table should be updated as hardware improves!
      dp::sg::gl::NVSGTexImageFmt recentTexImageFmts[Image::IMG_NUM_FORMATS][Image::IMG_NUM_TYPES] =
      {
        { // IMG_COLOR_INDEX - unsupported!! 
          { 0,0,0, 0,0, 0, 0,0 }   // IMG_BYTE
          ,  { 0,0,0, 0,0, 0, 0,0 }   // IMG_UNSIGNED_BYTE
          ,  { 0,0,0, 0,0, 0, 0,0 }   // IMG_SHORT
          ,  { 0,0,0, 0,0, 0, 0,0 }   // IMG_UNSIGNED_SHORT
          ,  { 0,0,0, 0,0, 0, 0,0 }   // IMG_INT
          ,  { 0,0,0, 0,0, 0, 0,0 }   // IMG_UNSIGNED_INT
          ,  { 0,0,0, 0,0, 0, 0,0 }   // IMG_FLOAT32
          ,  { 0,0,0, 0,0, 0, 0,0 }   // IMG_FLOAT16

          // custom formats
          ,  { 0,0,0,0,0,0,0,0 } //IMG_UNSIGNED_INT_2_10_10_10 //!< 2 Bits A, 10 bits RGB (RGB10/10A2)
          ,  { 0,0,0,0,0,0,0,0 } //IMG_UNSIGNED_INT_5_9_9_9    //!< 5 bits exponent 9 bits RGB mantissa
          //   (EXT_texture_shared_exponent) format
          ,  { 0,0,0,0,0,0,0,0 } //IMG_UNSIGNED_INT_10F_11F_11F //!< 5 bits exp + 5,6,6 bits RGB mantissa
          ,  { 0,0,0,0,0,0,0,0 } //IMG_UNSIGNED_INT_24_8 //!< 24 depth, 8 stencil
          //   (EXT_packed_depth_stencil) format
        }
        //
        // Formats that do not support the various combinations of these simply list the same enum for each
        // unsupported format.  There do not appear to be any support for compressed, signed, (NV?) formats
        // for the typical 1,2,3,4 component textures, other than LATC.
        //
        , { // IMG_RGB !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmt            type                upload

          { GL_SIGNED_RGB8_NV, GL_RGB16F_ARB,     GL_RGB8I_EXT,        GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT, 
            GL_RGB,            GL_BYTE,            0 }  // IMG_BYTE - NOTE: requires NV_texture_shader

          ,  { GL_RGB8,           GL_RGB16F_ARB,     GL_RGB8UI_EXT,        GL_COMPRESSED_RGB,  GL_SRGB8_EXT,
            GL_RGB,            GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_SIGNED_RGB8_NV, GL_RGB16F_ARB,     GL_RGB16I_EXT,       GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT,
            GL_RGB,            GL_SHORT,           0 }  // IMG_SHORT - NOTE: requires NV_texture_shader

          ,  { GL_RGB16,          GL_RGB16F_ARB,     GL_RGB16UI_EXT,      GL_COMPRESSED_RGB,  GL_SRGB8_EXT, 
            GL_RGB,            GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_RGB8_NV, GL_RGB32F_ARB,     GL_RGB32I_EXT,       GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT,  
            GL_RGB,            GL_INT,             0 }  // IMG_INT - NOTE: requires NV_texture_shader

          ,  { GL_RGB16,          GL_RGB32F_ARB,     GL_RGB32UI_EXT,      GL_COMPRESSED_RGB,  GL_SRGB8_EXT, 
            GL_RGB,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_RGB16,          GL_RGB32F_ARB,     GL_RGB32I_EXT,       GL_RGB32F_ARB,      GL_SRGB8_EXT,  
            GL_RGB,            GL_FLOAT,           0 }  // IMG_FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_RGB16,          GL_RGB16F_ARB,     GL_RGB32I_EXT,       GL_RGB16F_ARB,      GL_SRGB8_EXT,
            GL_RGB,            GL_HALF_FLOAT_NV,   0 }  // IMG_HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          //
          // requires 4-component format
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,   NVSG_TIF_INVALID,   NVSG_TIF_INVALID,    NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0}  // IMG_UNSIGNED_INT_2_10_10_10  

          // supported if EXT_texture_shared_exponent exported                                                                                                    
          ,  { GL_RGB9_E5_EXT,    GL_RGB9_E5_EXT,    GL_RGB9_E5_EXT,      GL_RGB9_E5_EXT,     GL_RGB9_E5_EXT,
            GL_RGB,            GL_UNSIGNED_INT_5_9_9_9_REV_EXT,   0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // supported if EXT_packed_float exported                                                                                                    
          ,  { GL_R11F_G11F_B10F_EXT,  GL_R11F_G11F_B10F_EXT,   GL_R11F_G11F_B10F_EXT,   GL_R11F_G11F_B10F_EXT,    GL_R11F_G11F_B10F_EXT,
            GL_RGB,            GL_UNSIGNED_INT_10F_11F_11F_REV_EXT, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,   NVSG_TIF_INVALID,   NVSG_TIF_INVALID,    NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8

        }
        , { // IMG_RGBA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          { GL_SIGNED_RGBA8_NV, GL_RGBA16F_ARB,   GL_RGBA8I_EXT,        GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT, 
            GL_RGBA,            GL_BYTE,            0 }  // IMG_BYTE - NOTE: requires NV_texture_shader

          ,  { GL_RGBA8,           GL_RGBA16F_ARB,   GL_RGBA8UI_EXT,       GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT,
            GL_RGBA,            GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_SIGNED_RGBA8_NV, GL_RGBA16F_ARB,   GL_RGBA16I_EXT,       GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT,
            GL_RGBA,            GL_SHORT,           0 }  // IMG_SHORT - NOTE: requires NV_texture_shader

          ,  { GL_RGBA16,          GL_RGBA16F_ARB,   GL_RGBA16UI_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT, 
            GL_RGBA,            GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_RGBA8_NV, GL_RGBA32F_ARB,   GL_RGBA32I_EXT,       GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT,  
            GL_RGBA,            GL_INT,             0 }  // IMG_INT - NOTE: requires NV_texture_shader

          ,  { GL_RGBA16,          GL_RGBA32F_ARB,   GL_RGBA32UI_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT, 
            GL_RGBA,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT - NOTE: precision loss!

          // fast upload format
          ,  { GL_RGBA16,          GL_RGBA32F_ARB,   GL_RGBA32I_EXT,       GL_RGBA32F_ARB,      GL_SRGB8_ALPHA8_EXT,  
            GL_RGBA,            GL_FLOAT,           1 }  // IMG_FLOAT - NOTE: requires ARB_texture_float

          // fast upload format
          ,  { GL_RGBA16,          GL_RGBA16F_ARB,   GL_RGBA32I_EXT,       GL_RGBA16F_ARB,      GL_SRGB8_ALPHA8_EXT,
            GL_RGBA,            GL_HALF_FLOAT_NV,   1 }  // IMG_HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 

          // G80 native    
          ,  { GL_RGB10_A2,       GL_RGBA16F_ARB,    GL_RGBA16I_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT,
            GL_RGBA,           GL_UNSIGNED_INT_2_10_10_10_REV,   0 }  // IMG_UNSIGNED_INT_2_10_10_10  

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,    NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9
          // EXT_texture_shared_exponent
          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,    NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,    NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0  } // IMG_UNSIGNED_INT_24_8

        }
        , { // IMG_BGR !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          { GL_SIGNED_RGB8_NV, GL_RGB16F_ARB,     GL_RGB8I_EXT,        GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT, 
            GL_BGR,            GL_BYTE,            0 }  // IMG_BYTE - NOTE: requires NV_texture_shader

          ,  { GL_RGB8,           GL_RGB16F_ARB,     GL_RGB8UI_EXT,       GL_COMPRESSED_RGB,  GL_SRGB8_EXT,
            GL_BGR,            GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_SIGNED_RGB8_NV, GL_RGB16F_ARB,     GL_RGB16I_EXT,       GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT,
            GL_BGR,            GL_SHORT,           0 }  // IMG_SHORT - NOTE: requires NV_texture_shader

          ,  { GL_RGB16,          GL_RGB16F_ARB,     GL_RGB16UI_EXT,      GL_COMPRESSED_RGB,  GL_SRGB8_EXT, 
            GL_BGR,            GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_RGB8_NV, GL_RGB32F_ARB,     GL_RGB32I_EXT,       GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT,  
            GL_BGR,            GL_INT,             0 }  // IMG_INT - NOTE: requires NV_texture_shader

          ,  { GL_RGB16,          GL_RGB32F_ARB,     GL_RGB32UI_EXT,      GL_COMPRESSED_RGB,  GL_SRGB8_EXT, 
            GL_BGR,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_RGB16,          GL_RGB32F_ARB,     GL_RGB32I_EXT,       GL_RGB32F_ARB,      GL_SRGB8_EXT,  
            GL_BGR,            GL_FLOAT,           0 }  // IMG_FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_RGB16,          GL_RGB16F_ARB,     GL_RGB32I_EXT,       GL_RGB16F_ARB,      GL_SRGB8_EXT,
            GL_BGR,            GL_HALF_FLOAT_NV,   0 }  // IMG_HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // meaningless because RGB layout doesn't map to 32-bit input data. would crash for 8-bit per channel user data

          // requires 4-component format
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0}  // IMG_UNSIGNED_INT_2_10_10_10  
          // G80 native
          // requires RGB                                                                                                     
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB                                                                                                     
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,    NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0  }  // IMG_UNSIGNED_INT_24_8

        }
        , { // IMG_BGRA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          // fast upload format
          { GL_SIGNED_RGBA8_NV, GL_RGBA16F_ARB,   GL_RGBA8I_EXT,        GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT, 
            GL_BGRA,            GL_BYTE,            1 }  // IMG_BYTE - NOTE: requires NV_texture_shader

          // fast upload format
          ,  { GL_RGBA8,           GL_RGBA16F_ARB,   GL_RGBA8UI_EXT,       GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT,
            GL_BGRA,            GL_UNSIGNED_BYTE,   1 }  // IMG_UNSIGNED_BYTE

          ,  { GL_SIGNED_RGBA8_NV, GL_RGBA16F_ARB,   GL_RGBA16I_EXT,       GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT,
            GL_BGRA,            GL_SHORT,           0 }  // IMG_SHORT - NOTE: requires NV_texture_shader

          ,  { GL_RGBA16,          GL_RGBA16F_ARB,   GL_RGBA16UI_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT, 
            GL_BGRA,            GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_RGBA8_NV, GL_RGBA32F_ARB,   GL_RGBA32I_EXT,       GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT,  
            GL_BGRA,            GL_INT,             0 }  // IMG_INT - NOTE: requires NV_texture_shader

          ,  { GL_RGBA16,          GL_RGBA32F_ARB,   GL_RGBA32UI_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT, 
            GL_BGRA,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_RGBA16,          GL_RGBA32F_ARB,   GL_RGBA32I_EXT,       GL_RGBA32F_ARB,      GL_SRGB8_ALPHA8_EXT,  
            GL_BGRA,            GL_FLOAT,           0 }  // IMG_FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_RGBA16,          GL_RGBA16F_ARB,   GL_RGBA32I_EXT,       GL_RGBA16F_ARB,      GL_SRGB8_ALPHA8_EXT,
            GL_BGRA,            GL_HALF_FLOAT_NV,   0 }  // IMG_HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // G80 native 
          ,  { GL_RGB10_A2,       GL_RGBA16F_ARB,    GL_RGBA16I_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT,
            GL_BGRA,           GL_UNSIGNED_INT_2_10_10_10_REV,   0 }  // IMG_UNSIGNED_INT_2_10_10_10  

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0  }  // IMG_UNSIGNED_INT_24_8

        }
        , { // IMG_LUMINANCE !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          { GL_SIGNED_LUMINANCE8_NV, GL_LUMINANCE16F_ARB,   GL_LUMINANCE8I_EXT,  GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT, GL_SLUMINANCE8_EXT, 
            GL_LUMINANCE,            GL_BYTE,            0 }  // IMG_BYTE - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE8,           GL_LUMINANCE16F_ARB,   GL_LUMINANCE8UI_EXT,       GL_COMPRESSED_LUMINANCE_LATC1_EXT,  GL_SLUMINANCE8_EXT,
            GL_LUMINANCE,            GL_UNSIGNED_BYTE,   1 }  // IMG_UNSIGNED_BYTE

          ,  { GL_SIGNED_LUMINANCE8_NV, GL_LUMINANCE16F_ARB,   GL_LUMINANCE16I_EXT,       GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT,  GL_SLUMINANCE8_EXT,
            GL_LUMINANCE,            GL_SHORT,           0 }  // IMG_SHORT - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE16,          GL_LUMINANCE16F_ARB,     GL_LUMINANCE16UI_EXT,    GL_COMPRESSED_LUMINANCE_LATC1_EXT,  GL_SLUMINANCE8_EXT, 
            GL_LUMINANCE,            GL_UNSIGNED_SHORT,  1 }  // IMG_UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_LUMINANCE8_NV, GL_LUMINANCE32F_ARB,     GL_LUMINANCE32I_EXT,     GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT,  
            GL_SLUMINANCE8_EXT,  
            GL_LUMINANCE,            GL_INT,             0 }  // IMG_INT - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE16,          GL_LUMINANCE32F_ARB,     GL_LUMINANCE32UI_EXT,      GL_COMPRESSED_LUMINANCE_LATC1_EXT,  
            GL_SLUMINANCE8_EXT, 
            GL_LUMINANCE,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_LUMINANCE16,          GL_LUMINANCE32F_ARB,     GL_LUMINANCE32I_EXT,       GL_COMPRESSED_LUMINANCE_LATC1_EXT,
            GL_SLUMINANCE8_EXT,  
            GL_LUMINANCE,            GL_FLOAT,           0 }  // IMG_FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_LUMINANCE16,          GL_LUMINANCE16F_ARB,     GL_LUMINANCE32I_EXT,       GL_COMPRESSED_LUMINANCE_LATC1_EXT, 
            GL_SLUMINANCE8_EXT,
            GL_LUMINANCE,            GL_HALF_FLOAT_NV,   0 }  // IMG_HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // requires 4-component format      
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,    NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0  }  // IMG_UNSIGNED_INT_24_8

        }
        , { // IMG_LUMINANCE_ALPHA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          { GL_SIGNED_LUMINANCE8_ALPHA8_NV, GL_LUMINANCE_ALPHA16F_ARB,   GL_LUMINANCE_ALPHA8I_EXT,  
            GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT,  GL_SLUMINANCE8_ALPHA8_EXT, 
            GL_LUMINANCE_ALPHA,            GL_BYTE,            0 }  // IMG_BYTE - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE8_ALPHA8,           GL_LUMINANCE_ALPHA16F_ARB,   GL_LUMINANCE_ALPHA8UI_EXT,       GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,  
            GL_SLUMINANCE8_ALPHA8_EXT,
            GL_LUMINANCE_ALPHA,            GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_SIGNED_LUMINANCE8_ALPHA8_NV, GL_LUMINANCE_ALPHA16F_ARB,   GL_LUMINANCE_ALPHA16I_EXT,       
            GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT,  GL_SLUMINANCE8_ALPHA8_EXT,
            GL_LUMINANCE_ALPHA,            GL_SHORT,           0 }  // IMG_SHORT - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE16_ALPHA16,          GL_LUMINANCE_ALPHA16F_ARB,     GL_LUMINANCE_ALPHA16UI_EXT,    
            GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,  GL_SLUMINANCE8_ALPHA8_EXT, 
            GL_LUMINANCE_ALPHA,            GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_LUMINANCE8_ALPHA8_NV, GL_LUMINANCE_ALPHA32F_ARB,     GL_LUMINANCE_ALPHA32I_EXT,     
            GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT,  
            GL_SLUMINANCE8_ALPHA8_EXT,  
            GL_LUMINANCE_ALPHA,             GL_INT,             0 }  // IMG_INT - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE16_ALPHA16,          GL_LUMINANCE_ALPHA32F_ARB,     GL_LUMINANCE_ALPHA32UI_EXT,      
            GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,  
            GL_SLUMINANCE8_ALPHA8_EXT, 
            GL_LUMINANCE_ALPHA,             GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_LUMINANCE16_ALPHA16,          GL_LUMINANCE_ALPHA32F_ARB,     GL_LUMINANCE_ALPHA32I_EXT,       
            GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,
            GL_SLUMINANCE8_ALPHA8_EXT,  
            GL_LUMINANCE_ALPHA,             GL_FLOAT,           0 }  // IMG_FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_LUMINANCE16_ALPHA16,          GL_LUMINANCE_ALPHA16F_ARB,     GL_LUMINANCE_ALPHA32I_EXT, GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,
            GL_SLUMINANCE8_ALPHA8_EXT,
            GL_LUMINANCE_ALPHA,             GL_HALF_FLOAT_NV,   0 }  // IMG_HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // requires 4-component format 
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F  

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8

        }
        , { // IMG_ALPHA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_SIGNED_ALPHA8_NV, GL_ALPHA16F_ARB,   GL_ALPHA8I_EXT,  GL_SIGNED_ALPHA8_NV, GL_SIGNED_ALPHA8_NV, 
            GL_ALPHA,            GL_BYTE,            0 }  // IMG_BYTE - NOTE: requires NV_texture_shader

          ,  { GL_ALPHA8,           GL_ALPHA16F_ARB,   GL_ALPHA8UI_EXT,       GL_ALPHA8,  GL_ALPHA8,
            GL_ALPHA,            GL_UNSIGNED_BYTE,   1 }  // IMG_UNSIGNED_BYTE

          ,  { GL_SIGNED_ALPHA8_NV, GL_ALPHA16F_ARB,   GL_ALPHA16I_EXT,       GL_SIGNED_ALPHA8_NV,  GL_SIGNED_ALPHA8_NV,
            GL_ALPHA,            GL_SHORT,           0 }  // IMG_SHORT - NOTE: requires NV_texture_shader

          ,  { GL_ALPHA16,          GL_ALPHA16F_ARB,     GL_ALPHA16UI_EXT,    GL_ALPHA16,  GL_ALPHA16, 
            GL_ALPHA,            GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT

          ,  { GL_SIGNED_ALPHA8_NV, GL_ALPHA32F_ARB,     GL_ALPHA32I_EXT,     GL_SIGNED_ALPHA8_NV,  GL_SIGNED_ALPHA8_NV,  
            GL_ALPHA,            GL_INT,             0 }  // IMG_INT - NOTE: requires NV_texture_shader

          ,  { GL_ALPHA16,          GL_ALPHA32F_ARB,     GL_ALPHA32UI_EXT,      GL_ALPHA16,  GL_ALPHA16, 
            GL_ALPHA,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT

          ,  { GL_ALPHA16,          GL_ALPHA32F_ARB,     GL_ALPHA32I_EXT,       GL_ALPHA16, GL_ALPHA16,  
            GL_ALPHA,            GL_FLOAT,           0 }  // IMG_FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_ALPHA16,          GL_ALPHA16F_ARB,     GL_ALPHA32I_EXT,       GL_ALPHA16, GL_ALPHA16, 
            GL_ALPHA,            GL_HALF_FLOAT_NV,   0 }  // IMG_HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // 1-component format does not match. requires 4-component format (RGBA|BGRA)
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 } // IMG_UNSIGNED_INT_2_10_10_10 

          // requires RGB format
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB format
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8
        }
        , { // IMG_DEPTH_COMPONENT - Requires ARB_depth_texture

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_BYTE,            0 }  // IMG_BYTE - NOTE: requires ARB_depth_texture

          ,  { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_UNSIGNED_BYTE,            0 }  // IMG_UNSIGNED_BYTE 

          ,  { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_SHORT,            0 }  // IMG_SHORT 

          ,  { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_SHORT,            0 }  // IMG_UNSIGNED_SHORT 

          ,  { GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB,   GL_DEPTH_COMPONENT32_ARB,  GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB, 
            GL_DEPTH_COMPONENT,            GL_INT,            0 }  // IMG_INT 

          ,  { GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB,   GL_DEPTH_COMPONENT32_ARB,  GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB, 
            GL_DEPTH_COMPONENT,            GL_UNSIGNED_INT,            0 }  // IMG_UNSIGNED_INT 

          ,  { GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB,   GL_DEPTH_COMPONENT32_ARB,  GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB, 
            GL_DEPTH_COMPONENT,            GL_FLOAT,            0 }  // IMG_FLOAT 

          ,  { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_HALF_FLOAT_NV,       0 }  // IMG_HALF 

          //
          // custom formats
          // 
          // requires 4-component format (RGBA|BGRA)
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 } // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB format
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB format
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 } // IMG_UNSIGNED_INT_10F_11F_11F 

          // load only depth pixels out of the depth stencil - supported if EXT_packed_depth_stencil exported
          ,  { GL_DEPTH_COMPONENT24_ARB,  GL_DEPTH_COMPONENT24_ARB,  GL_DEPTH_COMPONENT24_ARB,  GL_DEPTH_COMPONENT24_ARB,  GL_DEPTH_COMPONENT24_ARB,
            GL_DEPTH_STENCIL_EXT,     GL_UNSIGNED_INT_24_8_EXT, 0 } // IMG_UNSIGNED_INT_24_8
        }
        , { // IMG_DEPTH_STENCIL - Requires EXT_packed_depth_stencil

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0}  // IMG_BYTE

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0}  // IMG_UNSIGNED_BYTE 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_SHORT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_SHORT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_INT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_FLOAT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_HALF 

          //
          // custom formats
          // 
          // format - type mismatch 
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // format - type mismatch 
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // format - type mismatch 
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // perfect match of format and type - supported if EXT_packed_depth_stencil exported
          // fast upload format
          ,  { GL_DEPTH24_STENCIL8_EXT,  GL_DEPTH24_STENCIL8_EXT,   GL_DEPTH24_STENCIL8_EXT,   GL_DEPTH24_STENCIL8_EXT,    GL_DEPTH24_STENCIL8_EXT,
            GL_DEPTH_STENCIL_EXT,     GL_UNSIGNED_INT_24_8_EXT, 1 }  // IMG_UNSIGNED_INT_24_8
        }
        , { // IMG_INTEGER_ALPHA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_ALPHA8I_EXT, GL_ALPHA8I_EXT,   GL_ALPHA8I_EXT,  GL_ALPHA8I_EXT, GL_ALPHA8I_EXT, 
            GL_ALPHA_INTEGER_EXT,  GL_BYTE,            0 }  // IMG_BYTE

          ,  { GL_ALPHA8UI_EXT,        GL_ALPHA8UI_EXT,   GL_ALPHA8UI_EXT,       GL_ALPHA8UI_EXT,  GL_ALPHA8UI_EXT,
            GL_ALPHA_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_ALPHA16I_EXT, GL_ALPHA16I_EXT,   GL_ALPHA16I_EXT,    GL_ALPHA16I_EXT,  GL_ALPHA16I_EXT,
            GL_ALPHA_INTEGER_EXT,   GL_SHORT,           0 }  // IMG_SHORT

          ,  { GL_ALPHA16UI_EXT,          GL_ALPHA16UI_EXT,     GL_ALPHA16UI_EXT,    GL_ALPHA16UI_EXT,  GL_ALPHA16UI_EXT, 
            GL_ALPHA_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT

          ,  { GL_ALPHA32I_EXT, GL_ALPHA32I_EXT,     GL_ALPHA32I_EXT,     GL_ALPHA32I_EXT,  GL_ALPHA32I_EXT,  
            GL_ALPHA_INTEGER_EXT,            GL_INT,             0 }  // IMG_INT -

          ,  { GL_ALPHA32UI_EXT,          GL_ALPHA32UI_EXT,     GL_ALPHA32UI_EXT,      GL_ALPHA32UI_EXT,  GL_ALPHA32UI_EXT, 
            GL_ALPHA,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_FLOAT - float data with integer fmt = invalid

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 
          // requires 4-component format 
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INY_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8
        }
        , { // IMG_INTEGER_LUMINANCE !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_LUMINANCE8I_EXT,        GL_LUMINANCE8I_EXT,   GL_LUMINANCE8I_EXT,  GL_LUMINANCE8I_EXT, GL_LUMINANCE8I_EXT, 
            GL_LUMINANCE_INTEGER_EXT,  GL_BYTE,            0 }  // IMG_BYTE

          ,  { GL_LUMINANCE8UI_EXT,       GL_LUMINANCE8UI_EXT,   GL_LUMINANCE8UI_EXT,       GL_LUMINANCE8UI_EXT,  GL_LUMINANCE8UI_EXT,
            GL_LUMINANCE_INTEGER_EXT,  GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_LUMINANCE16I_EXT, GL_LUMINANCE16I_EXT,   GL_LUMINANCE16I_EXT,    GL_LUMINANCE16I_EXT,  GL_LUMINANCE16I_EXT,
            GL_LUMINANCE_INTEGER_EXT,  GL_SHORT,           0 }  // IMG_SHORT

          ,  { GL_LUMINANCE16UI_EXT,      GL_LUMINANCE16UI_EXT,     GL_LUMINANCE16UI_EXT,    GL_LUMINANCE16UI_EXT,  GL_LUMINANCE16UI_EXT, 
            GL_LUMINANCE_INTEGER_EXT,  GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT

          ,  { GL_LUMINANCE32I_EXT, GL_LUMINANCE32I_EXT,     GL_LUMINANCE32I_EXT,     GL_LUMINANCE32I_EXT,  GL_LUMINANCE32I_EXT,  
            GL_LUMINANCE_INTEGER_EXT,  GL_INT,             0 }  // IMG_INT -

          ,  { GL_LUMINANCE32UI_EXT,      GL_LUMINANCE32UI_EXT,     GL_LUMINANCE32UI_EXT,      GL_LUMINANCE32UI_EXT,  GL_LUMINANCE32UI_EXT, 
            GL_LUMINANCE,              GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_FLOAT - float data with integer fmt = invalid

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 
          // requires 4-component format (RGBA|BGRA) 
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8
        }
        , { // IMG_INTEGER_LUMINANCE_ALPHA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_LUMINANCE_ALPHA8I_EXT, GL_LUMINANCE_ALPHA8I_EXT,   GL_LUMINANCE_ALPHA8I_EXT,  GL_LUMINANCE_ALPHA8I_EXT, GL_LUMINANCE_ALPHA8I_EXT, 
            GL_LUMINANCE_ALPHA_INTEGER_EXT,  GL_BYTE,            0 }  // IMG_BYTE

          ,  { GL_LUMINANCE_ALPHA8UI_EXT,        GL_LUMINANCE_ALPHA8UI_EXT,   GL_LUMINANCE_ALPHA8UI_EXT,       GL_LUMINANCE_ALPHA8UI_EXT,  GL_LUMINANCE_ALPHA8UI_EXT,
            GL_LUMINANCE_ALPHA_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_LUMINANCE_ALPHA16I_EXT, GL_LUMINANCE_ALPHA16I_EXT,   GL_LUMINANCE_ALPHA16I_EXT,    GL_LUMINANCE_ALPHA16I_EXT,  GL_LUMINANCE_ALPHA16I_EXT,
            GL_LUMINANCE_ALPHA_INTEGER_EXT,   GL_SHORT,           0 }  // IMG_SHORT

          ,  { GL_LUMINANCE_ALPHA16UI_EXT,          GL_LUMINANCE_ALPHA16UI_EXT,     GL_LUMINANCE_ALPHA16UI_EXT,    GL_LUMINANCE_ALPHA16UI_EXT,  GL_LUMINANCE_ALPHA16UI_EXT, 
            GL_LUMINANCE_ALPHA_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT

          ,  { GL_LUMINANCE_ALPHA32I_EXT, GL_LUMINANCE_ALPHA32I_EXT,     GL_LUMINANCE_ALPHA32I_EXT,     GL_LUMINANCE_ALPHA32I_EXT,  GL_LUMINANCE_ALPHA32I_EXT,  
            GL_LUMINANCE_ALPHA_INTEGER_EXT,            GL_INT,             0 }  // IMG_INT -

          ,  { GL_LUMINANCE_ALPHA32UI_EXT,          GL_LUMINANCE_ALPHA32UI_EXT,     GL_LUMINANCE_ALPHA32UI_EXT,      GL_LUMINANCE_ALPHA32UI_EXT,  GL_LUMINANCE_ALPHA32UI_EXT, 
            GL_LUMINANCE_ALPHA,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_FLOAT - float data with integer fmt = invalid

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 

          // requires 4-component format (RGBA|BGRA) 
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8
        }
        , { // IMG_INTEGER_RGB !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_RGB8I_EXT, GL_RGB8I_EXT,   GL_RGB8I_EXT,  GL_RGB8I_EXT, GL_RGB8I_EXT, 
            GL_RGB_INTEGER_EXT,  GL_BYTE,            0 }  // IMG_BYTE

          ,  { GL_RGB8UI_EXT,        GL_RGB8UI_EXT,   GL_RGB8UI_EXT,       GL_RGB8UI_EXT,  GL_RGB8UI_EXT,
            GL_RGB_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_RGB16I_EXT, GL_RGB16I_EXT,   GL_RGB16I_EXT,    GL_RGB16I_EXT,  GL_RGB16I_EXT,
            GL_RGB_INTEGER_EXT,   GL_SHORT,           0 }  // IMG_SHORT

          ,  { GL_RGB16UI_EXT,          GL_RGB16UI_EXT,     GL_RGB16UI_EXT,    GL_RGB16UI_EXT,  GL_RGB16UI_EXT, 
            GL_RGB_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT

          ,  { GL_RGB32I_EXT, GL_RGB32I_EXT,     GL_RGB32I_EXT,     GL_RGB32I_EXT,  GL_RGB32I_EXT,  
            GL_RGB_INTEGER_EXT,            GL_INT,             0 }  // IMG_INT -

          ,  { GL_RGB32UI_EXT,          GL_RGB32UI_EXT,     GL_RGB32UI_EXT,      GL_RGB32UI_EXT,  GL_RGB32UI_EXT, 
            GL_RGB,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_FLOAT - float data with integer fmt = invalid

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 

          // requires 4-component format (RGBA|BGRA) 
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8
        }
        , { // IMG_INTEGER_BGR !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_RGB8I_EXT, GL_RGB8I_EXT,   GL_RGB8I_EXT,  GL_RGB8I_EXT, GL_RGB8I_EXT, 
            GL_BGR_INTEGER_EXT,  GL_BYTE,            0 }  // IMG_BYTE

          ,  { GL_RGB8UI_EXT,        GL_RGB8UI_EXT,   GL_RGB8UI_EXT,       GL_RGB8UI_EXT,  GL_RGB8UI_EXT,
            GL_BGR_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_RGB16I_EXT, GL_RGB16I_EXT,   GL_RGB16I_EXT,    GL_RGB16I_EXT,  GL_RGB16I_EXT,
            GL_BGR_INTEGER_EXT,   GL_SHORT,           0 }  // IMG_SHORT

          ,  { GL_RGB16UI_EXT,          GL_RGB16UI_EXT,     GL_RGB16UI_EXT,    GL_RGB16UI_EXT,  GL_RGB16UI_EXT, 
            GL_BGR_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT

          ,  { GL_RGB32I_EXT, GL_RGB32I_EXT,     GL_RGB32I_EXT,     GL_RGB32I_EXT,  GL_RGB32I_EXT,  
            GL_BGR_INTEGER_EXT,            GL_INT,             0 }  // IMG_INT -

          ,  { GL_RGB32UI_EXT,          GL_RGB32UI_EXT,     GL_RGB32UI_EXT,      GL_RGB32UI_EXT,  GL_RGB32UI_EXT, 
            GL_BGR_INTEGER_EXT,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_FLOAT - float data with integer fmt = invalid

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 

          // requires 4-component format (RGBA|BGRA) 
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8
        }
        , { // IMG_INTEGER_RGBA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_RGBA8I_EXT, GL_RGBA8I_EXT,   GL_RGBA8I_EXT,  GL_RGBA8I_EXT, GL_RGBA8I_EXT, 
            GL_RGBA_INTEGER_EXT,  GL_BYTE,            0 }  // IMG_BYTE

          ,  { GL_RGBA8UI_EXT,        GL_RGBA8UI_EXT,   GL_RGBA8UI_EXT,       GL_RGBA8UI_EXT,  GL_RGBA8UI_EXT,
            GL_RGBA_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_RGBA16I_EXT, GL_RGBA16I_EXT,   GL_RGBA16I_EXT,    GL_RGBA16I_EXT,  GL_RGBA16I_EXT,
            GL_RGBA_INTEGER_EXT,   GL_SHORT,           0 }  // IMG_SHORT

          ,  { GL_RGBA16UI_EXT,          GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,    GL_RGBA16UI_EXT,  GL_RGBA16UI_EXT, 
            GL_RGBA_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT

          ,  { GL_RGBA32I_EXT, GL_RGBA32I_EXT,     GL_RGBA32I_EXT,     GL_RGBA32I_EXT,  GL_RGBA32I_EXT,  
            GL_RGBA_INTEGER_EXT,            GL_INT,             0 }  // IMG_INT -

          ,  { GL_RGBA32UI_EXT,          GL_RGBA32UI_EXT,     GL_RGBA32UI_EXT,      GL_RGBA32UI_EXT,  GL_RGBA32UI_EXT, 
            GL_RGBA_INTEGER_EXT,            GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_FLOAT - float data with integer fmt = invalid

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 

          // not sure if this makes sense, but it should work according to the specs 
          ,  { GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,    GL_RGBA16UI_EXT,  GL_RGBA16UI_EXT,
            GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT_2_10_10_10_REV,   0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8
        }
        , { // IMG_INTEGER_BGRA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_RGBA8I_EXT, GL_RGBA8I_EXT,   GL_RGBA8I_EXT,  GL_RGBA8I_EXT, GL_RGBA8I_EXT, 
            GL_BGRA_INTEGER_EXT,  GL_BYTE,            0 }  // IMG_BYTE

          ,  { GL_RGBA8UI_EXT,        GL_RGBA8UI_EXT,   GL_RGBA8UI_EXT,       GL_RGBA8UI_EXT,  GL_RGBA8UI_EXT,
            GL_BGRA_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // IMG_UNSIGNED_BYTE

          ,  { GL_RGBA16I_EXT, GL_RGBA16I_EXT,   GL_RGBA16I_EXT,    GL_RGBA16I_EXT,  GL_RGBA16I_EXT,
            GL_BGRA_INTEGER_EXT,   GL_SHORT,           0 }  // IMG_SHORT

          ,  { GL_RGBA16UI_EXT,          GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,    GL_RGBA16UI_EXT,  GL_RGBA16UI_EXT, 
            GL_BGRA_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // IMG_UNSIGNED_SHORT

          ,  { GL_RGBA32I_EXT, GL_RGBA32I_EXT,     GL_RGBA32I_EXT,     GL_RGBA32I_EXT,  GL_RGBA32I_EXT,  
            GL_BGRA_INTEGER_EXT, GL_INT,             0 }  // IMG_INT -

          ,  { GL_RGBA32UI_EXT,     GL_RGBA32UI_EXT,     GL_RGBA32UI_EXT,      GL_RGBA32UI_EXT,  GL_RGBA32UI_EXT, 
            GL_BGRA_INTEGER_EXT, GL_UNSIGNED_INT,    0 }  // IMG_UNSIGNED_INT

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_FLOAT - float data with integer fmt = invalid

          ,  { NVSG_TIF_INVALID,    NVSG_TIF_INVALID,     NVSG_TIF_INVALID,       NVSG_TIF_INVALID, NVSG_TIF_INVALID,  
            NVSG_TIF_INVALID,    0,           0 }  // IMG_HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 
          // not sure if this makes sense, but it should work according to the specs 
          ,  { GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,    GL_RGBA16UI_EXT,  GL_RGBA16UI_EXT,
            GL_BGRA_INTEGER_EXT, GL_UNSIGNED_INT_2_10_10_10_REV,   0 }  // IMG_UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { NVSG_TIF_INVALID,  NVSG_TIF_INVALID,  NVSG_TIF_INVALID,    NVSG_TIF_INVALID,   NVSG_TIF_INVALID,
            NVSG_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INT_24_8
        }

        //
        // NOTE - THE REST ARE SET UP PROGRAMMATICALLY BELOW
        //
      };

      class TexImageFmts
      {
      public: 
        TexImageFmts();
        dp::sg::gl::NVSGTexImageFmt getFmt( Image::PixelFormat pf, Image::PixelDataType pdt );
        dp::sg::gl::NVSGTexImageFmt* getFmts( Image::PixelFormat pf );

      private:
        void initialize();

      private:
        bool initialized;
        dp::sg::gl::NVSGTexImageFmt m_texImageFmts[Image::IMG_NUM_FORMATS][Image::IMG_NUM_TYPES];     
      };

      TexImageFmts::TexImageFmts()
        : initialized( false )
      {}

      dp::sg::gl::NVSGTexImageFmt TexImageFmts::getFmt( Image::PixelFormat pf, Image::PixelDataType pdt )
      {
        if( !initialized )
        {
          initialize();
          initialized = true;
        }

        return m_texImageFmts[pf][pdt];
      }

      dp::sg::gl::NVSGTexImageFmt* TexImageFmts::getFmts( Image::PixelFormat pf )
      {
        if( !initialized )
        {
          initialize();
          initialized = true;
        }

        return &m_texImageFmts[pf][0];
      }

      void TexImageFmts::initialize()
      { 
        // initialize texture image formats according to current hardware and driver configuration

        // NOTE: we first setup for most recent hardware features.
        // later on we probably need to fallback if features are not available on current configuration!
        DP_ASSERT( sizeof(m_texImageFmts) == sizeof(recentTexImageFmts) );
        memcpy(m_texImageFmts, recentTexImageFmts, sizeof(m_texImageFmts));

        if ( !GLEW_ARB_texture_float    // NOTE: both extensions map to same
          && !GLEW_ATI_texture_float )  // GL_XXX defines (ARB based on ATI)
        {
          // internal floating point formats not available on current configuration!
          // fallback to fixed format as internal format and let the driver choose

          for( size_t i = 0; i < Image::IMG_NUM_FORMATS; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j ++ )
            {
              m_texImageFmts[i][j].floatFmt = m_texImageFmts[i][j].fixedPtFmt;
            }
          }

          // also turn off "fast upload"
          m_texImageFmts[Image::IMG_RGBA][Image::IMG_FLOAT16].uploadHint = 0;
          m_texImageFmts[Image::IMG_RGBA][Image::IMG_FLOAT32].uploadHint = 0;
        }

        if ( !GLEW_NV_texture_shader )
        {
          //
          // Just set the signed formats to be the same as the unsigned.
          //
          for( size_t i = 0; i < Image::IMG_NUM_FORMATS; i ++ )
          {
            m_texImageFmts[i][Image::IMG_BYTE].fixedPtFmt = 
              m_texImageFmts[i][Image::IMG_UNSIGNED_BYTE].fixedPtFmt;
            m_texImageFmts[i][Image::IMG_SHORT].fixedPtFmt = 
              m_texImageFmts[i][Image::IMG_UNSIGNED_SHORT].fixedPtFmt;
            m_texImageFmts[i][Image::IMG_INT].fixedPtFmt = 
              m_texImageFmts[i][Image::IMG_UNSIGNED_INT].fixedPtFmt;
          }
        }

        if ( !GLEW_EXT_texture_sRGB )
        {
          // get rid of all of the srgb formats
          for( size_t i = 0; i < Image::IMG_NUM_FORMATS; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j ++ )
            {
              m_texImageFmts[i][j].nonLinearFmt = m_texImageFmts[i][j].fixedPtFmt;
            }
          }

          for( size_t i = Image::IMG_COMPRESSED_SRGB_DXT1; i <= Image::IMG_COMPRESSED_SRGBA_DXT5; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = NVSG_TIF_INVALID;
            }
          }
        }
        else
        {
          // set up these formats
          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_SRGB_DXT1][j].fixedPtFmt    = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGB_DXT1][j].floatFmt      = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGB_DXT1][j].compressedFmt = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGB_DXT1][j].integerFmt    = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGB_DXT1][j].nonLinearFmt  = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_SRGB_DXT1][j].usrFmt   = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_SRGB_DXT1][j].type          = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGB_DXT1][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT1][j].fixedPtFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT1][j].floatFmt      = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT1][j].compressedFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT1][j].integerFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT1][j].nonLinearFmt  = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT1][j].usrFmt   = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT1][j].type          = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT1][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT3][j].fixedPtFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT3][j].floatFmt      = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT3][j].compressedFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT3][j].integerFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT3][j].nonLinearFmt  = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT3][j].usrFmt   = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT3][j].type          = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT3][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT5][j].fixedPtFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT5][j].floatFmt      = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT5][j].compressedFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT5][j].integerFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT5][j].nonLinearFmt  = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT5][j].usrFmt   = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT5][j].type          = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SRGBA_DXT5][j].uploadHint    = 0;
          }
        }

        if( !GLEW_EXT_texture_compression_latc )
        {
          //
          // If latc compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = Image::IMG_COMPRESSED_LUMINANCE_LATC1; i <= Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = NVSG_TIF_INVALID;
            }
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j ++ )
          {
            m_texImageFmts[Image::IMG_LUMINANCE][j].compressedFmt = m_texImageFmts[Image::IMG_LUMINANCE][j].fixedPtFmt;
            m_texImageFmts[Image::IMG_LUMINANCE_ALPHA][j].compressedFmt = 
              m_texImageFmts[Image::IMG_LUMINANCE_ALPHA][j].fixedPtFmt;
          }
        }
        else
        {
          // set up these formats
          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_LATC1][j].fixedPtFmt    = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_LATC1][j].floatFmt      = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_LATC1][j].compressedFmt = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_LATC1][j].integerFmt    = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_LATC1][j].nonLinearFmt  = GL_COMPRESSED_LUMINANCE_LATC1_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_LATC1][j].usrFmt   = GL_COMPRESSED_LUMINANCE_LATC1_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_LATC1][j].type          = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_LATC1][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1][j].fixedPtFmt    = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1][j].floatFmt      = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1][j].compressedFmt = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1][j].integerFmt    = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1][j].nonLinearFmt  = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1][j].usrFmt   = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1][j].type          = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2][j].fixedPtFmt    = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2][j].floatFmt      = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2][j].compressedFmt = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2][j].integerFmt    = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2][j].nonLinearFmt  = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2][j].usrFmt   = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2][j].type          = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2][j].fixedPtFmt    = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2][j].floatFmt      = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2][j].compressedFmt = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2][j].integerFmt    = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2][j].nonLinearFmt  = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2][j].usrFmt   = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2][j].type          = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2][j].uploadHint    = 0;
          }
        }

        if( !GLEW_EXT_texture_compression_rgtc )
        {
          //
          // If rgtc compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = Image::IMG_COMPRESSED_RED_RGTC1; i <= Image::IMG_COMPRESSED_SIGNED_RG_RGTC2; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = NVSG_TIF_INVALID;
            }
          }
        }
        else
        {
          // set up these formats
          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_RED_RGTC1][j].fixedPtFmt    = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RED_RGTC1][j].floatFmt      = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RED_RGTC1][j].compressedFmt = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RED_RGTC1][j].integerFmt    = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RED_RGTC1][j].nonLinearFmt  = GL_COMPRESSED_RED_RGTC1_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_RED_RGTC1][j].usrFmt   = GL_COMPRESSED_RED_RGTC1_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_RED_RGTC1][j].type          = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RED_RGTC1][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RED_RGTC1][j].fixedPtFmt    = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RED_RGTC1][j].floatFmt      = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RED_RGTC1][j].compressedFmt = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RED_RGTC1][j].integerFmt    = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RED_RGTC1][j].nonLinearFmt  = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RED_RGTC1][j].usrFmt   = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RED_RGTC1][j].type          = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RED_RGTC1][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_RG_RGTC2][j].fixedPtFmt    = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RG_RGTC2][j].floatFmt      = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RG_RGTC2][j].compressedFmt = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RG_RGTC2][j].integerFmt    = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RG_RGTC2][j].nonLinearFmt  = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_RG_RGTC2][j].usrFmt   = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_RG_RGTC2][j].type          = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RG_RGTC2][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RG_RGTC2][j].fixedPtFmt    = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RG_RGTC2][j].floatFmt      = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RG_RGTC2][j].compressedFmt = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RG_RGTC2][j].integerFmt    = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RG_RGTC2][j].nonLinearFmt  = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RG_RGTC2][j].usrFmt   = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RG_RGTC2][j].type          = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_SIGNED_RG_RGTC2][j].uploadHint    = 0;
          }
        }

        if( !GLEW_ARB_depth_texture )
        {
          //
          // If rgtc compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = 0; i < Image::IMG_NUM_FORMATS; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = NVSG_TIF_INVALID;
            }
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_DEPTH_COMPONENT][j].fixedPtFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[Image::IMG_DEPTH_COMPONENT][j].floatFmt      = NVSG_TIF_INVALID;
            m_texImageFmts[Image::IMG_DEPTH_COMPONENT][j].compressedFmt = NVSG_TIF_INVALID;
            m_texImageFmts[Image::IMG_DEPTH_COMPONENT][j].integerFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[Image::IMG_DEPTH_COMPONENT][j].nonLinearFmt  = NVSG_TIF_INVALID;
          }
        }

        if( !GLEW_EXT_packed_depth_stencil )
        {
          //
          // If rgtc compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = 0; i < Image::IMG_NUM_FORMATS; i ++ )
          {
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_24_8].fixedPtFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_24_8].floatFmt      = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_24_8].compressedFmt = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_24_8].integerFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_24_8].nonLinearFmt  = NVSG_TIF_INVALID;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_DEPTH_STENCIL][j].fixedPtFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[Image::IMG_DEPTH_STENCIL][j].floatFmt      = NVSG_TIF_INVALID;
            m_texImageFmts[Image::IMG_DEPTH_STENCIL][j].compressedFmt = NVSG_TIF_INVALID;
            m_texImageFmts[Image::IMG_DEPTH_STENCIL][j].integerFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[Image::IMG_DEPTH_STENCIL][j].nonLinearFmt  = NVSG_TIF_INVALID;
          }
        }

        if( !GLEW_EXT_texture_shared_exponent )
        {
          //
          // If shared_exponent is not available, then disallow IMG_UNSIGNED_INT_10F_11F_11F
          //
          for( size_t i = 0; i < Image::IMG_NUM_FORMATS; i ++ )
          {
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_10F_11F_11F].fixedPtFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_10F_11F_11F].floatFmt      = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_10F_11F_11F].compressedFmt = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_10F_11F_11F].integerFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_10F_11F_11F].nonLinearFmt  = NVSG_TIF_INVALID;
          }
        }

        if( !GLEW_EXT_packed_float )
        {
          //
          // If packed_float is not available, then disallow IMG_UNSIGNED_INT_5_9_9_9
          //
          for( size_t i = 0; i < Image::IMG_NUM_FORMATS; i ++ )
          {
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_5_9_9_9].fixedPtFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_5_9_9_9].floatFmt      = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_5_9_9_9].compressedFmt = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_5_9_9_9].integerFmt    = NVSG_TIF_INVALID;
            m_texImageFmts[i][Image::IMG_UNSIGNED_INT_5_9_9_9].nonLinearFmt  = NVSG_TIF_INVALID;
          }
        }

        if( !GLEW_ARB_texture_compression )
        {
          //
          // If texture compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = 0; i < Image::IMG_NUM_FORMATS; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j ++ )
            {
              m_texImageFmts[i][j].compressedFmt = m_texImageFmts[i][j].fixedPtFmt;
            }
          }
        }

        if( !GLEW_EXT_texture_compression_s3tc )
        {
          for( size_t i = Image::IMG_COMPRESSED_RGB_DXT1; i <= Image::IMG_COMPRESSED_SRGBA_DXT5; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = NVSG_TIF_INVALID;
            }
          }
        }
        else
        {
          // set up these formats
          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_RGB_DXT1][j].fixedPtFmt    = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGB_DXT1][j].floatFmt      = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGB_DXT1][j].compressedFmt = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGB_DXT1][j].integerFmt    = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGB_DXT1][j].nonLinearFmt  = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_RGB_DXT1][j].usrFmt   = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_RGB_DXT1][j].type          = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGB_DXT1][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT1][j].fixedPtFmt    = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT1][j].floatFmt      = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT1][j].compressedFmt = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT1][j].integerFmt    = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT1][j].nonLinearFmt  = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT1][j].usrFmt   = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT1][j].type          = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT1][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT3][j].fixedPtFmt    = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT3][j].floatFmt      = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT3][j].compressedFmt = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT3][j].integerFmt    = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT3][j].nonLinearFmt  = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT3][j].usrFmt   = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT3][j].type          = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT3][j].uploadHint    = 0;
          }

          for( size_t j = 0; j < Image::IMG_NUM_TYPES; j++ )
          {
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT5][j].fixedPtFmt    = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT5][j].floatFmt      = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT5][j].compressedFmt = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT5][j].integerFmt    = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT5][j].nonLinearFmt  = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;

            // these are not used
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT5][j].usrFmt   = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;

            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT5][j].type          = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[Image::IMG_COMPRESSED_RGBA_DXT5][j].uploadHint    = 0;
          }
        }

        if( !GLEW_EXT_texture_integer )
        {
          //
          // If texture integer is not available, then set the integer formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = 0; i < Image::IMG_NUM_FORMATS; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j ++ )
            {
              m_texImageFmts[i][j].integerFmt    = m_texImageFmts[i][j].fixedPtFmt;
            }
          }

          for( size_t i = Image::IMG_INTEGER_ALPHA; i <= Image::IMG_INTEGER_BGRA; i ++ )
          {
            for( size_t j = 0; j < Image::IMG_NUM_TYPES; j ++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = NVSG_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = NVSG_TIF_INVALID;
            }
          }
        }

        if (  !GLEW_ARB_vertex_buffer_object 
          || !GLEW_ARB_pixel_buffer_object )
        {
          m_texImageFmts[Image::IMG_BGRA][Image::IMG_BYTE].uploadHint = 0; 
          m_texImageFmts[Image::IMG_BGRA][Image::IMG_UNSIGNED_BYTE].uploadHint = 0; 
          m_texImageFmts[Image::IMG_RGBA][Image::IMG_HALF].uploadHint = 0;
          m_texImageFmts[Image::IMG_RGBA][Image::IMG_FLOAT].uploadHint = 0;
        }
      }

      TexImageFmts texImageFmts;

      //
      // Checks to see if this is an integer format.
      //
      static
        bool isInteger( GLint usrFmt )
      {
        if( usrFmt >= Image::IMG_INTEGER_ALPHA && 
          usrFmt <= Image::IMG_INTEGER_BGRA )
        {
          return true;
        }
        else
        {
          return false;
        }
      }

      //
      // Takes the given user format and tries to find an integer format that matches
      //
      GLint mapToInteger( GLint usrFmt )
      {
        switch( usrFmt )
        {
        case GL_RGB:              //!< RGB format  
        case GL_RGB_INTEGER_EXT:
          return GL_RGB_INTEGER_EXT;

        case GL_RGBA:            //!< RGBA format  
        case GL_RGBA_INTEGER_EXT:
          return GL_RGBA_INTEGER_EXT;

        case GL_BGR:              //!< BGR format  
        case GL_BGR_INTEGER_EXT:
          return GL_BGR_INTEGER_EXT;

        case GL_BGRA:            //!< BGRA format  
        case GL_BGRA_INTEGER_EXT:
          return GL_BGRA_INTEGER_EXT;

        case GL_LUMINANCE:        //!< luminance format   
        case GL_LUMINANCE_INTEGER_EXT:
          return GL_LUMINANCE_INTEGER_EXT;

        case GL_LUMINANCE_ALPHA:  //!< luminance alpha format  
        case GL_LUMINANCE_ALPHA_INTEGER_EXT:
          return GL_LUMINANCE_ALPHA_INTEGER_EXT;

        case GL_ALPHA:            //!< Alpha only format
        case GL_ALPHA_INTEGER_EXT:
          return GL_ALPHA_INTEGER_EXT;

        default:
          // punt?
          return GL_RGB_INTEGER_EXT;
        }
      }

      //
      // Looks up the internal and user formats based on the gpu format
      // Simply returns the user format if it is not an integer format.
      //
      bool getTextureGPUFormatValues( TextureHost::TextureGPUFormat gpufmt, GLint & gpuFmt, GLint & usrFmt )
      {
        switch( gpufmt )
        {
        case TextureHost::TGF_ALPHA8:
          gpuFmt = GL_ALPHA8;
          return true;

        case TextureHost::TGF_ALPHA16:
          gpuFmt = GL_ALPHA16;
          return true;

        case TextureHost::TGF_SIGNED_ALPHA8:
          gpuFmt = GL_SIGNED_ALPHA8_NV;
          return true;

        case TextureHost::TGF_LUMINANCE8:
          gpuFmt = GL_LUMINANCE8;
          return true;

        case TextureHost::TGF_LUMINANCE16:
          gpuFmt = GL_LUMINANCE16;
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE8:
          gpuFmt = GL_SIGNED_LUMINANCE8_NV;
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE16:
          gpuFmt = GL_SIGNED_LUMINANCE8_NV; // not supported
          return true;

        case TextureHost::TGF_LUMINANCE8_ALPHA8:
          gpuFmt = GL_LUMINANCE8_ALPHA8;
          return true;

        case TextureHost::TGF_LUMINANCE16_ALPHA16:
          gpuFmt = GL_LUMINANCE16_ALPHA16;
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE8_ALPHA8:
          gpuFmt = GL_SIGNED_LUMINANCE8_ALPHA8_NV;
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE16_ALPHA16:
          gpuFmt = GL_SIGNED_LUMINANCE8_ALPHA8_NV;
          return true;

        case TextureHost::TGF_COMPRESSED_LUMINANCE:
          gpuFmt = GL_COMPRESSED_LUMINANCE;
          return true;

        case TextureHost::TGF_COMPRESSED_LUMINANCE_ALPHA:
          gpuFmt = GL_COMPRESSED_LUMINANCE_ALPHA;
          return true;

        case TextureHost::TGF_COMPRESSED_LUMINANCE_LATC:         // ext_texture_compression_latc
          gpuFmt = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
          return true;

        case TextureHost::TGF_SIGNED_COMPRESSED_LUMINANCE_LATC:
          gpuFmt = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_LUMINANCE_ALPHA_LATC:
          gpuFmt = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
          return true;

        case TextureHost::TGF_SIGNED_COMPRESSED_LUMINANCE_ALPHA_LATC:
          gpuFmt = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
          return true;

        case TextureHost::TGF_RGB8:
          gpuFmt = GL_RGB8;
          return true;

        case TextureHost::TGF_SIGNED_RGB8:
          gpuFmt = GL_SIGNED_RGB8_NV;
          return true;

        case TextureHost::TGF_RGB16:       // native on G80
          gpuFmt = GL_RGB16;
          return true;

        case TextureHost::TGF_RGBA8:
          gpuFmt = GL_RGBA8;
          return true;

        case TextureHost::TGF_SIGNED_RGBA8:
          gpuFmt = GL_SIGNED_RGBA_NV;
          return true;

        case TextureHost::TGF_RGB10:       // native on G80
          gpuFmt = GL_RGB10;
          return true;

        case TextureHost::TGF_RGB10_A2:    // native on G80
          gpuFmt = GL_RGB10_A2;
          return true;

        case TextureHost::TGF_RGBA16:      // native on G80
          gpuFmt = GL_RGB16;
          return true;

        case TextureHost::TGF_SRGB:
          gpuFmt = GL_SRGB_EXT;
          return true;

        case TextureHost::TGF_SRGBA:
          gpuFmt = GL_SRGB_ALPHA_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_SRGB:        // ext_texture_srgb
          gpuFmt = GL_COMPRESSED_SRGB_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_SRGBA:
          gpuFmt = GL_COMPRESSED_SRGB_ALPHA_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_SRGB_DXT1:
          gpuFmt = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_SRGB_DXT3:
          gpuFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_SRGB_DXT5:
          gpuFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_SRGBA_DXT1:
          gpuFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_RGB:
          gpuFmt = GL_COMPRESSED_RGB;
          return true;

        case TextureHost::TGF_COMPRESSED_RGBA:
          gpuFmt = GL_COMPRESSED_RGBA;
          return true;

        case TextureHost::TGF_COMPRESSED_RGB_DXT1:
          gpuFmt = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_RGB_DXT3:
          gpuFmt = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_RGB_DXT5:
          gpuFmt = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
          return true;

        case TextureHost::TGF_COMPRESSED_RGBA_DXT1:
          gpuFmt = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
          return true;

        case TextureHost::TGF_DEPTH16:              // arb_shadow
          gpuFmt = GL_DEPTH_COMPONENT16_ARB;
          return true;

        case TextureHost::TGF_DEPTH24:
          gpuFmt = GL_DEPTH_COMPONENT24_ARB;
          return true;

        case TextureHost::TGF_DEPTH32:
          gpuFmt = GL_DEPTH_COMPONENT32_ARB;
          return true;

        case TextureHost::TGF_DEPTH24_STENCIL8:  // ext_packed_depth_stencil
          gpuFmt = GL_DEPTH24_STENCIL8_EXT;
          return true;

        case TextureHost::TGF_LUMINANCE16F:         // arb_texture_float
          gpuFmt = GL_LUMINANCE16F_ARB;
          return true;

        case TextureHost::TGF_LUMINANCE32F:
          gpuFmt = GL_LUMINANCE32F_ARB;
          return true;

        case TextureHost::TGF_LUMINANCE_ALPHA16F:
          gpuFmt = GL_LUMINANCE_ALPHA16F_ARB;
          return true;

        case TextureHost::TGF_LUMINANCE_ALPHA32F:
          gpuFmt = GL_LUMINANCE_ALPHA32F_ARB;
          return true;

        case TextureHost::TGF_RGB16F:
          gpuFmt = GL_RGB16F_ARB;
          return true;

        case TextureHost::TGF_RGB32F:
          gpuFmt = GL_RGB32F_ARB;
          return true;

        case TextureHost::TGF_RGBA16F:
          gpuFmt = GL_RGBA16F_ARB;
          return true;

        case TextureHost::TGF_RGBA32F:
          gpuFmt = GL_RGBA32F_ARB;
          return true;

        case TextureHost::TGF_SIGNED_RGB_INTEGER32: // ext_texture_integer
          gpuFmt = GL_RGB32I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_RGB_INTEGER16:
          gpuFmt = GL_RGB16I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_RGB_INTEGER8:
          gpuFmt = GL_RGB8I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_RGB_INTEGER32:
          gpuFmt = GL_RGB32UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_RGB_INTEGER16:
          gpuFmt = GL_RGB16UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_RGB_INTEGER8:
          gpuFmt = GL_RGB8UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_RGBA_INTEGER32:
          gpuFmt = GL_RGBA32I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_RGBA_INTEGER16:
          gpuFmt = GL_RGBA16I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_RGBA_INTEGER8:
          gpuFmt = GL_RGBA8I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_RGBA_INTEGER32:
          gpuFmt = GL_RGBA32UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_RGBA_INTEGER16:
          gpuFmt = GL_RGBA16UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_RGBA_INTEGER8:
          gpuFmt = GL_RGBA8UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE_INTEGER32:
          gpuFmt = GL_LUMINANCE32I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE_INTEGER16:
          gpuFmt = GL_LUMINANCE16I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE_INTEGER8:
          gpuFmt = GL_LUMINANCE8I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_LUMINANCE_INTEGER32:
          gpuFmt = GL_LUMINANCE32UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_LUMINANCE_INTEGER16:
          gpuFmt = GL_LUMINANCE16UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_LUMINANCE_INTEGER8:
          gpuFmt = GL_LUMINANCE8UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE_ALPHA_INTEGER32:
          gpuFmt = GL_LUMINANCE32I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE_ALPHA_INTEGER16:
          gpuFmt = GL_LUMINANCE16I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_SIGNED_LUMINANCE_ALPHA_INTEGER8:
          gpuFmt = GL_LUMINANCE8I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_LUMINANCE_ALPHA_INTEGER32:
          gpuFmt = GL_LUMINANCE_ALPHA32UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_LUMINANCE_ALPHA_INTEGER16:
          gpuFmt = GL_LUMINANCE_ALPHA16UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_LUMINANCE_ALPHA_INTEGER8:
          gpuFmt = GL_LUMINANCE_ALPHA8UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TGF_UNSIGNED_FLOAT_SHARED_EXPONENT:
          gpuFmt = GL_RGB9_E5_EXT;
          return true;

        case TextureHost::TGF_UNSIGNED_FLOAT_PACKED:
          gpuFmt = GL_R11F_G11F_B10F_EXT;
          return true;

        default:
          return false;
        }
      }
      
      TextureGLSharedPtr TextureGL::create( const dp::gl::SharedTexture& texture )
      {
        return( std::shared_ptr<TextureGL>( new TextureGL( texture ) ) );
      }

      dp::sg::core::HandledObjectSharedPtr TextureGL::clone() const
      {
        return( std::shared_ptr<TextureGL>( new TextureGL( *this ) ) );
      }

      TextureGL::TextureGL( const dp::gl::SharedTexture& texture )
        : m_texture( texture )
      {
        TextureTarget target;
        switch ( texture->getTarget() )
        {
        case GL_TEXTURE_1D:
            target = TT_TEXTURE_1D;
          break;
        case GL_TEXTURE_2D:
          target = TT_TEXTURE_2D;
          break;
        case GL_TEXTURE_3D:
          target = TT_TEXTURE_3D;
          break;
        case GL_TEXTURE_CUBE_MAP:
          target = TT_TEXTURE_CUBE;
          break;
        case GL_TEXTURE_1D_ARRAY:
          target = TT_TEXTURE_1D_ARRAY;
          break;
        case GL_TEXTURE_2D_ARRAY:
          target = TT_TEXTURE_2D_ARRAY;
          break;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
          target = TT_TEXTURE_CUBE_ARRAY;
          break;
        case GL_TEXTURE_BUFFER:
          target = TT_TEXTURE_BUFFER;
          break;
        default:
          target = TT_UNSPECIFIED_TEXTURE_TARGET;
          break;
        };
        setTextureTarget(target);
      }

      const dp::gl::SharedTexture& TextureGL::getTexture() const
      {
        return m_texture;
      }

      bool TextureGL::getTexImageFmt( GLTexImageFmt & tfmt, Image::PixelFormat fmt, Image::PixelDataType type, TextureHost::TextureGPUFormat gpufmt )
      {
        // give a hint while debugging. though, need a run-time check below
        DP_ASSERT(fmt!=Image::IMG_UNKNOWN_FORMAT || type!=Image::IMG_UNKNOWN_TYPE);  

        if( fmt==Image::IMG_UNKNOWN_FORMAT || type==Image::IMG_UNKNOWN_TYPE )  
        {
          return false;
        }

        // lookup the formats
        NVSGTexImageFmt  nvsgFmt    = texImageFmts.getFmt(fmt,type);
        // formats for this particular fmt
        NVSGTexImageFmt* nvsgFmts   = texImageFmts.getFmts(fmt);

        tfmt.type       = nvsgFmt.type;
        tfmt.uploadHint = nvsgFmt.uploadHint;

        //
        // If gpufmt == TGF_DEFAULT, and we have a floating point format texture, then the assumption is that
        // the texture will be loaded with a floating point format to match the way NVSG used to work.  So, unless
        // the user has selected a specific generic format, examine the PixelData and if it is float, reset the
        // GPUformat to be float as well. 
        //
        if( gpufmt == TextureHost::TGF_DEFAULT )
        {
          // set it up this way in case there are other formats we need to put in code for.
          switch( type )
          {
            // reset to float if they provided float data
          case Image::IMG_FLOAT32:
          case Image::IMG_FLOAT16:
            gpufmt = TextureHost::TGF_FLOAT;
            break;

          default:
            // no changes
            break;
          }
        }

        switch( gpufmt )
        {
          // default "defaults" to FIXED format
        case TextureHost::TGF_DEFAULT:
        case TextureHost::TGF_FIXED:
          tfmt.intFmt     = nvsgFmt.fixedPtFmt;
          tfmt.usrFmt     = nvsgFmt.usrFmt;
          break;

        case TextureHost::TGF_COMPRESSED_FIXED:
          tfmt.intFmt     = nvsgFmt.compressedFmt;
          tfmt.usrFmt     = nvsgFmt.usrFmt;
          break;

        case TextureHost::TGF_FLOAT:
          tfmt.intFmt     = nvsgFmt.floatFmt;
          tfmt.usrFmt     = nvsgFmt.usrFmt;
          break;

        case TextureHost::TGF_NONLINEAR:
          tfmt.intFmt     = nvsgFmt.nonLinearFmt;
          tfmt.usrFmt     = nvsgFmt.usrFmt;
          break;

        case TextureHost::TGF_INTEGER:
          tfmt.intFmt     = nvsgFmt.integerFmt;
          tfmt.usrFmt     = mapToInteger( nvsgFmt.usrFmt );
          break;

        case TextureHost::TGF_FLOAT16:
          tfmt.intFmt     = nvsgFmts[Image::IMG_FLOAT16].floatFmt;
          tfmt.usrFmt     = nvsgFmts[Image::IMG_FLOAT16].usrFmt;
          break;

        case TextureHost::TGF_FLOAT32:
          tfmt.intFmt     = nvsgFmts[Image::IMG_FLOAT32].floatFmt;
          tfmt.usrFmt     = nvsgFmts[Image::IMG_FLOAT32].usrFmt;
          break;

        case TextureHost::TGF_FIXED8:
          {
            switch( type )
            {
            case Image::IMG_BYTE:
            case Image::IMG_SHORT:
            case Image::IMG_INT:
              tfmt.intFmt     = nvsgFmts[Image::IMG_BYTE].fixedPtFmt;
              tfmt.usrFmt     = nvsgFmts[Image::IMG_BYTE].usrFmt;
              break;

            case Image::IMG_UNSIGNED_BYTE:
            case Image::IMG_UNSIGNED_SHORT:
            case Image::IMG_UNSIGNED_INT:
            default:
              tfmt.intFmt     = nvsgFmts[Image::IMG_UNSIGNED_BYTE].fixedPtFmt;
              tfmt.usrFmt     = nvsgFmts[Image::IMG_UNSIGNED_BYTE].usrFmt;
              break;
            }
            break;
          }

        case TextureHost::TGF_FIXED10:
          // only 1 format supported
          tfmt.intFmt     = nvsgFmts[Image::IMG_UNSIGNED_INT_2_10_10_10].fixedPtFmt;
          tfmt.usrFmt     = nvsgFmts[Image::IMG_UNSIGNED_INT_2_10_10_10].usrFmt;
          break;

        case TextureHost::TGF_FIXED16:
          {
            switch( type )
            {
            case Image::IMG_BYTE:
            case Image::IMG_SHORT:
            case Image::IMG_INT:
              tfmt.intFmt     = nvsgFmts[Image::IMG_SHORT].fixedPtFmt;
              tfmt.usrFmt     = nvsgFmts[Image::IMG_SHORT].usrFmt;
              break;

            case Image::IMG_UNSIGNED_BYTE:
            case Image::IMG_UNSIGNED_SHORT:
            case Image::IMG_UNSIGNED_INT:
            default:
              tfmt.intFmt     = nvsgFmts[Image::IMG_UNSIGNED_SHORT].fixedPtFmt;
              tfmt.usrFmt     = nvsgFmts[Image::IMG_UNSIGNED_SHORT].usrFmt;
              break;
            }
            break;
          }

        default:
          // set this so the routine knows the number of components
          tfmt.intFmt     = nvsgFmt.fixedPtFmt;
          tfmt.usrFmt     = nvsgFmt.usrFmt;

          return getTextureGPUFormatValues( gpufmt, tfmt.intFmt, tfmt.usrFmt );
        }

        return ( tfmt.intFmt != NVSG_TIF_INVALID );
      }

    } // namespace gl
  } // namespace sg
} // namespace dp

