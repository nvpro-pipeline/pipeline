// Copyright NVIDIA Corporation 2015
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
      dp::sg::gl::TexImageFmt recentTexImageFmts[Image::PixelFormat::NUM_FORMATS][Image::PixelDataType::NUM_TYPES] =
      {
        { // PixelFormat::COLOR_INDEX - unsupported!! 
          { 0,0,0, 0,0, 0, 0,0 }   // PixelDataType::BYTE
          ,  { 0,0,0, 0,0, 0, 0,0 }   // PixelDataType::UNSIGNED_BYTE
          ,  { 0,0,0, 0,0, 0, 0,0 }   // PixelDataType::SHORT
          ,  { 0,0,0, 0,0, 0, 0,0 }   // PixelDataType::UNSIGNED_SHORT
          ,  { 0,0,0, 0,0, 0, 0,0 }   // PixelDataType::INT
          ,  { 0,0,0, 0,0, 0, 0,0 }   // PixelDataType::UNSIGNED_INT
          ,  { 0,0,0, 0,0, 0, 0,0 }   // PixelDataType::FLOAT32
          ,  { 0,0,0, 0,0, 0, 0,0 }   // PixelDataType::FLOAT64

          // custom formats
          ,  { 0,0,0,0,0,0,0,0 } //PixelDataType::UNSIGNED_INT_2_10_10_10 //!< 2 Bits A, 10 bits RGB (RGB10/10A2)
          ,  { 0,0,0,0,0,0,0,0 } //PixelDataType::UNSIGNED_INT_5_9_9_9    //!< 5 bits exponent 9 bits RGB mantissa
          //   (EXT_texture_shared_exponent) format
          ,  { 0,0,0,0,0,0,0,0 } //PixelDataType::UNSIGNED_INT_10F_11F_11F //!< 5 bits exp + 5,6,6 bits RGB mantissa
          ,  { 0,0,0,0,0,0,0,0 } //PixelDataType::UNSIGNED_INT_24_8 //!< 24 depth, 8 stencil
          //   (EXT_packed_depth_stencil) format
        }
        //
        // Formats that do not support the various combinations of these simply list the same enum for each
        // unsupported format.  There do not appear to be any support for compressed, signed, (NV?) formats
        // for the typical 1,2,3,4 component textures, other than LATC.
        //
        , { // PixelFormat::RGB !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmt            type                upload

          { GL_SIGNED_RGB8_NV, GL_RGB16F_ARB,     GL_RGB8I_EXT,        GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT, 
            GL_RGB,            GL_BYTE,            0 }  // PixelDataType::BYTE - NOTE: requires NV_texture_shader

          ,  { GL_RGB8,           GL_RGB16F_ARB,     GL_RGB8UI_EXT,        GL_COMPRESSED_RGB,  GL_SRGB8_EXT,
            GL_RGB,            GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_SIGNED_RGB8_NV, GL_RGB16F_ARB,     GL_RGB16I_EXT,       GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT,
            GL_RGB,            GL_SHORT,           0 }  // PixelDataType::SHORT - NOTE: requires NV_texture_shader

          ,  { GL_RGB16,          GL_RGB16F_ARB,     GL_RGB16UI_EXT,      GL_COMPRESSED_RGB,  GL_SRGB8_EXT, 
            GL_RGB,            GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_RGB8_NV, GL_RGB32F_ARB,     GL_RGB32I_EXT,       GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT,  
            GL_RGB,            GL_INT,             0 }  // PixelDataType::INT - NOTE: requires NV_texture_shader

          ,  { GL_RGB16,          GL_RGB32F_ARB,     GL_RGB32UI_EXT,      GL_COMPRESSED_RGB,  GL_SRGB8_EXT, 
            GL_RGB,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_RGB16,          GL_RGB32F_ARB,     GL_RGB32I_EXT,       GL_RGB32F_ARB,      GL_SRGB8_EXT,  
            GL_RGB,            GL_FLOAT,           0 }  // PixelDataType::FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_RGB16,          GL_RGB16F_ARB,     GL_RGB32I_EXT,       GL_RGB16F_ARB,      GL_SRGB8_EXT,
            GL_RGB,            GL_HALF_FLOAT_NV,   0 }  // PixelDataType::HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          //
          // requires 4-component format
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,   DP_TIF_INVALID,   DP_TIF_INVALID,    DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0}  // PixelDataType::UNSIGNED_INT_2_10_10_10  

          // supported if EXT_texture_shared_exponent exported                                                                                                    
          ,  { GL_RGB9_E5_EXT,    GL_RGB9_E5_EXT,    GL_RGB9_E5_EXT,      GL_RGB9_E5_EXT,     GL_RGB9_E5_EXT,
            GL_RGB,            GL_UNSIGNED_INT_5_9_9_9_REV_EXT,   0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // supported if EXT_packed_float exported                                                                                                    
          ,  { GL_R11F_G11F_B10F_EXT,  GL_R11F_G11F_B10F_EXT,   GL_R11F_G11F_B10F_EXT,   GL_R11F_G11F_B10F_EXT,    GL_R11F_G11F_B10F_EXT,
            GL_RGB,            GL_UNSIGNED_INT_10F_11F_11F_REV_EXT, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,   DP_TIF_INVALID,   DP_TIF_INVALID,    DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8

        }
        , { // PixelFormat::RGBA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          { GL_SIGNED_RGBA8_NV, GL_RGBA16F_ARB,   GL_RGBA8I_EXT,        GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT, 
            GL_RGBA,            GL_BYTE,            0 }  // PixelDataType::BYTE - NOTE: requires NV_texture_shader

          ,  { GL_RGBA8,           GL_RGBA16F_ARB,   GL_RGBA8UI_EXT,       GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT,
            GL_RGBA,            GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_SIGNED_RGBA8_NV, GL_RGBA16F_ARB,   GL_RGBA16I_EXT,       GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT,
            GL_RGBA,            GL_SHORT,           0 }  // PixelDataType::SHORT - NOTE: requires NV_texture_shader

          ,  { GL_RGBA16,          GL_RGBA16F_ARB,   GL_RGBA16UI_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT, 
            GL_RGBA,            GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_RGBA8_NV, GL_RGBA32F_ARB,   GL_RGBA32I_EXT,       GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT,  
            GL_RGBA,            GL_INT,             0 }  // PixelDataType::INT - NOTE: requires NV_texture_shader

          ,  { GL_RGBA16,          GL_RGBA32F_ARB,   GL_RGBA32UI_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT, 
            GL_RGBA,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT - NOTE: precision loss!

          // fast upload format
          ,  { GL_RGBA16,          GL_RGBA32F_ARB,   GL_RGBA32I_EXT,       GL_RGBA32F_ARB,      GL_SRGB8_ALPHA8_EXT,  
            GL_RGBA,            GL_FLOAT,           1 }  // PixelDataType::FLOAT - NOTE: requires ARB_texture_float

          // fast upload format
          ,  { GL_RGBA16,          GL_RGBA16F_ARB,   GL_RGBA32I_EXT,       GL_RGBA16F_ARB,      GL_SRGB8_ALPHA8_EXT,
            GL_RGBA,            GL_HALF_FLOAT_NV,   1 }  // PixelDataType::HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 

          // G80 native    
          ,  { GL_RGB10_A2,       GL_RGBA16F_ARB,    GL_RGBA16I_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT,
            GL_RGBA,           GL_UNSIGNED_INT_2_10_10_10_REV,   0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10  

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,    DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9
          // EXT_texture_shared_exponent
          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,    DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,    DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0  } // PixelDataType::UNSIGNED_INT_24_8

        }
        , { // PixelFormat::BGR !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          { GL_SIGNED_RGB8_NV, GL_RGB16F_ARB,     GL_RGB8I_EXT,        GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT, 
            GL_BGR,            GL_BYTE,            0 }  // PixelDataType::BYTE - NOTE: requires NV_texture_shader

          ,  { GL_RGB8,           GL_RGB16F_ARB,     GL_RGB8UI_EXT,       GL_COMPRESSED_RGB,  GL_SRGB8_EXT,
            GL_BGR,            GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_SIGNED_RGB8_NV, GL_RGB16F_ARB,     GL_RGB16I_EXT,       GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT,
            GL_BGR,            GL_SHORT,           0 }  // PixelDataType::SHORT - NOTE: requires NV_texture_shader

          ,  { GL_RGB16,          GL_RGB16F_ARB,     GL_RGB16UI_EXT,      GL_COMPRESSED_RGB,  GL_SRGB8_EXT, 
            GL_BGR,            GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_RGB8_NV, GL_RGB32F_ARB,     GL_RGB32I_EXT,       GL_SIGNED_RGB8_NV,  GL_SRGB8_EXT,  
            GL_BGR,            GL_INT,             0 }  // PixelDataType::INT - NOTE: requires NV_texture_shader

          ,  { GL_RGB16,          GL_RGB32F_ARB,     GL_RGB32UI_EXT,      GL_COMPRESSED_RGB,  GL_SRGB8_EXT, 
            GL_BGR,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_RGB16,          GL_RGB32F_ARB,     GL_RGB32I_EXT,       GL_RGB32F_ARB,      GL_SRGB8_EXT,  
            GL_BGR,            GL_FLOAT,           0 }  // PixelDataType::FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_RGB16,          GL_RGB16F_ARB,     GL_RGB32I_EXT,       GL_RGB16F_ARB,      GL_SRGB8_EXT,
            GL_BGR,            GL_HALF_FLOAT_NV,   0 }  // PixelDataType::HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // meaningless because RGB layout doesn't map to 32-bit input data. would crash for 8-bit per channel user data

          // requires 4-component format
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0}  // PixelDataType::UNSIGNED_INT_2_10_10_10  
          // G80 native
          // requires RGB                                                                                                     
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB                                                                                                     
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,    DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0  }  // PixelDataType::UNSIGNED_INT_24_8

        }
        , { // PixelFormat::BGRA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          // fast upload format
          { GL_SIGNED_RGBA8_NV, GL_RGBA16F_ARB,   GL_RGBA8I_EXT,        GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT, 
            GL_BGRA,            GL_BYTE,            1 }  // PixelDataType::BYTE - NOTE: requires NV_texture_shader

          // fast upload format
          ,  { GL_RGBA8,           GL_RGBA16F_ARB,   GL_RGBA8UI_EXT,       GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT,
            GL_BGRA,            GL_UNSIGNED_BYTE,   1 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_SIGNED_RGBA8_NV, GL_RGBA16F_ARB,   GL_RGBA16I_EXT,       GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT,
            GL_BGRA,            GL_SHORT,           0 }  // PixelDataType::SHORT - NOTE: requires NV_texture_shader

          ,  { GL_RGBA16,          GL_RGBA16F_ARB,   GL_RGBA16UI_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT, 
            GL_BGRA,            GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_RGBA8_NV, GL_RGBA32F_ARB,   GL_RGBA32I_EXT,       GL_SIGNED_RGBA8_NV,  GL_SRGB8_ALPHA8_EXT,  
            GL_BGRA,            GL_INT,             0 }  // PixelDataType::INT - NOTE: requires NV_texture_shader

          ,  { GL_RGBA16,          GL_RGBA32F_ARB,   GL_RGBA32UI_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT, 
            GL_BGRA,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_RGBA16,          GL_RGBA32F_ARB,   GL_RGBA32I_EXT,       GL_RGBA32F_ARB,      GL_SRGB8_ALPHA8_EXT,  
            GL_BGRA,            GL_FLOAT,           0 }  // PixelDataType::FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_RGBA16,          GL_RGBA16F_ARB,   GL_RGBA32I_EXT,       GL_RGBA16F_ARB,      GL_SRGB8_ALPHA8_EXT,
            GL_BGRA,            GL_HALF_FLOAT_NV,   0 }  // PixelDataType::HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // G80 native 
          ,  { GL_RGB10_A2,       GL_RGBA16F_ARB,    GL_RGBA16I_EXT,      GL_COMPRESSED_RGBA,  GL_SRGB8_ALPHA8_EXT,
            GL_BGRA,           GL_UNSIGNED_INT_2_10_10_10_REV,   0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10  

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0  }  // PixelDataType::UNSIGNED_INT_24_8

        }
        , { // PixelFormat::LUMINANCE !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          { GL_SIGNED_LUMINANCE8_NV, GL_LUMINANCE16F_ARB,   GL_LUMINANCE8I_EXT,  GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT, GL_SLUMINANCE8_EXT, 
            GL_LUMINANCE,            GL_BYTE,            0 }  // PixelDataType::BYTE - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE8,           GL_LUMINANCE16F_ARB,   GL_LUMINANCE8UI_EXT,       GL_COMPRESSED_LUMINANCE_LATC1_EXT,  GL_SLUMINANCE8_EXT,
            GL_LUMINANCE,            GL_UNSIGNED_BYTE,   1 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_SIGNED_LUMINANCE8_NV, GL_LUMINANCE16F_ARB,   GL_LUMINANCE16I_EXT,       GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT,  GL_SLUMINANCE8_EXT,
            GL_LUMINANCE,            GL_SHORT,           0 }  // PixelDataType::SHORT - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE16,          GL_LUMINANCE16F_ARB,     GL_LUMINANCE16UI_EXT,    GL_COMPRESSED_LUMINANCE_LATC1_EXT,  GL_SLUMINANCE8_EXT, 
            GL_LUMINANCE,            GL_UNSIGNED_SHORT,  1 }  // PixelDataType::UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_LUMINANCE8_NV, GL_LUMINANCE32F_ARB,     GL_LUMINANCE32I_EXT,     GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT,  
            GL_SLUMINANCE8_EXT,  
            GL_LUMINANCE,            GL_INT,             0 }  // PixelDataType::INT - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE16,          GL_LUMINANCE32F_ARB,     GL_LUMINANCE32UI_EXT,      GL_COMPRESSED_LUMINANCE_LATC1_EXT,  
            GL_SLUMINANCE8_EXT, 
            GL_LUMINANCE,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_LUMINANCE16,          GL_LUMINANCE32F_ARB,     GL_LUMINANCE32I_EXT,       GL_COMPRESSED_LUMINANCE_LATC1_EXT,
            GL_SLUMINANCE8_EXT,  
            GL_LUMINANCE,            GL_FLOAT,           0 }  // PixelDataType::FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_LUMINANCE16,          GL_LUMINANCE16F_ARB,     GL_LUMINANCE32I_EXT,       GL_COMPRESSED_LUMINANCE_LATC1_EXT, 
            GL_SLUMINANCE8_EXT,
            GL_LUMINANCE,            GL_HALF_FLOAT_NV,   0 }  // PixelDataType::HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // requires 4-component format      
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,    DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0  }  // PixelDataType::UNSIGNED_INT_24_8

        }
        , { // PixelFormat::LUMINANCE_ALPHA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       type                upload

          { GL_SIGNED_LUMINANCE8_ALPHA8_NV, GL_LUMINANCE_ALPHA16F_ARB,   GL_LUMINANCE_ALPHA8I_EXT,  
            GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT,  GL_SLUMINANCE8_ALPHA8_EXT, 
            GL_LUMINANCE_ALPHA,            GL_BYTE,            0 }  // PixelDataType::BYTE - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE8_ALPHA8,           GL_LUMINANCE_ALPHA16F_ARB,   GL_LUMINANCE_ALPHA8UI_EXT,       GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,  
            GL_SLUMINANCE8_ALPHA8_EXT,
            GL_LUMINANCE_ALPHA,            GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_SIGNED_LUMINANCE8_ALPHA8_NV, GL_LUMINANCE_ALPHA16F_ARB,   GL_LUMINANCE_ALPHA16I_EXT,       
            GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT,  GL_SLUMINANCE8_ALPHA8_EXT,
            GL_LUMINANCE_ALPHA,            GL_SHORT,           0 }  // PixelDataType::SHORT - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE16_ALPHA16,          GL_LUMINANCE_ALPHA16F_ARB,     GL_LUMINANCE_ALPHA16UI_EXT,    
            GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,  GL_SLUMINANCE8_ALPHA8_EXT, 
            GL_LUMINANCE_ALPHA,            GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT - NOTE: precision loss!

          ,  { GL_SIGNED_LUMINANCE8_ALPHA8_NV, GL_LUMINANCE_ALPHA32F_ARB,     GL_LUMINANCE_ALPHA32I_EXT,     
            GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT,  
            GL_SLUMINANCE8_ALPHA8_EXT,  
            GL_LUMINANCE_ALPHA,             GL_INT,             0 }  // PixelDataType::INT - NOTE: requires NV_texture_shader

          ,  { GL_LUMINANCE16_ALPHA16,          GL_LUMINANCE_ALPHA32F_ARB,     GL_LUMINANCE_ALPHA32UI_EXT,      
            GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,  
            GL_SLUMINANCE8_ALPHA8_EXT, 
            GL_LUMINANCE_ALPHA,             GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT - NOTE: precision loss!

          ,  { GL_LUMINANCE16_ALPHA16,          GL_LUMINANCE_ALPHA32F_ARB,     GL_LUMINANCE_ALPHA32I_EXT,       
            GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,
            GL_SLUMINANCE8_ALPHA8_EXT,  
            GL_LUMINANCE_ALPHA,             GL_FLOAT,           0 }  // PixelDataType::FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_LUMINANCE16_ALPHA16,          GL_LUMINANCE_ALPHA16F_ARB,     GL_LUMINANCE_ALPHA32I_EXT, GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,
            GL_SLUMINANCE8_ALPHA8_EXT,
            GL_LUMINANCE_ALPHA,             GL_HALF_FLOAT_NV,   0 }  // PixelDataType::HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // requires 4-component format 
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F  

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8

        }
        , { // PixelFormat::ALPHA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_SIGNED_ALPHA8_NV, GL_ALPHA16F_ARB,   GL_ALPHA8I_EXT,  GL_SIGNED_ALPHA8_NV, GL_SIGNED_ALPHA8_NV, 
            GL_ALPHA,            GL_BYTE,            0 }  // PixelDataType::BYTE - NOTE: requires NV_texture_shader

          ,  { GL_ALPHA8,           GL_ALPHA16F_ARB,   GL_ALPHA8UI_EXT,       GL_ALPHA8,  GL_ALPHA8,
            GL_ALPHA,            GL_UNSIGNED_BYTE,   1 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_SIGNED_ALPHA8_NV, GL_ALPHA16F_ARB,   GL_ALPHA16I_EXT,       GL_SIGNED_ALPHA8_NV,  GL_SIGNED_ALPHA8_NV,
            GL_ALPHA,            GL_SHORT,           0 }  // PixelDataType::SHORT - NOTE: requires NV_texture_shader

          ,  { GL_ALPHA16,          GL_ALPHA16F_ARB,     GL_ALPHA16UI_EXT,    GL_ALPHA16,  GL_ALPHA16, 
            GL_ALPHA,            GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT

          ,  { GL_SIGNED_ALPHA8_NV, GL_ALPHA32F_ARB,     GL_ALPHA32I_EXT,     GL_SIGNED_ALPHA8_NV,  GL_SIGNED_ALPHA8_NV,  
            GL_ALPHA,            GL_INT,             0 }  // PixelDataType::INT - NOTE: requires NV_texture_shader

          ,  { GL_ALPHA16,          GL_ALPHA32F_ARB,     GL_ALPHA32UI_EXT,      GL_ALPHA16,  GL_ALPHA16, 
            GL_ALPHA,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT

          ,  { GL_ALPHA16,          GL_ALPHA32F_ARB,     GL_ALPHA32I_EXT,       GL_ALPHA16, GL_ALPHA16,  
            GL_ALPHA,            GL_FLOAT,           0 }  // PixelDataType::FLOAT - NOTE: requires ARB_texture_float

          ,  { GL_ALPHA16,          GL_ALPHA16F_ARB,     GL_ALPHA32I_EXT,       GL_ALPHA16, GL_ALPHA16, 
            GL_ALPHA,            GL_HALF_FLOAT_NV,   0 }  // PixelDataType::HALF - NOTE: requires ARB_texture_float

          //
          // custom formats
          // 
          // 1-component format does not match. requires 4-component format (RGBA|BGRA)
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 } // PixelDataType::UNSIGNED_INT_2_10_10_10 

          // requires RGB format
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB format
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8
        }
        , { // PixelFormat::DEPTH_COMPONENT - Requires ARB_depth_texture

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_BYTE,            0 }  // PixelDataType::BYTE - NOTE: requires ARB_depth_texture

          ,  { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_UNSIGNED_BYTE,            0 }  // PixelDataType::UNSIGNED_BYTE 

          ,  { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_SHORT,            0 }  // PixelDataType::SHORT 

          ,  { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_SHORT,            0 }  // PixelDataType::UNSIGNED_SHORT 

          ,  { GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB,   GL_DEPTH_COMPONENT32_ARB,  GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB, 
            GL_DEPTH_COMPONENT,            GL_INT,            0 }  // PixelDataType::INT 

          ,  { GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB,   GL_DEPTH_COMPONENT32_ARB,  GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB, 
            GL_DEPTH_COMPONENT,            GL_UNSIGNED_INT,            0 }  // PixelDataType::UNSIGNED_INT 

          ,  { GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB,   GL_DEPTH_COMPONENT32_ARB,  GL_DEPTH_COMPONENT32_ARB, GL_DEPTH_COMPONENT32_ARB, 
            GL_DEPTH_COMPONENT,            GL_FLOAT,            0 }  // PixelDataType::FLOAT 

          ,  { GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB,   GL_DEPTH_COMPONENT16_ARB,  GL_DEPTH_COMPONENT16_ARB, GL_DEPTH_COMPONENT16_ARB, 
            GL_DEPTH_COMPONENT,            GL_HALF_FLOAT_NV,       0 }  // PixelDataType::HALF 

          //
          // custom formats
          // 
          // requires 4-component format (RGBA|BGRA)
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 } // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB format
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB format
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 } // PixelDataType::UNSIGNED_INT_10F_11F_11F 

          // load only depth pixels out of the depth stencil - supported if EXT_packed_depth_stencil exported
          ,  { GL_DEPTH_COMPONENT24_ARB,  GL_DEPTH_COMPONENT24_ARB,  GL_DEPTH_COMPONENT24_ARB,  GL_DEPTH_COMPONENT24_ARB,  GL_DEPTH_COMPONENT24_ARB,
            GL_DEPTH_STENCIL_EXT,     GL_UNSIGNED_INT_24_8_EXT, 0 } // PixelDataType::UNSIGNED_INT_24_8
        }
        , { // PixelFormat::DEPTH_STENCIL - Requires EXT_packed_depth_stencil

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0}  // PixelDataType::BYTE

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0}  // PixelDataType::UNSIGNED_BYTE 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::SHORT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_SHORT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::INT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::FLOAT 

          // DEPTH_STENCIL requires UNSIGNED_INT_24_8
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::HALF 

          //
          // custom formats
          // 
          // format - type mismatch 
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // format - type mismatch 
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // format - type mismatch 
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // perfect match of format and type - supported if EXT_packed_depth_stencil exported
          // fast upload format
          ,  { GL_DEPTH24_STENCIL8_EXT,  GL_DEPTH24_STENCIL8_EXT,   GL_DEPTH24_STENCIL8_EXT,   GL_DEPTH24_STENCIL8_EXT,    GL_DEPTH24_STENCIL8_EXT,
            GL_DEPTH_STENCIL_EXT,     GL_UNSIGNED_INT_24_8_EXT, 1 }  // PixelDataType::UNSIGNED_INT_24_8
        }
        , { // PixelFormat::INTEGER_ALPHA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_ALPHA8I_EXT, GL_ALPHA8I_EXT,   GL_ALPHA8I_EXT,  GL_ALPHA8I_EXT, GL_ALPHA8I_EXT, 
            GL_ALPHA_INTEGER_EXT,  GL_BYTE,            0 }  // PixelDataType::BYTE

          ,  { GL_ALPHA8UI_EXT,        GL_ALPHA8UI_EXT,   GL_ALPHA8UI_EXT,       GL_ALPHA8UI_EXT,  GL_ALPHA8UI_EXT,
            GL_ALPHA_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_ALPHA16I_EXT, GL_ALPHA16I_EXT,   GL_ALPHA16I_EXT,    GL_ALPHA16I_EXT,  GL_ALPHA16I_EXT,
            GL_ALPHA_INTEGER_EXT,   GL_SHORT,           0 }  // PixelDataType::SHORT

          ,  { GL_ALPHA16UI_EXT,          GL_ALPHA16UI_EXT,     GL_ALPHA16UI_EXT,    GL_ALPHA16UI_EXT,  GL_ALPHA16UI_EXT, 
            GL_ALPHA_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT

          ,  { GL_ALPHA32I_EXT, GL_ALPHA32I_EXT,     GL_ALPHA32I_EXT,     GL_ALPHA32I_EXT,  GL_ALPHA32I_EXT,  
            GL_ALPHA_INTEGER_EXT,            GL_INT,             0 }  // PixelDataType::INT -

          ,  { GL_ALPHA32UI_EXT,          GL_ALPHA32UI_EXT,     GL_ALPHA32UI_EXT,      GL_ALPHA32UI_EXT,  GL_ALPHA32UI_EXT, 
            GL_ALPHA,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::FLOAT - float data with integer fmt = invalid

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 
          // requires 4-component format 
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // IMG_UNSIGNED_INY_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8
        }
        , { // PixelFormat::INTEGER_LUMINANCE !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_LUMINANCE8I_EXT,        GL_LUMINANCE8I_EXT,   GL_LUMINANCE8I_EXT,  GL_LUMINANCE8I_EXT, GL_LUMINANCE8I_EXT, 
            GL_LUMINANCE_INTEGER_EXT,  GL_BYTE,            0 }  // PixelDataType::BYTE

          ,  { GL_LUMINANCE8UI_EXT,       GL_LUMINANCE8UI_EXT,   GL_LUMINANCE8UI_EXT,       GL_LUMINANCE8UI_EXT,  GL_LUMINANCE8UI_EXT,
            GL_LUMINANCE_INTEGER_EXT,  GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_LUMINANCE16I_EXT, GL_LUMINANCE16I_EXT,   GL_LUMINANCE16I_EXT,    GL_LUMINANCE16I_EXT,  GL_LUMINANCE16I_EXT,
            GL_LUMINANCE_INTEGER_EXT,  GL_SHORT,           0 }  // PixelDataType::SHORT

          ,  { GL_LUMINANCE16UI_EXT,      GL_LUMINANCE16UI_EXT,     GL_LUMINANCE16UI_EXT,    GL_LUMINANCE16UI_EXT,  GL_LUMINANCE16UI_EXT, 
            GL_LUMINANCE_INTEGER_EXT,  GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT

          ,  { GL_LUMINANCE32I_EXT, GL_LUMINANCE32I_EXT,     GL_LUMINANCE32I_EXT,     GL_LUMINANCE32I_EXT,  GL_LUMINANCE32I_EXT,  
            GL_LUMINANCE_INTEGER_EXT,  GL_INT,             0 }  // PixelDataType::INT -

          ,  { GL_LUMINANCE32UI_EXT,      GL_LUMINANCE32UI_EXT,     GL_LUMINANCE32UI_EXT,      GL_LUMINANCE32UI_EXT,  GL_LUMINANCE32UI_EXT, 
            GL_LUMINANCE,              GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::FLOAT - float data with integer fmt = invalid

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 
          // requires 4-component format (RGBA|BGRA) 
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8
        }
        , { // PixelFormat::INTEGER_LUMINANCE_ALPHA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_LUMINANCE_ALPHA8I_EXT, GL_LUMINANCE_ALPHA8I_EXT,   GL_LUMINANCE_ALPHA8I_EXT,  GL_LUMINANCE_ALPHA8I_EXT, GL_LUMINANCE_ALPHA8I_EXT, 
            GL_LUMINANCE_ALPHA_INTEGER_EXT,  GL_BYTE,            0 }  // PixelDataType::BYTE

          ,  { GL_LUMINANCE_ALPHA8UI_EXT,        GL_LUMINANCE_ALPHA8UI_EXT,   GL_LUMINANCE_ALPHA8UI_EXT,       GL_LUMINANCE_ALPHA8UI_EXT,  GL_LUMINANCE_ALPHA8UI_EXT,
            GL_LUMINANCE_ALPHA_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_LUMINANCE_ALPHA16I_EXT, GL_LUMINANCE_ALPHA16I_EXT,   GL_LUMINANCE_ALPHA16I_EXT,    GL_LUMINANCE_ALPHA16I_EXT,  GL_LUMINANCE_ALPHA16I_EXT,
            GL_LUMINANCE_ALPHA_INTEGER_EXT,   GL_SHORT,           0 }  // PixelDataType::SHORT

          ,  { GL_LUMINANCE_ALPHA16UI_EXT,          GL_LUMINANCE_ALPHA16UI_EXT,     GL_LUMINANCE_ALPHA16UI_EXT,    GL_LUMINANCE_ALPHA16UI_EXT,  GL_LUMINANCE_ALPHA16UI_EXT, 
            GL_LUMINANCE_ALPHA_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT

          ,  { GL_LUMINANCE_ALPHA32I_EXT, GL_LUMINANCE_ALPHA32I_EXT,     GL_LUMINANCE_ALPHA32I_EXT,     GL_LUMINANCE_ALPHA32I_EXT,  GL_LUMINANCE_ALPHA32I_EXT,  
            GL_LUMINANCE_ALPHA_INTEGER_EXT,            GL_INT,             0 }  // PixelDataType::INT -

          ,  { GL_LUMINANCE_ALPHA32UI_EXT,          GL_LUMINANCE_ALPHA32UI_EXT,     GL_LUMINANCE_ALPHA32UI_EXT,      GL_LUMINANCE_ALPHA32UI_EXT,  GL_LUMINANCE_ALPHA32UI_EXT, 
            GL_LUMINANCE_ALPHA,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::FLOAT - float data with integer fmt = invalid

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 

          // requires 4-component format (RGBA|BGRA) 
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8
        }
        , { // PixelFormat::INTEGER_RGB !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_RGB8I_EXT, GL_RGB8I_EXT,   GL_RGB8I_EXT,  GL_RGB8I_EXT, GL_RGB8I_EXT, 
            GL_RGB_INTEGER_EXT,  GL_BYTE,            0 }  // PixelDataType::BYTE

          ,  { GL_RGB8UI_EXT,        GL_RGB8UI_EXT,   GL_RGB8UI_EXT,       GL_RGB8UI_EXT,  GL_RGB8UI_EXT,
            GL_RGB_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_RGB16I_EXT, GL_RGB16I_EXT,   GL_RGB16I_EXT,    GL_RGB16I_EXT,  GL_RGB16I_EXT,
            GL_RGB_INTEGER_EXT,   GL_SHORT,           0 }  // PixelDataType::SHORT

          ,  { GL_RGB16UI_EXT,          GL_RGB16UI_EXT,     GL_RGB16UI_EXT,    GL_RGB16UI_EXT,  GL_RGB16UI_EXT, 
            GL_RGB_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT

          ,  { GL_RGB32I_EXT, GL_RGB32I_EXT,     GL_RGB32I_EXT,     GL_RGB32I_EXT,  GL_RGB32I_EXT,  
            GL_RGB_INTEGER_EXT,            GL_INT,             0 }  // PixelDataType::INT -

          ,  { GL_RGB32UI_EXT,          GL_RGB32UI_EXT,     GL_RGB32UI_EXT,      GL_RGB32UI_EXT,  GL_RGB32UI_EXT, 
            GL_RGB,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::FLOAT - float data with integer fmt = invalid

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 

          // requires 4-component format (RGBA|BGRA) 
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8
        }
        , { // PixelFormat::INTEGER_BGR !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_RGB8I_EXT, GL_RGB8I_EXT,   GL_RGB8I_EXT,  GL_RGB8I_EXT, GL_RGB8I_EXT, 
            GL_BGR_INTEGER_EXT,  GL_BYTE,            0 }  // PixelDataType::BYTE

          ,  { GL_RGB8UI_EXT,        GL_RGB8UI_EXT,   GL_RGB8UI_EXT,       GL_RGB8UI_EXT,  GL_RGB8UI_EXT,
            GL_BGR_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_RGB16I_EXT, GL_RGB16I_EXT,   GL_RGB16I_EXT,    GL_RGB16I_EXT,  GL_RGB16I_EXT,
            GL_BGR_INTEGER_EXT,   GL_SHORT,           0 }  // PixelDataType::SHORT

          ,  { GL_RGB16UI_EXT,          GL_RGB16UI_EXT,     GL_RGB16UI_EXT,    GL_RGB16UI_EXT,  GL_RGB16UI_EXT, 
            GL_BGR_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT

          ,  { GL_RGB32I_EXT, GL_RGB32I_EXT,     GL_RGB32I_EXT,     GL_RGB32I_EXT,  GL_RGB32I_EXT,  
            GL_BGR_INTEGER_EXT,            GL_INT,             0 }  // PixelDataType::INT -

          ,  { GL_RGB32UI_EXT,          GL_RGB32UI_EXT,     GL_RGB32UI_EXT,      GL_RGB32UI_EXT,  GL_RGB32UI_EXT, 
            GL_BGR_INTEGER_EXT,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::FLOAT - float data with integer fmt = invalid

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 

          // requires 4-component format (RGBA|BGRA) 
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8
        }
        , { // PixelFormat::INTEGER_RGBA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_RGBA8I_EXT, GL_RGBA8I_EXT,   GL_RGBA8I_EXT,  GL_RGBA8I_EXT, GL_RGBA8I_EXT, 
            GL_RGBA_INTEGER_EXT,  GL_BYTE,            0 }  // PixelDataType::BYTE

          ,  { GL_RGBA8UI_EXT,        GL_RGBA8UI_EXT,   GL_RGBA8UI_EXT,       GL_RGBA8UI_EXT,  GL_RGBA8UI_EXT,
            GL_RGBA_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_RGBA16I_EXT, GL_RGBA16I_EXT,   GL_RGBA16I_EXT,    GL_RGBA16I_EXT,  GL_RGBA16I_EXT,
            GL_RGBA_INTEGER_EXT,   GL_SHORT,           0 }  // PixelDataType::SHORT

          ,  { GL_RGBA16UI_EXT,          GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,    GL_RGBA16UI_EXT,  GL_RGBA16UI_EXT, 
            GL_RGBA_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT

          ,  { GL_RGBA32I_EXT, GL_RGBA32I_EXT,     GL_RGBA32I_EXT,     GL_RGBA32I_EXT,  GL_RGBA32I_EXT,  
            GL_RGBA_INTEGER_EXT,            GL_INT,             0 }  // PixelDataType::INT -

          ,  { GL_RGBA32UI_EXT,          GL_RGBA32UI_EXT,     GL_RGBA32UI_EXT,      GL_RGBA32UI_EXT,  GL_RGBA32UI_EXT, 
            GL_RGBA_INTEGER_EXT,            GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::FLOAT - float data with integer fmt = invalid

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 

          // not sure if this makes sense, but it should work according to the specs 
          ,  { GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,    GL_RGBA16UI_EXT,  GL_RGBA16UI_EXT,
            GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT_2_10_10_10_REV,   0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8
        }
        , { // PixelFormat::INTEGER_BGRA !! 

          // fixedfmt           floatFmt           integerFmt           compressedFmt       nonlinear  
          // userFmtFixed       userFmtFloat       userFmtInteger       type                upload

          { GL_RGBA8I_EXT, GL_RGBA8I_EXT,   GL_RGBA8I_EXT,  GL_RGBA8I_EXT, GL_RGBA8I_EXT, 
            GL_BGRA_INTEGER_EXT,  GL_BYTE,            0 }  // PixelDataType::BYTE

          ,  { GL_RGBA8UI_EXT,        GL_RGBA8UI_EXT,   GL_RGBA8UI_EXT,       GL_RGBA8UI_EXT,  GL_RGBA8UI_EXT,
            GL_BGRA_INTEGER_EXT,   GL_UNSIGNED_BYTE,   0 }  // PixelDataType::UNSIGNED_BYTE

          ,  { GL_RGBA16I_EXT, GL_RGBA16I_EXT,   GL_RGBA16I_EXT,    GL_RGBA16I_EXT,  GL_RGBA16I_EXT,
            GL_BGRA_INTEGER_EXT,   GL_SHORT,           0 }  // PixelDataType::SHORT

          ,  { GL_RGBA16UI_EXT,          GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,    GL_RGBA16UI_EXT,  GL_RGBA16UI_EXT, 
            GL_BGRA_INTEGER_EXT,      GL_UNSIGNED_SHORT,  0 }  // PixelDataType::UNSIGNED_SHORT

          ,  { GL_RGBA32I_EXT, GL_RGBA32I_EXT,     GL_RGBA32I_EXT,     GL_RGBA32I_EXT,  GL_RGBA32I_EXT,  
            GL_BGRA_INTEGER_EXT, GL_INT,             0 }  // PixelDataType::INT -

          ,  { GL_RGBA32UI_EXT,     GL_RGBA32UI_EXT,     GL_RGBA32UI_EXT,      GL_RGBA32UI_EXT,  GL_RGBA32UI_EXT, 
            GL_BGRA_INTEGER_EXT, GL_UNSIGNED_INT,    0 }  // PixelDataType::UNSIGNED_INT

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::FLOAT - float data with integer fmt = invalid

          ,  { DP_TIF_INVALID,    DP_TIF_INVALID,     DP_TIF_INVALID,       DP_TIF_INVALID, DP_TIF_INVALID,  
            DP_TIF_INVALID,    0,           0 }  // PixelDataType::HALF - float data with integer fmt = invalid

          //
          // custom formats
          // 
          // not sure if this makes sense, but it should work according to the specs 
          ,  { GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,     GL_RGBA16UI_EXT,    GL_RGBA16UI_EXT,  GL_RGBA16UI_EXT,
            GL_BGRA_INTEGER_EXT, GL_UNSIGNED_INT_2_10_10_10_REV,   0 }  // PixelDataType::UNSIGNED_INT_2_10_10_10

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_5_9_9_9

          // requires RGB
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_10F_11F_11F

          // format - type mismatch
          ,  { DP_TIF_INVALID,  DP_TIF_INVALID,  DP_TIF_INVALID,    DP_TIF_INVALID,   DP_TIF_INVALID,
            DP_TIF_INVALID,  0, 0 }  // PixelDataType::UNSIGNED_INT_24_8
        }

        //
        // NOTE - THE REST ARE SET UP PROGRAMMATICALLY BELOW
        //
      };

      class TexImageFmts
      {
      public: 
        TexImageFmts();
        dp::sg::gl::TexImageFmt getFmt( Image::PixelFormat pf, Image::PixelDataType pdt );
        dp::sg::gl::TexImageFmt* getFmts( Image::PixelFormat pf );

      private:
        void initialize();

      private:
        bool initialized;
        dp::sg::gl::TexImageFmt m_texImageFmts[Image::PixelFormat::NUM_FORMATS][Image::PixelDataType::NUM_TYPES];     
      };

      TexImageFmts::TexImageFmts()
        : initialized( false )
      {}

      dp::sg::gl::TexImageFmt TexImageFmts::getFmt( Image::PixelFormat pf, Image::PixelDataType pdt )
      {
        if( !initialized )
        {
          initialize();
          initialized = true;
        }

        return m_texImageFmts[static_cast<size_t>(pf)][static_cast<size_t>(pdt)];
      }

      dp::sg::gl::TexImageFmt* TexImageFmts::getFmts( Image::PixelFormat pf )
      {
        if( !initialized )
        {
          initialize();
          initialized = true;
        }

        return &m_texImageFmts[static_cast<unsigned int>(pf)][0];
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

          for( size_t i = 0; i < static_cast<size_t>(Image::PixelFormat::NUM_FORMATS); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j ++ )
            {
              m_texImageFmts[i][j].floatFmt = m_texImageFmts[i][j].fixedPtFmt;
            }
          }

          // also turn off "fast upload"
          m_texImageFmts[static_cast<unsigned int>(Image::PixelFormat::RGBA)][static_cast<size_t>(Image::PixelDataType::FLOAT16)].uploadHint = 0;
          m_texImageFmts[static_cast<unsigned int>(Image::PixelFormat::RGBA)][static_cast<size_t>(Image::PixelDataType::FLOAT32)].uploadHint = 0;
        }

        if ( !GLEW_NV_texture_shader )
        {
          //
          // Just set the signed formats to be the same as the unsigned.
          //
          for( size_t i = 0; i < static_cast<size_t>(Image::PixelFormat::NUM_FORMATS); i ++ )
          {
            m_texImageFmts[i][static_cast<size_t>(Image::PixelDataType::BYTE)].fixedPtFmt = 
              m_texImageFmts[i][static_cast<size_t>(Image::PixelDataType::UNSIGNED_BYTE)].fixedPtFmt;
            m_texImageFmts[i][static_cast<size_t>(Image::PixelDataType::SHORT)].fixedPtFmt = 
              m_texImageFmts[i][static_cast<size_t>(Image::PixelDataType::UNSIGNED_SHORT)].fixedPtFmt;
            m_texImageFmts[i][static_cast<size_t>(Image::PixelDataType::INT)].fixedPtFmt = 
              m_texImageFmts[i][static_cast<size_t>(Image::PixelDataType::UNSIGNED_INT)].fixedPtFmt;
          }
        }

        if ( !GLEW_EXT_texture_sRGB )
        {
          // get rid of all of the srgb formats
          for( size_t i = 0; i < static_cast<size_t>(Image::PixelFormat::NUM_FORMATS); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j ++ )
            {
              m_texImageFmts[i][j].nonLinearFmt = m_texImageFmts[i][j].fixedPtFmt;
            }
          }

          for( size_t i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_SRGB_DXT1); i <= static_cast<size_t>(Image::PixelFormat::COMPRESSED_SRGBA_DXT5); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
            }
          }
        }
        else
        {
          // set up these formats
          size_t i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_SRGB_DXT1);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_SRGBA_DXT1);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_SRGBA_DXT3);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_SRGBA_DXT5);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }
        }

        if( !GLEW_EXT_texture_compression_latc )
        {
          //
          // If latc compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_LUMINANCE_LATC1); i <= static_cast<size_t>(Image::PixelFormat::COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
            }
          }

          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j ++ )
          {
            m_texImageFmts[static_cast<size_t>(Image::PixelFormat::LUMINANCE)][j].compressedFmt = m_texImageFmts[static_cast<size_t>(Image::PixelFormat::LUMINANCE)][j].fixedPtFmt;
            m_texImageFmts[static_cast<size_t>(Image::PixelFormat::LUMINANCE_ALPHA)][j].compressedFmt = 
              m_texImageFmts[static_cast<size_t>(Image::PixelFormat::LUMINANCE_ALPHA)][j].fixedPtFmt;
          }
        }
        else
        {
          // set up these formats
          size_t i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_LUMINANCE_LATC1);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_LUMINANCE_LATC1_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_LUMINANCE_LATC1_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_SIGNED_LUMINANCE_LATC1);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_LUMINANCE_ALPHA_LATC2);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }
        }

        if( !GLEW_EXT_texture_compression_rgtc )
        {
          //
          // If rgtc compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_RED_RGTC1); i <= static_cast<size_t>(Image::PixelFormat::COMPRESSED_SIGNED_RG_RGTC2); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
            }
          }
        }
        else
        {
          size_t i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_RED_RGTC1);
          // set up these formats
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_RED_RGTC1_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_RED_RGTC1_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_SIGNED_RED_RGTC1);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_RG_RGTC2);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_SIGNED_RG_RGTC2);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }
        }

        if( !GLEW_ARB_depth_texture )
        {
          //
          // If rgtc compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = 0; i < static_cast<size_t>(Image::PixelFormat::NUM_FORMATS); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
            }
          }

          size_t i = static_cast<size_t>(Image::PixelFormat::DEPTH_COMPONENT);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
            m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
            m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
          }
        }

        if( !GLEW_EXT_packed_depth_stencil )
        {
          //
          // If rgtc compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          size_t j = static_cast<size_t>(Image::PixelDataType::UNSIGNED_INT_24_8);
          for( size_t i = 0; i < static_cast<size_t>(Image::PixelFormat::NUM_FORMATS); i ++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
            m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
            m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
          }

          size_t i = static_cast<size_t>(Image::PixelFormat::DEPTH_STENCIL);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
            m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
            m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
          }
        }

        if( !GLEW_EXT_texture_shared_exponent )
        {
          //
          // If shared_exponent is not available, then disallow PixelDataType::UNSIGNED_INT_10F_11F_11F
          //
          size_t j = static_cast<size_t>(Image::PixelDataType::UNSIGNED_INT_10F_11F_11F);
          for( size_t i = 0; i < static_cast<size_t>(Image::PixelFormat::NUM_FORMATS); i ++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
            m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
            m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
          }
        }

        if( !GLEW_EXT_packed_float )
        {
          //
          // If packed_float is not available, then disallow PixelDataType::UNSIGNED_INT_5_9_9_9
          //
          size_t j = static_cast<size_t>(Image::PixelDataType::UNSIGNED_INT_5_9_9_9);
          for( size_t i = 0; i < static_cast<size_t>(Image::PixelFormat::NUM_FORMATS); i ++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
            m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
            m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
            m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
          }
        }

        if( !GLEW_ARB_texture_compression )
        {
          //
          // If texture compression is not available, then set the compressed formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = 0; i < static_cast<size_t>(Image::PixelFormat::NUM_FORMATS); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j ++ )
            {
              m_texImageFmts[i][j].compressedFmt = m_texImageFmts[i][j].fixedPtFmt;
            }
          }
        }

        if( !GLEW_EXT_texture_compression_s3tc )
        {
          for( size_t i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_RGB_DXT1); i <= static_cast<size_t>(Image::PixelFormat::COMPRESSED_SRGBA_DXT5); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
            }
          }
        }
        else
        {
          // set up these formats
          size_t i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_RGB_DXT1);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_RGBA_DXT1);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_RGBA_DXT3);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }

          i = static_cast<size_t>(Image::PixelFormat::COMPRESSED_RGBA_DXT5);
          for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j++ )
          {
            m_texImageFmts[i][j].fixedPtFmt    = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].floatFmt      = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].compressedFmt = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].integerFmt    = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].nonLinearFmt  = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;

            // these are not used
            m_texImageFmts[i][j].usrFmt   = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;

            m_texImageFmts[i][j].type          = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            m_texImageFmts[i][j].uploadHint    = 0;
          }
        }

        if( !GLEW_EXT_texture_integer )
        {
          //
          // If texture integer is not available, then set the integer formats
          // to be the same as the fixedPt formats.
          //
          for( size_t i = 0; i < static_cast<size_t>(Image::PixelFormat::NUM_FORMATS); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j ++ )
            {
              m_texImageFmts[i][j].integerFmt    = m_texImageFmts[i][j].fixedPtFmt;
            }
          }

          for( size_t i = static_cast<size_t>(Image::PixelFormat::INTEGER_ALPHA); i <= static_cast<size_t>(Image::PixelFormat::INTEGER_BGRA); i ++ )
          {
            for( size_t j = 0; j < static_cast<size_t>(Image::PixelDataType::NUM_TYPES); j ++ )
            {
              m_texImageFmts[i][j].fixedPtFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].floatFmt      = DP_TIF_INVALID;
              m_texImageFmts[i][j].compressedFmt = DP_TIF_INVALID;
              m_texImageFmts[i][j].integerFmt    = DP_TIF_INVALID;
              m_texImageFmts[i][j].nonLinearFmt  = DP_TIF_INVALID;
            }
          }
        }

        if (  !GLEW_ARB_vertex_buffer_object 
          || !GLEW_ARB_pixel_buffer_object )
        {
          m_texImageFmts[static_cast<size_t>(Image::PixelFormat::BGRA)][static_cast<size_t>(Image::PixelDataType::BYTE)].uploadHint = 0; 
          m_texImageFmts[static_cast<size_t>(Image::PixelFormat::BGRA)][static_cast<size_t>(Image::PixelDataType::UNSIGNED_BYTE)].uploadHint = 0; 
          m_texImageFmts[static_cast<size_t>(Image::PixelFormat::RGBA)][static_cast<size_t>(Image::PixelDataType::HALF)].uploadHint = 0;
          m_texImageFmts[static_cast<size_t>(Image::PixelFormat::RGBA)][static_cast<size_t>(Image::PixelDataType::FLOAT)].uploadHint = 0;
        }
      }

      TexImageFmts texImageFmts;

      //
      // Checks to see if this is an integer format.
      //
      static
        bool isInteger( GLint usrFmt )
      {
        if( usrFmt >= static_cast<size_t>(Image::PixelFormat::INTEGER_ALPHA) && 
            usrFmt <= static_cast<size_t>(Image::PixelFormat::INTEGER_BGRA) )
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
        case TextureHost::TextureGPUFormat::ALPHA8:
          gpuFmt = GL_ALPHA8;
          return true;

        case TextureHost::TextureGPUFormat::ALPHA16:
          gpuFmt = GL_ALPHA16;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_ALPHA8:
          gpuFmt = GL_SIGNED_ALPHA8_NV;
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE8:
          gpuFmt = GL_LUMINANCE8;
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE16:
          gpuFmt = GL_LUMINANCE16;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE8:
          gpuFmt = GL_SIGNED_LUMINANCE8_NV;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE16:
          gpuFmt = GL_SIGNED_LUMINANCE8_NV; // not supported
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE8_ALPHA8:
          gpuFmt = GL_LUMINANCE8_ALPHA8;
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE16_ALPHA16:
          gpuFmt = GL_LUMINANCE16_ALPHA16;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE8_ALPHA8:
          gpuFmt = GL_SIGNED_LUMINANCE8_ALPHA8_NV;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE16_ALPHA16:
          gpuFmt = GL_SIGNED_LUMINANCE8_ALPHA8_NV;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_LUMINANCE:
          gpuFmt = GL_COMPRESSED_LUMINANCE;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_LUMINANCE_ALPHA:
          gpuFmt = GL_COMPRESSED_LUMINANCE_ALPHA;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_LUMINANCE_LATC:         // ext_texture_compression_latc
          gpuFmt = GL_COMPRESSED_LUMINANCE_LATC1_EXT;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_COMPRESSED_LUMINANCE_LATC:
          gpuFmt = GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_LUMINANCE_ALPHA_LATC:
          gpuFmt = GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_COMPRESSED_LUMINANCE_ALPHA_LATC:
          gpuFmt = GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT;
          return true;

        case TextureHost::TextureGPUFormat::RGB8:
          gpuFmt = GL_RGB8;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_RGB8:
          gpuFmt = GL_SIGNED_RGB8_NV;
          return true;

        case TextureHost::TextureGPUFormat::RGB16:       // native on G80
          gpuFmt = GL_RGB16;
          return true;

        case TextureHost::TextureGPUFormat::RGBA8:
          gpuFmt = GL_RGBA8;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_RGBA8:
          gpuFmt = GL_SIGNED_RGBA_NV;
          return true;

        case TextureHost::TextureGPUFormat::RGB10:       // native on G80
          gpuFmt = GL_RGB10;
          return true;

        case TextureHost::TextureGPUFormat::RGB10_A2:    // native on G80
          gpuFmt = GL_RGB10_A2;
          return true;

        case TextureHost::TextureGPUFormat::RGBA16:      // native on G80
          gpuFmt = GL_RGB16;
          return true;

        case TextureHost::TextureGPUFormat::SRGB:
          gpuFmt = GL_SRGB_EXT;
          return true;

        case TextureHost::TextureGPUFormat::SRGBA:
          gpuFmt = GL_SRGB_ALPHA_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_SRGB:        // ext_texture_srgb
          gpuFmt = GL_COMPRESSED_SRGB_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_SRGBA:
          gpuFmt = GL_COMPRESSED_SRGB_ALPHA_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_SRGB_DXT1:
          gpuFmt = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_SRGB_DXT3:
          gpuFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_SRGB_DXT5:
          gpuFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_SRGBA_DXT1:
          gpuFmt = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_RGB:
          gpuFmt = GL_COMPRESSED_RGB;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_RGBA:
          gpuFmt = GL_COMPRESSED_RGBA;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_RGB_DXT1:
          gpuFmt = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_RGB_DXT3:
          gpuFmt = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_RGB_DXT5:
          gpuFmt = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
          return true;

        case TextureHost::TextureGPUFormat::COMPRESSED_RGBA_DXT1:
          gpuFmt = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
          return true;

        case TextureHost::TextureGPUFormat::DEPTH16:              // arb_shadow
          gpuFmt = GL_DEPTH_COMPONENT16_ARB;
          return true;

        case TextureHost::TextureGPUFormat::DEPTH24:
          gpuFmt = GL_DEPTH_COMPONENT24_ARB;
          return true;

        case TextureHost::TextureGPUFormat::DEPTH32:
          gpuFmt = GL_DEPTH_COMPONENT32_ARB;
          return true;

        case TextureHost::TextureGPUFormat::DEPTH24_STENCIL8:  // ext_packed_depth_stencil
          gpuFmt = GL_DEPTH24_STENCIL8_EXT;
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE16F:         // arb_texture_float
          gpuFmt = GL_LUMINANCE16F_ARB;
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE32F:
          gpuFmt = GL_LUMINANCE32F_ARB;
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE_ALPHA16F:
          gpuFmt = GL_LUMINANCE_ALPHA16F_ARB;
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE_ALPHA32F:
          gpuFmt = GL_LUMINANCE_ALPHA32F_ARB;
          return true;

        case TextureHost::TextureGPUFormat::RGB16F:
          gpuFmt = GL_RGB16F_ARB;
          return true;

        case TextureHost::TextureGPUFormat::RGB32F:
          gpuFmt = GL_RGB32F_ARB;
          return true;

        case TextureHost::TextureGPUFormat::RGBA16F:
          gpuFmt = GL_RGBA16F_ARB;
          return true;

        case TextureHost::TextureGPUFormat::RGBA32F:
          gpuFmt = GL_RGBA32F_ARB;
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_RGB_INTEGER32: // ext_texture_integer
          gpuFmt = GL_RGB32I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_RGB_INTEGER16:
          gpuFmt = GL_RGB16I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_RGB_INTEGER8:
          gpuFmt = GL_RGB8I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::RGB_INTEGER32:
          gpuFmt = GL_RGB32UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::RGB_INTEGER16:
          gpuFmt = GL_RGB16UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::RGB_INTEGER8:
          gpuFmt = GL_RGB8UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_RGBA_INTEGER32:
          gpuFmt = GL_RGBA32I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_RGBA_INTEGER16:
          gpuFmt = GL_RGBA16I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_RGBA_INTEGER8:
          gpuFmt = GL_RGBA8I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::RGBA_INTEGER32:
          gpuFmt = GL_RGBA32UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::RGBA_INTEGER16:
          gpuFmt = GL_RGBA16UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::RGBA_INTEGER8:
          gpuFmt = GL_RGBA8UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE_INTEGER32:
          gpuFmt = GL_LUMINANCE32I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE_INTEGER16:
          gpuFmt = GL_LUMINANCE16I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE_INTEGER8:
          gpuFmt = GL_LUMINANCE8I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE_INTEGER32:
          gpuFmt = GL_LUMINANCE32UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE_INTEGER16:
          gpuFmt = GL_LUMINANCE16UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE_INTEGER8:
          gpuFmt = GL_LUMINANCE8UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE_ALPHA_INTEGER32:
          gpuFmt = GL_LUMINANCE32I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE_ALPHA_INTEGER16:
          gpuFmt = GL_LUMINANCE16I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::SIGNED_LUMINANCE_ALPHA_INTEGER8:
          gpuFmt = GL_LUMINANCE8I_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE_ALPHA_INTEGER32:
          gpuFmt = GL_LUMINANCE_ALPHA32UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE_ALPHA_INTEGER16:
          gpuFmt = GL_LUMINANCE_ALPHA16UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::LUMINANCE_ALPHA_INTEGER8:
          gpuFmt = GL_LUMINANCE_ALPHA8UI_EXT;
          if( !isInteger( usrFmt ) )
          {
            usrFmt = mapToInteger( usrFmt );
          }
          return true;

        case TextureHost::TextureGPUFormat::UNSIGNED_FLOAT_SHARED_EXPONENT:
          gpuFmt = GL_RGB9_E5_EXT;
          return true;

        case TextureHost::TextureGPUFormat::UNSIGNED_FLOAT_PACKED:
          gpuFmt = GL_R11F_G11F_B10F_EXT;
          return true;

        default:
          return false;
        }
      }
      
      TextureGLSharedPtr TextureGL::create( const dp::gl::TextureSharedPtr& texture )
      {
        return( std::shared_ptr<TextureGL>( new TextureGL( texture ) ) );
      }

      dp::sg::core::HandledObjectSharedPtr TextureGL::clone() const
      {
        return( std::shared_ptr<TextureGL>( new TextureGL( *this ) ) );
      }

      TextureGL::TextureGL( const dp::gl::TextureSharedPtr& texture )
        : m_texture( texture )
      {
        TextureTarget target;
        switch ( texture->getTarget() )
        {
        case GL_TEXTURE_1D:
            target = TextureTarget::TEXTURE_1D;
          break;
        case GL_TEXTURE_2D:
          target = TextureTarget::TEXTURE_2D;
          break;
        case GL_TEXTURE_3D:
          target = TextureTarget::TEXTURE_3D;
          break;
        case GL_TEXTURE_CUBE_MAP:
          target = TextureTarget::TEXTURE_CUBE;
          break;
        case GL_TEXTURE_1D_ARRAY:
          target = TextureTarget::TEXTURE_1D_ARRAY;
          break;
        case GL_TEXTURE_2D_ARRAY:
          target = TextureTarget::TEXTURE_2D_ARRAY;
          break;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
          target = TextureTarget::TEXTURE_CUBE_ARRAY;
          break;
        case GL_TEXTURE_BUFFER:
          target = TextureTarget::TEXTURE_BUFFER;
          break;
        default:
          target = TextureTarget::UNSPECIFIED;
          break;
        };
        setTextureTarget(target);
      }

      const dp::gl::TextureSharedPtr& TextureGL::getTexture() const
      {
        return m_texture;
      }

      bool TextureGL::getTexImageFmt( GLTexImageFmt & tfmt, Image::PixelFormat fmt, Image::PixelDataType type, TextureHost::TextureGPUFormat gpufmt )
      {
        // give a hint while debugging. though, need a run-time check below
        DP_ASSERT(fmt!=Image::PixelFormat::UNKNOWN || type!=Image::PixelDataType::UNKNOWN);  

        if( fmt==Image::PixelFormat::UNKNOWN || type==Image::PixelDataType::UNKNOWN )  
        {
          return false;
        }

        // lookup the formats
        TexImageFmt  dpFmt    = texImageFmts.getFmt(fmt,type);
        // formats for this particular fmt
        TexImageFmt* dpFmts   = texImageFmts.getFmts(fmt);

        tfmt.type       = dpFmt.type;
        tfmt.uploadHint = dpFmt.uploadHint;

        //
        // If gpufmt == TextureGPUFormat::DEFAULT, and we have a floating point format texture, then the assumption is that
        // the texture will be loaded with a floating point format to match the way dp used to work.  So, unless
        // the user has selected a specific generic format, examine the PixelData and if it is float, reset the
        // GPUformat to be float as well. 
        //
        if( gpufmt == TextureHost::TextureGPUFormat::DEFAULT )
        {
          // set it up this way in case there are other formats we need to put in code for.
          switch( type )
          {
            // reset to float if they provided float data
          case Image::PixelDataType::FLOAT32:
          case Image::PixelDataType::FLOAT16:
            gpufmt = TextureHost::TextureGPUFormat::FLOAT;
            break;

          default:
            // no changes
            break;
          }
        }

        switch( gpufmt )
        {
          // default "defaults" to FIXED format
        case TextureHost::TextureGPUFormat::DEFAULT:
        case TextureHost::TextureGPUFormat::FIXED:
          tfmt.intFmt     = dpFmt.fixedPtFmt;
          tfmt.usrFmt     = dpFmt.usrFmt;
          break;

        case TextureHost::TextureGPUFormat::COMPRESSED_FIXED:
          tfmt.intFmt     = dpFmt.compressedFmt;
          tfmt.usrFmt     = dpFmt.usrFmt;
          break;

        case TextureHost::TextureGPUFormat::FLOAT:
          tfmt.intFmt     = dpFmt.floatFmt;
          tfmt.usrFmt     = dpFmt.usrFmt;
          break;

        case TextureHost::TextureGPUFormat::NONLINEAR:
          tfmt.intFmt     = dpFmt.nonLinearFmt;
          tfmt.usrFmt     = dpFmt.usrFmt;
          break;

        case TextureHost::TextureGPUFormat::INTEGER:
          tfmt.intFmt     = dpFmt.integerFmt;
          tfmt.usrFmt     = mapToInteger( dpFmt.usrFmt );
          break;

        case TextureHost::TextureGPUFormat::FLOAT16:
          tfmt.intFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::FLOAT16)].floatFmt;
          tfmt.usrFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::FLOAT16)].usrFmt;
          break;

        case TextureHost::TextureGPUFormat::FLOAT32:
          tfmt.intFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::FLOAT32)].floatFmt;
          tfmt.usrFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::FLOAT32)].usrFmt;
          break;

        case TextureHost::TextureGPUFormat::FIXED8:
          {
            switch( type )
            {
            case Image::PixelDataType::BYTE:
            case Image::PixelDataType::SHORT:
            case Image::PixelDataType::INT:
              tfmt.intFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::BYTE)].fixedPtFmt;
              tfmt.usrFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::BYTE)].usrFmt;
              break;

            case Image::PixelDataType::UNSIGNED_BYTE:
            case Image::PixelDataType::UNSIGNED_SHORT:
            case Image::PixelDataType::UNSIGNED_INT:
            default:
              tfmt.intFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::UNSIGNED_BYTE)].fixedPtFmt;
              tfmt.usrFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::UNSIGNED_BYTE)].usrFmt;
              break;
            }
            break;
          }

        case TextureHost::TextureGPUFormat::FIXED10:
          // only 1 format supported
          tfmt.intFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::UNSIGNED_INT_2_10_10_10)].fixedPtFmt;
          tfmt.usrFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::UNSIGNED_INT_2_10_10_10)].usrFmt;
          break;

        case TextureHost::TextureGPUFormat::FIXED16:
          {
            switch( type )
            {
            case Image::PixelDataType::BYTE:
            case Image::PixelDataType::SHORT:
            case Image::PixelDataType::INT:
              tfmt.intFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::SHORT)].fixedPtFmt;
              tfmt.usrFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::SHORT)].usrFmt;
              break;

            case Image::PixelDataType::UNSIGNED_BYTE:
            case Image::PixelDataType::UNSIGNED_SHORT:
            case Image::PixelDataType::UNSIGNED_INT:
            default:
              tfmt.intFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::UNSIGNED_SHORT)].fixedPtFmt;
              tfmt.usrFmt     = dpFmts[static_cast<size_t>(Image::PixelDataType::UNSIGNED_SHORT)].usrFmt;
              break;
            }
            break;
          }

        default:
          // set this so the routine knows the number of components
          tfmt.intFmt     = dpFmt.fixedPtFmt;
          tfmt.usrFmt     = dpFmt.usrFmt;

          return getTextureGPUFormatValues( gpufmt, tfmt.intFmt, tfmt.usrFmt );
        }

        return ( tfmt.intFmt != DP_TIF_INVALID );
      }

    } // namespace gl
  } // namespace sg
} // namespace dp

