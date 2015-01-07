// Copyright NVIDIA Corporation 2012
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
/** @file */

#include <dp/sg/core/nvsgapi.h>
#include <dp/sg/core/Object.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/Texture.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      //! Texture compare mode
      typedef enum
      {
        TCM_NONE = 0                        //!< Corresponds to GL_NONE in OpenGL
      , TCM_R_TO_TEXTURE                    //!< Corrensponds to GL_COMPARE_R_TO_TEXTURE in OpenGL
      } TextureCompareMode;

      //! Texture magnification mode
      typedef enum
      {
        TFM_MAG_NEAREST = 0                 //!< Corresponds to GL_NEAREST in OpenGL
      , TFM_MAG_LINEAR                      //!< Corresponds to GL_LINEAR in OpenGL
      } TextureMagFilterMode;

      //! Texture minification mode
      typedef enum
      {
        TFM_MIN_NEAREST = 0                               //!< Corresponds to GL_NEAREST in OpenGL
      , TFM_MIN_LINEAR                                    //!< Corresponds to GL_LINEAR in OpenGL
      , TFM_MIN_LINEAR_MIPMAP_LINEAR                      //!< Corresponds to GL_LINEAR_MIPMAP_LINEAR
      , TFM_MIN_NEAREST_MIPMAP_NEAREST                    //!< Corresponds to GL_NEAREST_MIPMAP_NEAREST
      , TFM_MIN_NEAREST_MIPMAP_LINEAR                     //!< Corresponds to GL_NEAREST_MIPMAP_LINEAR (the OpenGL default)
      , TFM_MIN_LINEAR_MIPMAP_NEAREST                     //!< Corresponds to GL_LINEAR_MIPMAP_NEAREST
      } TextureMinFilterMode;

      DP_SG_CORE_API bool requiresMipmaps( TextureMinFilterMode tmfm );

      //! Texture wrap coordinate axis
      typedef enum
      {
        TWCA_S = 0    //!< S axis
      , TWCA_T        //!< T axis
      , TWCA_R        //!< R axis
      } TexWrapCoordAxis;

      //! Texture wrap mode
      typedef enum
      {
        TWM_REPEAT = 0              //!< Corresponds to GL_REPEAT in OpenGL
      , TWM_CLAMP                   //!< Corresponds to GL_CLAMP in OpenGL
      , TWM_MIRROR_REPEAT           //!< Corresponds to GL_MIRRORED_REPEAT in OpenGL
      , TWM_CLAMP_TO_EDGE           //!< Corresponds to GL_CLAMP_TO_EDGE in OpenGL
      , TWM_CLAMP_TO_BORDER         //!< Corresponds to GL_CLAMP_TO_BORDER in OpenGL
      , TWM_MIRROR_CLAMP            //!< Corresponds to GL_MIRROR_CLAMP in OpenGL
      , TWM_MIRROR_CLAMP_TO_EDGE    //!< Corresponds to GL_MIRROR_CLAMP_TO_EDGE in OpenGL
      , TWM_MIRROR_CLAMP_TO_BORDER  //!< Corresponds to GL_MIRROR_CLAMP_TO_BORDER in OpenGL
      } TextureWrapMode;


      class Sampler : public Object
      {
        public:
          DP_SG_CORE_API static SamplerSharedPtr create( const TextureSharedPtr & texture = TextureSharedPtr() );

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~Sampler();

        public:
          DP_SG_CORE_API const TextureSharedPtr & getTexture() const;
          DP_SG_CORE_API void setTexture( const TextureSharedPtr & texture );

          DP_SG_CORE_API const dp::math::Vec4f & getBorderColor() const;
          DP_SG_CORE_API void setBorderColor( const dp::math::Vec4f & color );

          DP_SG_CORE_API TextureMagFilterMode getMagFilterMode() const;
          DP_SG_CORE_API void setMagFilterMode( TextureMagFilterMode filterMode );

          DP_SG_CORE_API TextureMinFilterMode getMinFilterMode() const;
          DP_SG_CORE_API void setMinFilterMode( TextureMinFilterMode filterMode );

          DP_SG_CORE_API TextureWrapMode getWrapMode( TexWrapCoordAxis axis ) const;
          DP_SG_CORE_API void setWrapMode( TexWrapCoordAxis axis, TextureWrapMode wrapMode );
          DP_SG_CORE_API void setWrapModes( TextureWrapMode wrapModeS, TextureWrapMode wrapModeT, TextureWrapMode wrapModeR );
          DP_SG_CORE_API TextureWrapMode getWrapModeS() const;
          DP_SG_CORE_API void setWrapModeS( TextureWrapMode wrapMode );
          DP_SG_CORE_API TextureWrapMode getWrapModeT() const;
          DP_SG_CORE_API void setWrapModeT( TextureWrapMode wrapMode );
          DP_SG_CORE_API TextureWrapMode getWrapModeR() const;
          DP_SG_CORE_API void setWrapModeR( TextureWrapMode wrapMode );
          DP_SG_CORE_API TextureCompareMode getCompareMode() const;
          DP_SG_CORE_API void setCompareMode( TextureCompareMode compareMode );

          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const;
          DP_SG_CORE_API Sampler & operator=( const Sampler & rhs );

          REFLECTION_INFO_API( DP_SG_CORE_API, Sampler );

          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Texture );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( BorderColor );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( MagFilterMode );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( MinFilterMode );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( WrapModeS );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( WrapModeT );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( WrapModeR );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          DP_SG_CORE_API Sampler( const TextureSharedPtr & texture );

          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          void releaseMipmapCount();
          void increaseMipmapCount();

          TextureSharedPtr      m_texture;
          dp::math::Vec4f       m_borderColor;
          TextureMagFilterMode  m_magFilterMode;
          TextureMinFilterMode  m_minFilterMode;
          TextureWrapMode       m_wrapMode[3];
          TextureCompareMode    m_compareMode;
      };

    } // namespace core
  } // namespace sg
} // namespace dp
