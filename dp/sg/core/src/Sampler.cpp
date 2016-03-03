// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Texture.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( Sampler, Texture);
      DEFINE_STATIC_PROPERTY( Sampler, BorderColor );
      DEFINE_STATIC_PROPERTY( Sampler, MagFilterMode );
      DEFINE_STATIC_PROPERTY( Sampler, MinFilterMode );
      DEFINE_STATIC_PROPERTY( Sampler, WrapModeS );
      DEFINE_STATIC_PROPERTY( Sampler, WrapModeT );
      DEFINE_STATIC_PROPERTY( Sampler, WrapModeR );

      BEGIN_REFLECTION_INFO ( Sampler )
        DERIVE_STATIC_PROPERTIES( Sampler, Object );
        INIT_STATIC_PROPERTY_RW(      Sampler, Texture      , TextureSharedPtr    , Semantic::OBJECT, const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW(      Sampler, BorderColor  , dp::math::Vec4f     , Semantic::COLOR , const_reference, const_reference );
        INIT_STATIC_PROPERTY_RW_ENUM( Sampler, MagFilterMode, TextureMagFilterMode, Semantic::VALUE , value          , value );
        INIT_STATIC_PROPERTY_RW_ENUM( Sampler, MinFilterMode, TextureMinFilterMode, Semantic::VALUE , value          , value );
        INIT_STATIC_PROPERTY_RW_ENUM( Sampler, WrapModeS    , TextureWrapMode     , Semantic::VALUE , value          , value );
        INIT_STATIC_PROPERTY_RW_ENUM( Sampler, WrapModeT    , TextureWrapMode     , Semantic::VALUE , value          , value );
        INIT_STATIC_PROPERTY_RW_ENUM( Sampler, WrapModeR    , TextureWrapMode     , Semantic::VALUE , value          , value );
      END_REFLECTION_INFO

      SamplerSharedPtr Sampler::create( const TextureSharedPtr & texture )
      {
        return( std::shared_ptr<Sampler>( new Sampler( texture ) ) );
      }

      HandledObjectSharedPtr Sampler::clone() const
      {
        return( std::shared_ptr<Sampler>( new Sampler( *this ) ) );
      }

      Sampler::Sampler( const TextureSharedPtr & texture )
        : m_texture( texture )
        , m_borderColor( dp::math::Vec4f( 0.0f, 0.0f, 0.0f, 0.0f ) )
        , m_magFilterMode( TextureMagFilterMode::LINEAR )
        , m_minFilterMode( TextureMinFilterMode::LINEAR )
        , m_compareMode( TextureCompareMode::NONE )
      {
        m_objectCode = ObjectCode::SAMPLER;

        m_wrapMode[0] = TextureWrapMode::REPEAT;
        m_wrapMode[1] = TextureWrapMode::REPEAT;
        m_wrapMode[2] = TextureWrapMode::REPEAT;
      }

      Sampler::~Sampler()
      {
        releaseMipmapCount();
      }

      const TextureSharedPtr & Sampler::getTexture() const
      {
        return( m_texture );
      }

      void Sampler::setTexture( const TextureSharedPtr & texture )
      {
        if ( m_texture != texture )
        {
          releaseMipmapCount();
          m_texture = texture;
          increaseMipmapCount();
          notify( Event(this ) );
        }
      }

      const dp::math::Vec4f & Sampler::getBorderColor() const
      {
        return( m_borderColor );
      }

      void Sampler::setBorderColor( const dp::math::Vec4f & color )
      {
        if ( m_borderColor != color )
        {
          m_borderColor = color;
          notify( Event(this ) );
        }
      }

      TextureMagFilterMode Sampler::getMagFilterMode() const
      {
        return( m_magFilterMode );
      }

      void Sampler::setMagFilterMode( TextureMagFilterMode filterMode )
      {
        if ( m_magFilterMode != filterMode )
        {
          m_magFilterMode = filterMode;
          notify( Event(this ) );
        }
      }

      TextureMinFilterMode Sampler::getMinFilterMode() const
      {
        return( m_minFilterMode );
      }

      void Sampler::setMinFilterMode( TextureMinFilterMode filterMode )
      {
        if ( m_minFilterMode != filterMode )
        {
          // adjust mipmap use count of the texture, depending on current and new filter mode
          if ( m_texture && ( requiresMipmaps( m_minFilterMode ) != requiresMipmaps( filterMode ) ) )
          {
            if ( requiresMipmaps( filterMode ) )
            {
              m_texture->incrementMipmapUseCount();
            }
            else
            {
              DP_ASSERT( m_texture->isMipmapRequired() );
              m_texture->decrementMipmapUseCount();
            }
          }

          m_minFilterMode = filterMode;
          notify( Event(this ) );
        }
      }

      TextureWrapMode Sampler::getWrapMode( TexWrapCoordAxis axis ) const
      {
        return( m_wrapMode[static_cast<unsigned int>(axis)] );
      }

      void Sampler::setWrapMode( TexWrapCoordAxis axis, TextureWrapMode wrapMode )
      {
        if ( m_wrapMode[static_cast<unsigned int>(axis)] != wrapMode )
        {
          m_wrapMode[static_cast<unsigned int>(axis)] = wrapMode;
          notify( Event(this ) );
        }
      }

      void Sampler::setWrapModes( TextureWrapMode wrapModeS, TextureWrapMode wrapModeT, TextureWrapMode wrapModeR )
      {
        setWrapMode( TexWrapCoordAxis::S, wrapModeS );
        setWrapMode( TexWrapCoordAxis::T, wrapModeT );
        setWrapMode( TexWrapCoordAxis::R, wrapModeR );
      }

      TextureWrapMode Sampler::getWrapModeS() const
      {
        return( getWrapMode( TexWrapCoordAxis::S ) );
      }

      void Sampler::setWrapModeS( TextureWrapMode wrapMode )
      {
        setWrapMode( TexWrapCoordAxis::S, wrapMode );
      }

      TextureWrapMode Sampler::getWrapModeT() const
      {
        return( getWrapMode( TexWrapCoordAxis::T ) );
      }

      void Sampler::setWrapModeT( TextureWrapMode wrapMode )
      {
        setWrapMode( TexWrapCoordAxis::T, wrapMode );
      }

      TextureWrapMode Sampler::getWrapModeR() const
      {
        return( getWrapMode( TexWrapCoordAxis::R ) );
      }

      void Sampler::setWrapModeR( TextureWrapMode wrapMode )
      {
        setWrapMode( TexWrapCoordAxis::R, wrapMode );
      }

      TextureCompareMode Sampler::getCompareMode() const
      {
        return( m_compareMode );
      }

      void Sampler::setCompareMode( TextureCompareMode compareMode )
      {
        if ( m_compareMode != compareMode )
        {
          m_compareMode = compareMode;
          notify( Event(this ) );
        }
      }

      bool Sampler::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object.get() == this )
        {
          return( true );
        }

        bool equi = std::dynamic_pointer_cast<Sampler>(object) && Object::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          SamplerSharedPtr const& s = std::static_pointer_cast<Sampler>(object);
          equi =    ( !!m_texture     == !!s->m_texture )
                &&  ( m_borderColor   == s->m_borderColor )
                &&  ( m_magFilterMode == s->m_magFilterMode )
                &&  ( m_minFilterMode == s->m_minFilterMode )
                &&  ( m_wrapMode[0]   == s->m_wrapMode[0] )
                &&  ( m_wrapMode[1]   == s->m_wrapMode[1] )
                &&  ( m_wrapMode[2]   == s->m_wrapMode[2] )
                &&  ( m_compareMode   == s->m_compareMode );
          if ( equi )
          {
            if ( deepCompare )
            {
              equi = ( m_texture ? m_texture->isEquivalent( s->m_texture, true ) : true );
            }
            else
            {
              equi = ( m_texture == s->m_texture );
            }
          }
        }
        return( equi );
      }

      Sampler & Sampler::operator=( const Sampler & rhs )
      {
        if ( this != &rhs )
        {
          Object::operator=( rhs );
          setTexture( rhs.m_texture );
          setBorderColor( rhs.m_borderColor );
          setMagFilterMode( rhs.m_magFilterMode );
          setMinFilterMode( rhs.m_minFilterMode );
          setWrapMode( TexWrapCoordAxis::S, rhs.m_wrapMode[0] );
          setWrapMode( TexWrapCoordAxis::T, rhs.m_wrapMode[1] );
          setWrapMode( TexWrapCoordAxis::R, rhs.m_wrapMode[2] );
          setCompareMode( rhs.m_compareMode );
        }
        return( *this );
      }

      void Sampler::feedHashGenerator( dp::util::HashGenerator & hg ) const
      {
        Object::feedHashGenerator( hg );
        if ( m_texture )
        {
          hg.update( m_texture );
        }
        hg.update( reinterpret_cast<const unsigned char *>( &m_borderColor ), sizeof( dp::math::Vec4f ) );
        hg.update( reinterpret_cast<const unsigned char *>( &m_magFilterMode ), sizeof( TextureMagFilterMode ) );
        hg.update( reinterpret_cast<const unsigned char *>( &m_minFilterMode ), sizeof( TextureMinFilterMode ) );
        hg.update( reinterpret_cast<const unsigned char *>( &m_wrapMode[0] ), 3 * sizeof( TextureWrapMode ) );
        hg.update( reinterpret_cast<const unsigned char *>( &m_compareMode ), sizeof( TextureCompareMode ) );
      }

      void Sampler::releaseMipmapCount()
      {
        if ( m_texture && requiresMipmaps( m_minFilterMode) )
        {
          m_texture->decrementMipmapUseCount();
        }
      }

      void Sampler::increaseMipmapCount()
      {
        if ( m_texture && requiresMipmaps( m_minFilterMode) )
        {
          m_texture->incrementMipmapUseCount();
        }
      }


      bool requiresMipmaps( TextureMinFilterMode tmfm )
      {
        switch( tmfm )
        {
          case TextureMinFilterMode::NEAREST :
          case TextureMinFilterMode::LINEAR :
            return( false );
            break;
          case TextureMinFilterMode::LINEAR_MIPMAP_LINEAR :
          case TextureMinFilterMode::NEAREST_MIPMAP_NEAREST :
          case TextureMinFilterMode::NEAREST_MIPMAP_LINEAR :
          case TextureMinFilterMode::LINEAR_MIPMAP_NEAREST :
            return( true );
            break;
          default :
            DP_ASSERT( false );
            return( false );
            break;
        }
      }

    } // namespace core
  } // namespace sg
} // namespace dp

namespace dp
{
  namespace util
  {
    template <> const std::string EnumReflection<dp::sg::core::TextureMagFilterMode>::name = "TextureMagFilterMode";

    template <> const std::map<dp::sg::core::TextureMagFilterMode,std::string> EnumReflection<dp::sg::core::TextureMagFilterMode>::values =
    {
      { dp::sg::core::TextureMagFilterMode::NEAREST, "nearest"  },
      { dp::sg::core::TextureMagFilterMode::LINEAR,  "linear"   }
    };

    template <> const std::string EnumReflection<dp::sg::core::TextureMinFilterMode>::name = "TextureMinFilterMode";

    template <> const std::map<dp::sg::core::TextureMinFilterMode,std::string> EnumReflection<dp::sg::core::TextureMinFilterMode>::values =
    {
      { dp::sg::core::TextureMinFilterMode::NEAREST,                "nearest"                 },
      { dp::sg::core::TextureMinFilterMode::LINEAR,                 "linear"                  },
      { dp::sg::core::TextureMinFilterMode::LINEAR_MIPMAP_LINEAR,   "linear_mipmap_linear"    },
      { dp::sg::core::TextureMinFilterMode::NEAREST_MIPMAP_NEAREST, "nearest_mipmap_nearest"  },
      { dp::sg::core::TextureMinFilterMode::NEAREST_MIPMAP_LINEAR,  "nearest_mipmap_linear"   },
      { dp::sg::core::TextureMinFilterMode::LINEAR_MIPMAP_NEAREST,  "linear_mipmap_nearest"   }
    };

    template <> const std::string EnumReflection<dp::sg::core::TextureWrapMode>::name = "TextureWrapMode";

    template <> const std::map<dp::sg::core::TextureWrapMode,std::string> EnumReflection<dp::sg::core::TextureWrapMode>::values =
    {
      { dp::sg::core::TextureWrapMode::REPEAT,                 "repeat"                  },
      { dp::sg::core::TextureWrapMode::CLAMP,                  "clamp"                   },
      { dp::sg::core::TextureWrapMode::MIRROR_REPEAT,          "mirror_repeat"           },
      { dp::sg::core::TextureWrapMode::CLAMP_TO_EDGE,          "clamp_to_edge"           },
      { dp::sg::core::TextureWrapMode::CLAMP_TO_BORDER,        "clamp_to_border"         },
      { dp::sg::core::TextureWrapMode::MIRROR_CLAMP,           "mirror_clamp"            },
      { dp::sg::core::TextureWrapMode::MIRROR_CLAMP_TO_EDGE,   "mirror_clamp_to_edge"    },
      { dp::sg::core::TextureWrapMode::MIRROR_CLAMP_TO_BORDER, "mirror_clamp_to_border"  }
    };

  } // namespace util
} // namespace dp
