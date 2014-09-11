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


#include <dp/sg/core/Texture.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( Texture, MipmapRequired );

      BEGIN_REFLECTION_INFO ( Texture )
        INIT_STATIC_PROPERTY_RO_BOOL( Texture, MipmapRequired , bool , SEMANTIC_VALUE, value );
      END_REFLECTION_INFO

      Texture::Texture()
        : m_mipmapUseCount(0)
        , m_textureTarget( TT_UNSPECIFIED_TEXTURE_TARGET )
        , m_hashKey(0)
        , m_hashKeyValid(false)
      {
      }

      Texture::Texture( TextureTarget textureTarget )
        : m_mipmapUseCount(0)
        , m_textureTarget( textureTarget )
        , m_hashKey(0)
        , m_hashKeyValid(false)
      {
      }

      Texture::~Texture()
      {
      }

      bool Texture::isEquivalent( TextureSharedPtr const& texture, bool deepCompare ) const
      {
        if ( texture == this )
        {
          return( true );
        }
        return(   ( m_mipmapUseCount  == texture->m_mipmapUseCount )
              &&  ( m_textureTarget   == texture->m_textureTarget ) );
      }

      void Texture::setTextureTarget( TextureTarget tt )
      {
        if ( m_textureTarget != tt )
        {
          m_textureTarget = tt;
          invalidateHashKey();
        }
      }

      void Texture::incrementMipmapUseCount()
      {
        m_mipmapUseCount++;

        if ( m_mipmapUseCount == 1 )
        {
          notify( PropertyEvent( this, PID_MipmapRequired ) );
        }

        invalidateHashKey();
        onIncrementMipmapUseCount();
      }

      void Texture::decrementMipmapUseCount()
      {
        DP_ASSERT(m_mipmapUseCount); // wraparounds? this must be avoided!
        m_mipmapUseCount--;

        if ( m_mipmapUseCount == 0 )
        {
          notify( PropertyEvent( this, PID_MipmapRequired ) );
        }

        invalidateHashKey();
        onDecrementMipmapUseCount();
      }

      bool Texture::isMipmapRequired() const
      {
        return( m_mipmapUseCount > 0 );
      }

      void Texture::onIncrementMipmapUseCount()
      {
      }

      void Texture::onDecrementMipmapUseCount()
      {
      }

      dp::util::HashKey Texture::getHashKey() const
      {
        if ( ! m_hashKeyValid )
        {
          dp::util::HashGeneratorMurMur hg;
          feedHashGenerator( hg );
          hg.finalize( (unsigned int *)&m_hashKey );
          m_hashKeyValid = true;
        }
        return( m_hashKey );
      }

      void Texture::feedHashGenerator( dp::util::HashGenerator & hg ) const
      {
        hg.update( reinterpret_cast<const unsigned char *>(&m_mipmapUseCount), sizeof(m_mipmapUseCount) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_textureTarget), sizeof(m_textureTarget) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
