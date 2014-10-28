// Copyright NVIDIA Corporation 2013
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


#include <dp/sg/renderer/rix/gl/inc/ResourceSampler.h>
#include <dp/sg/core/Sampler.h>

namespace dp
{
  namespace sg
  {
    namespace renderer
    {
      namespace rix
      {
        namespace gl
        {

          SmartResourceSampler ResourceSampler::get( const dp::sg::core::SamplerSharedPtr &sampler, const SmartResourceManager& resourceManager )
          {
            assert( sampler );
            assert( resourceManager );

            SmartResourceSampler resourceSampler = resourceManager->getResource<ResourceSampler>( reinterpret_cast<size_t>(sampler.getWeakPtr()) );
            if ( !resourceSampler )
            {
              resourceSampler = std::shared_ptr<ResourceSampler>( new ResourceSampler( sampler, resourceManager ) );
              resourceSampler->m_samplerHandle = resourceManager->getRenderer()->samplerCreate();
              resourceSampler->update();
            }
            return resourceSampler;
          }

          ResourceSampler::ResourceSampler( const dp::sg::core::SamplerSharedPtr &sampler, const SmartResourceManager& resourceManager )
            : ResourceManager::Resource( reinterpret_cast<size_t>( sampler.getWeakPtr() ), resourceManager )
            , m_sampler( sampler )
          {
            resourceManager->subscribe( this );
          }

          ResourceSampler::~ResourceSampler()
          {
            m_resourceManager->unsubscribe( this );
          }

          const dp::sg::core::HandledObjectSharedPtr& ResourceSampler::getHandledObject() const
          {
            return m_sampler.inplaceCast<dp::sg::core::HandledObject>();
          }

          void ResourceSampler::update()
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            if(  m_sampler )
            {
              /** Get resource for SamplerState **/
              dp::rix::core::SamplerStateDataCommon samplerStateDataCommon( minFilterSceniXToRiX( m_sampler->getMinFilterMode() ),
                                                                  magFilterSceniXToRiX( m_sampler->getMagFilterMode() ),
                                                                  wrapModeSceniXToRiX( m_sampler->getWrapModeS() ),
                                                                  wrapModeSceniXToRiX( m_sampler->getWrapModeT() ),
                                                                  wrapModeSceniXToRiX( m_sampler->getWrapModeR() ),
                                                                  compareModeSceniXToRiX( m_sampler->getCompareMode() ) );

              m_samplerStateHandle = renderer->samplerStateCreate( samplerStateDataCommon );

              /** Get resource for Texture**/
              dp::sg::core::TextureSharedPtr texture = m_sampler->getTexture();
              if ( texture )
              {
                m_resourceTexture = ResourceTexture::get( texture, m_resourceManager );
              }
              else
              {
                m_resourceTexture.reset();
              }
            }
            else
            {
              m_resourceTexture.reset();
              m_samplerStateHandle.reset();
            }

            renderer->samplerSetSamplerState( m_samplerHandle, m_samplerStateHandle );
            renderer->samplerSetTexture( m_samplerHandle, m_resourceTexture ? m_resourceTexture->m_textureHandle : nullptr);
          }

          // Helper functions to translate resources from SceniX to RiX.
          dp::rix::core::SamplerStateFilterMode ResourceSampler::magFilterSceniXToRiX( dp::sg::core::TextureMagFilterMode mf )
          {
            switch ( mf )
            {
              case dp::sg::core::TFM_MIN_NEAREST:
                return dp::rix::core::SSFM_NEAREST;
              default:
                assert(!"Unexpected TextureMagFilterMode");
                // fall through
              case dp::sg::core::TFM_MIN_LINEAR:
                return dp::rix::core::SSFM_LINEAR;
            }
          }

          dp::rix::core::SamplerStateFilterMode ResourceSampler::minFilterSceniXToRiX( dp::sg::core::TextureMinFilterMode mf )
          {
            switch ( mf )
            {
              case dp::sg::core::TFM_MIN_NEAREST:
                return dp::rix::core::SSFM_NEAREST;
              case dp::sg::core::TFM_MIN_LINEAR:
                return dp::rix::core::SSFM_LINEAR;
              case dp::sg::core::TFM_MIN_NEAREST_MIPMAP_NEAREST:
                return dp::rix::core::SSFM_NEAREST_MIPMAP_NEAREST;
              case dp::sg::core::TFM_MIN_LINEAR_MIPMAP_NEAREST:
                return dp::rix::core::SSFM_NEAREST_MIPMAP_LINEAR;
              case dp::sg::core::TFM_MIN_LINEAR_MIPMAP_LINEAR:
                return dp::rix::core::SSFM_LINEAR_MIPMAP_LINEAR;
              default:
                assert(!"Unexpected TextureMinFilterMode");
                // fall through
              case dp::sg::core::TFM_MIN_NEAREST_MIPMAP_LINEAR:
                return dp::rix::core::SSFM_NEAREST_MIPMAP_LINEAR;
            }
          }

          dp::rix::core::SamplerStateWrapMode ResourceSampler::wrapModeSceniXToRiX( dp::sg::core::TextureWrapMode wm )
          {
            switch ( wm )
            {
              case dp::sg::core::TWM_CLAMP:
                return dp::rix::core::SSWM_CLAMP;
              case dp::sg::core::TWM_CLAMP_TO_BORDER:
                return dp::rix::core::SSWM_CLAMP_TO_BORDER;
              case dp::sg::core::TWM_CLAMP_TO_EDGE:
                return dp::rix::core::SSWM_CLAMP_TO_EDGE;
              // DAR FIXME No RiX equivalents existing for the three mirror_clamp modes. Map them to mirror_repeat.
              case dp::sg::core::TWM_MIRROR_CLAMP:
              case dp::sg::core::TWM_MIRROR_CLAMP_TO_EDGE:
              case dp::sg::core::TWM_MIRROR_CLAMP_TO_BORDER:
              case dp::sg::core::TWM_MIRROR_REPEAT:
                return dp::rix::core::SSWM_MIRRORED_REPEAT;
              default:
                assert(!"Unexpected wrap mode");
                // fall through
              case dp::sg::core::TWM_REPEAT:
                return dp::rix::core::SSWM_REPEAT;
            }
          }

          dp::rix::core::SamplerStateCompareMode ResourceSampler::compareModeSceniXToRiX( dp::sg::core::TextureCompareMode tcm )
          {
            switch( tcm )
            {
              case dp::sg::core::TCM_R_TO_TEXTURE :
                return dp::rix::core::SSCM_R_TO_TEXTURE;
              default :
                assert( !"Unexpected compare mode" );
                // fall through
              case dp::sg::core::TCM_NONE :
                return dp::rix::core::SSCM_NONE;
            }
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
