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

#include <dp/sg/renderer/rix/gl/inc/ResourceManager.h>
#include <dp/sg/core/Texture.h>

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
          SMART_TYPES( ResourceTexture );
          typedef ResourceTexture* WeakResourceTexture;
    
          class ResourceTexture : public ResourceManager::Resource
          {
          public:
            /** \brief Fetch resource for the given object/resourceManager. If no resource exists it'll be created **/
            static SmartResourceTexture get( const dp::sg::core::TextureSharedPtr& texture, const SmartResourceManager& resourceManager );
      
            virtual ~ResourceTexture();

            virtual const dp::sg::core::HandledObjectSharedPtr& getHandledObject() const;
            virtual void update();

            dp::rix::core::TextureSharedHandle m_textureHandle;
            bool                              m_isNativeTexture;

          protected:
            dp::rix::core::TextureSharedHandle getRiXTexture( dp::sg::core::TextureHostSharedPtr const& texture );
            void updateRiXTexture( const dp::rix::core::TextureSharedHandle& rixTexture, dp::sg::core::TextureHostSharedPtr const& texture );
            ResourceTexture( const dp::sg::core::TextureSharedPtr& texture, const SmartResourceManager& resourceManager );

          protected:
            dp::sg::core::TextureSharedPtr m_texture;
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
