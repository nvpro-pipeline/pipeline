// Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/renderer/rix/gl/Config.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceManager.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceTexture.h>

#include <dp/sg/core/CoreTypes.h>
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
          DEFINE_PTR_TYPES( ResourceSampler );

          class ResourceSampler : public ResourceManager::Resource
          {
          public:
            /** \brief Fetch resource for the given object/resourceManager. If no resource exists it'll be created **/
            static ResourceSamplerSharedPtr get( const dp::sg::core::SamplerSharedPtr &Sampler, const ResourceManagerSharedPtr& resourceManager );
            virtual void update();

            ~ResourceSampler();

            virtual dp::sg::core::HandledObjectSharedPtr getHandledObject() const;

            dp::rix::core::SamplerSharedHandle       m_samplerHandle;
            ResourceTextureSharedPtr                 m_resourceTexture;
            dp::rix::core::SamplerStateSharedHandle  m_samplerStateHandle;

          protected:
            dp::sg::core::SamplerSharedPtr  m_sampler;

            ResourceSampler( const dp::sg::core::SamplerSharedPtr &Sampler, const ResourceManagerSharedPtr& resourceManager );

          private:
            dp::rix::core::SamplerStateCompareMode compareModeSceniXToRiX( dp::sg::core::TextureCompareMode tcm );
            dp::rix::core::SamplerStateFilterMode magFilterSceniXToRiX( dp::sg::core::TextureMagFilterMode mf );
            dp::rix::core::SamplerStateFilterMode minFilterSceniXToRiX( dp::sg::core::TextureMinFilterMode mf );
            dp::rix::core::SamplerStateWrapMode wrapModeSceniXToRiX( dp::sg::core::TextureWrapMode wm );
          };

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
