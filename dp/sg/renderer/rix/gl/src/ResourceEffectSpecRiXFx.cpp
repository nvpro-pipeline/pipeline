// Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/fx/EffectLibrary.h>

#include <dp/sg/renderer/rix/gl/inc/ResourceEffectSpecRiXFx.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceTexture.h>

#include <dp/sg/core/Sampler.h>


using namespace dp::fx;
using namespace dp::math;

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

          ResourceEffectSpecRiXFxSharedPtr ResourceEffectSpecRiXFx::get( const EffectSpecSharedPtr &effectSpec, const dp::rix::fx::ManagerSharedPtr& rixFx, const ResourceManagerSharedPtr& resourceManager )
          {
            assert( effectSpec );
            assert( !!resourceManager );

            ResourceEffectSpecRiXFxSharedPtr resourceEffectSpec = resourceManager->getResource<ResourceEffectSpecRiXFx>( reinterpret_cast<size_t>(effectSpec.operator->()) );   // Big Hack !!
            if ( !resourceEffectSpec )
            {
              resourceEffectSpec = std::shared_ptr<ResourceEffectSpecRiXFx>( new ResourceEffectSpecRiXFx( effectSpec, rixFx, resourceManager ) );
            }

            return resourceEffectSpec;
          }

          ResourceEffectSpecRiXFx::ResourceEffectSpecRiXFx( const EffectSpecSharedPtr& effectSpec, const dp::rix::fx::ManagerSharedPtr& rixfx, const ResourceManagerSharedPtr& resourceManager )
            : ResourceManager::Resource( reinterpret_cast<size_t>( effectSpec.operator->() ), resourceManager )   // Big Hack !!
            , m_resourceManager( resourceManager )
            , m_effectSpec( effectSpec )
          {
            for ( EffectSpec::iterator itpgs = effectSpec->beginParameterGroupSpecs(); itpgs != effectSpec->endParameterGroupSpecs(); ++itpgs )
            {
              dp::sg::core::ParameterGroupDataSharedPtr data = dp::sg::core::ParameterGroupData::create( *itpgs );
              m_defaultDatas.push_back( ResourceParameterGroupDataRiXFx::get( data, rixfx, m_resourceManager ) );
            }
          }

          ResourceEffectSpecRiXFx::~ResourceEffectSpecRiXFx()
          {

          }

          const ResourceParameterGroupDataRiXFxSharedPtr& ResourceEffectSpecRiXFx::getDefaultParameterGroupDataResource( const dp::fx::EffectSpec::iterator& parameterGroupSpec) const
          {
            return m_defaultDatas[distance( m_effectSpec->beginParameterGroupSpecs(), parameterGroupSpec )];
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

