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


#include <dp/sg/renderer/rix/gl/inc/ResourceParameterGroupDataRiXFx.h>

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

          // TODO make implementation global? should it be possible to have two implementations active at the same time? What happens if implementation gets changed and resources are still alive?
          // TODO where to store rixfx?
          ResourceParameterGroupDataRiXFxSharedPtr ResourceParameterGroupDataRiXFx::get( const dp::sg::core::ParameterGroupDataSharedPtr& parameterGroupData, const dp::rix::fx::ManagerSharedPtr& rixFx, const ResourceManagerSharedPtr& resourceManager )
          {
            assert( parameterGroupData );
            assert( !!resourceManager );

            ResourceParameterGroupDataRiXFxSharedPtr resourceParameterGroupData = resourceManager->getResource<ResourceParameterGroupDataRiXFx>( reinterpret_cast<size_t>(parameterGroupData.operator->()) );    // Big Hack !!
            if ( !resourceParameterGroupData )
            {
              resourceParameterGroupData = std::shared_ptr<ResourceParameterGroupDataRiXFx>( new ResourceParameterGroupDataRiXFx( parameterGroupData, rixFx, resourceManager ) );
              resourceParameterGroupData->update();
            }

            return resourceParameterGroupData;
          }

          ResourceParameterGroupDataRiXFx::ResourceParameterGroupDataRiXFx( const dp::sg::core::ParameterGroupDataSharedPtr& parameterGroupData, const dp::rix::fx::ManagerSharedPtr& rixFx, const ResourceManagerSharedPtr& resourceManager )
            : ResourceManager::Resource( reinterpret_cast<size_t>( parameterGroupData.operator->() ), resourceManager )   // Big Hack !!
            , m_parameterGroupData( parameterGroupData )
            , m_rixFx( rixFx )
          {
            DP_ASSERT( parameterGroupData );

            resourceManager->subscribe( this );

            m_groupData = m_rixFx->groupDataCreate( parameterGroupData->getParameterGroupSpec() );
          }

          ResourceParameterGroupDataRiXFx::~ResourceParameterGroupDataRiXFx()
          {
            if ( m_resourceManager )
            {
              m_resourceManager->unsubscribe( this );
            }
          }

          void ResourceParameterGroupDataRiXFx::update( )
          {
            if ( m_groupData )
            {
              std::vector<ResourceSamplerSharedPtr> resourceSamplers;

              const dp::fx::ParameterGroupSpecSharedPtr& groupSpec = m_parameterGroupData->getParameterGroupSpec();

              dp::fx::ParameterGroupSpec::iterator itEnd = groupSpec->endParameterSpecs();
              for (dp::fx::ParameterGroupSpec::iterator itps = groupSpec->beginParameterSpecs(); itps != itEnd; ++itps)
              {
                const dp::fx::ParameterSpec& spec = (*itps).first;
                if ( spec.getType() & dp::fx::PT_SAMPLER_TYPE_MASK )
                {
                  const dp::sg::core::SamplerSharedPtr& sampler = m_parameterGroupData->getParameter<dp::sg::core::SamplerSharedPtr>( itps );
                  DP_ASSERT( sampler ); // There must be a sampler with texture here or the renderer might fail validation with uninitialized sampler variables.

                  ResourceSamplerSharedPtr resourceSampler = ResourceSampler::get( sampler, m_resourceManager );
                  m_rixFx->groupDataSetValue( m_groupData, itps, dp::rix::core::ContainerDataSampler( resourceSampler->m_samplerHandle ) );
                  resourceSamplers.push_back( resourceSampler );
                }
                else
                {
                  // DAR HACK How to handle PT_BUFFER?
                  // DESIGN! This is the place where the data expansion would need to happen.
                  m_rixFx->groupDataSetValue( m_groupData, itps, dp::rix::core::ContainerDataRaw(0, m_parameterGroupData->getParameter(itps), spec.getSizeInByte()) );
                }

              }

              // keep referenced resources alive
              std::swap( m_resourceSamplers, resourceSamplers );
            }
          }

          bool ResourceParameterGroupDataRiXFx::update( const dp::util::Event& event )
          {
            const dp::sg::core::ParameterGroupData::Event& parameterEvent = static_cast<const dp::sg::core::ParameterGroupData::Event&>(event);
            const dp::fx::ParameterSpec& spec = parameterEvent.getParameter()->first;
            // support quick update for scalar types.
            if ( spec.getType() & dp::fx::PT_SCALAR_TYPE_MASK )
            {
              m_rixFx->groupDataSetValue( m_groupData, parameterEvent.getParameter()
                                        , dp::rix::core::ContainerDataRaw(0, m_parameterGroupData->getParameter(parameterEvent.getParameter()), spec.getSizeInByte()) );
              return true;
            }
            else
            {
              // textures require more management to keep the resources alive. Leave them out in the quick update path.
              return false;
            }
          }

          const dp::sg::core::HandledObjectSharedPtr& ResourceParameterGroupDataRiXFx::getHandledObject() const
          {
            return m_parameterGroupData.inplaceCast<dp::sg::core::HandledObject>();
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
