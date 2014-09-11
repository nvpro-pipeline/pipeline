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


#include <dp/sg/renderer/rix/gl/inc/ResourceEffectDataRiXFx.h>

#include <dp/sg/core/CoreTypes.h>

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
          SmartResourceEffectDataRiXFx ResourceEffectDataRiXFx::get( const dp::sg::core::EffectDataSharedPtr &effectData, const dp::rix::fx::SmartManager& rixFx, const SmartResourceManager& resourceManager )
          {
            assert( effectData );
            assert( !!resourceManager );

            SmartResourceEffectDataRiXFx resourceEffectData = resourceManager->getResource<ResourceEffectDataRiXFx>( reinterpret_cast<size_t>(effectData.getWeakPtr()) );
            if ( !resourceEffectData )
            {
              resourceEffectData = new ResourceEffectDataRiXFx( effectData, rixFx, resourceManager );
              resourceEffectData->update();
            }

            return resourceEffectData;
          }

          ResourceEffectDataRiXFx::ResourceEffectDataRiXFx( const dp::sg::core::EffectDataSharedPtr &effectData, const dp::rix::fx::SmartManager& rixFx, const SmartResourceManager& resourceManager )
            : ResourceManager::Resource( reinterpret_cast<size_t>( effectData.getWeakPtr() ), resourceManager )
            , m_rixFx( rixFx )
            , m_effectData( effectData )
          {
            DP_ASSERT( effectData );

            resourceManager->subscribe( this );
          }

          ResourceEffectDataRiXFx::~ResourceEffectDataRiXFx()
          {
            if ( m_resourceManager )
            {
              m_resourceManager->unsubscribe( this );
            }
          }

          void ResourceEffectDataRiXFx::update( )
          {
            const dp::fx::SmartEffectSpec& effectSpec = m_effectData->getEffectSpec();
            m_resourceEffectSpec = ResourceEffectSpecRiXFx::get( effectSpec, m_rixFx, m_resourceManager );

            std::vector<SmartResourceParameterGroupDataRiXFx> newResourceParameterGroupDataRiXFxs;
            for ( dp::fx::EffectSpec::iterator it = effectSpec->beginParameterGroupSpecs(); it != effectSpec->endParameterGroupSpecs(); ++it )
            {
              dp::sg::core::ParameterGroupDataSharedPtr const& parameterGroupData = m_effectData->getParameterGroupData(it);

              if ( parameterGroupData )
              {
                newResourceParameterGroupDataRiXFxs.push_back( ResourceParameterGroupDataRiXFx::get( parameterGroupData, m_rixFx, m_resourceManager ) );
              }
              else
              {
                newResourceParameterGroupDataRiXFxs.push_back( m_resourceEffectSpec->getDefaultParameterGroupDataResource( it ) );
              }
            }
            m_resourceParameterGroupDataRiXFxs.swap( newResourceParameterGroupDataRiXFxs );
          }

          ResourceEffectDataRiXFx::GroupDatas ResourceEffectDataRiXFx::getGroupDatas() const
          {
            GroupDatas groupDatas;

            for ( size_t idx = 0;idx < m_resourceParameterGroupDataRiXFxs.size(); ++idx )
            {
              // TODO this is currently required since it may happen that groups do not create a container for unsupported ParameterGroupSpecs
              dp::rix::fx::GroupDataSharedHandle const& handle = m_resourceParameterGroupDataRiXFxs[idx]->getGroupData();
              if( handle )
              {
                groupDatas.push_back( handle );
              }
            }

            return groupDatas;
          }


          const dp::sg::core::HandledObjectSharedPtr& ResourceEffectDataRiXFx::getHandledObject() const
          {
            return m_effectData.inplaceCast<dp::sg::core::HandledObject>();
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
