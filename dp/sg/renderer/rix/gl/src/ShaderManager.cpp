// Copyright NVIDIA Corporation 2011-2012
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


#include <dp/sg/renderer/rix/gl/inc/ShaderManager.h>

#include <dp/fx/EffectSpec.h>
#include <dp/fx/EffectLibrary.h>

#include <dp/sg/core/Camera.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/util/Array.h>

using namespace dp::fx;
using namespace dp::math;
using namespace dp::sg::xbar;
using dp::util::array;

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

          ShaderManager::ShaderManager( SceneTree* sceneTree, const SmartResourceManager& resourceManager, SmartTransparencyManager const & transparencyManager )
            : m_sceneTree( sceneTree )
            , m_resourceManager( resourceManager )
            , m_environmentNeedsUpdate( false )
            , m_environmentSampler( nullptr )
            , m_transparencyManager( transparencyManager )
          {
            m_renderer = resourceManager->getRenderer();
            m_defaultEffectData = dp::sg::core::createStandardMaterialData();
            m_transparencyManager->setShaderManager( this );
          }

          void ShaderManager::setEnvironmentSampler( const dp::sg::core::SamplerSharedPtr & sampler )
          {
            if ( m_environmentSampler != sampler )
            {
              m_environmentSampler = sampler;
              m_environmentResourceSampler = sampler ? ResourceSampler::get( sampler, m_resourceManager ) : nullptr;

              m_environmentNeedsUpdate = true;
            }
          }

          const dp::sg::core::SamplerSharedPtr & ShaderManager::getEnvironmentSampler() const
          {
            return( m_environmentSampler );
          }

          SmartTransparencyManager const & ShaderManager::getTransparencyManager() const
          {
            return( m_transparencyManager );
          }

          SmartShaderManagerInstance ShaderManager::registerGeometryInstance( const dp::sg::core::GeoNodeSharedPtr &geoNode,
                                                                              dp::sg::xbar::ObjectTreeIndex objectTreeIndex,
                                                                              dp::rix::core::GeometryInstanceSharedHandle &geometryInstance,
                                                                              RenderPassType rpt )
          {
            const dp::sg::core::EffectDataSharedPtr& effectData = geoNode->getMaterialEffect() ? geoNode->getMaterialEffect() : m_defaultEffectData;

            return registerGeometryInstance( effectData, objectTreeIndex, geometryInstance, rpt );
          }

          SmartShaderManagerInstance ShaderManager::registerGeometryInstance( const dp::sg::core::EffectDataSharedPtr &effectData,
                                                                              dp::sg::xbar::ObjectTreeIndex objectTreeIndex,
                                                                              dp::rix::core::GeometryInstanceSharedHandle &geometryInstance,
                                                                              RenderPassType rpt )
          {
            DP_ASSERT( !"should not hit this path");
            return SmartShaderManagerInstance::null;
          }

          void ShaderManager::update( dp::sg::ui::ViewStateSharedPtr const& viewState )
          {
            if ( m_environmentNeedsUpdate )
            {
              updateEnvironment( m_environmentResourceSampler );
              m_environmentNeedsUpdate = false;
            }
            updateLights( viewState);
            dp::sg::core::CameraSharedPtr const& camera = viewState->getCamera();
            updateCameraState( camera->getWorldToViewMatrix() * camera->getProjection(), camera->getViewToWorldMatrix() );
          }

          /************************************************************************/
          /* ShaderManagerLights                                                  */
          /************************************************************************/

          // LightState
          struct ShaderLight {
            Vec4f ambient;             
            Vec4f diffuse;             
            Vec4f specular;            
            Vec4f position;            
            Vec4f direction;           
            float spotExponent;        
            float spotCutoff;          
            float constantAttenuation; 
            float linearAttenuation;   
            float quadraticAttenuation;
            float _pad0[3];
          };

          struct ShaderLightState {
            Vec3f       sys_SceneAmbientLight;
            int         numLights;
            ShaderLight lights[MAXLIGHTS];
          };

          inline bool copyLight( ShaderLight& light, const dp::sg::core::LightSourceSharedPtr& ls, const Mat44f& world )
          {
            bool copied = false;
            if( ls->isEnabled() )
            {
              DP_ASSERT( ls->getLightEffect() );
              dp::sg::core::EffectDataSharedPtr const& le = ls->getLightEffect();
              const SmartEffectSpec & es = le->getEffectSpec();
              for ( EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
              {
                const dp::sg::core::ParameterGroupDataSharedPtr & parameterGroupData = le->getParameterGroupData( it );
                if ( parameterGroupData )
                {
                  std::string name = (*it)->getName();
                  if ( ( name == "standardDirectedLightParameters" )
                    || ( name == "standardPointLightParameters" )
                    || ( name == "standardSpotLightParameters" ) )
                  {
                    const Mat44f w = world;
                    const SmartParameterGroupSpec & pgs = parameterGroupData->getParameterGroupSpec();
                    light.ambient = Vec4f( parameterGroupData->getParameter<Vec3f>( pgs->findParameterSpec( "ambient" ) ), 1.0f );
                    light.diffuse = Vec4f( parameterGroupData->getParameter<Vec3f>( pgs->findParameterSpec( "diffuse" ) ), 1.0f );
                    light.specular = Vec4f( parameterGroupData->getParameter<Vec3f>( pgs->findParameterSpec( "specular" ) ), 1.0f );
                    if ( name == "standardDirectedLightParameters" )
                    {
                      light.position = Vec4f( - parameterGroupData->getParameter<Vec3f>( pgs->findParameterSpec( "direction" ) ), 0.0f ) * w;
                    }
                    else
                    {
                      light.position = Vec4f( parameterGroupData->getParameter<Vec3f>( pgs->findParameterSpec( "position" ) ), 1.0f ) * w;
                      std::vector<float> attenuations;
                      parameterGroupData->getParameterArray( pgs->findParameterSpec( "attenuations" ), attenuations );
                      DP_ASSERT( attenuations.size() == 3 );
                      light.constantAttenuation = attenuations[0];
                      light.linearAttenuation = attenuations[1];
                      light.quadraticAttenuation = attenuations[2];
                      if ( name == "standardPointLightParameters" )
                      {
                        light.direction = Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
                        light.spotExponent = 0.0f;
                        light.spotCutoff = 180.f;
                      }
                      else
                      {
                        light.direction = Vec4f( parameterGroupData->getParameter<Vec3f>( pgs->findParameterSpec( "direction" ) ), 0.0f ) * w;
                        light.spotExponent = parameterGroupData->getParameter<float>( pgs->findParameterSpec( "exponent" ) );
                        light.spotCutoff = parameterGroupData->getParameter<float>( pgs->findParameterSpec( "cutoff" ) );
                      }
                    }
                    copied = true;
                    break;
                  }
                }
              }
            }

            return( copied );
          }

          ShaderManagerLights::ShaderManagerLights( SceneTree *sceneTree, const SmartResourceManager& resourceManager )
            : m_sceneTree( sceneTree )
            , m_resourceManager( resourceManager )
          {
            // create light descriptor
            dp::rix::core::ProgramParameter programParametersLight[] = 
            {
              dp::rix::core::ProgramParameter( "sys_LightsBuffer", dp::rix::core::CPT_BUFFER, 0 )
            };

            m_descriptorLight = m_resourceManager->getRenderer()->containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptorCommon( programParametersLight, sizeof array(programParametersLight) ) );
          }

          ShaderManagerLights::~ShaderManagerLights()
          {

          }

          ShaderManagerLights::LightInformation ShaderManagerLights::getLightInformation()
          {
            if( !m_lightInformation.m_buffer )
            {
              dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

              DP_ASSERT( !m_lightInformation.m_container );

              ShaderLightState shaderLightState;
              shaderLightState.numLights = 0;
              shaderLightState.sys_SceneAmbientLight = Vec3f( 0.0f, 0.0f, 0.0f );

              dp::rix::core::BufferSharedHandle buffer = renderer->bufferCreate();
              renderer->bufferSetSize( buffer, sizeof( ShaderLightState ) );
              renderer->bufferUpdateData( buffer, 0, &shaderLightState, sizeof( ShaderLightState ) ); 

              dp::rix::core::ContainerSharedHandle container = renderer->containerCreate( m_descriptorLight );
              dp::rix::core::ContainerEntry  entry     = renderer->containerDescriptorGetEntry( m_descriptorLight, "sys_LightsBuffer" );

              renderer->containerSetData( container, entry, dp::rix::core::ContainerDataBuffer( buffer ) );

              m_lightInformation.m_container = container;
              m_lightInformation.m_buffer    = buffer;
            }

            return m_lightInformation;
          }


          void ShaderManagerLights::updateLights( dp::sg::ui::ViewStateSharedPtr const& vs )
          {
            dp::sg::core::CameraSharedPtr const& camera = vs->getCamera();
            Trafo trafo;
            trafo.setTranslation( camera->getPosition() );
            
            dp::sg::core::SceneSharedPtr scene = vs->getScene();
            Vec3f ambientColor = scene->getAmbientColor();

            ShaderLightState lightState;
            unsigned int lightId = 0;

            lightState.sys_SceneAmbientLight = ambientColor;

            // copy camera headlights
            dp::sg::core::Camera::HeadLightIterator cam_it, cam_it_end = camera->endHeadLights();
            for ( cam_it = camera->beginHeadLights(); cam_it != cam_it_end; ++cam_it )
            {
              ShaderLight &light = lightState.lights[lightId];
              if ( copyLight( light, *cam_it, trafo.getMatrix() ) )
              {
                ++lightId;
              }
            }

            // copy scene lights
            TransformTree const& transformTree = m_sceneTree->getTransformTree();
            const std::set< ObjectTreeIndex >& lightSources = m_sceneTree->getLightSources();
            for ( std::set< ObjectTreeIndex >::const_iterator itLight = lightSources.begin(); itLight != lightSources.end() && lightId < MAXLIGHTS; ++itLight )
            {
              ObjectTreeNode& otn = m_sceneTree->getObjectTreeNode(*itLight);

              DP_ASSERT( dynamic_cast<dp::sg::core::LightSourceWeakPtr>(otn.m_object) );
              dp::sg::core::LightSourceSharedPtr ls = otn.m_object->getSharedPtr<dp::sg::core::LightSource>();

              ShaderLight &light = lightState.lights[lightId];

              if ( copyLight( light, ls, transformTree[otn.m_transformIndex].m_worldMatrix ) )
              {
                ++lightId;
              }
            }

            lightState.numLights = lightId;

            dp::rix::core::Renderer* renderer = m_resourceManager->getRenderer();
            renderer->bufferUpdateData( m_lightInformation.m_buffer, 0, &lightState, sizeof(ShaderLightState)  );
          }


          ShaderManagerTransforms::~ShaderManagerTransforms()
          {

          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp


