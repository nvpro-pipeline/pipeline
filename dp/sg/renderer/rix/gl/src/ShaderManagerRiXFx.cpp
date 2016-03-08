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


#include <dp/DP.h>
#include <dp/sg/renderer/rix/gl/inc/ShaderManager.h>
#include <dp/sg/renderer/rix/gl/inc/ShaderManagerRiXFx.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/fx/EffectSpec.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/util/Array.h>
#include <dp/util/File.h>
#include <iostream>

using namespace dp::fx;
using namespace dp::math;
using namespace dp::sg::xbar;

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

          /************************************************************************/
          /* ShaderManagerTransformsRiXFx                                         */
          /************************************************************************/
          class ShaderManagerTransformsRiXFx : public dp::util::Observer
          {
          public:
            ShaderManagerTransformsRiXFx( dp::sg::xbar::SceneTreeSharedPtr const& sceneTree, const ResourceManagerSharedPtr& resourceManager, dp::rix::fx::ManagerSharedPtr m_rixFxManager );
            virtual ~ShaderManagerTransformsRiXFx();

            virtual void updateTransforms();
            virtual dp::rix::fx::GroupDataSharedHandle getGroupData(dp::sg::xbar::TransformIndex);
            virtual dp::fx::EffectSpecSharedPtr getSystemSpec() { return m_effectSpecMatrices; }

            virtual void onNotify( dp::util::Event const & event, dp::util::Payload * payload);
            virtual void onDestroyed(dp::util::Subject const & subject, dp::util::Payload * payload);

          protected:
            dp::fx::EffectSpecSharedPtr             m_transformEffectSpec;

            typedef std::vector<dp::rix::fx::GroupDataSharedHandle> TransformGroupDatas;

            TransformGroupDatas                       m_transformGroupDatas;

            dp::sg::xbar::SceneTreeSharedPtr      m_sceneTree;
            ResourceManagerSharedPtr              m_resourceManager;
            dp::rix::fx::ManagerSharedPtr         m_rixFxManager;
            dp::fx::ParameterGroupSpecSharedPtr   m_groupSpecWorldMatrices;
            dp::fx::EffectSpecSharedPtr           m_effectSpecMatrices;

            dp::fx::ParameterGroupSpec::iterator  m_itWorldMatrix;
            dp::fx::ParameterGroupSpec::iterator  m_itWorldMatrixIT;

            dp::util::BitArray m_dirtyWorldMatrices; // world matrices which have been changed since the last update call
            dp::util::BitArray m_usedTransforms;     // transforms which are being used by the renderer

          private:
            void updateTransformNode(const dp::rix::fx::ManagerSharedPtr& manager, const dp::rix::fx::GroupDataSharedHandle& groupHandle, dp::math::Mat44f const & matrix);
          };


          ShaderManagerTransformsRiXFx::ShaderManagerTransformsRiXFx( SceneTreeSharedPtr const& sceneTree, const ResourceManagerSharedPtr& resourceManager, dp::rix::fx::ManagerSharedPtr rixFxManager )
            : m_sceneTree( sceneTree )
            , m_resourceManager( resourceManager )
            , m_rixFxManager( rixFxManager )
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            // create system group specs
            std::vector<dp::fx::ParameterSpec> parameterSpecs;
            parameterSpecs.push_back( ParameterSpec( "sys_WorldMatrix", PT_MATRIX4x4 | PT_FLOAT32, dp::util::Semantic::VALUE ) );
            parameterSpecs.push_back( ParameterSpec( "sys_WorldMatrixIT", PT_MATRIX4x4 | PT_FLOAT32, dp::util::Semantic::VALUE ) );
            m_groupSpecWorldMatrices = ParameterGroupSpec::create( "sys_WorldMatrices", parameterSpecs );

            m_itWorldMatrix   = m_groupSpecWorldMatrices->findParameterSpec( "sys_WorldMatrix" );
            m_itWorldMatrixIT = m_groupSpecWorldMatrices->findParameterSpec( "sys_WorldMatrixIT" );

            // create system effect specs
            dp::fx::EffectSpec::ParameterGroupSpecsContainer groupSpecs;
            groupSpecs.push_back(m_groupSpecWorldMatrices);
            m_effectSpecMatrices = dp::fx::EffectSpec::create( "sys_matrices", dp::fx::EffectSpec::Type::UNKNOWN, groupSpecs);

            m_sceneTree->getTransformTree().attach(this);
          }

          ShaderManagerTransformsRiXFx::~ShaderManagerTransformsRiXFx()
          {
            m_sceneTree->getTransformTree().detach(this);
          }

          void ShaderManagerTransformsRiXFx::onNotify(dp::util::Event const & event, dp::util::Payload * payload)
          {
            dp::sg::xbar::TransformTree::EventTransform const & eventTransform = static_cast<dp::sg::xbar::TransformTree::EventTransform const &>(event);
            dp::util::BitArray changedTransforms = eventTransform.getChangedWorldMatrices();
            changedTransforms.resize(m_dirtyWorldMatrices.getSize(), false);
            changedTransforms &= m_usedTransforms; // Remove the unused transforms from the bitmask. They don't need an update.
            m_dirtyWorldMatrices |= changedTransforms;
          }

          void ShaderManagerTransformsRiXFx::onDestroyed(dp::util::Subject const & subject, dp::util::Payload * payload)
          {
            DP_ASSERT(!"Should not happen!");
          }

          dp::rix::fx::GroupDataSharedHandle ShaderManagerTransformsRiXFx::getGroupData(dp::sg::xbar::TransformIndex transformIndex)
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            // keep same size as transform tree for a 1-1 mapping of transforms
            if ( m_transformGroupDatas.size() <= transformIndex )
            {
              m_transformGroupDatas.resize( m_sceneTree->getTransformTree().getTransforms().size());
              m_dirtyWorldMatrices.resize(m_transformGroupDatas.size());
              m_usedTransforms.resize(m_transformGroupDatas.size());
            }
            DP_ASSERT( transformIndex < m_transformGroupDatas.size() && "passed invalid transform index");

            if ( !m_transformGroupDatas[transformIndex] )
            {
              m_transformGroupDatas[transformIndex] = m_rixFxManager->groupDataCreate( m_groupSpecWorldMatrices );
              m_dirtyWorldMatrices.enableBit(transformIndex);
              m_usedTransforms.enableBit(transformIndex);
            }

            return m_transformGroupDatas[transformIndex];
          }

          inline void ShaderManagerTransformsRiXFx::updateTransformNode(const dp::rix::fx::ManagerSharedPtr& manager, const dp::rix::fx::GroupDataSharedHandle& groupHandle, dp::math::Mat44f const &worldMatrix)
          {
              // compute world matrices

              Mat44f inverseTranspose;
              dp::math::invertTranspose(worldMatrix, inverseTranspose);
              manager->groupDataSetValue(groupHandle, m_itWorldMatrix, dp::rix::core::ContainerDataRaw(0, &worldMatrix, sizeof(Mat44f)));
              manager->groupDataSetValue(groupHandle, m_itWorldMatrixIT, dp::rix::core::ContainerDataRaw(0, &inverseTranspose, sizeof(Mat44f)));
          }

          void ShaderManagerTransformsRiXFx::updateTransforms()
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            TransformTree::Transforms const & transforms = m_sceneTree->getTransformTree().getTransforms();
            m_dirtyWorldMatrices.traverseBits([&] (size_t index)
            {
              dp::math::Mat44f const & worldMatrix = transforms[index].world;
              updateTransformNode(m_rixFxManager, m_transformGroupDatas[index], worldMatrix);
            } );
            m_dirtyWorldMatrices.clear();
          }

          /************************************************************************/
          /* ShaderManagerRiXFxInstance                                           */
          /************************************************************************/
          DEFINE_PTR_TYPES( ShaderManagerRiXFxInstance );

          class ShaderManagerRiXFxInstance : public ShaderManagerInstance
          {
          public:
            static ShaderManagerRiXFxInstanceSharedPtr create();
          protected:
            ShaderManagerRiXFxInstance();

          public:
            dp::rix::fx::ProgramSharedHandle program;
            dp::rix::fx::InstanceSharedHandle instance;

            ResourceEffectDataRiXFxSharedPtr resourceEffectDataRiXFx;
          };

          ShaderManagerRiXFxInstanceSharedPtr ShaderManagerRiXFxInstance::create()
          {
            return( std::shared_ptr<ShaderManagerRiXFxInstance>( new ShaderManagerRiXFxInstance() ) );
          }

          ShaderManagerRiXFxInstance::ShaderManagerRiXFxInstance()
          {
          }

          /************************************************************************/
          /* ShaderManagerRiXFxRenderGroup                                        */
          /************************************************************************/
          DEFINE_PTR_TYPES( ShaderManagerRiXFxRenderGroup );

          class ShaderManagerRiXFxRenderGroup : public ShaderManagerRenderGroup
          {
          public:
            static ShaderManagerRiXFxRenderGroupSharedPtr create();
          protected:
            ShaderManagerRiXFxRenderGroup();

          public:
            dp::rix::fx::InstanceSharedHandle instance;
          };

          ShaderManagerRiXFxRenderGroupSharedPtr ShaderManagerRiXFxRenderGroup::create()
          {
            return( std::shared_ptr<ShaderManagerRiXFxRenderGroup>( new ShaderManagerRiXFxRenderGroup() ) );
          }

          ShaderManagerRiXFxRenderGroup::ShaderManagerRiXFxRenderGroup()
          {
          }

          ShaderManagerRiXFx::ShaderManagerRiXFx( SceneTreeSharedPtr const& sceneTree, dp::fx::Manager managerType, const ResourceManagerSharedPtr& resourceManager, TransparencyManagerSharedPtr const & transparencyManager, bool multicast )
            : ShaderManager( sceneTree, resourceManager, transparencyManager )
            , m_shaderManagerLights( sceneTree, resourceManager )
            , m_multicast(multicast)
          {
            // create RiXFx Manager
            m_rixFxManager = dp::rix::fx::Manager::create( managerType, m_renderer );

            m_shaderManagerTransforms.reset( new ShaderManagerTransformsRiXFx( sceneTree, resourceManager, m_rixFxManager ));
            m_systemSpecs[ m_shaderManagerTransforms->getSystemSpec()->getName() ] = dp::rix::fx::Manager::EffectSpecInfo( m_shaderManagerTransforms->getSystemSpec(), false );

            // create GroupSpec for Camera
            std::vector<dp::fx::ParameterSpec> parameterSpecs;
            parameterSpecs.push_back( ParameterSpec( "sys_ViewProjMatrix", PT_MATRIX4x4 | PT_FLOAT32, dp::util::Semantic::VALUE ) );
            parameterSpecs.push_back( ParameterSpec( "sys_ViewMatrixI",    PT_MATRIX4x4 | PT_FLOAT32, dp::util::Semantic::VALUE ) );
            dp::fx::ParameterGroupSpecSharedPtr groupSpecCamera = ParameterGroupSpec::create( "sys_camera", parameterSpecs, m_multicast );

            // Get iterators for fast updates later
            m_itViewProjMatrix = groupSpecCamera->findParameterSpec( "sys_ViewProjMatrix" );
            m_itViewMatrixI    = groupSpecCamera->findParameterSpec( "sys_ViewMatrixI" );

            // generate EffectSpec for Camera and register as SystemSpec
            dp::fx::EffectSpec::ParameterGroupSpecsContainer groupSpecs;
            groupSpecs.push_back(groupSpecCamera);
            m_effectSpecCamera = dp::fx::EffectSpec::create( "sys_camera", dp::fx::EffectSpec::Type::UNKNOWN, groupSpecs);
            m_systemSpecs["sys_camera"] = dp::rix::fx::Manager::EffectSpecInfo( m_effectSpecCamera, true );

            // create a GroupData for the Camera
            m_groupDataCamera = m_rixFxManager->groupDataCreate( groupSpecCamera );

            std::vector<dp::rix::core::ProgramParameter> programParametersFragment;
            programParametersFragment.push_back( dp::rix::core::ProgramParameter( "sys_EnvironmentSampler",        dp::rix::core::ContainerParameterType::SAMPLER ) );
            programParametersFragment.push_back( dp::rix::core::ProgramParameter( "sys_EnvironmentSamplerEnabled", dp::rix::core::ContainerParameterType::BOOL ) );
            programParametersFragment.push_back( dp::rix::core::ProgramParameter( "sys_ViewportSize",              dp::rix::core::ContainerParameterType::UINT2_32 ) );
            getTransparencyManager()->addFragmentParameters( programParametersFragment );
            m_descriptorFragment = m_renderer->containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptorCommon( programParametersFragment.data(), programParametersFragment.size() ) );
            m_containerFragment = m_renderer->containerCreate( m_descriptorFragment );

            std::vector<dp::fx::ParameterSpec> fragmentSpecs;
            fragmentSpecs.push_back( ParameterSpec( "sys_EnvironmentSampler", PT_SAMPLER_PTR | PT_SAMPLER_2D, dp::util::Semantic::VALUE ) );
            fragmentSpecs.push_back( ParameterSpec( "sys_EnvironmentSamplerEnabled", PT_BOOL, dp::util::Semantic::VALUE ) );
            fragmentSpecs.push_back( ParameterSpec( "sys_ViewportSize", PT_VECTOR2 | PT_UINT32, dp::util::Semantic::VALUE ) );
            getTransparencyManager()->addFragmentParameterSpecs( fragmentSpecs );
            dp::fx::ParameterGroupSpecSharedPtr fragmentGroupSpec = dp::fx::ParameterGroupSpec::create( "sys_FragmentParameters", fragmentSpecs );

            dp::fx::EffectSpec::ParameterGroupSpecsContainer fragmentGroupSpecs;
            fragmentGroupSpecs.push_back( fragmentGroupSpec );
            m_fragmentSystemSpec = dp::fx::EffectSpec::create( "sys_Fragment", dp::fx::EffectSpec::Type::SYSTEM, fragmentGroupSpecs );

            m_systemSpecs["sys_Fragment"] = dp::rix::fx::Manager::EffectSpecInfo( m_fragmentSystemSpec, true );

            // get snippets for opaque|transparent and color|depth passes
            getTransparencyManager()->addFragmentCodeSnippets( false, false, m_additionalCodeSnippets[RGL_OPAQUE][static_cast<size_t>(RenderGroupPass::FORWARD)][dp::fx::Domain::FRAGMENT] );
            getTransparencyManager()->addFragmentCodeSnippets( false, true,  m_additionalCodeSnippets[RGL_OPAQUE][static_cast<size_t>(RenderGroupPass::DEPTH)][dp::fx::Domain::FRAGMENT] );
            getTransparencyManager()->addFragmentCodeSnippets( true, false, m_additionalCodeSnippets[RGL_TRANSPARENT][static_cast<size_t>(RenderGroupPass::FORWARD)][dp::fx::Domain::FRAGMENT] );
            getTransparencyManager()->addFragmentCodeSnippets( true, true,  m_additionalCodeSnippets[RGL_TRANSPARENT][static_cast<size_t>(RenderGroupPass::DEPTH)][dp::fx::Domain::FRAGMENT] );
          }

          void ShaderManagerRiXFx::updateCameraState(std::vector<dp::sg::core::CameraSharedPtr> const & cameras)
          {
            for (size_t index = 0;index < cameras.size(); ++index)
            {
              std::shared_ptr<dp::sg::core::Camera> const & camera = cameras[index];

              dp::math::Mat44f worldToProj = camera->getWorldToViewMatrix() * camera->getProjection();
              dp::math::Mat44f viewToWorld = camera->getViewToWorldMatrix();

              m_rixFxManager->groupDataSetValue(m_groupDataCamera, m_itViewProjMatrix, dp::rix::core::ContainerDataRaw(0, &worldToProj, sizeof(worldToProj), uint32_t(index)));
              m_rixFxManager->groupDataSetValue(m_groupDataCamera, m_itViewMatrixI, dp::rix::core::ContainerDataRaw(0, &viewToWorld, sizeof(viewToWorld), uint32_t(index)));
            }

            // TODO don't call from here, but at a generic update call
            updateTransforms();
          }

          void ShaderManagerRiXFx::updateFragmentParameter( std::string const & name, dp::rix::core::ContainerDataRaw const & data )
          {
            dp::rix::core::ContainerEntry entry = m_renderer->containerDescriptorGetEntry( m_descriptorFragment, name.c_str() );
            if ( entry != ~0 )
            {
              m_renderer->containerSetData( m_containerFragment, entry, data );
            }
          }

          void ShaderManagerRiXFx::updateFragmentParameter( std::string const &name, dp::rix::core::ContainerDataSampler const & data )
          {
            dp::rix::core::ContainerEntry entry = m_renderer->containerDescriptorGetEntry( m_descriptorFragment, name.c_str() );
            if ( entry != ~0 )
            {
              m_renderer->containerSetData( m_containerFragment, entry, data );
            }
          }

          void ShaderManagerRiXFx::updateTransforms()
          {
            m_shaderManagerTransforms->updateTransforms();
            m_rixFxManager->runPendingUpdates();
          }

          ShaderManagerRenderGroupSharedPtr ShaderManagerRiXFx::registerRenderGroup( dp::rix::core::RenderGroupSharedHandle const & renderGroup )
          {
            ShaderManagerRiXFxRenderGroupSharedPtr rixFxRenderGroup = ShaderManagerRiXFxRenderGroup::create();
            rixFxRenderGroup->renderGroup = renderGroup;
            rixFxRenderGroup->instance = m_rixFxManager->instanceCreate( renderGroup.get() );
            addSystemContainers(std::static_pointer_cast<ShaderManagerRenderGroup>(rixFxRenderGroup));

            return rixFxRenderGroup;
          }

          void ShaderManagerRiXFx::addSystemContainers( ShaderManagerInstanceSharedPtr const & shaderObject )
          {
            const dp::sg::xbar::ObjectTreeNode &objectTreeNode = m_sceneTree.lock()->getObjectTreeNode( shaderObject->objectTreeIndex );
            ShaderManagerRiXFxInstanceSharedPtr o = std::static_pointer_cast<ShaderManagerRiXFxInstance>(shaderObject);
            m_rixFxManager->instanceUseGroupData(o->instance.get(), m_shaderManagerTransforms->getGroupData(objectTreeNode.m_transform).get());
          }

          void ShaderManagerRiXFx::addSystemContainers( ShaderManagerRenderGroupSharedPtr const & renderGroup )
          {
            ShaderManagerRiXFxRenderGroupSharedPtr o = std::static_pointer_cast<ShaderManagerRiXFxRenderGroup>(renderGroup);

            m_rixFxManager->instanceUseGroupData( o->instance.get(), m_groupDataCamera.get() );
            m_renderer->renderGroupUseContainer( renderGroup->renderGroup, m_containerFragment );
            m_renderer->renderGroupUseContainer( renderGroup->renderGroup, m_shaderManagerLights.getLightInformation().m_container );
          }

          void ShaderManagerRiXFx::updateLights( dp::sg::ui::ViewStateSharedPtr const& vs )
          {
            m_shaderManagerLights.updateLights( vs );
          }

          ShaderManagerInstanceSharedPtr ShaderManagerRiXFx::registerGeometryInstance(
            const dp::sg::core::PipelineDataSharedPtr &pipelineData,
            dp::sg::xbar::ObjectTreeIndex objectTreeIndex,
            dp::rix::core::GeometryInstanceSharedHandle &geometryInstance,
            RenderPassType rpt )
          {
            ResourceEffectDataRiXFxSharedPtr resourceEffectData = ResourceEffectDataRiXFx::get( pipelineData, m_rixFxManager, m_resourceManager );
            if ( resourceEffectData )
            {
              // use the EffectSpecs to determine potential transparency
              dp::fx::EffectSpecSharedPtr const & effectSpec = pipelineData->getEffectSpec();

              dp::rix::fx::SourceFragments const& sf = m_additionalCodeSnippets[effectSpec->getTransparent()][rpt==RenderPassType::DEPTH];
              dp::rix::fx::ProgramSharedHandle program = m_rixFxManager->programCreate( effectSpec, m_systemSpecs
                                                                                      , ( rpt == RenderPassType::FORWARD ) ? "forward" : "depthPass"
                                                                                      , nullptr, 0, m_additionalCodeSnippets[effectSpec->getTransparent()][rpt==RenderPassType::DEPTH] ); // no user descriptors
              ShaderManagerRiXFxInstanceSharedPtr o;
              if ( program )
              {
                o = ShaderManagerRiXFxInstance::create();
                o->geometryInstance = geometryInstance;
                o->objectTreeIndex = objectTreeIndex;
                o->resourceEffectDataRiXFx = resourceEffectData;

                o->program = program;
                o->instance = m_rixFxManager->instanceCreate( o->geometryInstance.get() );
                m_rixFxManager->instanceSetProgram( o->instance.get(), o->program.get() );

                std::vector<dp::rix::fx::GroupDataSharedHandle> groupDatas;

                ResourceEffectDataRiXFx::GroupDatas gec = resourceEffectData->getGroupDatas();
                std::copy( gec.begin(), gec.end(), std::back_inserter(groupDatas) ) ;

                addSystemContainers(std::static_pointer_cast<ShaderManagerInstance>(o));
                for (size_t i = 0; i < groupDatas.size(); i++)
                {
                  m_rixFxManager->instanceUseGroupData( o->instance.get(), groupDatas[i].get() );
                }
              }
              return o;
            }

            // no supported effect, use the default one.
            DP_ASSERT( pipelineData != m_defaultPipelineData );
            return registerGeometryInstance( m_defaultPipelineData, objectTreeIndex, geometryInstance, rpt );
          }

          std::map<dp::fx::Domain,std::string> ShaderManagerRiXFx::getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const
          {
            DP_ASSERT( m_rixFxManager );
            const dp::sg::core::PipelineDataSharedPtr & pipelineData = geoNode->getMaterialPipeline() ? geoNode->getMaterialPipeline() : m_defaultPipelineData;

            // use the EffectSpecs to determine potential transparency
            dp::fx::EffectSpecSharedPtr const & effectSpec = pipelineData->getEffectSpec();

            return( m_rixFxManager->getShaderSources( effectSpec, depthPass, m_systemSpecs, m_additionalCodeSnippets[effectSpec->getTransparent()][depthPass] ) );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
