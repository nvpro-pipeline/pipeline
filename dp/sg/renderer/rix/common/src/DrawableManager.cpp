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


#include <dp/sg/renderer/rix/gl/inc/DrawableManagerDefault.h>
#include <dp/sg/renderer/rix/gl/inc/ShaderManagerRiXFx.h>
#include <dp/sg/xbar/SceneTree.h>
#include <dp/culling/cpu/Manager.h>
#include <dp/culling/opengl/Manager.h>
#include <dp/culling/cuda/Manager.h>

#include <dp/sg/core/Camera.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/ui/ViewState.h>

#include <dp/gl/RenderContextGL.h>

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
          /* DrawableManagerDefault::TransformObserfver                           */
          /************************************************************************/
          DrawableManagerDefault::TransformObserver::TransformObserver( DrawableManagerDefault* drawableManager )
            : m_drawableManager( drawableManager )
          {
            m_drawableManager->getSceneTree()->attach( this );
          }

          DrawableManagerDefault::TransformObserver::~TransformObserver()
          {
            m_drawableManager->getSceneTree()->detach( this );
          }

          void DrawableManagerDefault::TransformObserver::onNotify( dp::util::Event const& event, dp::util::Payload* payload )
          {
            DP_ASSERT( dynamic_cast<dp::sg::xbar::SceneTree::Event const*>(&event) );

            dp::sg::xbar::SceneTree::Event const& xbarEvent = static_cast<dp::sg::xbar::SceneTree::Event const&>(event);
            if ( xbarEvent.getType() == dp::sg::xbar::SceneTree::Event::Transform )
            {
              m_drawableManager->onTransformChanged( static_cast<dp::sg::xbar::SceneTree::EventTransform const&>( event ) );
            }
          }

          void DrawableManagerDefault::TransformObserver::onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload )
          {
            DP_ASSERT( !"shouldn't get called" );
          }

          /************************************************************************/
          /* DrawableManagerDefault::EffectDataObserver                           */
          /************************************************************************/
          DrawableManagerDefault::EffectDataObserver::EffectDataObserver( DrawableManagerDefault* drawableManager )
            : m_drawableManager( drawableManager )
          {
          }

          DrawableManagerDefault::EffectDataObserver::~EffectDataObserver()
          {
          }

          void DrawableManagerDefault::EffectDataObserver::onNotify( dp::util::Event const& event, dp::util::Payload* payload )
          {
            DP_ASSERT( dynamic_cast<Payload*>(payload) );
            Payload* p = static_cast<Payload*>(payload);
            if ( p )
            {
              m_drawableManager->updateDrawableInstance( p->getHandle() );
            }
          }

          void DrawableManagerDefault::EffectDataObserver::onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload )
          {
            DP_ASSERT( !"shouldn't get called" );
          }


          /************************************************************************/
          /* DrawableManagerDefault::Instance                                     */
          /************************************************************************/
          DrawableManagerDefault::Instance::Instance()
            : m_isVisible(true)
            , m_isActive(true)
            , m_isTraversalActive(true)
            , m_activeTraversalMask(~0)
            , m_objectTreeIndex( ~0 )
            , m_handle( nullptr )
            , m_currentEffectSurface( nullptr )
            , m_currentEffectGeometry( nullptr )
            , m_effectDataAttached( false )
          {
            m_payload.reset( new Payload() );
          }

          inline void DrawableManagerDefault::Instance::updateRendererVisibility( dp::rix::core::Renderer* renderer )
          {
            renderer->geometryInstanceSetVisible( m_geometryInstance, m_isTraversalActive && m_isActive && m_isVisible );
            if (m_geometryInstanceDepthPass)
            {
              renderer->geometryInstanceSetVisible( m_geometryInstanceDepthPass, m_isTraversalActive && m_isActive && m_isVisible );
            }
          }

          /************************************************************************/
          /* DrawableManagerDefault                                               */
          /************************************************************************/
          DrawableManagerDefault::DrawableManagerDefault( const SmartResourceManager & resourceManager
                                                        , SmartTransparencyManager const & transparencyManager
                                                        , dp::fx::Manager shaderManagerType
                                                        , dp::culling::Mode cullingMode )
            : m_sceneTree( nullptr )
            , m_resourceManager( resourceManager )
            , m_shaderManager( nullptr )
            , m_shaderManagerType( shaderManagerType )
            , m_cullingMode( cullingMode)
            , m_cullingManager( nullptr )
            , m_cullingEnabled( true )
            , m_activeTraversalMask( ~0 )
            , m_viewportSize( 0, 0 )
            , m_transparencyManager( transparencyManager )
          {
          }

          DrawableManagerDefault::~DrawableManagerDefault()
          {
            detachEffectDataObserver();
            m_transformObserver.reset();
            m_effectDataObserver.reset();
          }

          void DrawableManagerDefault::setSceneTree( SceneTree* sceneTree )
          {
            if ( sceneTree != m_sceneTree )
            {
              detachEffectDataObserver();
              m_transformObserver.reset();
              m_effectDataObserver.reset();

              m_cullingGroup.reset();
              m_cullingResult.reset();
              m_shaderManager.reset();
              m_instances.clear();

              m_sceneTree = sceneTree;

              if ( m_sceneTree )
              {
                dp::culling::Mode cullingMode = m_cullingMode;

                // determine if OpenGL 4.3 is available
                if ( m_cullingMode == dp::culling::MODE_AUTO )
                {
                  cullingMode = dp::culling::MODE_CUDA;
                }
                switch ( cullingMode )
                {
                case dp::culling::MODE_CPU:
                  m_cullingManager.reset(dp::culling::cpu::Manager::create());
                  break;
                case dp::culling::MODE_OPENGL_COMPUTE:
                  m_cullingManager.reset(dp::culling::opengl::Manager::create());
                  break;
                case dp::culling::MODE_CUDA:
                  m_cullingManager.reset(dp::culling::cuda::Manager::create());
                  break;
                default:
                  DP_ASSERT( !"unknown culling mode, falling back to CPU version" );
                  m_cullingManager.reset(dp::culling::cpu::Manager::create());
                }
                m_cullingGroup = m_cullingManager->groupCreate();
                m_cullingResult = m_cullingManager->groupCreateResult( m_cullingGroup );

                TransformTree const& transformTree = m_sceneTree->getTransformTree();
                m_cullingManager->groupSetMatrices( m_cullingGroup, &transformTree[0].m_worldMatrix, transformTree.size(), sizeof( transformTree[0]) );

                switch ( m_shaderManagerType )
                {
                case dp::fx::MANAGER_UNIFORM:
                case dp::fx::MANAGER_SHADERBUFFER:
                case dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX:
                case dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX:
                case dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT:
                  m_shaderManager.reset( new ShaderManagerRiXFx( sceneTree, m_shaderManagerType, m_resourceManager, m_transparencyManager ) );
                  break;
                default:
                  m_shaderManager.reset( new ShaderManagerRiXFx( sceneTree, dp::fx::MANAGER_UNIFORM, m_resourceManager, m_transparencyManager ) );
                }

                dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
                m_renderGroup = renderer->renderGroupCreate();
                m_renderGroupDepthPass = renderer->renderGroupCreate();
                m_renderGroupTransparent = renderer->renderGroupCreate();

                m_shaderManager->setEnvironmentSampler( m_environmentSampler );
                m_shaderManager->addSystemContainers( m_renderGroup );
                m_shaderManager->addSystemContainers( m_renderGroupDepthPass );
                m_shaderManager->addSystemContainers( m_renderGroupTransparent );
                m_shaderManager->updateFragmentParameter( std::string( "sys_ViewportSize" ), dp::rix::core::ContainerDataRaw( 0, &m_viewportSize[0], sizeof( dp::math::Vec2ui ) ) );
                m_transparencyManager->updateFragmentParameters();

                // Observe Transform of SceneTree
                m_transformObserver.reset( new TransformObserver( this ) );
                m_effectDataObserver.reset( new EffectDataObserver( this ) );
              }
            }
          }

          DrawableManager::Handle DrawableManagerDefault::addDrawableInstance( dp::sg::core::GeoNodeWeakPtr geoNode, ObjectTreeIndex objectTreeIndex )
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            ObjectTreeNode objectTreeNode = m_sceneTree->getObjectTreeNode( objectTreeIndex );

            // generate a new handle and fill it
            DefaultHandleDataHandle handle = new DefaultHandleData( dp::sg::xbar::ObjectTreeIndex(m_instances.size()) );

            // generate a new instance
            m_instances.push_back( Instance() );
            Instance &di = m_instances.back();
            di.m_handle = handle;
            di.m_objectTreeIndex = objectTreeIndex;
            di.m_transformIndex = objectTreeNode.m_transformIndex;
            di.m_transparent = false;
            di.m_currentRenderGroup = nullptr;
            di.m_isVisible = true; // gis are visible by default
            di.m_payload->setHandle( handle );
            di.m_effectDataAttached = false;

            di.m_cullingObject = m_cullingManager->objectCreate( handle );
            m_cullingManager->objectSetUserData( di.m_cullingObject, di.m_handle );
            m_cullingManager->groupAddObject( m_cullingGroup, di.m_cullingObject );

            // create geometry instance
            di.m_geometryInstance = renderer->geometryInstanceCreate();
            dp::sg::core::GeoNodeLock gnrl(geoNode);

            bool isTransparent = false;
            bool foundGLSL = true;

            // update material & primitive
            updateDrawableInstance( handle );

            return handle;
          }

          void DrawableManagerDefault::removeDrawableInstance( DrawableManager::Handle handle )
          {
            const DefaultHandleDataHandle& handleData = dp::util::smart_cast<DefaultHandleData>( handle );

            // TODO really remove data
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            Instance& di = m_instances[handleData->m_index];

            if( di.m_effectDataAttached )
            {
              di.m_currentEffectGeometry->detach( m_effectDataObserver.get(), di.m_payload.get() );
              di.m_currentEffectSurface->detach( m_effectDataObserver.get(), di.m_payload.get() );
              di.m_effectDataAttached = false;
            }

            if ( di.m_currentRenderGroup )
            {
              renderer->renderGroupRemoveGeometryInstance( di.m_currentRenderGroup, di.m_geometryInstance );
            }

            m_cullingManager->groupRemoveObject( m_cullingGroup, di.m_cullingObject );

            std::vector<DefaultHandleDataHandle>::const_iterator it = std::find( m_transparentDIs.begin(), m_transparentDIs.end(), handle );
            if ( it != m_transparentDIs.end() )
            {
              m_transparentDIs.erase( it );
            }

            // last one does not need a swap
            if ( handleData->m_index != m_instances.size() - 1)
            {
              Instance &instance = m_instances.back();
              instance.m_handle->m_index = handleData->m_index;
              m_instances[handleData->m_index] = instance;
            }
            m_instances.pop_back();
          }

          void DrawableManagerDefault::updateDrawableInstance( DrawableManager::Handle handle )
          {
            const DefaultHandleDataHandle& handleData = dp::util::smart_cast<DefaultHandleData>( handle );

            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            Instance& di = m_instances[handleData->m_index];

            dp::sg::core::GeoNodeSharedPtr geoNode = dp::sg::core::weakPtr_cast<dp::sg::core::GeoNode>(m_sceneTree->getObjectTreeNode(di.m_objectTreeIndex).m_object);
            dp::sg::core::GeoNodeLock gnrl(geoNode);

            const ObjectTreeNode& objectTreeNode = m_sceneTree->getObjectTreeNode( di.m_objectTreeIndex );
            bool newVisible = objectTreeNode.m_worldActive; // visible by default

            if ( di.m_isVisible != newVisible )
            {
              di.m_isVisible = newVisible;
              di.updateRendererVisibility( renderer );
            }

            // update primitive
            const dp::sg::core::PrimitiveSharedPtr &primitive = gnrl->getPrimitive();
            if (primitive)
            {
              di.m_resourcePrimitive = ResourcePrimitive::get( primitive, m_resourceManager );
              di.m_boundingBoxLower = Vec4f(dp::sg::core::PrimitiveLock(primitive)->getBoundingBox().getLower(), 1.0f );
              di.m_boundingBoxExtent = Vec4f(dp::sg::core::PrimitiveLock(primitive)->getBoundingBox().getSize(), 0.0f );
              renderer->geometryInstanceSetGeometry( di.m_geometryInstance, di.m_resourcePrimitive->m_geometryHandle );
              if ( di.m_geometryInstanceDepthPass )
              {
                renderer->geometryInstanceSetGeometry( di.m_geometryInstanceDepthPass, di.m_resourcePrimitive->m_geometryHandle );
              }

              m_cullingManager->objectSetBoundingBox( di.m_cullingObject, dp::sg::core::PrimitiveLock(primitive)->getBoundingBox() );
              m_cullingManager->objectSetTransformIndex( di.m_cullingObject, objectTreeNode.m_transformIndex );

              const dp::sg::core::EffectDataSharedPtr &effectDataSurface = gnrl->getMaterialEffect() ? gnrl->getMaterialEffect() : m_shaderManager->getDefaultMaterialEffectData();
              const dp::sg::core::EffectDataSharedPtr &effectDataGeometry = primitive->getGeometryEffect() ? primitive->getGeometryEffect() : m_shaderManager->getDefaultGeometryEffectData();
              {
                // update transparency
                bool newTransparent = dp::sg::core::EffectDataLock( effectDataSurface )->getTransparent() || effectDataGeometry->getTransparent();
                if ( di.m_transparent != newTransparent )
                {
                  if ( newTransparent )
                  {
                    m_transparentDIs.push_back( di.m_handle );
                  }
                  else
                  {
                    // TODO this can be done faster
                    m_transparentDIs.erase( std::find(m_transparentDIs.begin(), m_transparentDIs.end(), di.m_handle ) );
                  }
                  di.m_transparent = newTransparent;
                }
              }

              DP_ASSERT( effectDataSurface && effectDataGeometry );
              if ( di.m_currentEffectSurface != effectDataSurface.get() || di.m_currentEffectGeometry != effectDataGeometry.get() )
              {
                if ( di.m_effectDataAttached )
                {
                  di.m_currentEffectGeometry->detach( m_effectDataObserver.get(), di.m_payload.get() );
                  di.m_currentEffectSurface->detach( m_effectDataObserver.get(), di.m_payload.get() );
                }

                di.m_currentEffectSurface = effectDataSurface.get();
                di.m_currentEffectGeometry = effectDataGeometry.get();

                di.m_smartShaderObject = m_shaderManager->registerGeometryInstance(geoNode, di.m_objectTreeIndex, di.m_geometryInstance );

                di.m_currentEffectGeometry->attach( m_effectDataObserver.get(), di.m_payload.get() );
                di.m_currentEffectSurface->attach( m_effectDataObserver.get(), di.m_payload.get() );
                di.m_effectDataAttached = true;
              }

              dp::rix::core::RenderGroupHandle newRenderGroup = di.m_transparent ? m_renderGroupTransparent : m_renderGroup;
              if ( newRenderGroup != di.m_currentRenderGroup )
              {
                if ( di.m_currentRenderGroup )
                {
                  renderer->renderGroupRemoveGeometryInstance( di.m_currentRenderGroup, di.m_geometryInstance );
                }
                renderer->renderGroupAddGeometryInstance( newRenderGroup, di.m_geometryInstance );
                di.m_currentRenderGroup = newRenderGroup;
              }
            }
            else
            {
              if ( di.m_currentRenderGroup )
              {
                renderer->renderGroupRemoveGeometryInstance( di.m_currentRenderGroup, di.m_geometryInstance );
                di.m_resourcePrimitive = nullptr;
                di.m_geometryInstance = nullptr;
                di.m_currentRenderGroup = nullptr;
                di.m_smartShaderObject = nullptr;
                di.m_currentEffectSurface = nullptr;
                di.m_currentEffectGeometry = nullptr;
              }
            }
          }

          void DrawableManagerDefault::setDrawableInstanceActive( Handle handle, bool active )
          {
            DP_ASSERT( handle );

            const DefaultHandleDataHandle& handleData = dp::util::smart_cast<DefaultHandleData>( handle );
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            Instance& di = m_instances[handleData->m_index];
            if ( di.m_isActive != active )
            {
              di.m_isActive = active;
              di.updateRendererVisibility( renderer );
            }
          }

          void DrawableManagerDefault::setDrawableInstanceTraversalMask( Handle handle, dp::util::Uint32 traversalMask )
          {
            DP_ASSERT( handle );

            const DefaultHandleDataHandle& handleData = dp::util::smart_cast<DefaultHandleData>( handle );
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            Instance& di = m_instances[handleData->m_index];
            if ( traversalMask != di.m_activeTraversalMask )
            {
              di.m_activeTraversalMask = traversalMask;
              di.m_isTraversalActive = !!(m_activeTraversalMask & traversalMask);
              di.updateRendererVisibility( renderer );
            }
          }

          void DrawableManagerDefault::cull( const dp::sg::core::CameraSharedPtr &camera )
          {
            if ( m_cullingEnabled && !m_instances.empty() )
            {
              dp::sg::core::CameraLock cam( camera );
              const Mat44f worldToViewProjection = cam->getWorldToViewMatrix() * cam->getProjection();

              TransformTree const& transformTree = m_sceneTree->getTransformTree();
              ObjectTree    &objectTree = m_sceneTree->getObjectTree();
              dp::rix::core::Renderer* renderer = m_resourceManager->getRenderer();

              m_cullingManager->groupSetMatrices( m_cullingGroup, &transformTree[0].m_worldMatrix, transformTree.size(), sizeof( transformTree[0]) );
              m_cullingManager->cull( m_cullingGroup, m_cullingResult, worldToViewProjection );

              dp::culling::GroupHandle changed = m_cullingManager->resultGetChanged( m_cullingResult );
              size_t changedCount = m_cullingManager->groupGetCount( changed );
#if 0
              if ( changedCount )
              {
                std::cout << "culling error:" << changedCount << std::endl;
                exit(-1);
              }
#endif
              for ( size_t index = 0;index < changedCount; ++index )
              {
                dp::culling::ObjectHandle object = m_cullingManager->groupGetObject( changed, index );
                const DefaultHandleDataHandle& defaultHandleData = dp::util::smart_cast<DefaultHandleData>(m_cullingManager->objectGetUserData( object ));
                Instance& instance = m_instances[defaultHandleData->m_index];

                bool newVisible = objectTree.m_tree[instance.m_objectTreeIndex].m_worldActive && m_cullingManager->resultObjectIsVisible( m_cullingResult, object );
                if ( instance.m_isVisible != newVisible )
                {
                  instance.m_isVisible = newVisible;
                  instance.updateRendererVisibility( renderer );
                }
              }
            }
          }

          namespace 
          {
            struct SortInfo
            {
              dp::rix::core::GeometryInstanceHandle gi;
              float squaredDistance;
            };


            struct SortInfoSort
            {
              bool operator()( const SortInfo& u, const SortInfo& v )
              {
                return u.squaredDistance > v.squaredDistance;
              }
            };
          }

          std::vector<dp::rix::core::GeometryInstanceHandle>& DrawableManagerDefault::getSortedTransparentGIs( const Vec3f& cameraPosition )
          {
            TransformTree const& transformTree = m_sceneTree->getTransformTree();

            std::vector<SortInfo> sortInfo;
            for ( std::vector<DefaultHandleDataHandle>::iterator it = m_transparentDIs.begin(); it != m_transparentDIs.end(); ++it )
            {
              SortInfo info;
              Instance &instance = m_instances[ (*it)->m_index ];
              if ( instance.m_isVisible )
              {
                info.gi = instance.m_geometryInstance;

                Vec4f center = instance.m_boundingBoxLower + 0.5 * instance.m_boundingBoxExtent;
                center = center * transformTree[instance.m_transformIndex].m_worldMatrix;
                Vec3f distance = Vec3f(center) - cameraPosition;
                info.squaredDistance = lengthSquared(distance);
                sortInfo.push_back( info );
              }
            }

            std::sort( sortInfo.begin(), sortInfo.end(), SortInfoSort() );

            m_depthSortedTransparentGIs.clear();
            for ( std::vector<SortInfo>::iterator it = sortInfo.begin(); it != sortInfo.end(); ++it )
            {
              m_depthSortedTransparentGIs.push_back( it->gi );
            }

            return m_depthSortedTransparentGIs;
          }

          bool DrawableManagerDefault::containsTransparentGIs()
          {
            return( !m_transparentDIs.empty() );
          }

          void DrawableManagerDefault::update( const nvsg::ViewStateSharedPtr &viewState )
          {
            setActiveTraversalMask( nvsg::ViewStateLock(viewState)->getTraversalMask() );
            m_shaderManager->update( viewState );
          }

          void DrawableManagerDefault::update( dp::math::Vec2ui const & viewportSize )
          {
            DP_ASSERT( m_viewportSize != viewportSize );
            m_viewportSize = viewportSize;
            if ( m_shaderManager )
            {
              m_shaderManager->updateFragmentParameter( std::string( "sys_ViewportSize" ), dp::rix::core::ContainerDataRaw( 0, &viewportSize[0], sizeof( dp::math::Vec2ui ) ) );
            }
          }

          void DrawableManagerDefault::setActiveTraversalMask( unsigned int traversalMask )
          {
            if ( traversalMask != m_activeTraversalMask )
            {
              dp::rix::core::Renderer* renderer = m_resourceManager->getRenderer();
              for ( std::vector<Instance>::iterator it = m_instances.begin(); it != m_instances.end(); ++it )
              {
                bool newActive = !!(it->m_activeTraversalMask & traversalMask);
                if ( newActive != it->m_isTraversalActive )
                {
                  it->m_isTraversalActive = newActive;
                  it->updateRendererVisibility( renderer );
                }
              }
              m_activeTraversalMask = traversalMask;
            }
          }

          void DrawableManagerDefault::onTransformChanged( dp::sg::xbar::SceneTree::EventTransform const& event )
          {
            m_cullingManager->groupMatrixChanged( m_cullingGroup, event.getIndex() );
          }

          void DrawableManagerDefault::setEnvironmentSampler( const dp::sg::core::SamplerSharedPtr & sampler )
          {
            if ( m_environmentSampler != sampler )
            {
              m_environmentSampler = sampler;
              if ( m_shaderManager )
              {
                m_shaderManager->setEnvironmentSampler( sampler );
              }
            }
          }

          const dp::sg::core::SamplerSharedPtr & DrawableManagerDefault::getEnvironmentSampler() const
          {
            return( m_environmentSampler );
          }

          dp::sg::renderer::rix::gl::ShaderManager * DrawableManagerDefault::getShaderManager() const
          {
            return( m_shaderManager.get() );
          }

          dp::math::Box3f DrawableManagerDefault::getBoundingBox() const
          {
            return m_cullingManager->getBoundingBox( m_cullingGroup );
          }


          std::map<dp::fx::Domain,std::string> DrawableManagerDefault::getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode ) const
          {
            DP_ASSERT( m_shaderManager );
            return( m_shaderManager->getShaderSources( geoNode ) );
          }

          void DrawableManagerDefault::setCullingEnabled( bool cullingEnabled )
          {
            m_cullingEnabled = cullingEnabled;
          }

          bool DrawableManagerDefault::isCullingEnabled() const
          {
            return m_cullingEnabled;
          }

          void DrawableManagerDefault::detachEffectDataObserver()
          {
            for ( std::vector<Instance>::iterator it = m_instances.begin(); it != m_instances.end(); ++it )
            {
              if ( it->m_effectDataAttached )
              {
                it->m_currentEffectGeometry->detach( m_effectDataObserver.get(), it->m_payload.get() );
                it->m_currentEffectSurface->detach( m_effectDataObserver.get(), it->m_payload.get() );
                it->m_effectDataAttached = false;
              }
            }
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

