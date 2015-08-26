// Copyright NVIDIA Corporation 2011-2015
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

#include <dp/sg/core/Camera.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/ui/ViewState.h>

#include <dp/gl/RenderContext.h>

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
            , m_effectDataAttached( false )
          {
            m_payload = Payload::create();
          }

          inline void DrawableManagerDefault::Instance::updateRendererVisibility( dp::rix::core::Renderer* renderer )
          {
            renderer->geometryInstanceSetVisible( m_geometryInstance, m_isTraversalActive && m_isActive && m_isVisible );
            if ( m_geometryInstanceDepthPass )
            {
              renderer->geometryInstanceSetVisible( m_geometryInstanceDepthPass, m_isTraversalActive && m_isActive && m_isVisible );
            }
          }

          /************************************************************************/
          /* DrawableManagerDefault                                               */
          /************************************************************************/
          DrawableManagerDefault::DrawableManagerDefault( const ResourceManagerSharedPtr & resourceManager
                                                        , TransparencyManagerSharedPtr const & transparencyManager
                                                        , dp::fx::Manager shaderManagerType
                                                        , dp::culling::Mode cullingMode )
            : dp::sg::xbar::DrawableManager( )
            , m_resourceManager( resourceManager )
            , m_shaderManager( nullptr )
            , m_shaderManagerType( shaderManagerType )
            , m_cullingMode( cullingMode)
            , m_cullingEnabled( true )
            , m_activeTraversalMask( ~0 )
            , m_viewportSize( 0, 0 )
            , m_transparencyManager( transparencyManager )
          {
          }

          DrawableManagerDefault::~DrawableManagerDefault()
          {
            // reset resources
            setSceneTree( SceneTreeSharedPtr::null );
            detachEffectDataObserver();
            m_effectDataObserver.reset();
          }

          DrawableManager::Handle DrawableManagerDefault::addDrawableInstance( dp::sg::core::GeoNodeWeakPtr geoNode, ObjectTreeIndex objectTreeIndex )
          {
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            ObjectTreeNode objectTreeNode = getSceneTree()->getObjectTreeNode( objectTreeIndex );

            // generate a new handle and fill it
            DefaultHandleDataSharedPtr handle = DefaultHandleData::create( dp::sg::xbar::ObjectTreeIndex(m_instances.size()) );

            // generate a new instance
            m_instances.push_back( Instance() );
            Instance &di = m_instances.back();
            di.m_handle = handle;
            di.m_geoNode = geoNode.getSharedPtr();
            di.m_objectTreeIndex = objectTreeIndex;
            di.m_transformIndex = objectTreeNode.m_transform;
            di.m_transparent = false;
            di.m_currentRenderGroup = nullptr;
            di.m_isVisible = true; // gis are visible by default
            di.m_payload->setHandle( handle );
            di.m_effectDataAttached = false;

            DP_ASSERT(di.m_geoNode);

            // create geometry instance
            di.m_geometryInstance = renderer->geometryInstanceCreate();
            di.m_geometryInstanceDepthPass = renderer->geometryInstanceCreate();

            bool isTransparent = false;
            bool foundGLSL = true;

            // update material & primitive
            updateDrawableInstance( handle );

            return handle;
          }

          void DrawableManagerDefault::removeDrawableInstance( DrawableManager::Handle handle )
          {
            DP_ASSERT( handle.isPtrTo<DefaultHandleData>() );
            DefaultHandleDataSharedPtr const& handleData = handle.staticCast<DefaultHandleData>();

            // TODO really remove data
            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            Instance& di = m_instances[handleData->m_index];

            if( di.m_effectDataAttached )
            {
              di.m_currentEffectSurface->detach( m_effectDataObserver.get(), di.m_payload.operator->() );   // Big Hack !!
              di.m_effectDataAttached = false;
            }

            if ( di.m_currentRenderGroup )
            {
              DP_ASSERT( di.m_currentRenderGroup == m_renderGroups[di.m_transparent][RGP_FORWARD].get() );
              renderer->renderGroupRemoveGeometryInstance( di.m_currentRenderGroup, di.m_geometryInstance );
              if ( di.m_smartShaderObjectDepthPass )
              {
                renderer->renderGroupRemoveGeometryInstance( m_renderGroups[di.m_transparent][RGP_DEPTH], di.m_geometryInstanceDepthPass );
              }
              di.m_currentRenderGroup = nullptr;
            }

            std::vector<DefaultHandleDataSharedPtr>::iterator it = std::find( m_transparentDIs.begin(), m_transparentDIs.end(), handle );
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
            DP_ASSERT( handle.isPtrTo<DefaultHandleData>() );
            DefaultHandleDataSharedPtr const& handleData = handle.staticCast<DefaultHandleData>();

            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();

            Instance& di = m_instances[handleData->m_index];

            dp::sg::core::GeoNodeSharedPtr const& geoNode = di.m_geoNode;

            // update primitive
            const dp::sg::core::PrimitiveSharedPtr &primitive = geoNode->getPrimitive();
            if (primitive)
            {
              di.m_resourcePrimitive = ResourcePrimitive::get( primitive, m_resourceManager );
              di.m_boundingBoxLower = Vec4f(geoNode->getBoundingBox().getLower(), 1.0f );
              di.m_boundingBoxExtent = Vec4f(geoNode->getBoundingBox().getSize(), 0.0f );
              renderer->geometryInstanceSetGeometry( di.m_geometryInstance, di.m_resourcePrimitive->m_geometryHandle );
              if ( di.m_geometryInstanceDepthPass )
              {
                renderer->geometryInstanceSetGeometry( di.m_geometryInstanceDepthPass, di.m_resourcePrimitive->m_geometryHandle );
              }

              const dp::sg::core::EffectDataSharedPtr &effectDataSurface = geoNode->getMaterialEffect() ? geoNode->getMaterialEffect() : m_shaderManager->getDefaultEffectData();
              {
                // update transparency
                bool newTransparent = effectDataSurface->getTransparent();
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

              DP_ASSERT( effectDataSurface );
              ShaderManagerInstanceSharedPtr shaderObjectDepthPass = di.m_smartShaderObjectDepthPass;
              if ( di.m_currentEffectSurface != effectDataSurface )
              {
                if ( di.m_effectDataAttached )
                {
                  di.m_currentEffectSurface->detach( m_effectDataObserver.get(), di.m_payload.operator->() );   // Big Hack !!
                }

                di.m_currentEffectSurface = effectDataSurface;

                di.m_smartShaderObject = m_shaderManager->registerGeometryInstance(geoNode, di.m_objectTreeIndex, di.m_geometryInstance );
                di.m_smartShaderObjectDepthPass = m_shaderManager->registerGeometryInstance(geoNode, di.m_objectTreeIndex, di.m_geometryInstanceDepthPass, RPT_DEPTH );

                di.m_currentEffectSurface->attach( m_effectDataObserver.get(), di.m_payload.operator->() );   // Big Hack !!
                di.m_effectDataAttached = true;
              }

              dp::rix::core::RenderGroupHandle newRenderGroup = m_renderGroups[di.m_transparent][RGP_FORWARD].get();
              if ( newRenderGroup != di.m_currentRenderGroup )
              {
                if ( di.m_currentRenderGroup )
                {
                  DP_ASSERT( di.m_currentRenderGroup == m_renderGroups[!di.m_transparent][RGP_FORWARD].get() );
                  renderer->renderGroupRemoveGeometryInstance( di.m_currentRenderGroup, di.m_geometryInstance );
                  if ( shaderObjectDepthPass )
                  {
                    renderer->renderGroupRemoveGeometryInstance( m_renderGroups[!di.m_transparent][RGP_DEPTH], di.m_geometryInstanceDepthPass );
                  }
                }
                renderer->renderGroupAddGeometryInstance( newRenderGroup, di.m_geometryInstance );
                if ( di.m_smartShaderObjectDepthPass )
                {
                  renderer->renderGroupAddGeometryInstance( m_renderGroups[di.m_transparent][RGP_DEPTH], di.m_geometryInstanceDepthPass );
                }

                di.m_currentRenderGroup = newRenderGroup;
              }
              else if ( !shaderObjectDepthPass && di.m_smartShaderObjectDepthPass )
              {
                // if the smartShaderObjectDepthPass is newly added
                renderer->renderGroupAddGeometryInstance( m_renderGroups[di.m_transparent][RGP_DEPTH], di.m_geometryInstanceDepthPass );
              }
            }
            else
            {
              if ( di.m_currentRenderGroup )
              {
                DP_ASSERT( di.m_currentRenderGroup == m_renderGroups[di.m_transparent][RGP_FORWARD].get() );
                renderer->renderGroupRemoveGeometryInstance( di.m_currentRenderGroup, di.m_geometryInstance );
                if ( di.m_smartShaderObjectDepthPass )
                {
                  renderer->renderGroupRemoveGeometryInstance( m_renderGroups[di.m_transparent][RGP_DEPTH], di.m_geometryInstanceDepthPass );
                }
                di.m_resourcePrimitive = ResourcePrimitiveSharedPtr::null;
                di.m_geometryInstance = nullptr;
                di.m_geometryInstanceDepthPass = nullptr;
                di.m_currentRenderGroup = nullptr;
                di.m_smartShaderObject = ShaderManagerInstanceSharedPtr::null;
                di.m_smartShaderObjectDepthPass = ShaderManagerInstanceSharedPtr::null;
                di.m_currentEffectSurface.reset();
              }
            }
          }

          void DrawableManagerDefault::setDrawableInstanceActive( Handle handle, bool active )
          {
            DP_ASSERT( handle.isPtrTo<DefaultHandleData>() );
            DefaultHandleDataSharedPtr const& handleData = handle.staticCast<DefaultHandleData>();

            dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
            Instance& di = m_instances[handleData->m_index];
            if ( di.m_isActive != active )
            {
              di.m_isActive = active;
              di.updateRendererVisibility( renderer );
            }
          }

          void DrawableManagerDefault::setDrawableInstanceTraversalMask( Handle handle, dp::Uint32 traversalMask )
          {
            DP_ASSERT( handle.isPtrTo<DefaultHandleData>() );
            DefaultHandleDataSharedPtr const& handleData = handle.staticCast<DefaultHandleData>();

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
              const Mat44f worldToViewProjection = camera->getWorldToViewMatrix() * camera->getProjection();

              ObjectTree    &objectTree = getSceneTree()->getObjectTree();
              dp::rix::core::Renderer* renderer = m_resourceManager->getRenderer();

              m_cullingManager->cull( m_cullingResult, worldToViewProjection );

              std::vector<ObjectTreeIndex> const & changed = m_cullingManager->resultGetChangedIndices( m_cullingResult );

              for ( size_t index = 0;index < changed.size(); ++index )
              {
                DP_ASSERT( getDrawableInstance( changed[index] ).isPtrTo<DefaultHandleData>() );
                DefaultHandleDataSharedPtr const& defaultHandleData = getDrawableInstance( changed[index] ).staticCast<DefaultHandleData>();
                Instance & instance = m_instances[defaultHandleData->m_index];

                bool newVisible = m_cullingManager->resultIsVisible( m_cullingResult, changed[index] );
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

          std::vector<dp::rix::core::GeometryInstanceSharedHandle>& DrawableManagerDefault::getSortedTransparentGIs( const Vec3f& cameraPosition )
          {
            std::vector<SortInfo> sortInfo;
            for ( std::vector<DefaultHandleDataSharedPtr>::iterator it = m_transparentDIs.begin(); it != m_transparentDIs.end(); ++it )
            {
              SortInfo info;
              Instance &instance = m_instances[ (*it)->m_index ];
              if ( instance.m_isVisible )
              {
                info.gi = instance.m_geometryInstance.get();

                Vec4f center = instance.m_boundingBoxLower + 0.5 * instance.m_boundingBoxExtent;
                center = center * getSceneTree()->getTransformEntry(instance.m_transformIndex).world;
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

          void DrawableManagerDefault::update( dp::sg::ui::ViewStateSharedPtr const& viewState )
          {
            setActiveTraversalMask( viewState->getTraversalMask() );
            m_shaderManager->update( viewState );
            DrawableManager::update();
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
            return m_cullingManager->getBoundingBox( );
            return dp::math::Box3f();
          }


          std::map<dp::fx::Domain,std::string> DrawableManagerDefault::getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const
          {
            DP_ASSERT( m_shaderManager );
            return( m_shaderManager->getShaderSources( geoNode, depthPass ) );
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
                it->m_currentEffectSurface->detach( m_effectDataObserver.get(), it->m_payload.operator->() );   // Big Hack !!
                it->m_effectDataAttached = false;
              }
            }
          }

          void DrawableManagerDefault::onSceneTreeChanged()
          {
            if ( getSceneTree() )
            {
              dp::culling::Mode cullingMode = m_cullingMode;
              // determine if OpenGL 4.3 is available
              if ( m_cullingMode == dp::culling::MODE_AUTO )
              {
                cullingMode = dp::culling::MODE_CUDA;
              }
              m_cullingManager = dp::sg::xbar::culling::Culling::create( getSceneTree(), m_cullingMode );
              m_cullingResult = m_cullingManager->resultCreate();

              switch ( m_shaderManagerType )
              {
              case dp::fx::MANAGER_UNIFORM:
              case dp::fx::MANAGER_SHADERBUFFER:
              case dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX:
              case dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX:
              case dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT:
              case dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT_RIX:
                m_shaderManager.reset( new ShaderManagerRiXFx( getSceneTree(), m_shaderManagerType, m_resourceManager, m_transparencyManager ) );
                break;
              default:
                m_shaderManager.reset( new ShaderManagerRiXFx( getSceneTree(), dp::fx::MANAGER_UNIFORM, m_resourceManager, m_transparencyManager ) );
              }

              dp::rix::core::Renderer *renderer = m_resourceManager->getRenderer();
              for ( int i=0 ; i<RGL_COUNT ; i++ )
              {
                for ( int j=0 ; j<RGP_COUNT ; j++ )
                {
                  m_renderGroups[i][j] = renderer->renderGroupCreate();
                  m_renderGroupInstances[i][j] = m_shaderManager->registerRenderGroup( m_renderGroups[i][j] );
                }
              }

              m_shaderManager->setEnvironmentSampler( m_environmentSampler );
              m_shaderManager->updateFragmentParameter( std::string( "sys_ViewportSize" ), dp::rix::core::ContainerDataRaw( 0, &m_viewportSize[0], sizeof( dp::math::Vec2ui ) ) );
              m_transparencyManager->updateFragmentParameters();

              // Observe Effects of SceneTree
              m_effectDataObserver.reset( new EffectDataObserver( this ) );
            }
            else
            {
              m_cullingManager.reset();
              m_shaderManager.reset();
              m_effectDataObserver.reset( );
            }
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
