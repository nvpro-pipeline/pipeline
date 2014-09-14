// Copyright NVIDIA Corporation 2010-2011
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
#include <dp/sg/renderer/rix/gl/TransparencyManagerNone.h>
#include <dp/sg/renderer/rix/gl/TransparencyManagerOITAll.h>
#include <dp/sg/renderer/rix/gl/TransparencyManagerOITClosestArray.h>
#include <dp/sg/renderer/rix/gl/TransparencyManagerOITClosestList.h>
#include <dp/sg/renderer/rix/gl/TransparencyManagerSB.h>
#include <dp/sg/renderer/rix/gl/inc/SceneRendererImpl.h>
#include <dp/sg/renderer/rix/gl/inc/DrawableManagerDefault.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/rix/gl/RiXGL.h>
#include <dp/sg/core/FrustumCamera.h>

#define ENABLE_PROFILING 0
#include <dp/util/Profile.h>
#include <dp/util/File.h>

#define ENABLE_NSIGHT_PROFILING 0
#if ENABLE_NSIGHT_PROFILING
#include <nvToolsExt.h>
#define GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV 0x00000010
#define NSIGHT_START_RANGE(range)   nvtxRangePushA( range )
                                    // make sure, the rendering step has finished on nvtxRangePop
#define NSIGHT_STOP_RANGE()         glMemoryBarrier( GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV );  \
                                    nvtxRangePop()
#else
#define NSIGHT_START_RANGE(range)
#define NSIGHT_STOP_RANGE()
#endif



using namespace dp::math;
using namespace dp::sg::xbar;
using namespace dp::sg::core;

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

          SmartTransparencyManager createTransparencyManager( TransparencyMode mode, dp::math::Vec2ui const & viewportSize )
          {
            switch( mode )
            {
              case TM_NONE :
                return( TransparencyManagerNone::create() );
              case TM_ORDER_INDEPENDENT_ALL :
                return( TransparencyManagerOITAll::create( viewportSize ) );
              case TM_ORDER_INDEPENDENT_CLOSEST_ARRAY :
                return( TransparencyManagerOITClosestArray::create( viewportSize, 16 ) );
              case TM_ORDER_INDEPENDENT_CLOSEST_LIST :
                return( TransparencyManagerOITClosestList::create( viewportSize, 16, 8.0f ) );
              case TM_SORTED_BLENDED :
                return( TransparencyManagerSB::create() );
              case TM_UNKNOWN :
              default :
                DP_ASSERT( false );
                return( TransparencyManagerSB::create() );
            }
          }

          SmartSceneRenderer SceneRendererImpl::create( const char *renderEngine, dp::fx::Manager shaderManagerType
                                                      , dp::culling::Mode cullingMode, TransparencyMode transparencyMode
                                                      , const dp::gl::SmartRenderTarget &renderTarget )
          {
            return new SceneRendererImpl( renderEngine, shaderManagerType, cullingMode, transparencyMode, renderTarget );
          }

          SceneRendererImpl::SceneRendererImpl( const char *renderEngine, dp::fx::Manager shaderManagerType
                                              , dp::culling::Mode cullingMode, TransparencyMode transparencyMode
                                              , const dp::gl::SmartRenderTarget &renderTarget )
            : SceneRenderer( renderTarget )
            , m_renderer( nullptr )
            , m_drawableManager( nullptr )
            , m_contextRegistered( false )
            , m_rendererInitialized( false )
            , m_shaderManager( shaderManagerType )
            , m_renderEngine( renderEngine )
            , m_cullingMode( cullingMode )
            , m_cullingEnabled( true )
            , m_viewportSize( 0, 0 )
          {
            if ( renderTarget )
            {
              m_viewportSize[0] = renderTarget->getWidth();
              m_viewportSize[1] = renderTarget->getHeight();
            }
            m_transparencyManager = createTransparencyManager( transparencyMode, m_viewportSize );
          }

          SceneRendererImpl::~SceneRendererImpl()
          {
            // clear all objects which can contain gl resources here
            if ( m_userRenderContext )
            {
              dp::gl::RenderContextStack glStack;
              glStack.push( m_userRenderContext );

              delete m_drawableManager;

              m_sceneTree.reset();
              m_resourceManager.reset();

              if( m_renderer )
              {
                m_renderer->deleteThis();
              }

              glStack.pop();
            }
          }

          void SceneRendererImpl::addRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr &rendererOptions )
          {
            SceneRenderer::addRendererOptions( rendererOptions );

            if( m_sceneTree )
            {
              m_sceneTree->addRendererOptions( rendererOptions );
            }
          }

          void SceneRendererImpl::deletePrimitiveCaches()
          {
            //PrimitiveCacheDeleter pcd;
            //pcd.process( m_sceneTree );
          }

          void SceneRendererImpl::beginRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget )
          {
            if ( !m_rendererInitialized )
            {
              m_rendererInitialized = initializeRenderer();
            }

            if ( m_rendererInitialized )
            {
              SceneRenderer::beginRendering( viewState, renderTarget );

              dp::gl::SmartRenderTarget renderTargetGL = dynamic_cast<dp::gl::RenderTarget*>( renderTarget.get() );
              DP_ASSERT( renderTargetGL );

              if ( !m_userRenderContext )
              {
                m_userRenderContext = renderTargetGL->getRenderContext();
              }
              else
              {
                DP_ASSERT( renderTargetGL->getRenderContext() == m_userRenderContext && "Current RenderContext is not the same as in the first call.");
              }

              if ( renderTargetGL && ( ( m_viewportSize[0] != renderTargetGL->getWidth() ) || ( m_viewportSize[1] != renderTargetGL->getHeight() ) ) )
              {
                m_viewportSize[0] = renderTargetGL->getWidth();
                m_viewportSize[1] = renderTargetGL->getHeight();
                if ( m_drawableManager )
                {
                  m_drawableManager->update( m_viewportSize );
                }
                m_transparencyManager->setViewportSize( m_viewportSize );
              }

              Vec4f bgColor = viewState->getScene()->getBackColor();
              renderTargetGL->setClearColor( bgColor[0], bgColor[1], bgColor[2], bgColor[3] );
              renderTargetGL->beginRendering();

              if ( !m_contextRegistered )
              {
                DP_ASSERT( dynamic_cast<RiX::GL::RiXGL*>(m_renderer) );
                static_cast<RiX::GL::RiXGL*>(m_renderer)->registerContext();
                m_contextRegistered = true;
              }

              dp::sg::ui::RendererOptionsSharedPtr rendererOptions( viewState->getRendererOptions() );
              DP_ASSERT( rendererOptions );
            }
          }

          void SceneRendererImpl::endRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget )
          {
            if ( m_rendererInitialized )
            {
              dp::gl::SmartRenderTarget renderTargetGL = dynamic_cast<dp::gl::RenderTarget*>( renderTarget.get() );
              DP_ASSERT( renderTargetGL );
              renderTargetGL->endRendering();

              SceneRenderer::endRendering( viewState, renderTarget );
            }
          }

          void SceneRendererImpl::setDepthPass( bool depthPass )
          {
            m_depthPass = depthPass;
          }
  
          void SceneRendererImpl::doRender( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget )
          {
            if ( m_rendererInitialized )
            {
              dp::gl::SmartRenderTarget renderTargetGL = dynamic_cast<dp::gl::RenderTarget*>( renderTarget.get() );
              DP_ASSERT( renderTargetGL );

#if 0
              // store OpenGL state 
              glPushAttrib( GL_ALL_ATTRIB_BITS );
              glPushClientAttrib( GL_CLIENT_VERTEX_ARRAY_BIT | GL_CLIENT_PIXEL_STORE_BIT );
#endif

              // (re-)generate SceneTree if
              // 1) no SceneTree present
              // 2) Scene changed
              SceneSharedPtr scene = viewState->getScene();
              // FIXME m_lastScene != scene will fail if a new node has been allocated at exactly the same pointer as the old one.
              if( m_sceneTree == nullptr || m_lastScene != scene )
              {
                delete m_drawableManager;

                // generate a new render list
                m_sceneTree.reset();
                m_sceneTree = SceneTree::create( scene );

                m_drawableManager = createDrawableManager( m_resourceManager );
                m_drawableManager->setSceneTree( m_sceneTree );
                m_drawableManager->update( m_viewportSize );

                DrawableManagerDefault* drawableManagerDefault = dynamic_cast<DrawableManagerDefault*>( m_drawableManager );
                drawableManagerDefault->setCullingEnabled( m_cullingEnabled );

                m_lastScene = scene;

                m_transparencyManager->setShaderManager( drawableManagerDefault->getShaderManager() );
                m_transparencyManager->useParameterContainer( m_resourceManager->getRenderer(), drawableManagerDefault->getRenderGroupTransparent() );
                m_transparencyManager->useParameterContainer( m_resourceManager->getRenderer(), drawableManagerDefault->getRenderGroupTransparentDepthPass() );
              }
      
              // Refresh all observed data
              {
                // Refresh the Scene Tree (the caches need to be filled for the first render)
                m_sceneTree->update( viewState );

                // Refresh the observed resources
                m_resourceManager->updateResources();
              }

              {
        
                PROFILE( "Render");
                doRenderDrawables( viewState, renderTargetGL );
              }

#if 0
              // pop OpenGL state
              glPopClientAttrib( );
              glPopAttrib( );
#endif
            }
          }

          DrawableManager* SceneRendererImpl::createDrawableManager( const SmartResourceManager &resourceManager ) const
          {
            DrawableManagerDefault * dmd = new DrawableManagerDefault( resourceManager, m_transparencyManager, m_shaderManager, m_cullingMode );
            dmd->setEnvironmentSampler( getEnvironmentSampler() );
            dmd->setCullingEnabled( m_cullingEnabled );

            return( dmd );
          }

          void SceneRendererImpl::doRenderDrawables( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::gl::SmartRenderTarget const& renderTarget )
          {
            CameraSharedPtr camera = viewState->getCamera();

            DrawableManagerDefault *drawableManagerDefault = dynamic_cast<DrawableManagerDefault*>(m_drawableManager);
            DP_ASSERT( drawableManagerDefault );

            // update near/far plane before culling
            if ( viewState->getAutoClipPlanes() && camera.isPtrTo<dp::sg::core::FrustumCamera>() )
            {
              dp::math::Box3f bbox = drawableManagerDefault->getBoundingBox();

              if ( isValid( bbox ) )
              {
                dp::math::Sphere3f bs( bbox.getCenter(), length( bbox.getSize() ) * 0.5f );
                camera.staticCast<FrustumCamera>()->calcNearFarDistances( bs );
              }
            }

            // update the viewstate after updating the near/far planes of the camera
            ((DrawableManagerDefault*)m_drawableManager)->update( viewState );

            // cull
            drawableManagerDefault->cull( camera );

            if( drawableManagerDefault )
            {
              NSIGHT_START_RANGE( "Frame" );
              glEnable(GL_DEPTH_TEST);
              if ( m_depthPass )
              {
                NSIGHT_START_RANGE( "DepthPass" );
                glDepthFunc(GL_LEQUAL);
                glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
                m_renderer->render( drawableManagerDefault->getRenderGroupDepthPass() );
                glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
                NSIGHT_STOP_RANGE();
              }

              NSIGHT_START_RANGE( "OpaquePass" );
              m_renderer->render( drawableManagerDefault->getRenderGroup() );
              NSIGHT_STOP_RANGE();
              if ( drawableManagerDefault->containsTransparentGIs() )
              {
                bool done;
                do
                {
                  NSIGHT_START_RANGE( "Begin TransparentPass" );
                  m_transparencyManager->beginTransparentPass( m_renderer );
                  NSIGHT_STOP_RANGE();
                  if ( m_transparencyManager->supportsDepthPass() )
                  {
                    NSIGHT_START_RANGE( "TransparentPass Depth" );
                    m_renderer->render( drawableManagerDefault->getRenderGroupTransparentDepthPass() );
                    NSIGHT_STOP_RANGE();
                    NSIGHT_START_RANGE( "TransparentPass Resolve Depth" );
                    m_transparencyManager->resolveDepthPass();
                    NSIGHT_STOP_RANGE();
                  }
                  if ( m_transparencyManager->needsSortedRendering() )
                  {
                    std::vector<RiX::GeometryInstanceSharedHandle> const & sortedGIs = drawableManagerDefault->getSortedTransparentGIs( camera->getPosition() );
                    NSIGHT_START_RANGE( "TransparentPass Sorted" );
                    m_renderer->render( drawableManagerDefault->getRenderGroupTransparent(), &sortedGIs[0], sortedGIs.size() );
                    NSIGHT_STOP_RANGE();
                  }
                  else
                  {
                    NSIGHT_START_RANGE( "TransparentPass Unsorted" );
                    m_renderer->render( drawableManagerDefault->getRenderGroupTransparent() );
                    NSIGHT_STOP_RANGE();
                  }
                  NSIGHT_START_RANGE( "End TransparentPass" );
                  done = m_transparencyManager->endTransparentPass();
                  NSIGHT_STOP_RANGE();
                } while ( !done );
              }
              NSIGHT_STOP_RANGE();
            }
          }

          void SceneRendererImpl::onEnvironmentSamplerChanged()
          {
            if ( m_drawableManager )
            {
              m_drawableManager->setEnvironmentSampler( getEnvironmentSampler() );
            }
          }

          std::map<dp::fx::Domain,std::string> SceneRendererImpl::getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const
          {
            DP_ASSERT( m_drawableManager );
            return( m_drawableManager->getShaderSources( geoNode, depthPass ) );
          }

          void SceneRendererImpl::setCullingEnabled( bool cullingEnabled )
          {
            m_cullingEnabled = cullingEnabled;
            if ( m_drawableManager )
            {
              DP_ASSERT( dynamic_cast<DrawableManagerDefault*>(m_drawableManager) );
              static_cast<DrawableManagerDefault*>(m_drawableManager)->setCullingEnabled( cullingEnabled );
              DP_ASSERT( m_cullingEnabled == static_cast<DrawableManagerDefault*>(m_drawableManager)->isCullingEnabled() );
            }
          }

          bool SceneRendererImpl::isCullingEnabled() const
          {
            return m_cullingEnabled;
          }

          void SceneRendererImpl::setCullingMode( dp::culling::Mode mode )
          {
            if ( m_cullingMode != mode )
            {
              m_sceneTree.reset();
              m_cullingMode = mode;
            }
          }

          dp::culling::Mode SceneRendererImpl::getCullingMode() const
          {
            return m_cullingMode;
          }

          dp::sg::renderer::rix::gl::TransparencyMode SceneRendererImpl::getTransparencyMode() const
          {
            DP_ASSERT( m_transparencyManager );
            return( m_transparencyManager->getTransparencyMode() );
          }

          void SceneRendererImpl::setTransparencyMode( dp::sg::renderer::rix::gl::TransparencyMode mode )
          {
            if ( mode != m_transparencyManager->getTransparencyMode() )
            {
              m_transparencyManager = createTransparencyManager( mode, m_viewportSize );
              m_transparencyManager->initializeParameterContainer( m_renderer, m_viewportSize );
              m_sceneTree.reset();
            }
          }

          void SceneRendererImpl::setShaderManager( dp::fx::Manager shaderManager )
          {
            if ( m_shaderManager != shaderManager )
            {
              m_shaderManager = shaderManager;
              shutdownRenderer();
            }
          }

          dp::fx::Manager SceneRendererImpl::getShaderManager() const
          {
            return m_shaderManager;
          }

          void SceneRendererImpl::setRenderEngine( std::string const& renderEngine )
          {
            if ( renderEngine != m_renderEngine )
            {
              m_renderEngine = renderEngine;
              shutdownRenderer();
            }
          }

          std::string const& SceneRendererImpl::getRenderEngine() const
          {
            return m_renderEngine;
          }


          void SceneRendererImpl::shutdownRenderer()
          {
            m_sceneTree.reset();
            delete m_drawableManager;
            m_drawableManager = 0;
            m_resourceManager.reset();
            m_rix.reset();
            m_contextRegistered = false;
            m_rendererInitialized = false;
          }

          bool SceneRendererImpl::initializeRenderer()
          {
            if ( !m_rendererInitialized )
            {
              // clear all resources
#if defined(DP_OS_WINDOWS)
              m_rix = dp::util::DynamicLibrary::createFromFile( "RiXGL.rdr" );
#else
              m_rix = dp::util::DynamicLibrary::createFromFile("libRiXGL.rdr");
#endif
              DP_ASSERT( m_rix && "Could not load dynamic library RiXGL.rdr" );

              dp::rix::core::PFNCREATERENDERER createRenderer = reinterpret_cast<RiX::PFNCREATERENDERER>(m_rix->getSymbol("createRenderer"));
              m_renderer = dynamic_cast<RiX::GL::RiXGL*>((*createRenderer)( m_renderEngine.c_str() ));
              DP_ASSERT( m_renderer && "Could not create RiXGL renderer" );

              m_resourceManager = new ResourceManager( m_renderer, m_shaderManager );

              m_rendererInitialized = m_renderer && m_resourceManager;

              if ( !m_rendererInitialized )
              {
                shutdownRenderer();
              }
              else
              {
                m_transparencyManager->initializeParameterContainer( m_renderer, m_viewportSize );
              }
            }

            return m_rendererInitialized;
          }

          dp::sg::renderer::rix::gl::SmartTransparencyManager const & SceneRendererImpl::getTransparencyManager() const
          {
            return( m_transparencyManager );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

