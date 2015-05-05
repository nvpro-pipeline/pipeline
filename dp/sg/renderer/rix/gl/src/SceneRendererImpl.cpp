// Copyright NVIDIA Corporation 2010-2014
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
#include <dp/fx/EffectLibrary.h>
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
#include <dp/util/SharedPtr.h>

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

          TransparencyManagerSharedPtr createTransparencyManager( TransparencyMode mode, dp::math::Vec2ui const & viewportSize )
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

          SceneRendererSharedPtr SceneRendererImpl::create( const char *renderEngine, dp::fx::Manager shaderManagerType
                                                          , dp::culling::Mode cullingMode, TransparencyMode transparencyMode
                                                          , const dp::gl::RenderTargetSharedPtr &renderTarget )
          {
            return( std::shared_ptr<SceneRendererImpl>( new SceneRendererImpl( renderEngine, shaderManagerType, cullingMode, transparencyMode, renderTarget ) ) );
          }

          SceneRendererImpl::SceneRendererImpl( const char *renderEngineOptions, dp::fx::Manager shaderManagerType
                                              , dp::culling::Mode cullingMode, TransparencyMode transparencyMode
                                              , const dp::gl::RenderTargetSharedPtr &renderTarget )
            : SceneRenderer( renderTarget )
            , m_renderer( nullptr )
            , m_drawableManager( nullptr )
            , m_contextRegistered( false )
            , m_rendererInitialized( false )
            , m_shaderManager( shaderManagerType )
            , m_renderEngineOptions( renderEngineOptions )
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

              glStack.pop();
            }
            m_environmentRenderer.reset();
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

          void SceneRendererImpl::beginRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget )
          {
            if ( !m_rendererInitialized )
            {
              m_rendererInitialized = initializeRenderer();
            }

            if ( m_rendererInitialized )
            {
              SceneRenderer::beginRendering( viewState, renderTarget );

              dp::gl::RenderTargetSharedPtr renderTargetGL = dp::util::shared_cast<dp::gl::RenderTarget>( renderTarget );
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
                DP_ASSERT( dynamic_cast<dp::rix::gl::RiXGL*>(m_renderer.get()) );
                static_cast<dp::rix::gl::RiXGL*>(m_renderer.get())->registerContext();
                m_contextRegistered = true;
              }

              dp::sg::ui::RendererOptionsSharedPtr rendererOptions( viewState->getRendererOptions() );
              DP_ASSERT( rendererOptions );
            }
          }

          void SceneRendererImpl::endRendering( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget )
          {
            if ( m_rendererInitialized )
            {
              dp::gl::RenderTargetSharedPtr renderTargetGL = dp::util::shared_cast<dp::gl::RenderTarget>( renderTarget );
              DP_ASSERT( renderTargetGL );
              renderTargetGL->endRendering();

              SceneRenderer::endRendering( viewState, renderTarget );
            }
          }

          void SceneRendererImpl::setDepthPass( bool depthPass )
          {
            m_depthPass = depthPass;
          }
  
          void SceneRendererImpl::doRender( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget )
          {
            if ( m_rendererInitialized )
            {
              dp::gl::RenderTargetSharedPtr renderTargetGL = dp::util::shared_cast<dp::gl::RenderTarget>( renderTarget );
              DP_ASSERT( renderTargetGL );

              if (viewState->getSceneTree() != m_sceneTree)
              {
                delete m_drawableManager;
                m_sceneTree = viewState->getSceneTree();

                m_drawableManager = createDrawableManager( m_resourceManager );
                m_drawableManager->setSceneTree( m_sceneTree );
                m_drawableManager->setEnvironmentSampler( getEnvironmentRenderingEnabled() ? getEnvironmentSampler() : dp::sg::core::SamplerSharedPtr::null );
                m_drawableManager->update( m_viewportSize );

                DrawableManagerDefault* drawableManagerDefault = dynamic_cast<DrawableManagerDefault*>( m_drawableManager );
                drawableManagerDefault->setCullingEnabled( m_cullingEnabled );

                m_transparencyManager->setShaderManager( drawableManagerDefault->getShaderManager() );
                m_transparencyManager->useParameterContainer( m_resourceManager->getRenderer(), drawableManagerDefault->getRenderGroupTransparent() );
                m_transparencyManager->useParameterContainer( m_resourceManager->getRenderer(), drawableManagerDefault->getRenderGroupTransparentDepthPass() );
              }

              // Refresh all observed data
              {
                // Refresh the Scene Tree (the caches need to be filled for the first render)
                m_sceneTree->update(viewState->getCamera(), viewState->getLODRangeScale());

                // Refresh the observed resources
                m_resourceManager->updateResources();
              }

              doRenderDrawables( viewState, renderTargetGL );            }
          }

          DrawableManager* SceneRendererImpl::createDrawableManager( const ResourceManagerSharedPtr &resourceManager ) const
          {
            DrawableManagerDefault * dmd = new DrawableManagerDefault( resourceManager, m_transparencyManager, m_shaderManager, m_cullingMode );
            dmd->setEnvironmentSampler( getEnvironmentSampler() );
            dmd->setCullingEnabled( m_cullingEnabled );

            return( dmd );
          }

          void SceneRendererImpl::doRenderEnvironmentMap( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::gl::RenderTargetSharedPtr const& renderTarget )
          {
            NSIGHT_START_RANGE( "EnvironmentMap" );
            // must not be called if backdrop is deactivated
            DP_ASSERT(m_environmentRenderer);

            // texture coord setup for mapping the backdrop
            //
            dp::sg::core::FrustumCameraSharedPtr const& theCamera = viewState->getCamera().staticCast<dp::sg::core::FrustumCamera>();

            dp::math::Vec3f lookdir = theCamera->getDirection();
            DP_ASSERT( isNormalized( lookdir ) );

            dp::math::Vec3f up = theCamera->getUpVector();
            DP_ASSERT( isNormalized( up ) );

            dp::math::Vec3f camera_u = lookdir ^ up;
            dp::math::Vec3f camera_v = camera_u ^ lookdir;
            normalize(camera_u);
            normalize(camera_v);

            float focusDistance = theCamera->getFocusDistance();
            dp::math::Vec2f wo = theCamera->getWindowOffset();  // default (0, 0)
            dp::math::Vec2f ws = theCamera->getWindowSize();    // default (1, 1)
            dp::math::Box2f wr = theCamera->getWindowRegion();
            //theCamera->getWindowRegion(ll, ur); // defaults (0, 0) and (1, 1)
            dp::math::Vec2f ll = wr.getLower();
            dp::math::Vec2f ur = wr.getUpper();

            // This is the window into the world at focus distance.
            float l = wo[0] - 0.5f * ws[0];
            float b = wo[1] - 0.5f * ws[1];

            //  adjust the l/r/b/t values to the window region to view
            float r = l + ur[0] * ws[0];
            l += ll[0] * ws[0];

            float t = b + ur[1] * ws[1];
            b += ll[1] * ws[1];

            // The vector to the window region center.
            lookdir = lookdir * focusDistance;
            lookdir += camera_u * 0.5f * (l + r);
            lookdir += camera_v * 0.5f * (b + t);

            // Half sized vector of window region u and v directions.
            camera_u *= 0.5f * (r - l);
            camera_v *= 0.5f * (t - b);

            // Mind, lookdir and camera_u, camera_v are not normalized!

            // Seeding the texture coordinate 2 of the FSQ with the frustum corner vectors.
            std::vector<dp::math::Vec4f> worldFrustum;
            worldFrustum.push_back(dp::math::Vec4f(lookdir - camera_u - camera_v, 0.0f)); // lower left
            worldFrustum.push_back(dp::math::Vec4f(lookdir + camera_u - camera_v, 0.0f)); // lower right
            worldFrustum.push_back(dp::math::Vec4f(lookdir + camera_u + camera_v, 0.0f)); // upper right
            worldFrustum.push_back(dp::math::Vec4f(lookdir - camera_u + camera_v, 0.0f)); // upper left

            m_environmentRenderer->setTexCoords(2, worldFrustum);
            m_environmentRenderer->render( renderTarget );
            NSIGHT_STOP_RANGE();
          }

          void SceneRendererImpl::doRenderDrawables( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::gl::RenderTargetSharedPtr const& renderTarget )
          {
            PROFILE( "Render");

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

            NSIGHT_START_RANGE( "Frame" );
            dp::gl::RenderTargetSharedPtr const & renderTargetGL = dp::util::shared_cast<dp::gl::RenderTarget>(renderTarget);
            dp::gl::TargetBufferMask clearMask = renderTargetGL->getClearMask();
            if ( getEnvironmentRenderingEnabled() )
            {
              // render the backdrop instead of clearing the color buffer
              doRenderEnvironmentMap( viewState, renderTarget );
            }

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
                m_transparencyManager->beginTransparentPass( m_renderer.get() );
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
                  std::vector<dp::rix::core::GeometryInstanceSharedHandle> const & sortedGIs = drawableManagerDefault->getSortedTransparentGIs( camera->getPosition() );
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

          void SceneRendererImpl::onEnvironmentSamplerChanged()
          {
            if ( m_drawableManager )
            {
              m_drawableManager->setEnvironmentSampler( getEnvironmentSampler() );
            }
          }

          void SceneRendererImpl::onEnvironmentRenderingEnabledChanged()
          {
            DP_ASSERT( getRenderTarget().dynamicCast<dp::gl::RenderTarget>() );
            dp::gl::RenderTargetSharedPtr glRenderTarget = getRenderTarget().staticCast<dp::gl::RenderTarget>();
            if ( getEnvironmentRenderingEnabled() )
            {
              m_environmentRenderer = dp::sg::renderer::rix::gl::FSQRenderer::create( glRenderTarget );
              m_environmentRenderer->setEffect( dp::sg::core::EffectData::create( dp::fx::EffectLibrary::instance()->getEffectSpec( std::string("environmentBackdrop") ) ) );
              m_environmentRenderer->setSamplerByName( "environment", getEnvironmentSampler() );
              glRenderTarget->setClearMask( glRenderTarget->getClearMask() & ~dp::gl::TBM_COLOR_BUFFER );
            }
            else
            {
              m_environmentRenderer.reset();
              glRenderTarget->setClearMask( glRenderTarget->getClearMask() | dp::gl::TBM_COLOR_BUFFER );
            }
            if ( m_drawableManager )
            {
              m_drawableManager->setEnvironmentSampler( getEnvironmentRenderingEnabled() ? getEnvironmentSampler() : dp::sg::core::SamplerSharedPtr::null );
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
              m_transparencyManager->initializeParameterContainer( m_renderer.get(), m_viewportSize );
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
            if ( renderEngine != m_renderEngineOptions )
            {
              m_renderEngineOptions = renderEngine;
              shutdownRenderer();
            }
          }

          std::string const& SceneRendererImpl::getRenderEngine() const
          {
            return m_renderEngineOptions;
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

              dp::rix::core::PFNCREATERENDERER createRenderer = reinterpret_cast<dp::rix::core::PFNCREATERENDERER>(m_rix->getSymbol("createRenderer"));
              m_renderer.reset(dynamic_cast<dp::rix::gl::RiXGL*>((*createRenderer)( m_renderEngineOptions.c_str() )));
              DP_ASSERT( m_renderer && "Could not create RiXGL renderer" );

              m_resourceManager = ResourceManager::create( m_renderer.get(), m_shaderManager );

              m_rendererInitialized = m_renderer && m_resourceManager;

              if ( !m_rendererInitialized )
              {
                shutdownRenderer();
              }
              else
              {
                m_transparencyManager->initializeParameterContainer( m_renderer.get(), m_viewportSize );
              }
            }

            return m_rendererInitialized;
          }

          dp::sg::renderer::rix::gl::TransparencyManagerSharedPtr const & SceneRendererImpl::getTransparencyManager() const
          {
            return( m_transparencyManager );
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

