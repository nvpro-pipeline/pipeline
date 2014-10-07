// Copyright NVIDIA Corporation 2011
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


#include <dp/gl/RenderTargetFBO.h>
#include <dp/math/Vecnt.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/sg/gl/TextureGL.h>
#include <dp/util/SharedPtr.h>

#include "CFRPipeline.h"
#include "TextureTransfer.h"

#include <GL/wglew.h>

using namespace dp::fx;
using std::string;
using std::vector;

#define CFRPIPELINE_PRIMARY_GPU_IMPROVEMENT 0

dp::util::SmartPtr<CFRPipeline::MonoViewStateProvider> CFRPipeline::MonoViewStateProvider::create()
{
  return new CFRPipeline::MonoViewStateProvider();
}

// This does nothing except for providing the setSceneRenderer() override which 
// allows to set the scene renderer before any OpenGL resources have been allocated.
CFRPipeline::CFRPipeline( const char *renderEngine,
                          dp::fx::Manager shaderManagerType,
                          dp::culling::Mode cullingMode,
                          const dp::gl::SharedRenderTarget &renderTarget)
  : dp::sg::ui::SceneRenderer( renderTarget )
  , m_renderEngine( renderEngine )
  , m_shaderManager( shaderManagerType )
  , m_cullingMode( cullingMode )
  , m_primaryGpuIndex( ~0 )
  , m_rendererCount( 0 )
  , m_tileWidth( 16 )
  , m_tileHeight( 12 )
  , m_targetHeight( 0 )
  , m_targetWidth( 0 )
{
  m_monoViewStateProvider = MonoViewStateProvider::create();
}

CFRPipeline::~CFRPipeline()
{
}

SmartCFRPipeline CFRPipeline::create( const char *renderEngine,
                                      dp::fx::Manager shaderManagerType,
                                      dp::culling::Mode cullingMode,
                                      const dp::gl::SharedRenderTarget &renderTarget)
{
  return new CFRPipeline( renderEngine, shaderManagerType, cullingMode, renderTarget );
}

void CFRPipeline::setTileSize( size_t width, size_t height )
{
  DP_ASSERT( !"not yet functional" );
  m_tileWidth  = width;
  m_tileHeight = height;
}

// This needs to be called on-demand in doRender
bool CFRPipeline::init( const dp::gl::SharedRenderContext &renderContext,
                        const dp::gl::SharedRenderTarget  &renderTarget )
{
  // collect information about available GPUs
  unsigned int gpuIndex = 0;
  std::vector< HGPUNV > gpus;
  HGPUNV gpu;
  while( wglEnumGpusNV( gpuIndex, &gpu ) )
  {
    std::cout << "GPU " << gpuIndex;
    gpus.push_back( gpu );

    // find out which GPU has the primary display 
    // this GPU is used for rendering in win7, so we can use the renderTarget that was passed

#if CFRPIPELINE_PRIMARY_GPU_IMPROVEMENT
    unsigned int deviceIndex = 0;
    GPU_DEVICE gpuDevice;
    while( wglEnumGpuDevicesNV( gpu, deviceIndex, &gpuDevice ) )
    {
      if( gpuDevice.Flags & DISPLAY_DEVICE_PRIMARY_DEVICE )
      {
        std::cout << " PRIMARY DISPLAY ATTACHED";
        m_primaryGpuIndex = gpuIndex;
        break;
      }
      ++deviceIndex;
    }
#endif

    ++gpuIndex;
    std::cout << "\n";
    if( gpuIndex == 3 )
    {
      //break;
    }
  }
  m_rendererCount = gpus.size();

  m_gpuData.resize( m_rendererCount );

  m_compositeTexture = dp::gl::Texture2D::create( GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE );

  for ( size_t i = 0; i < m_rendererCount; ++i )
  {
    GpuData& gpuData = m_gpuData[i];
 
    gpuData.m_sceneRenderer = dp::sg::renderer::rix::gl::SceneRenderer::create( m_renderEngine.c_str(), m_shaderManager, m_cullingMode );

    if( i == m_primaryGpuIndex )
    {
      // for the primary gpu, just use the rendertarget used for the resulting image
      gpuData.m_renderTarget = renderTarget;

      // this gpuData doesn't get a TextureTransfer object, as no transfer is needed
    }
    else
    {
      // all other gpus get an own render context and render target that will be copied to the final rendertarget 
      std::vector< HGPUNV > gpu;
      gpu.push_back( gpus[i] );

      dp::gl::RenderContextFormat format;
      dp::gl::SharedRenderContext context = dp::gl::RenderContext::create( dp::gl::RenderContext::Headless( &format, nullptr, gpu ) );

      dp::gl::SharedRenderTargetFBO rt = dp::gl::RenderTargetFBO::create(context);
      gpuData.m_renderTarget = rt;

      rt->beginRendering();
      // Render to rectangle texture.
      rt->setAttachment(dp::gl::RenderTargetFBO::COLOR_ATTACHMENT0, dp::gl::Texture2D::create(GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE));

      // Depth and Stencil are Renderbuffers.
      dp::gl::SharedRenderbuffer depthStencil0(dp::gl::Renderbuffer::create(GL_DEPTH24_STENCIL8)); // maybe use GL_DEPTH32F_STENCIL8
      rt->setAttachment(dp::gl::RenderTargetFBO::DEPTH_ATTACHMENT,   depthStencil0);
      rt->setAttachment(dp::gl::RenderTargetFBO::STENCIL_ATTACHMENT, depthStencil0);

      rt->endRendering();

      SmartTextureTransfer tt = new TextureTransfer( renderContext, context );
      tt->setTileSize( m_tileWidth, m_tileHeight );
      tt->setMaxIndex( m_rendererCount );
      gpuData.m_textureTransfer = tt;
    }
  }

  m_outputRenderer = dp::sg::renderer::rix::gl::FSQRenderer::create( renderTarget );

  return true;
}

void CFRPipeline::resize( size_t width, size_t height )
{
  m_compositeTexture->resize( dp::util::checked_cast<unsigned int>( width ), dp::util::checked_cast<unsigned int>( height ) );

  for ( size_t i = 0; i < m_rendererCount; ++i )
  {
    GpuData& gpuData = m_gpuData[i];
    if( i != m_primaryGpuIndex )
    {
      // render target of primary gpu doesn't need a resize (it's the final rendertarget)
      gpuData.m_renderTarget->setSize( dp::util::checked_cast<unsigned int>( width ), dp::util::checked_cast<unsigned int>( height ) );
    }

    gpuData.m_renderTarget->setClearMask( dp::gl::TBM_COLOR_BUFFER | dp::gl::TBM_DEPTH_BUFFER | dp::gl::TBM_STENCIL_BUFFER );
    gpuData.m_renderTarget->beginRendering();

    // enable stencil test, set stencil func/value to only render to tile with value i
    glEnable( GL_STENCIL_TEST );
    glStencilOp( GL_KEEP, GL_KEEP, GL_KEEP );
    glStencilFunc( GL_EQUAL, dp::util::checked_cast<GLuint>(i), ~0 );

    // initialize stencil buffer of render targets with a regular pattern
    generateStencilPattern( gpuData.m_renderTarget );

    gpuData.m_renderTarget->endRendering();
    gpuData.m_renderTarget->setClearMask( dp::gl::TBM_COLOR_BUFFER | dp::gl::TBM_DEPTH_BUFFER );
  }
}


// Mind, this is called for left and right eye independently.
void CFRPipeline::doRender(dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget)
{
  const dp::gl::SharedRenderTarget renderTargetGL = dp::util::shared_cast<dp::gl::RenderTarget>(renderTarget);
  DP_ASSERT( renderTargetGL );
  if( m_gpuData.empty() )
  {
    init( renderTargetGL->getRenderContext() , renderTargetGL );
  }

  unsigned int width;
  unsigned int height;
  renderTarget->getSize(width, height);

  if( width != m_targetWidth || height != m_targetHeight )
  {
    m_targetWidth = width;
    m_targetHeight = height;
    resize( width, height );
  }

  // TODO: initialize the FBOs with a checker pattern here if we want to use a depth value

  // render the scene in all FBOS
  for( size_t i = 0; i < m_rendererCount; ++i )
  {
    m_gpuData[i].m_sceneRenderer->render( viewState, m_gpuData[i].m_renderTarget, renderTarget->getStereoTarget() );
  }

  // combine the FBOs into the output render target

  for( size_t i = 0; i < m_rendererCount; ++i )
  {
    // don't copy image data on the primary gpu (the image is already on the right frame buffer)
    if( i == m_primaryGpuIndex )
    {
      continue;
    }

    GpuData& gpuData = m_gpuData[i];

    const dp::gl::RenderTargetFBO::SharedAttachment &attachment = dp::util::shared_cast<dp::gl::RenderTargetFBO>(gpuData.m_renderTarget)->getAttachment(dp::gl::RenderTargetFBO::COLOR_ATTACHMENT0);
    const dp::gl::RenderTargetFBO::SharedAttachmentTexture &texAtt = dp::util::shared_cast<dp::gl::RenderTargetFBO::AttachmentTexture>(attachment);

    DP_ASSERT( texAtt );

    // HACK: workaround, glDispatchCompute (called in tt.transfer()) seems to need a valid framebuffer
    gpuData.m_renderTarget->setClearMask(0);
    gpuData.m_renderTarget->beginRendering();

    gpuData.m_textureTransfer->transfer( i, m_compositeTexture, dp::util::shared_cast<dp::gl::Texture2D>(texAtt->getTexture()) );

    gpuData.m_renderTarget->endRendering();
    gpuData.m_renderTarget->setClearMask( dp::gl::TBM_COLOR_BUFFER | dp::gl::TBM_DEPTH_BUFFER );
  }

//   for( size_t i = 0; i < m_rendererCount; ++i )
//   {
//     std::stringstream filename;
//     filename << "z:\\tmp\\debugimage_" << i << ".png";
//     dp::util::imageToFile( m_gpuData[i].m_renderTarget->getImage(), filename.str() );
//   }

  // present the composited texture from the other (non-primary-) GPUs on the primary gpu.
  renderTargetGL->setClearMask( 0 );
  renderTargetGL->beginRendering();

  // present via a fixed function FSQ draw, disable shaders
  glUseProgram( 0 );

#if CFRPIPELINE_PRIMARY_GPU_IMPROVEMENT
  //  use the stencil buffer to composite correctly (draw all except the primary gpu tiles)
  glStencilFunc( GL_NOTEQUAL, dp::util::checked_cast<GLuint>(m_primaryGpuIndex), ~0 );
#endif

  m_outputRenderer->presentTexture2D( m_compositeTexture, renderTargetGL, false );

#if CFRPIPELINE_PRIMARY_GPU_IMPROVEMENT
  glStencilFunc( GL_EQUAL, dp::util::checked_cast<GLuint>(m_primaryGpuIndex), ~0 );
#endif

  renderTargetGL->endRendering();
  renderTargetGL->setClearMask( dp::gl::TBM_COLOR_BUFFER | dp::gl::TBM_DEPTH_BUFFER );

//   dp::util::imageToFile( renderTargetGL->getImage(), "z:\\tmp\\output.png" );
}

void CFRPipeline::onEnvironmentSamplerChanged()
{
  DP_ASSERT(false);
  /*
  DP_ASSERT( m_sceneRenderer );
  m_sceneRenderer->setEnvironmentSampler( getEnvironmentSampler() );
  */
}

std::map<dp::fx::Domain,std::string> CFRPipeline::getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const
{
  DP_ASSERT(false);
  /*
  DP_ASSERT( m_sceneRenderer );
  return( m_sceneRenderer->getShaderSources( geoNode, depthPass ) );
  */
  return std::map<dp::fx::Domain,std::string>();
}

void CFRPipeline::setCullingEnabled( bool enabled )
{
  DP_ASSERT(false);
  /*
  DP_ASSERT( m_sceneRenderer );
  m_sceneRenderer->setCullingEnabled( enabled );
  */
}

bool CFRPipeline::isCullingEnabled() const
{
  DP_ASSERT(false);
  /*
  DP_ASSERT( m_sceneRenderer );
  return( m_sceneRenderer->isCullingEnabled() );
  */
  return false;
}

void CFRPipeline::setCullingMode( dp::culling::Mode mode )
{
  DP_ASSERT(false);
  /*
  DP_ASSERT( m_sceneRenderer );
  m_sceneRenderer->setCullingMode( mode );
  */
}

dp::culling::Mode CFRPipeline::getCullingMode( ) const
{
  DP_ASSERT(false);
  /*
  DP_ASSERT( m_sceneRenderer );
  return( m_sceneRenderer->getCullingMode() );
  */
  return dp::culling::Mode();
}

void CFRPipeline::setShaderManager( dp::fx::Manager manager )
{
  DP_ASSERT(false);
  /*
  DP_ASSERT( m_sceneRenderer );
  m_sceneRenderer->setShaderManager( manager );
  */
}

dp::fx::Manager CFRPipeline::getShaderManager( ) const
{
  return( m_shaderManager );
}

void CFRPipeline::generateStencilPattern( dp::gl::SharedRenderTarget renderTarget )
{
  unsigned int width;
  unsigned int height;
  renderTarget->getSize(width, height);

  size_t tilesX = (width  + m_tileWidth  - 1) / m_tileWidth;  // number of horizontal tiles
  size_t tilesY = (height + m_tileHeight - 1) / m_tileHeight; // number of vert1ical tiles

  GLboolean scissorEnabled = glIsEnabled(GL_SCISSOR_TEST);
  int scissorBox[4] = {0, 0, -1, -1};
  glGetIntegerv(GL_SCISSOR_BOX, scissorBox);
  glEnable( GL_SCISSOR_TEST );

  for( size_t tileY = 0; tileY < tilesY; ++tileY )
  {
    for( size_t tileX = 0; tileX < tilesX; ++tileX )
    {
      size_t x = tileX * m_tileWidth;
      size_t y = tileY * m_tileHeight;
      GLuint value = dp::util::checked_cast<GLuint>( (tileX + m_rendererCount - (tileY % m_rendererCount) ) % m_rendererCount );

      glScissor( dp::util::checked_cast<GLuint>(x),            dp::util::checked_cast<GLuint>(y)
               , dp::util::checked_cast<GLsizei>(m_tileWidth), dp::util::checked_cast<GLsizei>(m_tileHeight) );

      glClearBufferuiv( GL_STENCIL, 0, &value );
    }
  }

  if (!scissorEnabled)
  {
    glDisable(GL_SCISSOR_TEST);
  }
  glScissor(scissorBox[0], scissorBox[1], scissorBox[2], scissorBox[3]);
}
