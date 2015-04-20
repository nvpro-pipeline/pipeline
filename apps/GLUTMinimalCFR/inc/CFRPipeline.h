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


#pragma once

#include <dp/sg/ui/SceneRenderer.h>
#include <dp/sg/renderer/rix/gl/SceneRenderer.h>
#include <dp/sg/renderer/rix/gl/FSQRenderer.h>

#include "TextureTransfer.h"

DEFINE_PTR_TYPES( CFRPipeline );

class CFRPipeline : public dp::sg::ui::SceneRenderer
{
protected:
  CFRPipeline( const char* renderEngine = 0,
               dp::fx::Manager shaderManagerType = dp::fx::MANAGER_SHADERBUFFER,
               dp::culling::Mode cullingMode = dp::culling::MODE_AUTO,
               const dp::gl::RenderTargetSharedPtr &renderTarget = dp::gl::RenderTargetSharedPtr() );

public:
  static CFRPipelineSharedPtr create( const char* renderEngine = 0,
                                      dp::fx::Manager shaderManagerType = dp::fx::MANAGER_SHADERBUFFER,
                                      dp::culling::Mode cullingMode = dp::culling::MODE_AUTO,
                                      const dp::gl::RenderTargetSharedPtr &renderTarget = dp::gl::RenderTargetSharedPtr() );
  ~CFRPipeline();

  void setTileSize( size_t width, size_t height );

  bool init(const dp::gl::RenderContextSharedPtr &renderContext, const dp::gl::RenderTargetSharedPtr &renderTarget);
  void resize( size_t width, size_t height );

  virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const;

  virtual void setCullingEnabled( bool enabled );
  virtual bool isCullingEnabled() const;

  virtual void setCullingMode( dp::culling::Mode mode );
  virtual dp::culling::Mode getCullingMode( ) const;

  virtual void setShaderManager( dp::fx::Manager shaderManager );
  virtual dp::fx::Manager getShaderManager() const;

protected:
  virtual void doRender(dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTargetSharedPtr const& renderTarget);
  virtual void onEnvironmentSamplerChanged();
  virtual void onEnvironmentRenderingEnabledChanged();

private:
  DEFINE_PTR_TYPES( MonoViewStateProvider );

  class MonoViewStateProvider : public SceneRenderer::StereoViewStateProvider
  {
  public:
    static MonoViewStateProviderSharedPtr create();

  protected:
    dp::sg::ui::ViewStateSharedPtr calculateViewState( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTarget::StereoTarget eye )
    {
      return viewState;  // NOP, this lets SceneRenderer::doRender() use the incoming ViewState camera.
    };
  };

private:
  struct GpuData
  {
    dp::sg::ui::SceneRendererSharedPtr  m_sceneRenderer;
    dp::gl::RenderTargetSharedPtr       m_renderTarget;
    TextureTransferSharedPtr            m_textureTransfer;
  };

private:
  void generateStencilPattern( dp::gl::RenderTargetSharedPtr renderTarget );

private:
  std::string       m_renderEngine;    // the render engine used to render the images
  dp::fx::Manager   m_shaderManager;   // the shader manager for the renderers
  dp::culling::Mode m_cullingMode;     // culling mode used for rendering

  size_t m_primaryGpuIndex; // index of the primary gpu (for special handling: no transfer, etc.)

  size_t m_rendererCount;   // number of renderers used for checkered frame rendering
  size_t m_tileWidth;       // checker tile width
  size_t m_tileHeight;      // checker tile height

  size_t m_targetWidth;     // width and height of the rendertarget
  size_t m_targetHeight;    // cached to check whether a resize is necessary

  std::vector< GpuData > m_gpuData;

  dp::sg::renderer::rix::gl::FSQRendererSharedPtr m_outputRenderer;   // renderer for compositing the output image
  dp::gl::Texture2DSharedPtr                      m_compositeTexture; // texture to composite the image data from the different contexts

  MonoViewStateProviderSharedPtr m_monoViewStateProvider;

};

