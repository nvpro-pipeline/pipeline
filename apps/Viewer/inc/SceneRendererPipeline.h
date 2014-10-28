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
#include <dp/gl/RenderTargetFBO.h>
#include <dp/sg/renderer/rix/gl/SceneRenderer.h>
#include <dp/sg/renderer/rix/gl/FSQRenderer.h>

struct TonemapperValues
{
  float gamma;
  float whitePoint;
  float brightness;
  float saturation;
  float crushBlacks;
  float burnHighlights;
};

SMART_TYPES( SceneRendererPipeline );

class SceneRendererPipeline : public dp::sg::ui::SceneRenderer
{
public:
  static SmartSceneRendererPipeline create();
  ~SceneRendererPipeline();

  bool init(const dp::gl::SharedRenderContext &renderContext, const dp::gl::SharedRenderTarget &renderTarget);

  void setSceneRenderer(const dp::sg::ui::SmartSceneRenderer &sceneRenderer);
  dp::sg::ui::SmartSceneRenderer getSceneRenderer() const; 
  void enableHighlighting(bool onOff);
  
  void updateEnvironment();

  void setTransparencyMode( dp::sg::renderer::rix::gl::TransparencyMode mode );
  dp::sg::renderer::rix::gl::TransparencyMode getTransparencyMode() const;

  void setTonemapperEnabled( bool enabled );
  bool isTonemapperEnabled() const;
  TonemapperValues getTonemapperValues() const;
  void setTonemapperValues( const TonemapperValues& values );

  virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::sg::core::GeoNodeSharedPtr & geoNode, bool depthPass ) const;

  virtual void setCullingEnabled( bool enabled );
  virtual bool isCullingEnabled() const;

  virtual void setCullingMode( dp::culling::Mode mode );
  virtual dp::culling::Mode getCullingMode( ) const;

  virtual void setShaderManager( dp::fx::Manager shaderManager );
  virtual dp::fx::Manager getShaderManager() const;

protected:
  SceneRendererPipeline();
  virtual void doRender(dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget);
  virtual void onEnvironmentSamplerChanged();

private:
  SMART_TYPES( MonoViewStateProvider );
  class MonoViewStateProvider : public SceneRenderer::StereoViewStateProvider
  {
  public:
    static SmartMonoViewStateProvider create();

  protected:
    dp::sg::ui::ViewStateSharedPtr calculateViewState( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::RenderTarget::StereoTarget eye )
    {
      return viewState;  // NOP, this lets SceneRenderer::doRender() use the incoming ViewState camera.
    };
  };

  // helpers called from doRender to keep the code more readable
  void doRenderBackdrop(dp::sg::ui::ViewStateSharedPtr const& viewState);
  void doRenderTonemap(dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget);
  void doRenderStandard(dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget);
  void doRenderHighlight(dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget);
  void initBackdrop();
  void initTonemapper();

private:
  dp::sg::ui::SmartSceneRenderer                m_sceneRenderer; // The renderer for the main image on the framebuffer (rasterizer or ray tracer).  
  SmartMonoViewStateProvider                    m_monoViewStateProvider;

  dp::gl::SharedRenderTarget                    m_renderTarget;             // The render target passed into init.
  dp::gl::SharedRenderTargetFBO                 m_tonemapFBO;               // The monoscopic FBO for the tonemap texture rendering and processing.
  dp::gl::SharedRenderTargetFBO                 m_highlightFBO;             // The monoscopic FBO for the highlight rendering and processing.
  dp::sg::renderer::rix::gl::SmartSceneRenderer m_sceneRendererHighlight;   // The renderer for the highlighted objects into the FBO.
  dp::sg::renderer::rix::gl::SmartFSQRenderer   m_rendererStencilToColor;
  dp::sg::renderer::rix::gl::SmartFSQRenderer   m_rendererHighlight;
  dp::sg::renderer::rix::gl::SmartFSQRenderer   m_environmentBackdrop;
  dp::sg::renderer::rix::gl::SmartFSQRenderer   m_tonemapper;
  bool                                          m_highlighting;
  bool                                          m_backdropEnabled;
  bool                                          m_tonemapperEnabled;

  // Tonemapper values in the GUI:
  TonemapperValues m_tonemapperValues;
  bool             m_tonemapperValuesChanged;

  // Derived tonemapper data in the shader:
  dp::sg::core::EffectDataSharedPtr m_tonemapperData;
};
