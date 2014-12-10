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


#include <test/testfw/core/TestRender.h>
#include <test/rix/core/framework/RiXBackend.h>

#include <dp/rix/core/RiX.h>
#include <dp/rix/gl/RiXGL.h>
#include <dp/gl/RenderTargetFBO.h>
#include <dp/gl/Texture.h>

#include <dp/math/Trafo.h>

class Feature_FBO : public dp::testfw::core::TestRender
{
public:
  Feature_FBO();
  ~Feature_FBO();

  bool onInit( void );
  bool onRunInit( unsigned int i );
  bool onRun( unsigned int idx );
  bool onClear( void );
  bool onRunCheck( unsigned int i );

  bool option( const std::vector<std::string>& optionString );

private:

  void createCamera( void );

  void createScene( void );
  void createSecondPass( void );

  void setupCamera( dp::math::Vec3f eye, dp::math::Vec3f center, dp::math::Vec3f up );

private:

  dp::rix::core::test::framework::RenderDataRiX* m_renderData;
  dp::rix::core::Renderer* m_rix;

  dp::rix::core::ContainerDescriptorSharedHandle m_vertContainerDescriptorCamera;
  dp::rix::core::ContainerSharedHandle m_vertViewProjContainer;
  dp::rix::core::ContainerEntry m_containerEntryWorld2view;
  dp::rix::core::ContainerEntry m_containerEntryView2world;
  dp::rix::core::ContainerEntry m_containerEntryView2clip;
  dp::rix::core::ContainerEntry m_containerEntryWorld2clip;
  dp::rix::core::ContainerSharedHandle m_fragConstContainerScene;
  dp::rix::core::ContainerEntry m_containerEntryLightDir;
  dp::rix::core::RenderGroupSharedHandle m_renderGroupScene;

  dp::rix::core::ContainerSharedHandle m_fragContainerScreenPass;
  dp::rix::core::ContainerEntry m_containerEntryFBOTexture;

  dp::rix::core::RenderGroupSharedHandle m_renderGroupSecondPass;




  float m_aspectRatio;
  float m_nearPlane;
  float m_farPlane;
  float m_fovy;

  dp::math::Mat44f m_view2Clip;
  dp::math::Mat44f m_world2View;
  GLuint m_textureName;
  GLuint m_depthTextureName;

  dp::gl::Texture2DSharedPtr m_colorBuf;
  dp::gl::Texture2DSharedPtr m_depthBuf;

  dp::ui::RenderTargetSharedPtr m_fbo;
  bool m_screenshotFBO;
  std::string m_screenshotFBOName;
};

extern "C"
{
  DPTTEST_API dp::testfw::core::Test * create_feature_FBO()
  {
    return new Feature_FBO();
  }
}
