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

#include <test/sgrdr/framework/inc/Config.h>

#include <test/testfw/core/Backend.h>

#include <dp/gl/RenderTargetFBO.h>

#include <dp/util/DynamicLibrary.h>

#include <dp/sg/ui/ViewState.h>
#include <dp/sg/ui/SceneRenderer.h>

namespace dp
{
  namespace sgrdr
  {
    namespace test
    {
      namespace framework
      {

        class RenderDataSgRdr : public dp::testfw::core::RenderData
        {
        public:
          DPTSGRDR_API RenderDataSgRdr();
          DPTSGRDR_API RenderDataSgRdr(dp::sg::ui::ViewStateSharedPtr const& viewState);
          DPTSGRDR_API virtual ~RenderDataSgRdr();

          DPTSGRDR_API void setViewState(dp::sg::ui::ViewStateSharedPtr const& viewState);
          DPTSGRDR_API dp::sg::ui::ViewStateSharedPtr const& getViewState() const;

        private:
          dp::sg::ui::ViewStateSharedPtr m_viewState;
        };

        class SgRdrBackend : public testfw::core::Backend
        {
        public:
          DPTSGRDR_API SgRdrBackend( const std::string& rendererName, const std::vector<std::string>& options );

        public:
          DPTSGRDR_API virtual ~SgRdrBackend();
          DPTSGRDR_API virtual dp::ui::SmartRenderTarget createDisplay(int width, int height, bool visible);

          DPTSGRDR_API virtual void render( dp::testfw::core::RenderData* renderData, dp::ui::SmartRenderTarget renderTarget = nullptr );
          DPTSGRDR_API virtual void finish();

          DPTSGRDR_API dp::sg::ui::SmartSceneRenderer getRenderer() const;

        protected:
          dp::gl::RenderContextFormat m_format;
          dp::gl::SmartRenderContext  m_context;
          int                         m_windowId;

          dp::sg::ui::SmartSceneRenderer m_renderer;
        };

        typedef util::SmartPtr<SgRdrBackend> SmartSgRdrBackend;
      } // namespace framework
    } // namespace test
  } // namespace sgrdr
} // namespace dp

#ifdef DPTSGRDR_EXPORTS
extern "C"
{

  DPTSGRDR_API int getNumSupportedRenderers()
  {
    return 2;
  }

  DPTSGRDR_API const char* getSupportedRenderer(int i)
  {
    switch( i )
    {
    case 0:
      return "RiXGL.rdr";
    default:
      return nullptr;
    }
  }

  DPTSGRDR_API bool isRendererSupported( const char* rendererName )
  {
    return !strcmp(rendererName, "RiXGL.rdr");
  }

  DPTSGRDR_API dp::sgrdr::test::framework::SgRdrBackend * create( const char* rendererName, const std::vector<std::string>* options )
  {
    if( !isRendererSupported(rendererName) )
    {
      return nullptr;
    }

    return new dp::sgrdr::test::framework::SgRdrBackend( std::string(rendererName)
                                                         , *options );
  }
}
#endif