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

#include <test/rix/gl/framework/inc/Config.h>
#include <test/rix/core/framework/RiXBackend.h>

#include <dp/gl/RenderContext.h>
#include <dp/gl/RenderContextFormat.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      namespace test
      {
        namespace framework
        {
          class RiXGLBackend : public core::test::framework::RiXBackend
          {
          public:
            DPTRIXGL_API RiXGLBackend(const char* renderer, const char* options);

          public:
            DPTRIXGL_API virtual ~RiXGLBackend();
            DPTRIXGL_API virtual dp::ui::SmartRenderTarget createDisplay(int width, int height, bool visible);
            DPTRIXGL_API virtual dp::ui::SmartRenderTarget createAuxiliaryRenderTarget(int width, int height);
            DPTRIXGL_API virtual void finish();

            template <typename RTGLImpl>
            static dp::ui::SmartRenderTarget createContextedRenderTarget(dp::gl::SharedRenderContext glContext)
            {
              return RTGLImpl::create( glContext );
            }

          protected:
            dp::gl::RenderContextFormat m_format;
            dp::gl::SharedRenderContext m_context;

            int m_windowId;
          };

          typedef util::SmartPtr<RiXGLBackend> SmartRiXGLBackend;
        } // namespace framework
      } // namespace gl
    } // namespace RiX
  } // namespace util
} // namespace dp

#ifdef DPTRIXGL_EXPORTS
extern "C"
{
  DPTRIXGL_API dp::rix::gl::test::framework::RiXGLBackend * create(const char* rendererName, const std::vector<std::string>* options)
  {
    std::vector<std::string>::const_iterator it = std::find(options->begin(), options->end(), std::string("--renderengine"));

    const char* renderEngine = it == options->end() || it+1 == options->end()
                               ? "Bindless"
                               : (it+1)->c_str();

    return new dp::rix::gl::test::framework::RiXGLBackend( rendererName, renderEngine );
  }

  DPTRIXGL_API int getNumSupportedRenderers()
  {
    return 1;
  }

  DPTRIXGL_API const char* getSupportedRenderer(int i)
  {
    return i == 0 ? "RiXGL.rdr" : nullptr;
  }

  DPTRIXGL_API bool isRendererSupported( const char* rendererName )
  {
    return !strcmp(rendererName, "RiXGL.rdr");
  }
}
#endif