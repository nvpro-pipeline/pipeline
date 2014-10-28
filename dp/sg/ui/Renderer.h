// Copyright NVIDIA Corporation 2010
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

#include <dp/ui/RenderTarget.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/Scene.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {

      /** \brief Renderer is the base class for algorithms which operate on an dp::ui::RenderTarget.
      **/
      class Renderer : public dp::util::Reflection
      {
      protected:
        DP_SG_UI_API Renderer( const dp::ui::SmartRenderTarget &renderTarget = dp::ui::SmartRenderTarget() );

      public:
        DP_SG_UI_API virtual ~Renderer();

        /** \brief Set the dp::ui::RenderTarget used by all subsequent render calls. 
            \param renderTarget The new dp::ui::RenderTarget.
        **/
        DP_SG_UI_API void setRenderTarget( const dp::ui::SmartRenderTarget &renderTarget );

        /** \brief Get the dp::ui::RenderTarget used by render calls.
            \return The current dp::ui::RenderTarget.
        **/
        DP_SG_UI_API dp::ui::SmartRenderTarget getRenderTarget() const;

        /** \brief Executes the rendering algorithm on the given dp::ui::RenderTarget. It calls beginRendering,
                   doRender and endRendering with an dp::ui::RenderTarget.
            \param renderTarget dp::ui::RenderTarget to render on. If this parameter is specified
                                it overrides the dp::ui::RenderTarget set by setRenderTarget. Otherwise
                                the previously specified dp::ui::RenderTarget is used.
        **/
        DP_SG_UI_API void render( const dp::ui::SmartRenderTarget &renderTarget = dp::ui::SmartRenderTarget() );

        /** \brief Signals the renderer that a complete new image is going to be rendered. **/
        DP_SG_UI_API virtual void restartAccumulation();

      protected:
        /** \brief This function is called once per render call before the first doRender call
            \param renderTarget dp::ui::SmartRenderTarget which had been passed to the render call
        **/
        DP_SG_UI_API virtual void beginRendering( const dp::ui::SmartRenderTarget &renderTarget = dp::ui::SmartRenderTarget() );

        /** \brief This function is called once per render call after the last doRender call
            \param renderTarget dp::ui::SmartRenderTarget which had been passed to the render call
        **/
        DP_SG_UI_API virtual void endRendering( const dp::ui::SmartRenderTarget &renderTarget = dp::ui::SmartRenderTarget() );

        /** \brief Override this function to implement a rendering algorithm.
            \param renderTarget dp::ui::SmartRenderTarget determined by the render call.
        **/
        DP_SG_UI_API virtual void doRender( const dp::ui::SmartRenderTarget &renderTarget ) = 0;

      private:
        dp::ui::SmartRenderTarget m_renderTarget;
      };

    }  // namespace ui
  } // namespace sg
} // namespace dp
