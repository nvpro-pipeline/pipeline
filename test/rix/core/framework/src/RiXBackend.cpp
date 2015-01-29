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


#include <test/rix/core/framework/RiXBackend.h>

#include <dp/util/DPAssert.h>

namespace dp
{
  namespace rix
  {
    namespace core
    {
      namespace test
      {
        namespace framework
        {

          RenderDataRiX::RenderDataRiX()
            : m_renderGroup(0)
          {
          }

          RenderDataRiX::RenderDataRiX( const dp::rix::core::RenderGroupSharedHandle& renderGroup )
          {
            m_renderGroup = renderGroup;
          }

          RenderDataRiX::~RenderDataRiX()
          {
          }

          void RenderDataRiX::setRenderGroup( const dp::rix::core::RenderGroupSharedHandle& renderGroup )
          {
            m_renderGroup = renderGroup;
          }

          const dp::rix::core::RenderGroupSharedHandle& RenderDataRiX::getRenderGroup() const
          {
            return m_renderGroup;
          }

          RiXBackend::~RiXBackend()
          {
            if(m_rix)
            {
              m_rix->deleteThis();
            }
          }

          dp::rix::core::Renderer* RiXBackend::getRenderer() const
          {
            //DP_ASSERT( util::isSmartPtrOf<RendererRiX>(m_renderer) );
            return m_rix; //util::smart_cast<RendererRiX>(m_renderer)->getRenderer();
          }

          RiXBackend::RiXBackend( const char* renderer, const char* options )
          {
            m_rixLib = util::DynamicLibrary::createFromFile(renderer);
            DP_ASSERT(m_rixLib);

            dp::rix::core::PFNCREATERENDERER createRenderer = (dp::rix::core::PFNCREATERENDERER)m_rixLib->getSymbol("createRenderer");
            DP_ASSERT(createRenderer);

            m_rix = createRenderer( options );
            DP_ASSERT(m_rix);
          }

          void RiXBackend::render( dp::testfw::core::RenderData* renderData, dp::ui::RenderTargetSharedPtr renderTarget )
          {
            DP_ASSERT(m_rix);
            DP_ASSERT(!!renderTarget);

            renderTarget->beginRendering();
            
            RenderOptions renderOptions;
            m_rix->render( static_cast<RenderDataRiX*>(renderData)->getRenderGroup(), renderOptions );

            renderTarget->endRendering();
          }

/*        //TODO: Figure out a less damaging optimization such as this one that
          //      spares needless re-initialization and reloads of libraries and 
          //      render contexts
          bool RiXBackend::resetRenderer() const
          {
            DP_ASSERT( util::isSmartPtrOf<RendererRiX>(m_renderer) );
            return util::smart_cast<RendererRiX>(m_renderer)->reset();
          }
*/

        } // namespace framework
      } // namespace gl
    } // namespace RiX
  } // namespace util
} // namespace dp
