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

#include <test/rix/core/framework/inc/Config.h>

#include <test/testfw/core/Backend.h>

#include <dp/util/DynamicLibrary.h>

#include <dp/rix/core/RiX.h>


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

          class RenderDataRiX : public dp::testfw::core::RenderData
          {
          public:
            DPTRIX_API RenderDataRiX();
            DPTRIX_API RenderDataRiX(const dp::rix::core::RenderGroupSharedHandle& renderGroup);
            DPTRIX_API virtual ~RenderDataRiX();

            DPTRIX_API void setRenderGroup(const dp::rix::core::RenderGroupSharedHandle& renderGroup);
            DPTRIX_API const dp::rix::core::RenderGroupSharedHandle& getRenderGroup() const;

          private:
            dp::rix::core::RenderGroupSharedHandle m_renderGroup;
          };

          SMART_TYPES( RiXBackend );

          class RiXBackend : public testfw::core::Backend
          {
          public:
            DPTRIX_API virtual ~RiXBackend();

            DPTRIX_API virtual dp::ui::SmartRenderTarget createDisplay(int width, int height, bool visible) = 0;
            DPTRIX_API virtual dp::ui::SmartRenderTarget createAuxiliaryRenderTarget(int width, int height) = 0;

            DPTRIX_API virtual void render( dp::testfw::core::RenderData* renderData, dp::ui::SmartRenderTarget renderTarget = dp::ui::SmartRenderTarget::null );
            
            DPTRIX_API dp::rix::core::Renderer* getRenderer() const;

          protected:
            DPTRIX_API RiXBackend( const char* renderer, const char* options );

          protected:
            dp::rix::core::Renderer* m_rix;
            dp::util::SmartDynamicLibrary m_rixLib;
          
          };

        } // namespace framework
      } // namespace gl
    } // namespace RiX
  } // namespace util
} // namespace dp
