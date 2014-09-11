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


#include <dp/sg/algorithm/Intersect.h>
#include <dp/sg/algorithm/RayIntersectTraverser.h>
#include <dp/sg/core/FrustumCamera.h>
#include <dp/sg/ui/ViewState.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      dp::util::SmartPtr<RayIntersectTraverser> applyPicker( dp::sg::ui::ViewStateSharedPtr const& viewStatePtr, dp::ui::SmartRenderTarget const& renderTarget, int windowX, int windowY )
      {
        DP_ASSERT( viewStatePtr );
        DP_ASSERT( renderTarget );

        dp::util::SmartPtr<RayIntersectTraverser> picker;

        unsigned int windowWidth, windowHeight;
        renderTarget->getSize( windowWidth, windowHeight );

        dp::sg::core::CameraSharedPtr pCam = viewStatePtr->getCamera();
        if ( pCam && pCam.isPtrTo<dp::sg::core::FrustumCamera>() ) // requires a frustum camera attached to the ViewState
        {
          picker = new RayIntersectTraverser;

          // calculate ray origin and direction from the input point
          dp::math::Vec3f rayOrigin;
          dp::math::Vec3f rayDir;

          pCam.staticCast<dp::sg::core::FrustumCamera>()->getPickRay( windowX, windowHeight - 1 - windowY, windowWidth, windowHeight, rayOrigin, rayDir );

          // run the intersect traverser for intersections with the given ray
          picker->setRay( rayOrigin, rayDir );
          picker->setViewportSize( windowWidth, windowHeight );
          picker->apply( viewStatePtr );
        }
        return picker;
      }

      float getIntersectionDistance( dp::sg::ui::ViewStateSharedPtr const& smartViewState, dp::ui::SmartRenderTarget const& renderTarget, int windowX, int windowY )
      {
        float result = -1.0f;
        dp::util::SmartPtr<RayIntersectTraverser> picker = applyPicker( smartViewState, renderTarget, windowX, windowY );
        if (picker && picker->getNumberOfIntersections() > 0)
        {
          result = picker->getNearest().getDist();
        }

        return result;
      }

      bool intersectObject( dp::sg::ui::ViewStateSharedPtr const& smartViewState, dp::ui::SmartRenderTarget const& renderTarget
                          , unsigned int windowX, unsigned int windowY, Intersection & result )
      {
        dp::util::SmartPtr<RayIntersectTraverser> picker = applyPicker( smartViewState, renderTarget, windowX, windowY );
        if (picker && picker->getNumberOfIntersections() > 0)
        {
          result = picker->getNearest();
          return true;
        }

        return false;
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
