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


#pragma once

#include <dp/sg/core/CoreTypes.h>
#include <dp/ui/RenderTarget.h>
#include <dp/sg/algorithm/RayIntersectTraverser.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {
      /*! \brief Get the distance to the closest object
       * \param viewState ViewState describing camera setup
       * \param renderTarget describing the viewport window
       * \param windowX x position of mouse inside the viewport window
       * \param windowY y position of mouse inside the viewport window
       * \return float distance to closest object.  -1 if nothing intersected.
       */
      float getIntersectionDistance( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget,
                                     int windowX, int windowY );

      /*! \brief intersect the scene from a window-space point
       * \param viewState ViewState describing the camera setup
       * \param renderTarget describing the viewport window
       * \param windowX x position of mouse inside the viewport window
       * \param windowY y position of mouse inside the viewport window
       * \param baseSearch the base node in the scene to search from
       * \param result the dp::sg::algorithm::Intersection result, if the method returns true
       * \return true if an intersection was found
       */
      bool intersectObject( dp::sg::ui::ViewStateSharedPtr const& viewState, dp::ui::SmartRenderTarget const& renderTarget,
                            unsigned int windowX, unsigned int windowY, Intersection & result );
    }
  }
}
