// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/core/Camera.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/ui/manipulator/FlightCameraManipulator.h>

using namespace dp::math;
using namespace dp::sg::core;

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        static const uint32_t AREAOFPEACE = 1;

        FlightCameraManipulator::FlightCameraManipulator( const dp::math::Vec2f & sensitivity )
        : Manipulator()
        , CursorState()
        , m_sensitivity( sensitivity )
        , m_speed( 0.f )
        , m_upVector( 0.f, 1.f, 0.f ) // default to y-up
        {
        }

        FlightCameraManipulator::~FlightCameraManipulator(void)
        {
        }

        bool FlightCameraManipulator::updateFrame( float dt )
        {
          bool result = false;
          m_deltaT = dt;

          if( getViewState() && getRenderTarget() )
          {
            result = fly();
          }

          return result;
        }

        void FlightCameraManipulator::reset()
        {
          resetInput();
        }

        bool FlightCameraManipulator::fly()
        {
          // Since this is more or less a slow helicopter flight we only
          // rotate around the world axis.

          CameraSharedPtr const& camera = getViewState()->getCamera();

          if (camera && fabs( m_speed ) > 0.f )
          {
            // left / right
            float deltaX = getRenderTarget()->getWidth() * 0.5f - getLastX();
            if ( fabs(deltaX) > AREAOFPEACE )
            {
              // so we start at 0
              (deltaX < 0.f) ? deltaX += AREAOFPEACE : deltaX -= AREAOFPEACE;

              float alpha = m_deltaT * m_sensitivity[0] * deltaX;
              camera->rotate(m_upVector, degToRad(alpha), false);
            }

            // up / down
            float deltaY = getRenderTarget()->getHeight() * 0.5f - getLastY();
            if ( fabs(deltaY) > AREAOFPEACE )
            {
              // so we start at 0
              (deltaY < 0.f) ? deltaY += AREAOFPEACE : deltaY -= AREAOFPEACE;

              // determine the horizontal axis
              Vec3f side = camera->getDirection() ^ camera->getUpVector();
              side.normalize();

              float alpha = m_deltaT * m_sensitivity[1] * deltaY;
              camera->rotate(side, degToRad(alpha), false);
            }

            // move into fly direction
            camera->setPosition(camera->getPosition() + m_deltaT * m_speed * camera->getDirection());

            return true;
          }

          return false;
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
