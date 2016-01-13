// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/ui/manipulator/WalkCameraManipulator.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/algorithm/RayIntersectTraverser.h>

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

        WalkCameraManipulator::WalkCameraManipulator( const dp::math::Vec2f & sensitivity )
        : Manipulator()
        , CursorState()
        , m_cameraHeightAboveTerrain(2.f)
        , m_sensitivity( sensitivity )
        , m_mode( Mode::FREELOOK )
        , m_upVector(0.f,1.f,0.f)
        , m_saveCameraDirection(0.f,0.f,-1.f)
        {
        }

        WalkCameraManipulator::~WalkCameraManipulator(void)
        {
        }

        bool WalkCameraManipulator::updateFrame( float dt )
        {
          m_deltaT = dt;

          // walk, translate and tilt can be combined at
          // the same time in this manipulator
          bool retval = false;

          if ( getViewState() && getRenderTarget() && getRenderTarget()->getWidth()
                                                   && getRenderTarget()->getHeight() )
          {
            // first, set orientation
            // NOTE: Freelook now controls the direction of travel.  All other elements set the position only.
            if( m_mode & Mode::FREELOOK )
            {
              retval |= freeLook();
            }

            // now set position
            // findTerrainPosition will set the position to be current camera pos, or terrain pos (if found)
            findTerrainPosition();

            if( m_mode & Mode::WALK )
            {
              retval |= walk();
            }

            if( m_mode & Mode::TRANSLATE )
            {
              retval |= translate();
            }

            if( m_mode & ModeMask{Mode::STRAFE_LEFT, Mode::STRAFE_RIGHT} )
            {
              retval |= strafe( (m_mode & Mode::STRAFE_RIGHT) != 0 );
            }

            // now, set the new camera position
            getViewState()->getCamera()->setPosition( m_currentPosition );
          }

          return retval;
        }

        void WalkCameraManipulator::reset()
        {
          resetInput();
          recordPlanarCameraDirection();
        }

        bool WalkCameraManipulator::findTerrainPosition()
        {
          Vec3f camPos;
          Sphere3f bound;

          // in case we don't find any intersections
          m_currentPosition = camPos = getViewState()->getCamera()->getPosition();

          bound = getViewState()->getScene()->getRootNode()->getBoundingSphere();

          // move intersect start point high in the air
          camPos += (bound.getRadius() * m_upVector);
          Vec3f dir = -m_upVector;

          // wasteful?
          dp::sg::algorithm::RayIntersectTraverser rit;
          rit.setViewState( getViewState() );
          rit.setCamClipping( false );
          rit.setViewportSize( getRenderTarget()->getWidth(), getRenderTarget()->getHeight() );
          rit.setRay( camPos, dir );
          rit.apply( getViewState()->getScene() );

          if( rit.getNumberOfIntersections() )
          {
            const dp::sg::algorithm::Intersection & intr = rit.getNearest();

            // get "closest" point
            m_currentPosition = intr.getIsp();

            // place 2 meters above ground - should be scalable
            m_currentPosition += ( m_cameraHeightAboveTerrain * m_upVector );

            return true;
          }

          return false;
        }

        bool WalkCameraManipulator::walk()
        {
          CameraSharedPtr camera = getViewState()->getCamera();

          if (camera)
          {
            // The walk direction is not the viewing direction of the camera.
            // It is always parallel to the walking plane
            Vec3f side = camera->getDirection()^m_upVector;
            Vec3f walkdir = m_upVector^side;
            walkdir.normalize();

            m_currentPosition += (m_deltaT * m_speed * walkdir);

            // redraw required
            return true;
          }

          // no redraw required
          return false;
        }

        // note: the WalkCameraManipulatorHIDSync does not exercise this method..
        bool WalkCameraManipulator::translate()
        {
          CameraSharedPtr camera = getViewState()->getCamera();
          if (camera)
          {
            int dx = getCurrentX() - getLastX();
            int dy = getLastY() - getCurrentY();
            if (dx != 0 || dy != 0)
            {
              float stepX = m_deltaT * m_speed * getViewState()->getTargetDistance() * float(dx);
              float stepY = m_deltaT * m_speed * getViewState()->getTargetDistance() * float(dy);

              Vec3f side = camera->getDirection()^m_upVector;
              side.normalize();

              DP_ASSERT(isNormalized(m_upVector));

              m_currentPosition += (stepX * side + stepY * m_upVector);

              // redraw required
              return true;
            }
          }

          // no redraw required
          return false;
        }

        bool WalkCameraManipulator::strafe( bool right )
        {
          CameraSharedPtr camera = getViewState()->getCamera();
          if (camera)
          {
            // this gives a vector pointing "right" of camera pos
            Vec3f dir = camera->getDirection()^m_upVector;
            dir.normalize();

            dir *= m_speed * m_deltaT;

            if( !right )
            {
              dir = -dir;
            }

            m_currentPosition += dir;

            // redraw required
            return true;
          }

          // no redraw required
          return false;
        }

        bool WalkCameraManipulator::freeLook()
        {
          CameraSharedPtr camera = getViewState()->getCamera();
          if (camera)
          {
            // first, reset to default
            camera->setOrientation( m_saveCameraDirection, m_upVector );

            // now, update the position based on that default orientation, where the center of the screen will have
            // us pointing parallel to the ground
            float w2 = getRenderTarget()->getWidth()*0.5f;
            float h2 = getRenderTarget()->getHeight()*0.5f;
            float alpha = m_sensitivity[0] * -180.f * ((getCurrentX() / w2) - 1.f);
            float beta  = m_sensitivity[1] *  -90.f * ((getCurrentY() / h2) - 1.f);
            camera->rotate(m_upVector, degToRad(alpha), false);
            camera->rotateX(degToRad(beta));

            // redraw required
            return true;
          }

          // no redraw required
          return false;
        }

        void WalkCameraManipulator::recordPlanarCameraDirection()
        {
          if( getViewState() )
          {
            CameraSharedPtr camera = getViewState()->getCamera();
            if (camera)
            {
              // store the vector parallel to the walking plane
              Vec3f side = camera->getDirection()^m_upVector;
              m_saveCameraDirection = m_upVector^side;
              m_saveCameraDirection.normalize();
            }
          }
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
