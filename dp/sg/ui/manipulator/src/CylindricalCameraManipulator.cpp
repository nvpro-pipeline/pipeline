// Copyright NVIDIA Corporation 2002-2009
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


#include <dp/sg/ui/manipulator/CylindricalCameraManipulator.h>

#include <dp/sg/algorithm/RayIntersectTraverser.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/PerspectiveCamera.h>

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

        template <typename T>
        inline void CylindricalCameraManipulator::checkLockAxis(T dx, T dy)
        {
          if ( m_lockMajorAxis )
          {
            if ( !(m_lockAxis[AXIS_X] | m_lockAxis[AXIS_Y]) )
            {
              m_activeLockAxis[AXIS_X] = abs(dx)>abs(dy);
              m_activeLockAxis[AXIS_Y] = abs(dx)<abs(dy);
            }
          }
          else
          {
            m_activeLockAxis[AXIS_X] = m_lockAxis[AXIS_X];
            m_activeLockAxis[AXIS_Y] = m_lockAxis[AXIS_Y];
          }
        }


        CylindricalCameraManipulator::CylindricalCameraManipulator()
        : Manipulator()
        , CursorState()
        , m_lockMajorAxis( false )
        , m_speed(0.001f)
        {
          m_lockAxis[0] = m_lockAxis[1] = m_lockAxis[2] = false;
          m_activeLockAxis[0] = m_activeLockAxis[1] = m_activeLockAxis[2] = false;
        }

        CylindricalCameraManipulator::~CylindricalCameraManipulator()
        {
        }

        void CylindricalCameraManipulator::reset()
        {
          resetInput();
        }

        void CylindricalCameraManipulator::setMode( Mode mode )
        {
          if ( getMode() != mode && getViewState() )
          {
            // Store cursor position and camera position on mode change. These values are being used for non-incremental updates.
        #if 0
            m_orbitCursor = getCursorPosition();
            CameraLock camera(viewState->getCamera());
            m_orbitCameraPosition = camera->getPosition();
            m_orbitCameraOrientation = camera->getOrientation();
        #endif

            m_mode = mode;
          }
        }

        bool CylindricalCameraManipulator::updateFrame( float dt )
        {
          bool result = false;

          if ( getViewState() && getRenderTarget() )
          {
            switch ( getMode() )
            {
              case MODE_ORBIT:
                result = orbit();
                break;

              case MODE_PAN:
                result = pan();
                break;

              case MODE_ROTATE_XY:
                result = rotate();
                break;

              case MODE_DOLLY:
                result = dolly();
                break;

              case MODE_ROLL_Z:
                result = roll();
                break;

              case MODE_ZOOM_FOV:
                result = zoom();
                break;

              case MODE_ZOOM_DOLLY:
                result = dollyZoom();
                break;

              case MODE_LOOKAT:
                result = lookAt();
                break;

              default:
                break;
            }

            if ( getWheelTicksDelta() && (m_mode != MODE_DOLLY) )
            {
              result = dolly();
            }
          }

          // NOTE: we are missing:
         // New custom operations!
         //else if (isSetPivot())
         // {
         //   result = setPivot();
         // }
         // else if (isSetFocus())
         // {
         //   result = setFocus();
         // }

          return result;
        }

        bool CylindricalCameraManipulator::pan()
        {
          dp::sg::ui::ViewStateSharedPtr viewState = getViewState();
          if ( viewState )
          {
            CameraSharedPtr camera = viewState->getCamera();
            if (camera)
            {
              int dx = getLastX() - getCurrentX();
              int dy = getCurrentY() - getLastY();

              if (dx != 0 || dy != 0)
              {
                checkLockAxis(dx, dy);

                float stepX = m_speed * viewState->getTargetDistance() * float(dx);
                float stepY = m_speed * viewState->getTargetDistance() * float(dy);

                if( m_activeLockAxis[ AXIS_X ] )
                {
                  if(dx != 0)
                  {
                    stepY = 0;
                  }
                  else
                  {
                    return false;
                  }
                }
                else if( m_activeLockAxis[ AXIS_Y ] )
                {
                  if(dy != 0)
                  {
                    stepX = 0;
                  }
                  else
                  {
                    return false;
                  }
                }

                // construct the camera movement plane
                Vec3f side = camera->getDirection()^camera->getUpVector();
                side.normalize();

                DP_ASSERT(isNormalized(camera->getUpVector()));
                camera->setPosition(camera->getPosition() + stepX * side + stepY * camera->getUpVector());

                // redraw required
                return true;
              }
            }
          }

          // no redraw required
          return false;
        }

        // Dolly without moving the focus point for easier use with depth of field.
        // The minimum focus distance is clamped to DOFCamera::m_focalLength * NEAREST_FOCUS_SCALE.
        // Use RMB and Ctrl+RMB to change the focus distance.
        bool CylindricalCameraManipulator::dolly()
        {
          dp::sg::ui::ViewStateSharedPtr const& viewState = getViewState();
          if ( viewState )
          {
            if ( viewState->getCamera().isPtrTo<PerspectiveCamera>() )
            {
              PerspectiveCameraSharedPtr const& camera = viewState->getCamera().staticCast<PerspectiveCamera>();

              int dy = getCurrentY() - getLastY();
              float multiplier = 1.0f;

        #if 0
              if ( m_flags & NVSG_WHEELING )
              {
                // take accumulated deltas if user is wheeling
                dy = getWheelTicks();
         
                // Temporary speedup/slowdown of mouse wheel dolly operation.
                if (getKeyState() & NVSG_SHIFT) // Speedup has precedence.
                {
                  multiplier = 4.0f;
                }
                else if (getKeyState() & NVSG_CONTROL)
                {
                  multiplier = 0.25f;
                }
              }
        #else
              // see if we are wheeling
              if ( getWheelTicksDelta() )
              {
                // take accumulated deltas if user is wheeling
                dy = getWheelTicksDelta();
         
                // MMM - implement speed in sync
        #if 0
                // Temporary speedup/slowdown of mouse wheel dolly operation.
                if (getKeyState() & NVSG_SHIFT) // Speedup has precedence.
                {
                  multiplier = 4.0f;
                }
                else if (getKeyState() & NVSG_CONTROL)
                {
                  multiplier = 0.25f;
                }
        #endif
              }
        #endif

              if (dy != 0)
              {
                float targetDistance = viewState->getTargetDistance();
                float step = multiplier * m_speed * targetDistance * float(dy);

                DP_ASSERT( isNormalized(camera->getDirection()) );
                camera->setPosition(camera->getPosition() + step * camera->getDirection());

                // This is an arbitrarily chosen minimum distance, assumed to be 1 unit == 1 cm.  This
                // may not work in very tiny environments.
                camera->setFocusDistance(std::max(camera->getFocusDistance() - step, 1.0f));

                viewState->setTargetDistance(targetDistance - step);

                // redraw required
                return true;
              }
            }
          }

          // no redraw required
          return false;
        }


        bool CylindricalCameraManipulator::zoom()
        {
          dp::sg::ui::ViewStateSharedPtr const& viewState = getViewState();
          if ( viewState )
          {
            CameraSharedPtr camera = viewState->getCamera();
            if (camera)
            {
              int dy = getCurrentY()-getLastY();
              if (dy != 0)
              {
                float targetDistance = viewState->getTargetDistance();
                float step =  m_speed * targetDistance * float(dy);

                DP_ASSERT( isNormalized(camera->getDirection()));
                camera->zoom(targetDistance / (targetDistance-step));
                // redraw required
                return true;
              }
            }
          }
          // no redraw required
          return false;
        }

        bool CylindricalCameraManipulator::dollyZoom()
        {
          dp::sg::ui::ViewStateSharedPtr const& viewState = getViewState();
          if ( viewState )
          {
            CameraSharedPtr camera = viewState->getCamera();
            if (camera)
            {
              int dy = getCurrentY()-getLastY();
              if (dy != 0)
              {
                float targetDistance = viewState->getTargetDistance();
                float step =  m_speed * targetDistance * float(dy);

                DP_ASSERT( isNormalized(camera->getDirection()));
                camera->setPosition(camera->getPosition()  + step * camera->getDirection());
                camera->zoom(targetDistance / (targetDistance-step));

                viewState->setTargetDistance(targetDistance - step);
                // redraw required
                return true;
              }
            }
          }

          // no redraw required
          return false;
        }

        // Roll free rotate (Ctrl+LMB drag)
        bool CylindricalCameraManipulator::rotate()
        {
          dp::sg::ui::ViewStateSharedPtr const& viewState = getViewState();
          if ( viewState )
          {
            CameraSharedPtr camera = viewState->getCamera();
            if (camera)
            {
              float halfWndX = float(getRenderTarget()->getWidth())  * 0.5f;
              float halfWndY = float(getRenderTarget()->getHeight()) * 0.5f;

              Vec2f p0( (float(getLastX()) - halfWndX)  / halfWndX
                      , (float(halfWndY) - getLastY())  / halfWndY);

              Vec2f p1( (float(getCurrentX()) - halfWndX)  / halfWndX
                      , (float(halfWndY) - getCurrentY())  / halfWndY);

              if (p0 != p1)
              {
                float dx = p1[0] - p0[0];
                float dy = p1[1] - p0[1];

                camera->rotate( Vec3f(0.0f, 1.0f, 0.0f), -dx, false); // world relative
                camera->rotate( Vec3f(1.0f, 0.0f, 0.0f),  dy, true ); // camera relative

                // redraw required
                return true;
              }
            }
          }
          // no redraw required
          return false;
        }


        // Roll camera around z axis (Ctrl+MMB drag)
        bool CylindricalCameraManipulator::roll()
        {
          dp::sg::ui::ViewStateSharedPtr const& viewState = getViewState();
          if ( viewState )
          {
            CameraSharedPtr camera = viewState->getCamera();
            if (camera)
            {
              float halfWndX = float(getRenderTarget()->getWidth())  * 0.5f;
              float halfWndY = float(getRenderTarget()->getHeight()) * 0.5f;

              Vec2f p0( (float(getLastX()) - halfWndX) / halfWndX
                      , (float(halfWndY) - getLastY()) / halfWndY);

              Vec2f p1( (float(getCurrentX()) - halfWndX) / halfWndX
                      , (float(halfWndY) - getCurrentY()) / halfWndY);

              if (p0 != p1)
              {
                // The z-coordinate sign of the p1 x p0 cross product controls the direction.
                float r = distance(p0, p1) * sign(p1[0] * p0[1] - p0[0] * p1[1]);

                camera->rotate( Vec3f(0.0f, 0.0f, 1.0f), r ); // camera relative

                // redraw required
                return true;
              }
            }
          }
          // no redraw required
          return false;
        }

        // Roll free orbit (LMB drag)
        bool CylindricalCameraManipulator::orbit()
        {
          DP_ASSERT(getRenderTarget()->getWidth());
          DP_ASSERT(getRenderTarget()->getHeight());

          dp::sg::ui::ViewStateSharedPtr const& viewState = getViewState();
          if ( viewState )
          {
            CameraSharedPtr camera = viewState->getCamera();
            if (camera)
            {
              float halfWndX = float(getRenderTarget()->getWidth())  * 0.5f;
              float halfWndY = float(getRenderTarget()->getHeight()) * 0.5f;

              int lastX = getLastX();
              int lastY = getLastY();
              int currentX = getCurrentX();
              int currentY = getCurrentY();
      
        #if 0
              if ( m_flags & NVSG_SPIN )
              {
                lastX = m_startSpinX;
                lastY = m_startSpinY;
                currentX = m_currentSpinX;
                currentY = m_currentSpinY;
              }
        #endif

              Vec2f p0( (float(lastX) - halfWndX)  / halfWndX
                      , (float(halfWndY) - lastY)  / halfWndY);

              Vec2f p1( (float(currentX) - halfWndX)  / halfWndX
                      , (float(halfWndY) - currentY)  / halfWndY);

              if (p0 != p1)
              {
                float dx = p1[0] - p0[0];
                float dy = p1[1] - p0[1];

                camera->orbit( Vec3f(0.0f, 1.0f, 0.0f), viewState->getTargetDistance(),  dx * PI_HALF, false ); // world relative
                camera->orbit( Vec3f(1.0f, 0.0f, 0.0f), viewState->getTargetDistance(), -dy * PI_HALF, true  ); // camera relative

                // redraw required
                return true;
              }
            }
          }

          // no redraw required
          return false;
        }

        bool CylindricalCameraManipulator::lookAt()
        {
          DP_ASSERT(getRenderTarget()->getWidth());
          DP_ASSERT(getRenderTarget()->getHeight());

          bool needsRedraw = false;
          float hitDistance = 0.0f;
          Vec3f rayOrigin;
          Vec3f rayDir;

          dp::sg::ui::ViewStateSharedPtr const& viewStateHdl = getViewState();
          if ( viewStateHdl )
          {
            if ( viewStateHdl->getCamera().isPtrTo<FrustumCamera>() )
            {
              FrustumCameraSharedPtr const& cameraHdl = viewStateHdl->getCamera().staticCast<FrustumCamera>();

              // calculate ray origin and direction from the input point
              int vpW = getRenderTarget()->getWidth();
              int vpH = getRenderTarget()->getHeight();
              int pkX = getCurrentX();           // at mouse-up, not mouse-down
              int pkY = vpH - 1 - getCurrentY(); // pick point is lower-left-relative

              cameraHdl->getPickRay(pkX, pkY, vpW, vpH, rayOrigin, rayDir);

              // run the intersect traverser for intersections with the given ray
              dp::sg::algorithm::RayIntersectTraverser picker;
              picker.setRay(rayOrigin, rayDir);
              picker.setViewState(viewStateHdl);
              picker.setViewportSize(vpW, vpH);
              picker.apply(viewStateHdl->getScene());

              if (picker.getNumberOfIntersections() > 0)
              {
                needsRedraw = true;
                hitDistance = picker.getNearest().getDist();
              }
            }
          }

          if(needsRedraw)
          {
            viewStateHdl->setTargetDistance(hitDistance);

            CameraSharedPtr const& camera = viewStateHdl->getCamera();
            camera->setPosition(rayOrigin);
            camera->setDirection(rayDir);
          }

          return needsRedraw;
        }

        bool CylindricalCameraManipulator::setPivot()
        {
          dp::sg::ui::ViewStateSharedPtr const& viewState = getViewState();
          if ( viewState )
          {
            if ( viewState->getCamera().isPtrTo<PerspectiveCamera>() )
            {
              PerspectiveCameraSharedPtr const& camera = viewState->getCamera().staticCast<PerspectiveCamera>();

              int vpW = getRenderTarget()->getWidth();
              int vpH = getRenderTarget()->getHeight();

              if (vpW && vpH)
              {
                int pkX = getCurrentX();           // at mouse-up, not mouse-down
                int pkY = vpH - 1 - getCurrentY(); // pick point is lower-left-relative
                Vec3f rayOrigin;
                Vec3f rayDir;

                camera->getPickRay(pkX, pkY, vpW, vpH, rayOrigin, rayDir);

                // run the intersect traverser for intersections with the given ray
                dp::sg::algorithm::RayIntersectTraverser picker;
                picker.setRay(rayOrigin, rayDir);
                picker.setViewState(viewState);
                picker.setViewportSize(vpW, vpH);
                picker.apply(viewState->getScene());

                if (picker.getNumberOfIntersections() > 0)
                {
                  float dist = picker.getNearest().getDist();

                  if (0.0f < dist)
                  {
                    Vec3f pivotOld(camera->getPosition() + camera->getDirection() * viewState->getTargetDistance());
                    Vec3f pivotNew(rayOrigin + rayDir * dist);

                    // Change the camera position instead of direction to make this pivot point setting roll free.
                    camera->move(pivotNew - pivotOld, false); // world space
            
                    // Also set the focus distance!
                    // Do not focus nearer than lens' lower focus limit.
                    dist = distance( camera->getPosition(), pivotNew);
                    camera->setFocusDistance(std::max(0.1f, dist));

                    // redraw required
                    return true;
                  }
                }
              }
            }
          }

          // no redraw required
          return false;
        }

        bool CylindricalCameraManipulator::setFocus()
        {
          dp::sg::ui::ViewStateSharedPtr const& viewState = getViewState();
          if ( viewState )
          {
            if ( viewState->getCamera().isPtrTo<PerspectiveCamera>() )
            {
              PerspectiveCameraSharedPtr const& camera = viewState->getCamera().staticCast<PerspectiveCamera>();

              int vpW = getRenderTarget()->getWidth();
              int vpH = getRenderTarget()->getHeight();
      
              if (vpW && vpH)
              {
                int pkX = getCurrentX();           // at mouse-up, not mouse-down
                int pkY = vpH - 1 - getCurrentY(); // pick point is lower-left-relative
                Vec3f rayOrigin;
                Vec3f rayDir;

                camera->getPickRay(pkX, pkY, vpW, vpH, rayOrigin, rayDir);

                // run the intersect traverser for intersections with the given ray
                dp::sg::algorithm::RayIntersectTraverser picker;
                picker.setRay(rayOrigin, rayDir);
                picker.setViewState(viewState);
                picker.setViewportSize(vpW, vpH);
                picker.apply(viewState->getScene());

                if (picker.getNumberOfIntersections() > 0)
                {
                  float dist = picker.getNearest().getDist();
                  if (0.0f < dist)
                  {
                    // Only set the focus distance, pivot point remains the same.
                    // Do not focus nearer than lens' lower focus limit.
                    camera->setFocusDistance(std::max(0.1f, dist));
                  }
                  // redraw required
                  return true;
                }
              }
            }
          }
          // no redraw required
          return false;
        }

        void CylindricalCameraManipulator::lockAxis( Axis axis )
        {
          m_lockAxis[axis] = true;
        }

        void CylindricalCameraManipulator::unlockAxis( Axis axis )
        {
          m_lockAxis[axis] = false;
        }

        void CylindricalCameraManipulator::lockMajorAxis( )
        {
          m_lockMajorAxis = true;
        }

        void CylindricalCameraManipulator::unlockMajorAxis( )
        {
          m_lockMajorAxis = false;
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
