// Copyright NVIDIA Corporation 2002-2013
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


#include <dp/gl/RenderTarget.h>
#include <dp/sg/algorithm/RayIntersectTraverser.h>
#include <dp/sg/core/FrustumCamera.h>
#include <dp/sg/ui/manipulator/TrackballCameraManipulator.h>
#include <dp/util/SharedPtr.h>
#include <GL/glew.h>

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
        inline void TrackballCameraManipulator::checkLockAxis(T dx, T dy)
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

        TrackballCameraManipulator::TrackballCameraManipulator()
        : Manipulator()
        , CursorState()
        , m_mode(MODE_NONE)
        , m_speed( 0.001f )
        , m_lockMajorAxis( false )
        {
          m_lockAxis[AXIS_X]       = m_lockAxis[AXIS_Y]       = m_lockAxis[AXIS_Z]       = false;
          m_activeLockAxis[AXIS_X] = m_activeLockAxis[AXIS_Y] = m_activeLockAxis[AXIS_Z] = false;
        }

        TrackballCameraManipulator::~TrackballCameraManipulator()
        {
        }

        void TrackballCameraManipulator::reset()
        {
          resetInput();
        }

        bool TrackballCameraManipulator::updateFrame( float dt )
        {
          bool result = false;

          if ( getViewState() && getRenderTarget() )
          {
            switch ( m_mode )
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

            case MODE_LOOKAT_DEPTH:
              result = lookAtDepthGL();
              break;

            default:
              break;
            }

            if ( getWheelTicksDelta() && (m_mode != MODE_DOLLY) )
            {
              result = dolly();
            }
          }

          return result;
        }

        bool TrackballCameraManipulator::pan()
        {
          DP_ASSERT(m_viewState);
          if ( m_viewState )
          {
            CameraSharedPtr camera = m_viewState->getCamera();
            if (camera)
            {
              int dx = getLastX() - getCurrentX();
              int dy = getCurrentY() - getLastY();

              if (dx != 0 || dy != 0)
              {

                checkLockAxis(dx, dy);

                float stepX = m_speed * m_viewState->getTargetDistance() * float(dx);
                float stepY = m_speed * m_viewState->getTargetDistance() * float(dy);

                if( m_activeLockAxis[AXIS_X] )
                {
                  if(dx!=0)
                    stepY =0;
                  else
                    return false;
                }
                else if( m_activeLockAxis[AXIS_Y] )
                {
                  if(dy!=0)
                    stepX =0;
                  else
                    return false;
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

        bool TrackballCameraManipulator::dolly()
        {
          DP_ASSERT(m_viewState);
          if ( m_viewState )
          {
            CameraSharedPtr camera = m_viewState->getCamera();
            if (camera)
            {
              int dy = getWheelTicksDelta();
              if (!dy)
              {
                dy = getCurrentY()-getLastY();
              }

              if (dy != 0)
              {
                float targetDistance = m_viewState->getTargetDistance();
                float step =  m_speed * targetDistance * float(dy);

                DP_ASSERT( isNormalized(camera->getDirection()));
                camera->setPosition(camera->getPosition()  + step * camera->getDirection());

                m_viewState->setTargetDistance(targetDistance - step);
                // redraw required
                return true;
              }
            }
          }

          // no redraw required
          return false;
        }

        bool TrackballCameraManipulator::zoom()
        {
          DP_ASSERT(m_viewState);
          if ( m_viewState )
          {
            CameraSharedPtr camera = m_viewState->getCamera();
            if (camera)
            {
              int dy = getCurrentY()-getLastY();
              if (dy != 0)
              {
                float targetDistance = m_viewState->getTargetDistance();
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

        bool TrackballCameraManipulator::dollyZoom()
        {
          DP_ASSERT(m_viewState);
          if ( m_viewState )
          {
            CameraSharedPtr camera = m_viewState->getCamera();
            if (camera)
            {
              int dy = getCurrentY()-getLastY();
              if (dy != 0)
              {
                float targetDistance = m_viewState->getTargetDistance();
                float step =  m_speed * targetDistance * float(dy);

                DP_ASSERT( isNormalized(camera->getDirection()));
                camera->setPosition(camera->getPosition()  + step * camera->getDirection());
                camera->zoom(targetDistance / (targetDistance-step));

                m_viewState->setTargetDistance(targetDistance - step);
                // redraw required
                return true;
              }
            }
          }

          // no redraw required
          return false;
        }

        bool TrackballCameraManipulator::rotate()
        {
          DP_ASSERT(m_viewState);
          if ( m_viewState )
          {
            CameraSharedPtr camera = m_viewState->getCamera();
            if (camera)
            {
              float halfWndX = float(getRenderTarget()->getWidth())  * 0.5f;
              float halfWndY = float(getRenderTarget()->getHeight()) * 0.5f;

              int lastX = m_orbitCursor[0];
              int lastY = m_orbitCursor[1];
              {
                camera->setPosition( m_orbitCameraPosition );
                camera->setOrientation( m_orbitCameraOrientation );
              }

              Vec2f p0( (float(lastX) - halfWndX)  / halfWndX
                      , (float(halfWndY) - lastY)  / halfWndY);

              Vec2f p1( (float(getCurrentX()) - halfWndX)  / halfWndX
                      , (float(halfWndY) - getCurrentY())  / halfWndY);

              if (p0 != p1)
              {
                float dx = p1[0] - p0[0];
                float dy = p1[1] - p0[1];

                checkLockAxis(dx, dy);

                Vec3f axis;
                Vec2f m = p1-p0;
                axis = Vec3f(m[1], -m[0],0.0f);
                axis.normalize();

                if( m_activeLockAxis[AXIS_X] )
                {
                  if(dx>0)
                    axis = Vec3f(0.0f,-1.0f,0.0f);
                  else if(dx<0)
                    axis = Vec3f(0.0f,1.0f,0.0f);
                  else
                    return false;
                }
                else if( m_activeLockAxis[AXIS_Y] )
                {
                  if(dy>0)
                    axis = Vec3f(1.0f,0.0f,0.0f);
                  else if(dy<0)
                    axis = Vec3f(-1.0f,0.0f,0.0f);
                  else
                    return false;
                }
                camera->rotate( axis, distance(p0,p1) );
              }
              // redraw required
              return true;
            }
          }
          // no redraw required
          return false;
        }

        bool TrackballCameraManipulator::roll()
        {
          DP_ASSERT(m_viewState);
          if ( m_viewState )
          {
            CameraSharedPtr camera = m_viewState->getCamera();
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
              }
              // redraw required
              return true;
            }
          }
          // no redraw required
          return false;
        }

        bool TrackballCameraManipulator::orbit()
        {
          if ( m_viewState )
          {
            DP_ASSERT(getRenderTarget());
            DP_ASSERT(getRenderTarget()->getWidth());
            DP_ASSERT(getRenderTarget()->getHeight());

            CameraSharedPtr camera = m_viewState->getCamera();
            if (camera)
            {
              float halfWndX = float(getRenderTarget()->getWidth())  * 0.5f;
              float halfWndY = float(getRenderTarget()->getHeight()) * 0.5f;

              int lastX = m_orbitCursor[0];
              int lastY = m_orbitCursor[1];
              camera->setPosition( m_orbitCameraPosition );
              camera->setOrientation( m_orbitCameraOrientation );

              int currentX = getCurrentX();
              int currentY = getCurrentY();
      
              Vec2f p0( (float(lastX) - halfWndX)  / halfWndX
                      , (float(halfWndY) - lastY)  / halfWndY);

              Vec2f p1( (float(currentX) - halfWndX)  / halfWndX
                      , (float(halfWndY) - currentY)  / halfWndY);

              if (p0 != p1)
              {

                float dx = p1[0] - p0[0];
                float dy = p1[1] - p0[1];

                checkLockAxis(dx, dy);

                Vec3f axis;
                float angle;
                m_trackball.apply(p0, p1, axis, angle);

                if ( m_activeLockAxis[AXIS_X] )
                {
                  if(dx>0)
                    axis = Vec3f(0.0f,1.0f,0.0f);
                  else if(dx<0)
                    axis = Vec3f(0.0f,-1.0f,0.0f);
                  else
                    return false;
                }
                else if( m_activeLockAxis[AXIS_Y] )
                {
                  if(dy>0)
                    axis = Vec3f(-1.0f,0.0f,0.0f);
                  else if(dy<0)
                    axis = Vec3f(1.0f,0.0f,0.0f);
                  else
                    return false;
                }

                camera->orbit( axis, m_viewState->getTargetDistance(), angle );

                // redraw required
                return true;
              }
            }
          }

          // no redraw required
          return false;
        }

        bool TrackballCameraManipulator::lookAt()
        {
          DP_ASSERT(m_viewState);

          bool needsRedraw = false;
          float hitDistance = 0.0f;
          Vec3f rayOrigin;
          Vec3f rayDir;

          if ( m_viewState )
          {
            DP_ASSERT(getRenderTarget());
            DP_ASSERT(getRenderTarget()->getWidth());
            DP_ASSERT(getRenderTarget()->getHeight());

            CameraSharedPtr cameraHdl = m_viewState->getCamera();
            if (cameraHdl && cameraHdl.isPtrTo<FrustumCamera>() )
            {
              // calculate ray origin and direction from the input point
              int vpW = getRenderTarget()->getWidth();
              int vpH = getRenderTarget()->getHeight();
              int pkX = getCurrentX();       // at mouse-up, not mouse-down
              int pkY = vpH - getCurrentY(); // pick point is lower-left-relative

              cameraHdl.staticCast<FrustumCamera>()->getPickRay(pkX, pkY, vpW, vpH, rayOrigin, rayDir);

              // run the intersect traverser for intersections with the given ray
              dp::sg::algorithm::RayIntersectTraverser picker;
              picker.setRay(rayOrigin, rayDir);
              picker.setViewState(m_viewState);
              picker.setViewportSize(vpW, vpH);
              picker.apply(m_viewState->getScene());

              if (picker.getNumberOfIntersections() > 0)
              {
                needsRedraw = true;
                hitDistance = picker.getNearest().getDist();
              }
            }
          }

          if(needsRedraw)
          {
            m_viewState->setTargetDistance(hitDistance);

            CameraSharedPtr const& camera = m_viewState->getCamera();
            camera->setPosition(rayOrigin);
            camera->setDirection(rayDir);
          }

          return needsRedraw;
        }

        bool TrackballCameraManipulator::lookAtDepthGL()
        {
          DP_ASSERT(m_viewState);

          bool needsRedraw = false;

          if ( m_viewState )
          {
            DP_ASSERT(getRenderTarget());
            DP_ASSERT(getRenderTarget()->getWidth());
            DP_ASSERT(getRenderTarget()->getHeight());

            CameraSharedPtr cameraHdl = m_viewState->getCamera();
            if (cameraHdl && cameraHdl.isPtrTo<FrustumCamera>() )
            {
              Vec3f rayOrigin;
              Vec3f rayDir;

              // calculate ray origin and direction from the input point
              int vpW = getRenderTarget()->getWidth();
              int vpH = getRenderTarget()->getHeight();
              int pkX = getCurrentX();       // at mouse-up, not mouse-down
              int pkY = vpH - getCurrentY(); // pick point is lower-left-relative

              float depth;
              glReadPixels( pkX, pkY,  1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth );

              // something has been hit
              if ( depth < 1.0f )
              {
                dp::sg::core::CameraSharedPtr camera = getViewState()->getCamera();

                dp::math::Mat44d worldToView(camera->getWorldToViewMatrix());
                dp::math::Mat44d projection(camera->getProjection());

                GLint viewport[4];
                unsigned int width, height;
                dp::util::shared_cast<dp::gl::RenderTarget>(getRenderTarget())->getPosition( viewport[0], viewport[1] );
                dp::util::shared_cast<dp::gl::RenderTarget>(getRenderTarget())->getSize( width, height );
                width = GLint(width);
                height = GLint(height);

                dp::math::Vec3d pickPoint;
                gluUnProject( pkX,  pkY, depth, worldToView.getPtr(), projection.getPtr(), viewport, &pickPoint[0], &pickPoint[1], &pickPoint[2] );

                dp::math::Vec3f pickPointFloat(pickPoint);
                getViewState()->setTargetDistance( dp::math::length( camera->getPosition() - pickPointFloat ) );

                cameraHdl.staticCast<FrustumCamera>()->getPickRay(pkX, pkY, vpW, vpH, rayOrigin, rayDir);
                camera->setPosition( rayOrigin );
                camera->setDirection( rayDir );

                needsRedraw = true;
              }
            }
          }

          return needsRedraw;
        }

        void TrackballCameraManipulator::setMode( Mode mode )
        {
          if ( m_mode != mode && m_viewState )
          {
            // Store cursor position and camera position on mode change. These values are being used for non-incremental updates.
            m_orbitCursor = getCursorPosition();
            CameraSharedPtr const& camera = m_viewState->getCamera();
            m_orbitCameraPosition = camera->getPosition();
            m_orbitCameraOrientation = camera->getOrientation();

            m_mode = mode;
          }
        }

        void TrackballCameraManipulator::lockAxis( Axis axis )
        {
          m_lockAxis[axis] = true;
        }

        void TrackballCameraManipulator::unlockAxis( Axis axis )
        {
          m_lockAxis[axis] = false;
        }

        void TrackballCameraManipulator::lockMajorAxis( )
        {
          m_lockMajorAxis = true;
        }

        void TrackballCameraManipulator::unlockMajorAxis( )
        {
          m_lockMajorAxis = false;
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
