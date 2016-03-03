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


#include <dp/sg/ui/manipulator/TrackballTransformManipulator.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/FrustumCamera.h>
#include <dp/sg/core/Path.h>
#include <dp/sg/ui/ViewState.h>

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
        inline void TrackballTransformManipulator::checkLockAxis(T dx, T dy)
        {
          if ( m_lockMajorAxis )
          {
            if ( !(m_lockAxis[static_cast<size_t>(Axis::X)] | m_lockAxis[static_cast<size_t>(Axis::Y)]) )
            {
               m_activeLockAxis[static_cast<size_t>(Axis::X)] = abs(dx)>abs(dy);
               m_activeLockAxis[static_cast<size_t>(Axis::Y)] = abs(dx)<abs(dy);
            }
          }
          else
          {
            m_activeLockAxis[static_cast<size_t>(Axis::X)] = m_lockAxis[static_cast<size_t>(Axis::X)];
            m_activeLockAxis[static_cast<size_t>(Axis::Y)] = m_lockAxis[static_cast<size_t>(Axis::Y)];
          }
        }

        TrackballTransformManipulator::TrackballTransformManipulator()
          : Manipulator()
          , CursorState()
          , m_mode( Mode::NONE )
          , m_lockMajorAxis( false )
        {
          m_lockAxis[static_cast<size_t>(Axis::X)]       = m_lockAxis[static_cast<size_t>(Axis::Y)]       = m_lockAxis[static_cast<size_t>(Axis::Z)] = false;
          m_activeLockAxis[static_cast<size_t>(Axis::X)] = m_activeLockAxis[static_cast<size_t>(Axis::Y)] = m_activeLockAxis[static_cast<size_t>(Axis::Z)] = false;
        }

        TrackballTransformManipulator::~TrackballTransformManipulator()
        {
        }

        void TrackballTransformManipulator::reset()
        {
          resetInput();
        }

        bool TrackballTransformManipulator::updateFrame( float dt )
        {
          bool result = false;

          if( m_transformPath && getViewState() && getRenderTarget() )
          {
            switch( m_mode )
            {
              case Mode::ROTATE:
                result = rotate();
                break;

              case Mode::PAN:
                result = pan();
                break;

              case Mode::DOLLY:
                result = dolly();
                break;

              default:
                break;
            }
          }

          return result;
        }

        bool  TrackballTransformManipulator::pan()
        {
          int dxScreen = getCurrentX() - getLastX();
          int dyScreen = getLastY() - getCurrentY();
          if ( dxScreen || dyScreen )
          {
            DP_ASSERT( std::dynamic_pointer_cast<FrustumCamera>(getViewState()->getCamera()) );
            TransformSharedPtr transform = std::static_pointer_cast<Transform>(m_transformPath->getTail());
            FrustumCameraSharedPtr const& camera = std::static_pointer_cast<FrustumCamera>(getViewState()->getCamera());
            if ( camera && transform )
            {
              unsigned int rtWidth = getRenderTarget()->getWidth();
              unsigned int rtHeight = getRenderTarget()->getHeight();
              Vec2f  camWinSize = camera->getWindowSize();
              if (    ( 0 < rtHeight ) && ( 0 < rtWidth )
                  &&  ( FLT_EPSILON < fabs( camWinSize[0] ) )
                  &&  ( FLT_EPSILON < fabs( camWinSize[1] ) ) )
              {
                //  get all the matrices needed here
                Mat44f m2w, w2m;
                m_transformPath->getModelToWorldMatrix(m2w, w2m); // model->world and world->model
                Mat44f w2v = camera->getWorldToViewMatrix();   // world->view
                Mat44f v2w = camera->getViewToWorldMatrix();   // view->world

                //  center of the object in view coordinates
                Vec4f center = Vec4f( transform->getBoundingSphere().getCenter(), 1.0f ) * m2w * w2v;

                //  window size at distance of the center of the object
                Vec2f centerWindowSize = - center[2] / getViewState()->getTargetDistance() * camWinSize;

                checkLockAxis(dxScreen, dyScreen);
                if ( m_activeLockAxis[static_cast<size_t>(Axis::X)] )
                {
                  if ( dxScreen != 0 )
                  {
                    dyScreen = 0;
                  }
                  else
                  {
                    return false;
                  }
                }
                else if ( m_activeLockAxis[static_cast<size_t>(Axis::Y)] )
                {
                  if ( dyScreen != 0)
                  {
                    dxScreen = 0;
                  }
                  else
                  {
                    return false;
                  }
                }

                //  delta in model coordinates
                Vec4f viewCenter( centerWindowSize[0] * dxScreen / rtWidth
                                , centerWindowSize[1] * dyScreen / rtHeight, 0.f, 0.f );
                Vec4f modelDelta = viewCenter * v2w * w2m;

                // add the delta to the translation of the transform
                Trafo trafo = transform->getTrafo();
                trafo.setTranslation( trafo.getTranslation() + Vec3f( modelDelta ) );
                transform->setTrafo( trafo );

                return true;
              }
            }
          }
          return false;
        }

        bool TrackballTransformManipulator::rotate()
        {
          if ( ( getCurrentX() != getLastX() ) || ( getCurrentY() != getLastY() ) )
          {
            DP_ASSERT( std::dynamic_pointer_cast<FrustumCamera>(getViewState()->getCamera()) );
            TransformSharedPtr transform = std::static_pointer_cast<Transform>(m_transformPath->getTail());
            FrustumCameraSharedPtr const& camera = std::static_pointer_cast<FrustumCamera>(getViewState()->getCamera());
            if ( camera && transform )
            {
              unsigned int rtWidth    = getRenderTarget()->getWidth();
              unsigned int rtHeight   = getRenderTarget()->getHeight();
              Vec2f  camWinSize = camera->getWindowSize();
              if (    ( 0 < rtHeight ) && ( 0 < rtWidth )
                  &&  ( FLT_EPSILON < fabs( camWinSize[0] ) )
                  &&  ( FLT_EPSILON < fabs( camWinSize[1] ) ) )
              {
                //  get all the matrices needed here
                Mat44f m2w, w2m, w2v, v2w, v2s, m2v;
                m_transformPath->getModelToWorldMatrix( m2w, w2m ); // model->world and world->model
                w2v = camera->getWorldToViewMatrix();            // world->view
                v2w = camera->getViewToWorldMatrix();            // view->world
                v2s = camera->getProjection();                   // view->screen (normalized)
                m2v = m2w * w2v;

                const Sphere3f& bs = transform->getBoundingSphere();

                //  center of the object in view coordinates
                Vec4f centerV = Vec4f( bs.getCenter(), 1.0f ) * m2v;
                DP_ASSERT( fabs( centerV[3] - 1.0f ) < FLT_EPSILON );

                //  center of the object in normalized screen coordinates
                Vec4f centerNS = centerV * v2s;
                DP_ASSERT( centerNS[3] != 0.0f );
                centerNS /= centerNS[3];

                //  center of the object in screen space
                Vec2f centerS( rtWidth * ( 1 + centerNS[0] ) / 2, rtHeight * ( 1 - centerNS[1] ) / 2 );

                //  move the input points relative to the center
                //  move the input points absolutely
                //Vec2f last( m_orbitCursor );
                Vec2f last( getLastCursorPosition() );
                Vec2f p0( last[0]    - centerS[0], centerS[1] - last[1] );
                Vec2f p1( getCurrentX() - centerS[0], centerS[1] - getCurrentY() );
                DP_ASSERT( p0[0] != p1[0] || p0[1] != p1[1] );

                //  get the scaling (from model to view)
                Vec3f scaling, translation;
                Quatf orientation, scaleOrientation;
                decompose( m2v, translation, orientation, scaling, scaleOrientation );
                float maxScale = std::max( scaling[0], std::max( scaling[1], scaling[2] ) );
                DP_ASSERT( FLT_EPSILON < fabs( maxScale ) );

                //  determine the radius in screen space (in the centers depth)
                Vec2f centerWindowSize = - centerV[2] / getViewState()->getTargetDistance() * camWinSize;
                float radius = bs.getRadius() * maxScale * rtWidth / centerWindowSize[0];

                //  with p0, p1, and the radius determine the axis and angle of rotation via the Trackball utility
                //  => axis is in view space then
                Vec3f axis;
                float angle;
                m_trackball.setSize( radius );
                m_trackball.apply( p0, p1, axis, angle );

                float dx = p1[0]-p0[0];
                float dy = p1[1]-p0[1];

                checkLockAxis(dx, dy);

                if ( m_activeLockAxis[static_cast<size_t>(Axis::X)] )
                {
                  if ( dx < 0 )
                    axis = Vec3f(0.f, -1.f, 0.f);
                  else if ( dx > 0)
                    axis = Vec3f(0.f, 1.f, 0.f);
                  else
                    return false;
                }
                else if ( m_activeLockAxis[static_cast<size_t>(Axis::Y)] )
                {
                  if ( dy < 0 )
                    axis = Vec3f(1.f, 0.f, 0.f);
                  else if ( dy > 0)
                    axis = Vec3f(-1.f, 0.f, 0.f);
                  else
                    return false;
                }

                // transform axis back into model space
                axis = Vec3f( Vec4f( axis, 0.0f ) * v2w * w2m );
                axis.normalize();

                //  create the rotation around the center (in model space)
                Trafo trafo;
                trafo.setCenter( bs.getCenter() );
                trafo.setOrientation( Quatf( axis, angle ) );

                //  concatenate this rotation with the current transformation
                trafo.setMatrix( transform->getTrafo().getMatrix() * trafo.getMatrix() );

                //  concatenate this rotation with the original transformation
                //trafo.setMatrix( m_matrix * trafo.getMatrix() );

                //  set the current transform
                transform->setTrafo( trafo );

                return true;
              }
            }
          }

          return false;
        }


        bool TrackballTransformManipulator::dolly()
        {
          int dyScreen = getLastY() - getCurrentY();
          if( !dyScreen )
          {
            dyScreen = getWheelTicksDelta();
          }

          if ( dyScreen )
          {
            DP_ASSERT( std::dynamic_pointer_cast<FrustumCamera>(getViewState()->getCamera()) );
            TransformSharedPtr transform = std::static_pointer_cast<Transform>(m_transformPath->getTail());
            FrustumCameraSharedPtr const& camera = std::static_pointer_cast<FrustumCamera>(getViewState()->getCamera());
            if ( camera && transform )
            {
              unsigned int rtWidth = getRenderTarget()->getWidth();
              unsigned int rtHeight = getRenderTarget()->getHeight();
              Vec2f  camWinSize = camera->getWindowSize();
              if (    ( 0 < rtHeight ) && ( 0 < rtWidth )
                  &&  ( FLT_EPSILON < fabs( camWinSize[0] ) )
                  &&  ( FLT_EPSILON < fabs( camWinSize[1] ) ) )
              {
                //  get all the matrices needed here
                Mat44f m2w, w2m, w2v, v2w;
                m_transformPath->getModelToWorldMatrix(m2w, w2m);   // model->world and world->model
                w2v = camera->getWorldToViewMatrix();               // world->view
                v2w = camera->getViewToWorldMatrix();            // view->world

                // transfer mouse delta into view space
                float dyView = camWinSize[1]/rtHeight * dyScreen;

                // transfer the mouse delta vector into the model space
                Vec4f modelDelta = Vec4f( 0.0f, 0.0f, dyView, 0.0f ) * v2w * w2m;

                // minus the delta to the translation of the transform
                // minus, because we want mouse down to move the object into the direction of the user
                Trafo trafo = transform->getTrafo();
                trafo.setTranslation( trafo.getTranslation() - Vec3f( modelDelta ) );
                transform->setTrafo( trafo );

                return true;
              }
            }
          }
          return false;
        }

        void TrackballTransformManipulator::setMode( Mode mode )
        {
          if ( m_mode != mode && m_viewState )
          {
            // Store cursor position and camera position on mode change. These values are used for absolute updates.
            //m_orbitCursor = m_propCursor;
            //DP_ASSERT( getTransformPath() );
            //TransformWeakPtr transform = dynamic_cast<TransformWeakPtr>(getTransformPath()->getTail());
            //DP_ASSERT( transform );
            //m_matrix = transform->getTrafo().getMatrix();
            m_mode = mode;
          }
        }

        void TrackballTransformManipulator::lockAxis( Axis axis )
        {
          m_lockAxis[static_cast<size_t>(axis)] = true;
        }

        void TrackballTransformManipulator::unlockAxis( Axis axis )
        {
          m_lockAxis[static_cast<size_t>(axis)] = false;
        }

        void TrackballTransformManipulator::lockMajorAxis( )
        {
          m_lockMajorAxis = true;
        }

        void TrackballTransformManipulator::unlockMajorAxis( )
        {
          m_lockMajorAxis = false;
        }

        void TrackballTransformManipulator::setTransformPath( dp::sg::core::PathSharedPtr const& transformPath )
        {
          m_transformPath = transformPath;
        }

        dp::sg::core::PathSharedPtr const& TrackballTransformManipulator::getTransformPath() const
        {
          return( m_transformPath );
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
