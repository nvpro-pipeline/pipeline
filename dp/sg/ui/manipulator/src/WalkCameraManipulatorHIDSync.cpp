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


#include <dp/sg/ui/manipulator/WalkCameraManipulatorHIDSync.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        WalkCameraManipulatorHIDSync::WalkCameraManipulatorHIDSync( const dp::math::Vec2f & sensitivity )
          : WalkCameraManipulator( sensitivity )
          , PID_Pos(0)
          , PID_Wheel(0)
          , PID_ButtonForward(0)
          , PID_ButtonReverse(0)
          , PID_KeyForward(0)
          , PID_KeyReverse(0)
          , PID_KeyStrafeLeft(0)
          , PID_KeyStrafeRight(0)
          , PID_KeyUp(0)
          , PID_KeyDown(0)
          , PID_KeyRun(0)
          , m_speed(1.f)
          , m_reverseMouse( false )
        {
        }

        WalkCameraManipulatorHIDSync::~WalkCameraManipulatorHIDSync()
        {
        }

        bool WalkCameraManipulatorHIDSync::updateFrame( float dt )
        {
          if (!m_hid)
          {
            return false;
          }

          dp::math::Vec2i pos = m_hid->getValue<dp::math::Vec2i>( PID_Pos );
          if( getReverseMouse() && getRenderTarget() )
          {
            pos[1] = getRenderTarget()->getHeight() - pos[1] - 1;
          }

          setCursorPosition( pos );

          setWheelTicks( m_hid->getValue<int>( PID_Wheel ) );

          // set height above terrain
          if( m_hid->getValue<bool>( PID_KeyUp ) )
          {
            setCameraHeightAboveTerrain( getCameraHeightAboveTerrain() + m_sensitivity[1] );
          }
          else if( m_hid->getValue<bool>( PID_KeyDown ) )
          {
            float hat = getCameraHeightAboveTerrain() - m_sensitivity[1];
            if( hat < 0.f )
            {
              hat = 0.f;
            }

            setCameraHeightAboveTerrain( hat );
          }

          bool forward = m_hid->getValue<bool>( PID_ButtonForward ) ||
                         m_hid->getValue<bool>( PID_KeyForward );
          bool reverse = m_hid->getValue<bool>( PID_ButtonReverse ) ||
                         m_hid->getValue<bool>( PID_KeyReverse );

          bool strafeLeft  = m_hid->getValue<bool>( PID_KeyStrafeLeft );
          bool strafeRight = m_hid->getValue<bool>( PID_KeyStrafeRight );

          float run = m_hid->getValue<bool>( PID_KeyRun ) ? 2.0f : 1.0f;

          unsigned int mode = MODE_FREELOOK;
  
          // update speed based on wheel
          m_speed += getWheelTicksDelta() * 0.1f;
          if( m_speed < 0.f )
          {
            m_speed = 0.f;
          }

          if( forward || reverse )
          {
            // set forward, reverse here
            float speed = forward ? m_speed : -m_speed;
            WalkCameraManipulator::setSpeed( speed * run );

            mode |= MODE_WALK;
          }

          if( strafeLeft )
          {
            WalkCameraManipulator::setSpeed( m_speed * run );
            mode |= MODE_STRAFE_LEFT;
          }
          else if( strafeRight )
          {
            WalkCameraManipulator::setSpeed( m_speed * run );
            mode |= MODE_STRAFE_RIGHT;
          }

          // if not moving, set speed to 0
          if( mode == MODE_FREELOOK )
          {
            // stopped
            WalkCameraManipulator::setSpeed( 0.f );
          }

          setMode( mode );

          return WalkCameraManipulator::updateFrame( dt );
        }

        void WalkCameraManipulatorHIDSync::setHID( HumanInterfaceDevice *hid )
        {
          m_hid = hid;

          PID_Pos           = m_hid->getProperty( "Mouse_Position" );
          PID_ButtonForward = m_hid->getProperty( "Mouse_Left" );
          PID_ButtonReverse = m_hid->getProperty( "Mouse_Middle" );
          PID_Wheel         = m_hid->getProperty( "Mouse_Wheel" );

          PID_KeyForward     = m_hid->getProperty( "Key_W" );
          PID_KeyReverse     = m_hid->getProperty( "Key_S" );
          PID_KeyStrafeLeft  = m_hid->getProperty( "Key_A" );
          PID_KeyStrafeRight = m_hid->getProperty( "Key_D" );
          PID_KeyRun         = m_hid->getProperty( "Key_Shift" );

          PID_KeyUp          = m_hid->getProperty( "Key_Q" );
          PID_KeyDown        = m_hid->getProperty( "Key_E" );
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
