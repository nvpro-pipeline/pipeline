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


#include <dp/sg/ui/manipulator/FlightCameraManipulatorHIDSync.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        FlightCameraManipulatorHIDSync::FlightCameraManipulatorHIDSync( const dp::math::Vec2f & sensitivity )
          : FlightCameraManipulator( sensitivity )
          , PID_Pos(0)
          , PID_Wheel(0)
          , PID_Forward(0)
          , PID_Reverse(0)
          , m_speed(1.f)
        {

        }

        FlightCameraManipulatorHIDSync::~FlightCameraManipulatorHIDSync()
        {
        }

        bool FlightCameraManipulatorHIDSync::updateFrame( float dt )
        {
          if (!m_hid)
          {
            return false;
          }

          setCursorPosition( m_hid->getValue<dp::math::Vec2i>( PID_Pos ) );
          setWheelTicks( m_hid->getValue<int>( PID_Wheel ) );

          bool forward = m_hid->getValue<bool>( PID_Forward );
          bool reverse = m_hid->getValue<bool>( PID_Reverse );
  
          // set speed based on wheel and buttons
          m_speed  += getWheelTicksDelta() * 0.1f;
          if( m_speed < 0.f )
          {
            m_speed = 0.f;
          }

          if( forward || reverse )
          {
            // set forward, reverse here
            FlightCameraManipulator::setSpeed( forward ? m_speed : -m_speed );
          }
          else
          {
            // stopped
            FlightCameraManipulator::setSpeed( 0.f );
          }

          return FlightCameraManipulator::updateFrame( dt );
        }

        void FlightCameraManipulatorHIDSync::setHID( HumanInterfaceDevice *hid )
        {
          m_hid = hid;

          PID_Pos     = m_hid->getProperty( "Mouse_Position" );
          PID_Forward = m_hid->getProperty( "Mouse_Left" );
          PID_Reverse = m_hid->getProperty( "Mouse_Middle" );
          PID_Wheel   = m_hid->getProperty( "Mouse_Wheel" );
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
