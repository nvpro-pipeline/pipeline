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


#include <dp/sg/ui/manipulator/TrackballTransformManipulatorHIDSync.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        TrackballTransformManipulatorHIDSync::TrackballTransformManipulatorHIDSync( )
          : TrackballTransformManipulator()
          , PID_Pos(0)
          , PID_Rotate(0)
          , PID_Pan(0)
          , PID_Control(0)
          , PID_Wheel(0)
        {
        }

        TrackballTransformManipulatorHIDSync::~TrackballTransformManipulatorHIDSync()
        {
        }

        bool TrackballTransformManipulatorHIDSync::updateFrame( float dt )
        {
          if (!m_hid)
          {
            return false;
          }

          bool rotate  = m_hid->getValue<bool>( PID_Rotate );
          bool pan     = m_hid->getValue<bool>( PID_Pan );
          bool control = m_hid->getValue<bool>( PID_Control );

          if ( control )
          {
            lockMajorAxis();
          }
          else
          {
            unlockMajorAxis();
          }

          setCursorPosition( m_hid->getValue<dp::math::Vec2i>( PID_Pos ) );
          setWheelTicks( m_hid->getValue<int>( PID_Wheel ) );

          TrackballTransformManipulator::Mode mode = TrackballTransformManipulator::Mode::NONE;

          if ( rotate && pan || getWheelTicksDelta() )
          {
            mode = TrackballTransformManipulator::Mode::DOLLY;
          }
          else if ( !rotate && pan )
          {
            mode = TrackballTransformManipulator::Mode::PAN;
          }
          else if ( rotate && !pan )
          {
            mode = TrackballTransformManipulator::Mode::ROTATE;
          }

          setMode( mode );

          return TrackballTransformManipulator::updateFrame( dt );
        }

        void TrackballTransformManipulatorHIDSync::setHID( HumanInterfaceDevice *hid )
        {
          m_hid = hid;

          PID_Pos     = m_hid->getProperty("Mouse_Position" );
          PID_Rotate  = m_hid->getProperty("Mouse_Left" );
          PID_Pan     = m_hid->getProperty("Mouse_Middle" );
          PID_Wheel   = m_hid->getProperty("Mouse_Wheel");
          PID_Control = m_hid->getProperty("Key_Control");
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
