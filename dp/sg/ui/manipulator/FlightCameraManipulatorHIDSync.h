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


#pragma once

#include <dp/sg/ui/HumanInterfaceDevice.h>
#include <dp/sg/ui/manipulator/FlightCameraManipulator.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        /*! \brief FlightCameraManipulatorHIDSync "synchronizes" the HID input device to the CursorState of a FlightCameraManipulator.
         *
         * The FlightCameraManipulatorHIDSync controls the FlightCameraManipulator with the following HID inputs: \n\n
         * <table>
         * <tr><td> <b>Operation</b></td>    <td><b>MouseButton</b></td>     <td><b>KeyState</b></td></tr>
         * <tr><td>Fly forward</td>          <td>Mouse_Left</td>             <td></td></tr>
         * <tr><td>Fly backward</td>         <td>Mouse_Middle</td>           <td></td></tr>
         * <tr><td>Accelerate</td>           <td>Mouse_Wheel</td>            <td></td></tr> 
         * <tr><td>Decelerate</td>           <td>Mouse_Wheeltd>              <td></td></tr> 
         * </table>
         *  \n
         *  When the Manipulator is in flight mode (forward or backward), one can determine the 
         *  flight direction by moving the mouse to the left or right. One can also increase or
         *  decrease the flight speed by pressing the application defined acceleration or deceleration
         *  keys.
         *  \remarks Typically the application will want to run the FlightCameraManipulator continuously, passing appropriate
         *  delta-time passage as the 'dt' argument to updateFrame().  If this is not done, then the flight manipulator will only
         *  move when one of the control inputs changes.
         *  \sa FlightCameraManipulatorHID */
        class FlightCameraManipulatorHIDSync : public FlightCameraManipulator
        {
        public:
          /*! \brief Constructor .
           *  \param sensitivity The vector should be greater than (0,0).  This will set the flight controls sensitivity
           *  in the X and Y axes.  The sensitivity value is the ratio of window pixels to angular velocity in the flight
           *  manipulator.  A value of 1.0 means 1 window pixel results in an angular velocity of 1 degree per second on that axis.
           *  Values closer to zero will decrease the sensitivity (so a larger mouse movement has less affect on the controls).
           *  The default value works well for most applications.
           *  \remarks Initialize the FlightCameraManipulator object.  The world up direction is taken from the camera.
           *  If the world up is the default (0.f, 1.f, 0.f), you are flying in the x-z-plane and the ceiling/heaven 
           *  is in the y-direction. */
          DP_SG_UI_MANIPULATOR_API FlightCameraManipulatorHIDSync( const dp::math::Vec2f & sensitivity = dp::math::Vec2f( 0.2f, 0.2f ) );
          DP_SG_UI_MANIPULATOR_API virtual ~FlightCameraManipulatorHIDSync();

          DP_SG_UI_MANIPULATOR_API void setHID( HumanInterfaceDevice *hid );
          DP_SG_UI_MANIPULATOR_API virtual bool updateFrame( float dt );

          /*! \brief Specifies the speed used to perform camera motions. 
           * \param speed Speed to set.  The speed is in "units-per-second" where 'units' are database units, and 
           * the time reference comes from the argument to updateFrame().
           * \note This speed is cached even when the FlightCameraManipulatorHIDSync is "stopped" (not moving because no buttons are
           * pressed) whereas the Speed stored in the FlightCameraManipulator base class will be 0.0 when "stopped," and nonzero when
           * moving.
           * \remarks The initial speed is set to 1.0
           * \sa getSpeed */
          virtual void setSpeed( float speed );

          /*! \brief Get speed used to perform camera motions.
           *  \return Speed value that has been set.
           *  \sa setSpeed */
          virtual float getSpeed() const;

        private:
          /* Property ids for HID input */
          dp::util::PropertyId PID_Pos;
          dp::util::PropertyId PID_Wheel;
          dp::util::PropertyId PID_Forward;
          dp::util::PropertyId PID_Reverse;

          HumanInterfaceDevice *m_hid;
          float m_speed;
        };

        inline void FlightCameraManipulatorHIDSync::setSpeed( float speed )
        {
          m_speed = speed;
        }

        inline float FlightCameraManipulatorHIDSync::getSpeed() const
        {
          return m_speed;
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
