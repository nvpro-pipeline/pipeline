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
#include <dp/sg/ui/manipulator/WalkCameraManipulator.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        /*! \brief WalkCameraManipulatorHIDSync "synchronizes" the HID input devices to the CursorState of a WalkCameraManipulator.
         *
         *  The WalkCameraManipulator comes with the following HID input mapping: \n\n
         *  <table>
         *  <tr><td> <b>Operation</b></td>    <td><b>MouseButton</b></td>     <td><b>KeyState</b></td></tr>
         *  <tr><td>Walk forward</td>         <td>Mouse_Left</td>              <td></td></tr>
         *  <tr><td>Walk backward</td>        <td>Mouse_Middle</td>            <td></td></tr>
         *  <tr><td>Tilt</td>                 <td>Mouse_Position</td>          <td></td></tr> 
         *  <tr><td>Translate</td>            <td>Mouse_Middle</td>            <td>Key_Control</td></tr>
         *  <tr><td>Accelerate</td>           <td>Mouse_Wheel</td>             <td></td></tr> 
         *  <tr><td>Decelerate</td>           <td>Mouse_Wheel</td>             <td></td></tr> 
         *  </table>
         *
         *  The WalkCameraManipulator can also operate like a "First Person Shooter": \n\n
         *
         *  <table>
         *  <tr><td> <b>Operation</b></td>    <td><b>MouseButton</b></td>     <td><b>KeyState</b></td></tr>
         *  <tr><td>Walk forward</td>         <td></td>                       <td>Key_W</td></tr>
         *  <tr><td>Walk backward</td>        <td></td>                       <td>Key_S</td></tr>
         *  <tr><td>Look</td>                 <td>Mouse_Position</td>         <td></td></tr> 
         *  <tr><td>Strafe left</td>          <td></td>                       <td>Key_A</td></tr>
         *  <tr><td>Strafe right</td>         <td></td>                       <td>Key_D</td></tr>
         *  <tr><td>Run (2x speed)</td>       <td></td>                       <td>Key_Shift</td></tr>
         *  </table>
         *  \n
         *  The manipulator automatically connects the eyepoint to the model's surface.  See the \a setCameraHeightAboveTerrain
         *  and \a getCameraHeightAboveTerrain functions for how to manipulate the camera height above the terrain.  
         *  This simulates pushing a camera on a tripod through an environment that will always be this height
         *  above the actual terrain.
         *  \remarks Typically the application will want to run the WalkCameraManipulator continuously, passing appropriate
         *  delta-time passage as the 'dt' argument to updateFrame().  If this is not done, then the walk manipulator will only
         *  move when one of the control inputs changes.
         *  \sa Manipulator, WalkCameraManipulatorHIDSync */
        class WalkCameraManipulatorHIDSync : public WalkCameraManipulator
        {
        public:
          /* \brief Constructor
           *  \param sensitivity The vector should be greater than (0,0).  This will set the "look" controls sensitivity
           *  in the X and Y axes.  The sensitivity value is the ratio of window pixels to angular displacement in the walk
           *  manipulator.  A value of 1.0 means 1 window pixel = 1 degree of movement in that axis.  Values closer to zero 
           *  will decrease the sensitivity (so a larger mouse movement has less affect on the controls).  The default value
           *  works well for most applications.
           */
          DP_SG_UI_MANIPULATOR_API WalkCameraManipulatorHIDSync( const dp::math::Vec2f & sensitivity = dp::math::Vec2f( 0.8f, 0.8f ) );
          DP_SG_UI_MANIPULATOR_API virtual ~WalkCameraManipulatorHIDSync();

          DP_SG_UI_MANIPULATOR_API void setHID( HumanInterfaceDevice *hid );

          DP_SG_UI_MANIPULATOR_API virtual bool updateFrame( float dt );

          /*! \brief Specifies the speed used to perform camera motions. 
           * \param speed Speed to set.  The speed is in "units-per-second" where 'units' are database units, and 
           * the time reference comes from the argument to updateFrame().
           * \note This speed is cached even when the WalkCameraManipulatorHIDSync is "stopped" (not moving because no buttons are
           * pressed) whereas the Speed stored in the WalkCameraManipulator base class will be 0.0 when "stopped," and nonzero when
           * moving.
           * \remarks The initial speed is set to 1.0
           * \sa getSpeed */
          virtual void setSpeed( float speed );

          /*! \brief Get speed used to perform camera motions.
           *  \return Speed value that has been set.
           *  \sa setSpeed */
          virtual float getSpeed() const;

          void setReverseMouse( bool reverse );
          bool getReverseMouse() const;

        private:
          /* Property ids for HID input */
          dp::util::PropertyId PID_Pos;
          dp::util::PropertyId PID_Wheel;
          dp::util::PropertyId PID_ButtonForward;
          dp::util::PropertyId PID_ButtonReverse;

          dp::util::PropertyId PID_KeyForward;
          dp::util::PropertyId PID_KeyReverse;
          dp::util::PropertyId PID_KeyStrafeLeft;
          dp::util::PropertyId PID_KeyStrafeRight;
          dp::util::PropertyId PID_KeyUp;
          dp::util::PropertyId PID_KeyDown;
          dp::util::PropertyId PID_KeyRun;

          HumanInterfaceDevice *m_hid;
          float m_speed;
          bool  m_reverseMouse;
        };

        inline void WalkCameraManipulatorHIDSync::setReverseMouse( bool rev )
        {
          m_reverseMouse = rev;
        }

        inline bool WalkCameraManipulatorHIDSync::getReverseMouse() const
        {
          return m_reverseMouse;
        }

        inline void WalkCameraManipulatorHIDSync::setSpeed( float speed )
        {
          m_speed = speed;
        }

        inline float WalkCameraManipulatorHIDSync::getSpeed() const
        {
          return m_speed;
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
