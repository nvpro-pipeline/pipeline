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
#include <dp/sg/ui/manipulator/CylindricalCameraManipulator.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        /*! \brief CylindricalCameraManipulatorHIDSync "synchronizes" the HID input device to the CursorState of a CylindricalCameraManipulator
         *
         *  The CylindricalCameraManipulatorHIDSyc uses the following HID input mappings:
         *
         *  MMM - NOTE: THIS TABLE NEEDS TO BE UPDATED!!
         *  
         *  <table>
         *  <tr><td> <b>Camera Operation</b></td>      <td><b>MouseButton</b></td>            <td><b>KeyState</b></td></tr>
         *  <tr><td>Orbit</td>                         <td>Mouse_Left</td>                    <td></td></tr>
         *  <tr><td>Orbit - lock major axis</td>       <td>Mouse_Left</td>                    <td>Key_Shift</td></tr>
         *  <tr><td>Continuous Orbit</td>              <td>Mouse_Left</td>                    <td>Key_C</td></tr> 
         *  <tr><td>Pan</td>                           <td>Mouse_Middle</td>                  <td></td></tr>
         *  <tr><td>Pan - lock major axis</td>         <td>Mouse_Middle</td>                  <td>Key_Shift</td></tr> 
         *  <tr><td>Dolly</td>                         <td>Mouse_Left + Mouse_Middle</td>     <td></td></tr> 
         *  <tr><td>XY-Rotate</td>                     <td>Mouse_Left</td>                    <td>Key_Control</td></tr>
         *  <tr><td>XY-Rotate - lock major axis</td>   <td>Mouse_Left</td>                    <td>Key_Control + Key_Shift</td></tr>
         *  <tr><td>Z-Roll</td>                        <td>Mouse_Middle</td>                  <td>Key_Control</td></tr> 
         *  <tr><td>Zoom (FOV)</td>                    <td>Mouse_Left + Mouse_Middle</td>     <td>Key_Control</td></tr> 
         *  <tr><td>Dolly-Zoom (Vertigo)</td>          <td>Mouse_Left + Mouse_Middle</td>     <td>Key_Control + Key_Shift</td></tr>
         *  <tr><td>Set rotation point at cursor</td>  <td></td>                              <td>Key_H</td></tr> 
         *  </table>
         *  \sa Manipulator, CylindricalCameraManipulatorHID */
        class CylindricalCameraManipulatorHIDSync : public CylindricalCameraManipulator
        {
        public:
          DP_SG_UI_MANIPULATOR_API CylindricalCameraManipulatorHIDSync( );
          DP_SG_UI_MANIPULATOR_API virtual ~CylindricalCameraManipulatorHIDSync();

          DP_SG_UI_MANIPULATOR_API void setHID( HumanInterfaceDevice *hid );

          DP_SG_UI_MANIPULATOR_API virtual bool updateFrame( float dt );

        private:
          /* Property ids for HID input */
          dp::util::PropertyId PID_Pos;
          dp::util::PropertyId PID_Orbit;
          dp::util::PropertyId PID_Pan;
          dp::util::PropertyId PID_Shift;
          dp::util::PropertyId PID_Control;
          dp::util::PropertyId PID_HotSpot;
          dp::util::PropertyId PID_Wheel;

          HumanInterfaceDevice *m_hid;
        };

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
