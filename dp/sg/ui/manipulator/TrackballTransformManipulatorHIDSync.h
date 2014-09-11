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
#include <dp/sg/ui/manipulator/TrackballTransformManipulator.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        /*! \brief TrackballTransformManipulatorHIDSync "synchronizes" the HID input device to the CursorState of a TrackballTransformManipulator.
         *  
         *  This Manipulator drives the CursorState of a TrackballTransformManipulatorHID.
         *
         *  This Manipulator is a special Manipulator that interprets/converts
         *  mouse movement into dp::sg::core::Transform changes. You can rotate the object, pan, and  
         *  dolly it in and out. So you can freely move and place an object in your scene.
         *  \n
         *  The TrackballTransformManipulatorHIDSync uses the following HID input mapping: \n\n
         *
         *  <table>
         *  <tr><td> <b>Operation</b></td>             <td><b>MouseButton</b></td>            <td><b>KeyState</b></td></tr>
         *  <tr><td>XY-Rotate</td>                     <td>Mouse_Left</td>                    <td></td></tr>
         *  <tr><td>XY-Rotate - lock major axis</td>   <td>Mouse_Left</td>                    <td>Key_Shift</td></tr>
         *  <tr><td>Pan</td>                           <td>Mouse_Middle</td>                  <td></td></tr>
         *  <tr><td>Pan - lock major axis</td>         <td>Mouse_Middle</td>                  <td>Key_Shift</td></tr> 
         *  <tr><td>Dolly</td>                         <td>Mouse_Left + Mouse_Middle</td>     <td></td></tr> 
         *  </table>
         *
         *  \n
         *  See the TransformManipulatorViewer tutorial source code on how to incorporate a 
         *  TrackballTransformManipulatorHID in your application.
         *  \note Note, that a TrackballTransformManipulatorHID needs a FrustumCamera to work with. The behaviour is
         *  undefined in all other cases.
         *  \n\n
         *  \sa Manipulator, TrackballTransformManipulatorHID */
        class TrackballTransformManipulatorHIDSync : public TrackballTransformManipulator
        {
        public:
          DP_SG_UI_MANIPULATOR_API TrackballTransformManipulatorHIDSync( );
          DP_SG_UI_MANIPULATOR_API virtual ~TrackballTransformManipulatorHIDSync();

          DP_SG_UI_MANIPULATOR_API void setHID( HumanInterfaceDevice *hid );

          DP_SG_UI_MANIPULATOR_API virtual bool updateFrame( float dt );

        private:
          /* Property ids for HID input */
          dp::util::PropertyId PID_Pos;
          dp::util::PropertyId PID_Rotate;
          dp::util::PropertyId PID_Pan;
          dp::util::PropertyId PID_Control;
          dp::util::PropertyId PID_Wheel;

          HumanInterfaceDevice *m_hid;
        };

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
