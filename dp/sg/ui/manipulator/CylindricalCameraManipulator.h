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


#pragma once
/** \file */

#include <dp/sg/ui/manipulator/Config.h>
#include <dp/sg/ui/Manipulator.h>
#include <dp/sg/ui/Trackball.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        class CylindricalCameraManipulator : public Manipulator, public CursorState
        {
        public:
          DP_SG_UI_MANIPULATOR_API CylindricalCameraManipulator();
          DP_SG_UI_MANIPULATOR_API virtual ~CylindricalCameraManipulator();

          enum class Mode
          {
              NONE = 0
            , ORBIT
            , PAN
            , ROTATE_XY
            , DOLLY
            , ROLL_Z
            , ZOOM_FOV
            , ZOOM_DOLLY
            , LOOKAT
          };

          enum class Axis
          {
              X = 0
            , Y = 1
            , Z = 2
          };

          DP_SG_UI_MANIPULATOR_API void setMode( Mode mode );
          Mode getMode() const;

          DP_SG_UI_MANIPULATOR_API void setSpeed( float speed );
          float getSpeed() const;

          DP_SG_UI_MANIPULATOR_API virtual bool updateFrame( float dt );
          DP_SG_UI_MANIPULATOR_API virtual void reset();

          DP_SG_UI_MANIPULATOR_API void lockAxis( Axis axis );
          DP_SG_UI_MANIPULATOR_API void unlockAxis( Axis axis );

          DP_SG_UI_MANIPULATOR_API void lockMajorAxis();
          DP_SG_UI_MANIPULATOR_API void unlockMajorAxis();
  
        protected:
   
          unsigned int m_flags;
          dp::sg::ui::Trackball m_trackball;       //!< Trackball object that does all the calculations
          int m_startSpinX, m_startSpinY;
          int m_currentSpinX, m_currentSpinY;

          /* Better to have bool m_lock[3]? */
          bool m_lockAxis[3];       // requested locks by user
          bool m_activeLockAxis[3]; // current active locks
          bool m_lockMajorAxis;     // true if major axis should be locked

          float m_speed;

        protected:

          DP_SG_UI_MANIPULATOR_API bool orbit();     // Custom roll free orbit.
          DP_SG_UI_MANIPULATOR_API bool pan();
          DP_SG_UI_MANIPULATOR_API bool dolly();     // Custom dolly keeping the focus point until reaching camera minimum focus distance.
          DP_SG_UI_MANIPULATOR_API bool zoom();
          DP_SG_UI_MANIPULATOR_API bool dollyZoom();
          DP_SG_UI_MANIPULATOR_API bool rotate();    // Custom roll free rotate.
          DP_SG_UI_MANIPULATOR_API bool roll();
          DP_SG_UI_MANIPULATOR_API bool lookAt();
   
          DP_SG_UI_MANIPULATOR_API bool setPivot();  // Custom operation 0 (also sets the focus point)
          DP_SG_UI_MANIPULATOR_API bool setFocus();  // Custom operation 1

        protected:
          Mode m_mode;

        private:
          template<typename T> void checkLockAxis(T dx, T dy);
        };

        inline CylindricalCameraManipulator::Mode CylindricalCameraManipulator::getMode() const
        {
          return m_mode;
        }

        inline float CylindricalCameraManipulator::getSpeed() const
        {
          return m_speed;
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
