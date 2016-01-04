// Copyright NVIDIA Corporation 2002-2005
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
#include <dp/sg/core/CoreTypes.h>
#include <dp/ui/RenderTarget.h>
#include <dp/sg/ui/Trackball.h>
#include <dp/math/Vecnt.h>
#include <dp/math/Quatt.h>
#include <dp/util/Reflection.h>


namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        /*! \brief Simulates a trackball-like camera mouse interaction.
         *  
         *  This Manipulator uses the CursorState to drive a certain camera operations like
         *  orbit, pan, dolly, rotate, roll, and zoom.
         *
         *  CursorState is updated by the derived class TrackballCameraManipulatorSync, and therefore users 
         *  will probably not use this class directly.
         *
         *  \sa Manipulator, TrackballCameraManipulatorSync */
        class TrackballCameraManipulator : public Manipulator, public CursorState
        {
        public:

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
            , LOOKAT_DEPTH
          };

          enum class Axis
          {
              X = 0
            , Y = 1
            , Z = 2
          };

          /*! \brief Default constructor.
           *  \remarks Initialize the TrackballCameraManipulator object.*/
          DP_SG_UI_MANIPULATOR_API TrackballCameraManipulator();

          /*! \brief Default destructor. */
          DP_SG_UI_MANIPULATOR_API virtual ~TrackballCameraManipulator();
  
          DP_SG_UI_MANIPULATOR_API virtual bool updateFrame( float dt );
          DP_SG_UI_MANIPULATOR_API virtual void reset();

          DP_SG_UI_MANIPULATOR_API void setMode( Mode mode );
          DP_SG_UI_MANIPULATOR_API void lockAxis( Axis axis );
          DP_SG_UI_MANIPULATOR_API void unlockAxis( Axis axis );

          DP_SG_UI_MANIPULATOR_API void lockMajorAxis();
          DP_SG_UI_MANIPULATOR_API void unlockMajorAxis();

        protected:
          dp::sg::ui::Trackball m_trackball;       //!< Trackball object that does all the calculations

          Mode m_mode;

          // Properties
          dp::math::Vec2i m_startSpin;

          dp::math::Vec2i m_orbitCursor;
          dp::math::Quatf m_orbitCameraOrientation;
          dp::math::Vec3f m_orbitCameraPosition;

        protected:
          float m_speed;

          /* Better to have bool m_lock[3]? */
          bool m_lockAxis[3];       // requested locks by user
          bool m_activeLockAxis[3]; // current active locks
          bool m_lockMajorAxis;     // true if major axis should be locked

          /*! \brief Performs the orbit operation.
           *  \return This function returns true when a redraw is needed. */
          DP_SG_UI_MANIPULATOR_API virtual bool orbit();

          /*! \brief Performs the pan operation.
           *  \return This function returns true when a redraw is needed. */
          DP_SG_UI_MANIPULATOR_API virtual bool pan();

          /*! \brief Performs the dolly operation.
           *  \return This function returns true when a redraw is needed. */
          DP_SG_UI_MANIPULATOR_API virtual bool dolly();

          /*! \brief Performs the zoom operation.
           *  \return This function returns true when a redraw is needed. */
          DP_SG_UI_MANIPULATOR_API virtual bool zoom();

          /*! \brief Performs a dolly together with a zoom to mimic a 'vertigo effect'
           *  \return This function returns true when a redraw is needed. 
           *  \remarks This mimics a so called 'vertigo effect'.
           *  The effect is achieved by using the setting of a zoom lens to adjust the 
           *  angle of view (often referred to as field of view) while the camera dollies 
           *  (or moves) towards or away from the subject in such a way as to keep the subject 
           *  the same size in the frame throughout. */
          DP_SG_UI_MANIPULATOR_API virtual bool dollyZoom();

          /*! \brief Performs a XY-rotation.
           *  \return This function returns true when a redraw is needed. */
          DP_SG_UI_MANIPULATOR_API virtual bool rotate();

          /*! \brief Performs a Z-rotation.
          *  \return This function returns true when a redraw is needed. */
          DP_SG_UI_MANIPULATOR_API virtual bool roll();

          /*! \brief Performs a look-at operation.
           *  \return This function returns true when a redraw is needed. */
          DP_SG_UI_MANIPULATOR_API virtual bool lookAt();

          /*! \brief Performs a look-at operation based on the depth buffer of the current OpenGL Framebuffer.
           *  \return This function returns true when a redraw is needed. */
          DP_SG_UI_MANIPULATOR_API virtual bool lookAtDepthGL();

        private:
          template<typename T>
          void checkLockAxis(T dx, T dy);
        };

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
