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
#include <dp/sg/ui/Trackball.h>

#include <dp/sg/core/Path.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        /*! \brief Simulate a trackball like object interaction.
         *  
         *  This Manipulator uses the CursorState to drive a transform to do operations like
         *  pan, dolly, and rotate.
         *
         *  CursorState is updated by the derived class TrackballCameraManipulatorSync, and therefore users 
         *  will probably not use this class directly.
         *
         *  \sa Manipulator, TrackballTransformManipulatorSync */
        class TrackballTransformManipulator : public Manipulator, public CursorState
        {
        public:
  
          enum class Mode
          {
              NONE = 0
            , PAN
            , ROTATE
            , DOLLY
          };

          enum class Axis
          {
              X = 0
            , Y = 1
            , Z = 2
          };

          /*! \brief Default constructor.
           *  \remarks Initialize the TrackballTransformManipulator object.*/
          DP_SG_UI_MANIPULATOR_API TrackballTransformManipulator(void);
  
          /*! \brief Default destructor. */
          DP_SG_UI_MANIPULATOR_API virtual ~TrackballTransformManipulator(void);

          /*! \brief Updates the manipulator's timestamp, and runs the manipulator.
           *  \param dt Delta time passage since this function was called last, in seconds.
           *  \remarks This function should be called continuously when this manipulator is active to update the 
           *  ViewState's camera using the current flight model.
           */
          DP_SG_UI_MANIPULATOR_API virtual bool updateFrame( float dt );

          /*! \brief Resets the manipulator to defaults.
           * \remarks Resets the manipulator to initial state.
           */
          DP_SG_UI_MANIPULATOR_API virtual void reset();

          /*! \brief Set the mode this manipulator is operating in.
           *  \remarks The manipulator's mode determines how it updates the object upon every timestep. This 
           *  Manipulator supports the following list of modes:
           *
           *  Mode::PAN     Translate the objects below the transform in the XZ plane.
           *  Mode::ROTATE  Rotate the object using a trackball style based on the object's bounding volume.
           *  Mode::DOLLY   Translate the object toward or away from the viewer.
           *
           *  \sa getMode */
          DP_SG_UI_MANIPULATOR_API void setMode( Mode mode );

          /*! \brief Get the mode this manipulator is operating in.
           *  \return The manipulator's current mode.
           *  \sa setMode */
          Mode getMode() const;

          DP_SG_UI_MANIPULATOR_API void lockAxis( Axis axis );
          DP_SG_UI_MANIPULATOR_API void unlockAxis( Axis axis );

          DP_SG_UI_MANIPULATOR_API void lockMajorAxis();
          DP_SG_UI_MANIPULATOR_API void unlockMajorAxis();

          /*! \brief Set the dp::sg::core::Path to the dp::sg::core::Transform node.
           *  \param transformPath Complete path to the dp::sg::core::Transform node. Null is a valid value to 
           *  disconnect the TrackballTransformManipulator from the controlled object.
           *  \remarks Attach the Manipulator to the desired dp::sg::core::Transform in the tree by providing 
           *  a complete dp::sg::core::Path from the root node to the dp::sg::core::Transform node.\n
           *  This class takes care of incrementing and decrementing the reference count of the provided
           *  dp::sg::core::Path object.
           *  \n\n 
           *  The application is responsible to make sure that the dp::sg::core::Path stays 
           *  valid during the usage of the Manipulator.
           *  \sa getTransformPath */
          DP_SG_UI_MANIPULATOR_API void setTransformPath( dp::sg::core::PathSharedPtr const& transformPath );

          /*! \brief Get the dp::sg::core::Path to the dp::sg::core::Transform that currently is under control.
           *  \return Path to the controlled dp::sg::core::Transform. If the TrackballTransformManipulator is not 
           *  connected this function returns NULL.
           *  \remarks  The application is responsible to make sure that the dp::sg::core::Path stays 
           *  valid during the usage of the Manipulator.
           *  \sa setTransformPath */
          DP_SG_UI_MANIPULATOR_API dp::sg::core::PathSharedPtr const& getTransformPath() const;

        protected:
          dp::sg::ui::Trackball       m_trackball;      // Trackball object that does all the calculations
          dp::sg::core::PathSharedPtr m_transformPath;  //!< Complete dp::sg::core::Path to the dp::sg::core::Transform node.

          // new interface
          Mode m_mode;

          /* Better to have bool m_lock[3]? */
          bool m_lockAxis[3];       // requested locks by user
          bool m_activeLockAxis[3]; // current active locks
          bool m_lockMajorAxis;     // true if major axis should be locked

        protected:
          /*! \brief Do the pan.
          *  \return This function returns true when a redraw is needed. 
          *  \remarks Implementation of the pan functionality.*/
          DP_SG_UI_MANIPULATOR_API virtual bool pan();

          /*! \brief Do the dolly.
          *  \return This function returns true when a redraw is needed. 
          *  \remarks Implementation of the dolly functionality.*/
          DP_SG_UI_MANIPULATOR_API virtual bool dolly();

          /*! \brief Do the rotate.
          *  \return This function returns true when a redraw is needed. 
          *  \remarks Implementation of the rotate functionality. */
          DP_SG_UI_MANIPULATOR_API virtual bool rotate();

        private:
          template<typename T>
          void checkLockAxis(T dx, T dy);
        };

        inline TrackballTransformManipulator::Mode TrackballTransformManipulator::getMode() const
        {
          return m_mode;
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
