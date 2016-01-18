// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace manipulator
      {

        /*! \brief Simulate walk-through like camera-mouse interaction.
         *  \remarks This manipulator is a special Manipulator that interprets/converts CursorState
         *  into walk-through camera movements. The user can simulate walk forward, walk backward, 
         *  increase speed, decrease speed, tilt and translate.\n
         *
         *  CursorState is updated by the derived class WalkCameraManipulatorSync, and therefore users 
         *  will probably not use this class directly.
         *
         *  \sa Manipulator, WalkCameraManipulatorSync */
        class WalkCameraManipulator : public Manipulator, public CursorState
        {
          public:
            enum class Mode
            {
              FREELOOK     = 0x01,
              WALK         = 0x02,
              TRANSLATE    = 0x04,
              STRAFE_RIGHT = 0x08,
              STRAFE_LEFT  = 0x10
            };

            typedef dp::util::Flags<Mode> ModeMask;

            /*! \brief Constructor .
             *  \param sensitivity The vector should be greater than (0,0).  This will set the "look" controls sensitivity
             *  in the X and Y axes.  The sensitivity value is the ratio of window pixels to angular displacement in the walk
             *  manipulator.  A value of 1.0 means 1 window pixel = 1 degree of movement in that axis.  Values closer to zero 
             *  will decrease the sensitivity (so a larger mouse movement has less affect on the controls).  The default value
             *  works well for most applications.
             *  \remarks Initialize the WalkCameraManipulator object.
             **/
            DP_SG_UI_MANIPULATOR_API WalkCameraManipulator( const dp::math::Vec2f & sensitivity = dp::math::Vec2f( 0.8f, 0.8f ) );

            /*! \brief Default destructor. */
            DP_SG_UI_MANIPULATOR_API virtual ~WalkCameraManipulator();

            /*! \brief Set the Height Above Terrain offset.
             *  \param offset The distance, in the direction of the "UP" vector, to offset the viewpoint
             *         from the terrain surface.  The default is 2 units.
             *  \sa getCameraHeightAboveTerrain */
            void setCameraHeightAboveTerrain( float offset );

            /*! \brief Get the Height Above Terrain offset.
             *  \returns The distance, in the direction of the "UP" vector, to offset the viewpoint
             *           from the terrain surface.
             *  \sa setCameraHeightAboveTerrain */
            float getCameraHeightAboveTerrain();

            /*! \brief Communicates the world up-vector to the Manipulator.
             * \param up Indicates the world up-vector. 
             * \remarks In particular for implementing walk or fly operations it is essential 
             * for the manipulator to know the world's up vector to manipulate the camera 
             * orientation according. 
             * \remarks The initial up vector is (0,1,0) - Y-up
             * \sa getUpVector */
            void setUpVector( const dp::math::Vec3f & );

            /*! \brief Retrieves the world up-vector.
             * \return The world up-vector. 
             * \sa setUpVector */
            dp::math::Vec3f getUpVector() const;

            /*! \brief Specifies the speed used to perform camera motions. 
             * \param speed Speed to set.  The speed is in "units-per-second" where 'units' are database units. 
             * \remarks The initial speed is set to 0.0
             * \sa getSpeed */
            virtual void setSpeed( float speed );

            /*! \brief Get speed used to perform camera motions.
             *  \return Speed value that has been set.
             *  \sa setSpeed */
            virtual float getSpeed() const;

            /*! \brief Set the mode this manipulator is operating in.
             *  \remarks The manipulator's mode determines how it updates the camera upon every timestep. This 
             *  Manipulator supports the following list of modes (which may be or'ed together):
             *
             *  Mode::WALK           Typical walk mode where direction is based on camera "look" vector.
             *  Mode::TRANSLATE      Translate in a circle around the hitpoint.
             *  Mode::FREELOOK       Freeform "look" where viewing direction is based on mouse position.
             *  Mode::STRAFE_RIGHT   Strafe right (move right, in a direction parallel to Forward X Up vectors)
             *  Mode::STRAFE_LEFT    Strafe left (move left, in a direction parallel to Forward X Up vectors)
             *
             *  \sa getMode */
            void setMode( ModeMask mode );

            /*! \brief Get the mode this manipulator is operating in.
             *  \return The manipulator's current mode.
             *  \sa setMode */
            unsigned int getMode() const;

            /*! \brief Updates the manipulator's timestamp, and runs the manipulator.
             *  \param dt Delta time passage since this function was called last, in seconds.
             *  \remarks This function should be called continuously when this manipulator is active to update the 
             *  ViewState's camera using the current walk model.
             */
            virtual bool updateFrame( float dt );

            /*! \brief Resets the manipulator to defaults.
             * \remarks Resets the manipulator to initial state.
             */
            virtual void reset();

            /*! \brief Sets this manipulator's ViewState.
             *  \param viewState The ViewState this manipulator should use.
             *  \remarks Overridden to detect when a new scene has been set so that the camera orientation can be saved.
             *  \sa getViewState */
            virtual void setViewState( dp::sg::ui::ViewStateSharedPtr const& viewState );

          protected:

            /*! \brief Do the walk.
            *  \return This function returns true when a redraw is needed. 
            *  \remarks Implementation of the walk functionality.
            *  \sa translate, freeLook, strafe */
            DP_SG_UI_MANIPULATOR_API virtual bool walk();

            /*! \brief Do the translate.
            *  \return This function returns true when a redraw is needed. 
            *  \remarks Implementation of the translate functionality.
            *  \sa walk, freeLook, strafe */
            DP_SG_UI_MANIPULATOR_API virtual bool translate();

            /*! \brief Do the freeLook.
            *  \return This function returns true when a redraw is needed. 
            *  \remarks Implementation of the freeLook functionality.
            *  \sa walk, translate, strafe */
            DP_SG_UI_MANIPULATOR_API virtual bool freeLook();

            /*! \brief Do the strafe.
            *  \return This function returns true when a redraw is needed. 
            *  \remarks Implementation of the strafe functionality.
            *  \sa walk, translate, freeLook */
            DP_SG_UI_MANIPULATOR_API virtual bool strafe( bool right );

            /*! \brief Records current camera direction, minus any up vector contribution.
             */
            DP_SG_UI_MANIPULATOR_API void recordPlanarCameraDirection();

          protected:

            //! Used to find the terrain
            DP_SG_UI_MANIPULATOR_API bool findTerrainPosition();

            float m_cameraHeightAboveTerrain;//!< Camera's Height Above Terrain. Default: 2.f.
            dp::math::Vec3f m_upVector;
            ModeMask m_mode;
            float m_speed;
            dp::math::Vec2f m_sensitivity;
            dp::math::Vec3f m_saveCameraDirection;
            dp::math::Vec3f m_currentPosition;
            float m_deltaT;
        };

        inline WalkCameraManipulator::ModeMask operator|( WalkCameraManipulator::Mode bit0, WalkCameraManipulator::Mode bit1 )
        {
          return WalkCameraManipulator::ModeMask( bit0 ) | bit1;
        }

        inline void WalkCameraManipulator::setViewState( dp::sg::ui::ViewStateSharedPtr const& viewState)
        {
          Manipulator::setViewState( viewState );
          recordPlanarCameraDirection();
        }

        inline void WalkCameraManipulator::setSpeed( float spd )
        {
          m_speed = spd;
        }

        inline float WalkCameraManipulator::getSpeed() const
        {
          return m_speed;
        }

        inline void WalkCameraManipulator::setCameraHeightAboveTerrain( float offset )
        {
          m_cameraHeightAboveTerrain = offset; 
        }

        inline float WalkCameraManipulator::getCameraHeightAboveTerrain()
        {
          return m_cameraHeightAboveTerrain; 
        }

        inline void WalkCameraManipulator::setMode( ModeMask mode )
        {
          m_mode = mode;
        }

        inline unsigned int WalkCameraManipulator::getMode() const
        {
          return m_mode;
        }

        inline void WalkCameraManipulator::setUpVector( const dp::math::Vec3f & up )
        {
          m_upVector = up;
        }

        inline dp::math::Vec3f WalkCameraManipulator::getUpVector() const
        {
          return m_upVector;
        }

      } // namespace manipulator
    } // namespace ui
  } // namespace sg
} // namespace dp
