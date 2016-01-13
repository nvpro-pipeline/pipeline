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

#include <dp/math/Vecnt.h>
#include <dp/sg/ui/Config.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/ui/RenderTarget.h>
#include <dp/util/Flags.h>
#include <map>

namespace dp
{
  namespace sg
  {
    namespace ui
    {

      /*! \brief Base class for Manipulators.
       *  A Manipulator provides the ability to control a certain object on a user interaction.
       *  This class serves a base class for concrete Manipulators. As an example, a concrete 
       *  Manipulator can implement an interactive control of an dp::sg::core::Camera object, where
       *  user commands are passed through an InputHandler and executed by the implementation
       *  through the public Camera API. Same can be implemented for other SceniX core objects.
       *  The SceniX SDK already comes with a set of concrete CameraManipulator objects, like
       *  the TrackballCameraManipulator, the WalkCameraManipulator, and the FlightCameraManipulator.
       *  These objects can be instantiated and configured for your application right away.
       *  Besides the mentioned CameraManipulators, SceniX also provides a TrackballTransformManipulator
       *  as a concrete implementation of a TrackballTransformManipulator.
       *  \n\n
       *  Manipulator provides an API to assign RenderTarget and ViewState, which concrete implementations require 
       *  to retrieve necessary information to accurately update the controlled objects, like cameras and transforms. */
      class Manipulator
      {
      public:
        DP_SG_UI_API Manipulator();
        DP_SG_UI_API virtual ~Manipulator();

        /*! \brief Sets this manipulator's ViewState.
         *  \param viewState The ViewState this manipulator should use.
         *  \sa getViewState */
        DP_SG_UI_API virtual void setViewState( dp::sg::ui::ViewStateSharedPtr const& viewState);

        /*! \brief Gets this manipulator's ViewState.
         *  \return The ViewState this manipulator is using.
         *  \sa setViewState */
        DP_SG_UI_API virtual dp::sg::ui::ViewStateSharedPtr const& getViewState() const;

        /*! \brief Sets this manipulator's RenderTarget.
         *  \param renderTarget The RenderTarget this manipulator should use.
         *  \sa getRenderTarget */
        DP_SG_UI_API virtual void setRenderTarget( const dp::ui::RenderTargetSharedPtr &renderTarget );

        /*! \brief Gets this manipulator's RenderTarget.
         *  \return The RenderTarget this manipulator is using.
         *  \sa setRenderTarget */
        DP_SG_UI_API virtual dp::ui::RenderTargetSharedPtr getRenderTarget() const;

        /*! \brief Updates the manipulator's timestamp, and runs the manipulator.
         *  \param dt Delta time passage since this function was called last, in seconds.
         *  \remarks Some manipulators (walk, flight, animation) require this method to be called continuously 
         *  when they are active, to update the simulation.
         */
        DP_SG_UI_API virtual bool updateFrame( float dt ) = 0;

        /*! \brief Resets the manipulator to defaults.
         * \remarks Resets the manipulator to initial state.
         */
        virtual void reset();

      protected:
        dp::ui::RenderTargetSharedPtr   m_renderTarget;
        dp::sg::ui::ViewStateSharedPtr  m_viewState;
      };

      /*! \brief Internal helper class used to manage the "cursor" state.
       *  \remarks Manipulators that use the mouse cursor will use this class to manage its state.
       */
      class CursorState
      {
        public:
          CursorState();
          ~CursorState();

          void setCursorPosition( const dp::math::Vec2i &cursor );
          const dp::math::Vec2i & getCursorPosition() const;
          const dp::math::Vec2i & getLastCursorPosition() const;

          void setWheelTicks( int wheelTicks );
          int  getWheelTicks() const;
          int  getLastWheelTicks() const;
          int  getWheelTicksDelta() const;

          int getLastX() const;
          int getLastY() const;
          int getCurrentX() const;
          int getCurrentY() const;

          void resetInput();

        private:
          enum class InitialUpdate
          {
            WHEEL    = 0x01,
            POSITION = 0x02
          };

          typedef dp::util::Flags<InitialUpdate> InitialUpdateMask;

          dp::math::Vec2i     m_cursorPosition;
          dp::math::Vec2i     m_lastCursorPosition;
          int                 m_wheelTicks;
          int                 m_lastWheelTicks;
          InitialUpdateMask  m_initialUpdate;
      };

      inline void CursorState::resetInput()
      {
        m_cursorPosition     = dp::math::Vec2i(0,0);
        m_lastCursorPosition = dp::math::Vec2i(0,0);
        m_wheelTicks         = 0;
        m_lastWheelTicks     = 0;
        m_initialUpdate      = { InitialUpdate::WHEEL, InitialUpdate::POSITION };
      }

      inline CursorState::CursorState()
      {
        resetInput();
      }

      inline CursorState::~CursorState()
      {
      }

      inline int CursorState::getCurrentX() const
      {
        return getCursorPosition()[0];
      }

      inline int CursorState::getCurrentY() const
      {
        return getCursorPosition()[1];
      }

      inline int CursorState::getLastX() const
      {
        return getLastCursorPosition()[0];
      }

      inline int CursorState::getLastY() const
      {
        return getLastCursorPosition()[1];
      }

      inline void CursorState::setCursorPosition( const dp::math::Vec2i &cursor )
      {
        if( m_initialUpdate & InitialUpdate::POSITION )
        {
          m_lastCursorPosition = cursor;
          m_initialUpdate ^= InitialUpdate::POSITION;
        }
        else
        {
          m_lastCursorPosition = m_cursorPosition;
        }

        m_cursorPosition = cursor;
      }

      inline const dp::math::Vec2i & CursorState::getCursorPosition() const
      {
        return m_cursorPosition;
      }

      inline const dp::math::Vec2i & CursorState::getLastCursorPosition() const
      {
        return m_lastCursorPosition;
      }

      inline void CursorState::setWheelTicks( int wheelTicks )
      {
        if( m_initialUpdate & InitialUpdate::WHEEL )
        {
          m_lastWheelTicks = wheelTicks;
          m_initialUpdate ^= InitialUpdate::WHEEL;
        }
        else
        {
          m_lastWheelTicks = m_wheelTicks;
        }

        m_wheelTicks = wheelTicks;
      }

      inline int  CursorState::getWheelTicks() const
      {
        return m_wheelTicks;
      }

      inline int  CursorState::getLastWheelTicks() const
      {
        return m_lastWheelTicks;
      }

      inline int  CursorState::getWheelTicksDelta() const
      {
        return getWheelTicks() - getLastWheelTicks();
      }

    } // namespace ui
  } // namespace sg
} // namespace dp
