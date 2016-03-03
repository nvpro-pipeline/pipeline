// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/ui/glut/Config.h>
#include <dp/sg/ui/SceniXWidget.h>
#include <dp/gl/RenderTarget.h>
#include <dp/util/Timer.h>
#include <boost/utility.hpp>
#include <dp/sg/ui/HumanInterfaceDevice.h>

#pragma once

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace glut
      {
        class Widget : public HumanInterfaceDevice
        {
        public:
          DP_SG_UI_GLUT_API Widget( int x = -1, int y = -1 );
          DP_SG_UI_GLUT_API virtual ~Widget();

          DP_SG_UI_GLUT_API dp::gl::RenderContextSharedPtr getRenderContext() const;
          DP_SG_UI_GLUT_API const dp::gl::RenderTargetSharedPtr  & getRenderTarget() const;

          /** \brief Enforce a repaint of the viewport **/
          DP_SG_UI_GLUT_API virtual void doTriggerRepaint();

          /** \brief Set the title of the GLUT window
              \param title The new window title.
          **/
          DP_SG_UI_GLUT_API virtual void setWindowTitle( std::string const& title );

          /** \brief Resize the window to the given size
              \param width Width of the window
              \param height Height of the window
          **/
          DP_SG_UI_GLUT_API virtual void setWindowSize( size_t width, size_t height );

          DP_SG_UI_GLUT_API virtual void setWindowFullScreen();

          /** \brief Update whether the current framerate should be displayed in the window title.
              \param showFPS framerate will be displayed in the window title if showFPS is true.
          **/
          DP_SG_UI_GLUT_API virtual void setShowFrameRate( bool showFPS );

          /** \brief Continuous update repaints the viewport as fast as possible
              \param continous Continuous update will be enabled if continous is true
          **/
          DP_SG_UI_GLUT_API virtual void setContinuousUpdate( bool continuous );

        protected:
          DP_SG_UI_GLUT_API virtual void paint();
          DP_SG_UI_GLUT_API virtual void resize( GLint width, GLint height );
          DP_SG_UI_GLUT_API virtual void cleanup();
          DP_SG_UI_GLUT_API virtual void mouseButtonChanged( int button, int state, int x, int y );
          DP_SG_UI_GLUT_API virtual void mousePositionChanged( int x, int y );
          DP_SG_UI_GLUT_API virtual void mouseWheelChanged( int wheel, int direction, int x, int y );
          DP_SG_UI_GLUT_API virtual void keyPressed( unsigned char key, int x, int y );
          DP_SG_UI_GLUT_API virtual void keyReleased( unsigned char key, int x, int y );
          DP_SG_UI_GLUT_API virtual void specialKeyPressed( int key, int x, int y );
          DP_SG_UI_GLUT_API virtual void specialKeyReleased( int key, int x, int y );

          DP_SG_UI_GLUT_API virtual void updateWindowTitle( std::string const& title );

          DP_SG_UI_GLUT_API virtual void onManipulatorChanged( Manipulator *manipulator );
          DP_SG_UI_GLUT_API virtual void hidNotify( dp::util::PropertyId property );

          /** \brief Override to fetch HID events **/
          DP_SG_UI_GLUT_API virtual void onHIDEvent( dp::util::PropertyId propertyId );

        private:
#if !defined(HAVE_FREEGLUT_2_8)
          void updateModifiers();
#endif
        private:

          int m_windowId;
          size_t m_width;
          size_t m_height;

          // FPS
          bool m_showFPS;
          double m_fps;
          unsigned int m_frameCount;
          dp::util::Timer m_frameRateTimer;
          dp::util::Timer m_updateTimer;

          bool m_continuousUpdate;

          std::string m_windowTitle;
          std::string m_windowTitleBase;
          dp::gl::RenderTargetSharedPtr  m_renderTarget;

          /************************************************************************/
          /* static delegation functions                                          */
          /************************************************************************/
          static Widget* getCurrentWidget();
          static void renderFunction();
          static void reshapeFunction( GLint width, GLint height );
          static void closeFunction();
          static void mouseButtonFunction( int button, int state, int x, int y );
          static void mouseMotionFunction( int x, int y );
          static void mouseWheelFunction( int wheel, int direction, int x, int y );
          static void keyboardFunction( unsigned char key, int x, int y );
          static void keyboardUpFunction( unsigned char key, int x, int y );
          static void keyboardSpecialFunction( int key, int x, int y );
          static void keyboardSpecialUpFunction( int key, int x, int y );

          /************************************************************************/
          /* HumanInterfaceDevice                                                 */
          /************************************************************************/
        public:
          DP_SG_UI_GLUT_API virtual unsigned int getNumberOfAxes() const;
          DP_SG_UI_GLUT_API virtual std::string getAxisName( unsigned int axis ) const;
          DP_SG_UI_GLUT_API virtual unsigned int getNumberOfKeys() const;
          DP_SG_UI_GLUT_API virtual std::string getKeyName( unsigned int key ) const;

          // HID
          REFLECTION_INFO_API( DP_SG_UI_GLUT_API, Widget );

          BEGIN_DECLARE_STATIC_PROPERTIES
            // mouse
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Mouse_Left );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Mouse_Middle );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Mouse_Right );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Mouse_Position );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Mouse_Wheel );

            // keyboard
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Escape );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Tab );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Backtab );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Backspace );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Return );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Enter );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Insert );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Delete );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Pause );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Print );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_SysReq );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Clear );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Home );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_End );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Left );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Up );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Right );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Down );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_PageUp );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_PageDown );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Shift );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Control );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Meta );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Alt );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_AltGr );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_CapsLock );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_NumLock );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_ScrollLock );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F1 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F2 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F3 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F4 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F5 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F6 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F7 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F8 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F9 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F10 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F11 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F12 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Super_L );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Super_R );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Menu );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hyper_L );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hyper_R );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Help );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Direction_L );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Direction_R );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Space );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Any );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Exclam );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_QuoteDbl );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_NumberSign );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dollar );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Percent );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ampersand );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Apostrophe );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_ParenLeft );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_ParenRight );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Asterisk );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Plus );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Comma );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Minus );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Period );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Slash );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_0 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_1 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_2 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_3 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_4 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_5 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_6 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_7 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_8 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_9 );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Colon );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Semicolon );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Less );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Equal );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Greater );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Question );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_At );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_A );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_B );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_C );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_D );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_E );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_F );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_G );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_H );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_I );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_J );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_K );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_L );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_M );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_N );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_O );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_P );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Q );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_R );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_S );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_T );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_U );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_V );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_W );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_X );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Y );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Z );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_BracketLeft );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Backslash );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_BracketRight );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_AsciiCircum );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Underscore );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_QuoteLeft );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_BraceLeft );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Bar );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_BraceRight );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_AsciiTilde );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_nobreakspace );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_exclamdown );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_cent );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_sterling );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_currency );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_yen );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_brokenbar );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_section );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_diaeresis );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_copyright );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_ordfeminine );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_guillemotleft );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_notsign );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_hyphen );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_registered );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_macron );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_degree );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_plusminus );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_twosuperior );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_threesuperior );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_acute );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_mu );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_paragraph );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_periodcentered );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_cedilla );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_onesuperior );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_masculine );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_guillemotright );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_onequarter );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_onehalf );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_threequarters );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_questiondown );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Agrave );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Aacute );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Acircumflex );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Atilde );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Adiaeresis );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Aring );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_AE );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ccedilla );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Egrave );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Eacute );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ecircumflex );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ediaeresis );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Igrave );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Iacute );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Icircumflex );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Idiaeresis );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_ETH );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ntilde );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ograve );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Oacute );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ocircumflex );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Otilde );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Odiaeresis );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_multiply );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ooblique );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ugrave );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Uacute );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Ucircumflex );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Udiaeresis );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Yacute );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_THORN );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_ssharp );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_division );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_ydiaeresis );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Multi_key );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Codeinput );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_SingleCandidate );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_MultipleCandidate );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_PreviousCandidate );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Mode_switch );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Kanji );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Muhenkan );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Henkan );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Romaji );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hiragana );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Katakana );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hiragana_Katakana );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Zenkaku );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hankaku );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Zenkaku_Hankaku );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Touroku );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Massyo );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Kana_Lock );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Kana_Shift );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Eisu_Shift );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Eisu_toggle );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_Start );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_End );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_Hanja );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_Jamo );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_Romaja );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_Jeonja );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_Banja );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_PreHanja );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_PostHanja );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Hangul_Special );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Grave );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Acute );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Circumflex );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Tilde );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Macron );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Breve );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Abovedot );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Diaeresis );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Abovering );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Doubleacute );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Caron );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Cedilla );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Ogonek );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Iota );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Voiced_Sound );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Semivoiced_Sound );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Belowdot );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Hook );
            DP_SG_UI_GLUT_API DECLARE_STATIC_PROPERTY( Key_Dead_Horn );

          END_DECLARE_STATIC_PROPERTIES
        private:
          dp::util::PropertyId m_asciiMap[256];
          typedef std::map<int, dp::util::PropertyId> SpecialPropertyMap;
          SpecialPropertyMap m_specialPropertyMap;

          // mouse members
          bool m_propMouse_Left;
          bool m_propMouse_Middle;
          bool m_propMouse_Right;
          dp::math::Vec2i m_propMouse_Position;
          int m_propMouse_Wheel;

          // keyboard members
          bool m_propKey_Escape;
          bool m_propKey_Tab;
          bool m_propKey_Backtab;
          bool m_propKey_Backspace;
          bool m_propKey_Return;
          bool m_propKey_Enter;
          bool m_propKey_Insert;
          bool m_propKey_Delete;
          bool m_propKey_Pause;
          bool m_propKey_Print;
          bool m_propKey_SysReq;
          bool m_propKey_Clear;
          bool m_propKey_Home;
          bool m_propKey_End;
          bool m_propKey_Left;
          bool m_propKey_Up;
          bool m_propKey_Right;
          bool m_propKey_Down;
          bool m_propKey_PageUp;
          bool m_propKey_PageDown;
          bool m_propKey_Shift;
          bool m_propKey_Control;
          bool m_propKey_Meta;
          bool m_propKey_Alt;
          bool m_propKey_AltGr;
          bool m_propKey_CapsLock;
          bool m_propKey_NumLock;
          bool m_propKey_ScrollLock;
          bool m_propKey_F1;
          bool m_propKey_F2;
          bool m_propKey_F3;
          bool m_propKey_F4;
          bool m_propKey_F5;
          bool m_propKey_F6;
          bool m_propKey_F7;
          bool m_propKey_F8;
          bool m_propKey_F9;
          bool m_propKey_F10;
          bool m_propKey_F11;
          bool m_propKey_F12;
          bool m_propKey_F13;
          bool m_propKey_F14;
          bool m_propKey_F15;
          bool m_propKey_F16;
          bool m_propKey_F17;
          bool m_propKey_F18;
          bool m_propKey_F19;
          bool m_propKey_F20;
          bool m_propKey_F21;
          bool m_propKey_F22;
          bool m_propKey_F23;
          bool m_propKey_F24;
          bool m_propKey_F25;
          bool m_propKey_F26;
          bool m_propKey_F27;
          bool m_propKey_F28;
          bool m_propKey_F29;
          bool m_propKey_F30;
          bool m_propKey_F31;
          bool m_propKey_F32;
          bool m_propKey_F33;
          bool m_propKey_F34;
          bool m_propKey_F35;
          bool m_propKey_Super_L;
          bool m_propKey_Super_R;
          bool m_propKey_Menu;
          bool m_propKey_Hyper_L;
          bool m_propKey_Hyper_R;
          bool m_propKey_Help;
          bool m_propKey_Direction_L;
          bool m_propKey_Direction_R;
          bool m_propKey_Space;
          bool m_propKey_Any;
          bool m_propKey_Exclam;
          bool m_propKey_QuoteDbl;
          bool m_propKey_NumberSign;
          bool m_propKey_Dollar;
          bool m_propKey_Percent;
          bool m_propKey_Ampersand;
          bool m_propKey_Apostrophe;
          bool m_propKey_ParenLeft;
          bool m_propKey_ParenRight;
          bool m_propKey_Asterisk;
          bool m_propKey_Plus;
          bool m_propKey_Comma;
          bool m_propKey_Minus;
          bool m_propKey_Period;
          bool m_propKey_Slash;
          bool m_propKey_0;
          bool m_propKey_1;
          bool m_propKey_2;
          bool m_propKey_3;
          bool m_propKey_4;
          bool m_propKey_5;
          bool m_propKey_6;
          bool m_propKey_7;
          bool m_propKey_8;
          bool m_propKey_9;
          bool m_propKey_Colon;
          bool m_propKey_Semicolon;
          bool m_propKey_Less;
          bool m_propKey_Equal;
          bool m_propKey_Greater;
          bool m_propKey_Question;
          bool m_propKey_At;
          bool m_propKey_A;
          bool m_propKey_B;
          bool m_propKey_C;
          bool m_propKey_D;
          bool m_propKey_E;
          bool m_propKey_F;
          bool m_propKey_G;
          bool m_propKey_H;
          bool m_propKey_I;
          bool m_propKey_J;
          bool m_propKey_K;
          bool m_propKey_L;
          bool m_propKey_M;
          bool m_propKey_N;
          bool m_propKey_O;
          bool m_propKey_P;
          bool m_propKey_Q;
          bool m_propKey_R;
          bool m_propKey_S;
          bool m_propKey_T;
          bool m_propKey_U;
          bool m_propKey_V;
          bool m_propKey_W;
          bool m_propKey_X;
          bool m_propKey_Y;
          bool m_propKey_Z;
          bool m_propKey_BracketLeft;
          bool m_propKey_Backslash;
          bool m_propKey_BracketRight;
          bool m_propKey_AsciiCircum;
          bool m_propKey_Underscore;
          bool m_propKey_QuoteLeft;
          bool m_propKey_BraceLeft;
          bool m_propKey_Bar;
          bool m_propKey_BraceRight;
          bool m_propKey_AsciiTilde;
          bool m_propKey_nobreakspace;
          bool m_propKey_exclamdown;
          bool m_propKey_cent;
          bool m_propKey_sterling;
          bool m_propKey_currency;
          bool m_propKey_yen;
          bool m_propKey_brokenbar;
          bool m_propKey_section;
          bool m_propKey_diaeresis;
          bool m_propKey_copyright;
          bool m_propKey_ordfeminine;
          bool m_propKey_guillemotleft;
          bool m_propKey_notsign;
          bool m_propKey_hyphen;
          bool m_propKey_registered;
          bool m_propKey_macron;
          bool m_propKey_degree;
          bool m_propKey_plusminus;
          bool m_propKey_twosuperior;
          bool m_propKey_threesuperior;
          bool m_propKey_acute;
          bool m_propKey_mu;
          bool m_propKey_paragraph;
          bool m_propKey_periodcentered;
          bool m_propKey_cedilla;
          bool m_propKey_onesuperior;
          bool m_propKey_masculine;
          bool m_propKey_guillemotright;
          bool m_propKey_onequarter;
          bool m_propKey_onehalf;
          bool m_propKey_threequarters;
          bool m_propKey_questiondown;
          bool m_propKey_Agrave;
          bool m_propKey_Aacute;
          bool m_propKey_Acircumflex;
          bool m_propKey_Atilde;
          bool m_propKey_Adiaeresis;
          bool m_propKey_Aring;
          bool m_propKey_AE;
          bool m_propKey_Ccedilla;
          bool m_propKey_Egrave;
          bool m_propKey_Eacute;
          bool m_propKey_Ecircumflex;
          bool m_propKey_Ediaeresis;
          bool m_propKey_Igrave;
          bool m_propKey_Iacute;
          bool m_propKey_Icircumflex;
          bool m_propKey_Idiaeresis;
          bool m_propKey_ETH;
          bool m_propKey_Ntilde;
          bool m_propKey_Ograve;
          bool m_propKey_Oacute;
          bool m_propKey_Ocircumflex;
          bool m_propKey_Otilde;
          bool m_propKey_Odiaeresis;
          bool m_propKey_multiply;
          bool m_propKey_Ooblique;
          bool m_propKey_Ugrave;
          bool m_propKey_Uacute;
          bool m_propKey_Ucircumflex;
          bool m_propKey_Udiaeresis;
          bool m_propKey_Yacute;
          bool m_propKey_THORN;
          bool m_propKey_ssharp;
          bool m_propKey_division;
          bool m_propKey_ydiaeresis;
          bool m_propKey_Multi_key;
          bool m_propKey_Codeinput;
          bool m_propKey_SingleCandidate;
          bool m_propKey_MultipleCandidate;
          bool m_propKey_PreviousCandidate;
          bool m_propKey_Mode_switch;
          bool m_propKey_Kanji;
          bool m_propKey_Muhenkan;
          bool m_propKey_Henkan;
          bool m_propKey_Romaji;
          bool m_propKey_Hiragana;
          bool m_propKey_Katakana;
          bool m_propKey_Hiragana_Katakana;
          bool m_propKey_Zenkaku;
          bool m_propKey_Hankaku;
          bool m_propKey_Zenkaku_Hankaku;
          bool m_propKey_Touroku;
          bool m_propKey_Massyo;
          bool m_propKey_Kana_Lock;
          bool m_propKey_Kana_Shift;
          bool m_propKey_Eisu_Shift;
          bool m_propKey_Eisu_toggle;
          bool m_propKey_Hangul;
          bool m_propKey_Hangul_Start;
          bool m_propKey_Hangul_End;
          bool m_propKey_Hangul_Hanja;
          bool m_propKey_Hangul_Jamo;
          bool m_propKey_Hangul_Romaja;
          bool m_propKey_Hangul_Jeonja;
          bool m_propKey_Hangul_Banja;
          bool m_propKey_Hangul_PreHanja;
          bool m_propKey_Hangul_PostHanja;
          bool m_propKey_Hangul_Special;
          bool m_propKey_Dead_Grave;
          bool m_propKey_Dead_Acute;
          bool m_propKey_Dead_Circumflex;
          bool m_propKey_Dead_Tilde;
          bool m_propKey_Dead_Macron;
          bool m_propKey_Dead_Breve;
          bool m_propKey_Dead_Abovedot;
          bool m_propKey_Dead_Diaeresis;
          bool m_propKey_Dead_Abovering;
          bool m_propKey_Dead_Doubleacute;
          bool m_propKey_Dead_Caron;
          bool m_propKey_Dead_Cedilla;
          bool m_propKey_Dead_Ogonek;
          bool m_propKey_Dead_Iota;
          bool m_propKey_Dead_Voiced_Sound;
          bool m_propKey_Dead_Semivoiced_Sound;
          bool m_propKey_Dead_Belowdot;
          bool m_propKey_Dead_Hook;
          bool m_propKey_Dead_Horn;
        };

      }
    }
  }
}