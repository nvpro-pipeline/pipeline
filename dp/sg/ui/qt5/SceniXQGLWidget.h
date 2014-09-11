// Copyright NVIDIA Corporation 2009-2013
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

#include <QWidget>
#include <QExposeEvent>
#include <dp/util/SmartPtr.h>

#include <dp/sg/ui/qt5/Config.h>
#include <dp/sg/ui/SceneRenderer.h>
#include <dp/sg/ui/SceniXWidget.h>

#include <dp/sg/ui/HumanInterfaceDevice.h>

namespace dp
{
  namespace gl
  {
    class RenderTarget;
    typedef dp::util::SmartPtr<RenderTarget> SmartRenderTarget;

    class RenderContext;
    typedef dp::util::SmartPtr<RenderContext> SmartRenderContext;

    class RenderContextFormat;
  }
}

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace qt5
      {

        class SceniXQGLWidget : public QWidget, public HumanInterfaceDevice
        {
        public:
          DP_SG_UI_QT5_API SceniXQGLWidget(QWidget *parent, dp::gl::RenderContextFormat const & format, SceniXQGLWidget *shareWidget = 0);
          DP_SG_UI_QT5_API virtual ~SceniXQGLWidget();

          using QWidget::update;

          DP_SG_UI_QT5_API virtual void resizeEvent( QResizeEvent *event );

          DP_SG_UI_QT5_API const dp::gl::SmartRenderContext & getRenderContext() const;
          DP_SG_UI_QT5_API const dp::gl::SmartRenderTarget & getRenderTarget() const;

          DP_SG_UI_QT5_API bool event( QEvent *event );

          DP_SG_UI_QT5_API bool setFormat( const dp::gl::RenderContextFormat &format );
          DP_SG_UI_QT5_API dp::gl::RenderContextFormat getFormat() const;

          DP_SG_UI_QT5_API QPaintEngine *paintEngine() const;

          DP_SG_UI_QT5_API virtual void keyPressEvent( QKeyEvent *event );
          DP_SG_UI_QT5_API virtual void keyReleaseEvent( QKeyEvent *event );

          DP_SG_UI_QT5_API virtual void mousePressEvent( QMouseEvent *event );
          DP_SG_UI_QT5_API virtual void mouseReleaseEvent( QMouseEvent *event );
          DP_SG_UI_QT5_API virtual void mouseMoveEvent( QMouseEvent *event );
          DP_SG_UI_QT5_API virtual void wheelEvent( QWheelEvent *event );

          DP_SG_UI_QT5_API virtual unsigned int getNumberOfAxes() const;
          DP_SG_UI_QT5_API virtual std::string getAxisName( unsigned int axis ) const;
          DP_SG_UI_QT5_API virtual unsigned int getNumberOfKeys() const;
          DP_SG_UI_QT5_API virtual std::string getKeyName( unsigned int key ) const;

          DP_SG_UI_QT5_API virtual void triggerRepaint();

        protected:
          /** \brief This function is being called after setting a new format. At this time the new RenderTarget and the old RenderTarget
                     is available. All references to the old RenderTarget need to be released here.
              \param oldTarget The RenderTarget for the old format.
              \param newTarget The RenderTarget for the new format.
          **/
          DP_SG_UI_QT5_API virtual void onRenderTargetChanged( const dp::gl::SmartRenderTarget &oldTarget, const dp::gl::SmartRenderTarget &newTarget );

          DP_SG_UI_QT5_API virtual void initializeGL();
          DP_SG_UI_QT5_API virtual void resizeGL( int width, int height );
          DP_SG_UI_QT5_API virtual void paintGL();

        public:
          // HID
          REFLECTION_INFO_API( DP_SG_UI_QT5_API, SceniXQGLWidget );

          BEGIN_DECLARE_STATIC_PROPERTIES
            // mouse
            DECLARE_STATIC_PROPERTY( Mouse_Left );
            DECLARE_STATIC_PROPERTY( Mouse_Middle );
            DECLARE_STATIC_PROPERTY( Mouse_Right );
            DECLARE_STATIC_PROPERTY( Mouse_Position );
            DECLARE_STATIC_PROPERTY( Mouse_Wheel );

            // keyboard
            DECLARE_STATIC_PROPERTY( Key_Escape );
            DECLARE_STATIC_PROPERTY( Key_Tab );
            DECLARE_STATIC_PROPERTY( Key_Backtab );
            DECLARE_STATIC_PROPERTY( Key_Backspace );
            DECLARE_STATIC_PROPERTY( Key_Return );
            DECLARE_STATIC_PROPERTY( Key_Enter );
            DECLARE_STATIC_PROPERTY( Key_Insert );
            DECLARE_STATIC_PROPERTY( Key_Delete );
            DECLARE_STATIC_PROPERTY( Key_Pause );
            DECLARE_STATIC_PROPERTY( Key_Print );
            DECLARE_STATIC_PROPERTY( Key_SysReq );
            DECLARE_STATIC_PROPERTY( Key_Clear );
            DECLARE_STATIC_PROPERTY( Key_Home );
            DECLARE_STATIC_PROPERTY( Key_End );
            DECLARE_STATIC_PROPERTY( Key_Left );
            DECLARE_STATIC_PROPERTY( Key_Up );
            DECLARE_STATIC_PROPERTY( Key_Right );
            DECLARE_STATIC_PROPERTY( Key_Down );
            DECLARE_STATIC_PROPERTY( Key_PageUp );
            DECLARE_STATIC_PROPERTY( Key_PageDown );
            DECLARE_STATIC_PROPERTY( Key_Shift );
            DECLARE_STATIC_PROPERTY( Key_Control );
            DECLARE_STATIC_PROPERTY( Key_Meta );
            DECLARE_STATIC_PROPERTY( Key_Alt );
            DECLARE_STATIC_PROPERTY( Key_AltGr );
            DECLARE_STATIC_PROPERTY( Key_CapsLock );
            DECLARE_STATIC_PROPERTY( Key_NumLock );
            DECLARE_STATIC_PROPERTY( Key_ScrollLock );
            DECLARE_STATIC_PROPERTY( Key_F1 );
            DECLARE_STATIC_PROPERTY( Key_F2 );
            DECLARE_STATIC_PROPERTY( Key_F3 );
            DECLARE_STATIC_PROPERTY( Key_F4 );
            DECLARE_STATIC_PROPERTY( Key_F5 );
            DECLARE_STATIC_PROPERTY( Key_F6 );
            DECLARE_STATIC_PROPERTY( Key_F7 );
            DECLARE_STATIC_PROPERTY( Key_F8 );
            DECLARE_STATIC_PROPERTY( Key_F9 );
            DECLARE_STATIC_PROPERTY( Key_F10 );
            DECLARE_STATIC_PROPERTY( Key_F11 );
            DECLARE_STATIC_PROPERTY( Key_F12 );
            DECLARE_STATIC_PROPERTY( Key_F13 );
            DECLARE_STATIC_PROPERTY( Key_F14 );
            DECLARE_STATIC_PROPERTY( Key_F15 );
            DECLARE_STATIC_PROPERTY( Key_F16 );
            DECLARE_STATIC_PROPERTY( Key_F17 );
            DECLARE_STATIC_PROPERTY( Key_F18 );
            DECLARE_STATIC_PROPERTY( Key_F19 );
            DECLARE_STATIC_PROPERTY( Key_F20 );
            DECLARE_STATIC_PROPERTY( Key_F21 );
            DECLARE_STATIC_PROPERTY( Key_F22 );
            DECLARE_STATIC_PROPERTY( Key_F23 );
            DECLARE_STATIC_PROPERTY( Key_F24 );
            DECLARE_STATIC_PROPERTY( Key_F25 );
            DECLARE_STATIC_PROPERTY( Key_F26 );
            DECLARE_STATIC_PROPERTY( Key_F27 );
            DECLARE_STATIC_PROPERTY( Key_F28 );
            DECLARE_STATIC_PROPERTY( Key_F29 );
            DECLARE_STATIC_PROPERTY( Key_F30 );
            DECLARE_STATIC_PROPERTY( Key_F31 );
            DECLARE_STATIC_PROPERTY( Key_F32 );
            DECLARE_STATIC_PROPERTY( Key_F33 );
            DECLARE_STATIC_PROPERTY( Key_F34 );
            DECLARE_STATIC_PROPERTY( Key_F35 );
            DECLARE_STATIC_PROPERTY( Key_Super_L );
            DECLARE_STATIC_PROPERTY( Key_Super_R );
            DECLARE_STATIC_PROPERTY( Key_Menu );
            DECLARE_STATIC_PROPERTY( Key_Hyper_L );
            DECLARE_STATIC_PROPERTY( Key_Hyper_R );
            DECLARE_STATIC_PROPERTY( Key_Help );
            DECLARE_STATIC_PROPERTY( Key_Direction_L );
            DECLARE_STATIC_PROPERTY( Key_Direction_R );
            DECLARE_STATIC_PROPERTY( Key_Space );
            DECLARE_STATIC_PROPERTY( Key_Any );
            DECLARE_STATIC_PROPERTY( Key_Exclam );
            DECLARE_STATIC_PROPERTY( Key_QuoteDbl );
            DECLARE_STATIC_PROPERTY( Key_NumberSign );
            DECLARE_STATIC_PROPERTY( Key_Dollar );
            DECLARE_STATIC_PROPERTY( Key_Percent );
            DECLARE_STATIC_PROPERTY( Key_Ampersand );
            DECLARE_STATIC_PROPERTY( Key_Apostrophe );
            DECLARE_STATIC_PROPERTY( Key_ParenLeft );
            DECLARE_STATIC_PROPERTY( Key_ParenRight );
            DECLARE_STATIC_PROPERTY( Key_Asterisk );
            DECLARE_STATIC_PROPERTY( Key_Plus );
            DECLARE_STATIC_PROPERTY( Key_Comma );
            DECLARE_STATIC_PROPERTY( Key_Minus );
            DECLARE_STATIC_PROPERTY( Key_Period );
            DECLARE_STATIC_PROPERTY( Key_Slash );
            DECLARE_STATIC_PROPERTY( Key_0 );
            DECLARE_STATIC_PROPERTY( Key_1 );
            DECLARE_STATIC_PROPERTY( Key_2 );
            DECLARE_STATIC_PROPERTY( Key_3 );
            DECLARE_STATIC_PROPERTY( Key_4 );
            DECLARE_STATIC_PROPERTY( Key_5 );
            DECLARE_STATIC_PROPERTY( Key_6 );
            DECLARE_STATIC_PROPERTY( Key_7 );
            DECLARE_STATIC_PROPERTY( Key_8 );
            DECLARE_STATIC_PROPERTY( Key_9 );
            DECLARE_STATIC_PROPERTY( Key_Colon );
            DECLARE_STATIC_PROPERTY( Key_Semicolon );
            DECLARE_STATIC_PROPERTY( Key_Less );
            DECLARE_STATIC_PROPERTY( Key_Equal );
            DECLARE_STATIC_PROPERTY( Key_Greater );
            DECLARE_STATIC_PROPERTY( Key_Question );
            DECLARE_STATIC_PROPERTY( Key_At );
            DECLARE_STATIC_PROPERTY( Key_A );
            DECLARE_STATIC_PROPERTY( Key_B );
            DECLARE_STATIC_PROPERTY( Key_C );
            DECLARE_STATIC_PROPERTY( Key_D );
            DECLARE_STATIC_PROPERTY( Key_E );
            DECLARE_STATIC_PROPERTY( Key_F );
            DECLARE_STATIC_PROPERTY( Key_G );
            DECLARE_STATIC_PROPERTY( Key_H );
            DECLARE_STATIC_PROPERTY( Key_I );
            DECLARE_STATIC_PROPERTY( Key_J );
            DECLARE_STATIC_PROPERTY( Key_K );
            DECLARE_STATIC_PROPERTY( Key_L );
            DECLARE_STATIC_PROPERTY( Key_M );
            DECLARE_STATIC_PROPERTY( Key_N );
            DECLARE_STATIC_PROPERTY( Key_O );
            DECLARE_STATIC_PROPERTY( Key_P );
            DECLARE_STATIC_PROPERTY( Key_Q );
            DECLARE_STATIC_PROPERTY( Key_R );
            DECLARE_STATIC_PROPERTY( Key_S );
            DECLARE_STATIC_PROPERTY( Key_T );
            DECLARE_STATIC_PROPERTY( Key_U );
            DECLARE_STATIC_PROPERTY( Key_V );
            DECLARE_STATIC_PROPERTY( Key_W );
            DECLARE_STATIC_PROPERTY( Key_X );
            DECLARE_STATIC_PROPERTY( Key_Y );
            DECLARE_STATIC_PROPERTY( Key_Z );
            DECLARE_STATIC_PROPERTY( Key_BracketLeft );
            DECLARE_STATIC_PROPERTY( Key_Backslash );
            DECLARE_STATIC_PROPERTY( Key_BracketRight );
            DECLARE_STATIC_PROPERTY( Key_AsciiCircum );
            DECLARE_STATIC_PROPERTY( Key_Underscore );
            DECLARE_STATIC_PROPERTY( Key_QuoteLeft );
            DECLARE_STATIC_PROPERTY( Key_BraceLeft );
            DECLARE_STATIC_PROPERTY( Key_Bar );
            DECLARE_STATIC_PROPERTY( Key_BraceRight );
            DECLARE_STATIC_PROPERTY( Key_AsciiTilde );
            DECLARE_STATIC_PROPERTY( Key_nobreakspace );
            DECLARE_STATIC_PROPERTY( Key_exclamdown );
            DECLARE_STATIC_PROPERTY( Key_cent );
            DECLARE_STATIC_PROPERTY( Key_sterling );
            DECLARE_STATIC_PROPERTY( Key_currency );
            DECLARE_STATIC_PROPERTY( Key_yen );
            DECLARE_STATIC_PROPERTY( Key_brokenbar );
            DECLARE_STATIC_PROPERTY( Key_section );
            DECLARE_STATIC_PROPERTY( Key_diaeresis );
            DECLARE_STATIC_PROPERTY( Key_copyright );
            DECLARE_STATIC_PROPERTY( Key_ordfeminine );
            DECLARE_STATIC_PROPERTY( Key_guillemotleft );
            DECLARE_STATIC_PROPERTY( Key_notsign );
            DECLARE_STATIC_PROPERTY( Key_hyphen );
            DECLARE_STATIC_PROPERTY( Key_registered );
            DECLARE_STATIC_PROPERTY( Key_macron );
            DECLARE_STATIC_PROPERTY( Key_degree );
            DECLARE_STATIC_PROPERTY( Key_plusminus );
            DECLARE_STATIC_PROPERTY( Key_twosuperior );
            DECLARE_STATIC_PROPERTY( Key_threesuperior );
            DECLARE_STATIC_PROPERTY( Key_acute );
            DECLARE_STATIC_PROPERTY( Key_mu );
            DECLARE_STATIC_PROPERTY( Key_paragraph );
            DECLARE_STATIC_PROPERTY( Key_periodcentered );
            DECLARE_STATIC_PROPERTY( Key_cedilla );
            DECLARE_STATIC_PROPERTY( Key_onesuperior );
            DECLARE_STATIC_PROPERTY( Key_masculine );
            DECLARE_STATIC_PROPERTY( Key_guillemotright );
            DECLARE_STATIC_PROPERTY( Key_onequarter );
            DECLARE_STATIC_PROPERTY( Key_onehalf );
            DECLARE_STATIC_PROPERTY( Key_threequarters );
            DECLARE_STATIC_PROPERTY( Key_questiondown );
            DECLARE_STATIC_PROPERTY( Key_Agrave );
            DECLARE_STATIC_PROPERTY( Key_Aacute );
            DECLARE_STATIC_PROPERTY( Key_Acircumflex );
            DECLARE_STATIC_PROPERTY( Key_Atilde );
            DECLARE_STATIC_PROPERTY( Key_Adiaeresis );
            DECLARE_STATIC_PROPERTY( Key_Aring );
            DECLARE_STATIC_PROPERTY( Key_AE );
            DECLARE_STATIC_PROPERTY( Key_Ccedilla );
            DECLARE_STATIC_PROPERTY( Key_Egrave );
            DECLARE_STATIC_PROPERTY( Key_Eacute );
            DECLARE_STATIC_PROPERTY( Key_Ecircumflex );
            DECLARE_STATIC_PROPERTY( Key_Ediaeresis );
            DECLARE_STATIC_PROPERTY( Key_Igrave );
            DECLARE_STATIC_PROPERTY( Key_Iacute );
            DECLARE_STATIC_PROPERTY( Key_Icircumflex );
            DECLARE_STATIC_PROPERTY( Key_Idiaeresis );
            DECLARE_STATIC_PROPERTY( Key_ETH );
            DECLARE_STATIC_PROPERTY( Key_Ntilde );
            DECLARE_STATIC_PROPERTY( Key_Ograve );
            DECLARE_STATIC_PROPERTY( Key_Oacute );
            DECLARE_STATIC_PROPERTY( Key_Ocircumflex );
            DECLARE_STATIC_PROPERTY( Key_Otilde );
            DECLARE_STATIC_PROPERTY( Key_Odiaeresis );
            DECLARE_STATIC_PROPERTY( Key_multiply );
            DECLARE_STATIC_PROPERTY( Key_Ooblique );
            DECLARE_STATIC_PROPERTY( Key_Ugrave );
            DECLARE_STATIC_PROPERTY( Key_Uacute );
            DECLARE_STATIC_PROPERTY( Key_Ucircumflex );
            DECLARE_STATIC_PROPERTY( Key_Udiaeresis );
            DECLARE_STATIC_PROPERTY( Key_Yacute );
            DECLARE_STATIC_PROPERTY( Key_THORN );
            DECLARE_STATIC_PROPERTY( Key_ssharp );
            DECLARE_STATIC_PROPERTY( Key_division );
            DECLARE_STATIC_PROPERTY( Key_ydiaeresis );
            DECLARE_STATIC_PROPERTY( Key_Multi_key );
            DECLARE_STATIC_PROPERTY( Key_Codeinput );
            DECLARE_STATIC_PROPERTY( Key_SingleCandidate );
            DECLARE_STATIC_PROPERTY( Key_MultipleCandidate );
            DECLARE_STATIC_PROPERTY( Key_PreviousCandidate );
            DECLARE_STATIC_PROPERTY( Key_Mode_switch );
            DECLARE_STATIC_PROPERTY( Key_Kanji );
            DECLARE_STATIC_PROPERTY( Key_Muhenkan );
            DECLARE_STATIC_PROPERTY( Key_Henkan );
            DECLARE_STATIC_PROPERTY( Key_Romaji );
            DECLARE_STATIC_PROPERTY( Key_Hiragana );
            DECLARE_STATIC_PROPERTY( Key_Katakana );
            DECLARE_STATIC_PROPERTY( Key_Hiragana_Katakana );
            DECLARE_STATIC_PROPERTY( Key_Zenkaku );
            DECLARE_STATIC_PROPERTY( Key_Hankaku );
            DECLARE_STATIC_PROPERTY( Key_Zenkaku_Hankaku );
            DECLARE_STATIC_PROPERTY( Key_Touroku );
            DECLARE_STATIC_PROPERTY( Key_Massyo );
            DECLARE_STATIC_PROPERTY( Key_Kana_Lock );
            DECLARE_STATIC_PROPERTY( Key_Kana_Shift );
            DECLARE_STATIC_PROPERTY( Key_Eisu_Shift );
            DECLARE_STATIC_PROPERTY( Key_Eisu_toggle );
            DECLARE_STATIC_PROPERTY( Key_Hangul );
            DECLARE_STATIC_PROPERTY( Key_Hangul_Start );
            DECLARE_STATIC_PROPERTY( Key_Hangul_End );
            DECLARE_STATIC_PROPERTY( Key_Hangul_Hanja );
            DECLARE_STATIC_PROPERTY( Key_Hangul_Jamo );
            DECLARE_STATIC_PROPERTY( Key_Hangul_Romaja );
            DECLARE_STATIC_PROPERTY( Key_Hangul_Jeonja );
            DECLARE_STATIC_PROPERTY( Key_Hangul_Banja );
            DECLARE_STATIC_PROPERTY( Key_Hangul_PreHanja );
            DECLARE_STATIC_PROPERTY( Key_Hangul_PostHanja );
            DECLARE_STATIC_PROPERTY( Key_Hangul_Special );
            DECLARE_STATIC_PROPERTY( Key_Dead_Grave );
            DECLARE_STATIC_PROPERTY( Key_Dead_Acute );
            DECLARE_STATIC_PROPERTY( Key_Dead_Circumflex );
            DECLARE_STATIC_PROPERTY( Key_Dead_Tilde );
            DECLARE_STATIC_PROPERTY( Key_Dead_Macron );
            DECLARE_STATIC_PROPERTY( Key_Dead_Breve );
            DECLARE_STATIC_PROPERTY( Key_Dead_Abovedot );
            DECLARE_STATIC_PROPERTY( Key_Dead_Diaeresis );
            DECLARE_STATIC_PROPERTY( Key_Dead_Abovering );
            DECLARE_STATIC_PROPERTY( Key_Dead_Doubleacute );
            DECLARE_STATIC_PROPERTY( Key_Dead_Caron );
            DECLARE_STATIC_PROPERTY( Key_Dead_Cedilla );
            DECLARE_STATIC_PROPERTY( Key_Dead_Ogonek );
            DECLARE_STATIC_PROPERTY( Key_Dead_Iota );
            DECLARE_STATIC_PROPERTY( Key_Dead_Voiced_Sound );
            DECLARE_STATIC_PROPERTY( Key_Dead_Semivoiced_Sound );
            DECLARE_STATIC_PROPERTY( Key_Dead_Belowdot );
            DECLARE_STATIC_PROPERTY( Key_Dead_Hook );
            DECLARE_STATIC_PROPERTY( Key_Dead_Horn );
            DECLARE_STATIC_PROPERTY( Key_Back );
            DECLARE_STATIC_PROPERTY( Key_Forward );
            DECLARE_STATIC_PROPERTY( Key_Stop );
            DECLARE_STATIC_PROPERTY( Key_Refresh );
            DECLARE_STATIC_PROPERTY( Key_VolumeDown );
            DECLARE_STATIC_PROPERTY( Key_VolumeMute );
            DECLARE_STATIC_PROPERTY( Key_VolumeUp );
            DECLARE_STATIC_PROPERTY( Key_BassBoost );
            DECLARE_STATIC_PROPERTY( Key_BassUp );
            DECLARE_STATIC_PROPERTY( Key_BassDown );
            DECLARE_STATIC_PROPERTY( Key_TrebleUp );
            DECLARE_STATIC_PROPERTY( Key_TrebleDown );
            DECLARE_STATIC_PROPERTY( Key_MediaPlay );
            DECLARE_STATIC_PROPERTY( Key_MediaStop );
            DECLARE_STATIC_PROPERTY( Key_MediaPrevious );
            DECLARE_STATIC_PROPERTY( Key_MediaNext );
            DECLARE_STATIC_PROPERTY( Key_MediaRecord );
            DECLARE_STATIC_PROPERTY( Key_HomePage );
            DECLARE_STATIC_PROPERTY( Key_Favorites );
            DECLARE_STATIC_PROPERTY( Key_Search );
            DECLARE_STATIC_PROPERTY( Key_Standby );
            DECLARE_STATIC_PROPERTY( Key_OpenUrl );
            DECLARE_STATIC_PROPERTY( Key_LaunchMail );
            DECLARE_STATIC_PROPERTY( Key_LaunchMedia );
            DECLARE_STATIC_PROPERTY( Key_Launch0 );
            DECLARE_STATIC_PROPERTY( Key_Launch1 );
            DECLARE_STATIC_PROPERTY( Key_Launch2 );
            DECLARE_STATIC_PROPERTY( Key_Launch3 );
            DECLARE_STATIC_PROPERTY( Key_Launch4 );
            DECLARE_STATIC_PROPERTY( Key_Launch5 );
            DECLARE_STATIC_PROPERTY( Key_Launch6 );
            DECLARE_STATIC_PROPERTY( Key_Launch7 );
            DECLARE_STATIC_PROPERTY( Key_Launch8 );
            DECLARE_STATIC_PROPERTY( Key_Launch9 );
            DECLARE_STATIC_PROPERTY( Key_LaunchA );
            DECLARE_STATIC_PROPERTY( Key_LaunchB );
            DECLARE_STATIC_PROPERTY( Key_LaunchC );
            DECLARE_STATIC_PROPERTY( Key_LaunchD );
            DECLARE_STATIC_PROPERTY( Key_LaunchE );
            DECLARE_STATIC_PROPERTY( Key_LaunchF );
            DECLARE_STATIC_PROPERTY( Key_MonBrightnessUp );
            DECLARE_STATIC_PROPERTY( Key_MonBrightnessDown );
            DECLARE_STATIC_PROPERTY( Key_KeyboardLightOnOff );
            DECLARE_STATIC_PROPERTY( Key_KeyboardBrightnessUp );
            DECLARE_STATIC_PROPERTY( Key_KeyboardBrightnessDown );
            DECLARE_STATIC_PROPERTY( Key_PowerOff );
            DECLARE_STATIC_PROPERTY( Key_WakeUp );
            DECLARE_STATIC_PROPERTY( Key_Eject );
            DECLARE_STATIC_PROPERTY( Key_ScreenSaver );
            DECLARE_STATIC_PROPERTY( Key_WWW );
            DECLARE_STATIC_PROPERTY( Key_Memo );
            DECLARE_STATIC_PROPERTY( Key_LightBulb );
            DECLARE_STATIC_PROPERTY( Key_Shop );
            DECLARE_STATIC_PROPERTY( Key_History );
            DECLARE_STATIC_PROPERTY( Key_AddFavorite );
            DECLARE_STATIC_PROPERTY( Key_HotLinks );
            DECLARE_STATIC_PROPERTY( Key_BrightnessAdjust );
            DECLARE_STATIC_PROPERTY( Key_Finance );
            DECLARE_STATIC_PROPERTY( Key_Community );
            DECLARE_STATIC_PROPERTY( Key_AudioRewind );
            DECLARE_STATIC_PROPERTY( Key_BackForward );
            DECLARE_STATIC_PROPERTY( Key_ApplicationLeft );
            DECLARE_STATIC_PROPERTY( Key_ApplicationRight );
            DECLARE_STATIC_PROPERTY( Key_Book );
            DECLARE_STATIC_PROPERTY( Key_CD );
            DECLARE_STATIC_PROPERTY( Key_Calculator );
            DECLARE_STATIC_PROPERTY( Key_ToDoList );
            DECLARE_STATIC_PROPERTY( Key_ClearGrab );
            DECLARE_STATIC_PROPERTY( Key_Close );
            DECLARE_STATIC_PROPERTY( Key_Copy );
            DECLARE_STATIC_PROPERTY( Key_Cut );
            DECLARE_STATIC_PROPERTY( Key_Display );
            DECLARE_STATIC_PROPERTY( Key_DOS );
            DECLARE_STATIC_PROPERTY( Key_Documents );
            DECLARE_STATIC_PROPERTY( Key_Excel );
            DECLARE_STATIC_PROPERTY( Key_Explorer );
            DECLARE_STATIC_PROPERTY( Key_Game );
            DECLARE_STATIC_PROPERTY( Key_Go );
            DECLARE_STATIC_PROPERTY( Key_iTouch );
            DECLARE_STATIC_PROPERTY( Key_LogOff );
            DECLARE_STATIC_PROPERTY( Key_Market );
            DECLARE_STATIC_PROPERTY( Key_Meeting );
            DECLARE_STATIC_PROPERTY( Key_MenuKB );
            DECLARE_STATIC_PROPERTY( Key_MenuPB );
            DECLARE_STATIC_PROPERTY( Key_MySites );
            DECLARE_STATIC_PROPERTY( Key_News );
            DECLARE_STATIC_PROPERTY( Key_OfficeHome );
            DECLARE_STATIC_PROPERTY( Key_Option );
            DECLARE_STATIC_PROPERTY( Key_Paste );
            DECLARE_STATIC_PROPERTY( Key_Phone );
            DECLARE_STATIC_PROPERTY( Key_Calendar );
            DECLARE_STATIC_PROPERTY( Key_Reply );
            DECLARE_STATIC_PROPERTY( Key_Reload );
            DECLARE_STATIC_PROPERTY( Key_RotateWindows );
            DECLARE_STATIC_PROPERTY( Key_RotationPB );
            DECLARE_STATIC_PROPERTY( Key_RotationKB );
            DECLARE_STATIC_PROPERTY( Key_Save );
            DECLARE_STATIC_PROPERTY( Key_Send );
            DECLARE_STATIC_PROPERTY( Key_Spell );
            DECLARE_STATIC_PROPERTY( Key_SplitScreen );
            DECLARE_STATIC_PROPERTY( Key_Support );
            DECLARE_STATIC_PROPERTY( Key_TaskPane );
            DECLARE_STATIC_PROPERTY( Key_Terminal );
            DECLARE_STATIC_PROPERTY( Key_Tools );
            DECLARE_STATIC_PROPERTY( Key_Travel );
            DECLARE_STATIC_PROPERTY( Key_Video );
            DECLARE_STATIC_PROPERTY( Key_Word );
            DECLARE_STATIC_PROPERTY( Key_Xfer );
            DECLARE_STATIC_PROPERTY( Key_ZoomIn );
            DECLARE_STATIC_PROPERTY( Key_ZoomOut );
            DECLARE_STATIC_PROPERTY( Key_Away );
            DECLARE_STATIC_PROPERTY( Key_Messenger );
            DECLARE_STATIC_PROPERTY( Key_WebCam );
            DECLARE_STATIC_PROPERTY( Key_MailForward );
            DECLARE_STATIC_PROPERTY( Key_Pictures );
            DECLARE_STATIC_PROPERTY( Key_Music );
            DECLARE_STATIC_PROPERTY( Key_Battery );
            DECLARE_STATIC_PROPERTY( Key_Bluetooth );
            DECLARE_STATIC_PROPERTY( Key_WLAN );
            DECLARE_STATIC_PROPERTY( Key_UWB );
            DECLARE_STATIC_PROPERTY( Key_AudioForward );
            DECLARE_STATIC_PROPERTY( Key_AudioRepeat );
            DECLARE_STATIC_PROPERTY( Key_AudioRandomPlay );
            DECLARE_STATIC_PROPERTY( Key_Subtitle );
            DECLARE_STATIC_PROPERTY( Key_AudioCycleTrack );
            DECLARE_STATIC_PROPERTY( Key_Time );
            DECLARE_STATIC_PROPERTY( Key_Hibernate );
            DECLARE_STATIC_PROPERTY( Key_View );
            DECLARE_STATIC_PROPERTY( Key_TopMenu );
            DECLARE_STATIC_PROPERTY( Key_PowerDown );
            DECLARE_STATIC_PROPERTY( Key_Suspend );
            DECLARE_STATIC_PROPERTY( Key_ContrastAdjust );
            DECLARE_STATIC_PROPERTY( Key_MediaLast );
            DECLARE_STATIC_PROPERTY( Key_unknown );
            DECLARE_STATIC_PROPERTY( Key_Call );
            DECLARE_STATIC_PROPERTY( Key_Context1 );
            DECLARE_STATIC_PROPERTY( Key_Context2 );
            DECLARE_STATIC_PROPERTY( Key_Context3 );
            DECLARE_STATIC_PROPERTY( Key_Context4 );
            DECLARE_STATIC_PROPERTY( Key_Flip );
            DECLARE_STATIC_PROPERTY( Key_Hangup );
            DECLARE_STATIC_PROPERTY( Key_No );
            DECLARE_STATIC_PROPERTY( Key_Select );
            DECLARE_STATIC_PROPERTY( Key_Yes );
            DECLARE_STATIC_PROPERTY( Key_Execute );
            DECLARE_STATIC_PROPERTY( Key_Printer );
            DECLARE_STATIC_PROPERTY( Key_Play );
            DECLARE_STATIC_PROPERTY( Key_Sleep );
            DECLARE_STATIC_PROPERTY( Key_Zoom );
            DECLARE_STATIC_PROPERTY( Key_Cancel );
          END_DECLARE_STATIC_PROPERTIES

        protected:
  
          class SceniXQGLWidgetPrivate;

          SceniXQGLWidgetPrivate *m_glWidget;

          /** Called on hid (keyboard, mouse) events. Call always the base class in your own event handler! **/
          DP_SG_UI_QT5_API virtual void hidNotify( dp::util::PropertyId propertyId );

        private:
          // HID
          struct KeyInfo {
            KeyInfo() : propertyId(0), member(0), name(0) {}
            KeyInfo( dp::util::PropertyId id, bool SceniXQGLWidget::*keyMember, const char *keyName ) : propertyId(id), member( keyMember ), name(keyName) {}

            dp::util::PropertyId   propertyId;
            bool SceniXQGLWidget::*member;
            const char            *name;
          };

          DP_SG_UI_QT5_API dp::util::PropertyId getQtMouseButtonProperty( Qt::MouseButton button ) const;

          typedef std::vector<KeyInfo> KeyInfoVector;
          typedef std::map<Qt::Key, KeyInfo> KeyInfoMap;

          KeyInfoMap m_keyProperties;  //!< Map Qt key to propertyId
          KeyInfoVector m_keyInfos;  //!< For indexed access

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
          bool m_propKey_Back;
          bool m_propKey_Forward;
          bool m_propKey_Stop;
          bool m_propKey_Refresh;
          bool m_propKey_VolumeDown;
          bool m_propKey_VolumeMute;
          bool m_propKey_VolumeUp;
          bool m_propKey_BassBoost;
          bool m_propKey_BassUp;
          bool m_propKey_BassDown;
          bool m_propKey_TrebleUp;
          bool m_propKey_TrebleDown;
          bool m_propKey_MediaPlay;
          bool m_propKey_MediaStop;
          bool m_propKey_MediaPrevious;
          bool m_propKey_MediaNext;
          bool m_propKey_MediaRecord;
          bool m_propKey_HomePage;
          bool m_propKey_Favorites;
          bool m_propKey_Search;
          bool m_propKey_Standby;
          bool m_propKey_OpenUrl;
          bool m_propKey_LaunchMail;
          bool m_propKey_LaunchMedia;
          bool m_propKey_Launch0;
          bool m_propKey_Launch1;
          bool m_propKey_Launch2;
          bool m_propKey_Launch3;
          bool m_propKey_Launch4;
          bool m_propKey_Launch5;
          bool m_propKey_Launch6;
          bool m_propKey_Launch7;
          bool m_propKey_Launch8;
          bool m_propKey_Launch9;
          bool m_propKey_LaunchA;
          bool m_propKey_LaunchB;
          bool m_propKey_LaunchC;
          bool m_propKey_LaunchD;
          bool m_propKey_LaunchE;
          bool m_propKey_LaunchF;
          bool m_propKey_MonBrightnessUp;
          bool m_propKey_MonBrightnessDown;
          bool m_propKey_KeyboardLightOnOff;
          bool m_propKey_KeyboardBrightnessUp;
          bool m_propKey_KeyboardBrightnessDown;
          bool m_propKey_PowerOff;
          bool m_propKey_WakeUp;
          bool m_propKey_Eject;
          bool m_propKey_ScreenSaver;
          bool m_propKey_WWW;
          bool m_propKey_Memo;
          bool m_propKey_LightBulb;
          bool m_propKey_Shop;
          bool m_propKey_History;
          bool m_propKey_AddFavorite;
          bool m_propKey_HotLinks;
          bool m_propKey_BrightnessAdjust;
          bool m_propKey_Finance;
          bool m_propKey_Community;
          bool m_propKey_AudioRewind;
          bool m_propKey_BackForward;
          bool m_propKey_ApplicationLeft;
          bool m_propKey_ApplicationRight;
          bool m_propKey_Book;
          bool m_propKey_CD;
          bool m_propKey_Calculator;
          bool m_propKey_ToDoList;
          bool m_propKey_ClearGrab;
          bool m_propKey_Close;
          bool m_propKey_Copy;
          bool m_propKey_Cut;
          bool m_propKey_Display;
          bool m_propKey_DOS;
          bool m_propKey_Documents;
          bool m_propKey_Excel;
          bool m_propKey_Explorer;
          bool m_propKey_Game;
          bool m_propKey_Go;
          bool m_propKey_iTouch;
          bool m_propKey_LogOff;
          bool m_propKey_Market;
          bool m_propKey_Meeting;
          bool m_propKey_MenuKB;
          bool m_propKey_MenuPB;
          bool m_propKey_MySites;
          bool m_propKey_News;
          bool m_propKey_OfficeHome;
          bool m_propKey_Option;
          bool m_propKey_Paste;
          bool m_propKey_Phone;
          bool m_propKey_Calendar;
          bool m_propKey_Reply;
          bool m_propKey_Reload;
          bool m_propKey_RotateWindows;
          bool m_propKey_RotationPB;
          bool m_propKey_RotationKB;
          bool m_propKey_Save;
          bool m_propKey_Send;
          bool m_propKey_Spell;
          bool m_propKey_SplitScreen;
          bool m_propKey_Support;
          bool m_propKey_TaskPane;
          bool m_propKey_Terminal;
          bool m_propKey_Tools;
          bool m_propKey_Travel;
          bool m_propKey_Video;
          bool m_propKey_Word;
          bool m_propKey_Xfer;
          bool m_propKey_ZoomIn;
          bool m_propKey_ZoomOut;
          bool m_propKey_Away;
          bool m_propKey_Messenger;
          bool m_propKey_WebCam;
          bool m_propKey_MailForward;
          bool m_propKey_Pictures;
          bool m_propKey_Music;
          bool m_propKey_Battery;
          bool m_propKey_Bluetooth;
          bool m_propKey_WLAN;
          bool m_propKey_UWB;
          bool m_propKey_AudioForward;
          bool m_propKey_AudioRepeat;
          bool m_propKey_AudioRandomPlay;
          bool m_propKey_Subtitle;
          bool m_propKey_AudioCycleTrack;
          bool m_propKey_Time;
          bool m_propKey_Hibernate;
          bool m_propKey_View;
          bool m_propKey_TopMenu;
          bool m_propKey_PowerDown;
          bool m_propKey_Suspend;
          bool m_propKey_ContrastAdjust;
          bool m_propKey_MediaLast;
          bool m_propKey_unknown;
          bool m_propKey_Call;
          bool m_propKey_Context1;
          bool m_propKey_Context2;
          bool m_propKey_Context3;
          bool m_propKey_Context4;
          bool m_propKey_Flip;
          bool m_propKey_Hangup;
          bool m_propKey_No;
          bool m_propKey_Select;
          bool m_propKey_Yes;
          bool m_propKey_Execute;
          bool m_propKey_Printer;
          bool m_propKey_Play;
          bool m_propKey_Sleep;
          bool m_propKey_Zoom;
          bool m_propKey_Cancel;
        };

      } // namespace qt5
    } // namespace ui
  } // namespace sg
} // namespace dp
