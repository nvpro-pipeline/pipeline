// Copyright (c) 2009-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <GL/glew.h>

#include <QResizeEvent>

#include <dp/sg/ui/qt5/SceniXQGLWidget.h>

#include <dp/gl/RenderTarget.h>
#include <dp/gl/RenderTargetFB.h>
#include <dp/gl/RenderContextFormat.h>

#if defined(DP_OS_LINUX)
#include <QX11Info>
#endif

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace qt5
      {

        using namespace dp::util;

        class SceniXQGLWidget::SceniXQGLWidgetPrivate : public QWidget
        {
        public:
          SceniXQGLWidgetPrivate( SceniXQGLWidget *parent, const dp::gl::RenderContextFormat &format, SceniXQGLWidgetPrivate *shareWidget );
          virtual ~SceniXQGLWidgetPrivate();

          const dp::gl::RenderContextSharedPtr & getRenderContext() const;
          const dp::gl::RenderTargetSharedPtr  & getRenderTarget() const;

          QPaintEngine *paintEngine() const;

        protected:
          void initialize();
          virtual void paintEvent( QPaintEvent * paintEvent );
          virtual void resizeEvent( QResizeEvent *resizeEvent );

          dp::gl::RenderTargetSharedPtr   m_renderTarget;
          bool                            m_initialized;
          SceniXQGLWidgetPrivate        * m_shareWidget;
          dp::gl::RenderContextFormat     m_format;
        };


        // mouse
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Mouse_Left );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Mouse_Middle );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Mouse_Right );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Mouse_Position );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Mouse_Wheel );

        // keyboard
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Escape );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Tab );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Backtab );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Backspace );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Return );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Enter );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Insert );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Delete );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Pause );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Print );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_SysReq );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Clear );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Home );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_End );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Left );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Up );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Right );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Down );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_PageUp );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_PageDown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Shift );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Control );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Meta );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Alt );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AltGr );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_CapsLock );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_NumLock );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ScrollLock );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F1 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F2 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F3 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F4 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F5 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F6 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F7 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F8 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F9 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F10 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F11 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F12 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F13 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F14 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F15 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F16 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F17 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F18 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F19 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F20 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F21 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F22 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F23 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F24 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F25 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F26 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F27 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F28 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F29 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F30 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F31 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F32 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F33 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F34 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F35 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Super_L );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Super_R );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Menu );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hyper_L );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hyper_R );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Help );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Direction_L );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Direction_R );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Space );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Any );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Exclam );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_QuoteDbl );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_NumberSign );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dollar );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Percent );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ampersand );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Apostrophe );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ParenLeft );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ParenRight );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Asterisk );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Plus );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Comma );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Minus );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Period );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Slash );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_0 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_1 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_2 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_3 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_4 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_5 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_6 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_7 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_8 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_9 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Colon );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Semicolon );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Less );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Equal );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Greater );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Question );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_At );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_A );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_B );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_C );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_D );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_E );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_F );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_G );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_H );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_I );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_J );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_K );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_L );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_M );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_N );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_O );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_P );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Q );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_R );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_S );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_T );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_U );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_V );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_W );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_X );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Y );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Z );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_BracketLeft );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Backslash );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_BracketRight );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AsciiCircum );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Underscore );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_QuoteLeft );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_BraceLeft );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Bar );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_BraceRight );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AsciiTilde );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_nobreakspace );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_exclamdown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_cent );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_sterling );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_currency );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_yen );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_brokenbar );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_section );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_diaeresis );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_copyright );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ordfeminine );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_guillemotleft );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_notsign );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_hyphen );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_registered );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_macron );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_degree );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_plusminus );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_twosuperior );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_threesuperior );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_acute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_mu );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_paragraph );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_periodcentered );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_cedilla );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_onesuperior );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_masculine );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_guillemotright );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_onequarter );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_onehalf );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_threequarters );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_questiondown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Agrave );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Aacute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Acircumflex );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Atilde );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Adiaeresis );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Aring );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AE );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ccedilla );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Egrave );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Eacute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ecircumflex );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ediaeresis );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Igrave );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Iacute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Icircumflex );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Idiaeresis );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ETH );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ntilde );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ograve );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Oacute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ocircumflex );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Otilde );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Odiaeresis );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_multiply );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ooblique );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ugrave );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Uacute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Ucircumflex );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Udiaeresis );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Yacute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_THORN );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ssharp );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_division );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ydiaeresis );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Multi_key );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Codeinput );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_SingleCandidate );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MultipleCandidate );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_PreviousCandidate );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Mode_switch );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Kanji );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Muhenkan );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Henkan );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Romaji );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hiragana );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Katakana );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hiragana_Katakana );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Zenkaku );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hankaku );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Zenkaku_Hankaku );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Touroku );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Massyo );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Kana_Lock );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Kana_Shift );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Eisu_Shift );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Eisu_toggle );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_Start );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_End );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_Hanja );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_Jamo );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_Romaja );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_Jeonja );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_Banja );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_PreHanja );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_PostHanja );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangul_Special );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Grave );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Acute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Circumflex );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Tilde );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Macron );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Breve );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Abovedot );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Diaeresis );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Abovering );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Doubleacute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Caron );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Cedilla );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Ogonek );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Iota );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Voiced_Sound );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Semivoiced_Sound );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Belowdot );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Hook );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Dead_Horn );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Back );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Forward );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Stop );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Refresh );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_VolumeDown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_VolumeMute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_VolumeUp );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_BassBoost );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_BassUp );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_BassDown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_TrebleUp );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_TrebleDown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MediaPlay );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MediaStop );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MediaPrevious );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MediaNext );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MediaRecord );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_HomePage );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Favorites );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Search );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Standby );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_OpenUrl );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LaunchMail );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LaunchMedia );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch0 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch1 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch2 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch3 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch4 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch5 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch6 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch7 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch8 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Launch9 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LaunchA );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LaunchB );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LaunchC );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LaunchD );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LaunchE );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LaunchF );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MonBrightnessUp );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MonBrightnessDown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_KeyboardLightOnOff );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_KeyboardBrightnessUp );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_KeyboardBrightnessDown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_PowerOff );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_WakeUp );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Eject );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ScreenSaver );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_WWW );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Memo );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LightBulb );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Shop );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_History );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AddFavorite );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_HotLinks );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_BrightnessAdjust );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Finance );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Community );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AudioRewind );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_BackForward );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ApplicationLeft );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ApplicationRight );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Book );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_CD );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Calculator );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ToDoList );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ClearGrab );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Close );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Copy );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Cut );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Display );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_DOS );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Documents );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Excel );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Explorer );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Game );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Go );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_iTouch );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_LogOff );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Market );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Meeting );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MenuKB );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MenuPB );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MySites );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_News );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_OfficeHome );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Option );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Paste );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Phone );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Calendar );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Reply );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Reload );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_RotateWindows );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_RotationPB );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_RotationKB );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Save );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Send );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Spell );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_SplitScreen );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Support );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_TaskPane );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Terminal );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Tools );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Travel );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Video );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Word );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Xfer );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ZoomIn );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ZoomOut );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Away );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Messenger );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_WebCam );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MailForward );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Pictures );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Music );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Battery );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Bluetooth );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_WLAN );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_UWB );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AudioForward );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AudioRepeat );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AudioRandomPlay );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Subtitle );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_AudioCycleTrack );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Time );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hibernate );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_View );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_TopMenu );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_PowerDown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Suspend );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_ContrastAdjust );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_MediaLast );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_unknown );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Call );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Context1 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Context2 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Context3 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Context4 );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Flip );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Hangup );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_No );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Select );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Yes );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Execute );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Printer );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Play );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Sleep );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Zoom );
        DEFINE_STATIC_PROPERTY( SceniXQGLWidget, Key_Cancel );

        BEGIN_REFLECTION_INFO( SceniXQGLWidget )
          DERIVE_STATIC_PROPERTIES( SceniXQGLWidget, HumanInterfaceDevice );

          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Mouse_Left, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Mouse_Middle, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Mouse_Right, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Mouse_Position, dp::math::Vec2i, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Mouse_Wheel, int, Semantic::VALUE );

          // keyboard
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Escape, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Tab, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Backtab, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Backspace, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Return, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Enter, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Insert, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Delete, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Pause, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Print, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_SysReq, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Clear, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Home, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_End, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Left, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Up, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Right, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Down, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_PageUp, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_PageDown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Shift, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Control, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Meta, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Alt, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AltGr, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_CapsLock, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_NumLock, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ScrollLock, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F1, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F2, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F3, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F4, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F5, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F6, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F7, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F8, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F9, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F10, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F11, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F12, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F13, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F14, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F15, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F16, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F17, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F18, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F19, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F20, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F21, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F22, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F23, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F24, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F25, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F26, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F27, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F28, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F29, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F30, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F31, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F32, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F33, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F34, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F35, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Super_L, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Super_R, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Menu, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hyper_L, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hyper_R, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Help, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Direction_L, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Direction_R, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Space, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Any, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Exclam, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_QuoteDbl, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_NumberSign, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dollar, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Percent, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ampersand, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Apostrophe, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ParenLeft, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ParenRight, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Asterisk, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Plus, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Comma, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Minus, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Period, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Slash, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_0, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_1, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_2, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_3, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_4, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_5, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_6, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_7, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_8, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_9, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Colon, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Semicolon, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Less, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Equal, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Greater, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Question, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_At, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_A, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_B, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_C, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_D, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_E, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_F, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_G, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_H, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_I, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_J, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_K, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_L, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_M, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_N, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_O, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_P, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Q, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_R, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_S, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_T, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_U, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_V, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_W, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_X, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Y, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Z, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_BracketLeft, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Backslash, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_BracketRight, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AsciiCircum, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Underscore, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_QuoteLeft, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_BraceLeft, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Bar, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_BraceRight, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AsciiTilde, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_nobreakspace, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_exclamdown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_cent, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_sterling, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_currency, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_yen, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_brokenbar, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_section, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_diaeresis, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_copyright, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ordfeminine, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_guillemotleft, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_notsign, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_hyphen, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_registered, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_macron, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_degree, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_plusminus, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_twosuperior, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_threesuperior, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_acute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_mu, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_paragraph, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_periodcentered, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_cedilla, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_onesuperior, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_masculine, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_guillemotright, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_onequarter, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_onehalf, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_threequarters, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_questiondown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Agrave, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Aacute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Acircumflex, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Atilde, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Adiaeresis, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Aring, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AE, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ccedilla, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Egrave, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Eacute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ecircumflex, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ediaeresis, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Igrave, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Iacute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Icircumflex, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Idiaeresis, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ETH, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ntilde, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ograve, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Oacute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ocircumflex, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Otilde, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Odiaeresis, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_multiply, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ooblique, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ugrave, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Uacute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Ucircumflex, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Udiaeresis, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Yacute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_THORN, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ssharp, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_division, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ydiaeresis, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Multi_key, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Codeinput, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_SingleCandidate, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MultipleCandidate, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_PreviousCandidate, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Mode_switch, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Kanji, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Muhenkan, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Henkan, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Romaji, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hiragana, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Katakana, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hiragana_Katakana, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Zenkaku, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hankaku, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Zenkaku_Hankaku, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Touroku, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Massyo, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Kana_Lock, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Kana_Shift, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Eisu_Shift, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Eisu_toggle, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_Start, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_End, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_Hanja, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_Jamo, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_Romaja, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_Jeonja, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_Banja, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_PreHanja, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_PostHanja, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangul_Special, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Grave, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Acute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Circumflex, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Tilde, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Macron, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Breve, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Abovedot, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Diaeresis, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Abovering, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Doubleacute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Caron, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Cedilla, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Ogonek, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Iota, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Voiced_Sound, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Semivoiced_Sound, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Belowdot, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Hook, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Dead_Horn, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Back, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Forward, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Stop, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Refresh, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_VolumeDown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_VolumeMute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_VolumeUp, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_BassBoost, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_BassUp, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_BassDown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_TrebleUp, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_TrebleDown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MediaPlay, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MediaStop, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MediaPrevious, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MediaNext, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MediaRecord, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_HomePage, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Favorites, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Search, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Standby, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_OpenUrl, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LaunchMail, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LaunchMedia, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch0, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch1, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch2, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch3, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch4, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch5, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch6, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch7, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch8, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Launch9, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LaunchA, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LaunchB, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LaunchC, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LaunchD, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LaunchE, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LaunchF, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MonBrightnessUp, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MonBrightnessDown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_KeyboardLightOnOff, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_KeyboardBrightnessUp, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_KeyboardBrightnessDown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_PowerOff, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_WakeUp, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Eject, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ScreenSaver, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_WWW, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Memo, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LightBulb, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Shop, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_History, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AddFavorite, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_HotLinks, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_BrightnessAdjust, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Finance, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Community, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AudioRewind, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_BackForward, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ApplicationLeft, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ApplicationRight, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Book, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_CD, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Calculator, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ToDoList, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ClearGrab, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Close, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Copy, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Cut, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Display, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_DOS, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Documents, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Excel, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Explorer, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Game, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Go, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_iTouch, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_LogOff, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Market, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Meeting, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MenuKB, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MenuPB, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MySites, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_News, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_OfficeHome, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Option, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Paste, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Phone, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Calendar, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Reply, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Reload, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_RotateWindows, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_RotationPB, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_RotationKB, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Save, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Send, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Spell, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_SplitScreen, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Support, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_TaskPane, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Terminal, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Tools, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Travel, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Video, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Word, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Xfer, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ZoomIn, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ZoomOut, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Away, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Messenger, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_WebCam, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MailForward, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Pictures, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Music, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Battery, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Bluetooth, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_WLAN, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_UWB, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AudioForward, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AudioRepeat, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AudioRandomPlay, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Subtitle, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_AudioCycleTrack, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Time, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hibernate, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_View, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_TopMenu, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_PowerDown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Suspend, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_ContrastAdjust, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_MediaLast, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_unknown, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Call, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Context1, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Context2, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Context3, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Context4, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Flip, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Hangup, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_No, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Select, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Yes, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Execute, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Printer, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Play, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Sleep, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Zoom, bool, Semantic::VALUE );
          INIT_STATIC_PROPERTY_RO_MEMBER( SceniXQGLWidget, Key_Cancel, bool, Semantic::VALUE );
        END_REFLECTION_INFO


        SceniXQGLWidget::SceniXQGLWidget( QWidget *parent, const dp::gl::RenderContextFormat &format, SceniXQGLWidget *shareWidget )
        : QWidget( parent )
        {
          setAttribute( Qt::WA_NativeWindow );
          setAttribute( Qt::WA_PaintOnScreen ); // don't let qt paint anything on screen

          m_glWidget = new SceniXQGLWidgetPrivate( this, format, shareWidget ? shareWidget->m_glWidget : nullptr );
          m_glWidget->move(0,0);

          setMouseTracking( true );

          PID_Mouse_Left = getProperty( "Mouse_Left" );
          m_propMouse_Left = false;

          PID_Mouse_Middle = getProperty( "Mouse_Middle" );
          m_propMouse_Middle = false;

          PID_Mouse_Right = getProperty( "Mouse_Right" );
          m_propMouse_Right = false;

          PID_Mouse_Position = getProperty( "Mouse_Position" );
          m_propMouse_Position = dp::math::Vec2i(0,0);

          PID_Mouse_Wheel = getProperty("Mouse_Wheel");
          m_propMouse_Wheel = 0;

        #define INIT_KEY(key) m_prop##key = false; m_keyProperties[ Qt::key ] = KeyInfo(getProperty(#key), &SceniXQGLWidget::m_prop##key, #key); m_keyInfos.push_back( KeyInfo( getProperty(#key), &SceniXQGLWidget::m_prop##key, #key ) );

          INIT_KEY(Key_Escape);
          INIT_KEY(Key_Tab);
          INIT_KEY(Key_Backtab);
          INIT_KEY(Key_Backspace);
          INIT_KEY(Key_Return);
          INIT_KEY(Key_Enter);
          INIT_KEY(Key_Insert);
          INIT_KEY(Key_Delete);
          INIT_KEY(Key_Pause);
          INIT_KEY(Key_Print);
          INIT_KEY(Key_SysReq);
          INIT_KEY(Key_Clear);
          INIT_KEY(Key_Home);
          INIT_KEY(Key_End);
          INIT_KEY(Key_Left);
          INIT_KEY(Key_Up);
          INIT_KEY(Key_Right);
          INIT_KEY(Key_Down);
          INIT_KEY(Key_PageUp);
          INIT_KEY(Key_PageDown);
          INIT_KEY(Key_Shift);
          INIT_KEY(Key_Control);
          INIT_KEY(Key_Meta);
          INIT_KEY(Key_Alt);
          INIT_KEY(Key_AltGr);
          INIT_KEY(Key_CapsLock);
          INIT_KEY(Key_NumLock);
          INIT_KEY(Key_ScrollLock);
          INIT_KEY(Key_F1);
          INIT_KEY(Key_F2);
          INIT_KEY(Key_F3);
          INIT_KEY(Key_F4);
          INIT_KEY(Key_F5);
          INIT_KEY(Key_F6);
          INIT_KEY(Key_F7);
          INIT_KEY(Key_F8);
          INIT_KEY(Key_F9);
          INIT_KEY(Key_F10);
          INIT_KEY(Key_F11);
          INIT_KEY(Key_F12);
          INIT_KEY(Key_F13);
          INIT_KEY(Key_F14);
          INIT_KEY(Key_F15);
          INIT_KEY(Key_F16);
          INIT_KEY(Key_F17);
          INIT_KEY(Key_F18);
          INIT_KEY(Key_F19);
          INIT_KEY(Key_F20);
          INIT_KEY(Key_F21);
          INIT_KEY(Key_F22);
          INIT_KEY(Key_F23);
          INIT_KEY(Key_F24);
          INIT_KEY(Key_F25);
          INIT_KEY(Key_F26);
          INIT_KEY(Key_F27);
          INIT_KEY(Key_F28);
          INIT_KEY(Key_F29);
          INIT_KEY(Key_F30);
          INIT_KEY(Key_F31);
          INIT_KEY(Key_F32);
          INIT_KEY(Key_F33);
          INIT_KEY(Key_F34);
          INIT_KEY(Key_F35);
          INIT_KEY(Key_Super_L);
          INIT_KEY(Key_Super_R);
          INIT_KEY(Key_Menu);
          INIT_KEY(Key_Hyper_L);
          INIT_KEY(Key_Hyper_R);
          INIT_KEY(Key_Help);
          INIT_KEY(Key_Direction_L);
          INIT_KEY(Key_Direction_R);
          INIT_KEY(Key_Space);
          INIT_KEY(Key_Any);
          INIT_KEY(Key_Exclam);
          INIT_KEY(Key_QuoteDbl);
          INIT_KEY(Key_NumberSign);
          INIT_KEY(Key_Dollar);
          INIT_KEY(Key_Percent);
          INIT_KEY(Key_Ampersand);
          INIT_KEY(Key_Apostrophe);
          INIT_KEY(Key_ParenLeft);
          INIT_KEY(Key_ParenRight);
          INIT_KEY(Key_Asterisk);
          INIT_KEY(Key_Plus);
          INIT_KEY(Key_Comma);
          INIT_KEY(Key_Minus);
          INIT_KEY(Key_Period);
          INIT_KEY(Key_Slash);
          INIT_KEY(Key_0);
          INIT_KEY(Key_1);
          INIT_KEY(Key_2);
          INIT_KEY(Key_3);
          INIT_KEY(Key_4);
          INIT_KEY(Key_5);
          INIT_KEY(Key_6);
          INIT_KEY(Key_7);
          INIT_KEY(Key_8);
          INIT_KEY(Key_9);
          INIT_KEY(Key_Colon);
          INIT_KEY(Key_Semicolon);
          INIT_KEY(Key_Less);
          INIT_KEY(Key_Equal);
          INIT_KEY(Key_Greater);
          INIT_KEY(Key_Question);
          INIT_KEY(Key_At);
          INIT_KEY(Key_A);
          INIT_KEY(Key_B);
          INIT_KEY(Key_C);
          INIT_KEY(Key_D);
          INIT_KEY(Key_E);
          INIT_KEY(Key_F);
          INIT_KEY(Key_G);
          INIT_KEY(Key_H);
          INIT_KEY(Key_I);
          INIT_KEY(Key_J);
          INIT_KEY(Key_K);
          INIT_KEY(Key_L);
          INIT_KEY(Key_M);
          INIT_KEY(Key_N);
          INIT_KEY(Key_O);
          INIT_KEY(Key_P);
          INIT_KEY(Key_Q);
          INIT_KEY(Key_R);
          INIT_KEY(Key_S);
          INIT_KEY(Key_T);
          INIT_KEY(Key_U);
          INIT_KEY(Key_V);
          INIT_KEY(Key_W);
          INIT_KEY(Key_X);
          INIT_KEY(Key_Y);
          INIT_KEY(Key_Z);
          INIT_KEY(Key_BracketLeft);
          INIT_KEY(Key_Backslash);
          INIT_KEY(Key_BracketRight);
          INIT_KEY(Key_AsciiCircum);
          INIT_KEY(Key_Underscore);
          INIT_KEY(Key_QuoteLeft);
          INIT_KEY(Key_BraceLeft);
          INIT_KEY(Key_Bar);
          INIT_KEY(Key_BraceRight);
          INIT_KEY(Key_AsciiTilde);
          INIT_KEY(Key_nobreakspace);
          INIT_KEY(Key_exclamdown);
          INIT_KEY(Key_cent);
          INIT_KEY(Key_sterling);
          INIT_KEY(Key_currency);
          INIT_KEY(Key_yen);
          INIT_KEY(Key_brokenbar);
          INIT_KEY(Key_section);
          INIT_KEY(Key_diaeresis);
          INIT_KEY(Key_copyright);
          INIT_KEY(Key_ordfeminine);
          INIT_KEY(Key_guillemotleft);
          INIT_KEY(Key_notsign);
          INIT_KEY(Key_hyphen);
          INIT_KEY(Key_registered);
          INIT_KEY(Key_macron);
          INIT_KEY(Key_degree);
          INIT_KEY(Key_plusminus);
          INIT_KEY(Key_twosuperior);
          INIT_KEY(Key_threesuperior);
          INIT_KEY(Key_acute);
          INIT_KEY(Key_mu);
          INIT_KEY(Key_paragraph);
          INIT_KEY(Key_periodcentered);
          INIT_KEY(Key_cedilla);
          INIT_KEY(Key_onesuperior);
          INIT_KEY(Key_masculine);
          INIT_KEY(Key_guillemotright);
          INIT_KEY(Key_onequarter);
          INIT_KEY(Key_onehalf);
          INIT_KEY(Key_threequarters);
          INIT_KEY(Key_questiondown);
          INIT_KEY(Key_Agrave);
          INIT_KEY(Key_Aacute);
          INIT_KEY(Key_Acircumflex);
          INIT_KEY(Key_Atilde);
          INIT_KEY(Key_Adiaeresis);
          INIT_KEY(Key_Aring);
          INIT_KEY(Key_AE);
          INIT_KEY(Key_Ccedilla);
          INIT_KEY(Key_Egrave);
          INIT_KEY(Key_Eacute);
          INIT_KEY(Key_Ecircumflex);
          INIT_KEY(Key_Ediaeresis);
          INIT_KEY(Key_Igrave);
          INIT_KEY(Key_Iacute);
          INIT_KEY(Key_Icircumflex);
          INIT_KEY(Key_Idiaeresis);
          INIT_KEY(Key_ETH);
          INIT_KEY(Key_Ntilde);
          INIT_KEY(Key_Ograve);
          INIT_KEY(Key_Oacute);
          INIT_KEY(Key_Ocircumflex);
          INIT_KEY(Key_Otilde);
          INIT_KEY(Key_Odiaeresis);
          INIT_KEY(Key_multiply);
          INIT_KEY(Key_Ooblique);
          INIT_KEY(Key_Ugrave);
          INIT_KEY(Key_Uacute);
          INIT_KEY(Key_Ucircumflex);
          INIT_KEY(Key_Udiaeresis);
          INIT_KEY(Key_Yacute);
          INIT_KEY(Key_THORN);
          INIT_KEY(Key_ssharp);
          INIT_KEY(Key_division);
          INIT_KEY(Key_ydiaeresis);
          INIT_KEY(Key_Multi_key);
          INIT_KEY(Key_Codeinput);
          INIT_KEY(Key_SingleCandidate);
          INIT_KEY(Key_MultipleCandidate);
          INIT_KEY(Key_PreviousCandidate);
          INIT_KEY(Key_Mode_switch);
          INIT_KEY(Key_Kanji);
          INIT_KEY(Key_Muhenkan);
          INIT_KEY(Key_Henkan);
          INIT_KEY(Key_Romaji);
          INIT_KEY(Key_Hiragana);
          INIT_KEY(Key_Katakana);
          INIT_KEY(Key_Hiragana_Katakana);
          INIT_KEY(Key_Zenkaku);
          INIT_KEY(Key_Hankaku);
          INIT_KEY(Key_Zenkaku_Hankaku);
          INIT_KEY(Key_Touroku);
          INIT_KEY(Key_Massyo);
          INIT_KEY(Key_Kana_Lock);
          INIT_KEY(Key_Kana_Shift);
          INIT_KEY(Key_Eisu_Shift);
          INIT_KEY(Key_Eisu_toggle);
          INIT_KEY(Key_Hangul);
          INIT_KEY(Key_Hangul_Start);
          INIT_KEY(Key_Hangul_End);
          INIT_KEY(Key_Hangul_Hanja);
          INIT_KEY(Key_Hangul_Jamo);
          INIT_KEY(Key_Hangul_Romaja);
          INIT_KEY(Key_Hangul_Jeonja);
          INIT_KEY(Key_Hangul_Banja);
          INIT_KEY(Key_Hangul_PreHanja);
          INIT_KEY(Key_Hangul_PostHanja);
          INIT_KEY(Key_Hangul_Special);
          INIT_KEY(Key_Dead_Grave);
          INIT_KEY(Key_Dead_Acute);
          INIT_KEY(Key_Dead_Circumflex);
          INIT_KEY(Key_Dead_Tilde);
          INIT_KEY(Key_Dead_Macron);
          INIT_KEY(Key_Dead_Breve);
          INIT_KEY(Key_Dead_Abovedot);
          INIT_KEY(Key_Dead_Diaeresis);
          INIT_KEY(Key_Dead_Abovering);
          INIT_KEY(Key_Dead_Doubleacute);
          INIT_KEY(Key_Dead_Caron);
          INIT_KEY(Key_Dead_Cedilla);
          INIT_KEY(Key_Dead_Ogonek);
          INIT_KEY(Key_Dead_Iota);
          INIT_KEY(Key_Dead_Voiced_Sound);
          INIT_KEY(Key_Dead_Semivoiced_Sound);
          INIT_KEY(Key_Dead_Belowdot);
          INIT_KEY(Key_Dead_Hook);
          INIT_KEY(Key_Dead_Horn);
          INIT_KEY(Key_Back);
          INIT_KEY(Key_Forward);
          INIT_KEY(Key_Stop);
          INIT_KEY(Key_Refresh);
          INIT_KEY(Key_VolumeDown);
          INIT_KEY(Key_VolumeMute);
          INIT_KEY(Key_VolumeUp);
          INIT_KEY(Key_BassBoost);
          INIT_KEY(Key_BassUp);
          INIT_KEY(Key_BassDown);
          INIT_KEY(Key_TrebleUp);
          INIT_KEY(Key_TrebleDown);
          INIT_KEY(Key_MediaPlay);
          INIT_KEY(Key_MediaStop);
          INIT_KEY(Key_MediaPrevious);
          INIT_KEY(Key_MediaNext);
          INIT_KEY(Key_MediaRecord);
          INIT_KEY(Key_HomePage);
          INIT_KEY(Key_Favorites);
          INIT_KEY(Key_Search);
          INIT_KEY(Key_Standby);
          INIT_KEY(Key_OpenUrl);
          INIT_KEY(Key_LaunchMail);
          INIT_KEY(Key_LaunchMedia);
          INIT_KEY(Key_Launch0);
          INIT_KEY(Key_Launch1);
          INIT_KEY(Key_Launch2);
          INIT_KEY(Key_Launch3);
          INIT_KEY(Key_Launch4);
          INIT_KEY(Key_Launch5);
          INIT_KEY(Key_Launch6);
          INIT_KEY(Key_Launch7);
          INIT_KEY(Key_Launch8);
          INIT_KEY(Key_Launch9);
          INIT_KEY(Key_LaunchA);
          INIT_KEY(Key_LaunchB);
          INIT_KEY(Key_LaunchC);
          INIT_KEY(Key_LaunchD);
          INIT_KEY(Key_LaunchE);
          INIT_KEY(Key_LaunchF);
          INIT_KEY(Key_MonBrightnessUp);
          INIT_KEY(Key_MonBrightnessDown);
          INIT_KEY(Key_KeyboardLightOnOff);
          INIT_KEY(Key_KeyboardBrightnessUp);
          INIT_KEY(Key_KeyboardBrightnessDown);
          INIT_KEY(Key_PowerOff);
          INIT_KEY(Key_WakeUp);
          INIT_KEY(Key_Eject);
          INIT_KEY(Key_ScreenSaver);
          INIT_KEY(Key_WWW);
          INIT_KEY(Key_Memo);
          INIT_KEY(Key_LightBulb);
          INIT_KEY(Key_Shop);
          INIT_KEY(Key_History);
          INIT_KEY(Key_AddFavorite);
          INIT_KEY(Key_HotLinks);
          INIT_KEY(Key_BrightnessAdjust);
          INIT_KEY(Key_Finance);
          INIT_KEY(Key_Community);
          INIT_KEY(Key_AudioRewind);
          INIT_KEY(Key_BackForward);
          INIT_KEY(Key_ApplicationLeft);
          INIT_KEY(Key_ApplicationRight);
          INIT_KEY(Key_Book);
          INIT_KEY(Key_CD);
          INIT_KEY(Key_Calculator);
          INIT_KEY(Key_ToDoList);
          INIT_KEY(Key_ClearGrab);
          INIT_KEY(Key_Close);
          INIT_KEY(Key_Copy);
          INIT_KEY(Key_Cut);
          INIT_KEY(Key_Display);
          INIT_KEY(Key_DOS);
          INIT_KEY(Key_Documents);
          INIT_KEY(Key_Excel);
          INIT_KEY(Key_Explorer);
          INIT_KEY(Key_Game);
          INIT_KEY(Key_Go);
          INIT_KEY(Key_iTouch);
          INIT_KEY(Key_LogOff);
          INIT_KEY(Key_Market);
          INIT_KEY(Key_Meeting);
          INIT_KEY(Key_MenuKB);
          INIT_KEY(Key_MenuPB);
          INIT_KEY(Key_MySites);
          INIT_KEY(Key_News);
          INIT_KEY(Key_OfficeHome);
          INIT_KEY(Key_Option);
          INIT_KEY(Key_Paste);
          INIT_KEY(Key_Phone);
          INIT_KEY(Key_Calendar);
          INIT_KEY(Key_Reply);
          INIT_KEY(Key_Reload);
          INIT_KEY(Key_RotateWindows);
          INIT_KEY(Key_RotationPB);
          INIT_KEY(Key_RotationKB);
          INIT_KEY(Key_Save);
          INIT_KEY(Key_Send);
          INIT_KEY(Key_Spell);
          INIT_KEY(Key_SplitScreen);
          INIT_KEY(Key_Support);
          INIT_KEY(Key_TaskPane);
          INIT_KEY(Key_Terminal);
          INIT_KEY(Key_Tools);
          INIT_KEY(Key_Travel);
          INIT_KEY(Key_Video);
          INIT_KEY(Key_Word);
          INIT_KEY(Key_Xfer);
          INIT_KEY(Key_ZoomIn);
          INIT_KEY(Key_ZoomOut);
          INIT_KEY(Key_Away);
          INIT_KEY(Key_Messenger);
          INIT_KEY(Key_WebCam);
          INIT_KEY(Key_MailForward);
          INIT_KEY(Key_Pictures);
          INIT_KEY(Key_Music);
          INIT_KEY(Key_Battery);
          INIT_KEY(Key_Bluetooth);
          INIT_KEY(Key_WLAN);
          INIT_KEY(Key_UWB);
          INIT_KEY(Key_AudioForward);
          INIT_KEY(Key_AudioRepeat);
          INIT_KEY(Key_AudioRandomPlay);
          INIT_KEY(Key_Subtitle);
          INIT_KEY(Key_AudioCycleTrack);
          INIT_KEY(Key_Time);
          INIT_KEY(Key_Hibernate);
          INIT_KEY(Key_View);
          INIT_KEY(Key_TopMenu);
          INIT_KEY(Key_PowerDown);
          INIT_KEY(Key_Suspend);
          INIT_KEY(Key_ContrastAdjust);
          INIT_KEY(Key_MediaLast);
          INIT_KEY(Key_unknown);
          INIT_KEY(Key_Call);
          INIT_KEY(Key_Context1);
          INIT_KEY(Key_Context2);
          INIT_KEY(Key_Context3);
          INIT_KEY(Key_Context4);
          INIT_KEY(Key_Flip);
          INIT_KEY(Key_Hangup);
          INIT_KEY(Key_No);
          INIT_KEY(Key_Select);
          INIT_KEY(Key_Yes);
          INIT_KEY(Key_Execute);
          INIT_KEY(Key_Printer);
          INIT_KEY(Key_Play);
          INIT_KEY(Key_Sleep);
          INIT_KEY(Key_Zoom);
          INIT_KEY(Key_Cancel);
        #undef INIT_KEY
        }

        SceniXQGLWidget::~SceniXQGLWidget()
        {
        }

        void SceniXQGLWidget::initializeGL()
        {
        }

        void SceniXQGLWidget::resizeGL(int width, int height)
        {
        }

        void SceniXQGLWidget::paintGL()
        {
        }

        void SceniXQGLWidget::resizeEvent(QResizeEvent *event)
        {
          QWidget::resizeEvent( event );
          m_glWidget->resize( event->size() );
        }

        const dp::gl::RenderContextSharedPtr & SceniXQGLWidget::getRenderContext() const
        {
          return m_glWidget->getRenderContext();
        }

        const dp::gl::RenderTargetSharedPtr & SceniXQGLWidget::getRenderTarget() const
        {
          return m_glWidget->getRenderTarget();
        }

        /**************************/
        /* SceniXQGLWidgetPrivate */
        /**************************/
        SceniXQGLWidget::SceniXQGLWidgetPrivate::SceniXQGLWidgetPrivate( SceniXQGLWidget *parent, const dp::gl::RenderContextFormat &format, SceniXQGLWidgetPrivate *shareWidget )
          : QWidget( parent )
          , m_renderTarget( 0 )
          , m_initialized( false )
          , m_shareWidget( shareWidget )
          , m_format( format )
        {
          setAttribute( Qt::WA_NativeWindow );
          setAttribute( Qt::WA_PaintOnScreen ); // don't let qt paint anything on screen

#if defined(DP_OS_WINDOWS)
          dp::gl::RenderContextSharedPtr renderContextGL = dp::gl::RenderContext::create( dp::gl::RenderContext::FromHWND( (HWND)winId(), &m_format, m_shareWidget ? m_shareWidget->getRenderContext() : dp::gl::RenderContextSharedPtr() ) );
#elif defined(DP_OS_LINUX)
          // TODO support format
          dp::gl::RenderContextSharedPtr renderContextGL = dp::gl::RenderContext::create( dp::gl::RenderContext::FromDrawable( QX11Info::display(), QX11Info::appScreen(), winId(), m_shareWidget ? m_shareWidget->getRenderContext() : dp::gl::RenderContextSharedPtr() ) );
#endif
          m_renderTarget = dp::gl::RenderTargetFB::create( renderContextGL );
        }

        SceniXQGLWidget::SceniXQGLWidgetPrivate::~SceniXQGLWidgetPrivate()
        {
          dp::gl::RenderContextSharedPtr renderContextGL = m_renderTarget->getRenderContext();
          m_renderTarget.reset();

          if ( dp::gl::RenderContext::getCurrentRenderContext() == renderContextGL )
          {
            renderContextGL->makeNoncurrent();
          }
        }

        QPaintEngine *SceniXQGLWidget::SceniXQGLWidgetPrivate::paintEngine() const
        {
          return 0;
        }

        const dp::gl::RenderContextSharedPtr & SceniXQGLWidget::SceniXQGLWidgetPrivate::getRenderContext() const
        {
          return m_renderTarget->getRenderContext();
        }

        const dp::gl::RenderTargetSharedPtr & SceniXQGLWidget::SceniXQGLWidgetPrivate::getRenderTarget() const
        {
          return m_renderTarget;
        }

        void SceniXQGLWidget::SceniXQGLWidgetPrivate::initialize()
        {
          SceniXQGLWidget *scenixParent = dynamic_cast<SceniXQGLWidget*>(parentWidget());
          if (scenixParent)
          {
            scenixParent->initializeGL();
          }
        }

        void SceniXQGLWidget::SceniXQGLWidgetPrivate::resizeEvent( QResizeEvent * resizeEvent )
        {
          if (!m_renderTarget) {
            return;
          }

          m_renderTarget->setSize( resizeEvent->size().width(), resizeEvent->size().height() );

          if ( !m_initialized )
          {
            initialize();
            m_initialized = true;
          }

          SceniXQGLWidget *scenixParent = dynamic_cast<SceniXQGLWidget*>(parentWidget());
          if (scenixParent)
          {
            scenixParent->resizeGL( resizeEvent->size().width(), resizeEvent->size().height() );
          }
        }

        void SceniXQGLWidget::SceniXQGLWidgetPrivate::paintEvent( QPaintEvent * event )
        {
          SceniXQGLWidget *scenixParent = dynamic_cast<SceniXQGLWidget*>(parentWidget());
          if (scenixParent)
          {
            m_renderTarget->getRenderContext()->makeCurrent();
            scenixParent->paintGL();
            m_renderTarget->getRenderContext()->swap();
          }
        }

        bool SceniXQGLWidget::event( QEvent *event )
        {
          switch ( event->type() )
          {
          case QEvent::UpdateRequest:
            m_glWidget->update();
          break;
          case QEvent::MouseTrackingChange:
            m_glWidget->setMouseTracking( hasMouseTracking() );
          break;
          }

          return QWidget::event( event );
        }


        QPaintEngine *SceniXQGLWidget::paintEngine() const
        {
          return 0;
        }

        bool SceniXQGLWidget::setFormat( const dp::gl::RenderContextFormat &format )
        {
          bool valid = true;

          if ( format != getFormat() )
          {
            valid = format.isAvailable();
            if (valid)
            {
              // create a new gl viewport with the new format
              SceniXQGLWidgetPrivate *oldWidget = m_glWidget;

              m_glWidget = new SceniXQGLWidgetPrivate( this, format, oldWidget );
              m_glWidget->move(0, 0);
              m_glWidget->resize( width(), height() );
              m_glWidget->show();
              m_glWidget->setMouseTracking( hasMouseTracking() );

              // notify derived classes that the format has changed and a new RenderTarget is available.
              onRenderTargetChanged( oldWidget->getRenderTarget(), m_glWidget->getRenderTarget() );

              // delete the old viewport
              delete oldWidget;
            }
          }
          return valid;
        }

        dp::gl::RenderContextFormat SceniXQGLWidget::getFormat() const
        {
          return getRenderContext()->getFormat();
        }


        // HID
        void SceniXQGLWidget::keyPressEvent( QKeyEvent *event )
        {
          if ( !event->isAutoRepeat() )
          {
            KeyInfoMap::iterator it = m_keyProperties.find( static_cast<Qt::Key>(event->key()) );
            if ( it != m_keyProperties.end() )
            {
              const KeyInfo &info = it->second;
              if ( this->*info.member == false )
              {
                this->*info.member = true;
                hidNotify( info.propertyId );
              }
            }
          }
          event->ignore();
        }

        void SceniXQGLWidget::keyReleaseEvent( QKeyEvent *event )
        {
          if ( !event->isAutoRepeat() )
          {
            KeyInfoMap::iterator it = m_keyProperties.find( static_cast<Qt::Key>(event->key()) );
            if ( it != m_keyProperties.end() )
            {
              const KeyInfo &info = it->second;
              if ( this->*info.member == true )
              {
                this->*info.member = false;
                hidNotify( info.propertyId );
              }
            }
          }
          event->ignore();
        }

        PropertyId SceniXQGLWidget::getQtMouseButtonProperty( Qt::MouseButton button ) const
        {
          switch ( button )
          {
          case Qt::LeftButton:
            return PID_Mouse_Left;
          case Qt::RightButton:
            return PID_Mouse_Right;
          case Qt::MidButton:
            return PID_Mouse_Middle;
          default:
            return 0;
          }
        }

        void SceniXQGLWidget::mousePressEvent( QMouseEvent *event )
        {
          PropertyId propButton = getQtMouseButtonProperty( event->button() );
          if ( propButton )
          {
            if ( getValue<bool>( propButton ) == false )
            {
              setValue<bool>( propButton, true );
              hidNotify( propButton );
            }
          }
          event->ignore();
        }

        void SceniXQGLWidget::mouseReleaseEvent( QMouseEvent *event )
        {
          PropertyId propButton = getQtMouseButtonProperty( event->button() );
          if ( propButton )
          {
            if ( getValue<bool>( propButton ) == true )
            {
              setValue<bool>( propButton, false );
              hidNotify( propButton );
            }
          }
          event->ignore();
        }

        void SceniXQGLWidget::mouseMoveEvent( QMouseEvent *event )
        {
          QPoint pos = event->pos();
          if ( m_propMouse_Position[0] != pos.x() || m_propMouse_Position[1] != pos.y() )
          {
            m_propMouse_Position[0] = pos.x();
            m_propMouse_Position[1] = pos.y();
            hidNotify( PID_Mouse_Position );
          }
          event->ignore();
        }

        void SceniXQGLWidget::wheelEvent( QWheelEvent *event )
        {
          if ( event->orientation() == Qt::Vertical )
          {
            // accumulate here so that wheel is absolute, like position
            m_propMouse_Wheel += event->delta();
            hidNotify( PID_Mouse_Wheel );
          }
          event->ignore();
        }

        void SceniXQGLWidget::hidNotify( PropertyId property )
        {
          notify( dp::util::Reflection::PropertyEvent( this, property ) );
        }

        unsigned int SceniXQGLWidget::getNumberOfAxes() const
        {
          return 0;
        }

        std::string SceniXQGLWidget::getAxisName( unsigned int axis ) const
        {
          return "";
        }

        unsigned int SceniXQGLWidget::getNumberOfKeys() const
        {
          return dp::checked_cast<unsigned int>(m_keyInfos.size());
        }

        std::string SceniXQGLWidget::getKeyName( unsigned int key ) const
        {
          DP_ASSERT( key < m_keyInfos.size() );

          return m_keyInfos[key].name;
        }

        void SceniXQGLWidget::triggerRepaint()
        {
          update();
        }

        void SceniXQGLWidget::onRenderTargetChanged( const dp::gl::RenderTargetSharedPtr &oldTarget, const dp::gl::RenderTargetSharedPtr &newTarget)
        {
        }

      } // namespace qt5
    } // namespace ui
  } // namespace sg
} // namespace dp
