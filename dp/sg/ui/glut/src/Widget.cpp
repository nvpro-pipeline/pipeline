// Copyright NVIDIA Corporation 2012
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


#include <dp/sg/ui/glut/Widget.h>
#include <dp/gl/RenderTargetFB.h>
#include <GL/freeglut.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace glut
      {

        // mouse
        DEFINE_STATIC_PROPERTY( Widget, Mouse_Left );
        DEFINE_STATIC_PROPERTY( Widget, Mouse_Middle );
        DEFINE_STATIC_PROPERTY( Widget, Mouse_Right );
        DEFINE_STATIC_PROPERTY( Widget, Mouse_Position );
        DEFINE_STATIC_PROPERTY( Widget, Mouse_Wheel );

        DEFINE_STATIC_PROPERTY( Widget, Key_Escape );
        DEFINE_STATIC_PROPERTY( Widget, Key_Tab );
        DEFINE_STATIC_PROPERTY( Widget, Key_Backtab );
        DEFINE_STATIC_PROPERTY( Widget, Key_Backspace );
        DEFINE_STATIC_PROPERTY( Widget, Key_Return );
        DEFINE_STATIC_PROPERTY( Widget, Key_Enter );
        DEFINE_STATIC_PROPERTY( Widget, Key_Insert );
        DEFINE_STATIC_PROPERTY( Widget, Key_Delete );
        DEFINE_STATIC_PROPERTY( Widget, Key_Pause );
        DEFINE_STATIC_PROPERTY( Widget, Key_Print );
        DEFINE_STATIC_PROPERTY( Widget, Key_SysReq );
        DEFINE_STATIC_PROPERTY( Widget, Key_Clear );
        DEFINE_STATIC_PROPERTY( Widget, Key_Home );
        DEFINE_STATIC_PROPERTY( Widget, Key_End );
        DEFINE_STATIC_PROPERTY( Widget, Key_Left );
        DEFINE_STATIC_PROPERTY( Widget, Key_Up );
        DEFINE_STATIC_PROPERTY( Widget, Key_Right );
        DEFINE_STATIC_PROPERTY( Widget, Key_Down );
        DEFINE_STATIC_PROPERTY( Widget, Key_PageUp );
        DEFINE_STATIC_PROPERTY( Widget, Key_PageDown );
        DEFINE_STATIC_PROPERTY( Widget, Key_Shift );
        DEFINE_STATIC_PROPERTY( Widget, Key_Control );
        DEFINE_STATIC_PROPERTY( Widget, Key_Meta );
        DEFINE_STATIC_PROPERTY( Widget, Key_Alt );
        DEFINE_STATIC_PROPERTY( Widget, Key_AltGr );
        DEFINE_STATIC_PROPERTY( Widget, Key_CapsLock );
        DEFINE_STATIC_PROPERTY( Widget, Key_NumLock );
        DEFINE_STATIC_PROPERTY( Widget, Key_ScrollLock );
        DEFINE_STATIC_PROPERTY( Widget, Key_F1 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F2 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F3 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F4 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F5 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F6 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F7 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F8 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F9 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F10 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F11 );
        DEFINE_STATIC_PROPERTY( Widget, Key_F12 );
        DEFINE_STATIC_PROPERTY( Widget, Key_Super_L );
        DEFINE_STATIC_PROPERTY( Widget, Key_Super_R );
        DEFINE_STATIC_PROPERTY( Widget, Key_Menu );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hyper_L );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hyper_R );
        DEFINE_STATIC_PROPERTY( Widget, Key_Help );
        DEFINE_STATIC_PROPERTY( Widget, Key_Direction_L );
        DEFINE_STATIC_PROPERTY( Widget, Key_Direction_R );
        DEFINE_STATIC_PROPERTY( Widget, Key_Space );
        DEFINE_STATIC_PROPERTY( Widget, Key_Any );
        DEFINE_STATIC_PROPERTY( Widget, Key_Exclam );
        DEFINE_STATIC_PROPERTY( Widget, Key_QuoteDbl );
        DEFINE_STATIC_PROPERTY( Widget, Key_NumberSign );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dollar );
        DEFINE_STATIC_PROPERTY( Widget, Key_Percent );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ampersand );
        DEFINE_STATIC_PROPERTY( Widget, Key_Apostrophe );
        DEFINE_STATIC_PROPERTY( Widget, Key_ParenLeft );
        DEFINE_STATIC_PROPERTY( Widget, Key_ParenRight );
        DEFINE_STATIC_PROPERTY( Widget, Key_Asterisk );
        DEFINE_STATIC_PROPERTY( Widget, Key_Plus );
        DEFINE_STATIC_PROPERTY( Widget, Key_Comma );
        DEFINE_STATIC_PROPERTY( Widget, Key_Minus );
        DEFINE_STATIC_PROPERTY( Widget, Key_Period );
        DEFINE_STATIC_PROPERTY( Widget, Key_Slash );
        DEFINE_STATIC_PROPERTY( Widget, Key_0 );
        DEFINE_STATIC_PROPERTY( Widget, Key_1 );
        DEFINE_STATIC_PROPERTY( Widget, Key_2 );
        DEFINE_STATIC_PROPERTY( Widget, Key_3 );
        DEFINE_STATIC_PROPERTY( Widget, Key_4 );
        DEFINE_STATIC_PROPERTY( Widget, Key_5 );
        DEFINE_STATIC_PROPERTY( Widget, Key_6 );
        DEFINE_STATIC_PROPERTY( Widget, Key_7 );
        DEFINE_STATIC_PROPERTY( Widget, Key_8 );
        DEFINE_STATIC_PROPERTY( Widget, Key_9 );
        DEFINE_STATIC_PROPERTY( Widget, Key_Colon );
        DEFINE_STATIC_PROPERTY( Widget, Key_Semicolon );
        DEFINE_STATIC_PROPERTY( Widget, Key_Less );
        DEFINE_STATIC_PROPERTY( Widget, Key_Equal );
        DEFINE_STATIC_PROPERTY( Widget, Key_Greater );
        DEFINE_STATIC_PROPERTY( Widget, Key_Question );
        DEFINE_STATIC_PROPERTY( Widget, Key_At );
        DEFINE_STATIC_PROPERTY( Widget, Key_A );
        DEFINE_STATIC_PROPERTY( Widget, Key_B );
        DEFINE_STATIC_PROPERTY( Widget, Key_C );
        DEFINE_STATIC_PROPERTY( Widget, Key_D );
        DEFINE_STATIC_PROPERTY( Widget, Key_E );
        DEFINE_STATIC_PROPERTY( Widget, Key_F );
        DEFINE_STATIC_PROPERTY( Widget, Key_G );
        DEFINE_STATIC_PROPERTY( Widget, Key_H );
        DEFINE_STATIC_PROPERTY( Widget, Key_I );
        DEFINE_STATIC_PROPERTY( Widget, Key_J );
        DEFINE_STATIC_PROPERTY( Widget, Key_K );
        DEFINE_STATIC_PROPERTY( Widget, Key_L );
        DEFINE_STATIC_PROPERTY( Widget, Key_M );
        DEFINE_STATIC_PROPERTY( Widget, Key_N );
        DEFINE_STATIC_PROPERTY( Widget, Key_O );
        DEFINE_STATIC_PROPERTY( Widget, Key_P );
        DEFINE_STATIC_PROPERTY( Widget, Key_Q );
        DEFINE_STATIC_PROPERTY( Widget, Key_R );
        DEFINE_STATIC_PROPERTY( Widget, Key_S );
        DEFINE_STATIC_PROPERTY( Widget, Key_T );
        DEFINE_STATIC_PROPERTY( Widget, Key_U );
        DEFINE_STATIC_PROPERTY( Widget, Key_V );
        DEFINE_STATIC_PROPERTY( Widget, Key_W );
        DEFINE_STATIC_PROPERTY( Widget, Key_X );
        DEFINE_STATIC_PROPERTY( Widget, Key_Y );
        DEFINE_STATIC_PROPERTY( Widget, Key_Z );
        DEFINE_STATIC_PROPERTY( Widget, Key_BracketLeft );
        DEFINE_STATIC_PROPERTY( Widget, Key_Backslash );
        DEFINE_STATIC_PROPERTY( Widget, Key_BracketRight );
        DEFINE_STATIC_PROPERTY( Widget, Key_AsciiCircum );
        DEFINE_STATIC_PROPERTY( Widget, Key_Underscore );
        DEFINE_STATIC_PROPERTY( Widget, Key_QuoteLeft );
        DEFINE_STATIC_PROPERTY( Widget, Key_BraceLeft );
        DEFINE_STATIC_PROPERTY( Widget, Key_Bar );
        DEFINE_STATIC_PROPERTY( Widget, Key_BraceRight );
        DEFINE_STATIC_PROPERTY( Widget, Key_AsciiTilde );
        DEFINE_STATIC_PROPERTY( Widget, Key_nobreakspace );
        DEFINE_STATIC_PROPERTY( Widget, Key_exclamdown );
        DEFINE_STATIC_PROPERTY( Widget, Key_cent );
        DEFINE_STATIC_PROPERTY( Widget, Key_sterling );
        DEFINE_STATIC_PROPERTY( Widget, Key_currency );
        DEFINE_STATIC_PROPERTY( Widget, Key_yen );
        DEFINE_STATIC_PROPERTY( Widget, Key_brokenbar );
        DEFINE_STATIC_PROPERTY( Widget, Key_section );
        DEFINE_STATIC_PROPERTY( Widget, Key_diaeresis );
        DEFINE_STATIC_PROPERTY( Widget, Key_copyright );
        DEFINE_STATIC_PROPERTY( Widget, Key_ordfeminine );
        DEFINE_STATIC_PROPERTY( Widget, Key_guillemotleft );
        DEFINE_STATIC_PROPERTY( Widget, Key_notsign );
        DEFINE_STATIC_PROPERTY( Widget, Key_hyphen );
        DEFINE_STATIC_PROPERTY( Widget, Key_registered );
        DEFINE_STATIC_PROPERTY( Widget, Key_macron );
        DEFINE_STATIC_PROPERTY( Widget, Key_degree );
        DEFINE_STATIC_PROPERTY( Widget, Key_plusminus );
        DEFINE_STATIC_PROPERTY( Widget, Key_twosuperior );
        DEFINE_STATIC_PROPERTY( Widget, Key_threesuperior );
        DEFINE_STATIC_PROPERTY( Widget, Key_acute );
        DEFINE_STATIC_PROPERTY( Widget, Key_mu );
        DEFINE_STATIC_PROPERTY( Widget, Key_paragraph );
        DEFINE_STATIC_PROPERTY( Widget, Key_periodcentered );
        DEFINE_STATIC_PROPERTY( Widget, Key_cedilla );
        DEFINE_STATIC_PROPERTY( Widget, Key_onesuperior );
        DEFINE_STATIC_PROPERTY( Widget, Key_masculine );
        DEFINE_STATIC_PROPERTY( Widget, Key_guillemotright );
        DEFINE_STATIC_PROPERTY( Widget, Key_onequarter );
        DEFINE_STATIC_PROPERTY( Widget, Key_onehalf );
        DEFINE_STATIC_PROPERTY( Widget, Key_threequarters );
        DEFINE_STATIC_PROPERTY( Widget, Key_questiondown );
        DEFINE_STATIC_PROPERTY( Widget, Key_Agrave );
        DEFINE_STATIC_PROPERTY( Widget, Key_Aacute );
        DEFINE_STATIC_PROPERTY( Widget, Key_Acircumflex );
        DEFINE_STATIC_PROPERTY( Widget, Key_Atilde );
        DEFINE_STATIC_PROPERTY( Widget, Key_Adiaeresis );
        DEFINE_STATIC_PROPERTY( Widget, Key_Aring );
        DEFINE_STATIC_PROPERTY( Widget, Key_AE );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ccedilla );
        DEFINE_STATIC_PROPERTY( Widget, Key_Egrave );
        DEFINE_STATIC_PROPERTY( Widget, Key_Eacute );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ecircumflex );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ediaeresis );
        DEFINE_STATIC_PROPERTY( Widget, Key_Igrave );
        DEFINE_STATIC_PROPERTY( Widget, Key_Iacute );
        DEFINE_STATIC_PROPERTY( Widget, Key_Icircumflex );
        DEFINE_STATIC_PROPERTY( Widget, Key_Idiaeresis );
        DEFINE_STATIC_PROPERTY( Widget, Key_ETH );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ntilde );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ograve );
        DEFINE_STATIC_PROPERTY( Widget, Key_Oacute );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ocircumflex );
        DEFINE_STATIC_PROPERTY( Widget, Key_Otilde );
        DEFINE_STATIC_PROPERTY( Widget, Key_Odiaeresis );
        DEFINE_STATIC_PROPERTY( Widget, Key_multiply );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ooblique );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ugrave );
        DEFINE_STATIC_PROPERTY( Widget, Key_Uacute );
        DEFINE_STATIC_PROPERTY( Widget, Key_Ucircumflex );
        DEFINE_STATIC_PROPERTY( Widget, Key_Udiaeresis );
        DEFINE_STATIC_PROPERTY( Widget, Key_Yacute );
        DEFINE_STATIC_PROPERTY( Widget, Key_THORN );
        DEFINE_STATIC_PROPERTY( Widget, Key_ssharp );
        DEFINE_STATIC_PROPERTY( Widget, Key_division );
        DEFINE_STATIC_PROPERTY( Widget, Key_ydiaeresis );
        DEFINE_STATIC_PROPERTY( Widget, Key_Multi_key );
        DEFINE_STATIC_PROPERTY( Widget, Key_Codeinput );
        DEFINE_STATIC_PROPERTY( Widget, Key_SingleCandidate );
        DEFINE_STATIC_PROPERTY( Widget, Key_MultipleCandidate );
        DEFINE_STATIC_PROPERTY( Widget, Key_PreviousCandidate );
        DEFINE_STATIC_PROPERTY( Widget, Key_Mode_switch );
        DEFINE_STATIC_PROPERTY( Widget, Key_Kanji );
        DEFINE_STATIC_PROPERTY( Widget, Key_Muhenkan );
        DEFINE_STATIC_PROPERTY( Widget, Key_Henkan );
        DEFINE_STATIC_PROPERTY( Widget, Key_Romaji );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hiragana );
        DEFINE_STATIC_PROPERTY( Widget, Key_Katakana );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hiragana_Katakana );
        DEFINE_STATIC_PROPERTY( Widget, Key_Zenkaku );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hankaku );
        DEFINE_STATIC_PROPERTY( Widget, Key_Zenkaku_Hankaku );
        DEFINE_STATIC_PROPERTY( Widget, Key_Touroku );
        DEFINE_STATIC_PROPERTY( Widget, Key_Massyo );
        DEFINE_STATIC_PROPERTY( Widget, Key_Kana_Lock );
        DEFINE_STATIC_PROPERTY( Widget, Key_Kana_Shift );
        DEFINE_STATIC_PROPERTY( Widget, Key_Eisu_Shift );
        DEFINE_STATIC_PROPERTY( Widget, Key_Eisu_toggle );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_Start );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_End );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_Hanja );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_Jamo );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_Romaja );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_Jeonja );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_Banja );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_PreHanja );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_PostHanja );
        DEFINE_STATIC_PROPERTY( Widget, Key_Hangul_Special );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Grave );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Acute );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Circumflex );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Tilde );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Macron );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Breve );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Abovedot );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Diaeresis );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Abovering );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Doubleacute );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Caron );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Cedilla );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Ogonek );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Iota );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Voiced_Sound );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Semivoiced_Sound );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Belowdot );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Hook );
        DEFINE_STATIC_PROPERTY( Widget, Key_Dead_Horn );

        BEGIN_REFLECTION_INFO( Widget )
          DERIVE_STATIC_PROPERTIES( Widget, HumanInterfaceDevice );

        // mouse
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Mouse_Left, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Mouse_Middle, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Mouse_Right, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Mouse_Position, dp::math::Vec2i, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Mouse_Wheel, int, SEMANTIC_VALUE );

        // keyboard
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Escape, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Tab, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Backtab, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Backspace, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Return, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Enter, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Insert, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Delete, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Pause, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Print, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_SysReq, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Clear, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Home, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_End, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Left, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Up, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Right, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Down, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_PageUp, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_PageDown, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Shift, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Control, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Meta, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Alt, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_AltGr, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_CapsLock, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_NumLock, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_ScrollLock, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F1, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F2, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F3, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F4, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F5, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F6, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F7, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F8, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F9, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F10, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F11, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F12, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Super_L, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Super_R, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Menu, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hyper_L, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hyper_R, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Help, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Direction_L, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Direction_R, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Space, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Any, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Exclam, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_QuoteDbl, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_NumberSign, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dollar, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Percent, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ampersand, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Apostrophe, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_ParenLeft, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_ParenRight, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Asterisk, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Plus, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Comma, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Minus, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Period, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Slash, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_0, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_1, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_2, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_3, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_4, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_5, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_6, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_7, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_8, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_9, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Colon, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Semicolon, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Less, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Equal, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Greater, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Question, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_At, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_A, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_B, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_C, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_D, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_E, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_F, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_G, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_H, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_I, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_J, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_K, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_L, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_M, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_N, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_O, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_P, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Q, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_R, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_S, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_T, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_U, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_V, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_W, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_X, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Y, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Z, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_BracketLeft, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Backslash, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_BracketRight, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_AsciiCircum, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Underscore, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_QuoteLeft, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_BraceLeft, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Bar, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_BraceRight, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_AsciiTilde, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_nobreakspace, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_exclamdown, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_cent, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_sterling, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_currency, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_yen, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_brokenbar, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_section, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_diaeresis, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_copyright, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_ordfeminine, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_guillemotleft, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_notsign, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_hyphen, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_registered, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_macron, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_degree, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_plusminus, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_twosuperior, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_threesuperior, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_acute, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_mu, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_paragraph, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_periodcentered, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_cedilla, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_onesuperior, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_masculine, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_guillemotright, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_onequarter, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_onehalf, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_threequarters, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_questiondown, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Agrave, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Aacute, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Acircumflex, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Atilde, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Adiaeresis, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Aring, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_AE, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ccedilla, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Egrave, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Eacute, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ecircumflex, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ediaeresis, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Igrave, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Iacute, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Icircumflex, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Idiaeresis, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_ETH, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ntilde, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ograve, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Oacute, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ocircumflex, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Otilde, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Odiaeresis, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_multiply, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ooblique, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ugrave, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Uacute, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Ucircumflex, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Udiaeresis, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Yacute, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_THORN, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_ssharp, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_division, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_ydiaeresis, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Multi_key, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Codeinput, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_SingleCandidate, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_MultipleCandidate, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_PreviousCandidate, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Mode_switch, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Kanji, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Muhenkan, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Henkan, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Romaji, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hiragana, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Katakana, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hiragana_Katakana, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Zenkaku, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hankaku, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Zenkaku_Hankaku, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Touroku, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Massyo, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Kana_Lock, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Kana_Shift, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Eisu_Shift, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Eisu_toggle, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_Start, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_End, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_Hanja, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_Jamo, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_Romaja, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_Jeonja, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_Banja, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_PreHanja, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_PostHanja, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Hangul_Special, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Grave, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Acute, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Circumflex, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Tilde, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Macron, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Breve, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Abovedot, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Diaeresis, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Abovering, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Doubleacute, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Caron, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Cedilla, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Ogonek, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Iota, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Voiced_Sound, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Semivoiced_Sound, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Belowdot, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Hook, bool, SEMANTIC_VALUE );
        INIT_STATIC_PROPERTY_RO_MEMBER( Widget, Key_Dead_Horn, bool, SEMANTIC_VALUE );

        END_REFLECTION_INFO

          namespace
        {
          /** \brief Class to temporary bind the given window id **/
          class BindWidget : public boost::noncopyable
          {
          public:
            BindWidget( int windowId )
            {
              m_lastWindowId = glutGetWindow();
              glutSetWindow( windowId );
            }

            ~BindWidget()
            {
              glutSetWindow( m_lastWindowId );
            }
          private:
            int m_lastWindowId;
          };
        } // namespace

        Widget::Widget( int x, int y )
          : m_showFPS( false )
          , m_fps( 0.0f )
          , m_frameCount( 0 )
          , m_continuousUpdate( false )
          , m_width( 640 )
          , m_height( 480 )
        {
          memset( m_asciiMap, 0, sizeof( m_asciiMap ) );
#define INIT_KEY(key) m_prop##key = false;
#define INIT_ASCII_KEY(key, ascii) m_prop##key = false; m_asciiMap[ascii] = PID_##key;
#define INIT_SPECIAL_KEY(key, special) m_prop##key = false; m_specialPropertyMap[special] = PID_##key;

          INIT_ASCII_KEY(Key_Escape, 27);
          INIT_ASCII_KEY(Key_Tab, 9);
          INIT_KEY(Key_Backtab);
          INIT_KEY(Key_Backspace);
          INIT_ASCII_KEY(Key_Return, 13);
          INIT_ASCII_KEY(Key_Enter, 13);
          INIT_SPECIAL_KEY(Key_Insert, GLUT_KEY_INSERT);
          INIT_SPECIAL_KEY(Key_Delete, GLUT_KEY_DELETE);
          INIT_KEY(Key_Pause);
          INIT_KEY(Key_Print);
          INIT_KEY(Key_SysReq);
          INIT_KEY(Key_Clear);
          INIT_SPECIAL_KEY(Key_Home, GLUT_KEY_HOME);
          INIT_SPECIAL_KEY(Key_End, GLUT_KEY_END);
          INIT_SPECIAL_KEY(Key_Left, GLUT_KEY_LEFT);
          INIT_SPECIAL_KEY(Key_Up, GLUT_KEY_UP);
          INIT_SPECIAL_KEY(Key_Right, GLUT_KEY_RIGHT);
          INIT_SPECIAL_KEY(Key_Down, GLUT_KEY_DOWN);
          INIT_SPECIAL_KEY(Key_PageUp, GLUT_KEY_PAGE_DOWN);
          INIT_SPECIAL_KEY(Key_PageDown, GLUT_KEY_PAGE_UP);
#if HAVE_FREEGLUT_2_8
          INIT_SPECIAL_KEY(Key_Shift, GLUT_KEY_SHIFT_L);
          INIT_SPECIAL_KEY(Key_Control, GLUT_KEY_CTRL_L );
#else
          INIT_KEY(Key_Shift);
          INIT_KEY(Key_Control);
#endif
          INIT_KEY(Key_Meta);

#if HAVE_FREEGLUT_2_8
          INIT_SPECIAL_KEY(Key_Alt, GLUT_KEY_ALT_L);
#else
          INIT_KEY(Key_Alt);
#endif
          INIT_KEY(Key_AltGr);
          INIT_KEY(Key_CapsLock);
          INIT_KEY(Key_NumLock);
          INIT_KEY(Key_ScrollLock);
          INIT_SPECIAL_KEY(Key_F1, GLUT_KEY_F1);
          INIT_SPECIAL_KEY(Key_F2, GLUT_KEY_F2);
          INIT_SPECIAL_KEY(Key_F3, GLUT_KEY_F3);
          INIT_SPECIAL_KEY(Key_F4, GLUT_KEY_F4);
          INIT_SPECIAL_KEY(Key_F5, GLUT_KEY_F5);
          INIT_SPECIAL_KEY(Key_F6, GLUT_KEY_F6);
          INIT_SPECIAL_KEY(Key_F7, GLUT_KEY_F7);
          INIT_SPECIAL_KEY(Key_F8, GLUT_KEY_F8);
          INIT_SPECIAL_KEY(Key_F9, GLUT_KEY_F9);
          INIT_SPECIAL_KEY(Key_F10, GLUT_KEY_F10);
          INIT_SPECIAL_KEY(Key_F11, GLUT_KEY_F11);
          INIT_SPECIAL_KEY(Key_F12, GLUT_KEY_F12);
          INIT_KEY(Key_Super_L);
          INIT_KEY(Key_Super_R);
          INIT_KEY(Key_Menu);
          INIT_KEY(Key_Hyper_L);
          INIT_KEY(Key_Hyper_R);
          INIT_KEY(Key_Help);
          INIT_KEY(Key_Direction_L);
          INIT_KEY(Key_Direction_R);
          INIT_ASCII_KEY(Key_Space, ' ');
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
          INIT_ASCII_KEY(Key_Asterisk, '*');
          INIT_ASCII_KEY(Key_Plus, '+');
          INIT_ASCII_KEY(Key_Comma, ',');
          INIT_ASCII_KEY(Key_Minus, '-');
          INIT_ASCII_KEY(Key_Period, '.');
          INIT_ASCII_KEY(Key_Slash, '/');
          INIT_ASCII_KEY(Key_0, '0');
          INIT_ASCII_KEY(Key_1, '1');
          INIT_ASCII_KEY(Key_2, '2');
          INIT_ASCII_KEY(Key_3, '3');
          INIT_ASCII_KEY(Key_4, '4');
          INIT_ASCII_KEY(Key_5, '5');
          INIT_ASCII_KEY(Key_6, '6');
          INIT_ASCII_KEY(Key_7, '7');
          INIT_ASCII_KEY(Key_8, '8');
          INIT_ASCII_KEY(Key_9, '9');
          INIT_ASCII_KEY(Key_Colon, ',');
          INIT_ASCII_KEY(Key_Semicolon, ';');
          INIT_ASCII_KEY(Key_Less, '<');
          INIT_ASCII_KEY(Key_Equal, '=');
          INIT_ASCII_KEY(Key_Greater, '>');
          INIT_ASCII_KEY(Key_Question, '?');
          INIT_ASCII_KEY(Key_At, '@');
          INIT_ASCII_KEY(Key_A, 'a');
          INIT_ASCII_KEY(Key_B, 'b');
          INIT_ASCII_KEY(Key_C, 'c');
          INIT_ASCII_KEY(Key_D, 'd');
          INIT_ASCII_KEY(Key_E, 'e');
          INIT_ASCII_KEY(Key_F, 'f');
          INIT_ASCII_KEY(Key_G, 'g');
          INIT_ASCII_KEY(Key_H, 'h');
          INIT_ASCII_KEY(Key_I, 'i');
          INIT_ASCII_KEY(Key_J, 'j');
          INIT_ASCII_KEY(Key_K, 'k');
          INIT_ASCII_KEY(Key_L, 'l');
          INIT_ASCII_KEY(Key_M, 'm');
          INIT_ASCII_KEY(Key_N, 'n');
          INIT_ASCII_KEY(Key_O, 'o');
          INIT_ASCII_KEY(Key_P, 'p');
          INIT_ASCII_KEY(Key_Q, 'q');
          INIT_ASCII_KEY(Key_R, 'r');
          INIT_ASCII_KEY(Key_S, 's');
          INIT_ASCII_KEY(Key_T, 't');
          INIT_ASCII_KEY(Key_U, 'u');
          INIT_ASCII_KEY(Key_V, 'v');
          INIT_ASCII_KEY(Key_W, 'w');
          INIT_ASCII_KEY(Key_X, 'x');
          INIT_ASCII_KEY(Key_Y, 'y');
          INIT_ASCII_KEY(Key_Z, 'z');
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
#undef INIT_KEY

          m_updateTimer.start();

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


          glutInitWindowPosition( x, y );
          glutInitWindowSize( dp::checked_cast<int>(m_width), dp::checked_cast<int>(m_height) );

          glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA );
          m_windowId = glutCreateWindow( "SceniX minimal GLUT example" );
          glutSetWindowData( this );
          glutDisplayFunc( &renderFunction );
          glutReshapeFunc( &reshapeFunction );
          glutCloseFunc( &closeFunction );
          glutMouseFunc( &mouseButtonFunction );
          glutMotionFunc( &mouseMotionFunction );
          glutMouseWheelFunc( &mouseWheelFunction );
          glutPassiveMotionFunc( &mouseMotionFunction );
          glutKeyboardFunc( &keyboardFunction );
          glutKeyboardUpFunc( &keyboardUpFunction );
          glutSpecialFunc( &keyboardSpecialFunction );
          glutSpecialUpFunc( &keyboardSpecialUpFunction );

          setWindowTitle( "Powered by devtech platform" );

          glewInit();
          m_renderTarget = dp::gl::RenderTargetFB::create( dp::gl::RenderContext::create( dp::gl::RenderContext::Attach() ) );
          m_renderTarget->setClearMask( dp::gl::TBM_COLOR_BUFFER | dp::gl::TBM_DEPTH_BUFFER );
        }

        Widget::~Widget()
        {
          if ( m_windowId )
          {
            glutDestroyWindow( m_windowId );
          }
        }

        Widget* Widget::getCurrentWidget()
        {
          return reinterpret_cast<Widget*>( glutGetWindowData() );
        }

        void Widget::renderFunction()
        {
          getCurrentWidget()->paint();
        }

        void Widget::reshapeFunction( GLint width, GLint height )
        {
          getCurrentWidget()->resize( width, height );
        }

        void Widget::closeFunction()
        {
          getCurrentWidget()->cleanup();
        }

        void Widget::mouseButtonFunction( int button, int state, int x, int y )
        {
#if !defined(HAVE_FREEGLUT_2_8)
          getCurrentWidget()->updateModifiers();
#endif
          getCurrentWidget()->mouseButtonChanged( button, state, x, y );
        }

        void Widget::mouseMotionFunction( int x, int y )
        {
#if !defined(HAVE_FREEGLUT_2_8)
          getCurrentWidget()->updateModifiers();
#endif
          getCurrentWidget()->mousePositionChanged( x, y );
        }

        void Widget::mouseWheelFunction( int wheel, int direction, int x, int y )
        {
#if !defined(HAVE_FREEGLUT_2_8)
          getCurrentWidget()->updateModifiers();
#endif
          getCurrentWidget()->mouseWheelChanged( wheel, direction, x, y );
        }

        void Widget::keyboardFunction( unsigned char key, int x, int y )
        {
#if !defined(HAVE_FREEGLUT_2_8)
          getCurrentWidget()->updateModifiers();
#endif
          getCurrentWidget()->keyPressed( key, x, y );
        }

        void Widget::keyboardUpFunction( unsigned char key, int x, int y )
        {
#if !defined(HAVE_FREEGLUT_2_8)
          getCurrentWidget()->updateModifiers();
#endif
          getCurrentWidget()->keyReleased( key, x, y );
        }

        void Widget::keyboardSpecialFunction( int key, int x, int y )
        {
          getCurrentWidget()->specialKeyPressed( key, x, y );
        }

        void Widget::keyboardSpecialUpFunction( int key, int x, int y )
        {
          getCurrentWidget()->specialKeyReleased( key, x, y );
        }

        void Widget::paint()
        {
          BindWidget bind( m_windowId );
         
          if ( m_showFPS )
          {
            double elapsedSeconds = m_frameRateTimer.getTime();
            ++m_frameCount;
            if ( elapsedSeconds > 1.0 )
            {
              m_fps = double(m_frameCount) / elapsedSeconds;
              updateWindowTitle( m_windowTitle );
              m_frameCount = 0;
              m_frameRateTimer.restart();
            }
          }

          if ( m_continuousUpdate )
          {
            glutPostRedisplay();
          }
        }

        void Widget::resize( GLint width, GLint height )
        {
          m_renderTarget->setSize( width, height );

          m_width = width;
          m_height = height;
          std::ostringstream completeTitle;
          completeTitle << m_windowTitleBase << " (" << m_width << "," << m_height << ")";
          m_windowTitle = completeTitle.str();
          updateWindowTitle( m_windowTitle );
        }

        void Widget::cleanup()
        {
          m_renderTarget.reset();
          m_windowId = 0;
        }

        void Widget::mouseButtonChanged( int button, int state, int x, int y )
        {
          bool pressed = state == GLUT_DOWN;
          dp::util::PropertyId propertyButton = 0;
          switch ( button )
          {
          case GLUT_LEFT_BUTTON:
            propertyButton = PID_Mouse_Left;
            break;
          case GLUT_MIDDLE_BUTTON:
            propertyButton = PID_Mouse_Middle;
            break;
          case GLUT_RIGHT_BUTTON:
            propertyButton = PID_Mouse_Right;
            break;
          }
          if ( propertyButton && getValue<bool>( propertyButton ) != pressed )
          {
            setValue<bool>( propertyButton, pressed );
            hidNotify( propertyButton );
          }
        }

        void Widget::mousePositionChanged( int x, int y )
        {
          if ( m_propMouse_Position[0] != x || m_propMouse_Position[1] != y )
          {
            m_propMouse_Position[0] = x;
            m_propMouse_Position[1] = y;
            hidNotify( PID_Mouse_Position );
          }
        }

        void Widget::mouseWheelChanged( int wheel, int direction, int x, int y )
        {
          if ( wheel == 0 )
          {
            // GLUT supports only +1, -1 for direction. assume a reasonable value to emulate the real stepping difference.
            m_propMouse_Wheel += direction * 120;
            hidNotify( PID_Mouse_Wheel );
          }
        }

        void Widget::keyPressed( unsigned char key, int x, int y )
        {
          dp::util::PropertyId property = m_asciiMap[tolower(key)];
          if ( property && !getValue<bool>(property) )
          {
            setValue<bool>( property, true );
            hidNotify( property );
          }
        }

        void Widget::keyReleased( unsigned char key, int x, int y )
        {
          dp::util::PropertyId property = m_asciiMap[tolower(key)];
          if ( property && getValue<bool>(property) )
          {
            setValue<bool>( property, false );
            hidNotify( property );
          }
        }

        void Widget::specialKeyPressed( int key, int x, int y )
        {
          dp::util::PropertyId property = m_specialPropertyMap[key];
          if ( property && !getValue<bool>(property) )
          {
            setValue<bool>( property, true );
            hidNotify( property );
          }
        }

        void Widget::specialKeyReleased( int key, int x, int y )
        {
          dp::util::PropertyId property = m_specialPropertyMap[key];
          if ( property && getValue<bool>(property) )
          {
            setValue<bool>( property, false );
            hidNotify( property );
          }
        }

#if !defined(HAVE_FREEGLUT_2_8)
        void Widget::updateModifiers()
        {
          int modifiers = glutGetModifiers();
          if ( !!(modifiers & GLUT_ACTIVE_SHIFT) != getValue<bool>(PID_Key_Shift) )
          {
            setValue<bool>( PID_Key_Shift, !!(modifiers & GLUT_ACTIVE_SHIFT) );
            hidNotify( PID_Key_Shift );
          }
          if ( !!(modifiers & GLUT_ACTIVE_CTRL) != getValue<bool>(PID_Key_Control) )
          {
            setValue<bool>( PID_Key_Control, !!(modifiers & GLUT_ACTIVE_CTRL) );
            hidNotify( PID_Key_Control );
          }
          if ( !!(modifiers & GLUT_ACTIVE_ALT) != getValue<bool>(PID_Key_Alt) )
          {
            setValue<bool>( PID_Key_Alt, !!(modifiers & GLUT_ACTIVE_ALT) );
            hidNotify( PID_Key_Alt );
          }
        }
#endif

        void Widget::doTriggerRepaint()
        {
          BindWidget bind( m_windowId );
          glutPostRedisplay();
        }

        void Widget::setWindowTitle( std::string const& windowTitle )
        {
          BindWidget bind( m_windowId );
          m_windowTitleBase = windowTitle;
          std::ostringstream completeTitle;
          completeTitle << m_windowTitleBase << " (" << m_width << "," << m_height << ")";
          m_windowTitle = completeTitle.str();
          updateWindowTitle( m_windowTitle );
        }

        void Widget::setWindowSize( size_t width, size_t height )
        {
          BindWidget bindWidget( m_windowId );
          glutReshapeWindow( dp::checked_cast<int>(width), dp::checked_cast<int>(height) );
        }

        void Widget::setWindowFullScreen()
        {
          BindWidget bindWidget( m_windowId );
          glutFullScreen();
        }

        void Widget::updateWindowTitle( std::string const& title )
        {
          BindWidget bind( m_windowId );

          if ( m_showFPS )
          {
            std::ostringstream windowTitle;
            windowTitle.precision(2);
            windowTitle.setf( std::ios::fixed, std::ios::floatfield );
            windowTitle<< title << ", " << m_fps << " FPS";
            glutSetWindowTitle( windowTitle.str().c_str() );
          }
          else
          {
            glutSetWindowTitle( title.c_str() );
          }
        }

        void Widget::setShowFrameRate( bool showFPS )
        {
          m_showFPS = showFPS;
          if ( m_showFPS )
          {
            m_frameRateTimer.start();
            m_frameCount = 0;
          }
          else
          {
            m_fps = 0.0f;
          }
        }

        void Widget::setContinuousUpdate( bool continuous )
        {
          BindWidget bind( m_windowId );
          if ( m_continuousUpdate != continuous )
          {
            m_continuousUpdate = continuous;
            if ( continuous )
            {
              glutPostRedisplay();
            }
          }
        }

        dp::gl::RenderContextSharedPtr const& Widget::getRenderContext() const
        {
          return m_renderTarget ? m_renderTarget->getRenderContext() : dp::gl::RenderContextSharedPtr::null;
        }

        dp::gl::RenderTargetSharedPtr  const& Widget::getRenderTarget() const
        {
          return m_renderTarget;
        }

        void Widget::onManipulatorChanged( Manipulator *manipulator )
        {
          if ( manipulator )
          {
            manipulator->setRenderTarget( getRenderTarget() );
          }
        }

        void Widget::hidNotify( dp::util::PropertyId property )
        {
          notify( dp::util::Reflection::PropertyEvent( this, property ) );
          onHIDEvent( property );
        }

        /************************************************************************/
        /* HumanInterfaceDevice                                                 */
        /************************************************************************/
        unsigned int Widget::getNumberOfAxes() const
        {
          return 0;
        }

        std::string Widget::getAxisName( unsigned int axis ) const
        {
          return "";
        }

        unsigned int Widget::getNumberOfKeys() const
        {
          return 0;
          //return dp::checked_cast<unsigned int>(m_keyInfos.size());
        }

        std::string Widget::getKeyName( unsigned int key ) const
        {
          //DP_ASSERT( key < m_keyInfos.size() );

          //return m_keyInfos[key].name;
          return "";
        }

        void Widget::onHIDEvent( dp::util::PropertyId propertyId )
        {
        }

      }

    }
  }
}
