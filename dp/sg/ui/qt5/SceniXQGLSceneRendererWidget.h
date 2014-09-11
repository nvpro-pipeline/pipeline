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

#include <dp/sg/ui/qt5/SceniXQGLWidget.h>
#include <QTimer>
#include <dp/util/Timer.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      namespace qt5
      {

        /************************************************************************/
        /* Basic widget to render a scene/viewstate pair using a SceneRenderer. */
        /* The SceneRenderer must accept a dp::gl::RenderTarget as RenderTarget */
        /************************************************************************/
        class SceniXQGLSceneRendererWidget : public SceniXQGLWidget, public SceniXSceneRendererWidget
        {
        public:
          DP_SG_UI_QT5_API SceniXQGLSceneRendererWidget(QWidget *parent, const dp::gl::RenderContextFormat &format, SceniXQGLWidget *shareWidget = 0);
          DP_SG_UI_QT5_API virtual ~SceniXQGLSceneRendererWidget();

          DP_SG_UI_QT5_API virtual void triggerRepaint();
          DP_SG_UI_QT5_API virtual void setContinuousUpdate( bool tf );
          DP_SG_UI_QT5_API virtual bool getContinuousUpdate() const;

        protected:
          DP_SG_UI_QT5_API virtual void onManipulatorChanged( Manipulator *manipulator );
          // updates manipulator, calls triggerRepaint
          DP_SG_UI_QT5_API virtual void onTimer( float dt );

          DP_SG_UI_QT5_API virtual void hidNotify( dp::util::PropertyId property );

          DP_SG_UI_QT5_API virtual void initializeGL();
          DP_SG_UI_QT5_API virtual void paintGL();

          DP_SG_UI_QT5_API float getElapsedTime();

        private:
          DP_SG_UI_QT5_API virtual void timerEvent( QTimerEvent * event );

        protected:
          bool m_continuousUpdate;
          int  m_timerID;
          dp::util::Timer m_todTimer;
          double          m_lastTime;
        };

        inline bool SceniXQGLSceneRendererWidget::getContinuousUpdate() const
        {
          return m_continuousUpdate;
        }

      } // namespace qt5
    } // namespace ui
  } // namespace sg
} // namespace dp
