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


#pragma once

#include <QWidget>
#include <vector>

class QSplitterMovable;
class ViewerRendererWidget;

class ViewportLayout : public QWidget
{
  Q_OBJECT

public:
  enum class Type
  {
    ONE,
    TWO_LEFT,
    TWO_TOP,
    THREE_LEFT,
    THREE_TOP,
    FOUR,
    UNDEFINED
  };

  enum { MAX_VIEWS = 4 };

  ViewportLayout(QWidget * parent = 0);
  virtual ~ViewportLayout();
  void setViewport( unsigned int index, ViewerRendererWidget * viewport );

protected:
  virtual bool eventFilter( QObject * obj, QEvent * event );

public slots:
  void setViewportLayout(Type type);
  void setActiveViewport(int index);
  void clear();

private slots:
  void synchMoveSplitter1(int pos, int index);
  void synchMoveSplitter2(int pos, int index);

signals:
  void activeViewportChanged( int, QWidget * );

private:
  Type m_currentLayout;
  int m_activeViewportIndex;
  int m_displayViewport[MAX_VIEWS];
  QWidget *m_viewport[MAX_VIEWS];
  ViewerRendererWidget *m_renderWidgets[MAX_VIEWS];
  QSplitterMovable *m_splitter[MAX_VIEWS-1];
  bool m_preventSplitterMoveRecursion;
};

unsigned int viewportCount( ViewportLayout::Type viewportLayout );
