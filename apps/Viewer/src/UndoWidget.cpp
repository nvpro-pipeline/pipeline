
// Copyright NVIDIA Corporation 2009-2010
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


#include <QEvent>
#include <QUndoView>
#include <QContextMenuEvent>
#include "Viewer.h"
#include "UndoWidget.h"

UndoWidget::UndoWidget(QWidget *parent)
  : QDockWidget( "Undo Stack", parent )
{
  setObjectName( "UndoWidget" );
  m_undoView = new QUndoView( &GetSceneStateUndoStack(), this );

  setWidget( m_undoView );
}

UndoWidget::~UndoWidget()
{
}

bool UndoWidget::empty() const
{
  return( m_undoView->stack()->count() == 0 );
}

void UndoWidget::clear()
{
  m_undoView->stack()->clear();
}

void UndoWidget::contextMenuEvent( QContextMenuEvent * event )
{
  QDockWidget::contextMenuEvent( event );

  if ( m_undoView->stack()->count() )
  {
    QMenu menu( "Undo Context Menu", this );
    QAction * action = menu.addAction( QApplication::translate( VIEWER_APPLICATION_NAME, "&Clear Undo Stack" ) );
    connect( action, SIGNAL(triggered()), this, SLOT(clear()) );

    menu.exec( event->globalPos() );
  }
}
