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


#include "ViewportLayout.h"
#include "ViewerRendererWidget.h"
#include <dp/sg/ui/ViewState.h>
#include <QSplitter>
#include <QVBoxLayout>
// more fixes for Qt's busted headers
#ifdef KeyPress
#undef KeyPress
#undef KeyRelease
#undef None
#undef FocusIn
#undef FocusOut
#undef FontChange
#undef Bool
#endif
#include <QEvent>
#include <QAction>
#include <QActionEvent>

using namespace dp::util;

class QSplitterMovable : public QSplitter
{
  public:
    QSplitterMovable(QWidget * parent) : QSplitter(parent) {}
    QSplitterMovable(Qt::Orientation orientation, QWidget * parent) : QSplitter(orientation, parent) {}
    ~QSplitterMovable() {}
    void moveSplitterExt(int pos, int index)
    {
      moveSplitter(pos, index);
    }
};

ViewportLayout::ViewportLayout(QWidget * parent)
 : QWidget( parent )
 , m_currentLayout( Type::UNDEFINED )
 , m_activeViewportIndex( ~0 )
 , m_preventSplitterMoveRecursion( false )
{
  for(int i = 0; i < MAX_VIEWS; i++)
  {
    m_viewport[i] = NULL;
    m_displayViewport[i] = i;
  }

  for(int i = 0; i < MAX_VIEWS-1; i++)
  {
    m_splitter[i] = new QSplitterMovable( NULL );
  }

  QVBoxLayout * layout = new QVBoxLayout( this );

  layout->setSpacing(1);
  layout->setContentsMargins(0,0,0,0);
  setLayout( layout );
}

void ViewportLayout::clear()
{
  for(int i = 0; i < MAX_VIEWS; i++)
  {
    if(m_viewport[i])
    {
      m_viewport[i]->setParent(NULL);
      delete m_viewport[i];
      m_viewport[i] = nullptr;
    }
    m_displayViewport[i] = i;
  }

  // 0 is the parent of 1 and 2, delete backwards.
  for(int i = 2; i >= 0; i--)
  {
    m_splitter[i]->setParent(NULL);
    delete m_splitter[i];
    m_splitter[i] = new QSplitterMovable( NULL );
  }

  // reset these vars too
  m_currentLayout = Type::UNDEFINED;
  m_activeViewportIndex = ~0;
  m_preventSplitterMoveRecursion = false;
}

ViewportLayout::~ViewportLayout()
{
  for(int i = 0; i < MAX_VIEWS; i++)
  {
    if(m_viewport[i])
    {
      m_viewport[i]->setParent(NULL);
      delete m_viewport[i];
    }
  }

  // 0 is the parent of 1 and 2, delete backwards.
  for(int i = 2; i >= 0; i--)
  {
    m_splitter[i]->setParent(NULL);
    delete m_splitter[i];
  }
}

void ViewportLayout::setViewport( unsigned int idx, ViewerRendererWidget * viewport )
{
  DP_ASSERT( idx < MAX_VIEWS );

  if ( viewport )
  {
    m_renderWidgets[idx] = viewport;
    m_renderWidgets[idx]->installEventFilter( this );

    m_viewport[idx] = new QWidget( this );
    QVBoxLayout * layout = new QVBoxLayout( m_viewport[idx] );

    layout->setSpacing(0);
    layout->setContentsMargins(1,1,1,1);
    layout->addWidget( m_renderWidgets[idx] );
    m_viewport[idx]->setLayout( layout );
    m_viewport[idx]->setStyleSheet( "border: 1px solid #5a5a5a;" );
  }
  else
  {
    DP_ASSERT( m_viewport[idx] );
    m_viewport[idx]->setParent( nullptr );
    delete m_viewport[idx];
    m_viewport[idx] = nullptr;
  }

  m_displayViewport[idx] = idx;

  // set the active one to be zero initially
  emit activeViewportChanged( 0, m_viewport[0] );
  setActiveViewport(0);
}

//
//
bool ViewportLayout::eventFilter( QObject * obj, QEvent * event )
{
  QWidget * widget = dynamic_cast< QWidget * >( obj );

  if( widget )
  {
    for( unsigned int i = 0; i < MAX_VIEWS; i ++ )
    {
      if( widget == m_renderWidgets[i] )
      {
        if( event->type() == QEvent::MouseButtonPress ||
            event->type() == QEvent::KeyPress )
        {
          // this activates this window in our model
          emit activeViewportChanged( i, widget );
          setActiveViewport( i );
        }

        break;
      }
    }
  }

  // pass to parent
  return QWidget::eventFilter( obj, event );
}

void ViewportLayout::setActiveViewport(int index)
{
  if( index != m_activeViewportIndex && m_activeViewportIndex != ~0 )
  {
    // reset style sheet on former 
    m_viewport[m_activeViewportIndex]->setStyleSheet("border: 1px solid #5a5a5a;");
    // turn this off, if it was on
  }

  m_activeViewportIndex = index;

  // set new style sheet, but only if we have more than one view
  if( m_currentLayout != Type::ONE && m_currentLayout != Type::UNDEFINED )
  {
    m_viewport[m_activeViewportIndex]->setStyleSheet("border: 1px solid #76b900;"); // NVIDIA Green.
  }

  int newDisplayViewport[MAX_VIEWS];
  
  int newDisplayIndex = 1;
  while( m_displayViewport[newDisplayIndex-1] != index )
  {
    newDisplayViewport[newDisplayIndex] = m_displayViewport[newDisplayIndex-1];
    newDisplayIndex++;
  }

  while(newDisplayIndex < MAX_VIEWS)
  {
    newDisplayViewport[newDisplayIndex] = m_displayViewport[newDisplayIndex];
    newDisplayIndex++;
  }

  newDisplayViewport[0] = index;

  for(int i = 0; i < MAX_VIEWS; i++)
  {
    m_displayViewport[i] = newDisplayViewport[i];
  }
}

void ViewportLayout::setViewportLayout(Type type)
{
  // ignore same requests
  if( type == m_currentLayout )
  {
    return;
  }

  for(int i = 0; i < MAX_VIEWS; i++)
  {
    if(m_viewport[i])
    {
      m_viewport[i]->setParent(NULL);
    }
  }

  layout()->removeWidget( m_splitter[0] );

  for(int i = 0; i < MAX_VIEWS-1; i++)
  {
    m_splitter[i]->setParent(NULL);
  }

  if( m_currentLayout == ViewportLayout::Type::ONE )
  {
    // if we were in layout 0, then nothing was highlighted.  highlight it now
    m_viewport[m_activeViewportIndex]->setStyleSheet("border: 1px solid #76b900;"); // NVIDIA Green
  }
  else if( m_currentLayout == ViewportLayout::Type::FOUR )
  {
    // if we were in layout 5, then remove the signals
    disconnect( m_splitter[1], SIGNAL(splitterMoved(int, int)), this, SLOT(synchMoveSplitter2(int, int)));
    disconnect( m_splitter[2], SIGNAL(splitterMoved(int, int)), this, SLOT(synchMoveSplitter1(int, int)));
  }

  QSize size = this->size();

  switch(type)
  {
    case ViewportLayout::Type::ONE:
    {
      // if we are moving to layout 0, then remove the border
      m_viewport[m_activeViewportIndex]->setStyleSheet("border: 1px solid #5a5a5a;");

      m_splitter[0]->setParent(this);
      m_splitter[0]->setOrientation(Qt::Vertical);
      layout()->addWidget( m_splitter[0] );
      m_splitter[0]->addWidget(m_viewport[m_displayViewport[0]]);
    }
    break;

    case ViewportLayout::Type::TWO_LEFT:
    {
      m_splitter[0]->setParent(this);
      m_splitter[0]->setOrientation(Qt::Horizontal);
      layout()->addWidget( m_splitter[0] );
      m_splitter[0]->addWidget(m_viewport[m_displayViewport[0]]);
      m_splitter[0]->addWidget(m_viewport[m_displayViewport[1]]);

      QList<int> sizes;
      sizes.append( size.width()/2 );
      sizes.append( size.width()/2 );
      m_splitter[0]->setSizes( sizes );
    }
    break;

    case ViewportLayout::Type::TWO_TOP:
    {
      m_splitter[0]->setParent(this);
      m_splitter[0]->setOrientation(Qt::Vertical);
      layout()->addWidget( m_splitter[0] );
      m_splitter[0]->addWidget(m_viewport[m_displayViewport[0]]);
      m_splitter[0]->addWidget(m_viewport[m_displayViewport[1]]);

      QList<int> sizes;
      sizes.append( size.height()/2 );
      sizes.append( size.height()/2 );
      m_splitter[0]->setSizes( sizes );
    }
    break;

    case ViewportLayout::Type::THREE_LEFT:
    {
      QList<int> size0, size1;
      m_splitter[0]->setParent(this);
      m_splitter[0]->setOrientation(Qt::Horizontal);
      m_splitter[0]->addWidget(m_viewport[m_displayViewport[0]]);
      size0.append( size.width() / 2);
      size0.append( size.width() / 2);

      m_splitter[1]->setOrientation(Qt::Vertical);
      m_splitter[1]->addWidget(m_viewport[m_displayViewport[1]]);
      m_splitter[1]->addWidget(m_viewport[m_displayViewport[2]]);
      layout()->addWidget( m_splitter[0] );
      m_splitter[0]->addWidget(m_splitter[1]);

      size1.append( size.height()/2 );
      size1.append( size.height()/2 );

      m_splitter[0]->setSizes( size0 );
      m_splitter[1]->setSizes( size1 );
    }
    break;

    case ViewportLayout::Type::THREE_TOP:
    {
      QList<int> size0, size1;
      m_splitter[0]->setParent(this);
      m_splitter[0]->setOrientation(Qt::Vertical);
      m_splitter[0]->addWidget(m_viewport[m_displayViewport[0]]);
      size0.append( size.height()/2 );
      size0.append( size.height()/2 );

      m_splitter[1]->setOrientation(Qt::Horizontal);
      m_splitter[1]->addWidget(m_viewport[m_displayViewport[1]]);
      m_splitter[1]->addWidget(m_viewport[m_displayViewport[2]]);
      layout()->addWidget( m_splitter[0] );
      m_splitter[0]->addWidget(m_splitter[1]);

      size1.append( size.width()/2 );
      size1.append( size.width()/2 );

      m_splitter[0]->setSizes( size0 );
      m_splitter[1]->setSizes( size1 );
    }
    break;

    case ViewportLayout::Type::FOUR:
    {
      QList<int> size0, size1;

      m_splitter[1]->setOrientation(Qt::Horizontal);
      m_splitter[1]->addWidget(m_viewport[m_displayViewport[0]]);
      m_splitter[1]->addWidget(m_viewport[m_displayViewport[1]]);
      size1.append( size.width() /2 );
      size1.append( size.width() /2 );

      m_splitter[2]->setOrientation(Qt::Horizontal);
      m_splitter[2]->addWidget(m_viewport[m_displayViewport[2]]);
      m_splitter[2]->addWidget(m_viewport[m_displayViewport[3]]);

      m_splitter[0]->setParent(this);
      m_splitter[0]->setOrientation(Qt::Vertical);
      layout()->addWidget( m_splitter[0] );
      m_splitter[0]->addWidget(m_splitter[1]);
      m_splitter[0]->addWidget(m_splitter[2]);

      size0.append( size.height()/2 );
      size0.append( size.height()/2 );

      m_splitter[0]->setSizes( size0 );
      m_splitter[1]->setSizes( size1 );
      m_splitter[2]->setSizes( size1 );

      connect( m_splitter[1], SIGNAL(splitterMoved(int, int)), this, SLOT(synchMoveSplitter2(int, int)));
      connect( m_splitter[2], SIGNAL(splitterMoved(int, int)), this, SLOT(synchMoveSplitter1(int, int)));
    }
    break;
  }

  m_currentLayout = type;
}

void ViewportLayout::synchMoveSplitter1(int pos, int index)
{
  if(!m_preventSplitterMoveRecursion)
  {
    m_splitter[1]->moveSplitterExt(pos, index);
  }

  m_preventSplitterMoveRecursion = false;
}

void ViewportLayout::synchMoveSplitter2(int pos, int index)
{
  m_preventSplitterMoveRecursion = true;
  m_splitter[2]->moveSplitterExt(pos, index);
}

unsigned int viewportCount( ViewportLayout::Type type )
{
  switch( type )
  {
    case ViewportLayout::Type::ONE :
      return( 1 );
    case ViewportLayout::Type::TWO_LEFT :
    case ViewportLayout::Type::TWO_TOP :
      return( 2 );
    case ViewportLayout::Type::THREE_LEFT :
    case ViewportLayout::Type::THREE_TOP :
      return( 3 );
    case ViewportLayout::Type::FOUR :
      return( 4 );
    default :
      DP_ASSERT( false );
      return( 0 );
  }
}