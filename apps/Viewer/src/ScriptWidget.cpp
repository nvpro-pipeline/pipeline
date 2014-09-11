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


#include <QLineEdit>
#include <QTextEdit>
#include <QToolBar>
#include "Viewer.h"
#include "ScriptWidget.h"
#include "ScriptSystem.h"

ScriptWidget::ScriptWidget(QWidget *parent)
  : QDockWidget( "Script Sandbox", parent )
  , m_lineEditMode(false)
{
  setObjectName( "ScriptWidget" );

  QWidget * group = new QWidget( this );
  m_layout = new QVBoxLayout( group );

  m_toolBar = new QToolBar( group );
  m_toolBar->setIconSize( QSize( 8, 8 ) );

  m_textEdit = new QTextEdit( group );
  m_lineEdit = new QLineEdit( group );

  // set some nice defaults
  m_textEdit->setReadOnly( false );
  m_lineEdit->setReadOnly( false );

  m_layout->setSpacing(1);
  m_layout->setContentsMargins(0,0,0,0);
  m_layout->addWidget( m_toolBar );
  m_layout->addWidget( m_textEdit );
  m_layout->addWidget( m_lineEdit );
  m_lineEdit->setVisible( false );
  
  group->setLayout( m_layout );
  
  // Note that you must add the layout of the widget before you 
  // call this function; if not, the widget will not be visible.
  setWidget( group );

  // add some actions for the toolbar
  QAction * actionExecute = new QAction( this );
  actionExecute->setObjectName(QString("actionExecute"));
  actionExecute->setText(QApplication::translate(VIEWER_APPLICATION_NAME, "Execute Selection", 0));
  actionExecute->setShortcut( QKeySequence::Refresh );  // F5
  QIcon eicon;
  eicon.addFile(QString::fromUtf8(":/images/expandTree.png"), QSize(), QIcon::Normal, QIcon::Off);
  actionExecute->setIcon(eicon);

  QAction * actionLineEdit = new QAction( this );
  actionLineEdit->setObjectName(QString("actionLineEdit"));
  actionLineEdit->setText(QApplication::translate(VIEWER_APPLICATION_NAME, "Line Edit Mode", 0));
  actionLineEdit->setChecked( false );
  QIcon licon;
  licon.addFile(QString::fromUtf8(":/images/expandTree.png"), QSize(), QIcon::Normal, QIcon::Off);
  actionLineEdit->setIcon(licon);

  m_toolBar->addAction( actionExecute );
  m_toolBar->addAction( actionLineEdit );

  connect( actionExecute,  SIGNAL(triggered()),     this, SLOT(executeSelection()) );
  connect( actionLineEdit, SIGNAL(triggered()),     this, SLOT(switchEditMode()) );
  connect( m_lineEdit,     SIGNAL(returnPressed()), this, SLOT(executeLine()) );
}

void
ScriptWidget::executeSelection()
{
  QTextCursor cursor = m_textEdit->textCursor();
  if( cursor.hasSelection() )
  {
    QString text = cursor.selectedText();
    GetApp()->getScriptSystem()->executeScript( text, "sandbox" );
  }
}

void
ScriptWidget::executeLine()
{
  QString text = m_lineEdit->text();
  
  if( !text.isEmpty() )
  {
    GetApp()->getScriptSystem()->executeScript( text, "sandbox" );
  }

  m_lineEdit->clear();
}

void
ScriptWidget::switchEditMode()
{
  if( m_lineEditMode )
  {
    m_textEdit->setVisible( true );
    m_lineEdit->setVisible( false );
  }
  else
  {
    m_textEdit->setVisible( false );
    m_lineEdit->setVisible( true );
  }

  m_lineEditMode = !m_lineEditMode;
}

ScriptWidget::~ScriptWidget()
{
}

