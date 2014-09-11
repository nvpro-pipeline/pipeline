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


#include "LogWidget.h"
#include <QTime>

LogWidget::LogWidget(QWidget *parent)
  : QDockWidget( "Log", parent )
{
  m_textEdit = new QTextBrowser( this );

  // set some nice defaults
  m_textEdit->setReadOnly( true );

  m_warningColor = "#FF8000"; // orange
  m_errorColor   = "#FF0000"; // red

  setObjectName( "LogWidget" );
  setWidget( m_textEdit );
}

inline void escapeHTML( QString & str )
{
  // we can add more to this list if necessary
  str.replace( QString("<"),  QString("&lt;") );
  str.replace( QString(">"),  QString("&gt;") );
}

void LogWidget::message( const QString & inmessage, Severity severity ) const
{
  QTime tm = QTime::currentTime();
  QString output = tm.toString( "hh:mm:ss" ) + QString( "&nbsp;:&nbsp;" );

  // escape any HTML markup that may be in the text
  QString message( inmessage );
  escapeHTML( message );

  if ( severity == LOG_MESSAGE )
  {
    output += QString("<b>") + message + QString("</b>");
  }
  else
  {
    output.append( "<b><span style=\"color:" );
    output.append( (severity == LOG_WARNING) ? m_warningColor : m_errorColor );
    output.append( "\">" );
    output.append( message );
    output.append( "</span></b>" );
  }

  // add hard break where they had \n
  output.replace( QString("\n"), QString("<br>") );

  m_textEdit->insertHtml( output );
  // scroll down to find cursor
  m_textEdit->ensureCursorVisible();
}

LogWidget::~LogWidget()
{
}

