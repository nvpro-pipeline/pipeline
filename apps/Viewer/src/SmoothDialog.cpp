// Copyright NVIDIA Corporation 2011
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


#include <QBoxLayout>
#include <QDialogButtonBox>
#include <QGroupBox>
#include <QPushButton>
#include "SmoothDialog.h"
#include "Viewer.h"
#include <dp/sg/algorithm/SmoothTraverser.h>

using namespace dp::sg::core;
using namespace dp::math;

SmoothDialog::SmoothDialog( const SceneSharedPtr & scene, QWidget * parent )
  : QDialog( parent )
  , m_scene( scene )
{
  setWindowTitle( QApplication::translate( VIEWER_APPLICATION_NAME, "Smooth Scene" ) );

  // granularity for crease angle is selected to be 0.5°; that is range from 0..360 for 0..180°
  m_creaseAngleSlider = new QSlider;
  m_creaseAngleSlider->setMaximum( 360 );       // maximum: 180 degrees
  m_creaseAngleSlider->setMinimum( 0 );         // minimum: 0 degrees
  m_creaseAngleSlider->setOrientation( Qt::Horizontal );
  m_creaseAngleSlider->setPageStep( 10 );       // page step: 5 degrees
  m_creaseAngleSlider->setSingleStep( 1 );      // single step: 0.5 degrees
  m_creaseAngleSlider->setValue( 90 );          // value: 45 degrees

  QLabel * label = new QLabel( QApplication::translate( VIEWER_APPLICATION_NAME, "Crease Angle: " ) );
  m_creaseAngleLabel = new QLabel( QApplication::translate( VIEWER_APPLICATION_NAME, "45°" ) );
  QHBoxLayout * hLayout = new QHBoxLayout;
  hLayout->addWidget( label );
  hLayout->addWidget( m_creaseAngleLabel );

  QVBoxLayout * vLayout = new QVBoxLayout;
  vLayout->addWidget( m_creaseAngleSlider );
  vLayout->addLayout( hLayout );
  QGroupBox * optionsBox = new QGroupBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Options" ) );
  optionsBox->setLayout( vLayout );

  QDialogButtonBox * dbb = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );
  dbb->button( QDialogButtonBox::Ok )->setDefault ( true );

  QVBoxLayout * mainLayout = new QVBoxLayout;
  mainLayout->addWidget( optionsBox );
  mainLayout->addWidget( dbb );

  setLayout( mainLayout );
  adjustSize();
  setMinimumSize( size() );
  setMaximumSize( size() );

  // connect everything up
  connect( m_creaseAngleSlider, SIGNAL(valueChanged(int)), this, SLOT(valueChangedCreaseAngle(int)) );
  connect( dbb, SIGNAL(accepted()), this, SLOT(accept()) );
  connect( dbb, SIGNAL(rejected()), this, SLOT(reject()) );
}

SmoothDialog::~SmoothDialog()
{
}

void SmoothDialog::accept()
{
  GetApp()->setOverrideCursor( Qt::WaitCursor );
  {
    dp::util::SmartPtr<dp::sg::algorithm::SmoothTraverser> st( new dp::sg::algorithm::SmoothTraverser );
    st->setCreaseAngle( degToRad( 0.5f * m_creaseAngleSlider->value() ) );
    st->apply( m_scene );
  }
  GetApp()->restoreOverrideCursor();

  QDialog::accept();
}

void SmoothDialog::valueChangedCreaseAngle( int value )
{
  m_creaseAngleLabel->setText( QString::number( 0.5 * value, 'f', 1 ) + "°" );
}
