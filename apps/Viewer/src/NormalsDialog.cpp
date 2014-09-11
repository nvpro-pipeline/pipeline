// Copyright NVIDIA Corporation 2009-2011
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


#include "NormalsDialog.h"
#include "DisplayNormalsTraverser.h"
#include "Viewer.h"
#include <QColorDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>

NormalsDialog::NormalsDialog( QWidget * parent, bool state, float length, const QColor & color )
  : QDialog( parent )
  , m_displayState( state )
  , m_length( length )
  , m_color( color )
{
  setWindowTitle( "Display Normals" );

  QDoubleSpinBox * lengthSpinBox = new QDoubleSpinBox();
  lengthSpinBox->setMinimum( 0.01 );
  lengthSpinBox->setSingleStep( 0.01 );
  lengthSpinBox->setValue( m_length );

  QPushButton * colorButton = new QPushButton( m_color.name() );
  colorButton->setStyleSheet( QString( "QPushButton { color:" ) + m_color.name() + QString( "; }" ) );
  colorButton->setAutoFillBackground( true );

  QFormLayout * settingsLayout = new QFormLayout();
  settingsLayout->addRow( "Length", lengthSpinBox );
  settingsLayout->addRow( "Color", colorButton );

  QGroupBox * settingsBox = new QGroupBox( "Display Normals", this );
  settingsBox->setCheckable( true );
  settingsBox->setChecked( m_displayState );
  settingsBox->setLayout( settingsLayout );

  QDialogButtonBox * dbb = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );
  dbb->button( QDialogButtonBox::Ok )->setAutoDefault( true );

  QVBoxLayout * mainLayout = new QVBoxLayout;
  mainLayout->addWidget( settingsBox );
  mainLayout->addWidget( dbb );

  setLayout( mainLayout );
  adjustSize();
  setMinimumSize( size() );
  setMaximumSize( size() );

  // connect everything up
  connect( settingsBox,   SIGNAL(clicked(bool)),        this, SLOT(setDisplayNormals(bool)) );
  connect( lengthSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setNormalsLength(double)) );
  connect( colorButton,   SIGNAL(clicked()),            this, SLOT(setNormalsColor()) );
  connect( dbb,           SIGNAL(accepted()),           this, SLOT(accept()) );
  connect( dbb,           SIGNAL(rejected()),           this, SLOT(reject()) );
}

NormalsDialog::~NormalsDialog()
{
}

void NormalsDialog::getOptions( bool & state, float & length, QColor & color )
{
  state  = m_displayState;
  length = m_length;
  color  = m_color;
}

void NormalsDialog::apply( const dp::sg::core::SceneSharedPtr & scene )
{
  DP_ASSERT( scene );
  GetApp()->setOverrideCursor( Qt::WaitCursor );

  dp::util::SmartPtr<DisplayNormalsTraverser> normalsTraverser( new DisplayNormalsTraverser );

  // set the length of the normals to display
  normalsTraverser->setNormalLength( m_displayState ? m_length : 0.f );

  // set the color of the normals to display
  qreal r, g, b, a;
  m_color.getRgbF( &r, &g, &b, &a );

  dp::math::Vec3f color( (float)r, (float)g, (float)b );
  normalsTraverser->setNormalColor( color );
  normalsTraverser->apply( scene );

  GetApp()->restoreOverrideCursor();
}

void NormalsDialog::setDisplayNormals( bool state )
{
  m_displayState = state;
}

void NormalsDialog::setNormalsLength( double length )
{
  m_length = length;
}

void NormalsDialog::setNormalsColor()
{
  QColorDialog colorDialog( m_color, this );

  if ( colorDialog.exec() == QDialog::Accepted )
  {
    m_color = colorDialog.selectedColor();

    DP_ASSERT( dynamic_cast<QPushButton*>(sender()) );
    QPushButton * button = static_cast<QPushButton*>(sender());
    button->setStyleSheet( QString( "QPushButton { color:" ) + m_color.name() + QString( "; }" ) );
    button->setText( m_color.name() );
  }
}

