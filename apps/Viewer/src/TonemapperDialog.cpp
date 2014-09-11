// Copyright NVIDIA Corporation 2013
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


#include <QButtonGroup>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>
#include "TonemapperDialog.h"
#include "Viewer.h"

TonemapperDialog::TonemapperDialog( QWidget * parent, ViewerRendererWidget * vrw )
  : QDialog( parent )
  , m_renderer( vrw )
{
  DP_ASSERT( m_renderer );

  setWindowTitle( "Tonemapper Settings" );
  setWindowFlags( windowFlags() & ~Qt::WindowContextHelpButtonHint );

  m_restoreTonemapperEnable = m_renderer->isTonemapperEnabled();
  m_tonemapperValuesCurrent = m_renderer->getTonemapperValues();
  m_tonemapperValuesRestore = m_tonemapperValuesCurrent;

  m_spinGamma = new QDoubleSpinBox();
  m_spinGamma->setMinimum( 0.1 );
  m_spinGamma->setMaximum( 10.0 );
  m_spinGamma->setSingleStep( 0.1 );
  m_spinGamma->setValue( m_tonemapperValuesCurrent.gamma );
  connect( m_spinGamma, SIGNAL(valueChanged(double)), this, SLOT(gammaChanged(double)) );
  
  m_spinWhitePoint = new QDoubleSpinBox();
  m_spinWhitePoint->setMinimum( 0.1 );
  m_spinWhitePoint->setMaximum( 1000.0 );
  m_spinWhitePoint->setSingleStep( 0.1 );
  m_spinWhitePoint->setValue( m_tonemapperValuesCurrent.whitePoint );
  connect( m_spinWhitePoint, SIGNAL(valueChanged(double)), this, SLOT(whitePointChanged(double)) );

  m_spinBrightness = new QDoubleSpinBox();
  m_spinBrightness->setMinimum( 0.0 );
  m_spinBrightness->setMaximum( 100.0 );
  m_spinBrightness->setSingleStep( 0.01 );
  m_spinBrightness->setValue( m_tonemapperValuesCurrent.brightness );
  connect( m_spinBrightness, SIGNAL(valueChanged(double)), this, SLOT(brightnessChanged(double)) );

  m_spinSaturation = new QDoubleSpinBox();
  m_spinSaturation->setMinimum( 0.0 );
  m_spinSaturation->setMaximum( 10.0 );
  m_spinSaturation->setSingleStep( 0.01 );
  m_spinSaturation->setValue( m_tonemapperValuesCurrent.saturation );
  connect( m_spinSaturation, SIGNAL(valueChanged(double)), this, SLOT(saturationChanged(double)) );
  
  m_spinCrushBlacks = new QDoubleSpinBox();
  m_spinCrushBlacks->setMinimum( 0.0 );
  m_spinCrushBlacks->setMaximum( 10.0 );
  m_spinCrushBlacks->setSingleStep( 0.01 );
  m_spinCrushBlacks->setValue( m_tonemapperValuesCurrent.crushBlacks );
  connect( m_spinCrushBlacks, SIGNAL(valueChanged(double)), this, SLOT(crushBlacksChanged(double)) );

  m_spinBurnHighlights = new QDoubleSpinBox();
  m_spinBurnHighlights->setMinimum( 0.0 );
  m_spinBurnHighlights->setMaximum( 100.0 );
  m_spinBurnHighlights->setSingleStep( 0.01 );
  m_spinBurnHighlights->setValue( m_tonemapperValuesCurrent.burnHighlights );
  connect( m_spinBurnHighlights, SIGNAL(valueChanged(double)), this, SLOT(burnHighlightsChanged(double)) );

  QFormLayout * formLayout = new QFormLayout;
  formLayout->addRow( "Gamma", m_spinGamma );
  formLayout->addRow( "White Point", m_spinWhitePoint );
  formLayout->addRow( "Brighness", m_spinBrightness );
  formLayout->addRow( "Saturation", m_spinSaturation );
  formLayout->addRow( "Crush Blacks", m_spinCrushBlacks );
  formLayout->addRow( "Burn Highlights", m_spinBurnHighlights );

  QGroupBox * tonemapperBox = new QGroupBox( "Tone mapping" );
  tonemapperBox->setCheckable( true );
  tonemapperBox->setChecked( m_restoreTonemapperEnable );
  tonemapperBox->setLayout( formLayout );
  connect( tonemapperBox, SIGNAL(toggled(bool)), this, SLOT(setTonemapperEnable(bool)) );

  QDialogButtonBox * dbb = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );
  dbb->button( QDialogButtonBox::Ok )->setDefault ( true );
  connect( dbb, SIGNAL(accepted()), this, SLOT(accept()) );
  connect( dbb, SIGNAL(rejected()), this, SLOT(reject()) );

  QVBoxLayout * vLayout = new QVBoxLayout();
  vLayout->addWidget( tonemapperBox ); 
  vLayout->addWidget( dbb );

  setLayout( vLayout );
  adjustSize();
  setMinimumSize( size() );
  setMaximumSize( size() );
}

TonemapperDialog::~TonemapperDialog()
{
}

void TonemapperDialog::setTonemapperEnable( bool state )
{
  GetApp()->setTonemapperEnabled( state );
}

void TonemapperDialog::gammaChanged( double val )
{
  m_tonemapperValuesCurrent.gamma = float(val);
  m_renderer->setTonemapperValues( m_tonemapperValuesCurrent );
}

void TonemapperDialog::whitePointChanged( double val )
{
  m_tonemapperValuesCurrent.whitePoint = float(val);
  m_renderer->setTonemapperValues( m_tonemapperValuesCurrent );
}

void TonemapperDialog::brightnessChanged( double val )
{
  m_tonemapperValuesCurrent.brightness = float(val);
  m_renderer->setTonemapperValues( m_tonemapperValuesCurrent );
}

void TonemapperDialog::saturationChanged( double val )
{
  m_tonemapperValuesCurrent.saturation = float(val);
  m_renderer->setTonemapperValues( m_tonemapperValuesCurrent );
}

void TonemapperDialog::crushBlacksChanged( double val )
{
  m_tonemapperValuesCurrent.crushBlacks = float(val);
  m_renderer->setTonemapperValues( m_tonemapperValuesCurrent );
}

void TonemapperDialog::burnHighlightsChanged( double val )
{
  m_tonemapperValuesCurrent.burnHighlights = float(val);
  m_renderer->setTonemapperValues( m_tonemapperValuesCurrent );
}


void TonemapperDialog::reject()
{
  GetApp()->setTonemapperEnabled( m_restoreTonemapperEnable );
  m_renderer->setTonemapperValues( m_tonemapperValuesRestore );
  QDialog::reject();
}
