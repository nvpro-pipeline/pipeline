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


#include "Viewer.h"
#include "StereoDialog.h"
#include "ViewerRendererWidget.h"
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QGroupBox>
#include <QPushButton>

StereoDialog::StereoDialog( QWidget * parent, ViewerRendererWidget * vrw )
  : QDialog( parent )
  , m_renderer( vrw )
{
  DP_ASSERT( m_renderer );

  dp::gl::RenderContextFormat format = m_renderer->getFormat();
  m_restoreStereoEnable = format.isStereo();

  // Get the current setting from SceniX.
  {
    dp::sg::ui::ViewStateSharedPtr const& vs = m_renderer->getViewState();

    m_restoreAdjustment = vs->isStereoAutomaticEyeDistanceAdjustment(); 
    m_restoreFactor     = vs->getStereoAutomaticEyeDistanceFactor(); 
    m_restoreDistance   = vs->getStereoEyeDistance();
    m_restoreReversed   = vs->isStereoReversedEyes();
  }

  setWindowTitle( "Stereo" );

  QCheckBox * adjustmentCheck = new QCheckBox( "Automatic Eye Distance Adjustment" );
  adjustmentCheck->setChecked( true );

  m_adjustementLabel = new QLabel( m_restoreAdjustment ? "Distance Factor" : "Distance" );

  m_adjustmentSpin = new QDoubleSpinBox();
  m_adjustmentSpin->setDecimals( 4 );
  m_adjustmentSpin->setMaximum( 99999.0 );
  m_adjustmentSpin->setSingleStep( 0.001 );
  m_adjustmentSpin->setValue( m_restoreAdjustment ? m_restoreFactor : m_restoreDistance );

  QHBoxLayout * adjustmentLayout = new QHBoxLayout();
  adjustmentLayout->addWidget( m_adjustementLabel );
  adjustmentLayout->addWidget( m_adjustmentSpin );

  QCheckBox * reversedCheck = new QCheckBox( "Reversed Eyes" );
  reversedCheck->setChecked( m_restoreReversed );

  QVBoxLayout * stereoLayout = new QVBoxLayout();
  stereoLayout->addWidget( adjustmentCheck );
  stereoLayout->addLayout( adjustmentLayout );
  stereoLayout->addWidget( reversedCheck );

  // If we get here there is a stereo format available, the View->Stereo... menu entry enable state covered that.
  // There is no need to disable the group box.
  QGroupBox * stereoBox = new QGroupBox( "Stereo" );
  stereoBox->setCheckable( true );
  stereoBox->setChecked( m_restoreStereoEnable );
  stereoBox->setLayout( stereoLayout );

  QDialogButtonBox * dbb = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );
  dbb->button( QDialogButtonBox::Ok )->setDefault ( true );

  QVBoxLayout * mainLayout = new QVBoxLayout;
  mainLayout->addWidget( stereoBox );
  mainLayout->addWidget( dbb );

  setLayout( mainLayout );
  adjustSize();
  setMinimumSize( size() );
  setMaximumSize( size() );

  // Need to track these current states to use the SpinBox for either the factor or the distance.
  m_currentAdjustment = m_restoreAdjustment;
  m_currentFactor     = m_restoreFactor;
  m_currentDistance   = m_restoreDistance;

  connect( stereoBox,         SIGNAL(toggled(bool)),        this, SLOT(setStereoEnable(bool)) );
  connect( adjustmentCheck,   SIGNAL(toggled(bool)),        this, SLOT(setAdjustment(bool)) );
  connect( m_adjustmentSpin,  SIGNAL(valueChanged(double)), this, SLOT(setDistance(double)) );
  connect( reversedCheck,     SIGNAL(toggled(bool)),        this, SLOT(setReversed(bool)) );
  connect( dbb,               SIGNAL(accepted()),           this, SLOT(accept()) );
  connect( dbb,               SIGNAL(rejected()),           this, SLOT(reject()) );
}

StereoDialog::~StereoDialog()
{
}

void StereoDialog::setStereoEnable( bool state )
{
  m_renderer->triggeredViewportFormatStereo( state );
}

void StereoDialog::setAdjustment( bool state )
{
  m_currentAdjustment = state;

  m_renderer->getViewState()->setStereoAutomaticEyeDistanceAdjustment( m_currentAdjustment );

  // Change the display of the Eye Distance spin box to reflect the current usage.
  if ( m_currentAdjustment )
  {
    m_adjustementLabel->setText( "Distance Factor" );
    m_adjustmentSpin->setValue( m_currentFactor );
  }
  else
  {
    m_adjustementLabel->setText( "Distance" );
    m_adjustmentSpin->setValue( m_currentDistance );
  }
}

void StereoDialog::setDistance( double distance )
{
  // Change the display of the eye distance spin box to reflect the current usage.
  if ( m_currentAdjustment )
  {
    m_currentFactor = (float) distance;
    m_renderer->getViewState()->setStereoAutomaticEyeDistanceFactor( m_currentFactor );
  }
  else
  {
    m_currentDistance = (float) distance;
    m_renderer->getViewState()->setStereoEyeDistance( m_currentDistance );
  }
  
  m_renderer->present();
}

void StereoDialog::setReversed( bool state )
{
  m_renderer->getViewState()->setStereoReversedEyes( state );
  m_renderer->present();
}

void StereoDialog::restore()
{
  m_renderer->triggeredViewportFormatStereo( m_restoreStereoEnable );
  
  dp::sg::ui::ViewStateSharedPtr const& vs = m_renderer->getViewState();

  vs->setStereoAutomaticEyeDistanceAdjustment( m_restoreAdjustment );

  if ( m_restoreAdjustment )
  {
    vs->setStereoAutomaticEyeDistanceFactor( m_restoreFactor );
  }
  else
  {
    vs->setStereoEyeDistance( m_restoreDistance );
  }

  vs->setStereoReversedEyes( m_restoreReversed );

  m_renderer->present();
}
