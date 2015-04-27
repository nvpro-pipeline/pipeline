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
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>
#include "Viewer.h"
#include "TransparencyDialog.h"

TransparencyDialog::TransparencyDialog( QWidget * parent, ViewerRendererWidget * vrw )
  : QDialog( parent )
  , m_renderer( vrw )
{
  DP_ASSERT( m_renderer );

  setWindowTitle( "Transparency Settings" );
  setWindowFlags( windowFlags() & ~Qt::WindowContextHelpButtonHint );

  m_restoreTransparencyMode = m_renderer->getTransparencyMode();
  QRadioButton * noneButton = new QRadioButton( "None" );
  noneButton->setChecked( m_restoreTransparencyMode == dp::sg::renderer::rix::gl::TM_NONE );
  QRadioButton * sortedBlendedButton = new QRadioButton( "Sorted Blended" );
  sortedBlendedButton->setChecked( m_restoreTransparencyMode == dp::sg::renderer::rix::gl::TM_SORTED_BLENDED );
  QRadioButton * orderIndependentClosestListButton = new QRadioButton( "Order Independent Closest List" );
  orderIndependentClosestListButton->setChecked( m_restoreTransparencyMode == dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_LIST );
  QRadioButton * orderIndependentAllButton = new QRadioButton( "Order Independent All" );
  orderIndependentAllButton->setChecked( m_restoreTransparencyMode == dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_ALL );

  QButtonGroup * buttonGroup = new QButtonGroup();
  buttonGroup->addButton( noneButton, dp::sg::renderer::rix::gl::TM_NONE );
  buttonGroup->addButton( sortedBlendedButton, dp::sg::renderer::rix::gl::TM_SORTED_BLENDED );
  buttonGroup->addButton( orderIndependentClosestListButton, dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_LIST );
  buttonGroup->addButton( orderIndependentAllButton, dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_ALL );
  connect( buttonGroup, SIGNAL(buttonClicked(int)), this, SLOT(buttonClicked(int)) );

  m_restoreLayers = m_renderer->getOITDepth();
  m_layersBox = new QSpinBox();
  m_layersBox->setMinimum( 1 );
  m_layersBox->setValue( m_restoreLayers );
  m_layersBox->setEnabled( ( m_restoreTransparencyMode == dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_ARRAY )
                        || ( m_restoreTransparencyMode == dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_LIST ) );
  connect( m_layersBox, SIGNAL(valueChanged(int)), this, SLOT(layersChanged(int)) );
  QFormLayout * layersLayout = new QFormLayout;
  layersLayout->addRow( "Transparency Layers", m_layersBox );

  QFrame * separatorLine = new QFrame;
  separatorLine->setFrameShape( QFrame::HLine );
  separatorLine->setFrameShadow( QFrame::Sunken );

  QDialogButtonBox * dbb = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );
  dbb->button( QDialogButtonBox::Ok )->setDefault ( true );
  connect( dbb, SIGNAL(accepted()), this, SLOT(accept()) );
  connect( dbb, SIGNAL(rejected()), this, SLOT(reject()) );

  QVBoxLayout * vLayout = new QVBoxLayout();
  vLayout->addWidget( noneButton );
  vLayout->addWidget( sortedBlendedButton );
  vLayout->addWidget( orderIndependentClosestListButton );
  vLayout->addLayout( layersLayout );
  vLayout->addWidget( orderIndependentAllButton );
  vLayout->addWidget( separatorLine );
  vLayout->addWidget( dbb );

  setLayout( vLayout );
  adjustSize();
  setMinimumSize( size() );
  setMaximumSize( size() );
}

TransparencyDialog::~TransparencyDialog()
{
}

void TransparencyDialog::buttonClicked( int id )
{
  m_layersBox->setEnabled( ( id == dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_ARRAY )
                        || ( id == dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_LIST ) );
  if ( m_renderer->getTransparencyMode() != id )
  {
    m_renderer->setTransparencyMode( dp::sg::renderer::rix::gl::TransparencyMode(id) );
    GetApp()->getPreferences()->setTransparencyMode( id );
  }
}

void TransparencyDialog::layersChanged( int val )
{
  m_renderer->setOITDepth( val );
}

void TransparencyDialog::reject()
{
  if ( m_restoreTransparencyMode != m_renderer->getTransparencyMode() )
  {
    m_renderer->setTransparencyMode( m_restoreTransparencyMode );
  }
  if ( m_restoreLayers != m_renderer->getOITDepth() )
  {
    m_renderer->setOITDepth( m_restoreLayers );
  }
  QDialog::reject();
}
