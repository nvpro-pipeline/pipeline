// Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <QDialogButtonBox>
#include <QFormLayout>
#include <QPushButton>
#include <dp/sg/algorithm/CombineTraverser.h>
#include <dp/sg/algorithm/EliminateTraverser.h>
#include <dp/sg/algorithm/UnifyTraverser.h>
#include <dp/sg/algorithm/Optimize.h>
#include "OptimizerDialog.h"
#include "Viewer.h"

using namespace dp::sg::core;

OptimizerDialog::OptimizerDialog( const SceneSharedPtr & scene, QWidget * parent )
  : QDialog( parent )
  , m_scene( scene )
{
  setWindowTitle( QApplication::translate( VIEWER_APPLICATION_NAME, "Run Optimizers" ) );

  m_ignoreNamesButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Ignore Names" ) );
  m_ignoreNamesButton->setChecked( true );
  QVBoxLayout * vBoxLayout = new QVBoxLayout;
  vBoxLayout->addWidget( m_ignoreNamesButton );
  QGroupBox * settingsBox = new QGroupBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Settings" ) );
  settingsBox->setLayout( vBoxLayout );

  m_identityToGroupButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Identity Transforms to Groups" ) );
  vBoxLayout = new QVBoxLayout;
  vBoxLayout->addWidget( m_identityToGroupButton );
  QGroupBox * preprocessingBox = new QGroupBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Preprocessing" ) );
  preprocessingBox->setLayout( vBoxLayout );

  m_allButton = new QPushButton( QApplication::translate( VIEWER_APPLICATION_NAME, "All" ) );
  m_noneButton = new QPushButton( QApplication::translate( VIEWER_APPLICATION_NAME, "None" ) );
  QVBoxLayout * commandLayout = new QVBoxLayout;
  commandLayout->addWidget( m_allButton );
  commandLayout->addWidget( m_noneButton );

  QHBoxLayout * firstRowLayout = new QHBoxLayout;
  firstRowLayout->addWidget( settingsBox );
  firstRowLayout->addWidget( preprocessingBox );
  firstRowLayout->addStretch();
  firstRowLayout->addLayout( commandLayout );

  m_eliminateGroupsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Replace Groups by their children" ) );
  m_eliminateSingleChildGroupsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Replace single-child Groups by their child" ) );
  m_eliminateIndexSetsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Remove redundant IndexSets" ) );
  m_eliminateLODsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Replace LODs by their single child" ) );

  vBoxLayout = new QVBoxLayout;
  vBoxLayout->addWidget( m_eliminateGroupsButton );
  vBoxLayout->addWidget( m_eliminateSingleChildGroupsButton );
  vBoxLayout->addWidget( m_eliminateIndexSetsButton );
  vBoxLayout->addWidget( m_eliminateLODsButton );

  m_eliminateBox = new QGroupBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Eliminate degenerated or redundant objects" ) );
  m_eliminateBox->setCheckable( true );
  m_eliminateBox->setLayout( vBoxLayout );

  m_combineGeoNodesButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Combine GeoNodes" ) );
  m_combineLODsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Combine LODs" ) );
  m_combineLODRangesButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Combine LOD Ranges" ) );
  m_combineTransformsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Combine Transforms" ) );

  vBoxLayout = new QVBoxLayout;
  vBoxLayout->addWidget( m_combineGeoNodesButton );
  vBoxLayout->addWidget( m_combineLODsButton );
  vBoxLayout->addWidget( m_combineLODRangesButton );
  vBoxLayout->addWidget( m_combineTransformsButton );

  m_combineBox = new QGroupBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Combine compatible Objects" ) );
  m_combineBox->setCheckable( true );
  m_combineBox->setLayout( vBoxLayout );

  QVBoxLayout * firstColLayout = new QVBoxLayout;
  firstColLayout->addWidget( m_eliminateBox );
  firstColLayout->addWidget( m_combineBox );

  m_unifyEffectDataButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify EffectData" ) );
  m_unifyGeoNodesButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify GeoNodes" ) );
  m_unifyGroupsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify Groups" ) );
  m_unifyIndexSetsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify IndexSets" ) );
  m_unifyLODsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify LODs" ) );
  m_unifyParameterGroupDataButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify ParameterGroupData" ) );
  m_unifyPrimitivesButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify Primitives" ) );
  m_unifySamplersButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify Samplers" ) );
  m_unifyTexturesButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify Textures" ) );
  m_unifyTrafoAnimationsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify TrafoAnimations" ) );
  m_unifyVertexAttributeSetsButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify VertexAttributeSets" ) );
  m_unifyVerticesButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify Vertices" ) );

  m_epsilonEdit = new QLineEdit();
  QFormLayout * formLayout = new QFormLayout;
  formLayout->addRow( QApplication::translate( VIEWER_APPLICATION_NAME, "Epsilon:" ), m_epsilonEdit );

  vBoxLayout = new QVBoxLayout;
  vBoxLayout->addWidget( m_unifyEffectDataButton );
  vBoxLayout->addWidget( m_unifyGeoNodesButton );
  vBoxLayout->addWidget( m_unifyGroupsButton );
  vBoxLayout->addWidget( m_unifyIndexSetsButton );
  vBoxLayout->addWidget( m_unifyLODsButton );
  vBoxLayout->addWidget( m_unifyParameterGroupDataButton );
  vBoxLayout->addWidget( m_unifyPrimitivesButton );
  vBoxLayout->addWidget( m_unifySamplersButton );
  vBoxLayout->addWidget( m_unifyTexturesButton );
  vBoxLayout->addWidget( m_unifyTrafoAnimationsButton );
  vBoxLayout->addWidget( m_unifyVertexAttributeSetsButton );
  vBoxLayout->addWidget( m_unifyVerticesButton );
  vBoxLayout->addLayout( formLayout );

  m_unifyBox = new QGroupBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Unify equal Objects" ) );
  m_unifyBox->setCheckable( true );
  m_unifyBox->setLayout( vBoxLayout );

  QHBoxLayout * rowLayout = new QHBoxLayout;
  rowLayout->addLayout( firstColLayout );
  rowLayout->addWidget( m_unifyBox );

  QGroupBox * loopBox = new QGroupBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Optimizing Loop" ) );
  loopBox->setLayout( rowLayout );

  m_vertexCacheOptimizeButton = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Optimize Vertex Caches" ) );
  vBoxLayout = new QVBoxLayout;
  vBoxLayout->addWidget( m_vertexCacheOptimizeButton );
  QGroupBox * postProcessingBox = new QGroupBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Postprocessing" ) );
  postProcessingBox->setLayout( vBoxLayout );

  QDialogButtonBox * dbb = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );
  dbb->button( QDialogButtonBox::Ok )->setDefault ( true );

  QVBoxLayout * mainLayout = new QVBoxLayout;
  mainLayout->addLayout( firstRowLayout );
  mainLayout->addWidget( loopBox );
  mainLayout->addWidget( postProcessingBox );
  mainLayout->addWidget( dbb );

  setLayout( mainLayout );
  adjustSize();
  setMinimumSize( size() );
  setMaximumSize( size() );

  switchAllButtons( true );
  m_vertexCacheOptimizeButton->setChecked( false );   // switch this off by default, as it's very expensive and doesn't give that much!

  // connect everything up
  connect( m_allButton, SIGNAL(clicked(bool)), this, SLOT(clickedAll(bool)) );
  connect( m_noneButton, SIGNAL(clicked(bool)), this, SLOT(clickedNone(bool)) );

  connect( dbb, SIGNAL(accepted()), this, SLOT(accept()) );
  connect( dbb, SIGNAL(rejected()), this, SLOT(reject()) );

  QString epsilon;
  epsilon.sprintf( "%.12f", std::numeric_limits<float>::epsilon() );
  m_epsilonEdit->setText( epsilon );
  m_epsilonEdit->setValidator( new QDoubleValidator(this) );
}

OptimizerDialog::~OptimizerDialog()
{
}

void OptimizerDialog::clickedAll( bool checked )
{
  switchAllButtons( true );
}

void OptimizerDialog::clickedNone( bool checked )
{
  switchAllButtons( false );
}

void OptimizerDialog::accept()
{
  dp::sg::algorithm::CombineTraverser::TargetMask combineFlags;
  if ( m_combineBox->isChecked() )
  {
    if ( m_combineGeoNodesButton->isChecked() )
    {
      combineFlags |= dp::sg::algorithm::CombineTraverser::Target::GEONODE;
    }
    if ( m_combineLODsButton->isChecked() )
    {
      combineFlags |= dp::sg::algorithm::CombineTraverser::Target::LOD;
    }
    if ( m_combineLODRangesButton->isChecked() )
    {
      combineFlags |= dp::sg::algorithm::CombineTraverser::Target::LOD_RANGES;
    }
    if ( m_combineTransformsButton->isChecked() )
    {
      combineFlags |= dp::sg::algorithm::CombineTraverser::Target::TRANSFORM;
    }
  }
  unsigned int eliminateFlags = 0;
  if ( m_eliminateBox->isChecked() )
  {
    if ( m_eliminateGroupsButton->isChecked() )
    {
      eliminateFlags |= dp::sg::algorithm::EliminateTraverser::ET_GROUP;
    }
    if ( m_eliminateSingleChildGroupsButton->isChecked() )
    {
      eliminateFlags |= dp::sg::algorithm::EliminateTraverser::ET_GROUP_SINGLE_CHILD;
    }
    if ( m_eliminateIndexSetsButton->isChecked() )
    {
      eliminateFlags |= dp::sg::algorithm::EliminateTraverser::ET_INDEX_SET;
    }
    if ( m_eliminateLODsButton->isChecked() )
    {
      eliminateFlags |= dp::sg::algorithm::EliminateTraverser::ET_LOD;
    }
  }
  unsigned int unifyFlags = 0;
  float epsilon = 0.0f;
  if ( m_unifyBox->isChecked() )
  {
    if ( m_unifyEffectDataButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_PIPELINE_DATA;
    }
    if ( m_unifyGeoNodesButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_GEONODE;
    }
    if ( m_unifyGroupsButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_GROUP;
    }
    if ( m_unifyIndexSetsButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_INDEX_SET;
    }
    if ( m_unifyLODsButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_LOD;
    }
    if ( m_unifyParameterGroupDataButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_PARAMETER_GROUP_DATA;
    }
    if ( m_unifyPrimitivesButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_PRIMITIVE;
    }
    if ( m_unifySamplersButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_SAMPLER;
    }
    if ( m_unifyTexturesButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_TEXTURE;
    }
    if ( m_unifyTrafoAnimationsButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_TRAFO_ANIMATION;
    }
    if ( m_unifyVertexAttributeSetsButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_VERTEX_ATTRIBUTE_SET;
    }
    if ( m_unifyVerticesButton->isChecked() )
    {
      unifyFlags |= dp::sg::algorithm::UnifyTraverser::UT_VERTICES;
      epsilon = m_epsilonEdit->text().toFloat();
    }
  }

  GetApp()->setOverrideCursor( Qt::WaitCursor );
  dp::sg::algorithm::optimizeScene( m_scene, m_ignoreNamesButton->isChecked(), m_identityToGroupButton->isChecked()
               , combineFlags, eliminateFlags, unifyFlags, epsilon
               , m_vertexCacheOptimizeButton->isChecked() );
  GetApp()->restoreOverrideCursor();

  QDialog::accept();
}

void OptimizerDialog::switchAllButtons( bool on )
{
  m_identityToGroupButton->setChecked( on );

  m_combineBox->setChecked( on );
  m_eliminateBox->setChecked( on );
  m_unifyBox->setChecked( on );

  m_combineGeoNodesButton->setChecked( on );
  m_combineLODsButton->setChecked( on );
  m_combineLODRangesButton->setChecked( on );
  m_combineTransformsButton->setChecked( on );

  m_eliminateGroupsButton->setChecked( on );
  m_eliminateSingleChildGroupsButton->setChecked( false );  // as a subset of ET_GROUPS, this isn't needed
  m_eliminateIndexSetsButton->setChecked( on );
  m_eliminateLODsButton->setChecked( on );

  m_unifyEffectDataButton->setChecked( on );
  m_unifyGeoNodesButton->setChecked( on );
  m_unifyGroupsButton->setChecked( on );
  m_unifyIndexSetsButton->setChecked( on );
  m_unifyLODsButton->setChecked( on );
  m_unifyParameterGroupDataButton->setChecked( on );
  m_unifyPrimitivesButton->setChecked( on );
  m_unifySamplersButton->setChecked( on );
  m_unifyTexturesButton->setChecked( on );
  m_unifyTrafoAnimationsButton->setChecked( on );
  m_unifyVertexAttributeSetsButton->setChecked( on );
  m_unifyVerticesButton->setChecked( on );

  m_vertexCacheOptimizeButton->setChecked( on );
}
