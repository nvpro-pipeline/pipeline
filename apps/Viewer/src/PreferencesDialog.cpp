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


#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QSpinBox>
#include "Viewer.h"
#include "PreferencesDialog.h"

PreferencesDialog::PreferencesDialog( QWidget * parent )
  : QDialog( parent )
{
  Preferences *preferences = GetPreferences();
  m_restoreSearchPaths            = preferences->getSearchPaths();
  m_restoreEnvironmentEnabled     = preferences->getEnvironmentEnabled();
  m_restoreEnvironmentTextureName = preferences->getEnvironmentTextureName();
  m_restoreMaterialCatalogPath    = preferences->getMaterialCatalogPath();


  m_environmentMapLabel = new QLabel( m_restoreEnvironmentTextureName );
  m_environmentMapLabel->setFrameShadow( QFrame::Sunken );
  m_environmentMapLabel->setFrameShape( QFrame::StyledPanel );

  QPushButton * environmentMapButton = new QPushButton( "..." );
  connect( environmentMapButton, SIGNAL(clicked(bool)), this, SLOT(selectEnvironmentMap(bool)) );

  QHBoxLayout * environmentMapLayout = new QHBoxLayout();
  environmentMapLayout->addWidget( m_environmentMapLabel );
  environmentMapLayout->addWidget( environmentMapButton );

  QFormLayout * environmentLayout = new QFormLayout();
  environmentLayout->addRow( "Environment Map", environmentMapLayout );

  QGroupBox * environmentBox = new QGroupBox( "Environment" );
  environmentBox->setCheckable( true );
  environmentBox->setChecked( m_restoreEnvironmentEnabled );
  environmentBox->setLayout( environmentLayout );
  connect( environmentBox, SIGNAL(toggled(bool)), this, SLOT(toggledEnvironmentBox(bool)) );


  m_materialCatalogLabel = new QLabel( m_restoreMaterialCatalogPath );
  m_materialCatalogLabel->setFrameShadow( QFrame::Sunken );
  m_materialCatalogLabel->setFrameShape( QFrame::StyledPanel );

  QPushButton * materialCatalogButton = new QPushButton( "..." );
  connect( materialCatalogButton, SIGNAL(clicked(bool)), this, SLOT(selectMaterialCatalogPath(bool)) );

  QHBoxLayout * materialCatalogLayout = new QHBoxLayout();
  materialCatalogLayout->addWidget( m_materialCatalogLabel );
  materialCatalogLayout->addWidget( materialCatalogButton );

  QFormLayout * materialLayout = new QFormLayout();
  materialLayout->addRow( "Material Catalog", materialCatalogLayout );

  QGroupBox * materialBox = new QGroupBox( "Material" );
  materialBox->setLayout( materialLayout );


  m_searchPaths = new QListWidget();
  m_searchPaths->addItems( m_restoreSearchPaths );

  QPushButton * addPathButton = new QPushButton( "Add ..." );
  connect( addPathButton, SIGNAL(clicked(bool)), this, SLOT(addPath(bool)) );

  QPushButton * removePathButton = new QPushButton( "Remove" );
  connect( removePathButton, SIGNAL(clicked(bool)), this, SLOT(removePath(bool)) );

  QPushButton * moveUpPathButton = new QPushButton( "Move Up" );
  connect( moveUpPathButton, SIGNAL(clicked(bool)), this, SLOT(moveUpPath(bool)) );

  QPushButton * moveDownPathButton = new QPushButton( "Move Down" );
  connect( moveDownPathButton, SIGNAL(clicked(bool)), this, SLOT(moveDownPath(bool)) );

  QHBoxLayout * searchPathsButtons = new QHBoxLayout();
  searchPathsButtons->addWidget( addPathButton );
  searchPathsButtons->addWidget( removePathButton );
  searchPathsButtons->addWidget( moveUpPathButton );
  searchPathsButtons->addWidget( moveDownPathButton );

  QVBoxLayout * verticalLayout = new QVBoxLayout();
  verticalLayout->addWidget( m_searchPaths );
  verticalLayout->addLayout( searchPathsButtons );

  QGroupBox * searchPathsBox = new QGroupBox( "Search Paths" );
  searchPathsBox->setLayout( verticalLayout );

  QDialogButtonBox * buttonBox = new QDialogButtonBox( QDialogButtonBox::Cancel | QDialogButtonBox::Ok );
  connect( buttonBox, SIGNAL(accepted()), this, SLOT(accept()) );
  connect( buttonBox, SIGNAL(rejected()), this, SLOT(reject()) );

  verticalLayout = new QVBoxLayout();
  verticalLayout->addWidget( environmentBox );
  verticalLayout->addWidget( materialBox );
  verticalLayout->addWidget( searchPathsBox );
  verticalLayout->addWidget( buttonBox );

  setWindowTitle( "Preferences" );
  setLayout( verticalLayout );
}

PreferencesDialog::~PreferencesDialog()
{
}

void PreferencesDialog::addPath(bool checked)
{
  // Open folder selection dialog.
  QString path = QFileDialog::getExistingDirectory(this, "Add Search Path");
  if (!path.isEmpty())
  {
    // Append it to the preferences search paths if it doesn't exist already.
    QStringList searchPaths = GetPreferences()->getSearchPaths();
    if (!searchPaths.contains(path))
    {
      searchPaths.append(path);
      GetPreferences()->setSearchPaths(searchPaths);

      // Add it to the search path QListWidget
      m_searchPaths->addItem(path);
    }
  }
}

void PreferencesDialog::removePath(bool checked)
{
  int row = m_searchPaths->currentRow();
  if (row != -1)
  {
    QStringList searchPaths = GetPreferences()->getSearchPaths();
    searchPaths.removeAt(row);
    GetPreferences()->setSearchPaths(searchPaths);
    
    // Could use takeItem() and delete here.
    m_searchPaths->clear();
    m_searchPaths->addItems(searchPaths);
  }
}

void PreferencesDialog::moveUpPath(bool checked)
{
  int row = m_searchPaths->currentRow();
  if (1 <= row)
  {
    QStringList searchPaths = GetPreferences()->getSearchPaths();
    QString path = searchPaths.at(row);
    searchPaths.removeAt(row);
    searchPaths.insert(row - 1, path);
    GetPreferences()->setSearchPaths(searchPaths);
    
    m_searchPaths->clear();
    m_searchPaths->addItems(searchPaths);
    m_searchPaths->setCurrentRow(row - 1);
  }
}

void PreferencesDialog::moveDownPath(bool checked)
{
  int row = m_searchPaths->currentRow();
  if (row != -1 && row < m_searchPaths->count() - 1)
  {
    QStringList searchPaths = GetPreferences()->getSearchPaths();
    QString path = searchPaths.at(row);
    searchPaths.removeAt(row);
    searchPaths.insert(row + 1, path);
    GetPreferences()->setSearchPaths(searchPaths);
    
    m_searchPaths->clear();
    m_searchPaths->addItems(searchPaths);
    m_searchPaths->setCurrentRow(row + 1);
  }
}

void PreferencesDialog::toggledEnvironmentBox( bool on )
{
  DP_ASSERT( GetPreferences()->getEnvironmentEnabled() != on );
  GetPreferences()->setEnvironmentEnabled( on );
}

void PreferencesDialog::selectEnvironmentMap( bool checked )
{
  QString textureFile = GetApp()->getMainWindow()->getTextureFile( dp::fx::PT_SAMPLER_2D );
  if ( ! textureFile.isEmpty() )
  {
    m_environmentMapLabel->setText( textureFile );
    GetPreferences()->setEnvironmentTextureName( textureFile );
  }
}

void PreferencesDialog::selectMaterialCatalogPath( bool checked )
{
  QString directory = QFileDialog::getExistingDirectory( this, "Select the path for the Material Catalog", GetPreferences()->getMaterialCatalogPath()
                                                       , QFileDialog::ShowDirsOnly | QFileDialog::HideNameFilterDetails );
  if ( !directory.isEmpty() )
  {
    m_materialCatalogLabel->setText( directory );
    GetPreferences()->setMaterialCatalogPath( directory );
  }
}

void PreferencesDialog::restore()
{
  Preferences *preferences = GetPreferences();

  preferences->setSearchPaths( m_restoreSearchPaths );
  preferences->setEnvironmentEnabled( m_restoreEnvironmentEnabled );
  preferences->setEnvironmentTextureName( m_restoreEnvironmentTextureName );
  preferences->setMaterialCatalogPath( m_restoreMaterialCatalogPath );
}
