// Copyright (c) 2013-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <QDir>
#include <QDrag>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QUrl>
#include "MaterialBrowser.h"
#include "Viewer.h"
#include <dp/fx/EffectLibrary.h>

MaterialBrowser::MaterialBrowser( const QString & title, QWidget * parent )
  : QDockWidget( title, parent )
{
  DP_VERIFY( dp::fx::EffectLibrary::instance()->loadEffects( "PreviewScene.xml" ) );

  setObjectName( title );
  setAcceptDrops( true );

  std::vector<std::string> extensions = dp::fx::EffectLibrary::instance()->getRegisteredExtensions();
  for (std::vector<std::string>::const_iterator it = extensions.begin(); it != extensions.end(); ++it)
  {
    m_filterList.push_back(QString("*") + it->c_str());
  }

  m_catalog = new QTreeWidget();
  m_catalog->setHeaderHidden( true );
  initMaterialCatalog( GetApp()->getPreferences()->getMaterialCatalogPath() );

  setWidget( m_catalog );
  connect( m_catalog, SIGNAL(itemCollapsed(QTreeWidgetItem*)), this, SLOT(catalogItemCollapsed(QTreeWidgetItem*)) );
  connect( m_catalog, SIGNAL(itemExpanded(QTreeWidgetItem*)), this, SLOT(catalogItemExpanded(QTreeWidgetItem*)) );
  connect( m_catalog, SIGNAL(itemPressed(QTreeWidgetItem*,int)), this, SLOT(catalogItemPressed(QTreeWidgetItem*,int)) );
  connect( GetApp()->getPreferences(), SIGNAL(materialCatalogPathChanged(QString const&)), this, SLOT(materialCatalogPathChanged(QString const&)) );
}

MaterialBrowser::~MaterialBrowser()
{
}

void MaterialBrowser::addChildren(QTreeWidgetItem* item, QString const& filePath)
{
  QFileInfoList fil = QDir(filePath).entryInfoList(m_filterList, QDir::AllDirs | QDir::Files | QDir::NoDotAndDotDot, QDir::DirsFirst);
  for (int i = 0; i<fil.size(); i++)
  {
    QString filePath = fil[i].absoluteFilePath();
    if (fil[i].isDir())
    {
      QTreeWidgetItem * childItem = new QTreeWidgetItem();
      childItem->setData(0, Qt::UserRole, filePath);
      childItem->setText(0, filePath.section('/', -1));
      childItem->setChildIndicatorPolicy(QTreeWidgetItem::ShowIndicator);
      if (item)
      {
        item->addChild(childItem);
      }
      else
      {
        m_catalog->addTopLevelItem(childItem);
      }
    }
    else
    {
      std::string fp = filePath.toStdString();
      dp::util::FileFinder fileFinder(GetApp()->getPreferences()->getMaterialCatalogPath().toStdString());
      fileFinder.addSearchPaths(GetApp()->getPreferences()->getSearchPathsAsStdVector());
      dp::fx::EffectLibrary::instance()->loadEffects(fp, fileFinder);

      std::vector<std::string> materialNames;
      dp::fx::EffectLibrary::instance()->getEffectNames(fp, dp::fx::EffectSpec::Type::PIPELINE, materialNames);
      for (std::vector<std::string>::const_iterator it = materialNames.begin(); it != materialNames.end(); ++it)
      {
        QTreeWidgetItem * childItem = new QTreeWidgetItem();
        childItem->setData(0, Qt::UserRole, QString(it->c_str()));
        childItem->setText(0, QString(dp::fx::stripNameSpaces(*it).c_str()));
        if (item)
        {
          item->addChild(childItem);
        }
        else
        {
          m_catalog->addTopLevelItem(childItem);
        }
      }
    }
  }
}

void MaterialBrowser::catalogItemCollapsed(QTreeWidgetItem * item)
{
  QList<QTreeWidgetItem*> children = item->takeChildren();
  for (int i = 0; i<children.size(); i++)
  {
    delete children[i];
  }
}

void MaterialBrowser::catalogItemExpanded(QTreeWidgetItem * item)
{
  GetApp()->setOverrideCursor( Qt::WaitCursor );

  addChildren(item, item->data(0, Qt::UserRole).toString());

  GetApp()->restoreOverrideCursor();
}

void MaterialBrowser::catalogItemPressed( QTreeWidgetItem * item, int column )
{
  if ( item )
  {
    QString materialName = item->data( column, Qt::UserRole ).toString();
    dp::fx::EffectSpecSharedPtr const& pipelineSpec = dp::fx::EffectLibrary::instance()->getEffectSpec( materialName.toStdString() );
    if ( pipelineSpec )
    {
      QMimeData * mimeData = new QMimeData;
      mimeData->setText( materialName );

      QDrag * drag = new QDrag( this );
      drag->setMimeData( mimeData );
      drag->exec();
    }
  }
}

void MaterialBrowser::initMaterialCatalog( QString const & path )
{
  addChildren(nullptr, path);
}

void MaterialBrowser::materialCatalogPathChanged( QString const& path )
{
  m_catalog->clear();
  initMaterialCatalog( path );
}
