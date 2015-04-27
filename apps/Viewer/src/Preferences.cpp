
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


#include "Preferences.h"
#include "Viewer.h"
#include <QMetaProperty>
#include <QProcessEnvironment>
#include <QSettings>
#include <dp/fx/EffectLibrary.h>
#include <dp/util/File.h>

Preferences::Preferences( QObject * parent )
  : QObject( parent )
  , m_environmentEnabled( true )
  , m_normalsLineLength( 1.f )
  , m_transparencyMode( dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_ALL )
{
  load();

  QProcessEnvironment pe = QProcessEnvironment::systemEnvironment();
  QString home = pe.contains( "DPHOME" ) ? pe.value( "DPHOME" ) : QString( "." );

  // Add the DPHOME media folders if m_searchPath is empty
  // which is the case on the very first program run.
  // Different paths can be added inside the Preferences Dialog.
  if ( !m_searchPaths.count() )
  {
    m_searchPaths.append( home + QString("/media/effects") );
    m_searchPaths.append( home + QString("/media/effects/xml") );
    m_searchPaths.append( home + QString("/media/textures") );
    m_searchPaths.append( home + QString("/media/textures/maxbench") );
  }

  if ( m_environmentTextureName.isEmpty() )
  {
    m_environmentTextureName = home + "/media/textures/spheremaps/spherical_checker.png";
  }
  if ( m_materialCatalogPath.isEmpty() )
  {
    m_materialCatalogPath = home + "/media/effects";
  }
  if ( m_sceneSelectionPath.isEmpty() )
  {
    m_sceneSelectionPath = home + QString( "/media/scenes" );
  }
  if ( m_textureSelectionPath.isEmpty() )
  {
    m_textureSelectionPath = home + QString( "/media/textures" );
  }
}

Preferences::~Preferences()
{
  save();
}

void Preferences::load()
{
  QSettings settings( VIEWER_APPLICATION_VENDOR, VIEWER_APPLICATION_NAME );
  const QMetaObject * mo = metaObject();
  for( int i = 0; i < mo->propertyCount(); i ++ )
  {
    QMetaProperty mp = mo->property( i );
    if( settings.contains( mp.name() ) )
    {
      mp.write( this, settings.value( mp.name() ) );
    }
  }
}

void Preferences::save() const
{
  QSettings settings( VIEWER_APPLICATION_VENDOR, VIEWER_APPLICATION_NAME );
  const QMetaObject * mo = metaObject();
  for( int i = 0; i < mo->propertyCount(); i ++ )
  {
    QMetaProperty mp = mo->property( i );
    settings.setValue( mp.name(), mp.read(this) );
  }
}

void Preferences::setDepthPass( bool enabled )
{
  if ( m_depthPassEnabled != enabled )
  {
    m_depthPassEnabled = enabled;
    emit depthPassEnabled( enabled );
  }
}

bool Preferences::getDepthPass() const
{
  return( m_depthPassEnabled );
}

void Preferences::setEnvironmentEnabled( bool enabled )
{
  if ( m_environmentEnabled != enabled )
  {
    m_environmentEnabled = enabled;
    emit environmentEnabledChanged();
  }
}

bool Preferences::getEnvironmentEnabled() const
{
  return( m_environmentEnabled );
}

void Preferences::setEnvironmentTextureName( const QString & name )
{
  if ( m_environmentTextureName != name )
  {
    m_environmentTextureName = name;
    emit environmentTextureNameChanged( name );
  }
}

QString Preferences::getEnvironmentTextureName() const
{
  return( m_environmentTextureName );
}

void Preferences::setMaterialCatalogPath( QString const& name )
{
  if ( m_materialCatalogPath != name )
  {
    m_materialCatalogPath = name;
    emit materialCatalogPathChanged( name );
  }
}

QString Preferences::getMaterialCatalogPath() const
{
  return( m_materialCatalogPath );
}

void Preferences::setNormalsLineLength( float len )
{
  if( len != m_normalsLineLength )
  {
    m_normalsLineLength = len;

    emit normalsLineLengthChanged( len );
  }
}

float Preferences::getNormalsLineLength() const
{
  return m_normalsLineLength;
}

void Preferences::setSceneSelectionPath( QString const& path )
{
  m_sceneSelectionPath = path;
}

QString Preferences::getSceneSelectionPath() const
{
  return( m_sceneSelectionPath );
}

void Preferences::setSearchPaths( const QStringList & paths )
{
  if( paths != m_searchPaths )
  {
    m_searchPaths = paths;

    emit searchPathsChanged( paths );
  }
}

QStringList Preferences::getSearchPaths() const
{
  return m_searchPaths;
}

std::vector<std::string> Preferences::getSearchPathsAsStdVector() const
{
  std::vector< std::string > searchPaths;

  for( int i = 0; i < m_searchPaths.count(); i ++ )
  {
    searchPaths.push_back( m_searchPaths.at(i).toStdString() );
    dp::util::convertPath( searchPaths.back() );
  }
  
  return searchPaths;
}

void Preferences::setTextureSelectionPath( QString const& path )
{
  m_textureSelectionPath = path;
}

QString Preferences::getTextureSelectionPath() const
{
  return( m_textureSelectionPath );
}

void Preferences::setTransparencyMode( unsigned int tm )
{
  m_transparencyMode = tm;
}

unsigned int Preferences::getTransparencyMode() const
{
  return( m_transparencyMode );
}
