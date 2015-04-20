
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


#pragma once

#include <QObject>
#include <QStringList>
#include <vector>

class Preferences : public QObject
{
  Q_OBJECT

  //
  // Convention:
  //
  // Add a Q_PROPERTY for each element.  Add getter/setter to slots, and add a signal for changed.
  //

  Q_PROPERTY( bool          depthPass               READ  getDepthPass              WRITE setDepthPass )
  Q_PROPERTY( bool          environmentEnabled      READ  getEnvironmentEnabled     WRITE setEnvironmentEnabled )
  Q_PROPERTY( QString       environmentTextureName  READ  getEnvironmentTextureName WRITE setEnvironmentTextureName )
  Q_PROPERTY( QString       materialCatalogPath     READ  getMaterialCatalogPath    WRITE setMaterialCatalogPath )
  Q_PROPERTY( float         normalsLineLength       READ  getNormalsLineLength      WRITE setNormalsLineLength )
  Q_PROPERTY( QString       sceneSelectionPath      READ  getSceneSelectionPath     WRITE setSceneSelectionPath );
  Q_PROPERTY( QStringList   searchPaths             READ  getSearchPaths            WRITE setSearchPaths )
  Q_PROPERTY( QString       textureSelectionPath    READ  getTextureSelectionPath   WRITE setTextureSelectionPath )
  Q_PROPERTY( unsigned int  transparencyMode        READ  getTransparencyMode       WRITE setTransparencyMode )

  public:
    Preferences( QObject * parent = 0 );
    ~Preferences();

  public:
    void setDepthPass( bool enabled );
    bool getDepthPass() const;

    void setEnvironmentEnabled( bool enabled );
    bool getEnvironmentEnabled() const;

    void setEnvironmentTextureName( const QString & name );
    QString getEnvironmentTextureName() const;

    void setMaterialCatalogPath( QString const& name );
    QString getMaterialCatalogPath() const;

    void setNormalsLineLength( float ll );
    float getNormalsLineLength() const;

    void setSceneSelectionPath( QString const& path );
    QString getSceneSelectionPath() const;

    void setSearchPaths( const QStringList & paths );
    QStringList getSearchPaths() const;
    std::vector<std::string> getSearchPathsAsStdVector() const;

    void setTextureSelectionPath( QString const& path );
    QString getTextureSelectionPath() const;

    void setTransparencyMode( unsigned int tm );
    unsigned int getTransparencyMode() const;

  private:
    void load();
    void save() const;

signals:
    void depthPassEnabled( bool enabled );
    void environmentEnabledChanged();
    void environmentTextureNameChanged( QString const& name );
    void materialCatalogPathChanged( QString const& name );
    void normalsLineLengthChanged( float len );
    void searchPathsChanged( QStringList const& paths );

  private:
    bool          m_depthPassEnabled;
    bool          m_environmentEnabled;
    QString       m_environmentTextureName;
    QString       m_materialCatalogPath;
    float         m_normalsLineLength;
    QString       m_sceneSelectionPath;
    QStringList   m_searchPaths;
    QString       m_textureSelectionPath;
    unsigned int  m_transparencyMode;
};

