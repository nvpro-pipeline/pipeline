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

#include <QApplication>

#include "LogWidget.h"
#include "Preferences.h"
#include "ScriptSystem.h"
#include "ViewerCommand.h"
#include "MainWindow.h"

#include <dp/sg/ui/qt5/SceniXQGLWidget.h>

#define VIEWER_APPLICATION_NAME   "NVPro Pipeline Viewer"
#define VIEWER_APPLICATION_VENDOR "NVIDIA Corporation"

DEFINE_PTR_TYPES( viewerPlugInCallback );    // due to circular inclusion, we can't just include viewerPlugInCallback.h !!

class Viewer : public QApplication
{
  Q_OBJECT

  Q_PROPERTY( QString displayedSceneName READ getDisplayedSceneName )

  public:
    Viewer(int &, char **);
    ~Viewer();

    void startup();
    void log( const char * format, va_list valist, LogWidget::Severity severity ) const;
    dp::sg::ui::qt5::SceniXQGLWidget * getGlobalShareGLWidget() const;

    const dp::sg::core::SamplerSharedPtr & getEnvironmentSampler() const;

    void emitContinuousUpdate();
    void emitViewChanged();
    void emitMaterialChanged();
    void emitSceneTreeChanged();
    void executeCommand( ViewerCommand * command );

    Preferences * getPreferences() const;
    ScriptSystem * getScriptSystem() const;
    MainWindow * getMainWindow() const;
    const QString & getDisplayedSceneName() const;
    QUndoStack & getSceneStateUndoStack();
    const std::string & getRenderEngine() const;
    dp::fx::Manager getShaderManagerType() const;
    dp::culling::Mode getCullingMode() const;
    bool isBackdropEnabled() const;
    void setTonemapperEnabled( bool enabled );
    bool isTonemapperEnabled() const;
    const dp::sg::core::EffectDataSharedPtr & getEffectData( const std::string & effectName );
    bool holdsEffectData( const dp::sg::core::EffectDataSharedPtr & effectData );
    dp::sg::renderer::rix::gl::TransparencyMode getTransparencyMode() const;
    dp::sg::core::SceneSharedPtr getScene() const;
    dp::sg::ui::ViewStateSharedPtr const& getViewState() const;

    // if successful, will set displayedScene and displayedSceneName
    bool loadScene( const QString & fileName );
    bool saveScene( const QString & fileName ) const;
    void unloadScene();
    void outputStatistics();

    void runEventLoop();

  signals:
    void continuousUpdate();
    void viewChanged();
    void materialChanged();
    void environmentChanged(); // triggered after the environment sampler got updated
    void sceneChanged();
    void sceneTreeChanged();

  private slots:
    void setEnvironmentEnabledChanged();
    void setEnvironmentTextureName( const QString & name );
    void runStartupFile();
    void runStartupScript();

  private:
    void parseCommandLine( int & argc, char ** argv );

  private:
    typedef std::map<std::string,dp::sg::core::EffectDataSharedPtr> EffectDataMap;

  private:
    bool                                          m_backdropEnabled;
    bool                                          m_tonemapperEnabled;
    dp::culling::Mode                             m_cullingMode;
    QString                                       m_displayedSceneName;
    EffectDataMap                                 m_effectDataLibrary;
    dp::sg::core::SamplerSharedPtr                m_environmentSampler;
    dp::sg::ui::qt5::SceniXQGLWidget            * m_globalShareGLWidget; // Used to share a GL context among all renderers.
    MainWindow                                  * m_mainWindow;
    QUndoStack                                    m_parameterUndoStack;
    Preferences                                 * m_preferences;
    std::string                                   m_renderEngine;
    QString                                       m_runScript;
    QUndoStack                                    m_sceneStateUndoStack;
    ScriptSystem                                * m_scriptSystem;
    QTimer                                        m_scriptTimer;
    dp::fx::Manager                               m_shaderManagerType;
    QString                                       m_startupFile;
    viewerPlugInCallbackSharedPtr                 m_viewerPlugInCallback;
    dp::sg::ui::ViewStateSharedPtr                m_viewState;
    int                                           m_width;
    int                                           m_height;
};

inline Viewer * GetApp()
{
  return static_cast< Viewer * >( qApp );
}

inline dp::sg::ui::qt5::SceniXQGLWidget * Viewer::getGlobalShareGLWidget() const
{
  return m_globalShareGLWidget; 
}

inline const QString & Viewer::getDisplayedSceneName() const
{
  return m_displayedSceneName;
}

inline QUndoStack & Viewer::getSceneStateUndoStack()
{
  return m_sceneStateUndoStack;
}

inline const dp::sg::core::SamplerSharedPtr & Viewer::getEnvironmentSampler() const
{
  return( m_environmentSampler );
}

inline ScriptSystem * Viewer::getScriptSystem() const
{
  return m_scriptSystem;
}

inline MainWindow * Viewer::getMainWindow() const
{
  return m_mainWindow;
}

inline Preferences * Viewer::getPreferences() const
{
  return m_preferences;
}

inline const std::string & Viewer::getRenderEngine() const
{
  return( m_renderEngine );
}

inline dp::fx::Manager Viewer::getShaderManagerType() const
{
  return( m_shaderManagerType );
}

inline dp::culling::Mode Viewer::getCullingMode() const
{
  return( m_cullingMode );
}

inline dp::sg::renderer::rix::gl::TransparencyMode Viewer::getTransparencyMode() const
{
  return( dp::sg::renderer::rix::gl::TransparencyMode( m_preferences->getTransparencyMode() ) );
}

inline bool Viewer::isBackdropEnabled() const
{
  return( m_backdropEnabled );
}

inline bool Viewer::isTonemapperEnabled() const
{
  return( m_tonemapperEnabled );
}

inline dp::sg::core::SceneSharedPtr Viewer::getScene() const
{
  return( m_viewState ? m_viewState->getScene() : dp::sg::core::SceneSharedPtr() );
}

inline dp::sg::ui::ViewStateSharedPtr const& Viewer::getViewState() const
{
  return( m_viewState );
}


// some helpers
inline void ExecuteCommand( ViewerCommand * command )
{
  GetApp()->executeCommand( command );
}

inline Preferences * GetPreferences()
{
  return GetApp()->getPreferences();
}

inline QUndoStack & GetSceneStateUndoStack()
{
  return GetApp()->getSceneStateUndoStack();
}
