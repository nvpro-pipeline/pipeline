// Copyright (c) 2009-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <QLabel>
#include <QMainWindow>
#include <QComboBox>

#include "CameraAnimator.h"
#include "LogWidget.h"
#include "MaterialBrowser.h"
#include "ScenePropertiesWidget.h"
#include "SceneTreeBrowser.h"
#include "ScriptWidget.h"
#include "UndoWidget.h"
#include "ViewportLayout.h"
#include "ViewerRendererWidget.h"
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/Path.h>

// X11 defines Bool and Qt doesn't like this Bool
#if defined(Bool)
#undef Bool
#endif

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  enum class MenuID
  {
    ADD_HEADLIGHT,
    ADD_LIGHT_SOURCE,
    ANTIALIASING,
    CONVERT_SCENE,
    CULLING,
    MODIFY_SCENE,
    RENDER_ENGINE,
    VIEWPORT_FORMAT,
    COUNT
  };

public:
  MainWindow();
  ~MainWindow();

public:
  virtual bool eventFilter( QObject * obj, QEvent * event );

public:
  QAction               * getContinuousRedrawAction() const;
  ViewerRendererWidget  * getCurrentViewport() const;
  QWidget               * getLog() const;
  QMenu                 * getMenu( MenuID id ) const;
  QLabel                * getStatisticsLabel() const;
  QString                 getTextureFile( unsigned int samplerType );
  void                    loadFile( const QString & fileName, bool openOrImport );

signals:
  void manipulatorChanged( ViewerRendererWidget::ManipulatorType );

protected:
  virtual void closeEvent( QCloseEvent * event );
  virtual void timerEvent( QTimerEvent * te );

private slots:
  void aboutToShowConvertSceneMenu();
  void aboutToShowEditMenu();
  void aboutToShowFileMenu();
  void aboutToShowViewMenu();
  void activeViewportChanged( int index, QWidget * widget );
  void sceneChanged();
  void selectObject( dp::sg::core::PathSharedPtr const& path );
  void toggledCameraCycle( bool onoff );
  void toggledCameraOrbitX( bool onoff );
  void toggledCameraOrbitY( bool onoff );
  void toggledCameraOrbitZ( bool onoff );
  void toggledContinuousRedraw( bool onoff );
  void toggledManipulator( bool );
  void triggeredAbout();
  void triggeredAnalyzeScene( bool checked );
  void triggeredClose();
  void triggeredDepthPass( bool checked );
  void triggeredDestripScene( bool checked );
  void triggeredNormalizeScene( bool checked );
  void triggeredNormalsDialog();
  void triggeredOpen();
  void triggeredOptimizeScene( bool checked );
  void triggeredPreferencesDialog( bool checked );
  void triggeredQuit();
  void triggeredRecentFile();
  void triggeredSave();
  void triggeredSceneStatistics( bool checked );
  void triggeredSmoothScene( bool checked );
  void triggeredStereoDialog();
  void triggeredStripScene( bool checked );
  void triggeredTonemapperDialog();
  void triggeredTransparencyDialog();
  void triggeredTriangulateScene( bool checked );
  void triggeredZoomAll();
  void viewportLayoutChanged( int index );

private:
  CameraAnimator        * createAnimator( ViewerRendererWidget * buddy );
  ViewerRendererWidget  * createRenderer( QWidget * parent, dp::sg::core::SceneSharedPtr const & scene );
  QObject               * getCurrentCameraAnimator() const;
  void                    initMultisampleMenu( QMenu * menu );
  void                    removeNormals();
  void                    restoreSettings( const QString & window = "" );
  void                    saveSettings( const QString & window = "" );
  void                    setCurrentFile( const QString & fileName, bool addToRecentList );
  void                    setupActions();
  bool                    setupDefaultViewState( dp::sg::ui::ViewStateSharedPtr const& viewState );
  void                    setupDockWidgets();
  void                    setupMenus();
  void                    setupStatusBar();
  void                    setupToolbar();
  void                    startStopTimer( bool &currentOpt, bool newOpt, bool dontStop );
  void                    updateRecentFileActions();
  void                    updateRenderers( dp::sg::ui::ViewStateSharedPtr const& viewState );

private:
  enum { MaxRecentFiles = 9 };

private:    // all the simple widgets we need to hold
  QAction   * m_analyzeSceneAction;
  QAction   * m_cameraIterationAction;
  QAction   * m_clearUndoStackAction;
  QAction   * m_closeAction;
  QAction   * m_continuousRedrawAction;
  QAction   * m_depthPassAction;
  QAction   * m_destripSceneAction;
  QAction   * m_manipulatorAction[size_t(ViewerRendererWidget::ManipulatorType::COUNT)];
  QAction   * m_normalsDialogAction;
  QAction   * m_openAction;
  QAction   * m_optimizeSceneAction;
  QAction   * m_orbitXAction;
  QAction   * m_orbitYAction;
  QAction   * m_orbitZAction;
  QAction   * m_quitAction;
  QAction   * m_recentFileAction[MaxRecentFiles];
  QAction   * m_redoAction;
  QAction   * m_saveAction;
  QAction   * m_sceneStatisticsAction;
  QAction   * m_separatorAction;
  QAction   * m_stereoDialogAction;
  QAction   * m_stripSceneAction;
  QAction   * m_tonemapperDialogAction;
  QAction   * m_transparencyDialogAction;
  QAction   * m_triangulateSceneAction;
  QAction   * m_undoAction;
  QAction   * m_viewportFormat30BitAction;
  QAction   * m_viewportFormatSRGBAction;
  QAction   * m_viewportFormatStencilAction;
  QAction   * m_viewportFormatStereoAction;
  QAction   * m_zoomAllAction;
  QComboBox * m_viewportCombo;
  QLabel    * m_fpsLabel;
  QLabel    * m_statisticsLabel;
  QMenu     * m_menus[size_t(MenuID::COUNT)];

private:
  std::vector<CameraAnimator *>         m_cameraAnimators;
  bool                                  m_continuousRedraw;
  int                                   m_continuousRedrawTimerID;
  int                                   m_currentViewport;
  int                                   m_fpsTimerID;
  unsigned int                          m_lastFrame;
  double                                m_lastTime;
  LogWidget                           * m_log;
  MaterialBrowser                     * m_materialBrowser;
  QColor                                m_normalsColor;
  bool                                  m_normalsDisplayed;
  std::vector<ViewerRendererWidget *>   m_renderWidgets;
  ScenePropertiesWidget               * m_sceneProperties;
  SceneTreeBrowser                    * m_sceneTreeBrowser;
  std::string                           m_sceneLoaderFilter;
  ScriptWidget                        * m_scriptSandbox;
  std::string                           m_textureLoaderFilter;
  dp::util::Timer                       m_timer;
  ViewportLayout                      * m_viewportLayout;
  UndoWidget                          * m_undo;
};
