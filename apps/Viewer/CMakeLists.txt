# Copyright NVIDIA Corporation 2010
# TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
# *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
# OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
# BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
# WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
# BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
# ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
# BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0020 NEW)

find_package( Qt5Widgets QUIET )

if(Qt5Widgets_FOUND AND TARGET DPSgUIQt5)
  find_package( OpenGL REQUIRED )
  find_package( GLEW REQUIRED )

  find_package( Boost COMPONENTS program_options REQUIRED )

  set( SCENIX_COMMON_BASE "${SCENIX_HOME}/samples/common" )
  ###############################################################################
  # Set include directories
  include_directories("inc")
  include_directories("res/ui")
  include_directories( ${GLEW_INCLUDE_DIRS} )

  add_definitions(
    ${GLEW_DEFINITIONS}
    "-D_CRT_SECURE_NO_WARNINGS"
  )

  # this one is for the UI files that were generated
  include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

  file( GLOB uisources
        "${SCENIX_COMMON_BASE}/src/ui/*.cpp" )

  set( MANIP_UI_SOURCES "${uisources}" )

  set (VIEWER_SOURCES
    src/CameraAnimator.cpp
    src/CommandAddItem.cpp
    src/CommandReplaceItem.cpp
    src/DisplayNormalsTraverser.cpp
    src/LogWidget.cpp
    src/main.cpp
    src/MainWindow.cpp
    src/MaterialBrowser.cpp
    src/NormalizeDialog.cpp
    src/NormalsDialog.cpp
    src/OptimizerDialog.cpp
    src/PlainTextDialog.cpp
    src/Preferences.cpp
    src/PreferencesDialog.cpp
    src/ScenePropertiesWidget.cpp
    src/SceneRendererPipeline.cpp
    src/SceneTreeBrowser.cpp
    src/SceneTreeItem.cpp
    src/ScriptSystem.cpp
    src/ScriptWidget.cpp
    src/SmoothDialog.cpp
    src/StereoDialog.cpp
    src/TonemapperDialog.cpp
    src/TransparencyDialog.cpp
    src/UndoWidget.cpp
    src/Viewer.cpp
    src/ViewerCommand.cpp
    src/ViewerRendererWidget.cpp
    src/ViewportLayout.cpp
    ${MANIP_UI_SOURCES}
  )

  set (VIEWER_MOC_HEADERS
    inc/CameraAnimator.h
    inc/Viewer.h
    inc/LogWidget.h
    inc/MainWindow.h
    inc/MaterialBrowser.h
    inc/NormalizeDialog.h
    inc/NormalsDialog.h
    inc/OptimizerDialog.h
    inc/PlainTextDialog.h
    inc/Preferences.h
    inc/PreferencesDialog.h
    inc/ScenePropertiesWidget.h
    inc/SceneTreeBrowser.h
    inc/ScriptSystem.h
    inc/ScriptWidget.h
    inc/SmoothDialog.h
    inc/StereoDialog.h
    inc/TonemapperDialog.h
    inc/TransparencyDialog.h
    inc/UndoWidget.h
    inc/ViewportLayout.h
    inc/ViewerRendererWidget.h
  )

  set(VIEWER_HEADERS
    inc/CommandAddItem.h
    inc/CommandReplaceItem.h
    inc/DisplayNormalsTraverser.h
    inc/Log.h
    inc/SceneRendererPipeline.h
    inc/SceneTreeItem.h
    inc/ViewerCommand.h
    inc/viewerPlugInCallback.h
    res/ui/StyleSheet.h
  )

  set(VIEWER_SHADERS
    res/highlightFragment.glsl
    res/highlightVertex.glsl
    res/stencilToColorFragment.glsl
    res/stencilToColorVertex.glsl
    res/tonemapFragment.glsl
    res/tonemapVertex.glsl
    res/viewerEffects.xml
  )

  set (VIEWER_RESOURCES_MODELS
  )

  set(VIEWER_QRC
    res/Viewer.qrc
  )

  set(VIEWER_RC_FILES
    res/Viewer.rc
  )

  set(VIEWER_QRC_SOURCES ${VIEWER_QRC})

  # geneates rules to build sources from resources
  QT5_ADD_RESOURCES(VIEWER_QRC_SOURCES ${VIEWER_QRC})

  QT5_WRAP_CPP(VIEWER_MOC_SOURCES ${VIEWER_MOC_HEADERS})
  QT5_WRAP_UI(VIEWER_UI ${VIEWER_UI_SOURCES})

  add_executable(Viewer
    ${VIEWER_SOURCES}
    ${VIEWER_UI}
    ${VIEWER_MOC_SOURCES}
    ${VIEWER_QRC_SOURCES}
    ${VIEWER_HEADERS}
    ${VIEWER_SHADERS}
    ${VIEWER_MOC_HEADERS}
    ${VIEWER_QML_SOURCES}
    ${VIEWER_RESOURCES_MODELS}
    ${VIEWER_RC_FILES}
  )

  qt5_use_modules(Viewer Widgets Script Help OpenGL)

  if (WIN32)
    if (LARGE_ADDRESS_AWARE)
      set_target_properties( Viewer PROPERTIES LINK_FLAGS /LARGEADDRESSAWARE )
    endif(LARGE_ADDRESS_AWARE)
  endif(WIN32)

  if (UNIX)
    set_target_properties( Viewer PROPERTIES LINK_FLAGS -rdynamic )
  endif(UNIX)


  source_group("Header Files" FILES ${VIEWER_HEADERS} ${VIEWER_MOC_HEADERS})
  source_group("Source Files" FILES ${VIEWER_SOURCES})
  source_group("Generated Files" FILES ${VIEWER_MOC_SOURCES} ${VIEWER_QRC_SOURCES})
  source_group("Resource Files" FILES ${VIEWER_QRC})
  source_group("Resource Files\\Models" FILES ${VIEWER_RESOURCES_MODELS})
  source_group("Shader Files" FILES ${VIEWER_SHADERS})

  target_link_libraries(Viewer
    ${OPENGL_gl_LIBRARY}
    ${QT_LIBRARIES}
    ${QT_QTMAIN_LIBRARY}
    ${QT_LIB_SCRIPT}
    ${GLEW_LIBRARY}
    ${Boost_LIBRARIES}
    DP
    DPSgCore
    DPSgGL
    DPSgRdrRiXGL
    DPSgIO
    DPSgUIManipulator
    DPSgUIQt5
  )

  add_dependencies( Viewer
    3DSLoader
    DAELoader
    EXRLoader
    FLTLoader
    HOOPSLoader
    ILTexLoader
    MBSDFLoader
    NBFLoader
    NVSGLoader
    OBJLoader
    PLYLoader
    WRLLoader
    XMLLoader
    CSFSaver
    ILTexSaver
    NBFSaver
    NVSGSaver
    OBJSaver
  )

  CopyGLEW( Viewer "${DP_BINARY_PATH}" )

  # Target for file format optimizer
  add_executable(makenbf
    src/makenbf.cpp
  )

  target_link_libraries(makenbf
    ${QT_LIBRARIES}
    DPSgCore
    DPSgIO
  )

  # TODO Is this really required for makenbf? Currently it is using QApplication
  qt5_use_modules(makenbf Widgets)

  #
  # Not sure we really need these any more, probably just for convenience.
  #

  # Target which copies necessary files from Qt to bin folder
  add_custom_target(CopyQtDlls ${CMAKE_COMMAND}
    -D "build_configuration:STRING=$(ConfigurationName)"
    -D "SCENIX_BIN_PATH:PATH=${SCENIX_BIN_PATH}"
    -D "BUILDDIR:PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    -D "CMAKE_GENERATOR=${CMAKE_GENERATOR}"
    -D "QTDIR:PATH=$ENV{QTDIR}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/copyqt.cmake"
    )

  set_target_properties( Viewer PROPERTIES FOLDER "Apps")
  set_target_properties( CopyQtDlls PROPERTIES FOLDER "Tools")
  set_target_properties( makenbf PROPERTIES FOLDER "Tools")
else()
  message("Qt5Widgets or DpSgUIQt5 not found, skipping Viewer")
endif()
