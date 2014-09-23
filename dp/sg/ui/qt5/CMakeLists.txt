cmake_policy(SET CMP0020 NEW)

find_package( Qt5Widgets QUIET )

if( Qt5Widgets_FOUND )
  find_package( GLEW REQUIRED )
  find_package( OpenGL REQUIRED )

  #includes
  include_directories(
    ${GLEW_INCLUDE_DIRS}
  )

  #definitions
  add_definitions(
    -DDP_SG_UI_QT5_EXPORTS
    ${GLEW_DEFINITIONS}
  )

  #sources
  set(SOURCES
    src/SceniXQGLSceneRendererWidget.cpp
    src/SceniXQGLWidget.cpp
    src/SceniXQtUtil.cpp
  )

  set(PRIVATE_HEADERS
  )

  set(PUBLIC_HEADERS
    Config.h
    SceniXQGLSceneRendererWidget.h
    SceniXQGLWidget.h
    SceniXQtUtil.h
  )

  source_group(source FILES ${SOURCES})
  source_group(header FILES ${PRIVATE_HEADERS})
  source_group(""     FILES ${PUBLIC_HEADERS})

  #target
  add_library( DPSgUIQt5 SHARED
    ${SOURCES}
    ${HEADERS}
    ${PUBLIC_HEADERS}
  )

  target_link_libraries( DPSgUIQt5
    DPSgCore
    DPSgUI
    DPGL
    ${GLEW_LIBRARY}
    ${OPENGL_gl_LIBRARY}
  )

  if(UNIX)
    qt5_use_modules(DPSgUIQt5 Widgets X11Extras)
  else()
    qt5_use_modules(DPSgUIQt5 Widgets)
  endif()

  CopyGLEW( DPSgUIQt5 "${DP_BINARY_PATH}" )

  set_target_properties( DPSgUIQt5 PROPERTIES FOLDER "SG" )
else()
  message("Qt5Widgets not found, skipping dp::sg::ui::qt5")
endif()