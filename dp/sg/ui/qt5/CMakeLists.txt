cmake_policy(SET CMP0020 NEW)

find_package( NVGLEW REQUIRED )
find_package( OpenGL REQUIRED )
find_package( NVQt5  QUIET )
find_package( Qt5Core REQUIRED )

#includes
include_directories(
  ${GLEW_INCLUDES}
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

add_perforce_to_target( DPSgUIQt5 )

target_link_libraries( DPSgUIQt5
  DPSgCore
  DPSgUI
  DPGL
  ${GLEW_LIBS}
  ${OPENGL_gl_LIBRARY}
)

qt5_use_modules(DPSgUIQt5 Widgets OpenGL X11Extras )

CopyGLEW( DPSgUIQt5 "${DP_BINARY_PATH}" )

set_target_properties( DPSgUIQt5 PROPERTIES FOLDER "SG" )


message("Qt5")