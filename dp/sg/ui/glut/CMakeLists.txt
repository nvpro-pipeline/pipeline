find_package( GLEW REQUIRED )
find_package( GLUT REQUIRED)

#includes
include_directories(
  ${GLEW_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIR}
)

#definitions
add_definitions(
  -DDP_SG_UI_GLUT_EXPORTS
  ${GLEW_DEFINITIONS}
)

#sources
set(SOURCES
  src/SceneRendererWidget.cpp
  src/Widget.cpp
)

set(PRIVATE_HEADERS
)

set(PUBLIC_HEADERS
  Config.h
  SceneRendererWidget.h
  Widget.h
)

source_group(source FILES ${SOURCES})
source_group(header FILES ${PRIVATE_HEADERS})
source_group(""     FILES ${PUBLIC_HEADERS})

#target
add_library( DPSgUIGLUT SHARED
  ${SOURCES}
  ${HEADERS}
  ${PUBLIC_HEADERS}
)

target_link_libraries( DPSgUIGLUT
  DPSgCore
  DPSgUI
  DPGL
  ${GLEW_LIBRARY}
  ${GLUT_LIBRARIES}
  ${OPENGL_gl_LIBRARY}
)

CopyGLEW( DPSgUIGLUT "${DP_BINARY_PATH}" )
CopyGLUT( DPSgUIGLUT "${DP_BINARY_PATH}" )

set_target_properties( DPSgUIGLUT PROPERTIES FOLDER "SG" )
