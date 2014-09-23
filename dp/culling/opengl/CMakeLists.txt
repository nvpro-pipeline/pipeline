project(DPCullingOpenGL)

find_package(GLEW REQUIRED )

add_definitions(
  "-DDP_CULLING_EXPORTS"
  "-D_CRT_SECURE_NO_WARNINGS"
)

include_directories( ${GLEW_INCLUDE_DIRS} )

#definitions
add_definitions(
  ${GLEW_DEFINITIONS}
)

set(PUBLIC_HEADERS
  Manager.h
)

set(HEADERS
  inc/GroupImpl.h
  inc/ManagerImpl.h
)

#let cmake determine linker language
set(SOURCES
  src/GroupImpl.cpp
  src/ManagerImpl.cpp
)

source_group(sources FILES ${SOURCES})
source_group(headers FILES ${HEADERS})
source_group("" FILES ${PUBLIC_HEADERS})

add_library(DPCullingOpenGL STATIC
  ${PUBLIC_HEADERS}
  ${HEADERS}
  ${SOURCES}
)

CopyGLEW( DPCullingOpenGL "${DP_BINARY_PATH}" )

target_link_libraries( DPCullingOpenGL
  DPUtil
  DPMath
  ${OPENGL_gl_LIBRARY}
  ${GLEW_LIBRARY}
)

set_property(TARGET DPCullingOpenGL PROPERTY LINK_INTERFACE_LIBRARIES "")

set_target_properties( DPCullingOpenGL PROPERTIES FOLDER "Culling" )

if(UNIX)
  set_target_properties( DPCullingOpenGL PROPERTIES COMPILE_FLAGS -fPIC )
endif()
