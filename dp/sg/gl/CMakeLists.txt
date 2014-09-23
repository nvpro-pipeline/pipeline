find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )
find_package( Boost COMPONENTS thread date_time )

#includes
include_directories(
  ${GLEW_INCLUDE_DIRS}
)
#definitions
add_definitions(
  -DDP_SG_GL_EXPORTS 
  ${GLEW_DEFINITIONS}
)  

#sources
set(SOURCES
  src/BufferGL.cpp
  src/TextureGL.cpp
)

set(PUBLIC_HEADERS
  BufferGL.h
  Config.h
  CoreTypes.h
  TextureGL.h
)

set(PRIVATE_HEADERS
)

source_group(sources FILES ${SOURCES})
source_group(headers FILES ${PRIVATE_HEADERS})
source_group("" FILES ${PUBLIC_HEADERS})

#target
add_library( DPSgGL SHARED
  ${SOURCES}
  ${PUBLIC_HEADERS}
  ${PRIVATE_HEADERS}
)

target_link_libraries( DPSgGL
  DPSgCore
  DPSgUI
  DPUtil
  DPGL
  ${GLEW_LIBRARY}
  ${Boost_LIBRARIES}
)

CopyGLEW( DPSgGL  "${DP_BINARY_PATH}" )


set_target_properties( DPSgGL PROPERTIES FOLDER "SG" )
