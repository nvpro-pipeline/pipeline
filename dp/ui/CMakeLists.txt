find_package( OpenGL REQUIRED )

#includes
include_directories(
  ${GLEW_INCLUDE_DIRS}
)
#definitions
add_definitions(
  -DDP_UI_EXPORTS 
  ${GLEW_DEFINITIONS}
)  

#sources
set(SOURCES
  src/RenderTarget.cpp
)

set(PUBLIC_HEADERS
  Config.h
  RenderTarget.h
)

set(PRIVATE_HEADERS
)

source_group(sources FILES ${SOURCES})
source_group(headers FILES ${PRIVATE_HEADERS})
source_group("" FILES ${PUBLIC_HEADERS})

#target
add_library( DPUI SHARED
  ${SOURCES}
  ${PUBLIC_HEADERS}
  ${PRIVATE_HEADERS}
)

target_link_libraries( DPUI
  DPUtil
)

set_target_properties( DPUI PROPERTIES FOLDER "DP" )
