
set( HELPERS_SOURCES
  src/GeometryHelper.cpp
  src/TextureHelper.cpp
)

set( HELPERS_HEADERS
  GeometryHelper.h
  TextureHelper.h
)

add_definitions(
  "-DDPHELPERS_EXPORTS"
  "-D_CRT_SECURE_NO_WARNINGS"
)
  
source_group(headers FILES ${HELPERS_HEADERS})
source_group(sources FILES ${HELPERS_SOURCES})
  
add_library( DPHelpers SHARED
   ${HELPERS_SOURCES}
   ${HELPERS_HEADERS}
)

add_dependencies( DPHelpers RiXCore )

target_link_libraries( DPHelpers
  "${OPENGL_gl_LIBRARY}"
  ${GLEW_LIBRARY}
  DPTcore
  DPUtil
  DPTRiX
  RiXCore
  DPMath
)
  
set_target_properties( DPHelpers PROPERTIES FOLDER "DPT/RiX/Core")