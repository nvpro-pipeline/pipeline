
set( HELPERS_SOURCES
  src/Geometry.cpp
  src/GeometryHelper.cpp
  src/SimplexNoise1234.cpp
  src/TextureHelper.cpp
  src/Textures.cpp
)

set( HELPERS_HEADERS
  Geometry.h
  GeometryHelper.h
  SimplexNoise1234.h
  TextureHelper.h
  Textures.h
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
  
set_target_properties( DPHelpers PROPERTIES FOLDER "test/rix/core")