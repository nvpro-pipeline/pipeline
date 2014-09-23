
#includes

#definitions
add_definitions(
  -DDP_SG_GENERATOR_EXPORTS
)

#sources
set(SOURCES
  src/MeshGenerator.cpp
  src/GeoSphereScene.cpp
  src/PreviewScene.cpp
  src/SimpleScene.cpp
  src/Terrain.cpp
)

set(PRIVATE_HEADERS
)

set(PUBLIC_HEADERS
  Config.h
  MeshGenerator.h
  GeoSphereScene.h
  PreviewScene.h
  SimpleScene.h
  Terrain.h
)

source_group(source FILES ${SOURCES})
source_group(header FILES ${PRIVATE_HEADERS})
source_group(""     FILES ${PUBLIC_HEADERS})

#target
add_library( DPSgGenerator SHARED
  ${SOURCES}
  ${HEADERS}
  ${PUBLIC_HEADERS}
)

target_link_libraries( DPSgGenerator
  DP
  DPSgCore
  DPSgUI
  DPSgIO
)

set_target_properties( DPSgGenerator PROPERTIES FOLDER "SG" )
