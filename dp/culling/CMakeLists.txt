project(DPCulling)

cmake_policy(SET CMP0022 OLD)

add_definitions(
  "-DDP_CULLING_EXPORTS"
  "-D_CRT_SECURE_NO_WARNINGS"
)

add_subdirectory( cpu )
add_subdirectory( opengl )

set(PUBLIC_HEADERS
  Config.h
  GroupBitSet.h
  Manager.h
  ManagerBitSet.h
  ObjectBitSet.h
  ResultBitSet.h
)

set(HEADERS
)

#let cmake determine linker language
set(SOURCES
  src/GroupBitSet.cpp
  src/Manager.cpp
  src/ManagerBitSet.cpp
  src/ObjectBitSet.cpp
  src/ResultBitSet.cpp
)

source_group(sources FILES ${SOURCES})
source_group(headers FILES ${HEADERS})
source_group("" FILES ${PUBLIC_HEADERS})

add_library(DPCulling SHARED
  ${PUBLIC_HEADERS}
  ${HEADERS}
  ${SOURCES}
)

target_link_libraries( DPCulling DPCullingCPU DPCullingOpenGL )

set_property(TARGET DPCulling PROPERTY LINK_INTERFACE_LIBRARIES "")

set_target_properties( DPCulling PROPERTIES FOLDER "DP/Culling" )

