project(DPCullingCPU)

add_definitions(
  "-DDP_CULLING_EXPORTS"
  "-D_CRT_SECURE_NO_WARNINGS"
)

set(PUBLIC_HEADERS
  Config.h
  Manager.h
)

set(HEADERS
  inc/ManagerImpl.h
)

#let cmake determine linker language
set(SOURCES
  src/ManagerImpl.cpp
)

source_group(sources FILES ${SOURCES})
source_group(headers FILES ${HEADERS})
source_group("" FILES ${PUBLIC_HEADERS})

add_library(DPCullingCPU STATIC
  ${PUBLIC_HEADERS}
  ${HEADERS}
  ${SOURCES}
)

target_link_libraries( DPCullingCPU DPUtil DPMath )

set_target_properties( DPCullingCPU PROPERTIES FOLDER "DP/Culling" )

if(UNIX)
  set_target_properties( DPCullingCPU PROPERTIES COMPILE_FLAGS -fPIC )
endif()
