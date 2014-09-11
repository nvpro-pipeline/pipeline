#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DNVSGLOADER_EXPORTS
)

if (MSVC)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj" )
endif()

#source
set( NVSGLOADER_SOURCES
  NVSGLoader.cpp
)

set( NVSGLOADER_HEADERS
  inc/NVSGLoader.h
)

source_group(source FILES ${NVSGLOADER_SOURCES})
source_group(header FILES ${NVSGLOADER_HEADERS})

#target
add_library( NVSGLoader SHARED
  ${NVSGLOADER_SOURCES}
  ${NVSGLOADER_HEADERS}
)

add_perforce_to_target( NVSGLoader )

target_link_libraries( NVSGLoader
  DP
  DPSgCore
  DPSgUI
  DPMath
  DPUtil
  DPFx
  DPSgIO
)

set_target_properties( NVSGLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders" )
