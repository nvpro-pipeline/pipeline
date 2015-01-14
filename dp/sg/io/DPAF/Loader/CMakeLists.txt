#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DDPAFLOADER_EXPORTS
)

if (MSVC)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj" )
endif()

#source
set( DPAFLOADER_SOURCES
  DPAFLoader.cpp
)

set( DPAFLOADER_HEADERS
  inc/DPAFLoader.h
)

source_group(source FILES ${DPAFLOADER_SOURCES})
source_group(header FILES ${DPAFLOADER_HEADERS})

#target
add_library( DPAFLoader SHARED
  ${DPAFLOADER_SOURCES}
  ${DPAFLOADER_HEADERS}
)

target_link_libraries( DPAFLoader 
  DP
  DPSgCore
  DPMath
  DPUtil
  DPFx
  DPSgIO
)

set_target_properties( DPAFLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders" )
