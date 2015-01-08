#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  "-DDPBFLOADER_EXPORTS"
)

#sources
set(DPBFLOADER_SOURCES
  DPBFLoader.cpp
)

set(DPBFLOADER_HEADERS
  inc/DPBFLoader.h
  ../DPBF.h
)

source_group(source FILES ${DPBFLOADER_SOURCES})
source_group(header FILES ${DPBFLOADER_HEADERS})

#target
add_library( DPBFLoader SHARED
  ${DPBFLOADER_SOURCES}
  ${DPBFLOADER_HEADERS}
)

target_link_libraries( DPBFLoader 
  DP
  DPSgCore
  DPSgUI
  DPMath
  DPUtil
  DPFx
  DPSgIO
)

set_target_properties( DPBFLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders" )
