#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DDPBFSAVER_EXPORTS
)

if(WIN32)
  add_definitions("/wd4996")
endif()

#sources
set(DPBFSAVER_SOURCES
  DPBFSaver.cpp
)

set(DPBFSAVER_HEADERS
  inc/DPBFSaver.h
  ../DPBF.h
)

source_group(source FILES ${DPBFSAVER_SOURCES})
source_group(header FILES ${DPBFSAVER_HEADERS})

#target
add_library( DPBFSaver SHARED
  ${DPBFSAVER_SOURCES}
  ${DPBFSAVER_HEADERS}
)

target_link_libraries( DPBFSaver 
  DP
  DPSgCore
  DPSgUI
  DPSgAlgorithm
  DPMath
  DPUtil
  DPFx
  DPSgAlgorithm
  DPSgIO
)

set_target_properties( DPBFSaver PROPERTIES SUFFIX ".nxm" FOLDER "Savers" )
