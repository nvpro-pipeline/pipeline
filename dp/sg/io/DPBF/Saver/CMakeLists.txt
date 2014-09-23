#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DNBFSAVER_EXPORTS
)

if(WIN32)
  add_definitions("/wd4996")
endif()

#sources
set(NBFSAVER_SOURCES
  NBFSaver.cpp
)

set(NBFSAVER_HEADERS
  inc/NBFSaver.h
  ../NBF.h
)

source_group(source FILES ${NBFSAVER_SOURCES})
source_group(header FILES ${NBFSAVER_HEADERS})

#target
add_library( NBFSaver SHARED
  ${NBFSAVER_SOURCES}
  ${NBFSAVER_HEADERS}
)

target_link_libraries( NBFSaver 
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

set_target_properties( NBFSaver PROPERTIES SUFFIX ".nxm" FOLDER "Savers" )
