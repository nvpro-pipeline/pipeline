#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  "-DNBFLOADER_EXPORTS"
)

#sources
set(NBFLOADER_SOURCES
  NBFLoader.cpp
)

set(NBFLOADER_HEADERS
  inc/NBFLoader.h
  ../NBF.h
)

source_group(source FILES ${NBFLOADER_SOURCES})
source_group(header FILES ${NBFLOADER_HEADERS})

#target
add_library( NBFLoader SHARED
  ${NBFLOADER_SOURCES}
  ${NBFLOADER_HEADERS}
)

target_link_libraries( NBFLoader 
  DP
  DPSgCore
  DPSgUI
  DPMath
  DPUtil
  DPFx
  DPSgIO
)

set_target_properties( NBFLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders" )
