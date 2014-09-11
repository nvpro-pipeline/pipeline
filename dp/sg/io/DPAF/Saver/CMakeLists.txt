#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DNVSGSAVER_EXPORTS
  -D_CRT_SECURE_NO_WARNINGS
)

if (MSVC)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj" )
endif()

#sources
set( NVSGSAVER_SOURCES
  NVSGSaver.cpp
)

set( NVSGSAVER_HEADERS
  inc/NVSGSaver.h
)

source_group(source FILES ${NVSGSAVER_SOURCES})
source_group(header FILES ${NVSGSAVER_HEADERS})

#target
add_library( NVSGSaver SHARED
  ${NVSGSAVER_SOURCES}
  ${NVSGSAVER_HEADERS}
)

add_perforce_to_target( NVSGSaver )

target_link_libraries( NVSGSaver 
  DP
  DPSgCore
  DPSgUI
  DPSgAlgorithm
  DPMath
  DPUtil
  DPFx
  DPSgIO
)

set_target_properties( NVSGSaver PROPERTIES SUFFIX ".nxm" FOLDER "Savers" )
