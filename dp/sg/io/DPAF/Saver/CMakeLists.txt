#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DDPAFSAVER_EXPORTS
  -D_CRT_SECURE_NO_WARNINGS
)

if (MSVC)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj" )
endif()

#sources
set( DPAFSAVER_SOURCES
  DPAFSaver.cpp
)

set( DPAFSAVER_HEADERS
  inc/DPAFSaver.h
)

source_group(source FILES ${DPAFSAVER_SOURCES})
source_group(header FILES ${DPAFSAVER_HEADERS})

#target
add_library( DPAFSaver SHARED
  ${DPAFSAVER_SOURCES}
  ${DPAFSAVER_HEADERS}
)

target_link_libraries( DPAFSaver 
  DP
  DPSgCore
  DPMath
  DPUtil
  DPFx
  DPSgIO
)

set_target_properties( DPAFSaver PROPERTIES SUFFIX ".nxm" FOLDER "Savers" )
