#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

#definitions
add_definitions(
  -DPLYLOADER_EXPORTS
)

if(WIN32)
  add_definitions("/wd4996")
endif()


#sources
set(PLYLOADER_SOURCES
  PLYLoader.cpp
)

set(PLYLOADER_HEADERS
  PLYLoader.h
)

source_group(source FILES ${PLYLOADER_SOURCES})
source_group(header FILES ${PLYLOADER_HEADERS})

#target
add_library( PLYLoader SHARED
  ${PLYLOADER_SOURCES}
  ${PLYLOADER_HEADERS}
)

add_perforce_to_target( PLYLoader )

target_link_libraries( PLYLoader
  DP
  DPSgCore
  DPSgUI
  DPMath
  DPUtil
  DPFx
  DPSgIO
)

set_target_properties( PLYLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders" )
