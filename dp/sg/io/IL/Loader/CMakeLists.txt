#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  ${DEVIL_INCLUDES}
)

#definitions
add_definitions(
  -DILTEXLOADER_EXPORTS
  ${DEVIL_DEFINITIONS}
)

#sources
set(ILTEXLOADER_SOURCES
  ILTexLoader.cpp
)

set(ILTEXLOADER_HEADERS
  ILTexLoader.h
)

source_group(source FILES ${ILTEXLOADER_SOURCES})
source_group(header FILES ${ILTEXLOADER_HEADERS})

#target
add_library( ILTexLoader SHARED
  ${ILTEXLOADER_SOURCES}
  ${ILTEXLOADER_HEADERS}
)

add_perforce_to_target( ILTexLoader )

target_link_libraries( ILTexLoader
  DPSgCore
  DPSgUI
  DPMath
  DPUtil
  DPFx
  DPSgIO
  ${DEVIL_LIBS}
)

set_target_properties( ILTexLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders" )
