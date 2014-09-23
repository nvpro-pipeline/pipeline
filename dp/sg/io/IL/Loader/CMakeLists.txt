#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  ${IL_INCLUDE_DIR}
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

target_link_libraries( ILTexLoader
  DPSgCore
  DPSgUI
  DPMath
  DPUtil
  DPFx
  DPSgIO
  ${IL_LIBRARIES}
)

set_target_properties( ILTexLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders" )
