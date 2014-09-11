#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  ${DEVIL_INCLUDES}
)

#definitions
add_definitions(
  -DILTEXSAVER_EXPORTS
  ${DEVIL_DEFINITIONS}
)

#sources
set(ILTEXSAVER_SOURCES
  ILTexSaver.cpp
)
set(ILTEXSAVER_HEADERS
  ILTexSaver.h
)

source_group(source FILES ${ILTEXSAVER_SOURCES})
source_group(header FILES ${ILTEXSAVER_HEADERS})

#target
add_library( ILTexSaver SHARED
  ${ILTEXSAVER_SOURCES}
  ${ILTEXSAVER_HEADERS}
)

add_perforce_to_target( ILTexSaver )

target_link_libraries( ILTexSaver
  DPSgCore
  DPSgUI
  DPSgAlgorithm
  DPUtil
  DPSgIO
  ${DEVIL_LIBS}
)

set_target_properties( ILTexSaver PROPERTIES SUFFIX ".nxm" FOLDER "Savers")
