#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  ${IL_INCLUDE_DIR}
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

target_link_libraries( ILTexSaver
  DPSgCore
  DPUtil
  DPSgIO
  ${IL_LIBRARIES}
)

set_target_properties( ILTexSaver PROPERTIES SUFFIX ".nxm" FOLDER "Savers")
