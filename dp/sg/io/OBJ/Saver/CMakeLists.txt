#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DOBJSAVER_EXPORTS
  -D_CRT_SECURE_NO_WARNINGS
)

#sources
set(OBJSAVER_SOURCES
  ExtractGeometryTraverser.cpp
  OBJSaver.cpp
)

set(OBJSAVER_HEADERS
  inc/OBJSaver.h
  ExtractGeometryTraverser.h
)

source_group(source FILES ${OBJSAVER_SOURCES})
source_group(header FILES ${OBJSAVER_HEADERS})

#target
add_library( OBJSaver SHARED
  ${OBJSAVER_SOURCES}
  ${OBJSAVER_HEADERS}
)

target_link_libraries( OBJSaver
  DPSgCore
  DPSgUI
  DPSgAlgorithm
  DPMath
  DPUtil
  DPFx
  DPSgIO
)

set_target_properties( OBJSaver PROPERTIES SUFFIX ".nxm" FOLDER "Savers" )
