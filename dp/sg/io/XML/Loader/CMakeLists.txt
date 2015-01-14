find_package( NVTinyXML )

#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  include_directories( ${TINYXML_INCLUDE_DIRS} )
)

#definitions
add_definitions( 
  -DXMLLOADER_EXPORTS
)

if(WIN32)
  add_definitions("/wd4996")
endif()


#sources
set(XMLLOADER_SOURCES
  XMLLoader.cpp
)

set(XMLLOADER_HEADERS
  XMLLoader.h
)

source_group(source FILES ${XMLLOADER_SOURCES})
source_group(header FILES ${XMLLOADER_HEADERS})

#target
add_library( XMLLoader SHARED
  ${XMLLOADER_SOURCES}
  ${XMLLOADER_HEADERS}
)

target_link_libraries( XMLLoader
  DP
  DPSgCore
  DPMath
  DPUtil
  DPSgIO
  ${TINYXML_LIBRARIES}
)

set_target_properties( XMLLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders")
