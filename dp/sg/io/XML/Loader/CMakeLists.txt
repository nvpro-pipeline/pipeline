find_package( NVTinyXML )

#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  include_directories( ${TINYXML_INCLUDES} )
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
  nvsgXMLInterface.cpp
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

add_perforce_to_target( XMLLoader )

target_link_libraries( XMLLoader
  DP
  DPSgCore
  DPSgUI
  DPMath
  DPUtil
  DPSgIO
  ${TINYXML_LIBS}
)

set_target_properties( XMLLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders")
