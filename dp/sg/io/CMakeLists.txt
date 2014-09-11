cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0020 OLD)

FILE (GLOB linkunits ${CMAKE_CURRENT_SOURCE_DIR}/*)

FOREACH( linkunit ${linkunits} )
  if( IS_DIRECTORY ${linkunit} )
    if( EXISTS ${linkunit}/CMakeLists.txt )
      string( REGEX REPLACE "^.*/([^/]*)$" "\\1" LINK_NAME ${linkunit} )
      add_subdirectory( ${LINK_NAME} )
    endif()
  endif()
ENDFOREACH( linkunit ${linkunits} )


#includes

#definitions
add_definitions(
  -DDP_SG_IO_EXPORTS
)

#sources
set(SOURCES
  src/IO.cpp
  src/PlugInterface.cpp
)

set(PRIVATE_HEADERS
)

set(PUBLIC_HEADERS
  Config.h
  IO.h
  PlugInterface.h
  PlugInterfaceID.h
)

source_group(source FILES ${SOURCES})
source_group(header FILES ${PRIVATE_HEADERS})
source_group(""     FILES ${PUBLIC_HEADERS})

#target
add_library( DPSgIO SHARED
  ${SOURCES}
  ${HEADERS}
  ${PUBLIC_HEADERS}
)

add_perforce_to_target( DPSgIO )

target_link_libraries( DPSgIO
  DPSgCore
  DPSgUI
  DPUtil
)

set_target_properties( DPSgIO PROPERTIES FOLDER "SG" )
