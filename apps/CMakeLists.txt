cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0020 OLD)

SET( BUILD_OPENGL_APPS ON  CACHE BOOL "Build the RiXGL based apps" )

if( BUILD_OPENGL_APPS )
  FILE (GLOB linkunits ${CMAKE_CURRENT_SOURCE_DIR}/*)

  FOREACH( linkunit ${linkunits} )
    if( IS_DIRECTORY ${linkunit} )
      if( EXISTS ${linkunit}/CMakeLists.txt )
        string( REGEX REPLACE "^.*/([^/]*)$" "\\1" LINK_NAME ${linkunit} )
        add_subdirectory( ${LINK_NAME} )
      endif()
    endif()
  ENDFOREACH( linkunit ${linkunits} )
endif()


