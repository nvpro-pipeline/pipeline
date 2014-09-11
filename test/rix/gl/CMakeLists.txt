add_subdirectory( framework )

set( TEST_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/modules" )

FILE (GLOB linkunits ${TEST_DIRECTORY}/*)


FOREACH( linkunit ${linkunits} )


  if( IS_DIRECTORY ${linkunit} )
    if( EXISTS ${linkunit}/CMakeLists.txt )
      string( REGEX REPLACE "^.*/([^/]*)$" "\\1" LINK_NAME ${linkunit} )
      add_subdirectory( modules/${LINK_NAME} )
      set_target_properties( ${LINK_NAME} PROPERTIES SUFFIX ".dptest" FOLDER "DPT/RiX/GL/tests")
      add_dependencies( ${LINK_NAME} DPTRiXGL DPTestManager DPTapp )
      add_perforce_to_target( ${LINK_NAME} )
    endif()
  endif()
ENDFOREACH( linkunit ${linkunits} )
