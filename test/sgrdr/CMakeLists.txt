add_subdirectory( framework )

set( TEST_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/modules" )

FILE (GLOB linkunits ${TEST_DIRECTORY}/*)


FOREACH( linkunit ${linkunits} )


  if (TARGET DPTSgRdr)
    if( IS_DIRECTORY ${linkunit} )
      if( EXISTS ${linkunit}/CMakeLists.txt )
        string( REGEX REPLACE "^.*/([^/]*)$" "\\1" LINK_NAME ${linkunit} )
        add_subdirectory( modules/${LINK_NAME} )
        set_target_properties( ${LINK_NAME} PROPERTIES SUFFIX ".dptest" FOLDER "DPT/SgRdr/tests")
        add_dependencies( ${LINK_NAME} DPTSgRdr DPTestManager DPTapp )
      endif()
    endif()
  endif()
ENDFOREACH( linkunit ${linkunits} )
