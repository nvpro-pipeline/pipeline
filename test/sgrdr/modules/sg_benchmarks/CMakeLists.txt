find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )

FILE (GLOB tests ${linkunit}/*)

set( LINK_SOURCES "" )

add_definitions(
  "-D_CRT_SECURE_NO_WARNINGS"
)

FOREACH( test ${tests} )
  if( IS_DIRECTORY ${test} )
    if( EXISTS ${test}/CMakeLists.txt )
      string( REGEX REPLACE "^.*/([^/]*)$" "\\1" TEST_NAME ${test} )
        if( NOT (${TEST_NAME} MATCHES "^__") )
          add_subdirectory( ${TEST_NAME} )
        endif()
    endif()
  endif()
ENDFOREACH( test ${tests} )

include_directories(
  "${GLEW_INCLUDE_DIRS}"
)

if (TARGET DPTSgRdr)
  add_library( ${LINK_NAME} SHARED
     ${LINK_SOURCES}
  )

  target_link_libraries( ${LINK_NAME}
    ${GLEW_LIBRARY}
    DPTcore
    DPUtil
    DPTRiX
    RiXCore
    RiXGL
    DPTestManager
    DPHelpers
    DPTSgRdr
    DPSgIO
    DPSgGenerator
    DPSgCore
    DPSgRdrRiXGL
  )

  add_dependencies( ${LINK_NAME} DPHelpers )

endif()

