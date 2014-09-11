
find_package(OpenGL REQUIRED )
find_package(NVGLEW REQUIRED)

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

string( REGEX REPLACE "^.*/([^/]*)$" "\\1" CUR_TEST_MODULE_NAME ${CMAKE_CURRENT_SOURCE_DIR} )

include_directories(
  "${GLEW_INCLUDES}"
)

add_definitions(
  "-DCURRENT_MODULE_NAME=\"${CUR_TEST_MODULE_NAME}\""
  "-DCURRENT_MODULE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\""
  ${GLEW_DEFINITIONS}
)

add_library( ${LINK_NAME} SHARED
   ${LINK_SOURCES}
)

add_dependencies( ${LINK_NAME} DPTestManager DPHelpers )

target_link_libraries( ${LINK_NAME}
  ${GLEW_LIB}
  ${OPENGL_gl_LIBRARY}
  DPTcore
  DPUtil
  DPTRiX
  RiXCore
  RiXGL
  DPTestManager
  DPMath
  DPHelpers
  DPGL
)

