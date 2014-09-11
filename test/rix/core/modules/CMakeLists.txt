#Automatically adds all directories not starting with "__". No need to add tests here manually.

find_package(NVGLEW REQUIRED )

#includes
include_directories(
  "${DP_HOME}/dp/rix/core"
  "${DP_HOME}/dp/rix/gl"
  "${DP_HOME}/dp/rix/gl/test/framework/inc"
  "${DP_HOME}/dp/testfw/main/inc/" 
  "${DP_HOME}/dp/util/" 
  "${GLEW_ROOT}/include"
  #### Add additional includes here
)

###############################################################################
# Detect pieces of the bin/lib path
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  add_definitions("-DX86_64")
else ()
  set ( MSVC_CXX_FLAGS "/arch:SSE2" )
endif()

if(MSVC10)
  set (CRT "crt100")
  set ( COMPILER_ARCH "win32-msvc2010-${DP_ARCH}" )
  #enable multiprocessor build on visual studio 2010
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP ${MSVC_CXX_FLAGS}")
elseif(MSVC11)
  set (CRT "crt110")
  set ( COMPILER_ARCH "win32-msvc2012-${DP_ARCH}" )
  #enable multiprocessor build on visual studio 2012
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP ${MSVC_CXX_FLAGS}")
else()
  message(FATAL_ERROR "Compiler versio not supported")
endif()


FILE (GLOB list *)

FOREACH( test ${list} )
  if(IS_DIRECTORY ${test})
    string(REGEX REPLACE "^.*/([^/]*)$" "\\1" TEST_NAME ${test})
    
    #definitions
    add_definitions(
      "-DDPTTEST_EXPORTS"
      "-DTESTNAME=${TEST_NAME}"
      "-DCONFIGURATION=${OS}-${DP_ARCH}-${CRT}-${CNF}"
    )

    if(${TEST_NAME} MATCHES "^[^_][^_].*")
      if ( NOT ${TEST_NAME} STREQUAL "CMakeFiles")
        if ( EXISTS "${test}/CMakeLists.txt" )
          add_subdirectory( ${TEST_NAME} )
        endif( EXISTS "${test}/CMakeLists.txt" )
      endif ( NOT ${TEST_NAME} STREQUAL "CMakeFiles")
    endif(${TEST_NAME} MATCHES "^[^_][^_].*")
  endif(IS_DIRECTORY ${test})
ENDFOREACH( test ${list} )
 
