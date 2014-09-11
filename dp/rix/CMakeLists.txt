cmake_minimum_required(VERSION 2.8)

include( Perforce )

project( RiX )

SET( BUILD_DPT ON CACHE BOOL "Build DPT framework" )

option( RIX_BUILD_RIXFX "Build RiXFx with RiX" ON )
option( RIX_BUILD_RIXGL "Build RiXGL with RiX" ON )

if (CMAKE_COMPILER_IS_GNUCC)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()
         
if ( UNIX )
  set( RENDERERAPI_3RDPARTY_PATH "$ENV{SCENIXHOME}/../../buildtools/Linux" CACHE STRING "Devtech platform party libraries & compilers" )
elseif ( WIN32 )
  if ( "$ENV{DP_3RDPARTY_PATH}" STREQUAL "")
    set( RENDERERAPI_3RDPARTY_PATH "${DP_HOME}/../../3rdparty" CACHE STRING "Devtech platform party libraries & compilers" )
  else()
    set( RENDERERAPI_3RDPARTY_PATH ${DP_3RDPARTY_PATH} )
  endif()
endif(UNIX)

if (WIN32)
  add_definitions("-DNOMINMAX")
endif(WIN32)

include_directories( ${DP_HOME} )
add_subdirectory( core )

if( RIX_BUILD_RIXGL )
  add_subdirectory( gl )
endif()

if( RIX_BUILD_RIXFX )
  add_subdirectory( fx )
endif()
