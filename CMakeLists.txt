cmake_minimum_required(VERSION 2.8)

PROJECT( Pipeline )

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
include("dp")
include("dp3rdParty")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(WIN32)
  SET( BUILD_DPT ON CACHE BOOL "Build the test framework" )
endif(WIN32)

if(WIN32 AND MSVC_VERSION LESS 1700)
  SET( USE_CUDA ON CACHE BOOL "Use the CUDA library" )
else()
  SET( USE_CUDA OFF CACHE BOOL "Use the CUDA library" )
endif()

if( USE_CUDA )
  add_definitions("-DDP_USE_CUDA")
endif()

SET( USE_BOOST ON CACHE BOOL "Use the Boost 3rd party library" )

# check for DPHOME environment variable
if ( "$ENV{DPHOME}" STREQUAL "" )
  message( FATAL_ERROR "DPHOME environment variable not set!" )
endif()
file(TO_CMAKE_PATH "$ENV{DPHOME}" DP_HOME)

# check for DP_3RDPARTY_PATH environment variable
if ( "$ENV{DP_3RDPARTY_PATH}" STREQUAL "" )
  message( FATAL_ERROR "DP_3RDPARTY_PATH environment variable not set!" )
endif()
file(TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}" DP_3RDPARTY_PATH)

file(TO_CMAKE_PATH "$ENV{BUILD_TOOLS_DIR}" BUILD_TOOLS_DIR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake" "${DP_3RDPARTY_PATH}/CMake")

if ( "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  set(DP_ARCH "${CMAKE_SYSTEM_PROCESSOR}" CACHE STRING "CPU Architecture" )
elseif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set (DP_ARCH "amd64" CACHE STRING "CPU Architecture")
else ()
  set (DP_ARCH "x86" CACHE STRING "CPU Architecture")
endif()

find_package( Boost REQUIRED )

if(UNIX) 
  set(OS "linux")
  add_definitions("-DLINUX")
  add_definitions("-Wno-unused-local-typedefs -Wno-delete-non-virtual-dtor")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
else(UNIX)
  if(APPLE)
  else(APPLE)
    if(WIN32)
      set(OS "win")
      add_definitions("-DNOMINMAX")
    endif(WIN32)
  endif(APPLE)
endif(UNIX)

if ("${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
  set (DP_CONFIG "$(ConfigurationName)")
else()
  if( "${CMAKE_BUILD_TYPE}" STREQUAL "" )
    set (DP_CONFIG "release")
  endif()
endif()

if ( ${DP_ARCH} STREQUAL "armv7l" )
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mtune=cortex-a9 -mfloat-abi=hard -mfpu=neon" )
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mtune=cortex-a9 -mfloat-abi=hard -mfpu=neon")
endif()

if(UNIX)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
 
  if (DP_ARCH STREQUAL "amd64")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.1")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
  endif()
endif()

#set binary directory
if(WIN32 AND "${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib" )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" )
  else()
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib/${DP_CONFIG}" )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/${DP_CONFIG}" )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/${DP_CONFIG}" )
endif()

set(DP_BINARY_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${DP_CONFIG}" CACHE STRING "Devtech Platform binary path")

set(DP_TEST_LAUNCH_SITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release")

# some useful macros
MACRO(ADD_TARGET_PROPERTIES _target _name)

  SET(_properties)
  FOREACH(_prop ${ARGN})
    SET(_properties "${_properties} ${_prop}")
  ENDFOREACH(_prop)
  GET_TARGET_PROPERTY(_old_properties ${_target} ${_name})
  IF(NOT _old_properties)
    # in case it's NOTFOUND
    SET(_old_properties)
  ENDIF(NOT _old_properties)
  SET_TARGET_PROPERTIES(${_target} PROPERTIES ${_name} "${_old_properties} ${_properties}")

ENDMACRO(ADD_TARGET_PROPERTIES)

MACRO(TARGET_INCLUDE_SYMBOL target symbol)
  if (WIN32)
    if ( DP_ARCH STREQUAL "amd64" )
      add_target_properties( ${target} LINK_FLAGS /include:${symbol} )
    elseif( DP_ARCH STREQUAL "x86" )
      add_target_properties( ${target} LINK_FLAGS /include:_${symbol} )
    endif()
  endif()

  if(UNIX)
    add_target_properties( ${target} LINK_FLAGS "-Wl,--undefined=${symbol}" )
  endif()

ENDMACRO()

include_directories(${DP_HOME})
include_directories(${Boost_INCLUDE_DIR})


# find some libraries

#if (WIN32)
#  find_package(NVPMAPI REQUIRED)
#  add_definitions("-DHAVE_NVPMAPI")
#endif()

add_subdirectory( dp )

if( WIN32 AND BUILD_DPT )
  add_subdirectory( test )
endif()

add_subdirectory( apps )

