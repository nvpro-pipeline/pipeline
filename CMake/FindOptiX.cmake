set( OPTIX_PATH "${DP_3RDPARTY_PATH}/OptiX/3.0" )

if (WIN32)
  # Configure optix directory
  if(MSVC)
    set (COMPILER "win-${DP_ARCH}")
  elseif(CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    string(STRIP "${GCC_VERSION}" GCC_VERSION)
    set (COMPILER "mingw-gcc-${GCC_VERSION}-${DP_ARCH}")
  endif()

  set( OPTIX_INCLUDE_DIR "${OPTIX_PATH}/include" )
  set( OPTIX_BIN_DIR "${OPTIX_PATH}/bin/win-${DP_ARCH}" )
  set( OPTIX_LIB_DIR "${OPTIX_PATH}/lib/win-${DP_ARCH}" )
  set( OPTIX_LIBS 
    ${OPTIX_LIB_DIR}/optix.1.lib
    ${OPTIX_LIB_DIR}/optixu.1.lib
  )

  set( OPTIX_FOUND TRUE )
endif(WIN32)

if ( UNIX )
  message( STATUS "ERROR OptiX not yet supported for Linux.")
  # set ( COMPILER "gcc-4.1-${DP_ARCH}" )
endif( UNIX )

MACRO( CopyOptiX target )
  if ("${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
    if ("${DP_ARCH}" STREQUAL "amd64")
      copy_file_if_changed( ${target} "${OPTIX_BIN_DIR}/cudart64_42_9.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)" )
    else()
      copy_file_if_changed( ${target} "${OPTIX_BIN_DIR}/cudart32_42_9.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)" )
    endif()

    copy_file_if_changed( ${target} "${OPTIX_BIN_DIR}/optix.1.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)" )
    copy_file_if_changed( ${target} "${OPTIX_BIN_DIR}/optixu.1.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)" )
  else()
    message("FindOptiX.cmake: x86 not yet implemented")
    #copy_file_if_changed( ${target} "${NVPMAPI_BIN_DIR}/NvPmApi.DataProvider.Legacy.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}" )
  endif()
ENDMACRO()

