if (WIN32)

  set (NV_DEVIL_VERSION "1.7.8")
  
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set ( NV_DEVIL_ARCH "amd64" CACHE STRING "DEVIL library architecture" )
  else ()
    set ( NV_DEVIL_ARCH "x86" CACHE STRING "DEVIL library architecture" )
  endif()
  
  if(MSVC10)
    set (COMPILER "crt100")
  elseif(MSVC11)
    set (COMPILER "crt110")
  elseif(CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    string(STRIP "${GCC_VERSION}" GCC_VERSION)
    set (COMPILER "gcc-${GCC_VERSION}-mingw")
  else()
    message(FATAL_ERROR "Compiler versio not supported")
  endif()

  set (NV_DEVIL_COMPILER "${COMPILER}-${NV_DEVIL_ARCH}" CACHE STRING "DEVIL compiler")
  
  file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/DevIL/${NV_DEVIL_VERSION}" NV_DEVIL_PATH_TMP )

  set( NV_DEVIL_ROOT_PATH ${NV_DEVIL_PATH_TMP} CACHE STRING "Path to DEVIL" )
  
  if ( CMAKE_COMPILER_IS_GNUCC )
    #set(DEVIL_LIBS "${NV_DEVIL_ROOT_PATH}/lib/${NV_DEVIL_COMPILER}/libDEVIL.a" CACHE STRING "DEVIL libraries")
  else()
    set(DEVIL_LIBS "${NV_DEVIL_ROOT_PATH}/lib/${NV_DEVIL_COMPILER}/DevIL.lib" CACHE STRING "DEVIL libraries" )
  endif()
  
  set( DEVIL_INCLUDES "${NV_DEVIL_ROOT_PATH}/inc" CACHE STRING "DEVIL includes" )
  #set( DEVIL_DEFINITIONS "-DDEVIL_STATIC" CACHE STRING "DEVIL definitions" )
  set( DEVIL_DEFINITIONS "" CACHE STRING "DEVIL definitions" )

  FUNCTION(CopyDevIL target path)
    copy_file_if_changed( ${target} "${NV_DEVIL_ROOT_PATH}/bin/${NV_DEVIL_COMPILER}/DevIL.dll" "${path}" )
  ENDFUNCTION() 

endif()

if (UNIX)
  set( DEVIL_LIBS IL CACHE STRING "DEVIL Libraries")

  FUNCTION(CopyDevIL target path)
  ENDFUNCTION() 

endif()
