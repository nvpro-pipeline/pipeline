include(CopyFile)

if (WIN32)

  set (NV_GLEW_VERSION "1.10.0")
  
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set ( NV_GLEW_ARCH "amd64" CACHE STRING "GLEW library architecture" )
  else ()
    set ( NV_GLEW_ARCH "x86" CACHE STRING "GLEW library architecture" )
  endif()
  
  set (NV_GLEW_COMPILER "win32-${NV_GLEW_ARCH}" CACHE STRING "GLEW compiler")
  
  file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/glew/${NV_GLEW_VERSION}" NV_GLEW_PATH_TMP )

  set( NV_GLEW_ROOT_PATH ${NV_GLEW_PATH_TMP} CACHE STRING "Path to glew" )
  
  #if ( CMAKE_COMPILER_IS_GNUCC )
  #  set(GLEW_LIBS "${NV_GLEW_ROOT_PATH}/lib/${NV_GLEW_COMPILER}/libglew.a" CACHE STRING "GLEW libraries")
  #else()
  set(GLEW_LIBS "${NV_GLEW_ROOT_PATH}/lib/${NV_GLEW_COMPILER}/glew32.lib" CACHE STRING "GLEW libraries" )
  #endif()
  
  set( GLEW_INCLUDES "${NV_GLEW_ROOT_PATH}/include" CACHE STRING "GLEW includes" )
  #set( GLEW_DEFINITIONS "-DGLEW_STATIC" CACHE STRING "GLEW definitions" )
  
  FUNCTION(CopyGLEW target path)
    copy_file_if_changed( ${target} "${NV_GLEW_ROOT_PATH}/bin/${NV_GLEW_COMPILER}/glew32.dll" "${path}" )
  ENDFUNCTION()

endif()

if (UNIX)
  find_path( GLEW_INCLUDES GL/glew.h
    /usr/include
    /usr/local/include
    /sw/include
    /opt/local/include
    DOC "The directory where GL/glew.h resides"
  )

  find_library( GLEW_LIBS
    NAMES GLEW glew
    PATHS
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    /sw/lib
    /opt/local/lib
    DOC "The GLEW library"
  )

  FUNCTION(CopyGLEW target path)
  ENDFUNCTION()

endif()
