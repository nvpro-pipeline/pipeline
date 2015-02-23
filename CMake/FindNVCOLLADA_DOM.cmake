if (WIN32)

  #set (NV_COLLADA_VERSION "1.7.8")
  
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set ( NV_COLLADA_ARCH "amd64" CACHE STRING "COLLADA_DOM library architecture" )
  else ()
    set ( NV_COLLADA_ARCH "x86" CACHE STRING "COLLADA_DOM library architecture" )
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
    message(FATAL_ERROR "Compiler version not supported")
  endif()

  set (NV_COLLADA_DOM_COMPILER "win32-${COMPILER}-${NV_COLLADA_ARCH}" CACHE STRING "COLLADA_DOM compiler")
  
  file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/COLLADA_DOM" NV_COLLADA_PATH_TMP )

  set( NV_COLLADA_DOM_ROOT_PATH ${NV_COLLADA_PATH_TMP} CACHE STRING "Path to COLLADA_DOM" )
  
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(COLLADA_DOM_LIBS "${NV_COLLADA_DOM_ROOT_PATH}/lib/${NV_COLLADA_DOM_COMPILER}/libCOLLADA_DOM.a"
                         "${NV_COLLADA_DOM_ROOT_PATH}/lib/${NV_COLLADA_DOM_COMPILER}/libxml.a" 
                         CACHE STRING "COLLADA_DOM libraries")
  else()
    set(COLLADA_DOM_LIBS optimized "${NV_COLLADA_DOM_ROOT_PATH}/lib/${NV_COLLADA_DOM_COMPILER}/COLLADA_DOM.lib"
                         optimized "${NV_COLLADA_DOM_ROOT_PATH}/lib/${NV_COLLADA_DOM_COMPILER}/xml.lib"
                         debug "${NV_COLLADA_DOM_ROOT_PATH}/lib/${NV_COLLADA_DOM_COMPILER}/COLLADA_DOMd.lib"
                         debug "${NV_COLLADA_DOM_ROOT_PATH}/lib/${NV_COLLADA_DOM_COMPILER}/xmld.lib"
                         CACHE STRING "COLLADA_DOM libraries" )
  endif()
  
  set( COLLADA_DOM_INCLUDES "${NV_COLLADA_DOM_ROOT_PATH}/include" CACHE STRING "COLLADA_DOM_DOM includes" )
  set( COLLADA_DOM_INCLUDES_1_3 "${NV_COLLADA_DOM_ROOT_PATH}/include/1.3" CACHE STRING "COLLADA_DOM_DOM 1.3 includes" )
  set( COLLADA_DOM_INCLUDES_1_4 "${NV_COLLADA_DOM_ROOT_PATH}/include/1.4" CACHE STRING "COLLADA_DOM_DOM 1.4 includes" )
  set( COLLADA_DOM_DEFINITIONS "" CACHE STRING "COLLADA_DOM_DOM definitions" )
endif()
