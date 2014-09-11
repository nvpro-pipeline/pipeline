set (TINYXMLVERSION 2.6.2)

if (WIN32)
  # Configure tinyxml directory
  if(MSVC10)
    set (COMPILER "win32-msvc2010-${DP_ARCH}")
  elseif(MSVC11)
    set (COMPILER "win32-msvc2012-${DP_ARCH}")
  elseif(CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    string(STRIP "${GCC_VERSION}" GCC_VERSION)
    set (COMPILER "mingw-gcc-${GCC_VERSION}-${DP_ARCH}")
  else()
    message(FATAL_ERROR "Compiler version not supported")
  endif()
  
  set( TINYXML_INCLUDES "$ENV{DP_3RDPARTY_PATH}/tinyxml/${TINYXMLVERSION}/include" )
  if (MSVC)
    set( TINYXML_LIBS optimized "$ENV{DP_3RDPARTY_PATH}/tinyxml/${TINYXMLVERSION}/lib/${COMPILER}/tinyxml.lib" 
                      debug     "$ENV{DP_3RDPARTY_PATH}/tinyxml/${TINYXMLVERSION}/lib/${COMPILER}/tinyxmld.lib" )
  elseif(CMAKE_COMPILER_IS_GNUCC)
    set( TINYXML_LIBS optimized "$ENV{DP_3RDPARTY_PATH}/tinyxml/${TINYXMLVERSION}/lib/${COMPILER}/libtinyxml.a" 
                      debug     "$ENV{DP_3RDPARTY_PATH}/tinyxml/${TINYXMLVERSION}/lib/${COMPILER}/libtinyxmld.a" )
  endif()
  
endif(WIN32)

if ( UNIX )
  set( TINYXML_LIBS tinyxml )
endif( UNIX )


