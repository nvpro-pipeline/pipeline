include(CopyFile)

if (WIN32)

  set (NV_LUA_VERSION "5.1")
  
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set ( NV_LUA_ARCH "amd64" CACHE STRING "Lua library architecture" )
  else ()
    set ( NV_LUA_ARCH "x86" CACHE STRING "Lua library architecture" )
  endif()
  
  if(MSVC10)
    set (NV_LUA_CRT "msvc2010")
  elseif(MSVC11)
    set (NV_LUA_CRT "msvc2012")
  elseif(CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    string(STRIP "${GCC_VERSION}" GCC_VERSION)
    set (NV_LUA_CRT "gcc-${GCC_VERSION}-mingw")
  else()
    message(FATAL_ERROR "Compiler version not supported")
  endif()

  set (NV_LUA_COMPILER "win32-${NV_LUA_CRT}-${NV_LUA_ARCH}" CACHE STRING "Lua compiler")
  
  file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/Lua/${NV_LUA_VERSION}/${NV_LUA_COMPILER}" NV_LUA_PATH_TMP )

  set( NV_LUA_ROOT_PATH ${NV_LUA_PATH_TMP} CACHE STRING "Path to Lua" )
  
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(LUA_LIBS "${NV_LUA_ROOT_PATH}/lib/liblua.dll.a" CACHE STRING "Lua libraries")
  else()
    set(LUA_LIBS  "${NV_LUA_ROOT_PATH}/lib/lua.lib" CACHE STRING "Lua libraries" )
  endif()
  
  set( LUA_INCLUDES "${NV_LUA_ROOT_PATH}/include" CACHE STRING "Lua includes" )
  
  MACRO( CopyLua target )
    if ("${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
      message("copy_file_if_changed( ${target} ${NV_LUA_ROOT_PATH}/bin/lua.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)" )
      copy_file_if_changed( ${target} "${NV_LUA_ROOT_PATH}/bin/lua.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)" )
    else()
      copy_file_if_changed( ${target} "${NV_LUA_ROOT_PATH}/bin/lua.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}" )
    endif()
  ENDMACRO()
  
endif()
