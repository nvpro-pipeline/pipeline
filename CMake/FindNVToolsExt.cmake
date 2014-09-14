include(CopyFile)

file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}" DP_3RDPARTY_PATH )
set( NV_TOOLS_EXT_ROOT "${DP_3RDPARTY_PATH}/NvToolsExt" )

if ( EXISTS "${NV_TOOLS_EXT_ROOT}" AND IS_DIRECTORY "${NV_TOOLS_EXT_ROOT}" )

  set( NV_TOOLS_EXT_INCLUDE_DIR "${NV_TOOLS_EXT_ROOT}/include" )

  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( NV_TOOLS_EXT_ARCH "x64" CACHE STRING "NvToolsExt library architecture" )
  else ()
    set( NV_TOOLS_EXT_ARCH "Win32" CACHE STRING "NvToolsExt library architecture" )
  endif()

  set( NV_TOOLS_EXT_BIN_DIR "${NV_TOOLS_EXT_ROOT}/bin/${NV_TOOLS_EXT_ARCH}" )
  set( NV_TOOLS_EXT_LIB_DIR "${NV_TOOLS_EXT_ROOT}/lib/${NV_TOOLS_EXT_ARCH}" )

  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( NV_TOOLS_EXT_DLL "${NV_TOOLS_EXT_BIN_DIR}/nvToolsExt64_1.dll" )
    set( NV_TOOLS_EXT_LIB "${NV_TOOLS_EXT_LIB_DIR}/nvToolsExt64_1.lib" )
  else ()
    set( NV_TOOLS_EXT_DLL "${NV_TOOLS_EXT_BIN_DIR}/nvToolsExt32_1.dll" )
    set( NV_TOOLS_EXT_LIB "${NV_TOOLS_EXT_LIB_DIR}/nvToolsExt32_1.lib" )
  endif()

  if( EXISTS "${NV_TOOLS_EXT_DLL}" )
    set(NV_TOOLS_EXT_FOUND TRUE)
  endif()

  MACRO( CopyNV_TOOLS_EXT target )
    if ("${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
      copy_file_if_changed( ${target} "${NV_TOOLS_EXT_DLL}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)" )
    else()
      copy_file_if_changed( ${target} "${NV_TOOLS_EXT_DLL}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" )
    endif()
  ENDMACRO()
else()
  macro( CopyNV_TOOLS_EXT target )
  endmacro()
endif()
