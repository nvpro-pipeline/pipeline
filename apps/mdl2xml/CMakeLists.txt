# Copyright (c) 2013-2016, NVIDIA CORPORATION. All rights reserved.
# TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
# *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
# OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
# BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
# WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
# BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
# ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
# BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

cmake_minimum_required(VERSION 2.8)

project(MDL2XML)

find_package( Boost COMPONENTS program_options REQUIRED )
find_package( MDLSDK )
find_package( NVTinyXML)

if ( MDL_SDK_FOUND )

  add_definitions(
    "-D_CRT_SECURE_NO_WARNINGS"
    "-D_SCL_SECURE_NO_WARNINGS"
  )

  include_directories(
    ${MDL_SDK_INCLUDE_DIR}
    ${TINYXML_INCLUDE_DIRS}
  )

  set (HEADERS
    GLSLBuilder.h
    XMLBuilder.h
    ${DP_HOME}/dp/fx/mdl/inc/MaterialBuilder.h
    ${DP_HOME}/dp/fx/mdl/inc/MDLTokenizer.h
  )

  FILE( GLOB RESOURCES ${DP_HOME}/media/dpfx/mdl/*.glsl )
  set ( RESOURCES
    ${RESOURCES}
    ${DP_HOME}/media/dpfx/mdl/MDL.cfg
  )

  set (SOURCES
    GLSLBuilder.cpp
    mdl2xml.cpp
    XMLBuilder.cpp
    ${DP_HOME}/dp/fx/mdl/src/MaterialBuilder.cpp
    ${DP_HOME}/dp/fx/mdl/src/MDLTokenizer.cpp
  )

  add_executable(MDL2XML
    ${SOURCES}
    ${HEADERS}
    ${RESOURCES}
  )

  source_group(headers FILES ${HEADERS})
  source_group(resources FILES ${RESOURCES})
  source_group(sources FILES ${SOURCES})

  if (WIN32)
    if (LARGE_ADDRESS_AWARE)
      set_target_properties( MDL2XML PROPERTIES LINK_FLAGS /LARGEADDRESSAWARE )
    endif(LARGE_ADDRESS_AWARE)
    set_target_properties( MDL2XML PROPERTIES LINK_FLAGS_DEBUG "/NODEFAULTLIB:msvcrt" )
  endif(WIN32)

  if (UNIX)
    set_target_properties( Viewer PROPERTIES LINK_FLAGS -rdynamic )
  endif(UNIX)

  target_link_libraries(MDL2XML
    DP
    DPFx
    DPFxMDL
    DPUtil
    ${Boost_LIBRARIES}
    ${TINYXML_LIBRARIES}
  )

  CopyMDL_SDK( MDL2XML )

  set_target_properties( MDL2XML PROPERTIES FOLDER "Apps")

endif() # MDL_SDK_FOUND
