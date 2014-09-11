# Copyright NVIDIA Corporation 2009-2011
# TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
# *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
# OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
# BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
# WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
# BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
# ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
# BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

cmake_minimum_required(VERSION 2.6)

if(WIN32)
  project( GLUTMinimalCFR )

  find_package(OpenGL REQUIRED )
  find_package(NVGLUT REQUIRED )
  find_package(NVGLEW REQUIRED )

  find_package( NVBoost )
  find_package( Boost COMPONENTS program_options REQUIRED )

  set( SOURCES
    src/CFRPipeline.cpp
    src/GLUTMinimalCFR.cpp
    src/TextureTransfer.cpp
    main.cpp
  )

  set ( HEADERS
    inc/CFRPipeline.h
    inc/GLUTMinimalCFR.h
    inc/TextureTransfer.h
  )

  include_directories( "."
                       "inc"
                       ${Boost_INCLUDE_DIRS}
                       ${GLEW_INCLUDES}
                       ${GLUT_INCLUDE_DIR}
                     )

  add_definitions( ${GLEW_DEFINITIONS} )

  add_executable( GLUTMinimalCFR
    ${SOURCES}
    ${HEADERS}
  )

  target_link_libraries( GLUTMinimalCFR
    ${GLUT_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    ${GLEW_LIBS}
    ${Boost_LIBRARIES}
    DPSgAlgorithm
    DPSgCore
    DPSgGL
    DPSgRdrRiXGL
    DPSgIO
    DPSgUIManipulator
    DPSgGenerator
    DPSgUIGLUT
  )

  CopyGLEW( GLUTMinimalCFR "${DP_BINARY_PATH}" )
  CopyGLUT( GLUTMinimalCFR "${DP_BINARY_PATH}" )

  add_perforce_to_target( GLUTMinimalCFR )

  set_target_properties( GLUTMinimalCFR PROPERTIES FOLDER "Apps")
endif()
