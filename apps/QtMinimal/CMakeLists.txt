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
cmake_policy(SET CMP0020 OLD)

# Qt+OpenGL not available on arm yet
if ( NOT "${DP_ARCH}" STREQUAL "armv7l")
  project( QtMinimal )

  find_package(OpenGL REQUIRED )
  find_package(NVQt5  QUIET )
  find_package(Qt5Core REQUIRED )
  find_package(NVGLEW REQUIRED )

  find_package( NVBoost )
  find_package( Boost COMPONENTS program_options REQUIRED )

  set( SCENIX_COMMON_BASE "${SCENIX_HOME}/samples/common" )

  set( QTMINIMAL_SOURCES
    src/main.cpp
  )

  set (QTMINIMAL_HEADERS
  )

  include_directories( "."
                       ${Boost_INCLUDE_DIRS}
                     )

  include_directories( "inc" )
  include_directories( ${GLEW_INCLUDES} )

  add_definitions( ${GLEW_DEFINITIONS} )

  add_executable( QtMinimal
    ${QTMINIMAL_SOURCES}
    ${QTMINIMAL_HEADERS}
  )

  target_link_libraries( QtMinimal
    ${QT_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    ${GLEW_LIBS}
    ${Boost_LIBRARIES}
    DPSgAlgorithm
    DPSgCore
    DPSgGL
    DPSgRdrRiXGL
    DPSgIO
    DPSgUIManipulator
    DPSgUIQt5
    DPSgGenerator
  )

  qt5_use_modules(QtMinimal Widgets OpenGL)

  CopyGLEW( QtMinimal "${DP_BINARY_PATH}" )

  if(TARGET SceniX)
    add_dependencies( QtMinimal SceniX SceniXxbar SceniXUtils )
    add_perforce_to_target( QtMinimal )
  endif()

  set_target_properties( QtMinimal PROPERTIES FOLDER "Apps")
endif()