# Copyright NVIDIA Corporation 2009-2015
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
if (${CMAKE_VERSION} VERSION_EQUAL 3.0.0 OR ${CMAKE_VERSION} VERSION_GREATER 3.0.0)
  cmake_policy(SET CMP0043 OLD)
endif()

find_package( Qt5Widgets QUIET )

if (Qt5Widgets_FOUND AND TARGET DPSgUIQt5)
  project( QtMinimal )

  find_package( OpenGL REQUIRED )
  find_package( GLEW REQUIRED )
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
  include_directories( ${GLEW_INCLUDE_DIRS} )

  add_definitions( ${GLEW_DEFINITIONS} )

  add_executable( QtMinimal
    ${QTMINIMAL_SOURCES}
    ${QTMINIMAL_HEADERS}
  )

  target_link_libraries( QtMinimal
    ${QT_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    ${GLEW_LIBRARY}
    ${Boost_LIBRARIES}
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

  set_target_properties( QtMinimal PROPERTIES FOLDER "Apps")
else()
  message("Qt5Widgets or DpSgUIQt5 not found, skipping QtMinimal")
endif()

