# Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

find_package( Boost COMPONENTS system filesystem REQUIRED )

add_definitions(
  "-D_CRT_SECURE_NO_WARNINGS"
  "-D_SCL_SECURE_NO_WARNINGS"
)

#includes
include_directories(
  ${Boost_INCLUDE_DIRS}
)

set(PUBLIC_HEADERS
  Assert.h
  Config.h
  DP.h
  Types.h
  Exception.h
)

#let cmake determine linker language
set(SOURCES
  src/Assert.cpp
  src/DP.cpp
  src/Exception.cpp
)

source_group(sources FILES ${SOURCES})
source_group("" FILES ${PUBLIC_HEADERS})

add_subdirectory( util )

add_library(DP SHARED
  ${PUBLIC_HEADERS}
  ${SOURCES}
)

target_link_libraries(DP ${Boost_LIBRARIES} )

set_target_properties( DP PROPERTIES COMPILE_DEFINITIONS DP_EXPORTS )

set_target_properties( DP PROPERTIES FOLDER "DP" )

if (USE_CUDA)
  add_subdirectory( cuda )
endif()
add_subdirectory( culling )
add_subdirectory( fx )
add_subdirectory( gl )
add_subdirectory( math )
add_subdirectory( rix )
add_subdirectory( transform )
add_subdirectory( sg )
add_subdirectory( ui )
