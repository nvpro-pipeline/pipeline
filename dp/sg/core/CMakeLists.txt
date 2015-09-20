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


#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DDP_SG_CORE_EXPORTS
  "-D_CRT_SECURE_NO_WARNINGS"
)

#sources
set(CORE_SOURCES
  src/Billboard.cpp
  src/Buffer.cpp
  src/BufferHost.cpp
  src/Camera.cpp
  src/ClipPlane.cpp
  src/EffectData.cpp
  src/FrustumCamera.cpp
  src/GeoNode.cpp
  src/Group.cpp
  src/IndexSet.cpp
  src/Init.cpp
  src/LightSource.cpp
  src/LOD.cpp
  src/MatrixCamera.cpp
  src/Node.cpp
  src/Object.cpp
  src/ParallelCamera.cpp
  src/ParameterGroupData.cpp
  src/Path.cpp
  src/PerspectiveCamera.cpp
  src/Primitive.cpp
  src/Sampler.cpp
  src/Scene.cpp
  src/Switch.cpp
  src/Texture.cpp
  src/TextureFile.cpp
  src/TextureHost.cpp
  src/Transform.cpp
  src/VertexAttribute.cpp
  src/VertexAttributeSet.cpp
)

set(CORE_PUBLIC_HEADERS
  Billboard.h
  BoundingVolumeObject.h
  Buffer.h
  BufferHost.h
  Camera.h
  ClipPlane.h
  Config.h
  ConstIterator.h
  CoreTypes.h
  EffectData.h
  Event.h
  FrustumCamera.h
  GeoNode.h
  Group.h
  HandledObject.h
  IndexSet.h
  LightSource.h
  LOD.h
  MatrixCamera.h
  Node.h
  Object.h
  ParallelCamera.h
  ParameterGroupData.h
  Path.h
  PerspectiveCamera.h
  Primitive.h
  Sampler.h
  Scene.h
  Switch.h
  Texture.h
  TextureFile.h
  TextureHost.h
  Transform.h
  VertexAttribute.h
  VertexAttributeSet.h
)

set(CORE_PRIVATE_HEADERS
)

source_group(sources FILES ${CORE_SOURCES})
source_group(headers FILES ${CORE_PRIVATE_HEADERS})
source_group("" FILES ${CORE_PUBLIC_HEADERS})

#target
add_library( DPSgCore SHARED
  ${CORE_SOURCES}
  ${CORE_PUBLIC_HEADERS}
  ${CORE_PRIVATE_HEADERS}
  $<TARGET_OBJECTS:DPSgAnimation>
  $<TARGET_OBJECTS:DPSgAlgorithm>
  $<TARGET_OBJECTS:DPSgUI>
  $<TARGET_OBJECTS:DPSgXbar>
)

target_link_libraries( DPSgCore
  DPFx
  DPMath
  DPUtil
)

set_target_properties( DPSgCore PROPERTIES FOLDER "SG" )

if( BUILD_DPT )
#  add_subdirectory( test )
endif()
