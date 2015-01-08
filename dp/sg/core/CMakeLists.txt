
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
  src/FaceConnections.cpp
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
  FaceConnections.h
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
