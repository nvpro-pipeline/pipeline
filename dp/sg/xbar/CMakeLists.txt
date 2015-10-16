
#includes
include_directories(
  ${SCENIX_INCLUDES}
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DDP_SG_XBAR_EXPORTS
)

#sources
set(XBAR_SOURCES
  src/DrawableManager.cpp
  src/GeoNodeObserver.cpp
  src/GeneratorState.cpp
  src/ObjectObserver.cpp
  src/SceneObserver.cpp
  src/SceneTree.cpp
  src/SceneTreeGenerator.cpp
  src/SwitchObserver.cpp
  src/TransformObserver.cpp
  src/TransformTree.cpp
)

set(XBAR_PUBLIC_HEADERS
  DrawableManager.h
  ObjectTree.h
  SceneTree.h
  TransformTree.h
  Tree.h
  TreeResourceGroup.h
  xbar.h
)

set(XBAR_PRIVATE_HEADERS
  inc/GeneratorState.h
  inc/GeoNodeObserver.h
  inc/ObjectObserver.h
  inc/Observer.h
  inc/SceneObserver.h
  inc/SceneTreeGenerator.h
  inc/SwitchObserver.h
  inc/TransformObserver.h
  inc/UpdateObjectVisitor.h
)

source_group(sources FILES ${XBAR_SOURCES})
source_group(headers FILES ${XBAR_PRIVATE_HEADERS})
source_group("" FILES ${XBAR_PUBLIC_HEADERS})

#target
add_library( DPSgXbar OBJECT
  ${XBAR_SOURCES}
  ${XBAR_PUBLIC_HEADERS}
  ${XBAR_PRIVATE_HEADERS}
)

set_target_properties( DPSgXbar PROPERTIES FOLDER "SG" )

add_subdirectory( culling )