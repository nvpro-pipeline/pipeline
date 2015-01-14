find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )

#includes
include_directories(
  ${GLEW_INCLUDE_DIRS}
)

#definitions
add_definitions(
  -DDP_SG_UI_MANIPULATOR_EXPORTS
  ${GLEW_DEFINITIONS}
)

#sources
set(SOURCES
  src/CylindricalCameraManipulator.cpp
  src/CylindricalCameraManipulatorHIDSync.cpp
  src/FlightCameraManipulator.cpp
  src/FlightCameraManipulatorHIDSync.cpp
  src/TrackballCameraManipulator.cpp
  src/TrackballCameraManipulatorHIDSync.cpp
  src/TrackballTransformManipulator.cpp
  src/TrackballTransformManipulatorHIDSync.cpp
  src/WalkCameraManipulator.cpp
  src/WalkCameraManipulatorHIDSync.cpp
)

set(PRIVATE_HEADERS
)

set(PUBLIC_HEADERS
  Config.h
  CylindricalCameraManipulator.h
  CylindricalCameraManipulatorHIDSync.h
  FlightCameraManipulator.h
  FlightCameraManipulatorHIDSync.h
  TrackballCameraManipulator.h
  TrackballCameraManipulatorHIDSync.h
  TrackballTransformManipulator.h
  TrackballTransformManipulatorHIDSync.h
  WalkCameraManipulator.h
  WalkCameraManipulatorHIDSync.h
)

source_group(source FILES ${SOURCES})
source_group(header FILES ${PRIVATE_HEADERS})
source_group(""     FILES ${PUBLIC_HEADERS})

#target
add_library( DPSgUIManipulator SHARED
  ${SOURCES}
  ${HEADERS}
  ${PUBLIC_HEADERS}
)

target_link_libraries( DPSgUIManipulator
  DPSgCore
  DPFx
  DPMath
  ${OPENGL_glu_LIBRARY}
  ${OPENGL_gl_LIBRARY}
  ${GLEW_LIBRARY}
)

set_target_properties( DPSgUIManipulator PROPERTIES FOLDER "SG" )
