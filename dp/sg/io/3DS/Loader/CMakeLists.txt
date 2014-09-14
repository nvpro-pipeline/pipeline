# lib3ds 2.0 required, packages support only version 1.3
if(WIN32)

find_package( NVlib3ds )

#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib3ds"
  "${LIB3DS_INCLUDE_DIR}"
)

#definitions
add_definitions(
  -DTHREEDSLOADER_EXPORTS
)

#sources
set(3DSLOADER_SOURCES
  3DSLoader.cpp
  nvsgInterface.cpp
)

set(3DSLOADER_HEADERS
  3DSLoader.h
)

source_group(source FILES ${3DSLOADER_SOURCES})
source_group(header FILES ${3DSLOADER_HEADERS})

#target
add_library( 3DSLoader SHARED
  ${3DSLOADER_SOURCES}
  ${3DSLOADER_HEADERS}
)

add_perforce_to_target( 3DSLoader )

target_link_libraries( 3DSLoader
  DP
  DPSgCore
  DPSgUI
  DPMath
  DPUtil
  DPFx
  DPSgIO
  ${LIB3DS_LIBDIR}/Release/lib3ds.lib
  ${LIB3DS_LIBRARY}
)

set_target_properties( 3DSLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders")

endif()