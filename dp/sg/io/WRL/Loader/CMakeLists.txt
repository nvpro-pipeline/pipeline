#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  -DWRLLOADER_EXPORTS
  -D_CRT_SECURE_NO_WARNINGS
)

#sources
set(WRLLOADER_SOURCES
  VRMLTypes.cpp
  WRLLoader.cpp
)

set(WRLLOADER_HEADERS
  VRMLTypes.h
  inc/WRLLoader.h
)

source_group(source FILES ${WRLLOADER_SOURCES})
source_group(header FILES ${WRLLOADER_HEADERS})

#target
add_library( WRLLoader SHARED
  ${WRLLOADER_SOURCES}
  ${WRLLOADER_HEADERS}
)

target_link_libraries( WRLLoader
  DP
  DPSgCore
  DPSgUI
  DPSgAlgorithm
  DPMath
  DPUtil
  DPFx
  DPSgIO
)

set_target_properties( WRLLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders" )
