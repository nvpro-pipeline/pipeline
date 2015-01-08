find_package( NVlib3ds )

if (LIB3DS_FOUND)
  #includes
  include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/lib3ds"
    "${LIB3DS_INCLUDE_DIRS}"
  )

  #definitions
  add_definitions(
    -DTHREEDSLOADER_EXPORTS
  )

  #sources
  set(3DSLOADER_SOURCES
    3DSLoader.cpp
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

  target_link_libraries( 3DSLoader
    DP
    DPSgCore
    DPSgUI
    DPMath
    DPUtil
    DPFx
    DPSgIO
    ${LIB3DS_LIBRARIES}
  )

  set_target_properties( 3DSLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders")
else()
  message("lib3ds not found. Skipping 3DSLoader")
endif()
