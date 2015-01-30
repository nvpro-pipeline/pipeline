
include( CopyFile )
find_package( NVHOOPS3D )

if(HOOPS_FOUND)
  #includes
  include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${HOOPS3D_INCLUDES}
  )

  #definitions
  add_definitions( "-DHOOPSLOADER_EXPORTS" )

  #source
  set( HOOPSLOADER_SOURCES
        HOOPSLoader.cpp
     )

  set( HOOPSLOADER_HEADERS
        HOOPSLoader.h
     )

  source_group(source FILES ${HOOPSLOADER_SOURCES})
  source_group(header FILES ${HOOPSLOADER_HEADERS})

  #target
  add_library( HOOPSLoader SHARED
                ${HOOPSLOADER_HEADERS}
                ${HOOPSLOADER_SOURCES}
             )

  target_link_libraries( HOOPSLoader
    DP
    DPSgCore
    DPMath
    DPUtil
    DPFx
    DPSgIO
  )
           
  set_target_properties( HOOPSLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders")

  #
  # Copy dlls if they have changed.  Note, we don't need all of them, but we 
  # are not certain which ones we need for the time being.
  #
  file( GLOB hoops3ddlls "${HOOPS3D_BINDIR}/*.dll" )
  if (UNIX)
    file( TO_CMAKE_PATH "${SCENIX_BIN_PATH}" FDESTINATION )
  else()

  if ("${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
    set (SCENIX_CONFIG "$(ConfigurationName)")
  else()
    if( "${CMAKE_BUILD_TYPE}" STREQUAL "" )
      set (SCENIX_CONFIG "release")
    else()
      string(TOLOWER ${CMAKE_BUILD_TYPE} SCENIX_CONFIG )
    endif()
  endif()

    file( TO_CMAKE_PATH "${SCENIX_BIN_PATH}/${SCENIX_CONFIG}" FDESTINATION )
  endif()

  foreach( file ${hoops3ddlls} )
    file( TO_CMAKE_PATH "${file}" FSOURCE )
    copy_file_if_changed( HOOPSLoader "${FSOURCE}" "${DP_BINARY_PATH}" )
  endforeach()
  
endif()