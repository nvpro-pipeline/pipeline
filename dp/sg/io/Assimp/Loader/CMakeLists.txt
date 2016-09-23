find_package(Boost COMPONENTS filesystem system REQUIRED )

if (TARGET assimp)
  #includes
  include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    "${DP_3RDPARTY_PATH}/assimp/include"
    ${Boost_INCLUDE_DIRS}
  )

  #definitions
  add_definitions( 
    -DASSIMPLOADER_EXPORTS
  )

#  if(WIN32)
#    add_definitions("/wd4996")
#  endif()


  #sources
  set(ASSIMPLOADER_SOURCES
    AssimpLoader.cpp
  )

  set(ASSIMPLOADER_HEADERS
    AssimpLoader.h
  )

  source_group(source FILES ${ASSIMPLOADER_SOURCES})
  source_group(header FILES ${ASSIMPLOADER_HEADERS})

  #target
  add_library( AssimpLoader SHARED
    ${ASSIMPLOADER_SOURCES}
    ${ASSIMPLOADER_HEADERS}
  )

  target_link_libraries( AssimpLoader
    DP
    DPSgCore
    DPMath
    DPUtil
    DPSgIO
    ${Boost_LIBRARIES}
    assimp
    #debug "${NVjsoncpp_BASE}/lib/win32-msvc2013-x64-debug/jsoncpp.lib"
    #optimized "${NVjsoncpp_BASE}/lib/win32-msvc2013-x64-release/jsoncpp.lib"
  )

  #if (WIN32)
  #  set_target_properties( AssimpLoader PROPERTIES LINK_FLAGS_DEBUG "/NODEFAULTLIB:msvcrt" )
  #endif(WIN32)

  set_target_properties( AssimpLoader PROPERTIES SUFFIX ".nxm" FOLDER "DP/SG/IO")
else()
  message("Assimp not found, disabling AssimpLoader")
endif()
