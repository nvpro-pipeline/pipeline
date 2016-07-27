#find_package(NVjsoncpp)

find_package(Boost COMPONENTS filesystem system REQUIRED )

#set(NVjsoncpp_found TRUE)

#set(NVjsoncpp_BASE "${DP_3RDPARTY_PATH}/jsoncpp")

if (TARGET jsoncpp_lib_static)
  #includes
  include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    ${Boost_INCLUDE_DIRS}
    "${DP_3RDPARTY_PATH}/jsoncpp/include"
  )

  #definitions
  add_definitions( 
    -DGLTFLOADER_EXPORTS
  )

  if(WIN32)
    add_definitions("/wd4996")
  endif()


  #sources
  set(XMLLOADER_SOURCES
    glTFLoader.cpp
  )

  set(XMLLOADER_HEADERS
    glTFLoader.h
  )

  source_group(source FILES ${XMLLOADER_SOURCES})
  source_group(header FILES ${XMLLOADER_HEADERS})

  #target
  add_library( glTFLoader SHARED
    ${XMLLOADER_SOURCES}
    ${XMLLOADER_HEADERS}
  )

  target_link_libraries( glTFLoader
    DP
    DPSgCore
    DPMath
    DPUtil
    DPSgIO
    ${Boost_LIBRARIES}
    jsoncpp_lib_static
    #debug "${NVjsoncpp_BASE}/lib/win32-msvc2013-x64-debug/jsoncpp.lib"
    #optimized "${NVjsoncpp_BASE}/lib/win32-msvc2013-x64-release/jsoncpp.lib"
  )

  if (WIN32)
    set_target_properties( glTFLoader PROPERTIES LINK_FLAGS_DEBUG "/NODEFAULTLIB:msvcrt" )
  endif(WIN32)

  set_target_properties( glTFLoader PROPERTIES SUFFIX ".nxm" FOLDER "Loaders")
else()
  message("jsoncpp not found, disabling glTFLoader")
endif()
