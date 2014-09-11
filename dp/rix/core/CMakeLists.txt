project( RiXCore )

file(TO_CMAKE_PATH $ENV{DP_3RDPARTY_PATH} DP_3RDPARTY_PATH)
set(CMAKE_MODULE_PATH "${DP_3RDPARTY_PATH}/CMake")

find_package( NVBoost REQUIRED )
find_package( Boost COMPONENTS REQUIRED )

#includes
include_directories(
  "${PROJECT_SOURCE_DIR}/inc"
  ${Boost_INCLUDE_DIRS}
)

#definitions
add_definitions(
  -DRIX_CORE_EXPORTS
)

#sources
set( SOURCES 
  src/Renderer.cpp
  src/HandledObject.cpp
)

set( PUBLIC_HEADERS
  HandledObject.h
  RendererConfig.h
  RiX.h
)

source_group(sources FILES ${SOURCES})
source_group("" FILES ${PUBLIC_HEADERS})

add_library( RiXCore SHARED
  ${SOURCES}
  ${PUBLIC_HEADERS}
)

target_link_libraries( RiXCore DPUtil )

add_perforce_to_target( RiXCore )

set_target_properties( RiXCore PROPERTIES FOLDER "RiX"  )

if (WIN32 AND CMAKE_COMPILER_IS_GNUCC)
  # GCC requires the declspec(export) keyword behind the class keyword to export the vtable ( class declspec(export) MyKlass {} ). 
  # Export everything for now and adopt sources later.
  set_target_properties( RiXCore PROPERTIES LINK_FLAGS "-Wl,--export-all" )
endif()
