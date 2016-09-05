project( RiXCore )

find_package( Boost REQUIRED )

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

set_target_properties( RiXCore PROPERTIES FOLDER "DP/RiX"  )

if (WIN32 AND CMAKE_COMPILER_IS_GNUCC)
  # GCC requires the declspec(export) keyword behind the class keyword to export the vtable ( class declspec(export) MyKlass {} ). 
  # Export everything for now and adopt sources later.
  set_target_properties( RiXCore PROPERTIES LINK_FLAGS "-Wl,--export-all" )
endif()
