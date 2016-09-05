
find_package(NVPMAPI REQUIRED)

#includes
include_directories(
  ${NVPMAPI_INCLUDE_DIR}
)

add_definitions(
  "-D_CRT_SECURE_NO_WARNINGS"
)

if ( DPT_RENDERERAPI )
  add_definitions(
    "-DDPT_RENDERERAPI"
  )
endif()

link_directories( 
)

#sources
set (DPTAPP_SOURCES
  src/App.cpp
  src/main.cpp
)

set (DPTAPP_HEADERS
  App.h
)

#target
add_executable( DPTapp
  ${DPTAPP_SOURCES}
  ${DPTAPP_HEADERS}
)

target_link_libraries( DPTapp
  DPUtil
  DPTcore
)

if (NVPMAPI_FOUND)
  CopyNvPmApi( DPTapp )
endif()

add_dependencies( DPTapp DPTcore DPFx )

source_group(headers FILES ${DPTAPP_HEADERS})
source_group(sources FILES ${DPTAPP_SOURCES})

set_target_properties( DPTapp PROPERTIES FOLDER "test/testfw" )
