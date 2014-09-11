
#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

#definitions
add_definitions(
  "-DDPTRIX_EXPORTS"
  "-D_CRT_SECURE_NO_WARNINGS"
)

#sources
set (DPTRIX_SOURCES
  src/RiXBackend.cpp
)

set (DPTRIX_HEADERS
  inc/Config.h
)

set (DPTRIX_PUBLIC_HEADERS
  RiXBackend.h
)

source_group(headers FILES ${DPTRIX_HEADERS})
source_group(sources FILES ${DPTRIX_SOURCES})
source_group("" FILES ${DPTRIX_PUBLIC_HEADERS})

#target
add_library( DPTRiX SHARED
   ${DPTRIX_SOURCES}
   ${DPTRIX_HEADERS}
   ${DPTRIX_PUBLIC_HEADERS}
)

set_target_properties( DPTRiX PROPERTIES FOLDER "DPT/RiX/Core" )

target_link_libraries( DPTRiX
  DPTcore
  DPUtil
  RiXCore
)

add_dependencies( DPTRiX DPUtil DPTcore )