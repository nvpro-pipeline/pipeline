find_package( NVBoost REQUIRED )
find_package( Boost COMPONENTS system filesystem REQUIRED )

add_definitions(
  "-D_CRT_SECURE_NO_WARNINGS"
  "-D_SCL_SECURE_NO_WARNINGS"
)

#includes
include_directories(
  ${Boost_INCLUDE_DIRS}
)

set(PUBLIC_HEADERS
  Config.h
  DP.h
  Types.h
  Exception.h
)

#let cmake determine linker language
set(SOURCES
  src/DP.cpp
  src/Exception.cpp
)

source_group(sources FILES ${SOURCES})
source_group("" FILES ${PUBLIC_HEADERS})

add_subdirectory( util )

add_library(DP SHARED
  ${PUBLIC_HEADERS}
  ${SOURCES}
)

target_link_libraries(DP ${Boost_LIBRARIES} )

set_target_properties( DP PROPERTIES COMPILE_DEFINITIONS DP_EXPORTS )

add_subdirectory( fx )
add_subdirectory( math )
add_subdirectory( rix )

add_subdirectory( culling )
add_subdirectory( sg )
add_subdirectory( ui )
add_subdirectory( gl )
