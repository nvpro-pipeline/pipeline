
#includes

#definitions
add_definitions(
  -DDP_SG_ANIMATION_EXPORTS
)

#sources
set(SOURCES
  src/LinkManager.cpp
)

set(PRIVATE_HEADERS
)

set(PUBLIC_HEADERS
  Config.h
  LinkManager.h
)

source_group(source FILES ${SOURCES})
source_group(header FILES ${PRIVATE_HEADERS})
source_group(""     FILES ${PUBLIC_HEADERS})

#target
add_library( DPSgAnimation OBJECT
  ${SOURCES}
  ${HEADERS}
  ${PUBLIC_HEADERS}
)

set_target_properties( DPSgAnimation PROPERTIES FOLDER "DP/SG" )
