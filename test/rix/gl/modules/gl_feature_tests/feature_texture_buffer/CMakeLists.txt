
#Extract test name from directory
#string(REGEX REPLACE "^.*/([^/]*)$" "\\1" TEST_NAME ${CMAKE_CURRENT_SOURCE_DIR})


#definitions
add_definitions("-DDPT_QUOTEDTESTNAME=${TEST_NAME}")

set (TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/feature_texture_buffer.cpp      #### Add additional files here
)

set (TEST_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/feature_texture_buffer.h        #### Add additional files here
)


#source
source_group(${TEST_NAME}/headers FILES ${TEST_HEADERS})
source_group(${TEST_NAME}/sources FILES ${TEST_SOURCES})

LIST(APPEND LINK_SOURCES ${TEST_HEADERS} )
LIST(APPEND LINK_SOURCES ${TEST_SOURCES} )

set (LINK_SOURCES ${LINK_SOURCES} PARENT_SCOPE)
