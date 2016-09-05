add_definitions(
  "-D_CRT_SECURE_NO_WARNINGS"
)

set( MANAGER_SOURCES
  Manager.cpp
  Manager.h
)
  
source_group(headers FILES Manager.h)
source_group(sources FILES Manager.cpp)
  
add_library( DPTestManager STATIC
   ${MANAGER_SOURCES}
)

add_dependencies( DPTestManager DPTcore )
  
set_target_properties( DPTestManager PROPERTIES FOLDER "test/testfw" LINK_INTERFACE_LIBRARIES "")