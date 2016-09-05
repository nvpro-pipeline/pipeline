project( RiXFx )

#includes
include_directories(
  "${PROJECT_SOURCE_DIR}/inc"
  ${GLEW_INCLUDE_DIRS}
)
#definitions
add_definitions(
  "-DRIX_FX_EXPORTS"
  "-D_CRT_SECURE_NO_WARNINGS"
)

#sources
set( SOURCES 
  src/BufferManager.cpp
  src/BufferManagerImpl.cpp
  src/BufferManagerIndexed.cpp
  src/BufferManagerOffset.cpp
  src/GroupDataBuffered.cpp
  src/GroupDataBufferedCombined.cpp
  src/GroupDataUniform.cpp
  src/ParameterGroupSpecInfo.cpp
  src/Manager.cpp
  src/ManagerUniform.cpp
)

set( HEADERS
  inc/BufferManager.h
  inc/BufferManagerImpl.h
  inc/BufferManagerIndexed.h
  inc/BufferManagerOffset.h
  inc/GroupDataBase.h
  inc/GroupDataBuffered.h
  inc/GroupDataBufferedCombined.h
  inc/GroupDataUniform.h
  inc/ParameterGroupSpecInfo.h
  inc/ManagerUniform.h
)

set( PUBLIC_HEADERS
  Manager.h
  Config.h
)

source_group(headers FILES ${HEADERS})
source_group(sources FILES ${SOURCES})
source_group("" FILES ${PUBLIC_HEADERS})

add_library( RiXFx SHARED
  ${SOURCES}
  ${HEADERS}
  ${PUBLIC_HEADERS}
)

target_link_libraries( RiXFx
 DPFx
 DPUtil
 RiXCore
)

set_target_properties( RiXFx PROPERTIES FOLDER "DP/RiX"  )

if (WIN32 AND CMAKE_COMPILER_IS_GNUCC)
  # GCC requires the declspec(export) keyword behind the class keyword to export the vtable ( class declspec(export) MyKlass {} ). 
  # Export everything for now and adopt sources later.
  set_target_properties( RiXFx PROPERTIES LINK_FLAGS "-Wl,--export-all" )
endif()

