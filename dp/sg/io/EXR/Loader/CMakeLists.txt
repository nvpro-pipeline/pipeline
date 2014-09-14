project(EXRLoader)

find_package( NVOpenEXR )

include(CopyFile)

include_directories(
  ${OPENEXR_INCLUDE_DIRS}
)

#definitions
add_definitions(
  -DEXRLOADER_EXPORTS
)

if(WIN32)
  add_definitions( "/wd4996" )
endif()

#sources
set(EXRLOADER_SOURCES
  EXRLoader.cpp
  stdafx.cpp
)

set(EXRLOADER_HEADERS
  EXRLoader.h
  stdafx.h
)

source_group(source FILES ${EXRLOADER_SOURCES})
source_group(header FILES ${EXRLOADER_HEADERS})

#target
add_library( EXRLoader SHARED
  ${EXRLOADER_SOURCES}
  ${EXRLOADER_HEADERS}
  Readme.txt
)

if ("${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
  set (CONFIG "$(ConfigurationName)")
else()
  if( "${CMAKE_BUILD_TYPE}" STREQUAL "" )
    set (CONFIG "release")
  else()
    string(TOLOWER ${CMAKE_BUILD_TYPE} CONFIG )
  endif()
endif()

string(TOLOWER ${CONFIG} CNF)

if ( WIN32 )
  set(IMF_DLLS Half.dll Iex.dll IlmImf.dll IlmThread.dll Imath.dll)
  foreach (filename ${IMF_DLLS})
    copy_file_if_changed(EXRLoader "${OPENEXR_BINDIR}/Release/${filename}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
  endforeach()
endif()

add_perforce_to_target( EXRLoader )

target_link_libraries( EXRLoader
  DPSgCore
  DPSgIO
  ${OPENEXR_LIBRARIES}
)

if(WIN32)
target_link_libraries( EXRLoader
  ${OPENEXR_LIBDIR}/Release/IlmImf.lib
)
endif()

if(TARGET SceniX)
  add_dependencies( EXRLoader SceniX )
  add_perforce_to_target( EXRLoader )
endif()

set_target_properties( EXRLoader PROPERTIES SUFFIX ".nxm" )
set_target_properties( EXRLoader PROPERTIES FOLDER "Loaders")
