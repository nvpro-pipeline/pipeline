project(DPFx)

cmake_policy(SET CMP0022 OLD)

find_package( NVTinyXML)

add_definitions(
  "-DDP_FX_EXPORTS"
  "-D_CRT_SECURE_NO_WARNINGS"
)

include_directories(
  ${TINYXML_INCLUDES}
  "${PROJECT_SOURCE_DIR}/inc"
)

set(PUBLIC_HEADERS
  Config.h
  EffectData.h
  EffectDefs.h
  EffectLibrary.h
  EffectSpec.h
  EnumSpec.h
  ParameterConversion.h
  ParameterGroupData.h
  ParameterGroupLayout.h
  ParameterGroupSpec.h
  ParameterSpec.h
)

set(HEADERS
  inc/EffectDataPrivate.h
  inc/EffectLoader.h
  inc/EffectLibraryImpl.h
  inc/EnumSpecSnippet.h
  inc/ExtensionSnippet.h
  inc/FileSnippet.h
  inc/ParameterGroupDataPrivate.h
  inc/ParameterGroupSnippet.h
  inc/Snippet.h
  inc/SnippetListSnippet.h
  inc/StringSnippet.h
  inc/UniformGenerator.h
  inc/VersionSnippet.h
)

#let cmake determine linker language
set(SOURCES
  src/EffectLibraryImpl.cpp
  src/EffectLibrary.cpp
  src/EffectData.cpp
  src/EffectDataPrivate.cpp
  src/EffectSpec.cpp
  src/EnumSpec.cpp
  src/EnumSpecSnippet.cpp
  src/ExtensionSnippet.cpp
  src/FileSnippet.cpp
  src/ParameterConversion.cpp
  src/ParameterGroupData.cpp
  src/ParameterGroupDataPrivate.cpp
  src/ParameterGroupLayout.cpp
  src/ParameterGroupSnippet.cpp
  src/ParameterGroupSpec.cpp
  src/ParameterSpec.cpp
  src/Snippet.cpp
  src/SnippetListSnippet.cpp
  src/StringSnippet.cpp
  src/VersionSnippet.cpp
)

source_group(sources FILES ${SOURCES})
source_group("" FILES ${PUBLIC_HEADERS})

add_library(DPFx SHARED
  ${PUBLIC_HEADERS}
  ${HEADERS}
  ${SOURCES}
)

target_link_libraries( DPFx DPFxGLSL DP DPUtil ${TINYXML_LIBS} )

set_property(TARGET DPFx PROPERTY LINK_INTERFACE_LIBRARIES "")

set_target_properties( DPFx PROPERTIES FOLDER "Fx" )
if (WIN32)
  if (LARGE_ADDRESS_AWARE)
    set_target_properties( DPFx PROPERTIES LINK_FLAGS /LARGEADDRESSAWARE )
  endif(LARGE_ADDRESS_AWARE)
  set_target_properties( DPFx PROPERTIES LINK_FLAGS_RELEASE "/NODEFAULTLIB:libcmt" )
  set_target_properties( DPFx PROPERTIES LINK_FLAGS_DEBUG "/NODEFAULTLIB:libcmtd" )
endif(WIN32)

# add subdirectories in the end so that 'if available' linking works.
add_subdirectory( glsl )
add_subdirectory( mdl )
add_subdirectory( xml )

if (TARGET DPFxXML)
  target_link_libraries(DPFx DPFxMDL DPFxXML)
endif()
