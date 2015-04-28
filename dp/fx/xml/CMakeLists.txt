project(DPFxXML)

find_package(NVTinyXML)

if (TINYXML_FOUND)

  #Need to define this since this lib will be linked to DPFx
  add_definitions(
    "-DDP_FX_EXPORTS"
    "-D_CRT_SECURE_NO_WARNINGS"
  )

  include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/inc" )
  include_directories( ${TINYXML_INCLUDE_DIRS} )

  set(PUBLIC_HEADERS
    Config.h
    EffectLoader.h
  )

  #let cmake determine linker language
  set(SOURCES
    src/EffectLoader.cpp
  )

  source_group(sources FILES ${SOURCES})
  source_group("" FILES ${PUBLIC_HEADERS})

  add_library(DPFxXML STATIC
    ${PUBLIC_HEADERS}
    ${SOURCES}
  )

  target_link_libraries( DPFxXML DPUtil DPFxGLSL ${TINYXML_LIBRARIES} )

  set_target_properties( DPFxXML PROPERTIES FOLDER "Fx" )

  if(UNIX)
    set_target_properties( DPFxXML PROPERTIES COMPILE_FLAGS -fPIC )
  endif()

  target_include_symbol( DPFx dp_fx_xml_initialized )
else()
  message("TinyXML not found, disabling DPFxXML")
endif()