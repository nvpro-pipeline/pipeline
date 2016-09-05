project(DPFxMDL)

find_package( MDLSDK )
find_package(NVTinyXML)

if ( MDL_SDK_FOUND AND TINYXML_FOUND )

  #Need to define this since this lib will be linked to DPFx
  add_definitions(
    "-DDP_FX_EXPORTS"
    "-D_CRT_SECURE_NO_WARNINGS"
  )

  include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    ${MDL_SDK_INCLUDE_DIR}
    ${TINYXML_INCLUDE_DIRS}
  )

  set(PUBLIC_HEADERS
    Config.h
    EffectLoader.h
  )
  
  set(HEADERS
    inc/MaterialBuilder.h
    inc/MDLTokenizer.h
  )

  FILE( GLOB RESOURCES ${DP_HOME}/media/dpfx/mdl/*.glsl )
  set ( RESOURCES
    ${RESOURCES}
    ${DP_HOME}/media/dpfx/mdl/MDL.cfg
  )

  #let cmake determine linker language
  set(SOURCES
    src/EffectLoader.cpp
    src/MaterialBuilder.cpp
    src/MDLTokenizer.cpp
  )

  source_group("" FILES ${PUBLIC_HEADERS})
  source_group(headers FILES ${HEADERS})
  source_group(resources FILES ${RESOURCES})
  source_group(sources FILES ${SOURCES})

  add_library(DPFxMDL STATIC
    ${PUBLIC_HEADERS}
    ${HEADERS}
    ${RESOURCES}
    ${SOURCES}
  )

  target_link_libraries( DPFxMDL DPUtil ${TINYXML_LIBRARIES} )

  CopyMDL_SDK( DPFxMDL )
  
  set_target_properties( DPFxMDL PROPERTIES FOLDER "DP/Fx" )

  if(UNIX)
    set_target_properties( DPFxMDL PROPERTIES COMPILE_FLAGS -fPIC )
  endif()

  target_include_symbol( DPFx dp_fx_mdl_initialized )
else()
  message("MDLSDK not found or TinyXML not found, disabling DPFxMDL")
endif()