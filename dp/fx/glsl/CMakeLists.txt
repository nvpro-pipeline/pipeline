project(DPFxGLSL)

#Need to define this since this lib will be linked to DPFx
add_definitions(
  "-DDP_FX_EXPORTS"
  "-D_CRT_SECURE_NO_WARNINGS"
)

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/inc" )

set(PUBLIC_HEADERS
  Config.h
  ParameterInfo.h
  UniformGenerator.h
  UniformGeneratorGLSL.h
  UniformGeneratorGLSLStandard.h
  UniformGeneratorGLSLShaderBufferLoad.h
  UniformGeneratorUBOStd140.h
  UniformGeneratorSSBOStd140.h
)

set(HEADERS
  inc/ParameterInfoShaderBufferLoad.h
)

#let cmake determine linker language
set(SOURCES
  src/ParameterInfoShaderBufferLoad.cpp
  src/ParameterInfoUBOStd140.cpp
  src/UniformGenerator.cpp
  src/UniformGeneratorGLSL.cpp
  src/UniformGeneratorGLSLStandard.cpp
  src/UniformGeneratorGLSLShaderBufferLoad.cpp
  src/UniformGeneratorUBOStd140.cpp
  src/UniformGeneratorSSBOStd140.cpp
)

source_group(headers FILES ${HEADERS})
source_group(sources FILES ${SOURCES})
source_group("" FILES ${PUBLIC_HEADERS})

add_library(DPFxGLSL STATIC
  ${PUBLIC_HEADERS}
  ${HEADERS}
  ${SOURCES}
)

set_target_properties( DPFxGLSL PROPERTIES FOLDER "Fx" )

if(UNIX)
  set_target_properties( DPFxGLSL PROPERTIES COMPILE_FLAGS -fPIC )
endif()
