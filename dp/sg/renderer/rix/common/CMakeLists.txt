
find_package( NVGLEW REQUIRED )

#includes
include_directories( ${GLEW_INCLUDES} )

#definitions
add_definitions(
  -DDP_SG_RDR_RIX_GL_EXPORTS
)
add_definitions( ${GLEW_DEFINITIONS} )

#sources
set(SOURCES
  src/DrawableManagerDefault.cpp
  src/FSQRenderer.cpp
  src/FSQRendererImpl.cpp
  src/ResourceManager.cpp
  src/ResourceBuffer.cpp
  src/ResourceEffectDataRiXFx.cpp
  src/ResourceEffectSpecRiXFx.cpp
  src/ResourceIndexSet.cpp
  src/ResourceParameterGroupDataRiXFx.cpp
  src/ResourcePrimitive.cpp
  src/ResourceSampler.cpp
  src/ResourceSamplerState.cpp
  src/ResourceTexture.cpp
  src/ResourceVertexAttributeSet.cpp  
  src/SceneRenderer.cpp
  src/SceneRendererImpl.cpp
  src/ShaderManager.cpp
  src/ShaderManagerRiXFx.cpp
  src/ShaderManagerShaderBuffer.cpp
  src/ShaderManagerUniform.cpp
  src/TransparencyManagerOIT.cpp
  src/TransparencyManagerSB.cpp
)

set(PRIVATE_HEADERS
  inc/Culling.h
  inc/FSQRendererImpl.h
  inc/DrawableManagerDefault.h
  inc/ResourceManager.h
  inc/ResourceBuffer.h
  inc/ResourceEffectDataRiXFx.h
  inc/ResourceEffectSpecRiXFx.h
  inc/ResourceIndexSet.h
  inc/ResourceParameterGroupDataRiXFx.h
  inc/ResourcePrimitive.h
  inc/ResourceSampler.h
  inc/ResourceSamplerState.h
  inc/ResourceTexture.h  
  inc/ResourceVertexAttributeSet.h
  inc/SceneRendererImpl.h
  inc/ShaderManager.h
  inc/ShaderManagerRiXFx.h
  inc/ShaderManagerShaderBuffer.h
  inc/ShaderManagerUniform.h
  inc/Utility.h
)

set(PUBLIC_HEADERS
  Config.h
  FSQRenderer.h
  SceneRenderer.h
  TransparencyManager.h
  TransparencyManagerOIT.h
  TransparencyManagerSB.h
 )

source_group(source FILES ${SOURCES})
source_group(header FILES ${PRIVATE_HEADERS})
source_group(""     FILES ${PUBLIC_HEADERS})

#target
add_library( DPSgRdrRiXGL SHARED
  ${SOURCES}
  ${PRIVATE_HEADERS}
  ${PUBLIC_HEADERS}
)

add_perforce_to_target( DPSgRdrRiXGL )

target_link_libraries( DPSgRdrRiXGL
  RiXCore
  RiXGL
  RiXFx
  DP
  DPSgXbar
  DPSgCore
  DPGL
  DPSgGL
  DPSgUI
  DPSgIO
  DPMath
  DPCulling
  ${GLEW_LIBS}
)

set_target_properties( DPSgRdrRiXGL PROPERTIES FOLDER "SG" )
