
find_package( GLEW REQUIRED )
find_package( NVToolsExt REQUIRED )

#includes
include_directories(
  ${GLEW_INCLUDE_DIRS}
  ${NV_TOOLS_EXT_INCLUDE_DIR}
)

#definitions
add_definitions(
  -DDP_SG_RDR_RIX_GL_EXPORTS
)
add_definitions( ${GLEW_DEFINITIONS} )

#sources
set(SOURCES
  src/BufferAllocator.cpp
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
  src/ResourceTexture.cpp
  src/ResourceVertexAttributeSet.cpp  
  src/SceneRenderer.cpp
  src/SceneRendererImpl.cpp
  src/ShaderManager.cpp
  src/ShaderManagerRiXFx.cpp
  src/TransparencyManager.cpp
  src/TransparencyManagerNone.cpp
  src/TransparencyManagerOITAll.cpp
  src/TransparencyManagerOITClosestArray.cpp
  src/TransparencyManagerOITClosestList.cpp
  src/TransparencyManagerSB.cpp
)

set(PRIVATE_HEADERS
  inc/BufferAllocator.h
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
  inc/ResourceTexture.h  
  inc/ResourceVertexAttributeSet.h
  inc/SceneRendererImpl.h
  inc/ShaderManager.h
  inc/ShaderManagerRiXFx.h
  inc/Utility.h
)

set(PUBLIC_HEADERS
  Config.h
  FSQRenderer.h
  SceneRenderer.h
  TransparencyManager.h
  TransparencyManagerNone.h
  TransparencyManagerOITAll.h
  TransparencyManagerOITClosestArray.h
  TransparencyManagerOITClosestList.h
  TransparencyManagerSB.h
)

set(RESOURCES
  ../../../../../media/dpfx/emitColor.glsl
  ../../../../../media/dpfx/emitColorDepth.glsl
  ../../../../../media/dpfx/emitColorOITAllCounter.glsl
  ../../../../../media/dpfx/emitColorOITAllSample.glsl
  ../../../../../media/dpfx/emitColorOITClosestArray.glsl
  ../../../../../media/dpfx/emitColorOITClosestList.glsl
  ../../../../../media/dpfx/oitAllClear_fs.glsl
  ../../../../../media/dpfx/oitAllResolveCounters_fs.glsl
  ../../../../../media/dpfx/oitAllResolveSamples_fs.glsl
  ../../../../../media/dpfx/oitClosestArrayClear_fs.glsl
  ../../../../../media/dpfx/oitClosestArrayResolve_fs.glsl
  ../../../../../media/dpfx/oitClosestListClear_fs.glsl
  ../../../../../media/dpfx/oitClosestListResolve_fs.glsl
  ../../../../../media/dpfx/passThroughPosition_vs.glsl
)

source_group(source    FILES ${SOURCES})
source_group(header    FILES ${PRIVATE_HEADERS})
source_group(""        FILES ${PUBLIC_HEADERS})
source_group(resources FILES ${RESOURCES})

#target
add_library( DPSgRdrRiXGL SHARED
  ${SOURCES}
  ${PRIVATE_HEADERS}
  ${PUBLIC_HEADERS}
  ${RESOURCES}
)

target_link_libraries( DPSgRdrRiXGL
  RiXCore
  RiXGL
  RiXFx
  DP
  DPSgXbarCulling
  DPSgCore
  DPGL
  DPSgGL
  DPSgIO
  DPMath
  DPCulling
  ${GLEW_LIBRARY}
  ${NV_TOOLS_EXT_LIB}
)

set_target_properties( DPSgRdrRiXGL PROPERTIES FOLDER "SG" )

CopyNV_TOOLS_EXT( DPSgRdrRiXGL )

