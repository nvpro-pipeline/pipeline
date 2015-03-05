project( RiXGL )

find_package(OpenGL REQUIRED )
find_package(GLEW REQUIRED )

#includes
include_directories(
  "${PROJECT_SOURCE_DIR}/../Renderer/inc"
  "${PROJECT_SOURCE_DIR}/inc"
  ${GLEW_INCLUDE_DIRS}
)

#definitions
add_definitions(
  -DRIX_GL_EXPORTS
  ${GLEW_DEFINITIONS}
)

#sources

set( SOURCES 
  src/BufferGL.cpp
  src/ContainerGL.cpp
  src/GeometryDescriptionGL.cpp
  src/GeometryGL.cpp
  src/GeometryInstanceGL.cpp
  src/IndicesGL.cpp
  src/ParameterCacheStream.cpp
  src/ParameterCacheEntryStream.cpp
  src/ParameterCacheEntryStreamBuffer.cpp
  src/ParameterRenderer.cpp
  src/ParameterRendererBuffer.cpp
  src/ParameterRendererBufferAddressRange.cpp
  src/ParameterRendererBufferDSA.cpp
  src/ParameterRendererBufferRange.cpp
  src/ParameterRendererPersistentBufferMapping.cpp
  src/ParameterRendererPersistentBufferMappingUnifiedMemory.cpp
  src/ParameterRendererUniform.cpp
  src/ProgramGL.cpp
  src/ProgramParameterBuffer.cpp
  src/ProgramParameterGL.cpp
  src/ProgramPipelineGL.cpp
  src/RenderEngineGL.cpp
  src/RenderEngineGLBindless.cpp
  src/RenderEngineGLBindlessVAO.cpp
  src/RenderEngineGLIndirect.cpp
  src/RenderEngineGLVBO.cpp
  src/RenderEngineGLVBOVAO.cpp
  src/RenderEngineGLBindlessVertexAttribBinding.cpp
  src/RenderEngineGLVertexAttribBinding.cpp
  src/RiXGL.cpp
  src/RenderGroupGL.cpp
  src/Sampler.cpp
  src/SamplerStateGL.cpp
  src/TextureGL.cpp
  src/TextureGLBuffer.cpp
  src/TextureGL1D.cpp
  src/TextureGL1DArray.cpp
  src/TextureGL2D.cpp
  src/TextureGL2DRectangle.cpp
  src/TextureGL2DArray.cpp
  src/TextureGL3D.cpp
  src/TextureGLCubeMap.cpp
  src/TextureGLCubeMapArray.cpp
  src/VertexAttributesGL.cpp
  src/VertexDataGL.cpp
  src/VertexFormatGL.cpp
  src/VertexStateGL.cpp
)

#headers
set(HEADERS
  inc/BufferGL.h
  inc/ContainerGL.h
  inc/DataTypeConversionGL.h
  inc/GeometryDescriptionGL.h
  inc/GeometryGL.h
  inc/GeometryInstanceGL.h
  inc/IndicesGL.h
  inc/ProgramGL.h
  inc/ParameterCache.h
  inc/ParameterCacheStream.h
  inc/ParameterCacheEntryStream.h
  inc/ParameterCacheEntryStreamBuffer.h
  inc/ParameterRenderer.h
  inc/ParameterRendererBuffer.h
  inc/ParameterRendererBufferAddressRange.h
  inc/ParameterRendererBufferDSA.h
  inc/ParameterRendererBufferRange.h
  inc/ParameterRendererPersistentBufferMapping.h
  inc/ParameterRendererPersistentBufferMappingUnifiedMemory.h
  inc/ParameterRendererStream.h
  inc/ParameterRendererStreamBuffer.h
  inc/ParameterRendererUniform.h
  inc/ProgramParameterBuffer.h
  inc/ProgramParameterCache.h
  inc/ProgramParameterGL.h
  inc/ProgramPipelineGL.h
  inc/ProgramPipelineGroupCache.h
  inc/RenderEngineGL.h
  inc/RenderEngineGLDrawCall.h
  inc/RenderEngineGLImpl.hpp
  inc/RenderEngineGLBindless.h
  inc/RenderEngineGLBindlessVAO.h
  inc/RenderEngineGLIndirect.h
  inc/RenderEngineGLVBO.h
  inc/RenderEngineGLVBOVAO.h
  inc/RenderEngineGLBindlessVertexAttribBinding.h
  inc/RenderEngineGLVertexAttribBinding.h
  inc/RendererGLConfig.h
  inc/RenderGroupGL.h
  inc/Sampler.h
  inc/SamplerStateGL.h
  inc/TextureGL.h
  inc/TextureGLBuffer.h
  inc/TextureGL1D.h
  inc/TextureGL1DArray.h
  inc/TextureGL2D.h
  inc/TextureGL2DRectangle.h
  inc/TextureGL2DArray.h
  inc/TextureGL3D.h
  inc/TextureGLCubeMap.h
  inc/TextureGLCubeMapArray.h
  inc/VertexAttributesGL.h
  inc/VertexCacheGL.h
  inc/VertexDataGL.h
  inc/VertexFormatGL.h
  inc/VertexStateGL.h
)

set(PUBLIC_HEADERS
  RiXGL.h
  Config.h
)

source_group(headers FILES ${HEADERS})
source_group(sources FILES ${SOURCES})
source_group("" FILES ${PUBLIC_HEADERS})

#target
add_library( RiXGL SHARED
  ${SOURCES}
  ${HEADERS}
  ${PUBLIC_HEADERS}
)

target_link_libraries( RiXGL
  DP
  DPGL
  DPUtil
  RiXCore
  ${OPENGL_gl_LIBRARY}
  ${GLEW_LIBRARY}
)

if(WIN32)
  CopyGLEW( RiXGL "${DP_BINARY_PATH}" )
endif()

set_target_properties( RiXGL PROPERTIES SUFFIX ".rdr" FOLDER "RiX" )

