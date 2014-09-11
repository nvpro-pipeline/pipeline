project( RiXGL )

file(TO_CMAKE_PATH $ENV{DP_3RDPARTY_PATH} DP_3RDPARTY_PATH)
set(CMAKE_MODULE_PATH "${DP_3RDPARTY_PATH}/CMake")


find_package(OpenGL REQUIRED )
find_package(NVGLEW REQUIRED )

###############################################################################
# Detect pieces of the bin/lib path
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  add_definitions("-DX86_64")
else ()
  set ( MSVC_CXX_FLAGS "/arch:SSE2" )
endif()

if(MSVC10)
  set (CRT "crt100")
  set ( COMPILER_ARCH "win32-msvc2010-${DP_ARCH}" )
  #enable multiprocessor build on visual studio 2010
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP ${MSVC_CXX_FLAGS}")
elseif(MSVC11)
  set (CRT "crt110")
  set ( COMPILER_ARCH "win32-msvc2012-${DP_ARCH}" )
  #enable multiprocessor build on visual studio 2012
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP ${MSVC_CXX_FLAGS}")
else()
  message(FATAL_ERROR "Compiler version not supported")
endif()

#includes
include_directories(
  "${PROJECT_SOURCE_DIR}/../Renderer/inc"
  "${PROJECT_SOURCE_DIR}/inc"
  ${GLEW_INCLUDES}
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
  src/Matrix.cpp
  src/ParameterCacheStream.cpp
  src/ParameterCacheEntryStream.cpp
  src/ParameterCacheEntryStreamBuffer.cpp
  src/ParameterRenderer.cpp
  src/ParameterRendererBuffer.cpp
  src/ParameterRendererBufferDSA.cpp
  src/ParameterRendererBufferRange.cpp
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
  inc/Matrix.h
  inc/ProgramGL.h
  inc/ParameterCache.h
  inc/ParameterCacheStream.h
  inc/ParameterCacheEntryStream.h
  inc/ParameterCacheEntryStreamBuffer.h
  inc/ParameterRenderer.h
  inc/ParameterRendererBuffer.h
  inc/ParameterRendererBufferDSA.h
  inc/ParameterRendererBufferRange.h
  inc/ParameterRendererStream.h
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
  inc/Vector.h
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

add_perforce_to_target( RiXGL )

target_link_libraries( RiXGL
  RiXCore
  DPGL
  DPUtil
  ${GLEW_LIBS}
  ${OPENGL_gl_LIBRARY}
)

CopyGLEW( RiXGL "${DP_BINARY_PATH}" )

set_target_properties( RiXGL PROPERTIES SUFFIX ".rdr" FOLDER "RiX" )

