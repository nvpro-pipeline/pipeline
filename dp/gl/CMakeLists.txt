find_package( OpenGL REQUIRED )
find_package( NVGLEW REQUIRED )
find_package( NVBoost REQUIRED )
find_package( Boost COMPONENTS chrono date_time system thread )

#includes
include_directories(
  ${GLEW_INCLUDES}
)
#definitions
add_definitions(
  -DDP_GL_EXPORTS 
  ${GLEW_DEFINITIONS}
)  

#sources
set(SOURCES
  src/Buffer.cpp
  src/DisplayList.cpp
  src/Object.cpp
  src/Program.cpp
  src/Renderbuffer.cpp
  src/RenderContext.cpp
  src/RenderContextFormat.cpp
  src/RenderTarget.cpp
  src/RenderTargetFB.cpp
  src/RenderTargetFBO.cpp
  src/Shader.cpp
  src/Texture.cpp
)

set(PUBLIC_HEADERS
  Buffer.h
  Config.h
  DisplayList.h
  Object.h
  Program.h
  Renderbuffer.h
  RenderContext.h
  RenderContextFormat.h
  RenderTarget.h
  RenderTargetFB.h
  RenderTargetFBO.h
  Shader.h
  Texture.h
)

set(PRIVATE_HEADERS
  inc/BufferBind.h
  inc/BufferDSA.h
  inc/IBuffer.h
  inc/ITexture.h
  inc/TextureBind.h
  inc/TextureDSA.h
)

source_group(sources FILES ${SOURCES})
source_group(headers FILES ${PRIVATE_HEADERS})
source_group("" FILES ${PUBLIC_HEADERS})

#target
add_library( DPGL SHARED
  ${SOURCES}
  ${PUBLIC_HEADERS}
  ${PRIVATE_HEADERS}
)

add_perforce_to_target( DPGL )

target_link_libraries( DPGL
  DPUtil
  DPUI
  ${OPENGL_gl_LIBRARY}
  ${GLEW_LIBS}
  ${Boost_LIBRARIES}
)

CopyGLEW( DPGL "${DP_BINARY_PATH}" )
