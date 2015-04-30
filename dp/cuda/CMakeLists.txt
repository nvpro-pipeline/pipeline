find_package(GLEW REQUIRED)
find_package(CUDA)

if (CUDA_FOUND)
  #includes
  include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
  )
  #definitions
  add_definitions(
    -DDP_CUDA_EXPORTS
    ${CUDA_DEFINITIONS}
    ${GLEW_DEFINITIONS}
  )

  #sources
  set(SOURCES
    src/Buffer.cpp
    src/Buffer3D.cpp
    src/BufferHost.cpp
    src/Device.cpp
    src/Event.cpp
    src/GraphicsResource.cpp
    src/Stream.cpp
  )

  set(HEADERS
    Buffer.h
    Buffer3D.h
    BufferHost.h
    Config.h
    Device.h
    Event.h
    GraphicsResource.h
    Stream.h
    Types.h
  )

  source_group(sources FILES ${SOURCES})
  source_group(headers FILES ${HEADERS})

  #target
  cuda_add_library( DPCUDA SHARED
    ${SOURCES}
    ${HEADERS}
  )

  target_link_libraries( DPCUDA
    DPGL
    DPUtil
  )
else()
  message("CUDA not found, disabling DPCUDA")
endif()
