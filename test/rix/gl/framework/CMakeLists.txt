
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT)

if (GLUT_FOUND)
  #includes
  include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    "${GLEW_INCLUDE_DIRS}"
    "${GLUT_INCLUDE_DIR}"
  )

  #definitions
  add_definitions(
    "-DDPTRIXGL_EXPORTS"
    "-D_CRT_SECURE_NO_WARNINGS"
  )

  #sources
  set (DPTRIXGL_SOURCES
    src/RiXGLBackend.cpp
  )

  set (DPTRIXGL_HEADERS
    inc/Config.h
  )

  set (DPTRIXGL_PUBLIC_HEADERS
    RiXGLBackend.h
  )

  source_group(headers FILES ${DPTRIXGL_HEADERS})
  source_group(sources FILES ${DPTRIXGL_SOURCES})
  source_group("" FILES ${DPTRIXGL_PUBLIC_HEADERS})

  #target
  add_library( DPTRiXGL SHARED
     ${DPTRIXGL_SOURCES}
     ${DPTRIXGL_HEADERS}
     ${DPTRIXGL_PUBLIC_HEADERS}
  )

  set_target_properties( DPTRiXGL PROPERTIES SUFFIX ".bkd" FOLDER "DPT/RiX/GL" )
set_target_properties( DPTRiXGL PROPERTIES LINK_FLAGS "/NODEFAULTLIB:\"freeglutd.lib\" /NODEFAULTLIB:\"freeglut.lib\"" )

target_link_libraries( DPTRiXGL
  ${GLUT_LIBRARIES}
  DPUtil
  DPTcore
  RiXGL
  DPTRiX
  DPGL
)

  CopyGLUT( DPTRiXGL "${DP_BINARY_PATH}" )

  add_dependencies( DPTRiXGL DPUtil DPTcore DPTRiX RiXCore RiXGL )
  
else()
  message("GLUT not found, disabling DPTRiXGL.")
endif()