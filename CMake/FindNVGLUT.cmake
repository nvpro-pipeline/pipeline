include(CopyFile)
if (WIN32)
  # FindGLUT by CMake is not able to differentiate between 32-bit/64-bit on win32.
  # Use the freeglut windows binaries folder-structure here.
  
  set (GLUTVERSION "2.8.0")
  
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set ( GLUT_ARCH_PATH "x64/")
  else ()
    set ( GLUT_ARCH_PATH "" )
  endif()
  
  file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/freeglut/${GLUTVERSION}" GLUT_ROOT_PATH )
  
  FIND_PATH( GLUT_INCLUDE_DIR NAMES GL/glut.h PATHS "${GLUT_ROOT_PATH}/include" )
  FIND_LIBRARY( GLUT_glut_LIBRARY NAMES freeglut PATHS "${GLUT_ROOT_PATH}/lib/${GLUT_ARCH_PATH}" )
  
  SET( GLUT_FOUND "NO" )
  
  IF(GLUT_INCLUDE_DIR)
    IF(GLUT_glut_LIBRARY)
      SET( GLUT_LIBRARIES
        ${GLUT_glut_LIBRARY}
        )
      SET( GLUT_FOUND "YES" )
      
    ENDIF(GLUT_glut_LIBRARY)
  ENDIF(GLUT_INCLUDE_DIR)

  MARK_AS_ADVANCED(
    GLUT_INCLUDE_DIR
    GLUT_glut_LIBRARY
  )
  
  FUNCTION(CopyGLUT target path)
    copy_file_if_changed( ${target} "${GLUT_ROOT_PATH}/bin/${GLUT_ARCH_PATH}/freeglut.dll" "${path}" )
  ENDFUNCTION()
  
elseif(UNIX)
  find_package( GLUT REQUIRED )

  FUNCTION(CopyGLUT target path)
  ENDFUNCTION()
endif()
