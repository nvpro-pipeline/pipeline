if (WIN32)
  file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/fltLib/" FLTLIBROOT)
  if ( EXISTS "${FLTLIBROOT}" )

    file ( TO_NATIVE_PATH "${FLTLIBROOT}" FLTLIBROOT )
    set( ENV{FLTLIBROOT} "${FLTLIBROOT}" )
    set( FLTLIB_BINDIR "$ENV{FLTLIBROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/bin" )
    set( FLTLIB_LIBDIR "$ENV{FLTLIBROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/lib" )
    set( FLTLIB_INCLUDE_DIR "$ENV{FLTLIBROOT}" )
  endif()
elseif ( UNIX )
  #find_path(LIB3DS_INCLUDE_DIR lib3ds/material.h)
  #find_library(LIB3DS_LIBRARY NAMES 3ds)

  #TODO how to load the package from below?
  #include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
  #find_package_handle_standard_args(LIB3DS
  #  REQUIRED_VARS LIB3DS_INCLUDE_DIR LIB3DS_LIBRARY)

  #mark_as_advanced(LIB3DS_INCLUDE_DIR LIB3DS_LIBRARY)

endif()

