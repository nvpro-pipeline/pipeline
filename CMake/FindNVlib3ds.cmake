file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/lib3ds/" LIB3DSDROOT)

if (WIN32)
  if ( EXISTS "${LIB3DSDROOT}" )
    file ( TO_NATIVE_PATH "${LIB3DSDROOT}" LIB3DSDROOT )
    set( ENV{LIB3DSDROOT} "${LIB3DSDROOT}" )
    set( LIB3DS_BINDIR "$ENV{LIB3DSDROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/bin" )
    set( LIB3DS_LIBDIR "$ENV{LIB3DSDROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/lib" )
    set( LIB3DS_INCLUDE_DIR "$ENV{LIB3DSDROOT}" )

    # we only have windows version at the moment
  endif()

elseif ( UNIX )
  find_path(LIB3DS_INCLUDE_DIR lib3ds/material.h)
  find_library(LIB3DS_LIBRARY NAMES 3ds)

  #include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
  #find_package_handle_standard_args(LIB3DS
  #  REQUIRED_VARS LIB3DS_INCLUDE_DIR LIB3DS_LIBRARY)

  mark_as_advanced(LIB3DS_INCLUDE_DIR LIB3DS_LIBRARY)

endif()
