file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/fltLib/" FLTLIBROOT)

if ( EXISTS "${FLTLIBROOT}" )

  # we only have windows version at the moment
  if (WIN32)
  elseif ( UNIX )
  endif()

  file ( TO_NATIVE_PATH "${FLTLIBROOT}" FLTLIBROOT )
  set( ENV{FLTLIBROOT} "${FLTLIBROOT}" )
  set( FLTLIB_BINDIR "$ENV{FLTLIBROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/bin" )
  set( FLTLIB_LIBDIR "$ENV{FLTLIBROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/lib" )
  set( FLTLIB_INCDIR "$ENV{FLTLIBROOT}" )

endif()

