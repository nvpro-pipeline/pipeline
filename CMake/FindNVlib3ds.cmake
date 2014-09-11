file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/lib3ds/" LIB3DSDROOT)

if ( EXISTS "${LIB3DSDROOT}" )

  # we only have windows version at the moment
  if (WIN32)
  elseif ( UNIX )
  endif()

  file ( TO_NATIVE_PATH "${LIB3DSDROOT}" LIB3DSDROOT )
  set( ENV{LIB3DSDROOT} "${LIB3DSDROOT}" )
  set( LIB3DS_BINDIR "$ENV{LIB3DSDROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/bin" )
  set( LIB3DS_LIBDIR "$ENV{LIB3DSDROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/lib" )
  set( LIB3DS_INCDIR "$ENV{LIB3DSDROOT}" )

endif()

