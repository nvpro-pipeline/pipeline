file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/OpenEXR/" OPENEXRROOT)

if ( EXISTS "${OPENEXRROOT}" )

  # we only have windows version at the moment
  if (WIN32)
  elseif ( UNIX )
  endif()

  set( ENV{OPENEXRROOT} "${OPENEXRROOT}" )
  set( OPENEXR_BINDIR "$ENV{OPENEXRROOT}/win32-${DP_ARCH}/bin" )
  set( OPENEXR_LIBDIR "$ENV{OPENEXRROOT}/win32-${DP_ARCH}/lib" )
  set( OPENEXR_INCDIR "$ENV{OPENEXRROOT}/include" )

endif()

