set( HOOPS_VERSION "5.0" )
file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/HOOPS/${HOOPS_VERSION}/" HOOPS3DROOT)

if ( EXISTS "${HOOPS3DROOT}" )
  set( HOOPS_FOUND "true" )
  if ( "${DP_ARCH}" STREQUAL "x86" )
    set( HOOPS_ARCH "win32" )
  elseif ( "${DP_ARCH}" STREQUAL "amd64" )
    set( HOOPS_ARCH "win64" )
  endif()

  # we only have windows version at the moment
  if (WIN32)
  elseif ( UNIX )
  endif()

  file ( TO_NATIVE_PATH "${HOOPS3DROOT}" HOOPS3DROOT )
  set( ENV{HOOPS3DROOT} "${HOOPS3DROOT}" )
  set( HOOPS3D_BINDIR   "$ENV{HOOPS3DROOT}/bin/${HOOPS_ARCH}" )
  set( HOOPS3D_INCLUDES "$ENV{HOOPS3DROOT}/include" )

endif()

