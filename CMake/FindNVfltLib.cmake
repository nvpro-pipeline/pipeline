# - Try to find fltlib
# Once done this will define
#  FLTLIB_FOUND - System has fltlib
#  FLTLIB_INCLUDE_DIRS - The fltlib include directories
#  FLTLIB_LIBRARIES - The libraries needed to use fltlib
#  FLTLIB_DEFINITIONS - Compiler switches required for using fltlib

file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/fltLib/" FLTLIBROOT )

if ( EXISTS "${FLTLIBROOT}" )
  set( FLTLIB_FOUND        "true" )
  set( FLTLIB_INCLUDE_DIRS "${FLTLIBROOT}" )
  set( FLTLIB_LIBRARIES    optimized "${FLTLIBROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/lib/Release/OpenFlight.lib"
                           debug     "${FLTLIBROOT}/win32-msvc${MSVC_VERSION}-${DP_ARCH}/lib/Debug/OpenFlight.lib" )
endif()


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set FLTLIB_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(fltlib  DEFAULT_MSG
                                  FLTLIB_LIBRARIES FLTLIB_INCLUDE_DIRS)

mark_as_advanced(FLTLIB_INCLUDE_DIRS FLTLIB_LIBRARIES )