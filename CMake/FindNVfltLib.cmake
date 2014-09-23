# - Try to find fltlib
# Once done this will define
#  FLTLIB_FOUND - System has fltlib
#  FLTLIB_INCLUDE_DIRS - The fltlib include directories
#  FLTLIB_LIBRARIES - The libraries needed to use fltlib
#  FLTLIB_DEFINITIONS - Compiler switches required for using fltlib

find_path(FLTLIB_INCLUDE_DIR "flt.h")
find_library(FLTLIB_LIBRARY NAMES OpenFlight.lib)

set(FLTLIB_LIBRARIES ${FLTLIB_LIBRARY} )
set(FLTLIB_INCLUDE_DIRS ${FLTLIB_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set FLTLIB_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(fltlib  DEFAULT_MSG
                                  FLTLIB_LIBRARY FLTLIB_INCLUDE_DIR)

mark_as_advanced(FLTLIB_INCLUDE_DIR FLTLIB_LIBRARY )