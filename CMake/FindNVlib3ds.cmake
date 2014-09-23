# - Try to find lib3ds
# Once done this will define
#  LIB3DS_FOUND - System has lib3ds
#  LIB3DS_INCLUDE_DIRS - The lib3ds include directories
#  LIB3DS_LIBRARIES - The libraries needed to use lib3ds
#  LIB3DS_DEFINITIONS - Compiler switches required for using lib3ds

find_path(LIB3DS_INCLUDE_DIR "lib3ds/lib3ds.h")
find_library(LIB3DS_LIBRARY NAMES lib3ds.lib PATH_SUFFIXES "/lib")

set(LIB3DS_LIBRARIES ${LIB3DS_LIBRARY} )
set(LIB3DS_INCLUDE_DIRS ${LIB3DS_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIB3DS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(lib3ds DEFAULT_MSG
                                  LIB3DS_LIBRARY LIB3DS_INCLUDE_DIR)

mark_as_advanced(LIB3DS_INCLUDE_DIR LIB3DS_LIBRARY )