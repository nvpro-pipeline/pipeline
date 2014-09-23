# - Try to find OpenEXR
# Once done this will define
#  OPENEXR_FOUND - System has OpenEXR
#  OPENEXR_INCLUDE_DIRS - The OpenEXR include directories
#  OPENEXR_LIBRARIES - The libraries needed to use OpenEXR
#  OPENEXR_DEFINITIONS - Compiler switches required for using OpenEXR

find_path(OPENEXR_INCLUDE_DIR "Iex.h")
foreach(LIB Iex IlmImf IlmThread Imath Half)
  find_library(OPENEXR_${LIB}_LIBRARY_DEBUG NAMES ${LIB} PATH_SUFFIXES "/win32-${DP_ARCH}/lib/Debug")
  find_library(OPENEXR_${LIB}_LIBRARY_RELEASE NAMES ${LIB} PATH_SUFFIXES "/win32-${DP_ARCH}/lib/Release")
  
  list(APPEND OPENEXR_LIBRARIES debug ${OPENEXR_${LIB}_LIBRARY_DEBUG} optimized ${OPENEXR_${LIB}_LIBRARY_RELEASE})
  list(APPEND OPENEXR_LIBRARIES_REQUIRED OPENEXR_${LIB}_LIBRARY_DEBUG OPENEXR_${LIB}_LIBRARY_RELEASE)
endforeach()

set(OPENEXR_INCLUDE_DIRS ${OPENEXR_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OpenEXR DEFAULT_MSG
                                  ${OPENEXR_LIBRARIES_REQUIRED} OPENEXR_INCLUDE_DIR)
                                  
mark_as_advanced(OPENEXR_INCLUDE_DIR ${OPENEXR_LIBRARIES_REQUIRED} )