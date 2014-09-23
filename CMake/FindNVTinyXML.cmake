# - Try to find TinyXML
# Once done this will define
#  TINYXML_FOUND - System has TinyXML
#  TINYXML_INCLUDE_DIRS - The TinyXML include directories
#  TINYXML_LIBRARIES - The libraries needed to use TinyXML
#  TINYXML_DEFINITIONS - Compiler switches required for using TinyXML

find_path(TINYXML_INCLUDE_DIR "tinyxml.h")
find_library(TINYXML_LIBRARY NAMES tinyxml.lib)

set(TINYXML_LIBRARIES ${TINYXML_LIBRARY} )
set(TINYXML_INCLUDE_DIRS ${TINYXML_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set TINYXML_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(TinyXML  DEFAULT_MSG
                                  TINYXML_LIBRARY TINYXML_INCLUDE_DIR)

mark_as_advanced(TINYXML_INCLUDE_DIR TINYXML_LIBRARY )