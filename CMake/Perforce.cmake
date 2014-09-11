SET( DP_ADD_PERFORCE_BINDINGS  OFF CACHE BOOL "Add Perforce bindings" )

MACRO (add_perforce_to_target _target)
  IF ( DP_ADD_PERFORCE_BINDINGS )
    SET_TARGET_PROPERTIES(${_target} PROPERTIES
      VS_SCC_PROJECTNAME "Perforce Project"
      VS_SCC_LOCALPATH "${CMAKE_SOURCE_DIR}"
      VS_SCC_PROVIDER "MSSCCI:Perforce SCM"
    )
  ENDIF()
ENDMACRO()

