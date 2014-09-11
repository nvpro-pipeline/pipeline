SET( DPT_COVER_RIXGL    ON CACHE BOOL "Include test framework for RiXGL" ) 

add_subdirectory( core )

if ( DPT_COVER_RIXGL )
  add_subdirectory( gl )
endif()
