
SET( DPT_COVER_RIX ON CACHE BOOL "Test framework for RiX is included" )
SET( DPT_COVER_SGRDR ON CACHE BOOL "Test framework for SgRdr is included" )

add_subdirectory( testfw )

if ( DPT_COVER_RIX )
  add_subdirectory( rix )
endif()

if ( DPT_COVER_SGRDR )
  add_subdirectory( sgrdr )
endif()

add_subdirectory( scripts )