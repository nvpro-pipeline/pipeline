cmake_minimum_required(VERSION 2.6)

project( dpt )

set( DPUTIL_INCLUDES
  "${DP_HOME}/dp/util"
)

###############################################################################
# set target output directories

add_subdirectory( core )
add_subdirectory( manager )
add_subdirectory( app )
   
