find_package( DevIL )

if (IL_FOUND)
  add_subdirectory( Loader )
  add_subdirectory( Saver )
endif()