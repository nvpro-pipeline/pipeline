MACRO(copy_file_if_changed target infile outfile)
  add_custom_command( TARGET ${target}
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND}
                      ARGS -E copy_if_different "${infile}" "${outfile}"
                      DEPENDS "${infile}"
                     )
ENDMACRO()

FUNCTION(copy_files_if_changed target)
  # get integer with last element index and target removed
  list(LENGTH ARGN list_count)
  math(EXPR list_max_index ${list_count}-2)
  if ( ${list_max_index} GREATER 0 )
    list(GET ARGN -1 target_folder )
    foreach(i RANGE ${list_max_index})
      list(GET ARGN ${i} source)
      get_filename_component(basename "${source}" NAME)
      copy_file_if_changed( target "${source}" "${target_folder}/${basename}" )
    endforeach()
  endif()
ENDFUNCTION()
