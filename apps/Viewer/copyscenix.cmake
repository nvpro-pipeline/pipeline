# The following variables come from the caller
# build_configuration
# SCENIX_BIN_PATH
# BUILDDIR

# This will allow building Cheetah with configurations such as RelWithDebInfo.
if(build_configuration STREQUAL "Debug")
  set(scenix_build_config "Debug")
else()
  set(scenix_build_config "Release")
endif()

file(GLOB scenixdlls
     "${SCENIX_BIN_PATH}/${scenix_build_config}/*.dll"
     #"${SCENIX_BIN_PATH}/${scenix_build_config}/cg*.dll"
     #"${SCENIX_BIN_PATH}/${scenix_build_config}/cuda*.dll"
     #"${SCENIX_BIN_PATH}/${scenix_build_config}/lib3ds.dll"
     #"${SCENIX_BIN_PATH}/${scenix_build_config}/scenix*.dll"
     #"${SCENIX_BIN_PATH}/${scenix_build_config}/optix*.dll"
     )

file(GLOB scenixnxms 
     "${SCENIX_BIN_PATH}/${scenix_build_config}/*.nxm"
     #"${SCENIX_BIN_PATH}/${scenix_build_config}/DAELoader.nxm"
     #"${SCENIX_BIN_PATH}/${scenix_build_config}/3DSLoader.nxm"
     #"${SCENIX_BIN_PATH}/${scenix_build_config}/ILTexLoader.nxm"
     #"${SCENIX_BIN_PATH}/${scenix_build_config}/ILTexSaver.nxm"
     )

file(GLOB scenixpdbs
     "${SCENIX_BIN_PATH}/${scenix_build_config}/3dsloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/daeloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/fltloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/iltexloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/iltexsaver.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/lib3ds.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/modcpx9.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/mslcompiler.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/nbfloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/nbfsaver.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/nvbloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/nvsgloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/nvsgsaver.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/objloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/objsaver.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/openflight.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/plyloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/rtfxc.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/scenix9.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/scenixneuray9.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/scenixrt9.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/volloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/volsaver.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/wrlloader.pdb"
     "${SCENIX_BIN_PATH}/${scenix_build_config}/xmlloader.pdb"
     )

if(${build_configuration} STREQUAL "Debug")
  set(dlls   ${scenixdlls} ${scenixnxms} ${scenixpdbs})
else()
  set(dlls   ${scenixdlls} ${scenixnxms})
endif()
                   
foreach(file ${dlls})
  get_filename_component(filename_nopath "${file}" NAME)
  set(dest_file "${BUILDDIR}/${build_configuration}/${filename_nopath}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo copy "${file}" "${dest_file}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E      copy "${file}" "${dest_file}"
    RESULT_VARIABLE result
    )
  if(result)
    message(FATAL_ERROR "Error copying file ${file} to ${dest_file}")
  endif()
endforeach()

foreach(file ${qtplugins})
  set(src_file "${QTDIR}/plugins/${file}")
  set(dest_file "${BUILDDIR}/${build_configuration}/plugins/${file}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo copy "${src_file}" "${dest_file}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E      copy "${src_file}" "${dest_file}"
    RESULT_VARIABLE result
    )
  if(result)
    message(FATAL_ERROR "Error copying file ${src_file} to ${dest_file}")
  endif()
endforeach()

