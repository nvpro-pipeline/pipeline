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
     "${SCENIX_BIN_PATH}/${scenix_build_config}/*.pdb"
     )

file(GLOB qtrdlls 
     "${QTDIR}/bin/qtopengl4.dll" 
     "${QTDIR}/bin/qtgui4.dll" 
     "${QTDIR}/bin/qtcore4.dll"
     "${QTDIR}/bin/qtdeclarative4.dll"
     "${QTDIR}/bin/qtnetwork4.dll"
     "${QTDIR}/bin/qtscript4.dll"
     "${QTDIR}/bin/qtscripttools4.dll"
     "${QTDIR}/bin/qtsql4.dll"
     "${QTDIR}/bin/qtsvg4.dll"
     "${QTDIR}/bin/qtxml4.dll"
     "${QTDIR}/bin/qtxmlpatterns4.dll"
     "${QTDIR}/bin/qtwebkit4.dll" 
     "${QTDIR}/bin/phonon4.dll"
     )

file(GLOB qtddlls 
     "${QTDIR}/bin/qtopengld4.dll" 
     "${QTDIR}/bin/qtguid4.dll" 
     "${QTDIR}/bin/qtcored4.dll"
     "${QTDIR}/bin/qtdeclaratived4.dll"
     "${QTDIR}/bin/qtnetworkd4.dll"
     "${QTDIR}/bin/qtscriptd4.dll"
     "${QTDIR}/bin/qtscripttoolsd4.dll"
     "${QTDIR}/bin/qtsqld4.dll"
     "${QTDIR}/bin/qtsvgd4.dll"
     "${QTDIR}/bin/qtxmld4.dll"
     "${QTDIR}/bin/qtxmlpatternsd4.dll"
     "${QTDIR}/bin/qtwebkitd4.dll"
     "${QTDIR}/bin/phonond4.dll"
     )

file(GLOB qtpdbs 
     "${QTDIR}/bin/qtopengld4.pdb" 
     "${QTDIR}/bin/qtguid4.pdb" 
     "${QTDIR}/bin/qtcored4.pdb"
     "${QTDIR}/bin/qtdeclaratived4.pdb"
     "${QTDIR}/bin/qtnetworkd4.pdb"
     "${QTDIR}/bin/qtscriptd4.pdb"
     "${QTDIR}/bin/qtscripttoolsd4.pdb"
     "${QTDIR}/bin/qtsqld4.pdb"
     "${QTDIR}/bin/qtsvgd4.pdb"
     "${QTDIR}/bin/qtxmld4.pdb"
     "${QTDIR}/bin/qtxmlpatternsd4.pdb"
     "${QTDIR}/bin/qtwebkitd4.pdb"
     "${QTDIR}/bin/phonond4.pdb"
     )

set(qtplugins
  iconengines/qsvgicon4.dll
)

set(qtpluginsd
  iconengines/qsvgicond4.dll
)

set(qtpluginpdbs
  iconengines/qsvgicond4.pdb
)

if(${build_configuration} STREQUAL "Debug")
  set(dlls   ${scenixdlls} ${scenixnxms} ${scenixpdbs})
  set(qtdlls ${qtddlls} ${qtpdbs})
  set(qtplugins ${qtpluginsd} ${qtpluginpdbs})
else()
  set(dlls   ${scenixdlls} ${scenixnxms})
  set(qtdlls ${qtrdlls})
endif()
                   
foreach(file ${dlls} ${qtdlls})
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

