# The following variables come from the caller
# build_configuration
# SCENIX_BIN_PATH
# BUILDDIR

# This will allow building Cheetah with configurations such as RelWithDebInfo.

if ("${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
  if( build_configuration STREQUAL "Debug")
    set(scenix_build_config "Debug")
  else()
    set(scenix_build_config "Release")
  endif()
  set(scenix_build_dir "${BUILDDIR}/${scenix_build_config}")
else()
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(scenix_build_config "Debug")
  else()
    set(scenix_build_config "Release")
  endif()
  set(scenix_build_dir "${BUILDDIR}")
endif()

file(GLOB qtrdlls 
     "${QTDIR}/bin/Qt5OpenGL.dll" 
     "${QTDIR}/bin/Qt5Gui.dll" 
     "${QTDIR}/bin/Qt5Widgets.dll" 
     "${QTDIR}/bin/Qt5Core.dll"
     "${QTDIR}/bin/Qt5Script.dll"
     )

file(GLOB qtddlls 
     "${QTDIR}/bin/Qt5OpenGLd.dll" 
     "${QTDIR}/bin/Qt5Guid.dll" 
     "${QTDIR}/bin/Qt5Widgetsd.dll" 
     "${QTDIR}/bin/Qt5Cored.dll"
     "${QTDIR}/bin/Qt5Scriptd.dll"
     )

file(GLOB qtpdbs 
     "${QTDIR}/bin/Qt5OpenGLd.pdb" 
     "${QTDIR}/bin/Qt5Guid.pdb" 
     "${QTDIR}/bin/Qt5Widgetsd.pdb" 
     "${QTDIR}/bin/Qt5Cored.pdb"
     "${QTDIR}/bin/Qt5Scriptd.pdb"
     )

set(qtplugins
  iconengines/qsvgicon.dll
)

set(qtpluginsd
  iconengines/qsvgicond.dll
)

set(qtpluginpdbs
  iconengines/qsvgicond.pdb
)

if(scenix_build_config STREQUAL "Debug")
  set(qtdlls ${qtddlls} ${qtpdbs})
  set(qtplugins ${qtpluginsd} ${qtpluginpdbs})
  set(qtplatform qwindowsd.dll)
else()
  set(qtdlls ${qtrdlls})
  set(qtplatform qwindows.dll)
endif()
                   
foreach(file ${qtdlls})
  get_filename_component(filename_nopath "${file}" NAME)
  set(dest_file "${scenix_build_dir}/${filename_nopath}")
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
  set(dest_file "${scenix_build_dir}/plugins/${file}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo copy "${src_file}" "${dest_file}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E      copy "${src_file}" "${dest_file}"
    RESULT_VARIABLE result
    )
  if(result)
    message(FATAL_ERROR "Error copying file ${src_file} to ${dest_file}")
  endif()
endforeach()

execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${QTDIR}/plugins/platforms/${qtplatform}" "${scenix_build_dir}/platforms/${qtplatform}")


