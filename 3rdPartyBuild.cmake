cmake_minimum_required(VERSION 2.8.12)

#determine Visual Studio compiler version
execute_process(COMMAND "cl.exe" OUTPUT_VARIABLE dummy ERROR_VARIABLE cl_info_string)
string(REGEX REPLACE ".*Version (..).*" "\\1" cl_major_version ${cl_info_string})
string(REGEX REPLACE ".*for (...).*" "\\1" cl_architecture ${cl_info_string})

if ("${cl_architecture}" STREQUAL "x64")
  set(BUILD_ARCH x64)
  set(GENERATOR_ARCH Win64)
  set(BOOST_ADDRESS_MODEL 64)
elseif ("${cl_architecture}" STREQUAL "x86")
  set(BUILD_ARCH x86)
  set(GENERATOR_ARCH)
  set(BOOST_ADDRESS_MODEL 32)
else()
  message(FATAL_ERROR "unsupported CPU architecture")
endif()

if("${cl_major_version}" STREQUAL "17")
  set(GENERATOR "Visual Studio 11 ${GENERATOR_ARCH}")
  set(BOOST_TOOLSET msvc-11.0)
endif()

if("${cl_major_version}" STREQUAL "18")
  set(GENERATOR "Visual Studio 12 ${GENERATOR_ARCH}")
  set(BOOST_TOOLSET msvc-12.0)
endif()

if (NOT GENERATOR)
  message(FATAL_ERROR "no generator found, exit")
endif()

message("Creating 3rdparty library folder for ${GENERATOR}")


set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty" CACHE PATH "default install path" FORCE)
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/sources")
set(DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/downloads")
set(PATCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/patches")
set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/build")

message("Install prefix: ${CMAKE_INSTALL_PREFIX} ${ARGC} ${ARGV}")

file(MAKE_DIRECTORY ${SOURCE_DIR})
file(MAKE_DIRECTORY ${BUILD_DIR})

macro(lib3ds)
    message("Building lib3ds")
    set(FILENAME "lib3ds-20080909.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        file(DOWNLOAD "https://lib3ds.googlecode.com/files/${FILENAME}" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${DOWNLOAD_DIR}/${FILENAME} WORKING_DIRECTORY "${SOURCE_DIR}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${PATCH_DIR}/${FILENAME} WORKING_DIRECTORY "${SOURCE_DIR}")
    
    set(BUILD_DIRECTORY "${BUILD_DIR}/lib3ds")
    if (EXISTS "${BUILD_DIRECTORY}")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${BUILD_DIRECTORY}")
    endif()
    file(MAKE_DIRECTORY "${BUILD_DIRECTORY}")
    execute_process(COMMAND ${CMAKE_COMMAND} "-G${GENERATOR}" "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/lib3ds" "${SOURCE_DIR}/lib3ds-20080909" WORKING_DIRECTORY "${BUILD_DIRECTORY}")
    execute_process(COMMAND devenv.exe "${BUILD_DIRECTORY}/lib3ds.sln" /Build "Release|${BUILD_ARCH}" WORKING_DIRECTORY "${BUILD_DIRECTORY}")
    execute_process(COMMAND devenv.exe "${BUILD_DIRECTORY}/lib3ds.sln" /Build "Release|${BUILD_ARCH}" /Project INSTALL WORKING_DIRECTORY "${BUILD_DIRECTORY}")
endmacro()

macro(fltlib)
    message("Building fltlib")
    set(FILENAME "fltlib-1.0.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        file(DOWNLOAD "http://downloads.sourceforge.net/project/fltlib/fltlib/fltlib%201.0/${FILENAME}" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${SOURCE_DIR}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${PATCH_DIR}/${FILENAME}" WORKING_DIRECTORY "${SOURCE_DIR}")
    
    set(BUILD_DIRECTORY "${BUILD_DIR}/fltlib")
    if (EXISTS "${BUILD_DIRECTORY}")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${BUILD_DIRECTORY}")
    endif()
    file(MAKE_DIRECTORY "${BUILD_DIRECTORY}")
    execute_process(COMMAND ${CMAKE_COMMAND} "-G${GENERATOR}" "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/fltlib" "${SOURCE_DIR}/fltlib" WORKING_DIRECTORY "${BUILD_DIRECTORY}")
    execute_process(COMMAND devenv.exe "${BUILD_DIRECTORY}/fltlib.sln" /Build "Release|${BUILD_ARCH}" WORKING_DIRECTORY "${BUILD_DIRECTORY}")
    execute_process(COMMAND devenv.exe "${BUILD_DIRECTORY}/fltlib.sln" /Build "Release|${BUILD_ARCH}" /Project INSTALL WORKING_DIRECTORY "${BUILD_DIRECTORY}")
endmacro()

macro(tinyxml)
    message("Building TinyXML")
    set(FILENAME "tinyxml_2_6_2.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        file(DOWNLOAD "http://downloads.sourceforge.net/project/tinyxml/tinyxml/2.6.2/${FILENAME}" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${SOURCE_DIR}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${PATCH_DIR}/${FILENAME}" WORKING_DIRECTORY "${SOURCE_DIR}")
    
    set(BUILD_DIRECTORY "${BUILD_DIR}/tinyxml")
    if (EXISTS "${BUILD_DIRECTORY}")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${BUILD_DIRECTORY}")
    endif()
    file(MAKE_DIRECTORY "${BUILD_DIRECTORY}")
    execute_process(COMMAND ${CMAKE_COMMAND} "-G${GENERATOR}" "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/tinyxml" "${SOURCE_DIR}/tinyxml" WORKING_DIRECTORY "${BUILD_DIRECTORY}")
    execute_process(COMMAND devenv.exe "${BUILD_DIRECTORY}/tinyxml.sln" /Build "Release|${BUILD_ARCH}" WORKING_DIRECTORY "${BUILD_DIRECTORY}")
    execute_process(COMMAND devenv.exe "${BUILD_DIRECTORY}/tinyxml.sln" /Build "Release|${BUILD_ARCH}" /Project INSTALL WORKING_DIRECTORY "${BUILD_DIRECTORY}")
endmacro()

macro(boost)
    message("Building Boost.")
    if (NOT EXISTS "${DOWNLOAD_DIR}/boost_1_58_0.tar.bz2")
      message("Downloading Boost. This might take a while.")
      file(DOWNLOAD "http://downloads.sourceforge.net/project/boost/boost/1.58.0/boost_1_58_0.tar.bz2" "${DOWNLOAD_DIR}/boost_1_58_0.tar.bz2" STATUS downloaded)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/boost_1_58_0.tar.bz2" WORKING_DIRECTORY "${SOURCE_DIR}")
    
    execute_process(COMMAND cmd.exe "/C" "bootstrap.bat" WORKING_DIRECTORY "${SOURCE_DIR}/boost_1_58_0")
    execute_process(COMMAND cmd.exe /C b2 -j "$ENV{NUMBER_OF_PROCESSORS}" --toolset=${BOOST_TOOLSET} address-model=${BOOST_ADDRESS_MODEL} install --prefix=${CMAKE_INSTALL_PREFIX}/boost/ WORKING_DIRECTORY "${SOURCE_DIR}/boost_1_58_0")
endmacro()

macro(devil)
    if (NOT EXISTS "${DOWNLOAD_DIR}/DevIL-SDK-x64-1.7.8.zip")
      file(DOWNLOAD "http://downloads.sourceforge.net/project/openil/DevIL%20Windows%20SDK/1.7.8/DevIL-SDK-x64-1.7.8.zip" "${DOWNLOAD_DIR}/DevIL-SDK-x64-1.7.8.zip" STATUS downloaded)
    endif()
    file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/devil")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/DevIL-SDK-x64-1.7.8.zip" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}/devil")
endmacro()

macro(freeglut)
    message("downloading freeglut")
    if (NOT EXISTS "${DOWNLOAD_DIR}/freeglut-MSVC.zip")
        file(DOWNLOAD "http://files.transmissionzero.co.uk/software/development/GLUT/freeglut-MSVC.zip" "${DOWNLOAD_DIR}/freeglut-MSVC.zip" STATUS downloaded)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/freeglut-MSVC.zip" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
endmacro()

macro(glew)
    message("downloading GLEW")
    set(FILENAME "glew-1.12.0-win32.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        file(DOWNLOAD "http://downloads.sourceforge.net/project/glew/glew/1.12.0/${FILENAME}" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
    
    # remove old glew directory
    if (EXISTS "${CMAKE_INSTALL_PREFIX}/glew")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}/glew")
    endif()
    
    file(RENAME "${CMAKE_INSTALL_PREFIX}/glew-1.12.0" "${CMAKE_INSTALL_PREFIX}/glew")
endmacro()

lib3ds()
#fltlib()
tinyxml()
boost()
devil()
freeglut()
glew()

#message("Cleaning up")
#execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_SOURCE_DIR}/temp")
