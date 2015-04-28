include("dp")

if( UNIX )
  set(Boost_USE_STATIC_LIBS       OFF)
  set(Boost_USE_MULTITHREADED      ON)
  set(Boost_USE_STATIC_RUNTIME    OFF)
endif()

### Add 3rdparty libraries to the CMake search paths
if(WIN32)
  ### boost
  if (NOT BOOSTROOT)
    file( TO_CMAKE_PATH "${DP_3RDPARTY_PATH}/Boost" BOOSTROOT)

    if ( EXISTS "${BOOSTROOT}" )
      set( BOOST_ROOT "${BOOSTROOT}")
      set(Boost_USE_STATIC_LIBS "ON")
    endif()
  endif()

  ### fltlib
  # Currently disabled since a new version is required
  #list(APPEND CMAKE_INCLUDE_PATH "${DP_3RDPARTY_PATH}/fltlib/include")
  #list(APPEND CMAKE_LIBRARY_PATH "${DP_3RDPARTY_PATH}/fltlib/lib")

  ### lib3ds
  list(APPEND CMAKE_INCLUDE_PATH "${DP_3RDPARTY_PATH}/lib3ds/include")
  list(APPEND CMAKE_LIBRARY_PATH "${DP_3RDPARTY_PATH}/lib3ds/lib")

  ### tinyxml
  list(APPEND CMAKE_INCLUDE_PATH "${DP_3RDPARTY_PATH}/tinyxml/include")
  list(APPEND CMAKE_LIBRARY_PATH "${DP_3RDPARTY_PATH}/tinyxml/lib")

  ### OpenEXR
  list(APPEND CMAKE_INCLUDE_PATH "${DP_3RDPARTY_PATH}/OpenEXR/include")
  list(APPEND CMAKE_LIBRARY_PATH "${DP_3RDPARTY_PATH}/OpenEXR")

  ### DevIL
  list(APPEND CMAKE_INCLUDE_PATH "${DP_3RDPARTY_PATH}/DevIL/include")
  list(APPEND CMAKE_LIBRARY_PATH "${DP_3RDPARTY_PATH}/DevIL")

  ### GLEW
  list(APPEND CMAKE_INCLUDE_PATH "${DP_3RDPARTY_PATH}/glew/include")
  # TODO OS arch dependency?
  list(APPEND CMAKE_LIBRARY_PATH "${DP_3RDPARTY_PATH}/glew/lib/Release/x64")

  ### freeglut
  list(APPEND CMAKE_INCLUDE_PATH "${DP_3RDPARTY_PATH}/freeglut/include")
  # TODO OS arch dependency?
  list(APPEND CMAKE_LIBRARY_PATH "${DP_3RDPARTY_PATH}/freeglut/lib")

  ### Qt

  # Search for Windows SDK so that Qt5 can find glu32.lib
  if(WIN32)
    foreach(WINKIT "KitsRoot81" "KitsRoot")
      get_filename_component(WINKIT_DIR "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;${WINKIT}]" ABSOLUTE)
      if(EXISTS ${WINKIT_DIR})
        if ( ${DP_ARCH} STREQUAL "amd64" )
          set(WINKIT_ARCH "x64")
        elseif(${DP_ARCH} STREQUAL "x86")
          set(WINKIT_ARCH "x86")
        else()
          message(FATAL "unsupported architecture")
        endif()

        find_path(WINKIT_LIB GlU32.lib PATHS "${WINKIT_DIR}/Lib/win8/um/${WINKIT_ARCH}" "${WINKIT_DIR}/Lib/winv6.3/um/${WINKIT_ARCH}")

        list(APPEND CMAKE_PREFIX_PATH "${WINKIT_LIB}")
        message("Using Windows Kit ${WINKIT_DIR}")
        set(WINKIT_FOUND TRUE)
        break()
      endif()
    endforeach()

    if (NOT ${WINKIT_FOUND})
      message("Windows Kit not found. Qt might not find glu32.lib")
    endif()
  endif()

  get_filename_component(QtRoot "[HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Qt;InstallLocation]" ABSOLUTE)
  if (EXISTS ${QtRoot})
    if (${DP_ARCH} STREQUAL "amd64")
      set(QtArch "_64")
    endif()

    if (MSVC11)
      set(QtCompiler "msvc2012")
      if (${DP_ARCH} STREQUAL "amd64")
        set(QtVersion "5.2.1")
      else()
        set(QtVersion "5.3")
      endif()
    elseif(MSVC12)
      set(QtCompiler "msvc2013")
      set(QtVersion "5.4")

	  if(NOT EXISTS "${QtRoot}/${QtVersion}")
		message("Qt 5.4 not found. Fall back to Qt 5.3")
		set(QtVersion "5.3")
	  endif()
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${QtRoot}/${QtVersion}/${QtCompiler}${QtArch}_opengl/lib/cmake")

  endif()

endif()

macro(CopyDevIL)
endmacro()

macro(CopyGLEW)
endmacro()

macro(CopyGLUT)
endmacro()
