include("dp")

file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}" DP_3RDPARTY_PATH)

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
  list(APPEND CMAKE_LIBRARY_PATH "${DP_3RDPARTY_PATH}/freeglut/lib/x64")

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

  if (MSVC11)
    set(QtCompiler "msvc2012")
  elseif(MSVC12)
    set(QtCompiler "msvc2013")
  endif()

  if (${DP_ARCH} STREQUAL "amd64")
    set(QtArch "_64")
  endif()
    
  if (QtCompiler)
    set(QtRegistryKeys
     "[HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Qt;InstallLocation]"
     "[HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{0bdb0613-2f59-49c2-93b6-3016f4b1a43b};InstallLocation]"
    )
    
    foreach(QtRegistryKey ${QtRegistryKeys})
      get_filename_component(QtRoot ${QtRegistryKey} REALPATH)
      if (EXISTS ${QtRoot})
        message("Qt installation: ${QtRoot}")
        
        set(QtVersions 5.7 5.6 5.5 5.4 5.3 5.2.1)
        foreach(QtVersion ${QtVersions})
          set(QtPath "${QtRoot}/${QtVersion}/${QtCompiler}${QtArch}/lib/cmake")
          if (EXISTS ${QtPath})
            message("Using Qt ${QtVersion}: ${QtRoot}/${QtVersion}/${QtCompiler}${QtArch}")
            list(APPEND CMAKE_PREFIX_PATH ${QtPath})
            set(QtFound TRUE)
            break()
          endif()
          if (QtFound)
            break()
          endif()
        endforeach()
      endif()
    endforeach()
  endif()

endif()

macro(CopyDevIL)
endmacro()

macro(CopyGLEW)
endmacro()

macro(CopyGLUT)
endmacro()
