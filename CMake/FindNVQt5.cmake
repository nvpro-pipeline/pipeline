if (WIN32)

  # Search for Windows SDK so that Qt5 can find glu32.lib
  get_filename_component(WINSDK70_DIR  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows\\v7.0A;InstallationFolder]" ABSOLUTE CACHE)

  if(EXISTS ${WINSDK70_DIR})
    message(STATUS "Found Windows SDK 7.0A in ${WINSDK70_DIR}")
  else()
    message( STATUS "Windows SDK 7.0A not installed. Trying nv tools dir..." )

    if ( "$ENV{NV_TOOLS}" STREQUAL "" )
      message( FATAL_ERROR "NV_TOOLS environment variable not set!" )
    endif()

    SET(WINSDK70_DIR "$ENV{NV_TOOLS}/sdk/WinSDK/7.0a")

    if(EXISTS ${WINSDK70_DIR})
      message(STATUS "Found Windows SDK 7.0A in ${WINSDK70_DIR}")
    else()
      message(WARNING "Missing Windows SDK 7.0A. Should be in ${WINSDK70_DIR}")
    endif()
  endif()

  set (QTVERSION "5.1.0")
  if(MSVC10)
    set (ENV{QMAKESPEC} "win32-msvc2010")
    set (COMPILER "win32-msvc2010")
  elseif(MSVC11)
    set (ENV{QMAKESPEC} "win32-msvc2012")
    set (COMPILER "win32-msvc2012")
  elseif(CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    string(STRIP "${GCC_VERSION}" GCC_VERSION)
    set (ENV{QMAKESPEC} "win32-g++")
    set (COMPILER "win32-gcc-${GCC_VERSION}-mingw")
  else()
    message(FATAL_ERROR "Compiler version not supported")
  endif()

  file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/Qt/${QTVERSION}/${COMPILER}-${DP_ARCH}" QTPATH )
  
  if ( ${DP_ARCH} STREQUAL "amd64" )
    set( WINSDK_LIB "${WINSDK70_DIR}/lib/x64" )
  else()
    set( WINSDK_LIB "${WINSDK70_DIR}/lib" )
  endif()
  
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${WINSDK_LIB}")
  set(Qt5Core_DIR "${QTPATH}/lib/cmake/Qt5Core")

  if ( EXISTS "${QTPATH}" )
    file ( TO_NATIVE_PATH ${QTPATH} QTPATH )
    set( ENV{QTDIR} "${QTPATH}" )
    
    if ( NOT CMAKE_COMPILER_IS_GNUCC )
      file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/Qt/${QTVERSION}/$ENV{QMAKESPEC}-amd64" TMP)
      get_filename_component( QT32PATH ${TMP} ABSOLUTE )
      
      set( QT_QMAKE_EXECUTABLE "${QT32PATH}/bin/qmake.exe" CACHE FILEPATH "" )
      set( QT_UIC_EXECUTABLE "${QT32PATH}/bin/uic.exe" CACHE FILEPATH "" )
      set( QT_MOC_EXECUTABLE "${QT32PATH}/bin/moc.exe" CACHE FILEPATH "" )
      set( QT_RCC_EXECUTABLE "${QT32PATH}/bin/rcc.exe" CACHE FILEPATH "" )
    endif()
  endif()
  
endif()

if (UNIX)
  set( QTVERSION "4.7.1" )
  set( ENV{WS_PROJECT_COMPILER} "gcc-4.1" )
  set( ENV{WS_PROJECT_ARCH} ${DP_ARCH} ) 

  file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/Qt/${QTVERSION}" QTDIR)
  if ( EXISTS "${QTDIR}" )
    file ( TO_NATIVE_PATH ${QTDIR} QTDIR )
    set( ENV{QTDIR} "${QTDIR}" )
    set( QT_QMAKE_EXECUTABLE "${QTDIR}/bin/$ENV{WS_PROJECT_COMPILER}-$ENV{WS_PROJECT_ARCH}/qmake" CACHE FILEPATH "" )
    set( QT_UIC_EXECUTABLE "${QTDIR}/bin/$ENV{WS_PROJECT_COMPILER}-$ENV{WS_PROJECT_ARCH}/uic" CACHE FILEPATH "" )
    set( QT_MOC_EXECUTABLE "${QTDIR}/bin/$ENV{WS_PROJECT_COMPILER}-$ENV{WS_PROJECT_ARCH}/moc" CACHE FILEPATH "" )
    set( QT_RCC_EXECUTABLE "${QTDIR}/bin/$ENV{WS_PROJECT_COMPILER}-$ENV{WS_PROJECT_ARCH}/rcc" CACHE FILEPATH "" )
  endif()
endif()
