project(DPUtil)

find_package(Boost COMPONENTS filesystem system REQUIRED )
find_package(DevIL REQUIRED)

include( CopyFile )

include_directories( ${Boost_INCLUDE_DIRS}
                     ${IL_INCLUDE_DIR}
                     ${NVPMAPI_INCLUDE_DIR} 
                   )

add_definitions(
  "-DDP_UTIL_EXPORTS"
  ${GLEW_DEFINITIONS}
)

set(DPUTIL_INCLUDES
  Allocator.h
  Atom.h
  Backtrace.h
  BitArray.h
  BitMask.h
  Config.h
  DPAssert.h
  DynamicLibrary.h
  File.h
  FrameProfiler.h
  HashGenerator.h
  HashGeneratorMurMur.h
  HashGeneratorMD5.h
  Image.h
  Lock.h
  Memory.h
  NVPerfMon.h
  Observer.h
  PlugIn.h
  PlugInCallback.h
  Profile.h
  RCObject.h
  Reflection.h
  Semantic.h
  SharedPtr.h
  SimplexNoise1234.h
  Singleton.h
  SmartPtr.h
  StridedIterator.h
  Timer.h
  Tokenizer.h
  Tools.h
  Trace.h
  Types.h
)

SET( DPUTIL_INCLUDES ${DPUTIL_INCLUDES} Array.h )
add_definitions("-DDP_UTIL_USE_BOOST")

source_group(headers FILES ${DPUTIL_INCLUDES})

set(DPUTIL_UI_INCLUDES
  ui/RenderTarget.h
)

source_group(headers/ui FILES ${DPUTIL_UI_INCLUDES})

set(DPUTIL_GENERATOR_INCLUDES
  generator/Geometry.h
  generator/Textures.h
)

source_group(headers/generator FILES ${DPUTIL_GENERATOR_INCLUDES})

set(DPUTIL_SOURCES
  src/Allocator.cpp
  src/Backtrace.cpp
  src/BitArray.cpp
  src/DPAssert.cpp
  src/DynamicLibrary.cpp
  src/File.cpp
  src/FileMapping.cpp
  src/FrameProfiler.cpp
  src/HashGeneratorMurMur.cpp
  src/HashGeneratorMD5.cpp
  src/Image.cpp
  src/Memory.cpp
  src/NVPerfMon.cpp
  src/Observer.cpp
  src/PlugIn.cpp
  src/Profile.cpp
  src/RCObject.cpp
  src/Reflection.cpp
  src/SimplexNoise1234.cpp
  src/Timer.cpp
  src/Tokenizer.cpp
  src/Tools.cpp
)

source_group(sources FILES ${DPUTIL_SOURCES})


set(DPUTIL_UI_SOURCES
  ui/src/RenderTarget.cpp
)

source_group(sources/ui FILES ${DPUTIL_UI_SOURCES})


set(DPUTIL_GENERATOR_SOURCES
  generator/src/Geometry.cpp
  generator/src/Textures.cpp
)

source_group(sources/generator FILES ${DPUTIL_GENERATOR_SOURCES})


add_library(DPUtil SHARED
  ${DPUTIL_INCLUDES}
  ${DPUTIL_UI_INCLUDES}
  ${DPUTIL_GENERATOR_INCLUDES}
  ${DPUTIL_SOURCES}
  ${DPUTIL_UI_SOURCES}
  ${DPUTIL_GENERATOR_SOURCES}
)

CopyDevIL( DPUtil "${DP_BINARY_PATH}" )
CopyGLEW( DPUtil "${DP_BINARY_PATH}" )

target_link_libraries( DPUtil
    ${Boost_LIBRARIES}
    ${IL_LIBRARIES}
)

if (MSVC)
  target_link_libraries( DPUtil
    DbgHelp
  )
endif()

if (WIN32 AND CMAKE_COMPILER_IS_GNUCC)
  # GCC requires the declspec(export) keyword behind the class keyword to export the vtable ( class declspec(export) MyKlass {} ). 
  # Export everything for now and adopt sources later.
  set_target_properties( DPUtil PROPERTIES LINK_FLAGS "-Wl,--export-all" )
endif()

set_target_properties( DPUtil PROPERTIES FOLDER "Util" )
