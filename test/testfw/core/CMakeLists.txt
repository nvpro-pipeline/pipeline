
find_package(NVPMAPI REQUIRED)

find_package( Boost COMPONENTS program_options REQUIRED )

#includes
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  ${NVPMAPI_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

#definitions
add_definitions(
  "-DDPTCORE_EXPORTS"
  "-DCONFIGURATION=${OS}-${DP_ARCH}-${CRT}-${CNF}"
  "-DRESDIR=${CMAKE_CURRENT_SOURCE_DIR}/../resources"
  "-D_CRT_SECURE_NO_WARNINGS"
)

if ( DPT_RENDERERAPI )
  add_definitions(
    "-DDPT_RENDERERAPI"
  )
endif()


#sources
set (DPTCORE_SOURCES
  src/Backend.cpp
  src/MeasurementFunctor.cpp
  src/MeasurementFunctorGoldImage.cpp
  src/MeasurementFunctorNVPM.cpp
  src/MeasurementFunctorTimer.cpp
  src/Test.cpp
  src/TestObject.cpp
  src/TestObjectUtil.cpp
  src/TestRender.cpp
)

set (DPTCORE_HEADERS
  inc/Config.h
)

set (DPTCORE_PUBLIC_HEADERS
  Backend.h
  MeasurementFunctor.h
  MeasurementFunctorGoldImage.h
  MeasurementFunctorNVPM.h
  MeasurementFunctorTimer.h
  Test.h
  TestObject.h
  TestObjectUtil.h
  TestRender.h
)

source_group(headers FILES ${DPTCORE_HEADERS})
source_group(sources FILES ${DPTCORE_SOURCES})
source_group("" FILES ${DPTCORE_PUBLIC_HEADERS})

#target
add_library( DPTcore SHARED
   ${DPTCORE_SOURCES}
   ${DPTCORE_HEADERS}
   ${DPTCORE_PUBLIC_HEADERS}
)

target_link_libraries( DPTcore
  ${Boost_LIBRARIES}
  DPUtil
)

add_dependencies( DPTcore DPUtil )

set_target_properties( DPTcore PROPERTIES FOLDER "test/testfw")
