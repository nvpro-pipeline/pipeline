if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/jsoncpp)
  # configure jsoncpp
  option(JSONCPP_WITH_TESTS "" OFF)
  option(JSONCPP_WITH_POST_BUILD_UNITTEST "" OFF)
  option(JSONCPP_WITH_PKGCONFIG_SUPPORT "" OFF)
  add_subdirectory(jsoncpp)
endif()
