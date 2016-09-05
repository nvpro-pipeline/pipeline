if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/jsoncpp)
  # configure jsoncpp
  option(JSONCPP_WITH_TESTS "" OFF)
  option(JSONCPP_WITH_POST_BUILD_UNITTEST "" OFF)
  option(JSONCPP_WITH_PKGCONFIG_SUPPORT "" OFF)
  add_subdirectory(jsoncpp)
  set_target_properties( jsoncpp_lib_static PROPERTIES FOLDER "3rdparty")
endif()
