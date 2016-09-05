if(WIN32)
    #includes
    include_directories(
      "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    )

    #definitions
    add_definitions(
      -DCSFSAVER_EXPORTS
      -D_CRT_SECURE_NO_WARNINGS
    )

    #sources
    set(CSFSAVER_SOURCES
      ExtractGeometryTraverser.cpp
      CSFSaver.cpp
      OffsetManager.cpp
    )

    set(CSFSAVER_HEADERS
      inc/CSFSaver.h
      inc/ExtractGeometryTraverser.h
      inc/OffsetManager.h
      inc/CSFSGWrapper.h
    )

    source_group(source FILES ${CSFSAVER_SOURCES})
    source_group(header FILES ${CSFSAVER_HEADERS})

    #target
    add_library( CSFSaver SHARED
      ${CSFSAVER_SOURCES}
      ${CSFSAVER_HEADERS}
    )

    target_link_libraries( CSFSaver
      DPSgCore
      DPMath
      DPUtil
      DPFx
      DPSgIO
    )

    set_target_properties( CSFSaver PROPERTIES SUFFIX ".nxm" FOLDER "DP/SG/IO" )
endif()
