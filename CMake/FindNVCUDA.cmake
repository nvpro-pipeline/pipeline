if (WIN32)
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set ( NV_CUDA_ARCH "amd64" CACHE STRING "CUDA library architecture" )
  else ()
    set ( NV_CUDA_ARCH "x86" CACHE STRING "CUDA library architecture" )
  endif()
  
  file( TO_CMAKE_PATH "$ENV{DP_3RDPARTY_PATH}/CUDA/${NVCUDA_FIND_VERSION}/${NV_CUDA_ARCH}" NV_CUDA_PATH_TMP )
  if ( EXISTS "${NV_CUDA_PATH_TMP}" )
    set( CUDA_TOOLKIT_ROOT_DIR "${NV_CUDA_PATH_TMP}" )
  endif()
endif()

if ( NVCUDA_FIND_QUIET )
  set( QUIET "QUIET" )
endif()

if ( NVCUDA_FIND_REQUIRED )
  set( REQUIRED "REQUIRED" )
endif()

find_package(CUDA ${NVCUDA_FIND_VERSION} ${QUIET} ${REQUIRED})

# Generate PTX files
# NVCUDA_COMPILE_PTX( SOURCES file1.cu file2.cu DEPENDENCIES header1.h header2.h TARGET_PATH <path where ptxs should be stored> GENERATED_FILES ptx_sources NVCC_OPTIONS -arch=sm_20)
# Generates ptx files for the given source files. ptx_sources will contain the list of generated files.
FUNCTION(NVCUDA_COMPILE_PTX)
  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED_FILES)
  set(multiValueArgs NVCC_OPTIONS SOURCES DEPENDENCIES)
  CMAKE_PARSE_ARGUMENTS(NVCUDA_COMPILE_PTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  # Match the bitness of the ptx to the bitness of the application
  set( MACHINE "--machine=32" )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( MACHINE "--machine=64" )
  endif()
  
  # Custom build rule to generate ptx files from cuda files
  FOREACH( input ${NVCUDA_COMPILE_PTX_SOURCES} )
    get_filename_component( input_we ${input} NAME_WE )
    get_filename_component( ABS_PATH ${input} ABSOLUTE )
    string( REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" REL_PATH "${ABS_PATH}" )
    # message("rel ${REL_PATH}")
    # generate the *.ptx files inside "ptx" folder inside the executable's output directory.
    set( output "${CMAKE_CURRENT_BINARY_DIR}/ptx/${REL_PATH}/${input_we}.ptx" )
    # message( "output: ${output}")

    LIST( APPEND PTX_FILES  ${output} )
    
    add_custom_command(
      OUTPUT  ${output}
      DEPENDS ${input} ${NVCUDA_COMPILE_PTX_DEPENDENCIES}
      COMMAND ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --ptx ${NVCUDA_COMPILE_PTX_NVCC_OPTIONS} ${input} -o ${output} WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy ${output} "${NVCUDA_COMPILE_PTX_TARGET_PATH}/${NVCUDA_COMPILE_PTX_PATH_SUFFIX}/${input_we}.ptx"
    )
  ENDFOREACH( )
  
  set(${NVCUDA_COMPILE_PTX_GENERATED_FILES} ${PTX_FILES} PARENT_SCOPE)
ENDFUNCTION()