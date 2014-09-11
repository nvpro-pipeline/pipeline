// Copyright NVIDIA Corporation 2002-2005
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#pragma once
/** @file */

#include <dp/sg/core/nvsgapi.h>
#include <dp/util/BitMask.h>

#include <string>
#include <cstring>
#include <vector>

#if defined(_WIN32) && _MSC_VER >= 1500
#  if _BIND_TO_CURRENT_CRT_VERSION >= 1 || _BIND_TO_CURRENT_MFC_VERSION >= 1 || _BIND_TO_CURRENT_VCLIBS_VERSION >= 1 
#    pragma message ("WARNING: At least one of the macros  _BIND_TO_CURRENT_CRT_VERSION, _BIND_TO_CURRENT_VCLIBS_VERSION or BIND_TO_CURRENT_MFC_VERSION is defined. This can lead to runtime issues.")
#  endif
#endif

#if defined(LINUX)
// Linux specific library entry points
void nvsgLibInit() __attribute__((constructor)); 
void nvsgLibExit() __attribute__((destructor)); 
#endif

extern void enterCritSec();
extern void leaveCritSec();

//! Scene graph classes - core SceniX.
/** This namespace contains all core SceniX classes. 
  * Put all the scene graph and node related classes in here.*/ 
namespace dp
{
  namespace sg
  {
    namespace core
    {
  
    #if !defined(NDEBUG)
      enum
      {
        NVSG_DBG_ASSERT = BIT0, 
        NVSG_DBG_LEAK_DETECTION = BIT1
      };

      enum RuntimeDebugOp
      {
        NVSG_RUNTIME_DEBUG_POP_ALLOCATIONS,
        NVSG_RUNTIME_DEBUG_PUSH_ALLOCATIONS,
        NVSG_RUNTIME_DEBUG_DUMP_ALLOCATIONS,
        NVSG_RUNTIME_DEBUG_DUMP_ALLOCATION_DIFFERENCES
      };

      /*! \brief Lets the user control the debug runtime behavior
       * \param dbgFlags 
       * Flags to control the debug runtime behavior. 
       * Allowed flags are NVSG_DBG_ASSERT, NVSG_DBG_LEAK_DETECTION, or a bit combination of these.
       * \remarks
       * This entry point lets the user control SceniX's runtime behavior when running in debug mode.
       * It is only exported from the debug version of the SceniX shared library.\n
       * The \a dbgFlags you pass to this function control the runtime behavior.
       * If you omit the NVSG_DBG_ASSERT flag when you call this function, assertion failure reports 
       * will be suppressed. In general it is not recommended to suppress assertion failure reports, since
       * these are valuable means to detect possible errors as you develop your SceniX-based application.
       * By default, assertion failure reports are enabled. \n
       * Specify NVSG_DBG_LEAK_DETECTION to enable SceniX's memory leak tracking. This debug feature is off
       * by default, since memory leak detection has a huge impact on debug runtime performance. Dependent
       * on the data set you process with SceniX, enabled leak detection can degrade the debug runtime
       * performance by an order of magnitude. If leak detection is enabled, object allocation and
       * deallocation is tracked, and detected memory leaks will be reported to an attached debugger
       * when the application shuts down.\n
       * The nvsgSetDebugFlags can be used optionally to modify the default debug runtime behavior.
       * If you omit to call this function, assertion failure reports are enabled, and leak detection
       * is disabled. If you decide to call this function, it is recommended to do this immediately
       * prior to the call to nvsgInitialize, which is required to call prior to any other SceniX
       * entry point. Though, this is not mandatory. \n
       * When you call this function, previously specified debug flags will be overridden by the new flags
       * specified. If you intend to keep previously specified debug flags, you should use the
       * return value of nvsgGetDebugFlags, bitwise OR it with the flags you want additionally, and
       * pass the result to nvsgSetDebugFlags. 
       * \sa nvsgGetDebugFlags */
      DP_SG_CORE_API void nvsgSetDebugFlags(unsigned int dbgFlags);

      /*! \brief Retrieves the currently specified debug flags
       * \return 
       * The function returns the currently specified debug flags that are used to control the debug
       * runtime behavior. 
       * \remarks
       * When you call nvsgSetDebugFlags to modify the debug runtime behavior, previously specified 
       * debug flags will be overridden by the new flags you specify. If you intend to keep previously 
       * specified debug flags, you should use the return value of nvsgGetDebugFlags, bitwise OR it with 
       * the flags you want additionally, and pass the result to nvsgSetDebugFlags as shown in this
       * example:
       * \code
       *   // additionally enable leak detection, and leave current control flags unchanged
       *   unsigned int newDbgFlags = nvsgGetDebugFlags() | NVSG_DBG_LEAK_DETECTION;
       *   nvsgSetDebugFlags(newDbgFlags);
       * \endcode
       * \sa nvsgSetDebugFlags */
      DP_SG_CORE_API unsigned int nvsgGetDebugFlags();

      /*! \brief Execute certain runtime debug operations.
       *  \remarks add docs here when approved
       * \sa nvsgGetDebugFlags, nvsgSetDebugFlags */
      DP_SG_CORE_API bool nvsgRuntimeDebugControl( RuntimeDebugOp op );

    #else
    # define nvsgRuntimeDebugControl(op )
    # define nvsgSetDebugFlags(dbgFlags)
    # define nvsgGetDebugFlags() 0
    #endif

      enum NVSG3DAPIEnum
      {
        NVSG_3DAPI_UNINITIALIZED = 0, 
        NVSG_3DAPI_GL = 1
      };

      enum NVSGFeatureSupportedEnum
      {
        NVSG_FAST_TEXTURE_UPLOAD  = BIT0
      };

      //! Per-application initialization of SceniX global data
      /** \param graphicsAPI
        * Specifies the 3D graphics API to use for rendering.\n 
        * Specifying NVSG_3DAPI_GL to use OpenGL as 3D API currently is the only choice.
        * This is subject to change with future releases.
        * \param num_gpus
        * Specifies the number of GPUs to use for distributed rendering. Distributed
        * rendering is a feature that is available in 64 bit versions of SceniX. The
        * default value in 64 bit versions is 1 and this parameter is ignored on 32
        * bit versions.
        * \remarks
        * An application must call this routine once before any other SceniX API call. 
        * This routine properly initializes all global data required by the SceniX API.
        * \n\n
        * The nvsgInitialize entry point in addition lets you specify which 3D graphics API is
        * used for rendering. Note that using another 3D API for rendering than specified here
        * will result in undefined run-time behaviour. 
        * \n\n
        * It is recommended to call this routine inside the application's initial entry point. 
        * \note The nvsg runtime behaviour is undefined if an application issues nvsg API calls
        * without a prior call to this routine. */
      DP_SG_CORE_API void nvsgInitialize( NVSG3DAPIEnum graphicsAPI = NVSG_3DAPI_GL
                                  , int num_gpus = 1 );

      //! Per-application termination of nvsg global data
      /** It is strongly recommended to call this routine when the application terminates. 
        * This routine properly frees all global data required by the nvsg API.
        * \note The nvsg runtime behaviour is undefined if an application issues nvsg API calls
        * after having called this routine. */
      DP_SG_CORE_API void nvsgTerminate();

      //! Get the version string of the SceniX
      DP_SG_CORE_API void getVersionString(
        std::string & string  //!<  string to hold the SceniX version string
      );

      //! Get the SceniX name string
      DP_SG_CORE_API void getSDKName(
        std::string & string  //!<  string to hold the SceniX name string
      );

      //! Get the SceniX copyright string
      DP_SG_CORE_API void getCopyrightString(
        std::string & string  //!<  string to hold the SceniX coppyright string
      );

      //! Get the SceniX vendor name
      DP_SG_CORE_API void getVendorName(
        std::string & string //!<  string to hold the SceniX vendor name string
      );

      /*! \brief Get a bit field of the supported features.
       *  \return An unsigned int encoding the supported features.
       *  \remarks Currently, there is one feature supported on Quadro boards only: fast texture
       *  upload (NVSG_FAST_TEXTURE_UPLOAD). */
      DP_SG_CORE_API unsigned int getSupportedFeatures();

      /*! \brief Get the number of GPUs to be use for distributed rendering in a
       *  MultiGPU system
       *  \return The number of GPUs that was specified while calling nvsgInitialize
       */
      DP_SG_CORE_API int getNumGPUs();
 
      /*! \brief Get the 3D API 
       *  \return The API that was used to initialize SceniX.
       */
      DP_SG_CORE_API NVSG3DAPIEnum get3DAPI();

    } // namespace core
  } // namespace sg
} // namespace dp
