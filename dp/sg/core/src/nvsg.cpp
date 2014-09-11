// Copyright NVIDIA Corporation 2002-2012
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


// nvsg.cpp : Defines the entry point for the DLL application.
//

#if defined(DP_OS_WINDOWS)
#include <windows.h>
#endif

#include <dp/sg/core/nvsg.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/util/DPAssert.h>
#include <dp/util/Allocator.h>
#include <dp/util/Reflection.h>
#include <dp/util/Allocator.h>
#include <dp/util/PlugIn.h>
#include <dp/util/File.h>
#include <string>
#if defined(MSVC)
#include <crtdbg.h>
#endif
#if defined(LINUX)
#include <pthread.h>
#endif

// #pragma section is not supported by the GCC compiler

// with the ms compilers we use this to take over control 
// of when static objects are constructed and destructed
#if defined(_WIN32)
# define PRAGMA_INIT_SEG_SUPPORTED
#endif

int apiUsageCount = 0; // nvsg API reference count
int g_numGPUs = 0; // number of GPUs to use for distributed rendering
dp::sg::core::NVSG3DAPIEnum g_3DAPI = dp::sg::core::NVSG_3DAPI_UNINITIALIZED; // the rendering API that was initialized
// platform dependent definition of critical section/mutex
#if defined(_WIN32)
CRITICAL_SECTION cs;
#elif defined(LINUX)
pthread_mutex_t cs;
#else
// throw a compile error 
# error Unsupported Operating System!
#endif

// convenient platform independent helpers
//
void enterCritSec()
{
#if defined(_WIN32)
  EnterCriticalSection(&cs);
#elif defined(LINUX)
  pthread_mutex_lock(&cs);
#endif
}

void leaveCritSec()
{
#if defined(_WIN32)
  LeaveCriticalSection(&cs);
#elif defined(LINUX)
  pthread_mutex_unlock(&cs);
#endif
}

#if defined(PRAGMA_INIT_SEG_SUPPORTED)

// Note: If nvsg.dll gets linked against a mfc application the debug mfcXX.dll dumps pseudo memory leaks
// when ever it terminates before the nvsg.dll terminates. For this to avoid, we put our global/static
// data in our own data segment .nvsg, to manually control initialization and termination of that data.

// See MSDN documentation about '#pragma init_seg' for more details. 

#pragma warning(disable : 4075) // initializers put in unrecognized initialization area
typedef void (__cdecl * PFINITTERM)(void); // function pointer type for ctors/dtors 
int cpf = 0; //number of destructors we need to call at exit
PFINITTERM pfinitterm[64];  //ptrs to those dtors

// this gets called from the framework for every dtor that we need to call at exit 
int nvsgexit (PFINITTERM pf) 
{
  DP_ASSERT(cpf<64); // watch out for overflow!
  pfinitterm[cpf++] = pf; // register pointer to dtor
  return 0;
}
// declaration of our .nvsg data segment

// Note: The order here is important. Section names must be 8 characters or less.
// The sections with the same name before the '$' are merged into one section. 
// The order that they are merged is determined by sorting the characters after the '$'.
// nvsgStart and nvsgEnd are used to set boundaries so we can find the real functions 
// that we need to call for initialization.
# if _MSC_VER >= 1400 
#  pragma section(".nvsg$a", read)
#  pragma section(".nvsg$z", read)
# else
#  pragma section(".nvsg$a", read, write)
#  pragma section(".nvsg$z", read, write)
# endif
__declspec(allocate(".nvsg$a"))
PFINITTERM nvsgStart = (PFINITTERM)1;
__declspec(allocate(".nvsg$z"))
PFINITTERM nvsgEnd = (PFINITTERM)1;
# pragma data_seg() // reset data segment to .data
#endif // PRAGMA_INIT_SEG_SUPPORTED

namespace dp
{
  namespace sg
  {
    namespace core
    {


      #if !defined(NDEBUG)
      unsigned int nvsgDebugFlags = NVSG_DBG_ASSERT;
      #endif

      void nvsgInitialize(NVSG3DAPIEnum graphicsAPI, int num_gpus)
      {
        enterCritSec();
        if ( !apiUsageCount++ )
        {
      #if defined( PRAGMA_INIT_SEG_SUPPORTED )
          PFINITTERM *x = &nvsgStart;
          for (++x; x<&nvsgEnd; ++x) 
          {
            // The comparison for 0 is important. For now, each section is 256 bytes. 
            // When they are merged, they are padded with zeros. You can't depend on the 
            // section being 256 bytes, but you can depend on it being padded with zeros.
            if (*x)
            {
              (*x)();
            }
          }
      #endif

      #if !defined(X86_64) || defined(NVSG_DISABLE_CG_MULTIPLEX)  
          // No distributed rendering on 32 bit systems
          num_gpus = 1;
      #endif

          // Multiple calls to nvsgInitialize must specify the same number of GPUs
          DP_ASSERT(0 == g_numGPUs || num_gpus == g_numGPUs);
          g_numGPUs = num_gpus;

          // set 3dapi
          // Since we currently only have one usable API, this is not a problem, but in the
          // future should we assert that this is always being initialized to the same API?
          g_3DAPI = graphicsAPI;
        }

        leaveCritSec();

        // do other NVSG-specific initialization now...

        // the search path for loader DLLs
#if defined(_WIN32)
        std::string appPath = dp::util::getModulePath( GetModuleHandle( NULL ) );
#elif defined(LINUX)
        std::string appPath = dp::util::getModulePath();
#endif
        // TextureHost::createFromFile() relies on this to be set correctly!
        dp::util::addPlugInSearchPath( appPath );

        // load standard effects required for the scenegraph
        bool success = true;

        success &= dp::fx::EffectLibrary::instance()->loadEffects( "standard_lights.xml" );
        success &= dp::fx::EffectLibrary::instance()->loadEffects( "standard_material.xml" );
        success &= dp::fx::EffectLibrary::instance()->loadEffects( "collada.xml" );
        DP_ASSERT(success && "EffectLibrary::loadLibrary failed.");
      }

      void nvsgTerminate()
      {
        enterCritSec();
        if ( !--apiUsageCount )
        {
      #if defined( PRAGMA_INIT_SEG_SUPPORTED )
          while (cpf>0) 
          { // call dtors registered at init
            (pfinitterm[--cpf])();
          }
      #endif
        }

        leaveCritSec();
      }

      int getNumGPUs()
      {
        return g_numGPUs;
      }

      NVSG3DAPIEnum get3DAPI()
      {
        return g_3DAPI;
      }

      #if !defined(NDEBUG)
      void nvsgSetDebugFlags(unsigned int flags)
      {
        nvsgDebugFlags = flags;
        if ( nvsgDebugFlags & NVSG_DBG_LEAK_DETECTION )
        {
          dp::util::Singleton<dp::util::Allocator>::instance()->enableLeakDetection();
        }
      }

      unsigned int nvsgGetDebugFlags()
      {
        return nvsgDebugFlags;
      }

      bool nvsgRuntimeDebugControl( RuntimeDebugOp op )
      {
        DP_ASSERT( apiUsageCount && "nvsgRuntimeDebugControl can only be used after initializing SceniX" );
        DP_ASSERT( (nvsgDebugFlags & NVSG_DBG_LEAK_DETECTION) && "nvsgRuntimeDebugControl will have no effect if NVSG_DBG_LEAK_DETECTION is not enabled." );

        switch( op )
        {
          case NVSG_RUNTIME_DEBUG_POP_ALLOCATIONS:
            dp::util::Singleton<dp::util::Allocator>::instance()->popAllocations();
            break;

          case NVSG_RUNTIME_DEBUG_PUSH_ALLOCATIONS:
            dp::util::Singleton<dp::util::Allocator>::instance()->pushAllocations();
            break;

          case NVSG_RUNTIME_DEBUG_DUMP_ALLOCATIONS:
            dp::util::Singleton<dp::util::Allocator>::instance()->dumpAllocations();
            break;

          case NVSG_RUNTIME_DEBUG_DUMP_ALLOCATION_DIFFERENCES:
            dp::util::Singleton<dp::util::Allocator>::instance()->dumpDiffAllocations();
            break;

          default:
            return false;
        }

        return true;
      }
      #endif

    } // namespace core
  } // namespace sg
} // namespace dp

#if defined( PRAGMA_INIT_SEG_SUPPORTED )
// start our initialization area now
# pragma init_seg(".nvsg$m", nvsgexit)
#endif 

// for now on, we must call the ctors/dtors by ourselves, the c-runtime won't!

// globals/statics:

// note: According to "The C++ Programming Language" template class static members
//       have to be defined as below and then may be specialized with "template<> ..."
//       As this does not work with the gnu compiler we use the explicit instantiation 
//       through 'template ..." as a workaround.

#if defined(__GNUC__)
# define TEMPLATE template
#else
# define TEMPLATE template<>
#endif

// explicit instantiations follow:

// Allocator singleton object instantiation
TEMPLATE dp::util::Allocator dp::util::Singleton<dp::util::Allocator>::m_instance;

// PlugInServer singleton object instantiationn
TEMPLATE dp::util::PlugInServer dp::util::Singleton<dp::util::PlugInServer>::m_instance;

// global namespace's globals/statics
#if defined(_WIN32)
std::string NVSG_LOG_FILE(".\nvsg.log");
#elif defined(LINUX)
std::string NVSG_LOG_FILE("./nvsg.log");
#endif

#undef TEMPLATE

// win32 specific
#if defined(_WIN32)

extern "C" DP_SG_CORE_API 
// microsoft specific Dll entry point
BOOL APIENTRY DllMain( HANDLE hModule, 
                        DWORD  ul_reason_for_call, 
                        LPVOID lpReserved
                      )
{
  switch (ul_reason_for_call)
  {
  case DLL_PROCESS_ATTACH:
    InitializeCriticalSection(&cs);
#if defined(MSVC)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#if !defined(NDEBUG)
    _crtBreakAlloc = 0;
#endif
#endif

  // NOTE:
  // A thread calling the DLL entry-point function with DLL_PROCESS_ATTACH 
  // does not call the DLL entry-point function with DLL_THREAD_ATTACH.
  // For this reason we omit the break here!
  case DLL_THREAD_ATTACH:
    break;
  case DLL_THREAD_DETACH:
    break;
  case DLL_PROCESS_DETACH:
    DeleteCriticalSection(&cs);
    break;
  }
    return TRUE;
}

#elif defined(DP_OS_LINUX)
void nvsgLibInit()
{
  // Intialize Xlib for multithreaded access

  // TODO what's this mutex being used for?

  // we need a recursive mutex; a fast mutex (default) will deadlock
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
  pthread_mutex_init(&cs, &attr);
  pthread_mutexattr_destroy(&attr);
}

void nvsgLibExit()
{
  // release the mutex when the library quits
  pthread_mutex_destroy(&cs);
}
#endif
