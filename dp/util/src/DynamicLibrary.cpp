// Copyright NVIDIA Corporation 2011
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


#include <dp/util/DynamicLibrary.h>

//
// this file should be compatible with Windows, Linux, Mac
// 

#ifdef DP_OS_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace dp
{
  namespace util
  {

    DynamicLibrary::DynamicLibrary( void* dll )
    {
      m_dll = (void*)dll;
    }

    DynamicLibrary::~DynamicLibrary()
    {
#if defined( DP_OS_WINDOWS )
      FreeLibrary( (HMODULE)m_dll );
#else
      dlclose( m_dll );
#endif
    }

    void* DynamicLibrary::getSymbol( const char *name )
    {
#if defined(DP_OS_WINDOWS)
      return (void*)(GetProcAddress((HINSTANCE)m_dll, name));
#else
      m_dll = (m_dll == 0) ? RTLD_DEFAULT : m_dll;
      return dlsym(m_dll, name);
#endif
    }

    DynamicLibrarySharedPtr DynamicLibrary::createFromFile( const char* name )
    {
#if defined( DP_OS_WINDOWS )
      HINSTANCE dll = LoadLibrary(name);
#else
      void* dll = dlopen(name, RTLD_LAZY | RTLD_LOCAL );
#endif
      if(dll)
      {
        return( std::shared_ptr<DynamicLibrary>( new DynamicLibrary( (void*)dll ) ) );
      }
      else
      {
        return DynamicLibrarySharedPtr::null;
      }
    }

  } // namespace util 
} // namespace dp

