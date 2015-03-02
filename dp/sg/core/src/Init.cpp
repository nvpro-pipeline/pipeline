// Copyright NVIDIA Corporation 2002-2015
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


#include <dp/fx/EffectLibrary.h>
#include <dp/util/PlugIn.h>
#include <dp/util/File.h>
#include <string>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      static bool initialize()
      {
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

        return success;
      }

      bool initialized = initialize();

    } // namespace core
  } // namespace sg
} // namespace dp

