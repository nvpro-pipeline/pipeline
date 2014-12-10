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


//
// nvsgInterface.cpp
// interface to nvsg dll loading code
//

#include <dp/sg/core/nvsgapi.h>
#include <dp/sg/io/PlugInterface.h> // definition of UPITID_VERSION,
#include <dp/sg/io/PlugInterfaceID.h> // definition of UPITID_VERSION,
                                  // UPITID_SCENE_LOADER, and
                                  // UPITID_SCENE_SAVER
#include "3DSLoader.h"

#if defined(_WIN32)
BOOL APIENTRY DllMain(HANDLE hModule, DWORD reason, LPVOID lpReserved)
{
  if (reason == DLL_PROCESS_ATTACH)
  {
    int i=0;
  }

  return TRUE;
}
#elif defined(LINUX)
void lib_init()
{
  int i=0;
}
#endif


// unique plug-in types
const dp::util::UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION);

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(dp::util::UPIID(".3DS", PITID_SCENE_LOADER));
}

bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  const dp::util::UPIID PIID_3DS_SCENE_LOADER = dp::util::UPIID(".3DS", PITID_SCENE_LOADER);

  if ( piid == PIID_3DS_SCENE_LOADER )
  {
    pi = ThreeDSLoader::create();
    return( !!pi );
  }

  return false;
}
