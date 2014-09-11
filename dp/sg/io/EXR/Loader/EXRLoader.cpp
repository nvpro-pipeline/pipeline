// Copyright NVIDIA Corporation 2002-2004
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


// EXRLoader.cpp : Defines the entry point for the DLL application.
//

#include "stdafx.h"

#include "EXRLoader.h"

// NVSG includes
#include <dp/sg/core/TextureHost.h>
#include <dp/util/File.h>

// OpenEXR includes
#include "ImfRgbaFile.h"
#include "ImfRgba.h"
#include "ImfArray.h"
#include "ImathBox.h"
#include "ImfArray.h"
#include "half.h"

#include <vector>
#include <string>

using namespace dp::sg::core;

using std::vector;
using std::string;

using namespace Imf;
using namespace Imath;

// supported Plug Interface ID
const dp::util::UPITID PITID_TEXTURE_LOADER(UPITID_TEXTURE_LOADER, UPITID_VERSION); // plug-in type
const dp::util::UPIID  PIID_EXR_TEXTURE_LOADER(".EXR", PITID_TEXTURE_LOADER);       // a plug-in interface ID


// convenient macro
#define INVOKE_CALLBACK(cb) if ( callback() ) callback()->cb

#if defined(_WIN32)
BOOL APIENTRY DllMain( HANDLE hModule, 
                      DWORD  ul_reason_for_call, 
                      LPVOID lpReserved
                      )
{
  return TRUE;
}
#endif

#if defined(LINUX)
extern "C"
{
#endif

  EXRLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugIn *& pi)
  {
    if( piid==PIID_EXR_TEXTURE_LOADER )
    {      
      pi = new EXRLoader();
      return pi!=NULL;
    }

    return false;
  }

  EXRLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
  {
    piids.clear();

    piids.push_back(PIID_EXR_TEXTURE_LOADER);
  }

#if defined(LINUX)
}
#endif


EXRLoader::EXRLoader()
{
}

EXRLoader::~EXRLoader()
{
}

bool EXRLoader::onLoad( dp::sg::core::TextureHostSharedPtr const& texImg
  , const std::vector<std::string> & searchPaths )
{
  const std::string& filename = texImg->getFileName();
  DP_ASSERT( dp::util::fileExists( filename ) );

  int width, height, depth;

  Array2D<Rgba> pixels;
  
  RgbaInputFile file(filename.c_str());
  Box2i dw = file.dataWindow();

  width = dw.max.x - dw.min.x + 1;
  height = dw.max.y - dw.min.y + 1;
  depth = 1;

  pixels.resizeErase( height, width );

  file.setFrameBuffer( &pixels[0][0] - dw.min.x - dw.min.y * width, 1, width );
  file.readPixels( dw.min.y, dw.max.y );

  // now create a TextureHost for use with NVSG...
  size_t img = texImg->createImage( width, height, depth, Image::IMG_RGBA, Image::IMG_HALF
                                  , &pixels[0][0] - dw.min.x - dw.min.y * width);

  return( img != -1 );
}
