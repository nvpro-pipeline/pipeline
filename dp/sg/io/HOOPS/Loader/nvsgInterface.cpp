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


//
// nvsgInterface.cpp
// interface to nvsg dll loading code
//

#include <dp/sg/core/nvsgapi.h>
#include <dp/sg/io/PlugInterface.h> // definition of UPITID_VERSION,
#include <dp/sg/io/PlugInterfaceID.h> // definition of UPITID_VERSION,
                                  // UPITID_SCENE_LOADER, and
                                  // UPITID_SCENE_SAVER
#include "HOOPSLoader.h"

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

static char * cadFormats[] =
{
  ".3dxml",       // Dassault Systems: CATIA V6
  ".arc",         // Siemens: I-deas
  ".asm",         // Siemens: Solid Edge, PTC
  ".catpart",     // Dassault Systems: CATIA V5
  ".catproduct",  // Dassault Systems: CATIA V5
  ".catshape",    // Dassault Systems: CATIA V5
  ".cgr",         // Dassault Systems: CATIA Graphics Representation
  ".dlv",         // Dassault Systems: CATIA V4
  ".exp",         // Dassault Systems: CATIA V4
  ".iam",         // Autodesk: Inventor
  ".ifc",         // Industry Foundation Classes
  ".iges",        // IGES
  ".igs",         // IGES
  ".ipt",         // Autodesk: Inventor
  ".jt",          // Siemens: JT
  ".mf1",         // Siemens: I-deas
  ".model",       // Dassault Systems: CATIA V4
  ".neu",         // PTC
  ".par",         // Siemens: Solid Edge
  ".pkg",         // Siemens: I-deas
  ".prc",         // PRC
  ".prt",         // PTC, Siemens: NX
  ".psm",         // Siemens: Solid Edge
  ".pwd",         // Siemens: Solid Edge
  ".session",     // Dassault Systems: CATIA V4
  ".sldasm",      // Dassault Systems: SolidWorks
  ".sldprt",      // Dassault Systems: SolidWorks
  ".step",        // STEP
  ".stl",         // Stereo Lithography
  ".stp",         // STEP
  ".vda"          // VDA-FS
  ".vrml",        // VRML V1.0 and V2.0
//  ".wrl",         // VRML V1.0 and V2.0
  ".x_b",         // Siemens: Parasolid
  ".x_t",         // Siemens: Parasolid
  ".xas",         // PTC
  ".xmt",         // Siemens: Parasolid
  ".xmt_txt",     // Siemens: Parasolid
  ".xpr",         // PTC
};
#define NUM_CAD_FORMATS (sizeof( cadFormats ) / sizeof( char * ))

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  for ( unsigned int i=0 ; i<NUM_CAD_FORMATS ; i++ )
  {
    piids.push_back(dp::util::UPIID(cadFormats[i], PITID_SCENE_LOADER));
  }
}

bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugIn *& pi)
{
  for( unsigned int i = 0; i < NUM_CAD_FORMATS; i ++ )
  {
    if ( piid == dp::util::UPIID(cadFormats[i], PITID_SCENE_LOADER) )
    {
      pi = new HOOPSLoader();
      return( true );
    }
  }

  return false;
}
