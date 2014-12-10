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


// ILTexSaver.cpp : Defines the entry point for the DLL application.
//

#if defined(_WIN32)
# define WIN32_LEAN_AND_MEAN    // Exclude rarely-used stuff from Windows headers
# include <windows.h>
#endif
#include <typeinfo>

#include <dp/sg/core/NVSGVersion.h> // NVSG version info
#include <dp/sg/io/PlugInterfaceID.h>
#include <dp/sg/core/TextureHost.h>

#include <dp/util/DPAssert.h>
#include <dp/util/File.h>
#include <dp/util/Tools.h>

#include "ILTexSaver.h"
#include "il.h"

using namespace dp::sg::core;
using namespace dp::util;

using std::vector;
using std::string;

// a pointer to our single instance of the Loader
ILTexSaverSharedPtr ILTexSaver::m_instance;

// supported Plug Interface ID
const UPITID PITID_TEXTURE_SAVER(UPITID_TEXTURE_SAVER, UPITID_VERSION); // plug-in type
#define NUM_SUPPORTED_EXTENSIONS 22
static string SUPPORTED_EXTENSIONS[] = { ".TGA",
                                         ".JPG",
                                         ".JPEG",
                                         ".DDS",
                                         ".PNG",
                                         ".BMP",
                                         ".PCX",
                                         ".PBM",
                                         ".PGM",
                                         ".PPM",
                                         ".PSD",
                                         ".SGI",
                                         ".BW",
                                         ".RGB",
                                         ".RGBA",
                                         ".TIF",
                                         ".TIFF",
                                         ".JP2",
                                         ".PAL",
                                         ".VTF",
                                         ".RAW",
                                         ".HDR" };


// convenient macro
#ifndef  INVOKE_CALLBACK
  #define INVOKE_CALLBACK(cb) if ( callback() ) callback()->cb
#endif


#if defined(_WIN32)
BOOL APIENTRY DllMain( HANDLE hModule, 
                       DWORD  ul_reason_for_call, 
                       LPVOID lpReserved
          )
{
    return TRUE;
}
#endif

bool getPlugInterface(const UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  for( unsigned int i=0; i<NUM_SUPPORTED_EXTENSIONS; ++i)
  {
    UPIID tempPiid(SUPPORTED_EXTENSIONS[i].c_str(), PITID_TEXTURE_SAVER);
    if( piid==tempPiid )
    {
      if ( !ILTexSaver::m_instance )
      {
        ILTexSaver::m_instance = ILTexSaver::create();
      }
      pi = ILTexSaver::m_instance;
      return( !!pi );
    }
  }
  return false;
}

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  for( unsigned int i=0; i<NUM_SUPPORTED_EXTENSIONS; ++i)
  {
    // generate piids for all supported extensions dynamically
    UPIID tempPiid(SUPPORTED_EXTENSIONS[i].c_str(), PITID_TEXTURE_SAVER);
    piids.push_back(tempPiid);
  }
}

static int determineImage( int i, bool isDDS, bool isCube )
{
  int image = i;
  if ( isDDS )
  {
    if ( isCube )
    {
      if ( 4 == image )
      {
        image = 5;
      }
      else if ( 5 == image )
      {
        image = 4;
      }
    }
  }
  return( image );
}

static ILenum determineILFormat(Image::PixelFormat format)
{
  ILenum ilFormat = 0;
  switch( format )
  {
    case Image::IMG_COLOR_INDEX :
      ilFormat = IL_COLOR_INDEX;
      break;

    case Image::IMG_RGB :
    case Image::IMG_INTEGER_RGB :
      ilFormat = IL_RGB;
      break;

    case Image::IMG_RGBA :
    case Image::IMG_INTEGER_RGBA :
      ilFormat = IL_RGBA;
      break;

    case Image::IMG_BGR :
    case Image::IMG_INTEGER_BGR :
      ilFormat = IL_BGR;
      break;

    case Image::IMG_BGRA :
    case Image::IMG_INTEGER_BGRA :
      ilFormat = IL_BGRA;
      break;

    case Image::IMG_LUMINANCE :
    case Image::IMG_ALPHA :
    case Image::IMG_DEPTH_COMPONENT :
    case Image::IMG_INTEGER_ALPHA :
    case Image::IMG_INTEGER_LUMINANCE :
      ilFormat = IL_LUMINANCE;
      break;

    case Image::IMG_LUMINANCE_ALPHA :
    case Image::IMG_INTEGER_LUMINANCE_ALPHA :
      ilFormat = IL_LUMINANCE_ALPHA;
      break;

    default :
      ilFormat = 0;
      DP_ASSERT( !"Unknown pixel format" );
      break;
  }
  return ilFormat;
}

static ILenum determineILType(Image::PixelDataType type)
{
  ILenum ilType = 0;
  switch( type )
  {
    case Image::IMG_BYTE :  
      ilType = IL_BYTE;
      break;
    case Image::IMG_UNSIGNED_BYTE :
      ilType = IL_UNSIGNED_BYTE;
      break;
    case Image::IMG_SHORT :
      ilType = IL_SHORT;
      break;
    case Image::IMG_UNSIGNED_SHORT :
      ilType = IL_UNSIGNED_SHORT;
      break;
    case Image::IMG_INT :
      ilType = IL_INT;
      break;
    case Image::IMG_UNSIGNED_INT :
      ilType = IL_UNSIGNED_INT;
      break;
    case Image::IMG_FLOAT :
      ilType = IL_FLOAT;
      break;
    case Image::IMG_UNKNOWN_TYPE:
    default:
      ilType = 0;
      DP_ASSERT( !"Unknown pixel type" );
      break;
  } 
  return ilType;
}

static void faceTwiddling(int face, TextureHostSharedPtr const& texImage)
{
  if ( face == 0 || face == 1 || face == 4 || face == 5 ) // px, nx, pz, nz
  {
    texImage->mirrorY(face);
  }
  else
  {
    texImage->mirrorX(face);
  }
}

ILTexSaverSharedPtr ILTexSaver::create()
{
  return( std::shared_ptr<ILTexSaver>( new ILTexSaver() ) );
}

ILTexSaver::ILTexSaver()
{
  ilInit();
}

ILTexSaver::~ILTexSaver()
{
  ilShutDown();
  ILTexSaver::m_instance = ILTexSaverSharedPtr::null;
}

bool ILTexSaver::save( const TextureHostSharedPtr & image, const string & fileName )
{
  DP_ASSERT(image);
  
  // set locale temporarily to standard "C" locale
  dp::util::TempLocale tl("C");

  bool isCube;
  unsigned int imageID;

  ilGenImages( 1, (ILuint *) &imageID );
  ilBindImage( imageID );  

  string ext = dp::util::getFileExtension( fileName );
  bool isDDS = !_stricmp(".DDS", ext.c_str());   // .dds needs special handling

  if ( isDDS )
  {      
    // DirectDraw Surfaces have their origin at upper left
    ilEnable( IL_ORIGIN_SET );
    ilOriginFunc( IL_ORIGIN_UPPER_LEFT );
  }
  else
  {
    ilDisable( IL_ORIGIN_SET );
  }

  // DevIL does not know how to handle .jps and .pns. Since those formats are just renamed .jpgs and .pngs
  // pass over filename.(jps|pns).(jpg|png) and rename the file after saving it.
  // FIXME Sent bug report to DevIL. Remove this once jps/pns is added to DevIL.
  bool isStereoFormat = false;
  std::string devilFilename = fileName;
  
  if (!_stricmp(".JPS", ext.c_str()))
  {
    isStereoFormat = true;
    devilFilename += ".JPG";
    
  }
  else if (!_stricmp(".PNS", ext.c_str()))
  {
    isStereoFormat = true;
    devilFilename += ".PNG";
  }

  unsigned int numImages = image->getNumberOfImages();
  isCube = image->isCubeMap();
  unsigned char * pixels = NULL;

  // we only handle these cases properly
  DP_ASSERT(isCube == (numImages==6));
  DP_ASSERT(!isCube == (numImages==1));

  for ( unsigned int i = 0; i < numImages; ++i )
  {
    // for DDS cube maps we need to juggle with the faces 
    // to get them into the right order for DDS formats
    int face = determineImage( i, isDDS, isCube );
   
    ilBindImage(imageID);
    ilActiveImage(0);
    ilActiveFace(face);

    DP_ASSERT(IL_NO_ERROR == ilGetError());

    // TODO: Do not know how to handle paletted! This information is already destroyed
    //    // pixel format
    //    unsigned int format = ilGetInteger(IL_IMAGE_FORMAT);
    //    if ( IL_COLOR_INDEX == format )
    //    {
    //      // convert color index to whatever the base type of the palette is
    //      if ( !ilConvertImage(ilGetInteger(IL_PALETTE_BASE_TYPE), IL_UNSIGNED_BYTE) )
    //      {
    //        NVSG_TRACE_OUT("ERROR: conversion from color index format failed!\n");        
    //        INVOKE_CALLBACK(onInvalidFile(fileName, "DevIL Loadable Color-Indexed Image"));
    //        goto ERROREXIT;
    //      }
    //      // now query format of the converted image
    //      format = ilGetInteger(IL_IMAGE_FORMAT);
    //    }
    //

    // Determine the Pixel Format and type
    ILenum ilFormat = determineILFormat(image->getFormat());
    ILenum ilType = determineILType(image->getType());
    
    // Retrieve the image dimensions
    unsigned int width = image->getWidth(i);
    unsigned int height = image->getHeight(i);
    unsigned int depth = image->getDepth(i);

    // If assertion fires, something is wrong
    DP_ASSERT( (width > 0) && (height > 0) && (depth > 0) );
    

    // again some twiddling for DDS format necessary prior to 
    // specify the IL image
    if ( isDDS && isCube )
    {
      faceTwiddling( face, image );
    }

    // temporary cache to retrieve SceniX' TextureHost pixels
    pixels = new unsigned char[image->getNumberOfBytes()];

    // specify the IL image
    image->getSubImagePixels(i, 0, 0, 0, 0, width, height, depth, pixels);
    ilTexImage( width, height, depth, image->getBytesPerPixel(), ilFormat, ilType, NULL  );
    void* destpixels = ilGetData();
    memcpy(destpixels, pixels, image->getNumberOfBytes());

    // done with the temporary pixel cache
    delete[] pixels;
    

    DP_ASSERT(IL_NO_ERROR == ilGetError());

    // undo the face twiddling from before
    if ( isDDS && isCube )
    {
      faceTwiddling( face, image );
    }
  }

  // By default, always overwrite
  ilEnable(IL_FILE_OVERWRITE);
  if ( ilSaveImage( (const ILstring)devilFilename.c_str() ) )
  {
    // For stereo formats rename the file to the original filename
    if (isStereoFormat)
    {
      // Windows will not rename a file if the destination filename already does exist.
      remove( fileName.c_str() );
      rename( devilFilename.c_str(), fileName.c_str() );
    }

    ilDeleteImages(1, &imageID);
    DP_ASSERT(IL_NO_ERROR == ilGetError());
    return true;
  }
  else
  {
#if 0
    NVSG_TRACE_OUT("ERROR: save image failed!\n");      
#endif

    // clean up errors
    while( ilGetError() != IL_NO_ERROR )
    {}
       
    // report that an error has occured
    INVOKE_CALLBACK( onInvalidFile( fileName, "saving image file failed!") );

  }
  // free all resources associated with the DevIL image
  ilDeleteImages(1, &imageID);
  return false;
}
