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


// ILTexLoader.cpp : Defines the entry point for the DLL application.
//

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>
#endif

#include <typeinfo>

#include <dp/sg/core/NVSGVersion.h> // NVSG version info
#include <dp/sg/io/PlugInterfaceID.h>
#include <dp/sg/core/TextureHost.h>

#include <dp/util/DPAssert.h>
#include <dp/util/File.h>
#include <dp/util/Tools.h>

#include "ILTexLoader.h"
#include "il.h"

using namespace dp::util;
using namespace dp::sg::core;

using std::vector;
using std::string;

// a pointer to our single instance of the Loader
dp::util::SmartPtr<ILTexLoader> ILTexLoader::m_instance;

// supported Plug Interface ID
const UPITID PITID_TEXTURE_LOADER(UPITID_TEXTURE_LOADER, UPITID_VERSION); // plug-in type
#define NUM_SUPPORTED_EXTENSIONS 42
static string SUPPORTED_EXTENSIONS[] = { ".TGA", 
                                         ".VDA",
                                         ".ICB",
                                         ".VST",
                                         ".JPG",
                                         ".JPE",
                                         ".JPEG",
                                         ".DDS",
                                         ".PNG",
                                         ".BMP",
                                         ".DIB",
                                         ".GIF",
                                         ".CUT",
                                         ".HDR",
                                         ".ICO",
                                         ".CUR",
                                         ".JNG",
                                         ".LIF",
                                         ".MDL",
                                         ".MNG",
                                         ".PCD",
                                         ".PCX",
                                         ".PIC",
                                         ".PIX",
                                         ".PBM",
                                         ".PGM",
                                         ".PNM",
                                         ".PPM",
                                         ".PSD",
                                         ".PDD",
                                         ".PSP",
                                         ".PXR",
                                         ".SGI",
                                         ".BW",
                                         ".RGB",
                                         ".RGBA",
                                         ".INT",
                                         ".INTA",
                                         ".TIF",
                                         ".TIFF",
                                         ".WAL",
                                         ".XPM" };


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

bool getPlugInterface(const UPIID& piid, dp::util::SmartPtr<PlugIn> & pi)
{
  for( size_t i=0; i<NUM_SUPPORTED_EXTENSIONS; ++i)
  {
    UPIID tempPiid(SUPPORTED_EXTENSIONS[i].c_str(), PITID_TEXTURE_LOADER);
    if( piid==tempPiid )
    {      
      if ( !ILTexLoader::m_instance )
      {
        ILTexLoader::m_instance = dp::util::SmartPtr<ILTexLoader>( new ILTexLoader );
      }
      pi = ILTexLoader::m_instance;
      return !!pi;
    }
  }
  return false;
}


void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();
  for( size_t i=0; i<NUM_SUPPORTED_EXTENSIONS; ++i)
  {
    // generate piids for all supported extensions dynamically
    UPIID tempPiid(SUPPORTED_EXTENSIONS[i].c_str(), PITID_TEXTURE_LOADER);
    piids.push_back(tempPiid);
  }
}

static int determineFace( int i, bool isDDS, bool isCube )
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

static Image::PixelFormat determinePixelFormat(int format) 
{
  Image::PixelFormat pixelFormat;
  switch( format )
  {
    case IL_COLOR_INDEX :
      pixelFormat = Image::IMG_COLOR_INDEX;
      break;
    case IL_RGB :
      pixelFormat = Image::IMG_RGB;
      break;
    case IL_RGBA :
      pixelFormat = Image::IMG_RGBA;
      break;
    case IL_BGR :
      pixelFormat = Image::IMG_BGR;
      break;
    case IL_BGRA :
      pixelFormat = Image::IMG_BGRA;
      break;
    case IL_LUMINANCE :
      pixelFormat = Image::IMG_LUMINANCE;
      break;
    case IL_LUMINANCE_ALPHA :
      pixelFormat = Image::IMG_LUMINANCE_ALPHA;
      break;
    default :
      pixelFormat = Image::IMG_UNKNOWN_FORMAT;
      DP_ASSERT( false );
      break;
  }
  return pixelFormat;
}

static Image::PixelDataType determinePixelDataType(int type)
{
  Image::PixelDataType pixelType;
  switch( type )
  {
    case IL_BYTE :
      pixelType = Image::IMG_BYTE;
      break;
    case IL_UNSIGNED_BYTE :
      pixelType = Image::IMG_UNSIGNED_BYTE;
      break;
    case IL_SHORT :
      pixelType = Image::IMG_SHORT;
      break;
    case IL_UNSIGNED_SHORT :
      pixelType = Image::IMG_UNSIGNED_SHORT;
      break;
    case IL_INT :
      pixelType = Image::IMG_INT;
      break;
    case IL_UNSIGNED_INT :
      pixelType = Image::IMG_UNSIGNED_INT;
      break;
    case IL_FLOAT :
      pixelType = Image::IMG_FLOAT;
      break;
    default :
      pixelType = Image::IMG_UNKNOWN_TYPE;
      DP_ASSERT( false );
      break;
  }
  return pixelType;
}

ILTexLoader::ILTexLoader()
{
  ilInit();
}

ILTexLoader::~ILTexLoader()
{
  ilShutDown();
}

void ILTexLoader::deleteThis()
{
  if ( m_instance->isShared() )
  {
    m_instance->removeRef();
  }
  else
  {
    delete this;
  }
}

bool ILTexLoader::onLoad( TextureHostSharedPtr const& texImg
                        , const vector<string> & searchPaths )
{
  const string & filename = texImg->getFileName();

  // set locale temporarily to standard "C" locale
  TempLocale tl("C");

  bool cube;
  unsigned int imageID;
  ilGenImages( 1, (ILuint *) &imageID );
  ilBindImage( imageID );  

  std::string ext = dp::util::getFileExtension( filename );
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

  // load the image from file
  if ( ilLoadImage( (const ILstring) filename.c_str() ) )
  {
    int numImages = ilGetInteger(IL_NUM_IMAGES)+1; 
    // ^^^^^^^^^^^^^ Querying for IL_NUM_IMAGES returns the number of images
    //               following the current one. Add 1 for the right image count!
    
    int numMipMaps = ilGetInteger(IL_NUM_MIPMAPS);

    vector<const void*> mipmaps; // empty after instantiation
    if ( numImages > 1 && !numMipMaps ) 
    {
      // check if we have mipmaps as top-level images
      // NOTE: this, for example, could apply for tiff images

      bool failed = false; // indicates if test below has failed
      unsigned int w = 0; // width of the predecessor image
      
      for ( int i=0; i<numImages; ++i )
      {
        ilBindImage(imageID);  
        ilActiveImage(i);
        DP_ASSERT(IL_NO_ERROR == ilGetError()); 

        unsigned int ww = ilGetInteger( IL_IMAGE_WIDTH ); // actual image width
        if ( i ) // start checking with the second image
        {
          if ( ww == (w >> 1) ) // criteria for next mipmap level
          { 
            // top-level image actually is the i-th mipmap level
            mipmaps.push_back(ilGetData()); 
          } 
          else
          {
            // could not identify top-level image as a mipmap level
            // --> no further testing required
            failed = true; // test failed
            break; 
          }
        }
        w = ww; // prepare next-level test
      }
      if ( !failed && !mipmaps.empty() )
      {
        // reset # images to 1 in case we could have identified all but 
        // the first top-level image as actually being mipmap levels
        numImages = 1;
      }
    }

    cube =  ilGetInteger(IL_IMAGE_CUBEFLAGS)!=0 || numImages==6;

    // re-bind
    ilBindImage(imageID);

    for ( int image=0; image < numImages; ++image )
    {

      // cube faces within DevIL philosophy are organized like this:
      //
      //   image -> 1st face -> face index 0
      //   face1 -> 2nd face -> face index 1
      //   ...
      //   face5 -> 6th face -> face index 5

      int numFaces = ilGetInteger(IL_NUM_FACES)+1;

      for ( int f=0; f<numFaces; ++f )
      {
        // need to juggle with the faces to get them aligned with
        // how OpenGL expects cube faces ...
        int face = determineFace( f, isDDS, cube );

        // DevIL frequently loses track of the current state
        // 
        ilBindImage(imageID);
        ilActiveImage(image);
        ilActiveFace(face);
        //
        DP_ASSERT(IL_NO_ERROR == ilGetError()); 

        // pixel format
        unsigned int format = ilGetInteger(IL_IMAGE_FORMAT);
        if ( IL_COLOR_INDEX == format )
        {
          // convert color index to whatever the base type of the palette is
          if ( !ilConvertImage(ilGetInteger(IL_PALETTE_BASE_TYPE), IL_UNSIGNED_BYTE) )
          {
#if 0
            NVSG_TRACE_OUT("ERROR: conversion from color index format failed!\n");
#endif
            INVOKE_CALLBACK(onInvalidFile(filename, "DevIL Loadable Color-Indexed Image"));
            goto ERROREXIT;
          }
          // now query format of the converted image
          format = ilGetInteger(IL_IMAGE_FORMAT);
        }

        Image::PixelFormat pixelFormat = determinePixelFormat(format);
        Image::PixelDataType pixelType = determinePixelDataType(ilGetInteger( IL_IMAGE_TYPE ));

        // image dimension in pixels
        unsigned int width = ilGetInteger( IL_IMAGE_WIDTH );
        unsigned int height = ilGetInteger( IL_IMAGE_HEIGHT );
        unsigned int depth = ilGetInteger( IL_IMAGE_DEPTH );

        // should be at least on pixel in each direction 
        width  += (width==0);
        height += (height==0);
        depth  += (depth==0);

        unsigned int nImg = texImg->addImage( width, height, depth, pixelFormat, pixelType );
        if ( !( texImg->getCreationFlags() & TextureHost::F_NO_IMAGE_CREATION ) )
        {
          // Use mipmaps, if any, in the original file
          if ( numMipMaps > 0 )
          {
            mipmaps.clear(); // clear this for currently processed image
            for (int j = 1; j <= numMipMaps; ++j)
            { 
              // DevIL frequently loses track of the current state
              // 
              ilBindImage(imageID);
              ilActiveImage(image);
              ilActiveFace(face);
              //
              ilActiveMipmap(j);
              mipmaps.push_back((const void*)ilGetData());
            }

            // DevIL frequently loses track of the current state
            // 
            ilBindImage(imageID);
            ilActiveImage(image);
            ilActiveFace(face);
            //
            ilActiveMipmap(0);
          }
          texImg->setImageData( nImg, (const void*)ilGetData(), mipmaps );

          if ( isDDS )
          {
            // !!WARNING: 
            // This piece of code MUST NOT be visited twice for the same image,
            // because this would falsify the desired effect!

            if ( cube )
            {
              // the images at this position are flipped at the x-axis (due to devil)
              // flipping at x-axis will result in original image
              // mirroring at y-axis will result in rotating the image 180 degree
              if ( face == 0 || face == 1 || face == 4 || face == 5 ) // px, nx, pz, nz
              {
                texImg->mirrorY(nImg); // mirror over y-axis              
              }
              else // py, ny
              {
                texImg->mirrorX(nImg); // flip over x-axis              
              }
            }
          }            

          ILuint origin = ilGetInteger(IL_IMAGE_ORIGIN);  // need to call this before ilDeleteImages()!!

          if ( !cube && (origin==IL_ORIGIN_UPPER_LEFT) )    
          {      
            // OpenGL expects origin at lower left, so the image has to be flipped at the x-axis
            // for dds cubemaps we handle the separate face rotations above
            texImg->mirrorX(nImg);   // reverse rows
          }
        }
      }
    }
    // free all resources associated with the DevIL image
    // note: free this memory before we eventually flip the image (that will allocate more memory)
    ilDeleteImages(1, &imageID);
    DP_ASSERT(IL_NO_ERROR == ilGetError());

    if ( cube )
    {
      texImg->setTextureTarget( TT_TEXTURE_CUBE );
    }
    if ( numMipMaps )
    {
      texImg->incrementMipmapUseCount();
    }
  }
  else
  {
#if 0
    NVSG_TRACE_OUT("ERROR: load image failed!\n");
#endif
    INVOKE_CALLBACK(onInvalidFile(filename, "DevIL Loadable Image"));
    goto ERROREXIT;
  }

  return( true );

ERROREXIT:
  // free all resources associated with the DevIL image
  ilDeleteImages(1, &imageID);
  DP_ASSERT(IL_NO_ERROR == ilGetError());
  return( false );
}
