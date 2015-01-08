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


#include <sstream>
#include <locale.h>
#include <dp/DP.h>
#include <dp/fx/EffectLibrary.h>
#if defined(HAVE_HALF_FLOAT)
#include <dp/math/half.h>
#endif
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/ClipPlane.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/MatrixCamera.h>
#include <dp/sg/core/ParallelCamera.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/io/PlugInterfaceID.h>
#include <dp/util/File.h>
#include <dp/util/Tools.h>
#include <dp/sg/io/DPAF/Saver/inc/DPAFSaver.h>

using namespace dp::sg::core;
using namespace dp::math;
using namespace dp::util;
using dp::util::PlugIn;
using dp::util::UPIID;
using dp::util::UPITID;
using std::set;
using std::vector;
using std::string;
using std::pair;

// convenient macro
#ifndef  INVOKE_CALLBACK
#define INVOKE_CALLBACK(cb) if ( callback() ) callback()->cb
#endif

// supported Plug Interface ID
const UPITID PITID_SCENE_SAVER(UPITID_SCENE_SAVER, UPITID_VERSION); // plug-in type
UPIID PIID_DP_SCENE_SAVER = UPIID(".DPAF", PITID_SCENE_SAVER); 

#if defined( _WIN32 )
BOOL APIENTRY DllMain(HANDLE hModule, DWORD reason, LPVOID lpReserved)
{
  if (reason == DLL_PROCESS_ATTACH)
  {
    // initialize supported Plug Interface ID
    PIID_DP_SCENE_SAVER = UPIID(".DPAF", PITID_SCENE_SAVER); 
    int i=0;
  }

  return TRUE;
}
#elif defined( LINUX )
#endif

bool getPlugInterface(const UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  // check if UPIID is properly initialized 
  DP_ASSERT(PIID_DP_SCENE_SAVER==UPIID(".DPAF", PITID_SCENE_SAVER));

  if ( piid==PIID_DP_SCENE_SAVER )
  {
    pi = DPAFSaver::create();
    return( !!pi );
  }
  return false;
}

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(PIID_DP_SCENE_SAVER);
}

void  fprintfCompareMode( FILE *fh, const char *prefix, TextureCompareMode mode )
{
  fprintf( fh, "%s", prefix );
  switch( mode )
  {
    case TCM_R_TO_TEXTURE :
      fprintf( fh, "R_TO_TEXTURE\n" );
      break;
    default :
      DP_ASSERT( !"Unexpected TextureCompareMode. Using default TCM_NONE" );
      // fall through
    case TCM_NONE :
      fprintf( fh, "NONE\n" );
      break;
  }
}

void  fprintfMagFilterMode( FILE *fh, const char *prefix, TextureMagFilterMode mode )
{
  fprintf( fh, "%s", prefix );
  switch( mode )
  {
    case TFM_MAG_NEAREST :
      fprintf( fh, "NEAREST\n" );
      break;
    default :
      DP_ASSERT( !"Unexpected TextureMagFilterMode. Using default TFM_MAG_LINEAR" );
      // fall through
    case TFM_MAG_LINEAR :
      fprintf( fh, "LINEAR\n" );
      break;
  }
}

void  fprintfMinFilterMode( FILE *fh, const char *prefix, TextureMinFilterMode mode )
{
  fprintf( fh, "%s", prefix );
  switch( mode )
  {
    case TFM_MIN_NEAREST :
      fprintf( fh, "NEAREST\n" );
      break;
    case TFM_MIN_LINEAR :
      fprintf( fh, "LINEAR\n" );
      break;
    default :
      DP_ASSERT( !"Unexpected TextureMinFilterMode. Using default TFM_MIN_LINEAR_MIPMAP_LINEAR" );
      // fall through
    case TFM_MIN_LINEAR_MIPMAP_LINEAR :
      fprintf( fh, "LINEAR_MIPMAP_LINEAR\n" );
      break;
    case TFM_MIN_NEAREST_MIPMAP_NEAREST :
      fprintf( fh, "NEAREST_MIPMAP_NEAREST\n" );
      break;
    case TFM_MIN_NEAREST_MIPMAP_LINEAR :
      fprintf( fh, "NEAREST_MIPMAP_LINEAR\n" );
      break;
    case TFM_MIN_LINEAR_MIPMAP_NEAREST :
      fprintf( fh, "LINEAR_MIPMAP_NEAREST\n" );
      break;
  }
}

const char * wrapMode( TextureWrapMode mode )
{
  switch( mode )
  {
    default :
      DP_ASSERT( !"Unexpected TextureWrapMode. Using default TWM_REPEAT" );
      // fall through
    case TWM_REPEAT :
      return( "REPEAT" );
      break;
    case TWM_CLAMP :
      return( "CLAMP" );
      break;
    case TWM_MIRROR_REPEAT :
      return( "MIRROR_REPEAT" );
      break;
    case TWM_CLAMP_TO_EDGE :
      return( "CLAMP_TO_EDGE" );
      break;
    case TWM_CLAMP_TO_BORDER :
      return( "CLAMP_TO_BORDER" );
      break;
    case TWM_MIRROR_CLAMP :
      return( "MIRROR_CLAMP" );
      break;
    case TWM_MIRROR_CLAMP_TO_EDGE :
      return( "MIRROR_CLAMP_TO_EDGE" );
      break;
    case TWM_MIRROR_CLAMP_TO_BORDER :
      return( "MIRROR_CLAMP_TO_BORDER" );
      break;
  }
}

void  fprintfWrapModes( FILE *fh, const char *prefix, TextureWrapMode wrapModeS, TextureWrapMode wrapModeT, TextureWrapMode wrapModeR )
{
  fprintf( fh, "%s%s %s %s\n", prefix, wrapMode( wrapModeS ), wrapMode( wrapModeT ), wrapMode( wrapModeR ) );
}

void  fprintfPixelFormat( FILE *fh, const char *prefix, Image::PixelFormat pf )
{
  fprintf( fh, "%s", prefix );
  switch( pf )
  {
  case Image::IMG_COLOR_INDEX :
    fprintf( fh, "COLOR_INDEX\n" );
    break;
  case Image::IMG_RGB :
    fprintf( fh, "RGB\n" );
    break;
  case Image::IMG_RGBA :
    fprintf( fh, "RGBA\n" );
    break;
  case Image::IMG_BGR :
    fprintf( fh, "BGR\n" );
    break;
  case Image::IMG_BGRA :
    fprintf( fh, "BGRA\n" );
    break;
  case Image::IMG_LUMINANCE :
    fprintf( fh, "LUMINANCE\n" );
    break;
  case Image::IMG_LUMINANCE_ALPHA :
    fprintf( fh, "LUMINANCE_ALPHA\n" );
    break;
  case Image::IMG_ALPHA :
    fprintf( fh, "IMG_ALPHA\n" );
    break;
  case Image::IMG_DEPTH_COMPONENT :
    fprintf( fh, "IMG_DEPTH_COMPONENT\n" );
    break;
  case Image::IMG_DEPTH_STENCIL :
    fprintf( fh, "IMG_DEPTH_STENCIL\n" );
    break;
  case Image::IMG_INTEGER_ALPHA :
    fprintf( fh, "IMG_INTEGER_ALPHA\n" );
    break;
  case Image::IMG_INTEGER_LUMINANCE :
    fprintf( fh, "IMG_INTEGER_LUMINANCE\n" );
    break;
  case Image::IMG_INTEGER_LUMINANCE_ALPHA :
    fprintf( fh, "IMG_INTEGER_LUMINANCE_ALPHA\n" );
    break;
  case Image::IMG_INTEGER_RGB :
    fprintf( fh, "IMG_INTEGER_RGB\n" );
    break;
  case Image::IMG_INTEGER_BGR :
    fprintf( fh, "IMG_INTEGER_BGR\n" );
    break;
  case Image::IMG_INTEGER_RGBA :
    fprintf( fh, "IMG_INTEGER_RGBA\n" );
    break;
  case Image::IMG_INTEGER_BGRA :
    fprintf( fh, "IMG_INTEGER_BGRA\n" );
    break;
  case Image::IMG_COMPRESSED_LUMINANCE_LATC1 :
    fprintf( fh, "IMG_COMPRESSED_LUMINANCE_LATC1\n" );
    break;
  case Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1 :
    fprintf( fh, "IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1\n" );
    break;
  case Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2 :
    fprintf( fh, "IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2\n" );
    break;
  case Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2 :
    fprintf( fh, "IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2\n" );
    break;
  case Image::IMG_COMPRESSED_RED_RGTC1 :
    fprintf( fh, "IMG_COMPRESSED_RED_RGTC1\n" );
    break;
  case Image::IMG_COMPRESSED_SIGNED_RED_RGTC1 :
    fprintf( fh, "IMG_COMPRESSED_SIGNED_RED_RGTC1\n" );
    break;
  case Image::IMG_COMPRESSED_RG_RGTC2 :
    fprintf( fh, "IMG_COMPRESSED_RG_RGTC2\n" );
    break;
  case Image::IMG_COMPRESSED_SIGNED_RG_RGTC2 :
    fprintf( fh, "IMG_COMPRESSED_SIGNED_RG_RGTC2\n" );
    break;
  case Image::IMG_COMPRESSED_RGB_DXT1 :
    fprintf( fh, "IMG_COMPRESSED_RGB_DXT1\n" );
    break;
  case Image::IMG_COMPRESSED_RGBA_DXT1 :
    fprintf( fh, "IMG_COMPRESSED_RGBA_DXT1\n" );
    break;
  case Image::IMG_COMPRESSED_RGBA_DXT3 :
    fprintf( fh, "IMG_COMPRESSED_RGBA_DXT3\n" );
    break;
  case Image::IMG_COMPRESSED_RGBA_DXT5 :
    fprintf( fh, "IMG_COMPRESSED_RGBA_DXT5\n" );
    break;
  case Image::IMG_COMPRESSED_SRGB_DXT1 :
    fprintf( fh, "IMG_COMPRESSED_SRGB_DXT1\n" );
    break;
  case Image::IMG_COMPRESSED_SRGBA_DXT1 :
    fprintf( fh, "IMG_COMPRESSED_SRGBA_DXT1\n" );
    break;
  case Image::IMG_COMPRESSED_SRGBA_DXT3 :
    fprintf( fh, "IMG_COMPRESSED_SRGBA_DXT3\n" );
    break;
  case Image::IMG_COMPRESSED_SRGBA_DXT5 :
    fprintf( fh, "IMG_COMPRESSED_SRGBA_DXT5\n" );
    break;
  case Image::IMG_UNKNOWN_FORMAT :
  default :
    DP_ASSERT( false );
    fprintf( fh, "\n" );
    break;
  }
}

void  fprintfPixelType( FILE *fh, const char *prefix, Image::PixelDataType dt )
{
  fprintf( fh, "%s", prefix );
  switch( dt )
  {
  case Image::IMG_BYTE :
    fprintf( fh, "BYTE\n" );
    break;
  case Image::IMG_UNSIGNED_BYTE :
    fprintf( fh, "UNSIGNED_BYTE\n" );
    break;
  case Image::IMG_SHORT :
    fprintf( fh, "SHORT\n" );
    break;
  case Image::IMG_UNSIGNED_SHORT :
    fprintf( fh, "UNSIGNED_SHORT\n" );
    break;
  case Image::IMG_INT :
    fprintf( fh, "INT\n" );
    break;
  case Image::IMG_UNSIGNED_INT :
    fprintf( fh, "UNSIGNED_INT\n" );
    break;
  case Image::IMG_FLOAT :
    fprintf( fh, "FLOAT\n" );
    break;
  case Image::IMG_HALF :
    fprintf( fh, "HALF\n" );
    break;
  case Image::IMG_UNKNOWN_TYPE :
  default:
    DP_ASSERT( false );
    fprintf( fh, "\n" );
    break;
  }
}

void writePixelComponent( FILE *fh, char v )
{
  fprintf( fh, "%d ", v );
}

void writePixelComponent( FILE *fh, unsigned char v )
{
  fprintf( fh, "0x%02x ", v );
}

void writePixelComponent( FILE *fh, short v )
{
  fprintf( fh, "%d ", v );
}

void writePixelComponent( FILE *fh, unsigned short v )
{
  fprintf( fh, "0x%04x ", v );
}

void writePixelComponent( FILE *fh, int v )
{
  fprintf( fh, "%d ", v );
}

void writePixelComponent( FILE *fh, unsigned int v )
{
  fprintf( fh, "0x%08x ", v );
}

void writePixelComponent( FILE *fh, float v )
{
  fprintf( fh, "%f ", v );
}

#if defined(HAVE_HALF_FLOAT)
void writePixelComponent( FILE *fh, half v )
{
  // cannot pass object of non-POD type like 'half' through '...'
  // fortunately, half provides a float cast operator ;-)
  fprintf( fh, "%f ", (float)v );
}
#endif

template<typename T>
void writePixels( FILE *fh, const char * prefix, TextureHostSharedPtr const& ti, unsigned int image
                 , unsigned int mipmap )
{
  unsigned int depth = ti->getDepth( image, mipmap );
  unsigned int height = ti->getHeight( image, mipmap );
  unsigned int width = ti->getWidth( image, mipmap );
  unsigned int noc = numberOfComponents( ti->getFormat( image, mipmap ) );
  if ( ti->getPixels( image, mipmap ) )
  {
    Buffer::DataReadLock buffer( ti->getPixels( image, mipmap ) );
    const T * pixels = buffer.getPtr<T>();
    for ( unsigned int idx=0, i=0 ; i<depth ; i++ )
    {
      for ( unsigned int j=0 ; j<height ; j++ )
      {
        fprintf( fh, "%s", prefix );
        for ( unsigned int k=0 ; k<width ; k++ )
        {
          for ( unsigned int l=0 ; l<noc ; l++, idx++ )
          {
            writePixelComponent( fh, pixels[idx] );
          }
          fprintf( fh, " " );
        }
        fprintf( fh, "\n" );
      }
      if ( i < depth-1 )
      {
        fprintf( fh, "\n" );
      }
    }
  }
}

void fprintfPixels( FILE *fh, const char *prefix, TextureHostSharedPtr const& ti, unsigned int image
                   , unsigned int mipmap )
{
  switch( ti->getType( image, mipmap ) )
  {
  case Image::IMG_BYTE :
    writePixels<char>( fh, prefix, ti, image, mipmap );
    break;
  case Image::IMG_UNSIGNED_BYTE :
    writePixels<unsigned char>( fh, prefix, ti, image, mipmap );
    break;
  case Image::IMG_SHORT :
    writePixels<short>( fh, prefix, ti, image, mipmap );
    break;
  case Image::IMG_UNSIGNED_SHORT :
    writePixels<unsigned short>( fh, prefix, ti, image, mipmap );
    break;
  case Image::IMG_INT :
    writePixels<int>( fh, prefix, ti, image, mipmap );
    break;
  case Image::IMG_UNSIGNED_INT :
    writePixels<unsigned int>( fh, prefix, ti, image, mipmap );
    break;
  case Image::IMG_FLOAT :
    writePixels<float>( fh, prefix, ti, image, mipmap );
    break;
#if defined(HAVE_HALF_FLOAT)
  case Image::IMG_HALF :
    writePixels<half>( fh, prefix, ti, image, mipmap );
    break;
#endif
  case Image::IMG_UNKNOWN_TYPE :
  default:
    DP_ASSERT( false );
    break;
  }
}

void fprintfPatchesMode( FILE *fh, const char *prefix, PatchesMode pm )
{
  fprintf( fh, "%s", prefix );
  switch( pm )
  {
    case PATCHES_MODE_TRIANGLES :
      fprintf( fh, "Triangles\n" );
      break;
    case PATCHES_MODE_QUADS :
      fprintf( fh, "Quads\n" );
      break;
    case PATCHES_MODE_ISOLINES :
      fprintf( fh, "Isolines\n" );
      break;
    case PATCHES_MODE_POINTS :
      fprintf( fh, "Points\n" );
      break;
    default :
      DP_ASSERT( false );
      break;
  }
}

void fprintfPatchesOrdering( FILE *fh, const char *prefix, PatchesOrdering po )
{
  fprintf( fh, "%s", prefix );
  switch( po )
  {
    case PATCHES_ORDERING_CW :
      fprintf( fh, "CW\n" );
      break;
    case PATCHES_ORDERING_CCW :
      fprintf( fh, "CCW\n" );
      break;
    default :
      DP_ASSERT( false );
      break;
  }
}

void fprintfPatchesSpacing( FILE *fh, const char *prefix, PatchesSpacing ps )
{
  fprintf( fh, "%s", prefix );
  switch( ps )
  {
    case PATCHES_SPACING_EQUAL :
      fprintf( fh, "Equal\n" );
      break;
    case PATCHES_SPACING_FRACTIONAL_EVEN :
      fprintf( fh, "Even\n" );
      break;
    case PATCHES_SPACING_FRACTIONAL_ODD :
      fprintf( fh, "Odd\n" );
      break;
    default :
      DP_ASSERT( false );
      break;
  }
}

void  fprintfAlignment( FILE *fh, const char *prefix, Billboard::Alignment ba )
{
  fprintf( fh, "%s", prefix );
  switch( ba )
  {
  case Billboard::BA_AXIS :
    fprintf( fh, "AXIS\n" );
    break;
  case Billboard::BA_SCREEN :
    fprintf( fh, "SCREEN\n" );
    break;
  case Billboard::BA_VIEWER :
    fprintf( fh, "VIEWER\n" );
    break;
  default :
    DP_ASSERT( false );
    break;
  }
}

void  fprintfBool( FILE *fh, const char *prefix, bool b )
{
  fprintf( fh, "%s%s\n", prefix, b ? "TRUE" : "FALSE" );
}

template<typename IntType>
void  fprintfInt1Array( FILE *fh, const char * prefix, unsigned int count, const IntType *fInput, unsigned int stride )
{
  DP_ASSERT( count );
  const char *ptr = reinterpret_cast<const char *>(fInput);
  if ( count == 1 )
  {
    const IntType *f = reinterpret_cast<const IntType *>(ptr);
    fprintf( fh, "%s[ %d ]\n", prefix, f[0] );
  }
  else
  {
    fprintf( fh, "%s[ ", prefix );

    for ( unsigned int i=0 ; i<count-1 ; i++ )
    {
      // start a new line every 20 elements
      if ( i && (i % 20) == 0 )
      {
        fprintf( fh, "\n%s  ", prefix );
      }

      const IntType *f = reinterpret_cast<const IntType *>(ptr);
      fprintf( fh, "%d, ", f[0] );
      ptr += stride;
    }
    const IntType *f = reinterpret_cast<const IntType *>(ptr);
    fprintf( fh, "%d ]\n", f[0] );
  }
}

template<typename IntType>
void  fprintfInt2Array( FILE *fh, const char *prefix, unsigned int count, const IntType *fInput, unsigned int stride )
{
  DP_ASSERT( count );
  const char *ptr = reinterpret_cast<const char *>(fInput);
  if ( count == 1 )
  {
    const IntType *f = reinterpret_cast<const IntType *>(ptr);
    fprintf( fh, "%s[ 0x%X, 0x%X ]\n", prefix, f[0], f[1] );
  }
  else
  {
    fprintf( fh, "%s[\n", prefix );
    for ( unsigned int i=0 ; i<count-1 ; i++ )
    {
      const IntType *f = reinterpret_cast<const IntType *>(ptr);
      fprintf( fh, "%s\t0x%X, 0x%X;\n", prefix, f[0], f[1] );
      ptr += stride;
    }
    const IntType *f = reinterpret_cast<const IntType *>(ptr);
    fprintf( fh, "%s\t0x%X, 0x%X\n", prefix, f[0], f[1] );
    fprintf( fh, "%s]\n", prefix );
  }
}

template<typename IntType>
void  fprintfInt3Array( FILE *fh, const char *prefix, unsigned int count, const IntType *fInput, unsigned int stride )
{
  DP_ASSERT( count );
  const char *ptr = reinterpret_cast<const char *>(fInput);
  if ( count == 1 )
  {
    const IntType *f = reinterpret_cast<const IntType *>(ptr);
    fprintf( fh, "%s[ 0x%X, 0x%X, 0x%X ]\n", prefix, f[0], f[1], f[2] );
  }
  else
  {
    fprintf( fh, "%s[\n", prefix );
    for ( unsigned int i=0 ; i<count-1 ; i++ )
    {
      const IntType *f = reinterpret_cast<const IntType *>(ptr);
      fprintf( fh, "%s\t0x%X, 0x%X, 0x%X;\n", prefix, f[0], f[1], f[2] );
      ptr += stride;
    }
    const IntType *f = reinterpret_cast<const IntType *>(ptr);
    fprintf( fh, "%s\t0x%X, 0x%X, 0x%X\n", prefix, f[0], f[1], f[2] );
    fprintf( fh, "%s]\n", prefix );
  }
}

template<typename IntType>
void  fprintfInt4Array( FILE *fh, const char *prefix, unsigned int count, const IntType *fInput, unsigned int stride )
{
  DP_ASSERT( count );
  const char *ptr = reinterpret_cast<const char *>(fInput);
  if ( count == 1 )
  {
    const IntType *f = reinterpret_cast<const IntType *>(ptr);
    fprintf( fh, "%s[ 0x%X, 0x%X, 0x%X, 0x%X ]\n", prefix, f[0], f[1], f[2], f[3] );
  }
  else
  {
    fprintf( fh, "%s[\n", prefix );
    for ( unsigned int i=0 ; i<count-1 ; i++ )
    {
      const IntType *f = reinterpret_cast<const IntType *>(ptr);
      fprintf( fh, "%s\t0x%X, 0x%X, 0x%X, 0x%X;\n", prefix, f[0], f[1], f[2], f[3] );
      ptr += stride;
    }
    const IntType *f = reinterpret_cast<const IntType *>(ptr);
    fprintf( fh, "%s\t0x%X, 0x%X, 0x%X, 0x%X\n", prefix, f[0], f[1], f[2], f[3] );
    fprintf( fh, "%s]\n", prefix );
  }
}

template <typename FloatType>
void  fprintfFloat1Array( FILE *fh, const char *prefix, unsigned int count, const FloatType *fInput, unsigned int stride )
{
  DP_ASSERT( count );
  const char *ptr = reinterpret_cast<const char *>(fInput);
  if ( count == 1 )
  {
    const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
    fprintf( fh, "%s[ %f ]\n", prefix, f[0] );
  }
  else
  {
    fprintf( fh, "%s[ ", prefix );

    for ( unsigned int i=0 ; i<count-1 ; i++ )
    {
      const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
      fprintf( fh, "%f, ", f[0] );
      ptr += stride;
    }
    const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
    fprintf( fh, "%f ]\n", f[0] );
  }
}

template <typename FloatType>
void  fprintfFloat2Array( FILE *fh, const char *prefix, unsigned int count, const FloatType *fInput, unsigned int stride )
{
  DP_ASSERT( count );
  const char *ptr = reinterpret_cast<const char *>(fInput);
  if ( count == 1 )
  {
    const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
    fprintf( fh, "%s[ %f, %f ]\n", prefix, f[0], f[1] );
  }
  else
  {
    fprintf( fh, "%s[\n", prefix );
    for ( unsigned int i=0 ; i<count-1 ; i++ )
    {
      const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
      fprintf( fh, "%s\t%f, %f;\n", prefix, f[0], f[1] );
      ptr += stride;
    }
    const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
    fprintf( fh, "%s\t%f, %f\n", prefix, f[0], f[1] );
    fprintf( fh, "%s]\n", prefix );
  }
}

template <typename FloatType>
void  fprintfFloat3Array( FILE *fh, const char *prefix, unsigned int count, const FloatType *fInput, unsigned int stride )
{
  DP_ASSERT( count );
  const char *ptr = reinterpret_cast<const char *>(fInput);
  if ( count == 1 )
  {
    const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
    fprintf( fh, "%s[ %f, %f, %f ]\n", prefix, f[0], f[1], f[2] );
  }
  else
  {
    fprintf( fh, "%s[\n", prefix );
    for ( unsigned int i=0 ; i<count-1 ; i++ )
    {
      const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
      fprintf( fh, "%s\t%f, %f, %f;\n", prefix, f[0], f[1], f[2] );
      ptr += stride;
    }
    const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
    fprintf( fh, "%s\t%f, %f, %f\n", prefix, f[0], f[1], f[2] );
    fprintf( fh, "%s]\n", prefix );
  }
}

template <typename FloatType>
void  fprintfFloat4Array( FILE *fh, const char *prefix, unsigned int count, const FloatType *fInput, unsigned int stride )
{
  DP_ASSERT( count );
  const char *ptr = reinterpret_cast<const char *>(fInput);
  if ( count == 1 )
  {
    const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
    fprintf( fh, "%s[ %f, %f, %f, %f ]\n", prefix, f[0], f[1], f[2], f[3] );
  }
  else
  {
    fprintf( fh, "%s[\n", prefix );
    for ( unsigned int i=0 ; i<count-1 ; i++ )
    {
      const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
      fprintf( fh, "%s\t%f, %f, %f, %f;\n", prefix, f[0], f[1], f[2], f[3] );
      ptr += stride;
    }
    const FloatType *f = reinterpret_cast<const FloatType *>(ptr);
    fprintf( fh, "%s\t%f, %f, %f, %f\n", prefix, f[0], f[1], f[2], f[3] );
    fprintf( fh, "%s]\n", prefix );
  }
}

void  fprintfMat44f( FILE *fh, const char *firstLine, const char *nextLines, const Mat44f &m )
{
  fprintf( fh, "%s( ( %f, %f, %f, %f )\n", firstLine, m[0][0], m[0][1], m[0][2], m[0][3] );
  fprintf( fh, "%s  ( %f, %f, %f, %f )\n", nextLines, m[1][0], m[1][1], m[1][2], m[1][3] );
  fprintf( fh, "%s  ( %f, %f, %f, %f )\n", nextLines, m[2][0], m[2][1], m[2][2], m[2][3] );
  fprintf( fh, "%s  ( %f, %f, %f, %f ) )\n", nextLines, m[3][0], m[3][1], m[3][2], m[3][3] );
}

void  fprintfQuatf( FILE *fh, const char * prefix, const char * postfix, const Quatf &q )
{
  fprintf( fh, "%s%f, %f, %f %f%s\n", prefix, q[0], q[1], q[2], q[3], postfix );
}

template<typename T>
void writeScalar( FILE * fh, const T & v )
{
  DP_ASSERT( false );
}

template<>
void writeScalar<float>( FILE * fh, const float & v )
{
  fprintf( fh, "%f ", v );
}

template<unsigned n, typename T>
void writeVector( FILE * fh, const char * prefix, const Vecnt<n,T> & v )
{
  fprintf( fh, "%s( ", prefix );
  for ( unsigned int i=0 ; i<n ; i++ )
  {
    writeScalar<T>( fh, v[i] );
  }
  fprintf( fh, ")\n" );
}

void  fprintfArray( FILE *fh, unsigned int type, unsigned int size, unsigned int count, Buffer::DataReadLock lock, unsigned int stride, const char *prefix )
{
  const void *vdata = lock.getPtr();
  switch ( type )
  {
  case dp::util::DT_INT_8:
  case dp::util::DT_UNSIGNED_INT_8:
    switch( size )
    {
    case 1: fprintfInt1Array(fh, prefix, count, (const unsigned char*)vdata, stride); break;
    case 2: fprintfInt2Array(fh, prefix, count, (const unsigned char*)vdata, stride); break;
    case 3: fprintfInt3Array(fh, prefix, count, (const unsigned char*)vdata, stride); break;
    case 4: fprintfInt4Array(fh, prefix, count, (const unsigned char*)vdata, stride); break;
    }
    break;
  case dp::util::DT_INT_16:
  case dp::util::DT_UNSIGNED_INT_16:
    switch( size )
    {
    case 1: fprintfInt1Array(fh, prefix, count, (const unsigned short*)vdata, stride); break;
    case 2: fprintfInt2Array(fh, prefix, count, (const unsigned short*)vdata, stride); break;
    case 3: fprintfInt3Array(fh, prefix, count, (const unsigned short*)vdata, stride); break;
    case 4: fprintfInt4Array(fh, prefix, count, (const unsigned short*)vdata, stride); break;
    }
    break;
  case dp::util::DT_INT_32:
  case dp::util::DT_UNSIGNED_INT_32:
    switch( size )
    {
    case 1: fprintfInt1Array(fh, prefix, count, (const unsigned int*)vdata, stride); break;
    case 2: fprintfInt2Array(fh, prefix, count, (const unsigned int*)vdata, stride); break;
    case 3: fprintfInt3Array(fh, prefix, count, (const unsigned int*)vdata, stride); break;
    case 4: fprintfInt4Array(fh, prefix, count, (const unsigned int*)vdata, stride); break;
    }
    break;
  case dp::util::DT_FLOAT_32:
    switch( size )
    {
    case 1: fprintfFloat1Array(fh, prefix, count, (const float*)vdata, stride); break;
    case 2: fprintfFloat2Array(fh, prefix, count, (const float*)vdata, stride); break;
    case 3: fprintfFloat3Array(fh, prefix, count, (const float*)vdata, stride); break;
    case 4: fprintfFloat4Array(fh, prefix, count, (const float*)vdata, stride); break;
    }
    break;
  case dp::util::DT_FLOAT_64:
    switch( size )
    {
    case 1: fprintfFloat1Array(fh, prefix, count, (const double*)vdata, stride); break;
    case 2: fprintfFloat2Array(fh, prefix, count, (const double*)vdata, stride); break;
    case 3: fprintfFloat3Array(fh, prefix, count, (const double*)vdata, stride); break;
    case 4: fprintfFloat4Array(fh, prefix, count, (const double*)vdata, stride); break;
    }
    break;
  default:
    DP_ASSERT( !"unsupported datatype" );
  }
}

void  fprintfTextureTarget( FILE *fh, const char * prefix, TextureTarget target, const char * postfix )
{
  string name;
  switch( target)
  {
  case TT_TEXTURE_1D :        name = "TEXTURE_1D";          break;
  case TT_TEXTURE_2D :        name = "TEXTURE_2D";          break;
  case TT_TEXTURE_3D :        name = "TEXTURE_3D";          break;
  case TT_TEXTURE_CUBE :      name = "TEXTURE_CUBE";        break;
  case TT_TEXTURE_1D_ARRAY :  name = "TEXTURE_1D_ARRAY";    break;
  case TT_TEXTURE_2D_ARRAY :  name = "TEXTURE_2D_ARRAY";    break;
  case TT_TEXTURE_RECTANGLE : name = "TEXTURE_RECTANGLE";   break;
  case TT_TEXTURE_CUBE_ARRAY :name = "TEXTURE_CUBE_ARRAY";  break;
  case TT_TEXTURE_BUFFER :    name = "TEXTURE_BUFFER";      break;
  default:                      name = "TEXTURE_UNSPECIFIED"; break;
  }
  fprintf( fh, "%s%s%s", prefix, name.c_str(), postfix );
}

DPAFSaverSharedPtr DPAFSaver::create()
{
  return( std::shared_ptr<DPAFSaver>( new DPAFSaver() ) );
}

DPAFSaver::DPAFSaver()
{
}

DPAFSaver::~DPAFSaver()
{
}

bool  DPAFSaver::save( SceneSharedPtr const& scene, dp::sg::ui::ViewStateSharedPtr const& viewState, string const& filename )
{
  // set the locale temporarily to the default "C" to make printf behave predictably
  TempLocale tl("C");

  FILE *fh = fopen( filename.c_str(), "w" );
  if ( fh )
  {
    DPAFSaveTraverser saver;

    int s = setvbuf( fh, NULL, _IOFBF, 64 * BUFSIZ );
    DP_ASSERT( s == 0 );
    saver.setFILE( fh, filename );
    saver.setViewState( viewState );
    saver.apply( scene );
    fclose( fh );
  }
  else
  {
    INVOKE_CALLBACK( onInvalidFile( filename, "Failed to open file for writing.") );
  }

  return( !!fh );
}

DPAFSaveTraverser::DPAFSaveTraverser()
: m_nameCount(0)
{
  m_basePaths.resize( 2 );
  m_basePaths[1] = dp::home() + std::string( "/" );
  dp::util::convertPath( m_basePaths[1] );
}

void  DPAFSaveTraverser::setFILE( FILE *fh, std::string const& filename )
{
  m_fh = fh;

  m_basePaths[0] = dp::util::getFilePath( filename );;
  dp::util::convertPath( m_basePaths[0] );
}

void  DPAFSaveTraverser::handleBillboard( const Billboard *p )
{
  if ( isFirstTime( p ) )
  {
    string name( getObjectName( p ) );

    SharedTraverser::handleBillboard( p );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "Billboard\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "Billboard\t%s\n{\n", name.c_str() );
      groupData( p );
      Vec3f rotationAxis = p->getRotationAxis();
      fprintfAlignment( m_fh, "\talignment\t", p->getAlignment() );
      writeVector<3,float>( m_fh, "\trotationAxis\t", rotationAxis );
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handleGeoNode( const GeoNode *p )
{
  if ( isFirstTime( p ) )
  {
    string name( getObjectName( p ) );

    SharedTraverser::handleGeoNode( p );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "GeoNode\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "GeoNode\t%s\n{\n", name.c_str() );
      nodeData( p );
      if ( p->getMaterialEffect() )
      {
        fprintf( m_fh, "\tmaterialEffect\t%s\n", m_objectNames[p->getMaterialEffect().getWeakPtr()].c_str() );
      }
      if ( p->getPrimitive() )
      {
        fprintf( m_fh, "\tprimitive\t%s\n", m_objectNames[p->getPrimitive().getWeakPtr()].c_str() );
      }
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handleGroup( const Group * p )
{
  if ( isFirstTime( p ) )
  {
    string name( getObjectName( p ) );
    SharedTraverser::handleGroup( p );
    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "Group\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "Group\t%s\n{\n", name.c_str() );
      groupData( p );
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handleLOD( const LOD *p )
{
  if ( isFirstTime( p ) )
  {
    string name( getObjectName( p ) );

    //  Save _all_ children, not only the active one. Therefore we need to loop ourselves.
    for ( Group::ChildrenConstIterator gcci = p->beginChildren() ; gcci != p->endChildren() ; ++gcci )
    {
      traverseObject( *gcci );
    }

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "LOD\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "LOD\t%s\n{\n", name.c_str() );
      groupData( p );
      writeVector<3,float>( m_fh, "\tcenter\t\t", p->getCenter() );
      if ( p->getNumberOfRanges() )
      {
        fprintf( m_fh, "\tranges" );
        fprintfFloat1Array( m_fh, "\t\t", p->getNumberOfRanges(), p->getRanges(), sizeof(float) );
      }
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handleParallelCamera( const ParallelCamera *p )
{
  if ( isFirstTime( p ) )
  {
    SharedTraverser::handleParallelCamera( p );

    string name( getObjectName( p ) );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "ParallelCamera\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "ParallelCamera\t%s\n{\n", name.c_str() );
      frustumCameraData( p );
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handlePerspectiveCamera( const PerspectiveCamera *p )
{
  if ( isFirstTime( p ) )
  {
    SharedTraverser::handlePerspectiveCamera( p );

    string name( getObjectName( p ) );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "PerspectiveCamera\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "PerspectiveCamera\t%s\n{\n", name.c_str() );
      frustumCameraData( p );
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handleMatrixCamera( const MatrixCamera * p )
{
  if ( isFirstTime( p ) )
  {
    SharedTraverser::handleMatrixCamera( p );

    string name( getObjectName( p ) );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "MatrixCamera\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "MatrixCamera\t%s\n{\n", name.c_str() );
      cameraData( p );
      fprintfMat44f( m_fh, "\tprojectionMatrix \t", "\t                \t", p->getProjection() );
      fprintfMat44f( m_fh, "\tinverseMatrix    \t", "\t                \t", p->getInverseProjection() );
      fprintfMat44f( m_fh, "\tworldToViewMatrix\t", "\t                \t", p->getWorldToViewMatrix() );
      fprintfMat44f( m_fh, "\tviewToWorldMatrix\t", "\t                \t", p->getViewToWorldMatrix() );
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handleLightSource( const LightSource * p )
{
  if ( isFirstTime( p ) )
  {
    string name( getObjectName( p ) );

    SharedTraverser::handleLightSource( p );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "LightSource\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "LightSource\t%s\n{\n", name.c_str() );
      lightSourceData( p );
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handleSwitch( const Switch *p )
{
  if ( isFirstTime( p ) )
  {
    string name( getObjectName( p ) );

    //  Save _all_ children, not only the active ones. Therefore we need to loop ourselves.
    for ( Group::ChildrenConstIterator gcci = p->beginChildren() ; gcci != p->endChildren() ; ++gcci )
    {
      traverseObject( *gcci );
    }

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "Switch\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "Switch\t%s\n{\n", name.c_str() );
      groupData( p );
      if ( p->isActive() )
      {
        vector<unsigned int> actives;
        unsigned int numActives = p->getActive(actives);

        if ( numActives )
        {
          unsigned int i;
          fprintf( m_fh, "\tactive\t\t[ " );
          for ( i=0 ; i<numActives-1 ; i++ )
          {
            fprintf( m_fh, "%u, ", actives[i] );
          }
          fprintf( m_fh, "%u", actives[i] );
          fprintf( m_fh, " ]\n");
        }
      }
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handleTransform( const Transform *p )
{
  if ( isFirstTime( p ) )
  {
    string name( getObjectName( p ) );

    SharedTraverser::handleTransform( p );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "Transform\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "Transform\t%s\n{\n", name.c_str() );
      transformData( p );
      fprintf( m_fh, "}\n\n" );
    }
  }
}

void  DPAFSaveTraverser::handlePrimitive( const Primitive *p )
{
  string name;

  if ( isFirstTime( p ) )
  {
    name = getObjectName( p );

    SharedTraverser::handlePrimitive( p );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "Primitive\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "Primitive\t%s\n{\n", name.c_str() );
      primitiveData( p );
      fprintf( m_fh, "}\n\n" );
    }
  }
}

template <typename T> inline void
fprintfIndexData( FILE * fh, const T * ptr, unsigned int numElements )
{
  const unsigned int cols = 16;
  unsigned int rows = numElements / cols;
  unsigned int idx = 0;
  for ( unsigned int i=0 ; i<rows ; i++ )
  {
    fprintf( fh, "\t\t" );
    for ( unsigned int j=0 ; j<cols ; j++, idx++ )
    {
      fprintf( fh, " %d ", ptr[idx] );
    }
    fprintf( fh, "\n" );
  }
  if ( idx < numElements )
  {
    fprintf( fh, "\t\t" );
    for (  ; idx<numElements ; idx++ )
    {
      fprintf( fh, " %d ", ptr[idx] );
    }
    fprintf( fh, "\n" );
  }
}

void DPAFSaveTraverser::handleIndexSet( const IndexSet * p )
{
  if ( isFirstTime( p ) )
  {
    string name = getObjectName( p );

    SharedTraverser::handleIndexSet( p );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "IndexSet\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "IndexSet\t%s\n{\n", name.c_str() );
      fprintf( m_fh, "\tnumberOfIndices\t%u\n",       p->getNumberOfIndices() );
      fprintf( m_fh, "\tdataType\t%d\n",              p->getIndexDataType() );
      fprintf( m_fh, "\tprimitiveRestartIndex\t%u\n", p->getPrimitiveRestartIndex() );
      fprintf( m_fh, "\tindices\t[\n" );
      Buffer::DataReadLock reader( p->getBuffer() );
      switch( p->getIndexDataType() )
      {
        case dp::util::DT_UNSIGNED_INT_32:
          fprintfIndexData( m_fh, reader.getPtr<unsigned int>(), p->getNumberOfIndices() );
          break;
        case dp::util::DT_UNSIGNED_INT_16:
          fprintfIndexData( m_fh, reader.getPtr<unsigned short>(), p->getNumberOfIndices() );
          break;
        case dp::util::DT_UNSIGNED_INT_8:
          fprintfIndexData( m_fh, reader.getPtr<unsigned char>(), p->getNumberOfIndices() );
          break;
        default:
          DP_ASSERT(!"invalid data type!");
          break;
      }
      fprintf( m_fh, "\t]\n");
      fprintf( m_fh, "}\n\n" );
    }
  }
  else
  {
    DP_ASSERT(m_objectNames.find(getWeakPtr<Object>(p))!=m_objectNames.end());
  }
}

void  DPAFSaveTraverser::handleVertexAttributeSet( const VertexAttributeSet *p )
{
  if ( isFirstTime( p ) )
  {

    for ( unsigned int i=0; i<VertexAttributeSet::DP_SG_VERTEX_ATTRIB_COUNT; ++i )
    {
      if ( p->getSizeOfVertexData(i) )
      {
        buffer( p->getVertexBuffer(i).getWeakPtr() );
      }
    }

    string name( getObjectName( p ) );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "VertexAttributeSet\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "VertexAttributeSet\t%s\n{\n", name.c_str() );
      vertexAttributeSetData(p);
      fprintf( m_fh, "}\n\n" );
    }
  }
  else
  {
    DP_ASSERT(m_objectNames.find(getWeakPtr<Object>(p))!=m_objectNames.end());
  }
}

void DPAFSaveTraverser::handleEffectData( const EffectData * p )
{
  if ( isFirstTime( p ) )
  {
    const dp::fx::EffectSpecSharedPtr & es = p->getEffectSpec();
    m_effectSpecName = es->getName();

    SharedTraverser::handleEffectData( p );

    string name( getObjectName( p ) );
    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "EffectData\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "EffectData\t%s\n{\n", name.c_str() );
      objectData( p );
      fprintf( m_fh, "\teffectFile\t%s\n", dp::util::makePathRelative( dp::fx::EffectLibrary::instance()->getEffectFile( m_effectSpecName ), m_basePaths ).c_str() );
      fprintf( m_fh, "\teffectSpec\t%s\n", m_effectSpecName.c_str() );
      fprintf( m_fh, "\tparameterGroupData\n\t[\n" );
      for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
      {
        const ParameterGroupDataSharedPtr & pgd = p->getParameterGroupData( it );
        if ( pgd )
        {
          DP_ASSERT( m_objectNames.find( pgd.getWeakPtr() ) != m_objectNames.end() );
          fprintf( m_fh, "\t\t%s\n", m_objectNames[pgd.getWeakPtr()].c_str() );
        }
      }
      fprintf( m_fh, "\t]\n" );
      fprintfBool( m_fh, "\ttransparent\t", p->getTransparent() );
      fprintf( m_fh, "}\n\n" );
    }
  }
  else
  {
    DP_ASSERT(m_objectNames.find(getWeakPtr<Object>(p))!=m_objectNames.end());
  }
}

template<typename T>
void parameterOut( std::ostringstream & ost, const T & v )
{
  ost << v << " ";
}

template<>
void parameterOut( std::ostringstream & ost, const bool & v )
{
  ost << ( v ? "TRUE " : "FALSE " );
}

template<>
void parameterOut( std::ostringstream & ost, const char & v )
{
  ost << (int)v << " ";
}

template<>
void parameterOut( std::ostringstream & ost, const unsigned char & v )
{
  ost << (unsigned int)v << " ";
}

template<unsigned int n, typename T>
string parameterStringVNT( const Vecnt<n,T> & v )
{
  std::ostringstream ost;
  ost << "( ";
  for ( unsigned int i=0 ; i<n ; i++ )
  {
    parameterOut<T>( ost, v[i] );
  }
  ost << ")";
  return( ost.str() );
}

template<unsigned int n, typename T>
string parameterStringVNT( const ParameterGroupData * p, dp::fx::ParameterGroupSpec::iterator it )
{
  std::ostringstream ost;
  if ( it->first.getArraySize() )
  {
    vector<Vecnt<n,T> > data;
    p->getParameterArray( it, data );
    ost << "[ ";
    for ( size_t i=0 ; i<data.size() ; i++ )
    {
      ost << parameterStringVNT( data[i] ) << " ";
    }
    ost << "]";
  }
  else
  {
    ost << parameterStringVNT( p->getParameter<Vecnt<n,T> >( it ) );
  }
  return( ost.str() );
}

template<unsigned int m, unsigned int n, typename T>
string parameterStringMNT( const Matmnt<m,n,T> & mat )
{
  std::ostringstream ost;
  ost << "( ";
  for ( unsigned int i=0 ; i<m ; i++ )
  {
    ost << "( ";
    for ( unsigned int j=0 ; j<n ; j++ )
    {
      parameterOut<T>( ost, mat[i][j] );
    }
    ost << ") ";
  }
  ost << ")";
  return( ost.str() );
}

template<unsigned int m, unsigned int n, typename T>
string parameterStringMNT( const ParameterGroupData * p, dp::fx::ParameterGroupSpec::iterator it )
{
  std::ostringstream ost;
  if ( it->first.getArraySize() )
  {
    vector<Matmnt<m,n,T> > data;
    p->getParameterArray( it, data );
    ost << "[ ";
    for ( size_t i=0 ; i<data.size() ; i++ )
    {
      ost << parameterStringMNT( data[i] ) << " ";
    }
    ost << "]";
  }
  else
  {
    ost << parameterStringMNT( p->getParameter<Matmnt<m,n,T> >( it ) );
  }
  return( ost.str() );
}

template<typename T>
string parameterStringT( const ParameterGroupData * p, dp::fx::ParameterGroupSpec::iterator it )
{
  unsigned int type = it->first.getType();
  DP_ASSERT( type & dp::fx::PT_SCALAR_TYPE_MASK );
  if ( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
  {
    switch( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
    {
      case dp::fx::PT_VECTOR2 :
        return( parameterStringVNT<2,T>( p, it ) );
        break;
      case dp::fx::PT_VECTOR3 :
        return( parameterStringVNT<3,T>( p, it ) );
        break;
      case dp::fx::PT_VECTOR4 :
        return( parameterStringVNT<4,T>( p, it ) );
        break;
      case dp::fx::PT_MATRIX2x2 :
        return( parameterStringMNT<2,2,T>( p, it ) );
        break;
      case dp::fx::PT_MATRIX2x3 :
        return( parameterStringMNT<2,3,T>( p, it ) );
        break;
      case dp::fx::PT_MATRIX2x4 :
        return( parameterStringMNT<2,4,T>( p, it ) );
        break;
      case dp::fx::PT_MATRIX3x2 :
        return( parameterStringMNT<3,2,T>( p, it ) );
        break;
      case dp::fx::PT_MATRIX3x3 :
        return( parameterStringMNT<3,3,T>( p, it ) );
        break;
      case dp::fx::PT_MATRIX3x4 :
        return( parameterStringMNT<3,4,T>( p, it ) );
        break;
      case dp::fx::PT_MATRIX4x2 :
        return( parameterStringMNT<4,2,T>( p, it ) );
        break;
      case dp::fx::PT_MATRIX4x3 :
        return( parameterStringMNT<4,3,T>( p, it ) );
        break;
      case dp::fx::PT_MATRIX4x4 :
        return( parameterStringMNT<4,4,T>( p, it ) );
        break;
      default :
        DP_ASSERT( false );
        return( "" );
        break;
    }
  }
  else
  {
    std::ostringstream ost;
    if ( it->first.getArraySize() )
    {
      vector<T> data;
      p->getParameterArray( it, data );
      ost << "[ ";
      for ( size_t i=0 ; i<data.size() ; i++ )
      {
        parameterOut( ost, data[i] );
      }
      ost << "]";
    }
    else
    {
      parameterOut( ost, p->getParameter<T>( it ) );
    }
    return( ost.str() );
  }
}

template<unsigned int n>
string parameterStringEnumVN( const Vecnt<n,char> & v, dp::fx::ParameterGroupSpec::iterator it )
{
  const dp::fx::EnumSpecSharedPtr & enumSpec = it->first.getEnumSpec();
  DP_ASSERT( enumSpec );
  std::ostringstream ost;
  ost << "( ";
  for ( unsigned int i=0 ; i<n ; i++ )
  {
    ost << enumSpec->getValueName( v[i] ) << " ";
  }
  ost << ")";
  return( ost.str() );
}

template<unsigned int n>
string parameterStringEnumVN( const ParameterGroupData * p, dp::fx::ParameterGroupSpec::iterator it )
{
  std::ostringstream ost;
  if ( it->first.getArraySize() )
  {
    vector<Vecnt<n,char> > data;
    p->getParameterArray( it, data );
    ost << "[ ";
    for ( size_t i=0 ; i<data.size() ; i++ )
    {
      ost << parameterStringEnumVN( data[i], it ) << " ";
    }
    ost << "]";
  }
  else
  {
    ost << parameterStringEnumVN( p->getParameter<Vecnt<n,char> >( it ), it );
  }
  return( ost.str() );
}

template<unsigned int m, unsigned int n>
string parameterStringEnumMN( const Matmnt<m,n,char> & mat, dp::fx::ParameterGroupSpec::iterator it )
{
  const dp::fx::EnumSpecSharedPtr & enumSpec = it->first.getEnumSpec();
  DP_ASSERT( enumSpec );
  std::ostringstream ost;
  ost << "( ";
  for ( unsigned int i=0 ; i<m ; i++ )
  {
    ost << "( ";
    for ( unsigned int j=0 ; j<n ; j++ )
    {
      ost << enumSpec->getValueName( mat[i][j] ) << " ";
    }
    ost << ") ";
  }
  ost << ")";
  return( ost.str() );
}

template<unsigned int m, unsigned int n>
string parameterStringEnumMN( const ParameterGroupData * p, dp::fx::ParameterGroupSpec::iterator it )
{
  std::ostringstream ost;
  if ( it->first.getArraySize() )
  {
    vector<Matmnt<m,n,char> > data;
    p->getParameterArray( it, data );
    ost << "[ ";
    for ( size_t i=0 ; i<data.size() ; i++ )
    {
      ost << parameterStringEnumMN( data[i], it ) << " ";
    }
    ost << "]";
  }
  else
  {
    ost << parameterStringEnumMN( p->getParameter<Matmnt<m,n,char> >( it ), it );
  }
  return( ost.str() );
}

string parameterStringEnum( const ParameterGroupData * p, dp::fx::ParameterGroupSpec::iterator it )
{
  unsigned int type = it->first.getType();
  DP_ASSERT( type & dp::fx::PT_SCALAR_TYPE_MASK );
  if ( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
  {
    switch( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
    {
      case dp::fx::PT_VECTOR2 :
        return( parameterStringEnumVN<2>( p, it ) );
        break;
      case dp::fx::PT_VECTOR3 :
        return( parameterStringEnumVN<3>( p, it ) );
        break;
      case dp::fx::PT_VECTOR4 :
        return( parameterStringEnumVN<4>( p, it ) );
        break;
      case dp::fx::PT_MATRIX2x2 :
        return( parameterStringEnumMN<2,2>( p, it ) );
        break;
      case dp::fx::PT_MATRIX2x3 :
        return( parameterStringEnumMN<2,3>( p, it ) );
        break;
      case dp::fx::PT_MATRIX2x4 :
        return( parameterStringEnumMN<2,4>( p, it ) );
        break;
      case dp::fx::PT_MATRIX3x2 :
        return( parameterStringEnumMN<3,2>( p, it ) );
        break;
      case dp::fx::PT_MATRIX3x3 :
        return( parameterStringEnumMN<3,3>( p, it ) );
        break;
      case dp::fx::PT_MATRIX3x4 :
        return( parameterStringEnumMN<3,4>( p, it ) );
        break;
      case dp::fx::PT_MATRIX4x2 :
        return( parameterStringEnumMN<4,2>( p, it ) );
        break;
      case dp::fx::PT_MATRIX4x3 :
        return( parameterStringEnumMN<4,3>( p, it ) );
        break;
      case dp::fx::PT_MATRIX4x4 :
        return( parameterStringEnumMN<4,4>( p, it ) );
        break;
      default :
        DP_ASSERT( false );
        return( "" );
        break;
    }
  }
  else
  {
    std::ostringstream ost;
    const dp::fx::EnumSpecSharedPtr & enumSpec = it->first.getEnumSpec();
    DP_ASSERT( enumSpec );
    if ( it->first.getArraySize() )
    {
      vector<int> data;
      p->getParameterArray( it, data );
      ost << "[ ";
      for ( size_t i=0 ; i<data.size() ; i++ )
      {
        ost << enumSpec->getValueName( data[i] ) << " ";
      }
      ost << "]";
    }
    else
    {
      ost << enumSpec->getValueName( p->getParameter<dp::fx::EnumSpec::StorageType>( it ) ) << " ";
    }
    return( ost.str() );
  }
}

string DPAFSaveTraverser::parameterString( const ParameterGroupData * p, dp::fx::ParameterGroupSpec::iterator it )
{
  unsigned int type = it->first.getType();
  if ( type & dp::fx::PT_SCALAR_TYPE_MASK )
  {
    switch( type & dp::fx::PT_SCALAR_TYPE_MASK )
    {
      case dp::fx::PT_BOOL :
        return( parameterStringT<bool>( p, it ) );
        break;
      case dp::fx::PT_ENUM :
        return( parameterStringEnum( p, it ) );
        break;
      case dp::fx::PT_INT8 :
        return( parameterStringT<char>( p, it ) );
        break;
      case dp::fx::PT_UINT8 :
        return( parameterStringT<unsigned char>( p, it ) );
        break;
      case dp::fx::PT_INT16 :
        return( parameterStringT<short>( p, it ) );
        break;
      case dp::fx::PT_UINT16 :
        return( parameterStringT<unsigned short>( p, it ) );
        break;
      case dp::fx::PT_FLOAT32 :
        return( parameterStringT<float>( p, it ) );
        break;
      case dp::fx::PT_INT32 :
        return( parameterStringT<int>( p, it ) );
        break;
      case dp::fx::PT_UINT32 :
        return( parameterStringT<unsigned int>( p, it ) );
        break;
      case dp::fx::PT_FLOAT64 :
        return( parameterStringT<double>( p, it ) );
        break;
      case dp::fx::PT_INT64 :
        return( parameterStringT<long long>( p, it ) );
        break;
      case dp::fx::PT_UINT64 :
        return( parameterStringT<unsigned long long>( p, it ) );
        break;
      default :
        DP_ASSERT( false );
        break;
    }
  }
  else
  {
    DP_ASSERT( type & dp::fx::PT_POINTER_TYPE_MASK );
    switch( type & dp::fx::PT_POINTER_TYPE_MASK )
    {
      case dp::fx::PT_BUFFER_PTR :
        {
          const BufferSharedPtr & buffer = p->getParameter<BufferSharedPtr>( it );
          DP_ASSERT( buffer );
          DP_ASSERT( m_storedBuffers.find( buffer.getWeakPtr() ) != m_storedBuffers.end() );
          return( m_storedBuffers[buffer.getWeakPtr()] );
        }
        break;
      case dp::fx::PT_SAMPLER_PTR :
        {
          const SamplerSharedPtr & sampler = p->getParameter<SamplerSharedPtr>( it );
          DP_ASSERT( sampler );
          DP_ASSERT( m_storedSamplers.find( sampler.getWeakPtr() ) != m_storedSamplers.end() );
          return( m_storedSamplers[sampler.getWeakPtr()] );
        }
        break;
      default :
        DP_ASSERT( false );
        break;
    }
  }
  return( "" );
}

void DPAFSaveTraverser::handleParameterGroupData( const ParameterGroupData * p )
{
  if ( isFirstTime( p ) )
  {
    SharedTraverser::handleParameterGroupData( p );

    string name( getObjectName( p ) );
    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "ParameterGroupData\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }
      fprintf( m_fh, "ParameterGroupData\t%s\n{\n", name.c_str() );
      objectData( p );
      fprintf( m_fh, "\teffectFile\t%s\n", dp::util::makePathRelative( dp::fx::EffectLibrary::instance()->getEffectFile( m_effectSpecName ), m_basePaths ).c_str() );
      const dp::fx::ParameterGroupSpecSharedPtr & pgs = p->getParameterGroupSpec();
      fprintf( m_fh, "\tparameterGroupSpec\t%s %s\n", m_effectSpecName.c_str(), pgs->getName().c_str() );
      fprintf( m_fh, "\tparameters\n\t[\n" );
      for ( dp::fx::ParameterGroupSpec::iterator it = pgs->beginParameterSpecs() ; it != pgs->endParameterSpecs() ; ++it )
      {
        fprintf( m_fh, "\t\t%s\t%s\n", it->first.getName().c_str(), parameterString( p, it ).c_str() );
      }
      fprintf( m_fh, "\t]\n" );
      fprintf( m_fh, "}\n\n" );
    }
  }
  else
  {
    DP_ASSERT(m_objectNames.find(getWeakPtr<Object>(p))!=m_objectNames.end());
  }
}

void DPAFSaveTraverser::handleSampler( const Sampler * p )
{
  if ( isFirstTime( p ) )
  {
    string name( getObjectName( p ) );

    SharedTraverser::handleSampler( p );

    if ( p->isDataShared() && ( m_sharedData.find( p->getDataID() ) != m_sharedData.end() ) )
    {
      fprintf( m_fh, "Sampler\t%s\t%s\n\n", name.c_str(), m_sharedData[p->getDataID()].c_str() );
    }
    else
    {
      if ( p->isDataShared() )
      {
        m_sharedData[p->getDataID()] = name;
      }

      // first handle TextureHosts not defined by filename
      const TextureSharedPtr & texture = p->getTexture();
      if ( texture && texture.isPtrTo<TextureHost>() )
      {
        TextureHostSharedPtr const& textureHost = texture.staticCast<TextureHost>();
        if ( textureHost->getFileName().empty() )
        {
          textureImage( textureHost );
        }
      }

      // then the Sampler itself
      fprintf( m_fh, "Sampler\t%s\n{\n", name.c_str() );
      objectData( p );
      if ( texture && texture.isPtrTo<TextureHost>() )
      {
        TextureHostSharedPtr const& textureHost = texture.staticCast<TextureHost>();
        if ( textureHost->getFileName().empty() )
        {
          DP_ASSERT( m_textureImageNames.find(textureHost.getWeakPtr()) != m_textureImageNames.end() );
          fprintf( m_fh, "\ttextureImage\t%s\n", m_textureImageNames[textureHost.getWeakPtr()].c_str() );
        }
        else
        {
          fprintf( m_fh, "\tcreationFlags\t0x%x\n", textureHost->getCreationFlags() );
          fprintfTextureTarget(m_fh, "\ttextureTarget\t", textureHost->getTextureTarget(), "\n" );
          fprintf( m_fh, "\ttextureImage\t%s\n", getName( textureHost->getFileName() ).c_str() );
        }
      }
      writeVector<4,float>( m_fh, "\tborderColor\t", p->getBorderColor() );
      fprintfMagFilterMode( m_fh, "\tmagFilterMode\t", p->getMagFilterMode() );
      fprintfMinFilterMode( m_fh, "\tminFilterMode\t", p->getMinFilterMode() );
      fprintfWrapModes( m_fh, "\twrapMode\t", p->getWrapModeS(), p->getWrapModeT(), p->getWrapModeR() );
      fprintfCompareMode( m_fh, "\tcompareMode\t", p->getCompareMode() );
      fprintf( m_fh, "}\n\n" );

      DP_ASSERT( m_storedSamplers.find( p ) == m_storedSamplers.end() );
      m_storedSamplers.insert( std::pair<const Sampler *, std::string>( p, name ) );
    }
  }
  else
  {
    DP_ASSERT(m_objectNames.find(getWeakPtr<Object>(p))!=m_objectNames.end());
  }
}

string typeToString( unsigned int type )
{
  std::ostringstream ost;
  if ( type & dp::fx::PT_SCALAR_TYPE_MASK )
  {
    switch( type & dp::fx::PT_SCALAR_TYPE_MASK )
    {
      case dp::fx::PT_BOOL :
        ost << "bool";
        break;
      case dp::fx::PT_ENUM :
        ost << "enum";
        break;
      case dp::fx::PT_INT8 :
        ost << "char";
        break;
      case dp::fx::PT_UINT8 :
        ost << "uchar";
        break;
      case dp::fx::PT_INT16 :
        ost << "short";
        break;
      case dp::fx::PT_UINT16 :
        ost << "ushort";
        break;
      case dp::fx::PT_FLOAT32 :
        ost << "float";
        break;
      case dp::fx::PT_INT32 :
        ost << "int";
        break;
      case dp::fx::PT_UINT32 :
        ost << "uint";
        break;
      case dp::fx::PT_FLOAT64 :
        ost << "double";
        break;
      case dp::fx::PT_INT64 :
        ost << "longlong";
        break;
      case dp::fx::PT_UINT64 :
        ost << "ulonglong";
        break;
      default :
        DP_ASSERT( false );
        ost << "unknown scalar type";
        break;
    }
    if ( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
    {
      DP_ASSERT( type & dp::fx::PT_SCALAR_TYPE_MASK );
      switch( type & dp::fx::PT_SCALAR_MODIFIER_MASK )
      {
        case dp::fx::PT_VECTOR2 :
          ost << "2";
          break;
        case dp::fx::PT_VECTOR3 :
          ost << "3";
          break;
        case dp::fx::PT_VECTOR4 :
          ost << "4";
          break;
        case dp::fx::PT_MATRIX2x2 :
          ost << "2x2";
          break;
        case dp::fx::PT_MATRIX2x3 :
          ost << "2x3";
          break;
        case dp::fx::PT_MATRIX2x4 :
          ost << "2x4";
          break;
        case dp::fx::PT_MATRIX3x2 :
          ost << "3x2";
          break;
        case dp::fx::PT_MATRIX3x3 :
          ost << "3x3";
          break;
        case dp::fx::PT_MATRIX3x4 :
          ost << "3x4";
          break;
        case dp::fx::PT_MATRIX4x2 :
          ost << "4x2";
          break;
        case dp::fx::PT_MATRIX4x3 :
          ost << "4x3";
          break;
        case dp::fx::PT_MATRIX4x4 :
          ost << "4x4";
          break;
        default :
          DP_ASSERT( false );
          ost << "unknown scalar modifier type " << type;
      }
    }
  }
  else
  {
    DP_ASSERT( type & dp::fx::PT_POINTER_TYPE_MASK );
    switch( type & dp::fx::PT_POINTER_TYPE_MASK )
    {
      case dp::fx::PT_BUFFER_PTR :
        ost << "Buffer";
        break;
      case dp::fx::PT_SAMPLER_PTR :
        ost << "Sampler";
        break;
      default :
        DP_ASSERT( false );
        ost << "unknown texture type " << type;
        break;
    }
  }
  return( ost.str() );
}

void DPAFSaveTraverser::doApply( const NodeSharedPtr & root )
{
  DP_ASSERT( m_scene && "This traverser needs a valid Scene. Use setScene() prior calling apply()");
  DP_ASSERT( m_scene->getRootNode() == root );
  DP_ASSERT( m_storedBuffers.empty() && m_storedSamplers.empty() );

  initUnnamedCounters();

  fprintf( m_fh, "#DPAF V1.0\n\n" );

  SharedTraverser::doApply( root );
  for ( Scene::CameraIterator scci = m_scene->beginCameras() ; scci != m_scene->endCameras() ; ++scci )
  {
    traverseObject( *scci );
  }
  TextureHostSharedPtr backImage = m_scene->getBackImage();
  if ( backImage )
  {
    if ( backImage->getFileName().empty() )
    {
      textureImage( backImage );
    }
  }
  fprintf( m_fh, "Scene\n{\n" );
  writeVector<3,float>( m_fh, "\tambientColor\t\t", m_scene->getAmbientColor() );
  writeVector<4,float>( m_fh, "\tbackColor\t\t", m_scene->getBackColor() );
  if ( backImage )
  {
    fprintf( m_fh, "\tbackImage\t%s\n", backImage->getFileName().empty()
      ? m_textureImageNames[backImage.getWeakPtr()].c_str()
      : getName( backImage->getFileName() ).c_str() );
  }
  if ( m_scene->getNumberOfCameras() )
  {
    if ( m_scene->getNumberOfCameras() == 1 )
    {
      fprintf( m_fh, "\tcameras\t\t\t[ %s ]\n", m_objectNames[m_scene->beginCameras()->getWeakPtr()].c_str() );
    }
    else
    {
      fprintf( m_fh, "\tcameras\t\t\t[\n" );
      for ( Scene::CameraIterator scci = m_scene->beginCameras() ; scci != m_scene->endCameras() ; ++scci )
      {
        fprintf( m_fh, "\t       \t\t\t \t%s\n", m_objectNames[scci->getWeakPtr()].c_str() );
      }
      fprintf( m_fh, "\t       \t\t\t]\n" );
    }
  }
  if ( m_scene->getRootNode() )
  {
    fprintf( m_fh, "\troot\t\t\t%s\n", m_objectNames[m_scene->getRootNode().getWeakPtr()].c_str() );
  }


  fprintf( m_fh, "}\n\n" );

  if ( m_viewState )
  {
    DP_ASSERT( m_viewState->getCamera() );
    traverseObject(m_viewState->getCamera());

    fprintf( m_fh, "ViewState\n{\n" );
    fprintfBool( m_fh, "\tautoClip\t", m_viewState->getAutoClipPlanes() );
    fprintf( m_fh, "\tcamera\t%s\n", m_objectNames[m_viewState->getCamera().getWeakPtr()].c_str() );
    fprintfBool( m_fh, "\tstereoState\t", false );//m_viewState->getRenderTarget() ? m_viewState->getRenderTarget()->isStereoEnabled() : false );
    fprintfBool( m_fh, "\tstereoAutomaticEyeDistanceAdjustment\t", m_viewState->isStereoAutomaticEyeDistanceAdjustment() );
    fprintf( m_fh, "\tstereoAutomaticEyeDistanceFactor\t%f\n", m_viewState->getStereoAutomaticEyeDistanceFactor() );
    fprintf( m_fh, "\tstereoEyeDistance\t%f\n", m_viewState->getStereoEyeDistance() );
    fprintf( m_fh, "\ttargetDistance\t%f\n", m_viewState->getTargetDistance() );
    fprintf( m_fh, "}\n\n" );
  }

  //  write out the links
  if ( ! m_links.empty() )
  {
    fprintf( m_fh, "Links\n{\n" );
    for ( unsigned int i=0 ; i<m_links.size() ; i++ )
    {
      DP_ASSERT( m_objectNames.find( m_links[i].subject ) != m_objectNames.end() );
      DP_ASSERT( m_objectNames.find( m_links[i].observer ) != m_objectNames.end() );
      fprintf( m_fh, "\t%s %s, %s;\n" , m_links[i].name.c_str()
        , m_objectNames[m_links[i].subject].c_str()
        , m_objectNames[m_links[i].observer].c_str() );
    }
    fprintf( m_fh, "}\n\n" );
    m_links.clear();
  }

  m_storedBuffers.clear();
  m_storedSamplers.clear();
}

void  DPAFSaveTraverser::cameraData( const Camera *p )
{
  objectData( p );
  writeVector<3,float>( m_fh, "\tdirection\t", p->getDirection() );
  fprintf( m_fh, "\tfocusDistance\t%f\n", p->getFocusDistance() );
  if ( p->getNumberOfHeadLights() )
  {
    Vec3f trans;
    Quatf rot;
    fprintf( m_fh, "\theadLights\t[\n" );
    for ( Camera::HeadLightConstIterator hlci = p->beginHeadLights() ; hlci != p->endHeadLights() ; ++hlci )
    {
      fprintf( m_fh, "\t          \t %s\n", m_objectNames[hlci->getWeakPtr()].c_str() );
    }
    fprintf( m_fh, "\t          \t]\n" );
  }
  writeVector<3,float>( m_fh, "\tposition\t", p->getPosition() );
  writeVector<3,float>( m_fh, "\tupVector\t", p->getUpVector() );
}

void DPAFSaveTraverser::frustumCameraData( const FrustumCamera *p )
{
  cameraData( p );
  fprintf( m_fh, "\tfarDistance\t%f\n", p->getFarDistance() );
  fprintf( m_fh, "\tnearDistance\t%f\n", p->getNearDistance() );
  writeVector<2,float>( m_fh, "\twindowOffset\t", p->getWindowOffset() );
  writeVector<2,float>( m_fh, "\twindowSize\t", p->getWindowSize() );
}

const string  DPAFSaveTraverser::getName( const string &name )
{
  return( ( name.find( " " ) != string::npos ) ? "\"" + name + "\"" : name );
}

string  DPAFSaveTraverser::getObjectName( const Object *p )
{
  // Each object with no name, or with a non-unique name, gets the string "_dpaf#" appended, where
  // "#" stands for a unique number. That way, each saved objects will have it's own unique name in
  // the stored file.
  // The DPAFLoader, on the other side, will remove the "_dpaf#" part from each objects name, such
  // that the original (empty or non-unique) name is reconstructed.
  string name = p->getName();
  if ( name.empty() || m_nameSet.insert( name ).second )
  {
    // if there's no name, or that same name has been encountered before, decorate it
    static const char * postfix = "_dpaf";
    std::ostringstream ost;
    if ( name.find_first_of( " ," ) != string::npos )
    {
      // there's a name containing space or comma -> wrap it with ""
      ost << "\"" << name.c_str() << postfix << m_nameCount++ << "\"";
    }
    else
    {
      ost << name.c_str() << postfix << m_nameCount++;
    }
    name = ost.str();
  }
  else if ( name.find_first_of( " ," ) != string::npos )
  {
    // there's a name containing space or comma -> wrap it with ""
    std::ostringstream ost;
    ost << "\"" << name.c_str() << "\"";
    name = ost.str();
  }
  DP_VERIFY( m_objectNames.insert( std::make_pair( getWeakPtr<Object>(p), name ) ).second );
  return( name );
}

void  DPAFSaveTraverser::objectData( const Object * p )
{
  if ( !p->getAnnotation().empty() )
  {
    fprintf(m_fh, "\tannotation\t\"%s\"\n", p->getAnnotation().c_str());
  }

  // don't write them out unless they have value
  unsigned int hints = p->getHints();
  if( hints )
  {
    fprintf(m_fh, "\thints\t\t0x%08x\n", hints );
  }
}

void  DPAFSaveTraverser::groupData( const Group *p )
{
  nodeData( p );
  if ( p->getNumberOfClipPlanes() )
  {
    if ( p->getNumberOfClipPlanes() == 1 )
    {
      ClipPlaneSharedPtr const& plane = *p->beginClipPlanes();
      fprintf( m_fh, "\tclipPlanes\t[ ( %f, %f, %f ), %f, %s ]\n"
        , plane->getNormal()[0], plane->getNormal()[1], plane->getNormal()[2]
      , plane->getOffset(), plane->isEnabled() ? "TRUE" : "FALSE" );
    }
    else
    {
      fprintf( m_fh, "\tclipPlanes\t[\n" );
      for ( Group::ClipPlaneConstIterator gcpci = p->beginClipPlanes() ; gcpci != p->endClipPlanes() ; ++gcpci )
      {
        ClipPlaneSharedPtr const& plane = *gcpci;
        fprintf( m_fh, "\t          \t \t( %f, %f, %f ), %f, %s\n"
          , plane->getNormal()[0], plane->getNormal()[1], plane->getNormal()[2]
        , plane->getOffset(), plane->isEnabled() ? "TRUE" : "FALSE" );
      }
      fprintf( m_fh, "\t          \t]\n" );
    }
  }
  if ( p->getNumberOfChildren() )
  {
    if ( p->getNumberOfChildren() == 1 )
    {
      fprintf( m_fh, "\tchildren\t[ %s ]\n", m_objectNames[p->beginChildren()->getWeakPtr()].c_str() );
    }
    else
    {
      fprintf( m_fh, "\tchildren\t[\n" );
      for ( Group::ChildrenConstIterator gcci = p->beginChildren() ; gcci != p->endChildren() ; ++gcci )
      {
        fprintf( m_fh, "\t        \t \t%s\n", m_objectNames[gcci->getWeakPtr()].c_str() );
      }
      fprintf( m_fh, "\t        \t]\n" );
    }
  }
}

void  DPAFSaveTraverser::primitiveData( const Primitive *p )
{
  objectData( p );

  // primitive data
  fprintf( m_fh, "\tprimitiveType\t%s", primitiveTypeToName( p->getPrimitiveType() ).c_str() );
  if ( p->getPrimitiveType() == PRIMITIVE_PATCHES )
  {
    fprintf( m_fh, "\tpatchesType\t%s", patchesTypeToName( p->getPatchesType() ).c_str() );
    fprintfPatchesMode( m_fh, "\tpatchesMode\t", p->getPatchesMode() );
    fprintfPatchesOrdering( m_fh, "\tpatchesOrdering\t", p->getPatchesOrdering() );
    fprintfPatchesSpacing( m_fh, "\tpatchesSpacing\t", p->getPatchesSpacing() );
  }
  fprintf( m_fh, "\tinstanceCount %u\n",       p->getInstanceCount() ); 
  fprintf( m_fh, "\tvertexAttributeSet\t%s\n", m_objectNames[p->getVertexAttributeSet().getWeakPtr()].c_str() );
  if( p->isIndexed() )
  {
    fprintf( m_fh, "\tindexSet\t%s\n", m_objectNames[p->getIndexSet().getWeakPtr()].c_str() );
  }

  // set these after setting VAS and IndexSet so that they are read in the proper order
  unsigned int offset;
  unsigned int count;
  p->getElementRange( offset, count ); // Make sure to write the user values, count may be ~0 here.

  fprintf( m_fh, "\telementOffset %u\n", offset ); 
  fprintf( m_fh, "\telementCount %u\n",  count ); 
}

void DPAFSaveTraverser::initUnnamedCounters( void )
{
  m_objectNames.clear();
  m_nameSet.clear();
  m_sharedObjects.clear();
  m_nameCount = 0;
}

bool  DPAFSaveTraverser::isFirstTime( const HandledObject *p )
{
  return( m_sharedObjects.insert( p ).second );
}

void  DPAFSaveTraverser::lightSourceData( const LightSource *p )
{
  objectData( p );
  fprintfBool(m_fh, "\tenabled\t", p->isEnabled() );
  fprintfBool( m_fh, "\tshadowCasting\t", p->isShadowCasting() );
  if ( p->getLightEffect() )
  {
    fprintf( m_fh, "\tlightEffect\t%s\n", m_objectNames[p->getLightEffect().getWeakPtr()].c_str() );
  }
}

void  DPAFSaveTraverser::nodeData( const Node *p )
{
  // nothing left here really, but may as well leave the function in
  objectData( p );
}

void  DPAFSaveTraverser::textureImage( const TextureHostSharedPtr & tih )
{
  DP_ASSERT( tih );
  static int count = 0;

  if ( m_textureImageNames.find( tih.getWeakPtr() ) == m_textureImageNames.end() )
  {
    std::ostringstream ost;
    ost << "_dpafTextureHost" << count++;
    string name(ost.str());
    m_textureImageNames[tih.getWeakPtr()] = name;

    fprintf( m_fh, "TextureHost\t%s\n{\n", name.c_str() );
    fprintf( m_fh, "\tcreationFlags\t0x%x\n", tih->getCreationFlags() );
    fprintfTextureTarget( m_fh, "\ttextureTarget\t\t", tih->getTextureTarget(), "\n" );

    if ( ! tih->isImageStream() )
    {
      fprintf( m_fh, "\timages\n\t[\n" );
      for ( unsigned int i=0 ; i<tih->getNumberOfImages() ; i++ )
      {
        fprintf( m_fh, "\t\t{\n" );
        fprintfPixelFormat( m_fh, "\t\t\tpixelFormat\t", tih->getFormat(i,0) );
        fprintfPixelType( m_fh, "\t\t\tpixelType\t", tih->getType(i,0) );
        fprintf( m_fh, "\t\t\twidth\t%u\n", tih->getWidth(i,0) );
        fprintf( m_fh, "\t\t\theight\t%u\n", tih->getHeight(i,0) );
        fprintf( m_fh, "\t\t\tdepth\t%u\n", tih->getDepth(i,0) );
        fprintf( m_fh, "\t\t\tpixels\n\t\t\t[\n" );
        fprintfPixels( m_fh, "\t\t\t\t", tih, i, 0 );
        fprintf( m_fh, "\t\t\t]\n" );
        if ( 0 < tih->getNumberOfMipmaps() )
        {
          fprintf( m_fh, "\t\t\tmipmaps\n\t\t\t{\n" );
          for ( unsigned int j=0 ; j<=tih->getNumberOfMipmaps(i) ; j++ )
          {
            fprintf( m_fh, "\t\t\t\t[\n" );
            fprintfPixels( m_fh, "\t\t\t\t\t", tih, i, j );
            fprintf( m_fh, "\t\t\t\t]\n" );
          }
          fprintf( m_fh, "\t\t\t}\n" );
        }
        fprintf( m_fh, "\t\t}\n" );
      }
    }
    fprintf( m_fh, "\t]\n}\n\n" );
  }
}

void DPAFSaveTraverser::transformData( const Transform * p )
{
  groupData( p );

  const Trafo & trafo = p->getTrafo();
  writeVector<3,float>( m_fh, "\tcenter\t", trafo.getCenter() );
  fprintfQuatf( m_fh, "\torientation\t", "", trafo.getOrientation() );
  fprintfQuatf( m_fh, "\tscaleOrientation\t", "", trafo.getScaleOrientation() );
  writeVector<3,float>( m_fh, "\tscaling\t", trafo.getScaling() );
  writeVector<3,float>( m_fh, "\ttranslation\t", trafo.getTranslation() );
}

void DPAFSaveTraverser::vertexAttributeSetData( const VertexAttributeSet * p )
{
  objectData( p );

  unsigned int enableFlags = 0;
  unsigned int normalizeFlags =0;

  for ( unsigned int i=0; i<VertexAttributeSet::DP_SG_VERTEX_ATTRIB_COUNT; ++i )
  {
    enableFlags |= (p->isEnabled(i)<<i) | (p->isEnabled(i+16)<<(i+16));
    normalizeFlags |= (p->isNormalizeEnabled(i+16)<<(i+16));
    if ( p->getSizeOfVertexData(i) )
    {
      unsigned int count = p->getNumberOfVertexData(i);
      DP_ASSERT( p->getNumberOfVertexData(i) <= UINT_MAX );
      DP_ASSERT( m_storedBuffers.find( p->getVertexBuffer(i).getWeakPtr() ) != m_storedBuffers.end() );

      fprintf( m_fh, "\tvattrib%u\t%u\t%u\t%s\t%u\t%u\t%u\n", i, 
        p->getSizeOfVertexData(i), p->getTypeOfVertexData(i)
        , m_storedBuffers[ p->getVertexBuffer(i).getWeakPtr() ].c_str()
        , p->getOffsetOfVertexData(i), p->getStrideOfVertexData(i)
        , count);
    }
  }
  fprintf( m_fh, "\tenableFlags\t0x%X\n", enableFlags);  
  fprintf( m_fh, "\tnormalizeFlags\t0x%X\n", normalizeFlags);  
}

void DPAFSaveTraverser::buffer( const Buffer *p )
{
  DP_ASSERT( p->getSize() <= UINT_MAX );

  if ( m_storedBuffers.find(p) == m_storedBuffers.end() )
  {
    std::ostringstream name;
    name << "buffer_" << m_nameCount++;
    fprintf( m_fh, "Buffer %s\n{\n\ttype host\n\tdata", name.str().c_str() );
    fprintfArray( m_fh, dp::util::DT_UNSIGNED_INT_8, 1, checked_cast<unsigned int>(p->getSize()), Buffer::DataReadLock( p->getSharedPtr<Buffer>() ), 1, "\t\t" );
    fprintf( m_fh, "}\n\n" );
    m_storedBuffers.insert( std::pair<const Buffer *, std::string>( p, name.str() ) );
  }
}
