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


#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/BufferHost.h>
#include <dp/util/File.h>
#if defined(HAVE_HALF_FLOAT)
#include <dp/math/half.h>
#endif
#if defined(LINUX)
# include <climits>
#endif
#include <cstdint>
#include <iterator>

using namespace dp::math;
  
namespace dp
{
  namespace sg
  {
    namespace core
    {

      // pixel data memory management helpers
      struct PixelMemHdr
      {
        unsigned char * ptr;
        unsigned int size;
      };

      static TextureTarget determineTextureTarget( const TextureHost * th )
      {
        // !FALLBACK! if not unique (e.g. 1x1x1)
        // assuming 2D since 2D is the most common target used
        if ( th->is2D() )
        {
          return TT_TEXTURE_2D;
        }

        if ( th->is1D() )
        {
          return TT_TEXTURE_1D;
        }

        if ( th->is3D() )
        {
          return TT_TEXTURE_3D;
        }

        if ( th->isCubeMap() )
        {
          return TT_TEXTURE_CUBE;
        }

        if ( th->is1DArray() )
        {
          return TT_TEXTURE_1D_ARRAY;
        }

        if ( th->is2DArray() )
        {
          return TT_TEXTURE_2D_ARRAY;
        }

        return TT_UNSPECIFIED_TEXTURE_TARGET;
      }

      TextureTarget determineTextureTarget( const TextureHostSharedPtr & himg )
      {
        if ( himg )
        {
          return( determineTextureTarget( himg ) );
        }
        return TT_UNSPECIFIED_TEXTURE_TARGET;
      }

      //
      // These two methods check for formats and types that may cause problems
      // with the scaling routines.
      //
      inline bool isProblematicType( Image::PixelDataType type ) 
      {
        return ( type == Image::IMG_UNKNOWN_TYPE || 
                 type == Image::IMG_UNSIGNED_INT_2_10_10_10 || 
                 type == Image::IMG_UNSIGNED_INT_5_9_9_9 || 
                 type == Image::IMG_UNSIGNED_INT_10F_11F_11F || 
                 type == Image::IMG_UNSIGNED_INT_24_8 );
      }

      inline bool isProblematicFormat( Image::PixelFormat fmt )
      {
        return ( fmt == Image::IMG_COLOR_INDEX
              || fmt == Image::IMG_COMPRESSED_LUMINANCE_LATC1         
              || fmt == Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1  
              || fmt == Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2   
              || fmt == Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2 
              || fmt == Image::IMG_COMPRESSED_RED_RGTC1 
              || fmt == Image::IMG_COMPRESSED_SIGNED_RED_RGTC1 
              || fmt == Image::IMG_COMPRESSED_RG_RGTC2 
              || fmt == Image::IMG_COMPRESSED_SIGNED_RG_RGTC2 
       
              || fmt == Image::IMG_COMPRESSED_RGB_DXT1 
              || fmt == Image::IMG_COMPRESSED_RGBA_DXT1 
              || fmt == Image::IMG_COMPRESSED_RGBA_DXT3 
              || fmt == Image::IMG_COMPRESSED_RGBA_DXT5 
       
              || fmt == Image::IMG_COMPRESSED_SRGB_DXT1 
              || fmt == Image::IMG_COMPRESSED_SRGBA_DXT1 
              || fmt == Image::IMG_COMPRESSED_SRGBA_DXT3 
              || fmt == Image::IMG_COMPRESSED_SRGBA_DXT5 
              || fmt == Image::IMG_DEPTH_COMPONENT 
              || fmt == Image::IMG_DEPTH_STENCIL 
              || fmt == Image::IMG_UNKNOWN_FORMAT );
      }

      //  local helper function to add source pixel index i and corresponding weight to the
      //  contribution vector
      static void _addContribution( std::vector<std::pair<unsigned int,float> > &cont, int i, unsigned int width, float weight )
      {
        //  if the source pixel index i is off the overlaps (below 0 or above width), take the
        //  "mirrored" index
        unsigned int n = ( i < 0 ) ? -i : ( width <= static_cast<unsigned int>(i) ) ? 2*width-i-1 : i;
        DP_ASSERT( n < width );
        cont.push_back( std::pair<unsigned int,float>( n, weight ) );
      }

      static ScaleFilter * _createScaleFilter( unsigned int flags )
      {
        ScaleFilter * sf = NULL;
        switch( flags & TextureHost::F_SCALE_FILTER_MASK )
        {
          case TextureHost::F_SCALE_FILTER_BOX :
            sf = new BoxFilter;
            break;
          case TextureHost::F_SCALE_FILTER_TRIANGLE :
            sf = new TriangleFilter;
            break;
          case TextureHost::F_SCALE_FILTER_BELL :
            sf = new BellFilter;
            break;
          case TextureHost::F_SCALE_FILTER_BSPLINE :
            sf = new BSplineFilter;
            break;
          case TextureHost::F_SCALE_FILTER_LANCZOS3 :
            sf = new Lanczos3Filter;
            break;
          default :
            DP_ASSERT( false );
            sf = new TriangleFilter;
            break;
        }
        return( sf );
      }

    #if 0
      template<typename T>
        T * _pixel( const Image * img, unsigned int x, unsigned int y, unsigned int z, unsigned int c )
      {
        return( (T*)( img->m_pixels + z * img->m_bps + y * img->m_bpl + x * img->m_bpp + c * sizeof(T) ) );
      }
    #else
      template<typename T>
        unsigned int _pixelOffsetElements( const Image * img, unsigned int x, unsigned int y, unsigned int z, unsigned int c )
      {
        return( (z * img->m_bps + y * img->m_bpl + x * img->m_bpp) / sizeof(T) + c );
      }
    #endif

      static unsigned int _sizeOfImagePixels(const unsigned char * pixels)
      {
        DP_ASSERT(pixels);
        PixelMemHdr * hdr = (PixelMemHdr*)pixels-1;
        return hdr->size;
      }

    #if defined(NDEBUG)
      unsigned char * allocImagePixels(unsigned int nbytes)
    #else
      unsigned char * dbgAllocImagePixels(unsigned int nbytes, unsigned int line)
    #endif
      {
        static unsigned int dbgAllocImagePixelsCount(0);
        const uintptr_t align = 7; // 8-byte alignment
        nbytes = (nbytes+align)&~align; // always allocate a multiple of 8 bytes

        unsigned char * ptr = new unsigned char[nbytes+align+sizeof(PixelMemHdr)]; 
        if ( ptr )
        {
          // start of pixel memory should be aligned on a 8-byte boundary
          unsigned char * pixels = (unsigned char*)(((uintptr_t)ptr+align+sizeof(PixelMemHdr))&~align); 
          DP_ASSERT(!((uintptr_t)pixels & align));
          // 
    #if !defined(NDEBUG)
          memset(pixels, 'N', nbytes);
    #endif

          // store header right in front of pixel memory
          PixelMemHdr * hdr = (PixelMemHdr*)pixels-1;
          hdr->ptr = ptr;
          hdr->size = nbytes;

          return pixels;
        }
        return NULL;
      }

      void freeImagePixels(unsigned char * pixels)
      {
        DP_ASSERT(pixels);
        PixelMemHdr * hdr = (PixelMemHdr*)pixels-1;
        delete[] hdr->ptr;
      }

      unsigned int numberOfComponents( Image::PixelFormat format )
      {
        unsigned int noc;
        switch( format )
        {
          case Image::IMG_COLOR_INDEX :
          case Image::IMG_LUMINANCE :
          case Image::IMG_ALPHA :
          case Image::IMG_DEPTH_COMPONENT :
          case Image::IMG_DEPTH_STENCIL :
          case Image::IMG_INTEGER_LUMINANCE :
          case Image::IMG_COMPRESSED_LUMINANCE_LATC1 :
          case Image::IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1 :
          case Image::IMG_COMPRESSED_RED_RGTC1 :
          case Image::IMG_COMPRESSED_SIGNED_RED_RGTC1 :
            noc = 1;
            break;

          case Image::IMG_LUMINANCE_ALPHA :
          case Image::IMG_INTEGER_LUMINANCE_ALPHA :
          case Image::IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2 :
          case Image::IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2 :
          case Image::IMG_COMPRESSED_RG_RGTC2 :
          case Image::IMG_COMPRESSED_SIGNED_RG_RGTC2 :
            noc = 2;
            break;

          case Image::IMG_RGB :
          case Image::IMG_BGR :
          case Image::IMG_INTEGER_RGB :
          case Image::IMG_INTEGER_BGR :
          case Image::IMG_COMPRESSED_RGB_DXT1 :
          case Image::IMG_COMPRESSED_SRGB_DXT1 :
            noc = 3;
            break;

          case Image::IMG_RGBA :
          case Image::IMG_BGRA :
          case Image::IMG_COMPRESSED_RGBA_DXT1 :
          case Image::IMG_COMPRESSED_RGBA_DXT3 :
          case Image::IMG_COMPRESSED_RGBA_DXT5 :
          case Image::IMG_COMPRESSED_SRGBA_DXT1 :
          case Image::IMG_COMPRESSED_SRGBA_DXT3 :
          case Image::IMG_COMPRESSED_SRGBA_DXT5 :
            noc = 4;
            break;

          default :
            DP_ASSERT( false );
            noc = 0;
            break;
        }
        return( noc );
      }

      unsigned int sizeOfComponents( Image::PixelDataType type )
      {
        unsigned int soc;
        switch( type )
        {
          case Image::IMG_BYTE :
          case Image::IMG_UNSIGNED_BYTE :
            soc = sizeof(char);
            break;

          case Image::IMG_SHORT :
          case Image::IMG_UNSIGNED_SHORT :
            soc = sizeof(short);
            break;

          case Image::IMG_INT :
          case Image::IMG_UNSIGNED_INT :
          case Image::IMG_UNSIGNED_INT_2_10_10_10 :
          case Image::IMG_UNSIGNED_INT_5_9_9_9 :
          case Image::IMG_UNSIGNED_INT_10F_11F_11F :
          case Image::IMG_UNSIGNED_INT_24_8 :
            soc = sizeof(int);
            break;

          case Image::IMG_FLOAT :
            soc = sizeof(float);
            break;

          case Image::IMG_HALF :
            soc = sizeof(float)/2;
            break;

          default:
            DP_ASSERT( false );
            soc = 0;
            break;
        }
        return( soc );
      }

      inline unsigned int numberOfMipmaps(unsigned int w, unsigned int h, unsigned int d)
      {
        unsigned int bits = max( w, h, d );
        unsigned int i=1;
        while (bits >>= 1) ++i;
        return i;
      }

      Image::Image()
        : m_width(0)
        , m_height(0)
        , m_depth(0)
        , m_format(IMG_UNKNOWN_FORMAT)
        , m_type(IMG_UNKNOWN_TYPE)
        , m_bpp(0)
        , m_bpl(0)
        , m_bps(0)
        , m_nob(0)
        , m_pixels(0)
      {
      }

      Image::Image( unsigned int width, unsigned int height, unsigned int depth, PixelFormat format
                  , PixelDataType type )
        : m_width(width)
        , m_height(height)
        , m_depth(depth)
        , m_format(format)
        , m_type(type)
        , m_pixels( NULL )
      {
        m_bpp = numberOfComponents( m_format ) * sizeOfComponents( m_type );
        m_bpl = m_width * m_bpp;
        m_bps = m_height * m_bpl;
        m_nob = m_depth * m_bps;
      }

      Image::Image( const Image &rhs )
        : m_width(rhs.m_width)
        , m_height(rhs.m_height)
        , m_depth(rhs.m_depth)
        , m_format(rhs.m_format)
        , m_type(rhs.m_type)
        , m_bpp(rhs.m_bpp)
        , m_bpl(rhs.m_bpl)
        , m_bps(rhs.m_bps)
        , m_nob(rhs.m_nob)
        , m_pixels(rhs.m_pixels)
      {
      }

      Image &Image::operator=( const Image &rhs )
      {
        m_width = rhs.m_width;
        m_height = rhs.m_height;
        m_depth = rhs.m_depth;
        m_format = rhs.m_format;
        m_type = rhs.m_type;
        m_bpp = rhs.m_bpp;
        m_bpl = rhs.m_bpl;
        m_bps = rhs.m_bps;
        m_nob = rhs.m_nob;
        m_pixels = rhs.m_pixels;

        return *this;
      }

      void calculateContributions( std::vector<std::vector<std::pair<unsigned int,float> > > &contribs, unsigned int srcSize
                                 , unsigned int dstSize, const ScaleFilter * filter )
      {
        contribs.resize( dstSize );
        if ( srcSize == dstSize )
        {
          //  if src and dst are of same size, just add one weight of value 1.0f for each contribution
          //  element.
          for ( unsigned int i=0 ; i<dstSize ; i++ )
          {
            contribs[i].push_back( std::pair<unsigned int,float>( i, 1.0f ) );
          }
        }
        else
        {
          float dstPerSrc = (float)dstSize / (float)srcSize;
          float srcPerDst = 1.0f / dstPerSrc;
          if ( dstPerSrc < 1.0f )
          {
            //  less destination texels than source texels => downscaling
            float width = filter->getSupportWidth() * srcPerDst;  // filter support width in source space
            for ( unsigned int i=0 ; i<dstSize ; i++ )
            {
              //  for each destination texel determine one std::vector of contributions
              float center = (i+0.5f) * srcPerDst - 0.5f;     // position of destination texel in source space
              int left = (int)ceil( center - width );         // first (lower) source texel to consider
              int right = (int)floor( center + width );       // last (upper) source texel to consider
              contribs[i].reserve( right - left + 1 );
              for ( int j=left ; j<=right ; j++ )
              {
                //  for each source texel to consider, determine a contribution info
                //  call the filter with the distance between destination texel and source texel in
                //  destination space
                _addContribution( contribs[i], j, srcSize
                                , (*filter)( abs( center - (float)j ) * dstPerSrc ) * dstPerSrc );
              }
            }
          }
          else
          {
            //  more destination texels than source texels => upscaling
            float width = filter->getSupportWidth();  // filter support width just taken as if in source space
            for ( unsigned int i=0 ; i<dstSize ; i++ )
            {
              //  for each destination texel determine one vector of contributions
              float center = (i+0.5f) * srcPerDst - 0.5f;     // position of destination texel in source space
              int left = (int)ceil( center - width );     // first (lower) source texel to consider
              int right = (int)floor( center + width );   // last (upper) source texel to consider
              for ( int j=left ; j<=right ; j++ )
              {
                //  for each source texel to consider, determine a contribution info
                //  call the filter with the distance between destination texel and source texel as if
                //  in destination space
                _addContribution( contribs[i], j, srcSize, (*filter)( abs( center - (float)j ) ) );
              }
            }
          }
        }
      }

      template<typename T>
        void _insert( const Image * src, Image * dst, unsigned int x, unsigned int y, unsigned int z )
      {
        Buffer::DataReadLock  bufferSrc(src->m_pixels);
        Buffer::DataWriteLock bufferDst(dst->m_pixels, Buffer::MAP_WRITE);

        const T *srcData = bufferSrc.getPtr<T>();
        T *dstData = bufferDst.getPtr<T>();
        for ( unsigned int srcZ=0, dstZ=z ; srcZ<src->m_depth ; srcZ++, dstZ++ )
        {
          for ( unsigned int srcY=0, dstY=y ; srcY<src->m_height ; srcY++, dstY++ )
          {
            memcpy( dstData + _pixelOffsetElements<T>( dst, x, dstY, dstZ, 0 )
                  , srcData + _pixelOffsetElements<T>( src, 0, srcY, srcZ, 0 ), src->m_bpl );
          }
        }
      }

      void insert( const Image * src, Image * dst, unsigned int x, unsigned int y, unsigned int z )
      {
        DP_ASSERT(    ( x + src->m_width  <= dst->m_width )
                    &&  ( y + src->m_height <= dst->m_height )
                    &&  ( z + src->m_depth  <= dst->m_depth ) );
        switch( src->m_type )
        {
          case Image::IMG_UNSIGNED_BYTE:
            _insert<unsigned char>( src, dst, x, y, z );
            break;
          case Image::IMG_BYTE:                                 
            _insert<char>( src, dst, x, y, z );
            break;
          case Image::IMG_UNSIGNED_SHORT:
            _insert<unsigned short>( src, dst, x, y, z );
            break;
          case Image::IMG_SHORT:
            _insert<short>( src, dst, x, y, z );
            break;
          case Image::IMG_UNSIGNED_INT:
          case Image::IMG_UNSIGNED_INT_2_10_10_10:
          case Image::IMG_UNSIGNED_INT_5_9_9_9:
          case Image::IMG_UNSIGNED_INT_10F_11F_11F:
          case Image::IMG_UNSIGNED_INT_24_8:
            _insert<unsigned int>( src, dst, x, y, z );
            break;
          case Image::IMG_INT:
            _insert<int>( src, dst, x, y, z );
            break;
          case Image::IMG_FLOAT:
            _insert<float>( src, dst, x, y, z );
            break;
#if defined(HAVE_HALF_FLOAT)
          case Image::IMG_HALF:
            _insert<half>( src, dst, x, y, z );
            break;
#endif
          default:
            DP_ASSERT(false);
            break;
        }
      }

      template<typename T>
        void _rescale( const Image * src, Image * dst, std::vector<std::vector<std::pair<unsigned int,float> > > * contributions )
      {
        unsigned int numChannels = src->m_bpp / sizeof( T );
        DP_ASSERT(numChannels<=4);

        Buffer::DataReadLock  bufferSrc(src->m_pixels);
        Buffer::DataWriteLock bufferDst(dst->m_pixels, Buffer::MAP_WRITE);

        const T *srcData = bufferSrc.getPtr<T>();
        T *dstData = bufferDst.getPtr<T>();

        for ( unsigned int z=0 ; z<dst->m_depth ; z++ )
        {
          for ( unsigned int y=0 ; y<dst->m_height ; y++ )
          {
            for ( unsigned int x=0 ; x<dst->m_width ; x++ )
            {
              for ( unsigned int c=0 ; c<numChannels ; c++ )
              {
                //  for each value in the destination texture, determine it's value by considering all
                //  contributing source texture values
                float dstValue = 0.0f;
                float w = 0.0f;
                for ( size_t k=0 ; k<contributions[2][z].size() ; k++ )
                {
                  for ( size_t j=0 ; j<contributions[1][y].size() ; j++ )
                  {
                    for ( size_t i=0 ; i<contributions[0][x].size() ; i++ )
                    {
                      float v =   contributions[0][x][i].second
                                * contributions[1][y][j].second
                                * contributions[2][z][k].second;
                      dstValue += srcData[_pixelOffsetElements<T>( src, contributions[0][x][i].first
                                                 , contributions[1][y][j].first
                                                 , contributions[2][z][k].first, c )] * v;
                      w += v;
                    }
                  }
                }
                //  set the destination texel value, normalized (is this normalizing really necessary?)
                dstData[_pixelOffsetElements<T>( dst, x, y, z, c )] = T(dstValue/w);
              }
            }
          }
        }
      }

      void rescale( const Image * src, Image * dst, std::vector<std::vector<std::pair<unsigned int,float> > > * contributions )
      {
        switch( src->m_type )
        {
          case Image::IMG_UNSIGNED_BYTE:
            _rescale<unsigned char>( src, dst, contributions );
            break;
          case Image::IMG_BYTE:                                 
            _rescale<char>( src, dst, contributions );
            break;
          case Image::IMG_UNSIGNED_SHORT:
            _rescale<unsigned short>( src, dst, contributions );
            break;
          case Image::IMG_SHORT:
            _rescale<short>( src, dst, contributions );
            break;
          case Image::IMG_UNSIGNED_INT:
          case Image::IMG_UNSIGNED_INT_2_10_10_10:
          case Image::IMG_UNSIGNED_INT_5_9_9_9:
          case Image::IMG_UNSIGNED_INT_10F_11F_11F:
          case Image::IMG_UNSIGNED_INT_24_8:
            _rescale<unsigned int>( src, dst, contributions );
            break;
          case Image::IMG_INT:
            _rescale<int>( src, dst, contributions );
            break;
          case Image::IMG_FLOAT:
            _rescale<float>( src, dst, contributions );
            break;
#if defined(HAVE_HALF_FLOAT)
          case Image::IMG_HALF:
            _rescale<half>( src, dst, contributions );
            break;
#endif
          default:
            DP_ASSERT(false);
            break;
        }
      }

      template<typename T>
      void _convertPixelFormat( const Image * src, Image * dst, Image::PixelFormat format )
      {
        unsigned int srcChannels = numberOfComponents( src->m_format );
        unsigned int dstChannels = numberOfComponents( format );

        if( isProblematicFormat( format ) )
        {
          // give a warning in debug
          DP_ASSERT( !"UNABLE TO CONVERT TO THE GIVEN FORMAT!" );
          return;
        }

        if( isProblematicFormat( src->m_format ) )
        {
          // give a warning in debug
          DP_ASSERT( !"UNABLE TO CONVERT FROM THE GIVEN FORMAT!" );
          return;
        }

        Buffer::DataReadLock  bufferSrc(src->m_pixels);
        Buffer::DataWriteLock bufferDst(dst->m_pixels, Buffer::MAP_WRITE);

        const T *srcData = bufferSrc.getPtr<T>();
        T *dstData = bufferDst.getPtr<T>();

        for ( unsigned int z=0 ; z<dst->m_depth ; z++ )
        {
          for ( unsigned int y=0 ; y<dst->m_height ; y++ )
          {
            for ( unsigned int x=0 ; x<dst->m_width ; x++ )
            {
              T component[4];

              for ( unsigned int c=0 ; c<srcChannels ; c++ )
              {
                component[c] = srcData[_pixelOffsetElements<T>( src, x, y, z, c )];
              }

              // slow and klunky but works
              switch( src->m_format )
              {
                case Image::IMG_RGB:            //!< RGB format
                case Image::IMG_RGBA:           //!< RGBA format
                case Image::IMG_INTEGER_RGB:    // treat integer as same
                case Image::IMG_INTEGER_RGBA:
                  switch( format )
                  {
                  case Image::IMG_RGB:          //!< RGB format
                  case Image::IMG_INTEGER_RGB:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[1];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[2];
                    break;
                  case Image::IMG_RGBA:         //!< RGBA format
                  case Image::IMG_INTEGER_RGBA:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[1];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[2];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 3 )] = component[3];
                    break;
                  case Image::IMG_BGR:          //!< BGR format
                  case Image::IMG_INTEGER_BGR:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[2];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[1];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[0];
                    break;
                  case Image::IMG_BGRA:         //!< BGRA format
                  case Image::IMG_INTEGER_BGRA:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[2];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[1];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 3 )] = component[3];
                    break;
                  case Image::IMG_LUMINANCE:    //!< luminance format
                  case Image::IMG_INTEGER_LUMINANCE: // correct?
                    {
                      T pix = (component[0]+component[1]+component[2])/3;
                      dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = pix;
                    }
                    break;
                  case Image::IMG_LUMINANCE_ALPHA:  //!< luminance alpha format
                    {
                      T pix = (component[0]+component[1]+component[2])/3;
                      dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = pix;
                    }
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[3];
                    break;

                  case Image::IMG_ALPHA:
                  case Image::IMG_INTEGER_ALPHA:
                    {
                      T pix = component[0];
                      dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = pix;
                    }
                    break;
                  }
                  break;

                case Image::IMG_BGR:          //!< BGR format
                case Image::IMG_BGRA:         //!< BGRA format
                  switch( format )
                  {
                  case Image::IMG_RGB:        //!< RGB format
                  case Image::IMG_INTEGER_RGB:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[2];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[1];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[0];
                    break;
                  case Image::IMG_RGBA:       //!< RGBA format
                  case Image::IMG_INTEGER_RGBA:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[2];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[1];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 3 )] = component[3];
                    break;
                  case Image::IMG_BGR:        //!< BGR format
                  case Image::IMG_INTEGER_BGR:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[1];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[2];
                    break;
                  case Image::IMG_BGRA:       //!< BGRA format
                  case Image::IMG_INTEGER_BGRA:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[1];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[2];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 3 )] = component[3];
                    break;
                  case Image::IMG_LUMINANCE:  //!< luminance format
                  case Image::IMG_INTEGER_LUMINANCE:
                    {
                      T pix = (component[0]+component[1]+component[2])/3;
                      dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = pix;
                    }
                    break;
                  case Image::IMG_LUMINANCE_ALPHA:  //!< luminance alpha format
                  case Image::IMG_INTEGER_LUMINANCE_ALPHA:
                    {
                      T pix = (component[0]+component[1]+component[2])/3;
                      dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = pix;
                    }
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[3];
                    break;
                  case Image::IMG_ALPHA:
                  case Image::IMG_INTEGER_ALPHA:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[0];
                    break;
                  }
                  break;

                case Image::IMG_ALPHA:
                case Image::IMG_INTEGER_ALPHA:
                case Image::IMG_LUMINANCE:        //!< luminance format
                case Image::IMG_LUMINANCE_ALPHA:  //!< luminance alpha format
                  switch( format )
                  {
                  case Image::IMG_BGR:            //!< BGR format
                  case Image::IMG_RGB:            //!< RGB format
                  case Image::IMG_INTEGER_BGR:
                  case Image::IMG_INTEGER_RGB:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[0];
                    break;
                  case Image::IMG_BGRA:           //!< BGRA format
                  case Image::IMG_RGBA:           //!< RGBA format
                  case Image::IMG_INTEGER_BGRA:
                  case Image::IMG_INTEGER_RGBA:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 2 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 3 )] = component[1];
                    break;
                  case Image::IMG_LUMINANCE_ALPHA:  //!< luminance alpha format
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[0];
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 1 )] = component[1];
                    break;
                  case Image::IMG_LUMINANCE:        //!< luminance format
                  case Image::IMG_ALPHA:
                  case Image::IMG_INTEGER_ALPHA:
                    dstData[_pixelOffsetElements<T>( dst, x, y, z, 0 )] = component[0];
                  }
                  break;

                default:
                  DP_ASSERT( src->m_format /* UNKNOWN FORMAT */ );
                  break;
              }
            }
          }
        }

      }

      void cvtPixelFormat( const Image * src, Image * dst, Image::PixelFormat format )
      {
        // we don't convert these pixel formats at the moment.
        DP_ASSERT( isProblematicFormat( format ) == false );
        DP_ASSERT( isProblematicFormat( src->m_format ) == false );

        switch( src->m_type )
        {
          case Image::IMG_UNSIGNED_BYTE:
            _convertPixelFormat<unsigned char>( src, dst, format );
            break;
          case Image::IMG_BYTE:                                 
            _convertPixelFormat<char>( src, dst, format );
            break;
          case Image::IMG_UNSIGNED_SHORT:
            _convertPixelFormat<unsigned short>( src, dst, format );
            break;
          case Image::IMG_SHORT:
            _convertPixelFormat<short>( src, dst, format );
            break;
          case Image::IMG_UNSIGNED_INT:
            _convertPixelFormat<unsigned int>( src, dst, format );
            break;
          case Image::IMG_INT:
            _convertPixelFormat<int>( src, dst, format );
            break;
          case Image::IMG_FLOAT32:
            _convertPixelFormat<float>( src, dst, format );
            break;
#if defined(HAVE_HALF_FLOAT)
          case Image::IMG_FLOAT16:
            _convertPixelFormat<half>( src, dst, format );
            break;
#endif
          default:
            DP_ASSERT(false);
            break;
        }
      }

    #if !defined(NDEBUG)
      void assertMipMapChainComplete(const TextureHost * img)
      {
        DP_ASSERT(img); 
        DP_ASSERT(img->getNumberOfMipmaps() + 1 == numberOfMipmaps(img->getWidth(), img->getHeight(), img->getDepth()));
      }
    #endif

      TextureHostSharedPtr TextureHost::create( const std::string & filename )
      {
        return( std::shared_ptr<TextureHost>( new TextureHost( filename ) ) );
      }

      HandledObjectSharedPtr TextureHost::clone() const
      {
        return( std::shared_ptr<TextureHost>( new TextureHost( *this ) ) );
      }

      TextureHost::TextureHost( const std::string & filename )
      : m_creationFlags(F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD)
      , m_filename(filename)
      , m_internalFlags(0)
      , m_totalNumBytes(0)
      , m_gpuFormat( TGF_DEFAULT )
      {
        DP_ASSERT( !isHashKeyValid() );
      }

      void TextureHost::setCreationFlags(unsigned int flags)
      {
        if ( m_creationFlags != flags )
        {
          m_creationFlags = flags;

          if ( m_creationFlags & F_IMAGE_STREAM )
          {
            // for image streams ... 
            // ... never release image data after upload
            // ... never scale the image
            m_creationFlags |= F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD;
          }
          invalidateHashKey();
        }
      }

      void TextureHost::setTextureGPUFormat( TextureHost::TextureGPUFormat fmt )
      {
        if ( m_gpuFormat != fmt )
        {
          // FIXME find out proper "representation"
          m_gpuFormat = fmt;
          demandUpload(); // change in GPU format requires a re-upload
          invalidateHashKey();
        }
      }

      TextureHost::TextureHost(const TextureHost& rhs)
      : m_creationFlags(rhs.m_creationFlags)
      , m_internalFlags(0)
      , m_totalNumBytes(rhs.m_totalNumBytes)
      , m_filename(rhs.m_filename)
      , m_gpuFormat(rhs.m_gpuFormat)
      {
        copyImages( rhs.m_images );
        DP_ASSERT( !isHashKeyValid() );
      }

      TextureHost::~TextureHost()
      {
      }

      TextureHost& TextureHost::operator=(const TextureHost& rhs)
      {
        // avoid self-assignment
        if ( this != &rhs )
        {
          m_images.clear();

          // don't touch
          // ... m_internalFlags (will be evaluated in copyImages below)

          // copy new contents
          m_creationFlags = rhs.m_creationFlags;  // ... creation flags
          copyImages( rhs.m_images );             // ... image data
          m_filename = rhs.m_filename;            // ... file name
        }

        return *this;
      }

      void TextureHost::copyImages( const std::vector<std::vector<Image> > & srcImages )
      {
        m_images = srcImages;

        bool upload;
        if ( m_creationFlags & F_IMAGE_STREAM )
        {
          upload = true;
          for ( size_t i=0 ; i<srcImages.size() ; i++ )
          {
            for ( size_t j=0 ; j<srcImages[i].size() ; j++ )
            {
              DP_ASSERT( srcImages[i][j].m_pixels );
              m_images[i][j].m_pixels = srcImages[i][j].m_pixels;
            }
          }
        }
        else if ( !srcImages.empty() && !srcImages[0].empty() && srcImages[0][0].m_pixels )
        {
          upload = true;
          for ( size_t i=0 ; i<srcImages.size() ; i++ )
          {
            for ( size_t j=0 ; j<srcImages[i].size() ; j++ )
            {
              DP_ASSERT( srcImages[i][j].m_pixels );
              m_images[i][j].m_pixels = BufferHost::create();
              if ( !m_images[i][j].m_pixels )
              { // image allocation failed
    #if 0
                NVSG_TRACE_OUT("ERROR: image allocation failed!\n");
    #endif
                return;
              }

              {
                BufferSharedPtr const& bufferDst = m_images[i][j].m_pixels;
                bufferDst->setSize(   m_images[i][j].m_nob );
                bufferDst->setData(0, m_images[i][j].m_nob, srcImages[i][j].m_pixels, 0);
              }
            }
          }
        }
        else
        {
          upload = false;
        }

        // force an upload only if valid image data is available
        if ( upload )
        {
          recalcTotalNumberOfBytes();
          demandUpload();
          invalidateHashKey();
        }
      }

      unsigned int TextureHost::addImage( unsigned int width, unsigned int height, unsigned int depth
                                         , Image::PixelFormat format, Image::PixelDataType type )
      {
        DP_ASSERT( ( 0 < width ) && ( 0 < height ) && ( 0 < depth ) );

        m_images.push_back( std::vector<Image>() );
        m_images.back().push_back( Image(width, height, depth, format, type));
        invalidateHashKey();
        return( util::checked_cast<unsigned int>(m_images.size() - 1) ); 
      }

      void TextureHost::setImageData( unsigned int image, const void * pixels
                                     , const std::vector<const void *> & mipmaps )
      {
        DP_ASSERT( image < m_images.size() );

        Image *img = &m_images[image][0];
        if ( m_creationFlags & F_IMAGE_STREAM )
        {
          DP_ASSERT( pixels );
          BufferHostSharedPtr bufferHost = BufferHost::create();
          bufferHost->setUnmanagedDataPtr( const_cast<void *>(pixels) );
          bufferHost->setSize( img->m_nob );
          img->m_pixels = bufferHost;
        }
        else
        {
          if ( ! img->m_pixels )
          {
            img->m_pixels = BufferHost::create();
            if ( !img->m_pixels )
            { // image allocation failed
    #if 0
              NVSG_TRACE_OUT("ERROR: image allocation failed!\n");
    #endif
              return;
            }

            img->m_pixels->setSize( img->m_nob );

            if ( pixels )
            {
              img->m_pixels->setData( 0, img->m_nob, pixels );
            }

            // consider mipmaps passed through mipmaps vector
            if ( !mipmaps.empty() )
            {
              createMipmaps( m_images[image], mipmaps );
            }
          }
        }
        if ( getTextureTarget() == TT_UNSPECIFIED_TEXTURE_TARGET )
        {
          setTextureTarget( determineTextureTarget( this ) );
        }

        recalcTotalNumberOfBytes();
        demandUpload();
        invalidateHashKey();
      }

      void TextureHost::setImageData( unsigned int image, const BufferSharedPtr & pixels, const std::vector<BufferSharedPtr> & mipmaps )
      {
        DP_ASSERT( image < m_images.size() );

        Image *img = &m_images[image][0];
        img->m_pixels = pixels;

        if ( !mipmaps.empty() )
        {
          createMipmaps( m_images[image], mipmaps );
        }
        if ( getTextureTarget() == TT_UNSPECIFIED_TEXTURE_TARGET )
        {
          setTextureTarget( determineTextureTarget( this ) );
        }

        recalcTotalNumberOfBytes();
        demandUpload();
        invalidateHashKey();
      }

      unsigned int TextureHost::createImage( unsigned int            width
                                            , unsigned int            height
                                            , unsigned int            depth
                                            , Image::PixelFormat      format
                                            , Image::PixelDataType    type
                                            , const void            * pixels 
                                            , const std::vector<const void*>& mipmaps )
      {
        DP_ASSERT( ( 0 < width ) && ( 0 < height ) && ( 0 < depth ) );

        unsigned int img = addImage( width, height, depth, format, type );
        setImageData( img, pixels, mipmaps );
        DP_ASSERT( !isHashKeyValid() );
        return( img );
      }

      bool TextureHost::createMipmaps()
      {
        if ( (m_creationFlags & F_IMAGE_STREAM) && // we don't own data of image streams!
             isProblematicFormat(m_images[0][0].m_format) ) // we don't handle these formats right
        {
          // cannot handle
          // -> quit
          return false;
        }

        for ( size_t i=0 ; i<m_images.size() ; i++ )
        {
          if ( ! createMipmaps( m_images[i] ) )
          {
            // can't do with an incomplete chain
            // -> release all previously created mipmaps and quit
            releaseMipmaps();
            return false;
          }
        }

        // mipmap creation went successful if we get here
        recalcTotalNumberOfBytes();
        demandUpload();
        invalidateHashKey();
        return true;
      }

      bool TextureHost::createMipmaps( std::vector<Image> &images, const std::vector<const void*>& mipmaps )
      {
        DP_ASSERT( images.size() );   // must hold at least the top level image
        DP_ASSERT( ! ( m_creationFlags & F_IMAGE_STREAM ) );

        // get rid of old mipmaps, if any
        releaseMipmaps( images );

        // reserve enough space to hold all mipmap levels that will be generated below.
        // this in particular avoids reallocations on push_backs below, which would
        // invalidate the the running src pointer used to reference the respective 
        // source image within the for-loop below.
        unsigned int numMipMaps = numberOfMipmaps(images[0].m_width, images[0].m_height, images[0].m_depth);
        images.reserve(numMipMaps);
    
        Image * src=&images[0];
        for (size_t i=0; i<mipmaps.size() && ( 1<src->m_width || 1<src->m_height || 1<src->m_depth ) ; i++ )
        {
          DP_ASSERT(mipmaps[i]);

          // append the next level image to the images
          images.push_back( Image( 1<src->m_width  ? src->m_width>> 1 : 1
                                 , 1<src->m_height ? src->m_height>>1 : 1
                                 , 1<src->m_depth  ? src->m_depth>> 1 : 1
                                 , src->m_format, src->m_type ) );

          // if this fires, the above push_back invalidated the src pointer through reallocation.
          // this should not happen as we reserve enough images in the vector to hold all mipmap
          // levels.
          DP_ASSERT(src==&images[images.size()-2]);
      
          Image * dst = &images.back(); // points to new created Image

          dst->m_pixels = BufferHost::create();
          if ( !dst->m_pixels )
          { // image allocation failed
    #if 0
            NVSG_TRACE_OUT("ERROR: image allocation failed!\n");
    #endif
            goto ERROREXIT;
          }
          dst->m_pixels->setSize( dst->m_nob );
          dst->m_pixels->setData( 0, dst->m_nob, mipmaps[i] );
          src = dst; // prepare for next level creation
        }

        //  complete the mipmap chain, if there are an inadequate number of mipmaps, and the image
        // is not a F_IMAGE_STREAM
        if ( images.size() < numMipMaps )
        {
          if ( m_creationFlags & F_IMAGE_STREAM )
          {
    #if 0
            NVSG_TRACE_OUT("ERROR: incomplete mipmap chain with F_IMAGE_STREAM set!\n");
    #endif
            goto ERROREXIT;
          }

          if ( isProblematicFormat(images[0].m_format) )
          {
    #if 0
            NVSG_TRACE_OUT("ERROR: incomplete mipmap chain with problematic format!\n");
    #endif
            goto ERROREXIT;
          }

          std::vector<Image> mipmapchain;
          // push the last 'biggest' image from which to reconstruct the missing mipmaps, as the first image
          // in our mipmap chain
          mipmapchain.push_back(images.back());
          createMipmaps(mipmapchain);
          // copy the additional missing mipmaps created
          if ( mipmapchain.size() > 1 )
          {
           copy(mipmapchain.begin() + 1, mipmapchain.end(), back_inserter(images));
          }      
        }

        return true; // succeeded if we get here

    ERROREXIT:
        releaseMipmaps( images );
        return false;
      }

      bool TextureHost::createMipmaps( std::vector<Image> &images, const std::vector<BufferSharedPtr>& mipmaps )
      {
        DP_ASSERT( images.size() );   // must hold at least the top level image
        DP_ASSERT( ! ( m_creationFlags & F_IMAGE_STREAM ) );

        // get rid of old mipmaps, if any
        releaseMipmaps( images );

        // reserve enough space to hold all mipmap levels that will be generated below.
        // this in particular avoids reallocations on push_backs below, which would
        // invalidate the the running src pointer used to reference the respective 
        // source image within the for-loop below.
        unsigned int numMipMaps = numberOfMipmaps(images[0].m_width, images[0].m_height, images[0].m_depth);
        images.reserve(numMipMaps);
    
        Image * src=&images[0];
        for (size_t i=0; i<mipmaps.size() && ( 1<src->m_width || 1<src->m_height || 1<src->m_depth ) ; i++ )
        {
          DP_ASSERT(mipmaps[i]);

          // append the next level image to the images
          images.push_back( Image( 1<src->m_width  ? src->m_width>> 1 : 1
                                 , 1<src->m_height ? src->m_height>>1 : 1
                                 , 1<src->m_depth  ? src->m_depth>> 1 : 1
                                 , src->m_format, src->m_type ) );

          // if this fires, the above push_back invalidated the src pointer through reallocation.
          // this should not happen as we reserve enough images in the vector to hold all mipmap
          // levels.
          DP_ASSERT(src==&images[images.size()-2]);
      
          Image * dst = &images.back(); // points to new created Image

          dst->m_pixels = mipmaps[i];
          src = dst; // prepare for next level creation
        }

        //  complete the mipmap chain, if there are an inadequate number of mipmaps, and the image
        // is not a F_IMAGE_STREAM
        if ( images.size() < numMipMaps )
        {
          if ( m_creationFlags & F_IMAGE_STREAM )
          {
    #if 0
            NVSG_TRACE_OUT("ERROR: incomplete mipmap chain with F_IMAGE_STREAM set!\n");
    #endif
            goto ERROREXIT;
          }

          if ( isProblematicFormat(images[0].m_format) )
          {
    #if 0
            NVSG_TRACE_OUT("ERROR: incomplete mipmap chain with problematic format!\n");
    #endif
            goto ERROREXIT;
          }

          std::vector<Image> mipmapchain;
          // push the last 'biggest' image from which to reconstruct the missing mipmaps, as the first image
          // in our mipmap chain
          mipmapchain.push_back(images.back());
          createMipmaps(mipmapchain);
          // copy the additional missing mipmaps created
          if ( mipmapchain.size() > 1 )
          {
           copy(mipmapchain.begin() + 1, mipmapchain.end(), back_inserter(images));
          }      
        }

        return true; // succeeded if we get here

    ERROREXIT:
        releaseMipmaps( images );
        return false;
      }

      bool TextureHost::createMipmaps( std::vector<Image> &images )
      {
        // never use this function for streams 
        // this should have been caught at higher calling level!
        DP_ASSERT(!(m_creationFlags & F_IMAGE_STREAM));

        // get rid of old mipmaps, if any
        releaseMipmaps( images );

        // reserve enough space to hold all mipmap levels that will be generated below.
        // this in particular avoids reallocations on push_backs below, which would
        // invalidate the the running src pointer used to reference the respective 
        // source image within the for-loop below.
        images.reserve(numberOfMipmaps(images[0].m_width, images[0].m_height, images[0].m_depth));

        // this scale filter will be used for mipmap interpolation below
        ScaleFilter * sf = _createScaleFilter( m_creationFlags );
    
        for( Image * src=&images[0]; 1<src->m_width || 1<src->m_height || 1<src->m_depth ; /**/)
        {
          // append the next level image to the images
          images.push_back( Image( 1<src->m_width  ? src->m_width>> 1 : 1
                                 , 1<src->m_height ? src->m_height>>1 : 1
                                 , 1<src->m_depth  ? src->m_depth>> 1 : 1
                                 , src->m_format, src->m_type ) );

          // if this fires, the above push_back invalidated the src pointer through reallocation.
          // this should not happen as we reserve enough images in the vector to hold all mipmap
          // levels.
          DP_ASSERT(src==&images[images.size()-2]);
      
          Image * dst = &images.back(); // points to new created Image

          dst->m_pixels = BufferHost::create();
          if ( !dst->m_pixels )
          { // image allocation failed
    #if 0
            NVSG_TRACE_OUT("ERROR: image allocation failed!\n");
    #endif
            goto ERROREXIT;
          }
          else
          {
            dst->m_pixels->setSize( dst->m_nob );
          }

          filter( src, dst, sf );
          src = dst; // prepare for next level creation
        }
        delete sf;
        return true; // succeeded if we get here

    ERROREXIT:
        releaseMipmaps( images );
        return false;
      }

      void TextureHost::releaseMipmaps()
      {
        for ( size_t i=0 ; i<m_images.size() ; i++ )
        {
          releaseMipmaps( m_images[i] );
        }
      }

      void TextureHost::releaseMipmaps( std::vector<Image> & images )
      {
        if ( 1 < images.size() )
        {
          DP_ASSERT( ! ( m_creationFlags & F_IMAGE_STREAM ) );
          for ( size_t i=1 ; i<images.size() ; i++ )
          {
            images[i].m_pixels.reset();
          }
          images.resize(1);
        }
      }

      template<typename T>
        void _getSubImagePixels( const Image * srcImg, unsigned int x_offset, unsigned int y_offset
                               , unsigned int z_offset, unsigned int width, unsigned int height
                               , unsigned int depth, void * subPixels )
      {
        Buffer::DataReadLock bufferSrc(srcImg->m_pixels);

        const T *srcData = bufferSrc.getPtr<T>();

        unsigned int spanSize = width * srcImg->m_bpp;
        unsigned char * dst = (unsigned char*)subPixels;
        for ( unsigned int z=z_offset ; z<(z_offset+depth) ; z++ )
        {
          for ( unsigned int y=y_offset ; y<(y_offset+height) ; y++ )
          {
            memcpy( dst, srcData + _pixelOffsetElements<T>( srcImg, x_offset, y, z, 0 ), spanSize );
            dst += spanSize;
          }
        }
      }

  
      bool TextureHost::getSubImagePixels( unsigned int image, unsigned int mipmap, unsigned int x_offset
                                          , unsigned int y_offset, unsigned int z_offset
                                          , unsigned int width, unsigned int height
                                          , unsigned int depth, void * subPixels ) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        DP_ASSERT(subPixels);

        // This check is not really required, since if anything is zero
        // the memcpy will copy zero bytes, anyway, however
        // do no more work than required !
        if ( width == 0 || height == 0 || depth == 0 )
        {
          return true;
        }

        // Get the image
        const Image* srcImg = &m_images[image][mipmap];

        // Check if the sub image boundary off-shoots that of the existing image
        // or there is no pixel data available
        if (    ( (x_offset + width)  > srcImg->m_width )
            ||  ( (y_offset + height) > srcImg->m_height )
            ||  ( (z_offset + depth)  > srcImg->m_depth )
            ||  ! srcImg->m_pixels )
        {
          return false;
        }

        switch( srcImg->m_type )
        {
          case Image::IMG_UNSIGNED_BYTE:
            _getSubImagePixels<unsigned char>(srcImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_BYTE:                                 
            _getSubImagePixels<char>(srcImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_UNSIGNED_SHORT:
            _getSubImagePixels<unsigned short>(srcImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_SHORT:
            _getSubImagePixels<short>(srcImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_UNSIGNED_INT:
          case Image::IMG_UNSIGNED_INT_2_10_10_10:
          case Image::IMG_UNSIGNED_INT_5_9_9_9:
          case Image::IMG_UNSIGNED_INT_10F_11F_11F:
          case Image::IMG_UNSIGNED_INT_24_8:
            _getSubImagePixels<unsigned int>(srcImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_INT:
            _getSubImagePixels<int>(srcImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_FLOAT:
            _getSubImagePixels<float>(srcImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
#if defined(HAVE_HALF_FLOAT)
          case Image::IMG_HALF:
            _getSubImagePixels<half>(srcImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
#endif
          default:
            DP_ASSERT(false);
            break;
        }
        return true;
      }

      bool TextureHost::getSubImagePixels( unsigned int image, unsigned int mipmap, unsigned int x_offset
                                          , unsigned int y_offset, unsigned int z_offset
                                          , unsigned int width, unsigned int height
                                          , unsigned int depth, const BufferSharedPtr &buffer ) const
      {
        // Set new size of buffer
        const Image* srcImg = &m_images[image][mipmap];
        unsigned int bufferSize = width * height * srcImg->m_bpp;
        buffer->setSize(bufferSize);

        Buffer::DataWriteLock bufferLock( buffer, Buffer::MAP_WRITE );
        bool result = getSubImagePixels( image, mipmap, x_offset, y_offset, z_offset, width, height, depth, bufferLock.getPtr() );

        return result;
      }
   
      template<typename T>
        void _setSubImagePixels( Image * dstImg, unsigned int x_offset, unsigned int y_offset
                               , unsigned int z_offset, unsigned int width, unsigned int height
                               , unsigned int depth, const void * subPixels )
      {
        Buffer::DataWriteLock bufferDst(dstImg->m_pixels, Buffer::MAP_WRITE);

        T *dstData = bufferDst.getPtr<T>();

        unsigned int spanSize = width * dstImg->m_bpp;
        unsigned char * src = (unsigned char*)subPixels;
        for ( unsigned int z=z_offset ; z<(z_offset+depth) ; z++ )
        {
          for ( unsigned int y=y_offset ; y<(y_offset+height) ; y++ )
          {
            memcpy( dstData + _pixelOffsetElements<T>( dstImg, x_offset, y, z, 0 ), src, spanSize );
            src += spanSize;
          }
        }
      }
  
      bool TextureHost::setSubImagePixels( unsigned int image, unsigned int mipmap, unsigned int x_offset
                                          , unsigned int y_offset, unsigned int z_offset
                                          , unsigned int width, unsigned int height
                                          , unsigned int depth, const void * subPixels )
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        DP_ASSERT(subPixels);

        // This check is not really required, since if anything is zero
        // the memcpy will copy zero bytes, anyway, however
        // do no more work than required !
        if ( width == 0 || height == 0 || depth == 0 )
        {
          return true;
        }

        // Get the image
        Image* dstImg = &m_images[image][mipmap];

        // Check if the sub image boundary off-shoots that of the existing image
        // or there is no pixel to copy to
        if (    ( (x_offset + width)  > dstImg->m_width )
            ||  ( (y_offset + height) > dstImg->m_height )
            ||  ( (z_offset + depth)  > dstImg->m_depth )
            ||  ! dstImg->m_pixels )
        {
          return false;
        }

        switch( dstImg->m_type )
        {
          case Image::IMG_UNSIGNED_BYTE:
            _setSubImagePixels<unsigned char>(dstImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_BYTE:                                 
            _setSubImagePixels<char>(dstImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_UNSIGNED_SHORT:
            _setSubImagePixels<unsigned short>(dstImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_SHORT:
            _setSubImagePixels<short>(dstImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_UNSIGNED_INT:
          case Image::IMG_UNSIGNED_INT_2_10_10_10:
          case Image::IMG_UNSIGNED_INT_5_9_9_9:
          case Image::IMG_UNSIGNED_INT_10F_11F_11F:
          case Image::IMG_UNSIGNED_INT_24_8:
            _setSubImagePixels<unsigned int>(dstImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_INT:
            _setSubImagePixels<int>(dstImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
          case Image::IMG_FLOAT:
            _setSubImagePixels<float>(dstImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
#if defined(HAVE_HALF_FLOAT)
          case Image::IMG_HALF:
            _setSubImagePixels<half>(dstImg, x_offset, y_offset, z_offset, width, height, depth, subPixels);
            break;
#endif
          default:
            DP_ASSERT(false);
            break;
        }

        invalidateHashKey();
        return true;
      }

      bool TextureHost::setSubImagePixels( unsigned int image, unsigned int mipmap, unsigned int x_offset
                                          , unsigned int y_offset, unsigned int z_offset
                                          , unsigned int width, unsigned int height
                                          , unsigned int depth, const BufferSharedPtr &pixelBuffer )
      {
        Buffer::DataReadLock buffer( pixelBuffer );
        bool result = setSubImagePixels( image, mipmap, x_offset, y_offset, z_offset, width, height, depth, buffer.getPtr() );
        return result;
      }

      void TextureHost::filter( const Image * src, Image * dst, ScaleFilter * sf )
      {
        std::vector<std::vector<std::pair<unsigned int,float> > > contributions[3];
        //  contributions[0..2]: contributions in x-, y-, and z-direction
        //  contributions[0..2][i] : vector of contributions to i-th destination pixel
        //  contributions[0..2][i][j].first: index of source pixel
        //  contributions[0..2][i][j].second: weight of source pixel
        calculateContributions( contributions[0], src->m_width, dst->m_width, sf );
        calculateContributions( contributions[1], src->m_height, dst->m_height, sf );
        calculateContributions( contributions[2], src->m_depth, dst->m_depth, sf );

        rescale( src, dst, contributions );
        invalidateHashKey();
      }

      bool TextureHost::scale(unsigned int image, unsigned int width, unsigned int height, unsigned int depth)
      {
        DP_ASSERT(image<m_images.size()); // range fault! serious!

        if (  (m_creationFlags & F_IMAGE_STREAM)               // never use this function for streams 
           || (m_images.size() <= image)                       // bad index
           || ( isProblematicType( m_images[image][0].m_type ) )
           || ( isProblematicFormat( m_images[image][0].m_format ) )
           || (m_images[image][0].m_pixels == NULL ) ) // no pixel data to scale
        {
          return false;
        }

        Image * src = &m_images[image][0]; // shortcut to source image    

        if ( ( width != src->m_width ) || ( height != src->m_height ) || ( depth != src->m_depth ) )
        {
          Image dst( width, height, depth, src->m_format, src->m_type );
          dst.m_pixels = BufferHost::create();
          if ( !dst.m_pixels )
          { // image allocation failed
    #if 0
            NVSG_TRACE_OUT("ERROR: image allocation failed!\n");
    #endif
            goto ERROREXIT;
          }
          dst.m_pixels->setSize(dst.m_nob);

          ScaleFilter * sf = _createScaleFilter( m_creationFlags );
          filter( src, &dst, sf );
          delete sf;

          // now replace image by the scaled image ...
          m_images[image][0] = dst;

          // ... and create mipmaps for the scaled image in case the source image has mipmaps
          if ( 1 < m_images[image].size() )
          {
            DP_VERIFY( createMipmaps( m_images[image] ) );
          }
        }

        invalidateHashKey();
        return true;

    ERROREXIT:
        return false;
      }

      bool TextureHost::convertPixelFormat( Image::PixelFormat newFormat )
      {
        DP_ASSERT(m_images.size()); // range fault! serious!

        if (  (m_creationFlags & F_IMAGE_STREAM)               // never use this function for streams 
           || ( isProblematicType( m_images[0][0].m_type )) 
           || ( isProblematicFormat( m_images[0][0].m_format )) 
           || (m_images[0][0].m_pixels == NULL ) ) // no pixel data to scale
        {
          return false;
        }

        if( newFormat == m_images[0][0].m_format )
        {
          //formats are the same - don't bother checking mipmaps
          return true;
        }

        //
        // convert all images, all mipmaps
        //
        for( size_t i = 0; i < m_images.size(); i ++ )
        {
          for( size_t j = 0; j < m_images[i].size(); j++ )
          {
            Image * src = &m_images[i][j];
            Image dst( src->m_width, src->m_height, src->m_depth, newFormat, src->m_type );

            dst.m_pixels = BufferHost::create();
            if ( !dst.m_pixels )
            { // image allocation failed
    #if 0
              NVSG_TRACE_OUT("ERROR: image allocation failed!\n");
    #endif
              goto ERROREXIT;
            }
            dst.m_pixels->setSize( dst.m_nob );

            // cvt format
            cvtPixelFormat( src, &dst, newFormat );

            // reset
            m_images[i][j] = dst;
          }
        }

        invalidateHashKey();
        return true;

    ERROREXIT:
        return false;
      }

      bool TextureHost::scaleToPowerOfTwo(unsigned int sizeLimit)
      {
        DP_ASSERT(!(m_creationFlags & F_IMAGE_STREAM)); // must not modify streams!

        DP_ASSERT(!m_images.empty()); // requires at least one image available
        Image& img = m_images[0][0]; // shortcut to first top level image

        // we don't convert these pixel formats at the moment.
        DP_ASSERT( isProblematicFormat( img.m_format ) == false );
        DP_ASSERT( isProblematicType( img.m_type ) == false );

        if( isProblematicFormat( img.m_format ) || isProblematicType( img.m_type ) )
        {
          return false;
        }

        DP_ASSERT(img.m_pixels); // requires pixel data
    
        unsigned int w = img.m_width;
        unsigned int h = img.m_height;
        unsigned int d = img.m_depth;

        if (  sizeLimit < w || !isPowerOfTwo(w) 
           || sizeLimit < h || !isPowerOfTwo(h) 
           || sizeLimit < d || !isPowerOfTwo(d) )
        {
    #if 0
          NVSG_TRACE_OUT_F(("unscaled width  = %d\n", w));
          NVSG_TRACE_OUT_F(("unscaled height = %d\n", h));
          NVSG_TRACE_OUT_F(("unscaled depth  = %d\n", d));
    #endif

          // clamp dimensions to specified size limit
          w = std::min(w, sizeLimit);
          h = std::min(h, sizeLimit);
          d = std::min(d, sizeLimit);

          if ( !(m_creationFlags & F_SCALE_POT_MASK) 
             /* fallback to nearest if both, F_SCALE_POT_ABOVE and F_SCALE_POT_BELOW are specified */
             || (m_creationFlags & F_SCALE_POT_MASK)==(F_SCALE_POT_ABOVE | F_SCALE_POT_BELOW) )
          {
    #if 0
            NVSG_TRACE_OUT("scaling to nearest power-of-two\n");
    #endif
            w = powerOfTwoNearest(w);
            h = powerOfTwoNearest(h);
            d = powerOfTwoNearest(d);
          }
          else if ( m_creationFlags & F_SCALE_POT_BELOW )
          {
    #if 0
            NVSG_TRACE_OUT("scaling to previous power-of-two\n");
    #endif
            w = powerOfTwoBelow(w);
            h = powerOfTwoBelow(h);
            d = powerOfTwoBelow(d);
          }
          else
          {
    #if 0
            NVSG_TRACE_OUT("scaling to next power-of-two\n");
    #endif
            w = powerOfTwoAbove(w);
            h = powerOfTwoAbove(h);
            d = powerOfTwoAbove(d);
          }
      
    #if 0
          NVSG_TRACE_OUT_F(("scaled width  = %d\n", w));
          NVSG_TRACE_OUT_F(("scaled height = %d\n", h));
          NVSG_TRACE_OUT_F(("scaled depth  = %d\n", d));
    #endif

          // apply to all first level images
          DP_ASSERT( m_images.size() <= UINT_MAX );
          for ( unsigned int i=0; i<m_images.size(); ++i )
          {
            if ( !scale(i, w, h, d) )
            {
    #if 0
              NVSG_TRACE_OUT_F(("ERROR: scaling failed for image #%d\n"));
    #endif
              return false;
            }
          }
        }
        invalidateHashKey();
        return true;
      }

      bool TextureHost::isFloatingPoint() const
      {
        if ( ! m_images.empty() )
        {
          Image::PixelDataType type = m_images[0][0].m_type;
      
          return ( type==Image::IMG_FLOAT32 || type==Image::IMG_FLOAT16 ||
                   type==Image::IMG_UNSIGNED_INT_5_9_9_9 ||
                   type==Image::IMG_UNSIGNED_INT_10F_11F_11F );
        }
        return false;
      }

      bool TextureHost::isFixedPoint() const
      {
        // all compressed formats are fixed point
        return !isFloatingPoint();
      }

      bool TextureHost::isCompressed() const
      {
        if ( ! m_images.empty() )
        {
          Image::PixelFormat fmt = m_images[0][0].m_format;

          return ( (fmt >= Image::IMG_COMPRESSED_LUMINANCE_LATC1) &&
                   (fmt <= Image::IMG_COMPRESSED_SRGBA_DXT5) );
        }

        return false;
      }
  
      void TextureHost::convert2DToCubeMap()
      {
        DP_ASSERT((m_creationFlags & F_IMAGE_STREAM)==0);
        DP_ASSERT(is2D());
        DP_ASSERT(m_images.size()==1 && m_images[0].size()==1);
        DP_ASSERT(VERTICAL_CROSS_FORMAT(m_images[0][0].m_width, m_images[0][0].m_height));

        // the following calls the default copy constructor of Image, 
        // which does a flat copy! this is exactly what we need here!
        // after this we have srcImg.m_pixels==m_images[0][0].m_pixels
        Image srcImg(m_images[0][0]);
        m_images.clear();
    
        // run through cube map faces
        for ( unsigned int i = 0; i < 6; i++ ) 
        {
          unsigned int nImg = addImage( srcImg.m_width/3, srcImg.m_height/4, srcImg.m_depth
                                      , srcImg.m_format, srcImg.m_type );
          DP_ASSERT( nImg == i );

          Image* img = &m_images[nImg][0];
          DP_ASSERT( img->m_width == img->m_height );
          img->m_pixels = BufferHost::create();
          if ( !img->m_pixels )
          { // image allocation failed
    #if 0
            NVSG_TRACE_OUT("ERROR: image allocation failed!\n");
    #endif
            goto ERROREXIT;
          }
          else
          {
            img->m_pixels->setSize( img->m_nob );
          }

          // horizontal and vertical extents of the face relative to the 2D image
          int x_begin = 0;
          int x_end = 0;
          int y_begin =0;
          int y_end = 0; 
          int length = img->m_width;      

          //    | nz|            
          //    | ny|
          // ---     ---
          //| px pz   nx|
          // ---     ---
          //    | py|      
          //    |---|
          // note: we assume the image is ORIGIN_LOWER_LEFT
          switch ( i )
          {
            case 2: // py
              y_begin = length * 3;
              y_end   = (length * 4) - 1;
              x_begin = length;
              x_end   = (length * 2) - 1;
              break;
            case 0: // px
              y_begin = length * 2;
              y_end   = (length * 3) - 1;
              x_begin = 0;
              x_end   = length - 1;
              break;
            case 4: // pz
              y_begin = length * 2;
              y_end   = (length * 3) - 1;
              x_begin = length;
              x_end   = (length * 2) - 1;
              break;
            case 1: // nx
              y_begin = length * 2;
              y_end   = (length * 3) - 1;
              x_begin = length * 2 ;
              x_end   = (length * 3) - 1;
              break;
            case 3: // ny 
              y_begin = length;
              y_end   = (length * 2) - 1;
              x_begin = length;
              x_end   = (length * 2) - 1;
              break;
            case 5: // nz 
              y_begin = 0;
              y_end   = length - 1;
              x_begin = length;
              x_end   = (length * 2) - 1;        
              break;
            default:            
              DP_ASSERT(false);
              break;
          }

          int nTgt = 0;

          Buffer::DataReadLock  bufferSrc( srcImg.m_pixels );
          Buffer::DataWriteLock bufferDst( img->m_pixels, Buffer::MAP_WRITE );

          const char *src = bufferSrc.getPtr<char>();
          char *dst = bufferDst.getPtr<char>();

          if(i<=4) // needs to be rotated 180°
          {
            for ( int y = y_end; y >= y_begin; y-- ) 
            {
              for ( int x = x_end; x >= x_begin; x-- )
              {
                int srcIndex = y * srcImg.m_bpl + x * srcImg.m_bpp;
                memcpy( &dst[nTgt], &src[srcIndex], img->m_bpp );
                nTgt += img->m_bpp;
              }
            }
          }
          else
          {
            // just copy these
            for ( int y = y_begin; y <= y_end; y++ ) 
            {
              for ( int x = x_begin; x <= x_end; x++ )
              {          
                int srcIndex = y * srcImg.m_bpl + x * srcImg.m_bpp;
                memcpy( &dst[nTgt], &src[srcIndex], img->m_bpp );
                nTgt += img->m_bpp;
              }
            }
          }

          DP_ASSERT(nTgt == img->m_bps);
        }

        //  finally delete the source image pixels
        srcImg.m_pixels.reset();

        invalidateHashKey();
        return;

    ERROREXIT:
        m_images.clear();
        m_images.back().push_back( srcImg );
      }

      void TextureHost::finishUpload( )
      {
        m_internalFlags &= ~F_UPLOAD_DEMANDED;
        // no need to invalidateHashKey on change of m_internalFlags
      }

      void TextureHost::mirrorX( unsigned int image )
      {
        DP_ASSERT(image < m_images.size());

        std::vector<BufferSharedPtr> pixels( m_images[image].size() );   // the mirrored pixels

        for(size_t i=0; i<m_images[image].size(); i++)  // flip all images
        {
          Image * src = &m_images[image][i];
          pixels[i] = BufferHost::create();
      
          if ( !pixels[i] )
          { // image allocation failed
    #if 0
            NVSG_TRACE_OUT("ERROR: image allocation failed!\n");
    #endif
            return;
          }
          else
          {
            pixels[i]->setSize( src->m_nob );
          }

          Buffer::DataReadLock  bufferSrc( src->m_pixels );
          Buffer::DataWriteLock bufferDst( pixels[i], Buffer::MAP_WRITE );

          const char *srcData = bufferSrc.getPtr<char>();
          char *dstData = bufferDst.getPtr<char>();
      
          unsigned int dstIndex = 0;
          for ( unsigned int z = 0; z < src->m_depth  ; z++ ) 
          {
            for ( int y = src->m_height-1; y >= 0 ; y-- ) 
            {
              for ( unsigned int x = 0; x < src->m_width; x++ )
              {
                unsigned int srcIndex = z * src->m_bps + y * src->m_bpl + x * src->m_bpp;
                memcpy( &dstData[dstIndex], &srcData[srcIndex], src->m_bpp );
                dstIndex += src->m_bpp;
              }
            }
          }

        }

        for ( size_t i=0 ; i<m_images[image].size() ; i++ )
        {
          m_images[image][i].m_pixels = pixels[i];
        }

        invalidateHashKey();
      }

      void TextureHost::mirrorY( unsigned int image )
      {
        DP_ASSERT(image < m_images.size());

        std::vector<BufferSharedPtr> pixels( m_images[image].size() );   // the mirrored pixels

        for(size_t i=0; i<m_images[image].size(); i++)  // flip all images
        {   
          Image * src = &m_images[image][i];
          pixels[i] = BufferHost::create();
          if ( !pixels[i] )
          { // image allocation failed
    #if 0
            NVSG_TRACE_OUT("ERROR: image allocation failed!\n");
    #endif
            return;
          }
          else
          {
            pixels[i]->setSize( src->m_nob );
          }
      
          Buffer::DataReadLock  bufferSrc( src->m_pixels );
          Buffer::DataWriteLock bufferDst( pixels[i], Buffer::MAP_WRITE );

          const char *srcData = bufferSrc.getPtr<char>();
          char *dstData = bufferDst.getPtr<char>();

          unsigned int dstIndex = 0;
          for ( unsigned int z = 0; z < src->m_depth  ; z++ ) 
          {   
            for ( unsigned int y = 0; y < src->m_height ; y++ ) 
            {
              for ( int x = src->m_width-1; x >= 0 ; x-- )
              {          
                unsigned int srcIndex = z * src->m_bps + y * src->m_bpl + x * src->m_bpp;
                memcpy( &dstData[dstIndex], &srcData[srcIndex], src->m_bpp );
                dstIndex += src->m_bpp;
              }
            }
          }

        }

        for ( size_t i=0 ; i<m_images[image].size() ; i++ )
        {
          m_images[image][i].m_pixels = pixels[i];
        }

        invalidateHashKey();
      }

      void TextureHost::recalcTotalNumberOfBytes()
      {
        m_totalNumBytes = 0;
        for ( size_t i=0; i<m_images.size(); ++i )
        {
          for ( size_t j=0; j<m_images[i].size(); ++j )
          {
            m_totalNumBytes += m_images[i][j].m_nob;
          }
        }
      }

      bool TextureHost::setTextureTarget( TextureTarget target )
      {
        DP_ASSERT( m_images.size() > 0 && "Image Data must be set before");

        bool valid;
        switch(target)
        {
         case TT_TEXTURE_1D:
           valid = is1D();
           break;
         case TT_TEXTURE_2D:
           valid = is2D();
           break;
         case TT_TEXTURE_3D:
           valid = is3D();
           break;
         case TT_TEXTURE_CUBE:
           valid = isCubeMap();
           break;
         case TT_TEXTURE_1D_ARRAY:
           valid = is1DArray();
           break;
         case TT_TEXTURE_2D_ARRAY:
           valid = is2DArray();
           break;
         case TT_TEXTURE_RECTANGLE:
           valid = is2D();
           break;
         case TT_TEXTURE_CUBE_ARRAY:
           valid = isCubeMapArray();
           break;
         default:
           return false;
        }

        if (valid)
        {
          Texture::setTextureTarget( target );
        }

        return valid;
      }

      bool TextureHost::convertToTextureTarget( TextureTarget target )
      {
        if ( getTextureTarget() == target )
        {
          return true;
        }
        // don't allow a change if already set to something useful
        if ( getTextureTarget() != TT_UNSPECIFIED_TEXTURE_TARGET )
        {
          return false;
        }

        bool valid = setTextureTarget(target);
        if (valid)
        {
          return true;
        }

        switch( target )
        {
        case TT_TEXTURE_CUBE:
          valid = is2D() && VERTICAL_CROSS_FORMAT(getWidth(), getHeight());
          if (valid)
          {
            convert2DToCubeMap();
          }
          break;

        default:
          break;
        }

        if (valid)
        {
          setTextureTarget( target );
          invalidateHashKey();
        }

        return valid;
      }

      void TextureHost::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Texture::feedHashGenerator( hg );
        // no need to handle m_internalFlags here
        hg.update( reinterpret_cast<const unsigned char *>(&m_creationFlags), sizeof(m_creationFlags) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_totalNumBytes), sizeof(m_totalNumBytes) );
        if ( m_filename.empty() )
        {
          for ( size_t i=0 ; i<m_images.size() ; ++i )
          {
            for ( size_t j=0 ; j<m_images[i].size() ; ++j )
            {
              hg.update( reinterpret_cast<const unsigned char *>(&m_images[i][j].m_width), sizeof(m_images[i][j].m_width) );
              hg.update( reinterpret_cast<const unsigned char *>(&m_images[i][j].m_height), sizeof(m_images[i][j].m_height) );
              hg.update( reinterpret_cast<const unsigned char *>(&m_images[i][j].m_depth), sizeof(m_images[i][j].m_depth) );
              hg.update( reinterpret_cast<const unsigned char *>(&m_images[i][j].m_format), sizeof(m_images[i][j].m_format) );
              hg.update( reinterpret_cast<const unsigned char *>(&m_images[i][j].m_type), sizeof(m_images[i][j].m_type) );
              hg.update( reinterpret_cast<const unsigned char *>(&m_images[i][j].m_bpp), sizeof(m_images[i][j].m_bpp) );
              hg.update( reinterpret_cast<const unsigned char *>(&m_images[i][j].m_bpl), sizeof(m_images[i][j].m_bpl) );
              hg.update( reinterpret_cast<const unsigned char *>(&m_images[i][j].m_bps), sizeof(m_images[i][j].m_bps) );
              hg.update( reinterpret_cast<const unsigned char *>(&m_images[i][j].m_nob), sizeof(m_images[i][j].m_nob) );
              // need to handle m_pixels here too !!
              //util::HashKey hk = BufferLock(m_images[i][j].m_pixels)->getHashKey();
              //hg.update( reinterpret_cast<const unsigned char *>(&hk), sizeof(hk) );
            }
          }
        }
        else
        {
          hg.update( reinterpret_cast<const unsigned char *>(m_filename.c_str()), util::checked_cast<unsigned int>(m_filename.length()) );
        }
        hg.update( reinterpret_cast<const unsigned char *>(&m_gpuFormat), sizeof(m_gpuFormat) );
      }

      bool TextureHost::isEquivalent( TextureSharedPtr const& texture, bool deepCompare ) const
      {
        if ( texture == this )
        {
          return( true );
        }

        bool equi = texture.isPtrTo<TextureHost>() && Texture::isEquivalent( texture, deepCompare );
        if ( equi )
        {
          TextureHostSharedPtr const& th = texture.staticCast<TextureHost>();

          equi =  ( th != nullptr )
              &&  ( m_creationFlags   == th->m_creationFlags )
              &&  ( m_totalNumBytes   == th->m_totalNumBytes )
              &&  ( m_filename        == th->m_filename )
              &&  ( m_gpuFormat       == th->m_gpuFormat );
          if ( equi && m_filename.empty() )
          {
            for ( size_t i=0 ; i<m_images.size() && equi ; ++i )
            {
              for ( size_t j=0 ; j<m_images[i].size() && equi ; ++j )
              {
                equi =  ( m_images[i][j].m_width  == th->m_images[i][j].m_width )
                    &&  ( m_images[i][j].m_height == th->m_images[i][j].m_height )
                    &&  ( m_images[i][j].m_depth  == th->m_images[i][j].m_depth )
                    &&  ( m_images[i][j].m_format == th->m_images[i][j].m_format )
                    &&  ( m_images[i][j].m_type   == th->m_images[i][j].m_type )
                    &&  ( m_images[i][j].m_bpp    == th->m_images[i][j].m_bpp )
                    &&  ( m_images[i][j].m_bpl    == th->m_images[i][j].m_bpl )
                    &&  ( m_images[i][j].m_bps    == th->m_images[i][j].m_bps )
                    &&  ( m_images[i][j].m_nob    == th->m_images[i][j].m_nob );
                if ( equi )
                {
                  if ( deepCompare )
                  {
                    Buffer::DataReadLock data( th->m_images[i][j].m_pixels );
                    Buffer::DataReadLock self( m_images[i][j].m_pixels );
                    equi = ( memcmp( data.getPtr(), self.getPtr(), m_images[i][j].m_nob ) == 0 );
                    // the above sequence should be replaced by the following command, when Buffer learns about isEquivalent !!
                    //equi = BufferLock( m_images[i][j].m_pixels )->isEquivalent( BufferLock( th->m_images[i][j].m_pixels ), deepCompare );
                  }
                  else
                  {
                    equi = ( m_images[i][j].m_pixels == th->m_images[i][j].m_pixels );
                  }
                }
              }
            }
          }
        }
        return( equi );
      }

      TextureHostSharedPtr createStandardTexture()
      {
        static std::vector<Vec4f> tex;
        if ( tex.empty() )
        {
          tex.resize(64);

          // Create pattern
          for( unsigned int i = 0; i < 8; ++i )
          {
            for( unsigned int j = 0; j < 8; ++j )
            {
              unsigned int pos = i * 8 + j;
              Vec4f col(float(( i ^ j ) & 1), float((( i ^ j ) & 2) / 2), float(((i  ^ j ) & 4) / 4), 1.0f);
              tex[pos] = col;
            }
          }
        }

        TextureHostSharedPtr textureHost = TextureHost::create();
        textureHost->setCreationFlags( TextureHost::F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD );
        unsigned int index = textureHost->addImage( 8, 8, 1, Image::IMG_RGBA, Image::IMG_FLOAT32 );
        DP_ASSERT( index != -1 );
        textureHost->setImageData( index, (const void *) &tex[0] );
        textureHost->setTextureTarget( TT_TEXTURE_2D );
        textureHost->setTextureGPUFormat(TextureHost::TGF_FIXED8);

        return( textureHost );
      }

      float BellFilter::operator()( float t ) const
      {
        DP_ASSERT( 0.0f <= t );
        if ( t < 0.5f )
        {
          return( 0.75f - t * t );
        }
        else if ( t < 1.5f )
        {
          t -= 1.5f;
          return( 0.5f * t * t );
        }
        else
        {
          return( 0.0f );
        }
      }

      float BSplineFilter::operator()( float t ) const
      {
        DP_ASSERT( 0.0f <= t );
        if ( t < 1.0f )
        {
          float tt = t * t;
          return( 0.5f * tt * t - tt + 2.0f / 3.0f );
        }
        else if ( t < 2.0f )
        {
          t = 2.0f - t;
          return( 1.0f / 6.0f * t * t * t );
        }
        else
        {
          return( 0.0f );
        }
      }

    } // namespace core
  } // namespace sg
} // namespace dp
