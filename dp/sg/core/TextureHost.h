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


#pragma once
/** @file */

#include <dp/Types.h>
#include <dp/math/math.h>
#include <dp/sg/core/Config.h>
#include <dp/sg/core/Buffer.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/Texture.h>
#include <dp/util/BitMask.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>

// VERTICAL_CROSS_FORMAT(w,h) macro evaluates to true if 
// width and height comply with vertical cross format:
//      
//        ___     
//       |   |
// 3  ___|___|____
//   |   |   |    |    
// 2 |___|___|____|    
//       |   |          3xheight == 4xwidth
// 1     |___|
//       |   |
// 0     |___|
//
//     0   1   2
#define VERTICAL_CROSS_FORMAT(w,h) ((3*(h))==(4*(w)))

namespace dp
{
  namespace sg
  {
    namespace core
    {

      class ScaleFilter;
      class TextureLoader;
      class TextureHost;

      /*! \brief Retrieves the best-matching TextureTarget from A TextureHost. 
       * \return The function returns the best-matching TextureTarget for the specified TextureHost, or
       * TT_UNSPECIFIED_TEXTURE_TARGET if no matching TextureTarget can be determined.
       */
      DP_SG_CORE_API TextureTarget determineTextureTarget( const TextureHostSharedPtr & img );

      /*! \brief Helper structure to hold the data of an image.
       *  \par Namespace: dp::sg::core
       *  \sa insert, rescale */
      struct Image
      {
        //! Enumeration type to describe a pixel data format
        //! Essentially the "Format" parameter to TexImage
        //
        // NOTE WELL: THE ORDERING OF THIS TABLE IS VERY IMPORTANT - DO NOT
        // CHANGE THE ORDER OF ANY OF THE VALUES WITHOUT KNOWING EXACTLY WHAT
        // YOU ARE GETTING YOURSELF INTO!!
        //
        // TextureHost.cpp and GLContext.cpp contain dependencies.
        //
        enum PixelFormat
        {
            IMG_UNKNOWN_FORMAT = -1 //!< unknown pixel format
          , IMG_COLOR_INDEX = 0     //!< color index format
          , IMG_RGB                 //!< RGB format
          , IMG_RGBA                //!< RGBA format
          , IMG_BGR                 //!< BGR format
          , IMG_BGRA                //!< BGRA format
          , IMG_LUMINANCE           //!< luminance format
                                    //   luminace and intensity are the same cept for A
          , IMG_LUMINANCE_ALPHA     //!< luminance alpha format

          //
          // New Formats
          // 
          , IMG_ALPHA           //!< Alpha only format

          , IMG_DEPTH_COMPONENT //!< Depth only ( ARB_depth_texture )
          , IMG_DEPTH_STENCIL   //!< Depth + stencil - EXT_packed_depth_stencil

          // Integer
          , IMG_INTEGER_ALPHA
          , IMG_INTEGER_LUMINANCE
          , IMG_INTEGER_LUMINANCE_ALPHA
          , IMG_INTEGER_RGB
          , IMG_INTEGER_BGR
          , IMG_INTEGER_RGBA
          , IMG_INTEGER_BGRA

          //
          // Compressed texture formats
          // 
          // not really a "Format", but these require Compressed entry
          // point because they signify that the data has already been
          // compressed
          //
          , IMG_COMPRESSED_LUMINANCE_LATC1        
          , IMG_COMPRESSED_SIGNED_LUMINANCE_LATC1 
          , IMG_COMPRESSED_LUMINANCE_ALPHA_LATC2  
          , IMG_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2
          , IMG_COMPRESSED_RED_RGTC1
          , IMG_COMPRESSED_SIGNED_RED_RGTC1
          , IMG_COMPRESSED_RG_RGTC2
          , IMG_COMPRESSED_SIGNED_RG_RGTC2

          , IMG_COMPRESSED_RGB_DXT1
          , IMG_COMPRESSED_RGBA_DXT1
          , IMG_COMPRESSED_RGBA_DXT3
          , IMG_COMPRESSED_RGBA_DXT5

          , IMG_COMPRESSED_SRGB_DXT1
          , IMG_COMPRESSED_SRGBA_DXT1
          , IMG_COMPRESSED_SRGBA_DXT3
          , IMG_COMPRESSED_SRGBA_DXT5

          , IMG_NUM_FORMATS     //!< specifies number of valid format descriptors
        };

        //! Enumeration type to describe the data type of pixel data
        //
        // NOTE WELL: THE ORDERING OF THIS TABLE IS VERY IMPORTANT - DO NOT
        // CHANGE THE ORDER OF ANY OF THE VALUES WITHOUT KNOWING EXACTLY WHAT
        // YOU ARE GETTING YOURSELF INTO!!
        //
        // TextureHost.cpp and GLContext.cpp contain dependencies.
        //
        enum PixelDataType
        {
            IMG_UNKNOWN_TYPE = -1 //!< unknown data type
          , IMG_BYTE = 0          //!< signed byte type (8-bit)
          , IMG_UNSIGNED_BYTE     //!< unsigned byte type (8-bit)
          , IMG_SHORT             //!< signed short type (16-bit)  
          , IMG_UNSIGNED_SHORT    //!< unsigned short type (16-bit)
          , IMG_INT               //!< signed integer type (32-bit)
          , IMG_UNSIGNED_INT      //!< unsigned integer type (32-bit)
          , IMG_FLOAT32           //!< single-precision floating point (32-bit)
          , IMG_FLOAT16           //!< half-precision floating point (16-bit)

          // custom formats
          , IMG_UNSIGNED_INT_2_10_10_10 //!< 2 Bits A, 10 bits RGB (RGB10/10A2)
          , IMG_UNSIGNED_INT_5_9_9_9    //!< 5 bits exponent 9 bits RGB mantissa
                                        //   (EXT_texture_shared_exponent) format
          , IMG_UNSIGNED_INT_10F_11F_11F //!< 5 bits exp + 5,6,6 bits RGB mantissa
                                         //   (EXT_packed_float) format
          , IMG_UNSIGNED_INT_24_8      //!< 24 bits depth, 8 bits stencil
                                         //   (EXT_packed_depth_stencil) format
          , IMG_NUM_TYPES         //!< specifies number of valid type descriptors

          // compatibility
          , IMG_FLOAT = IMG_FLOAT32
          , IMG_HALF  = IMG_FLOAT16
        };


        /*! \brief Default constructor of Image.
         *  \remarks Initializes an empty Image with no data. */
        DP_SG_CORE_API Image();

        /*! \brief Constructor of an Image structure.
         *  \param width The width of the Image.
         *  \param height The height of the Image.
         *  \param depth The depth of the Image.
         *  \param format The PixelFormat of the Image.
         *  \param type The PixelDataType of the Image.
         *  \remarks This constructor allocates no memory for the pixel data, it just initialized the
         *  structure Image to hold the information for it.
         *  \note The bytes per pixel can be determined out of \a format (giving number of components)
         *  and \a type (giving size of components). */
        DP_SG_CORE_API Image( unsigned int width, unsigned int height, unsigned int depth
                      , PixelFormat format, PixelDataType type );
  
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // NOTE:
        // We need the copy constructor performing just a flat copy. That is, the 
        // copy must not allocate its own pixel storage and copy pixels from the 
        // source, but have m_pixels pointing to the same storage instead!
        //
        // This is exactly what a default copy constructor does! 
        // Therefore, we don't need to implement a copy constructor by our own!
        // Same is true for the assignment operator!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        DP_SG_CORE_API Image( const Image &rhs );
        DP_SG_CORE_API Image &operator=( const Image &rhs );

        unsigned int     m_width;      //!< width of the texture in pixels
        unsigned int     m_height;     //!< height of the texture in pixels
        unsigned int     m_depth;      //!< depth of the texture in pixels
        PixelFormat      m_format;     //!< specifies format of pixel data 
        PixelDataType    m_type;       //!< specifies type of pixel data
        unsigned int     m_bpp;        //!< bytes per pixel
        unsigned int     m_bpl;        //!< bytes per scanline
        unsigned int     m_bps;        //!< bytes per slice (plane)
        unsigned int     m_nob;        //!< number of bytes (complete image)        
        BufferSharedPtr m_pixels;    //!< raw image data (we use unsigned char as internal format)
      };

    #if defined(NDEBUG)
      /*! \brief Helper function to allocate memory for pixel data.
       *  \param nbytes The number of bytes to allocate.
       *  \return A pointer to the allocated memory.
       *  \remarks This function allocates the memory and assures it to be eight byte aligned. You
       *  should use freeImagePixels to free that memory.
       *  \sa freeImagePixels */
      DP_SG_CORE_API unsigned char * allocImagePixels( unsigned int nbytes );
    #else
      /*! \brief Same as allocImagePixels + leak detection */
      DP_SG_CORE_API unsigned char * dbgAllocImagePixels( unsigned int nbytes, unsigned int line );
      // This macro allows the same API usage as in the release mode
      #define allocImagePixels(nbytes) dbgAllocImagePixels( nbytes, __LINE__)
    #endif

      /*! \brief Helper function to free memory of pixel data.
       *  \param pixels A pointer to the pixel data to free.
       *  \remarks This function assumes that \a pixels has been allocated by allocImagePixels.
       *  \sa allocImagePixels */
      DP_SG_CORE_API void freeImagePixels( unsigned char * pixels );

      /*! \brief Helper function to determine the number of components of a PixelFormat.
       *  \param format The PixelFormat to determine the number of components from.
       *  \return The number of components described by \a format.
       *  \sa sizeOfComponents */
      DP_SG_CORE_API unsigned int numberOfComponents( Image::PixelFormat format );

      /*! \brief Helper function to determine the size of components of a PixelDataType.
       *  \param type The PixelDataType to determine the size of components from.
       *  \return The size of components described by \a type.
       *  \sa numberOfComponents */
      DP_SG_CORE_API unsigned int sizeOfComponents( Image::PixelDataType type );

      /*! \brief Helper function to determine the contributions of a ScaleFilter on scaling an Image.
       *  \param contribs A reference to a vector of vector of pairs of an int and a float. The i'th
       *  element of \a contribs holds the vector of contributions for the i'th destination value.
       *  The j'th element of \a contribs[i] holds the index and the contribution of the source
       *  values.
       *  \param srcSize The number of the source values.
       *  \param dstSize The number of the destination values.
       *  \param filter A pointer to the ScaleFilter to use for contribution determination.
       *  \remarks To determine the contributions of \a filter on a multidimensional array of values
       *  call calculateContributions once per dimension.
       *  \par Example
       *  \code
       *  vector<vector<pair<unsigned int,float> > > contributions[3];
       *  //  contributions[0..2]: contributions in x-, y-, and z-direction
       *  calculateContributions( contributions[0], srcWidth, dstWidth, sf );
       *  calculateContributions( contributions[1], srcHeight, dstHeight, sf );
       *  calculateContributions( contributions[2], srcDepth, dstDepth, sf );
       *  rescale( src, dst, contributions );
       *  \endcode
       *  \sa insert, rescale, TextureHost::filter */
      DP_SG_CORE_API void calculateContributions( std::vector<std::vector<std::pair<unsigned int,float> > > &contribs
                                          , unsigned int srcSize, unsigned int dstSize
                                          , const ScaleFilter * filter );

      /*! \brief Helper function to insert a source image into a destination image at a given position.
       *  \param src A pointer to the constant source image.
       *  \param dst A pointer to the destination image.
       *  \param x x-position of the source image in the destination image.
       *  \param y y-position of the source image in the destination image.
       *  \param z y-position of the source image in the destination image.
       *  \remarks It is assumed the the source image offset by x/y/z completely fits into the
       *  destination image.
       *  \note If the source image has overlaps, the overlap pixels are not copied (and need not fit
       *  into the destination image).
       *  \sa rescale */
      DP_SG_CORE_API void insert( const Image * src, Image * dst, unsigned int x, unsigned int y, unsigned int z );

      /*! \brief Helper function to rescale an Image using the contributions of a ScaleFilter.
       *  \param src A pointer to the constant source Image to scale.
       *  \param dst A pointer to the destination Image.
       *  \param contributions A pointer to three vectors of vectors of pairs of an int and a float
       *  that holds the contributions of each source pixel to each destination pixel in x, y, and z
       *  direction.
       *  \remarks The \a contributions can be calculated by calculateContributions.
       *  \note If the destination image is defined to have overlaps, those pixel values are not set.
       *  \sa calculateContributions */
      DP_SG_CORE_API void rescale( const Image * src, Image *dst
                           , std::vector<std::vector<std::pair<unsigned int,float> > > *contributions );

      /*! \brief Helper function to convert the image format from one to another.
       *  \param src A pointer to the constant source Image to scale.
       *  \param dst A pointer to the destination Image.
       *  \param destFormat The destination format.
       *  \remarks If destFormat is the same as the source image, src is simply copied to dst.  If
       *           an image format containing no alpha is converted to a format that contains alpha
       *           (for instance, RGB -> BGRA) the alpha channel will be set to 0xff.  */
      DP_SG_CORE_API void convertPixelFormat( const Image * src, Image *dst, Image::PixelFormat destFormat );

    #if !defined(NDEBUG)
      /*! \brief Asserts mipmap chain to be complete for the specified TextureHost */ 
      DP_SG_CORE_API void assertMipMapChainComplete(const TextureHost * img);
    #else
    # define assertMipMapChainComplete(i) static_cast<void*>(0)    
    #endif

      //! A texture image abstraction class.
      class TextureHost : public Texture
      {
        public:
          /*  \brief Create a TextureHost.
           *  \param filename The name of the texture file.
           *  \note A TextureHost created by this function can be used to explicitly create a texture, without
           *  loading data from a texture file. By default, such a texture has the F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD
           *  flag set. Explicitly unsetting that flag might be dangerous in multi-threaded environments with multiple
           *  render contexts, that don't share resources.
           *  \sa createFromFile */
          DP_SG_CORE_API static TextureHostSharedPtr create( const std::string & filename = "" );

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~TextureHost();

      public:

        //! Enumeration type to describe the format of the data when it is resident 
        //  on the GPU as a texture map
        enum TextureGPUFormat 
        { 
          TGF_DEFAULT, // TYPICALLY, PICK THE FASTEST FORMAT FOR THE GIVEN DATA

          // "generic formats" 
          TGF_FIXED, // USE A FIXED-POINT GPU FORMAT
          TGF_FLOAT, // USE A FLOATING-POINT GPU FORMAT (choose format based on size of float data supplied) 
          TGF_NONLINEAR, // USE AN SRGB FORMAT (choose SRGB/SRGBA based on the data supplied) 
          TGF_INTEGER, // USE AN INTEGER FORMAT, BASED ON THE DATA
          TGF_COMPRESSED_FIXED, // USE A COMPRESSED FIXED POINT FORMAT 
                                // (compress on upload)

          // some "not so generic" formats
          TGF_FLOAT16, // USE 16 BIT PER COMPONENT FLOAT FORMAT 
          TGF_FLOAT32, // USE 32 BIT PER COMPONENT FLOAT FORMAT 

          TGF_FIXED8,  // USE an 8 bit per component fixed point format
          TGF_FIXED10, // USE a 10 BIT PER COMPONENT FIXED POINT FORMAT (10,10,10,2)
          TGF_FIXED16, // USE a 16 bit per component fixed point format

          // explicit formats
          TGF_ALPHA8,
          TGF_ALPHA16,
          TGF_SIGNED_ALPHA8,
          TGF_SIGNED_ALPHA16,
          TGF_LUMINANCE8,
          TGF_LUMINANCE16,
          TGF_SIGNED_LUMINANCE8,
          TGF_SIGNED_LUMINANCE16,
          TGF_LUMINANCE8_ALPHA8,
          TGF_LUMINANCE16_ALPHA16,
          TGF_SIGNED_LUMINANCE8_ALPHA8,
          TGF_SIGNED_LUMINANCE16_ALPHA16,
          TGF_COMPRESSED_LUMINANCE,
          TGF_COMPRESSED_LUMINANCE_ALPHA,

          TGF_COMPRESSED_LUMINANCE_LATC,         // ext_texture_compression_latc
          TGF_SIGNED_COMPRESSED_LUMINANCE_LATC,
          TGF_COMPRESSED_LUMINANCE_ALPHA_LATC,
          TGF_SIGNED_COMPRESSED_LUMINANCE_ALPHA_LATC,
          TGF_COMPRESSED_RED_RGTC,
          TGF_SIGNED_COMPRESSED_RED_RGTC,
          TGF_COMPRESSED_RG_RGTC,
          TGF_SIGNED_COMPRESSED_RG_RGTC,

          TGF_RGB8,
          TGF_SIGNED_RGB8,
          TGF_RGB16,       // native on G80
          TGF_RGBA8,
          TGF_SIGNED_RGBA8,
          TGF_RGB10,       // native on G80
          TGF_RGB10_A2,    // native on G80
          TGF_RGBA16,      // native on G80

          TGF_SRGB,         // ext_texture_srgb
          TGF_SRGBA,
          TGF_COMPRESSED_SRGB,
          TGF_COMPRESSED_SRGBA,
          TGF_COMPRESSED_SRGB_DXT1,
          TGF_COMPRESSED_SRGB_DXT3,
          TGF_COMPRESSED_SRGB_DXT5,
          TGF_COMPRESSED_SRGBA_DXT1,
          TGF_COMPRESSED_RGB,
          TGF_COMPRESSED_RGBA,
          TGF_COMPRESSED_RGB_DXT1,
          TGF_COMPRESSED_RGB_DXT3,
          TGF_COMPRESSED_RGB_DXT5,
          TGF_COMPRESSED_RGBA_DXT1,

          TGF_DEPTH16,              // arb_depth_texture
          TGF_DEPTH24,
          TGF_DEPTH32,

          TGF_DEPTH24_STENCIL8,     // EXT_packed_depth_stencil

          TGF_LUMINANCE16F,         // arb_texture_float
          TGF_LUMINANCE32F,
          TGF_LUMINANCE_ALPHA16F,
          TGF_LUMINANCE_ALPHA32F,
          TGF_RGB16F,
          TGF_RGB32F,
          TGF_RGBA16F,
          TGF_RGBA32F,

          TGF_SIGNED_RGB_INTEGER32, // ext_texture_integer - require an integer
                                    // PixelDataFormat!
          TGF_SIGNED_RGB_INTEGER16,
          TGF_SIGNED_RGB_INTEGER8,
          TGF_RGB_INTEGER32,
          TGF_RGB_INTEGER16,
          TGF_RGB_INTEGER8,

          TGF_SIGNED_RGBA_INTEGER32,
          TGF_SIGNED_RGBA_INTEGER16,
          TGF_SIGNED_RGBA_INTEGER8,
          TGF_RGBA_INTEGER32,
          TGF_RGBA_INTEGER16,
          TGF_RGBA_INTEGER8,

          TGF_SIGNED_LUMINANCE_INTEGER32,
          TGF_SIGNED_LUMINANCE_INTEGER16,
          TGF_SIGNED_LUMINANCE_INTEGER8,
          TGF_LUMINANCE_INTEGER32,
          TGF_LUMINANCE_INTEGER16,
          TGF_LUMINANCE_INTEGER8,

          TGF_SIGNED_LUMINANCE_ALPHA_INTEGER32,
          TGF_SIGNED_LUMINANCE_ALPHA_INTEGER16,
          TGF_SIGNED_LUMINANCE_ALPHA_INTEGER8,
          TGF_LUMINANCE_ALPHA_INTEGER32,
          TGF_LUMINANCE_ALPHA_INTEGER16,
          TGF_LUMINANCE_ALPHA_INTEGER8,

          TGF_UNSIGNED_FLOAT_SHARED_EXPONENT, //USE RGB9_E5 format (EXT_texture_shared_exponent)
          TGF_UNSIGNED_FLOAT_PACKED          // USE R11F_G11F_B10F_EXT format, (EXT_texture_packed_float)
        };

        //! Flags considered for image processing
        enum 
        {
          F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD = BIT0 /*!<
            Serves as a hint to the upload code to keep the image data in process memory even after image upload. */
        , F_IMAGE_STREAM = BIT2 /*!<
            Forces the image to be treated as an image stream, that is, no pixel data will be copied to internal structures. */
        , F_NO_IMAGE_CREATION = BIT4  /*!<
            Only create the information data for this TextureHost, don't load the texture data. */
    
        , F_SCALE_FILTER_BOX = 0                //!< scale the image with a box filter (nearest)
        , F_SCALE_FILTER_TRIANGLE = BIT7        //!< scale the image with a triangle filter (linear)
        , F_SCALE_FILTER_BELL = BIT8            //!< scale the image with a bell filter
        , F_SCALE_FILTER_BSPLINE = BIT7 | BIT8  //!< scale the image with a bspline filter
        , F_SCALE_FILTER_LANCZOS3 = BIT9        //!< scale the image with a lanczos3 filter
        , F_SCALE_FILTER_MASK = BIT7 | BIT8 | BIT9

        , F_SCALE_POT_NEAREST = 0  /*!< always scale to nearest power of two if scaling is required (default) */
        , F_SCALE_POT_ABOVE = BIT16 /*!< always scale to the next power of two if scaling is required */
        , F_SCALE_POT_BELOW = BIT17 /*!< always scale to the highest power of two below if scaling is required */
        , F_SCALE_POT_MASK= F_SCALE_POT_ABOVE | F_SCALE_POT_BELOW
        };

        /*! \brief Sets the TextureTarget, returns false if incompatible with current image data
            \param target The TextureTarget defines how the texture is uploaded to the renderers
            \remarks Renderers require the TextureTarget to be specified in advance. Make sure you have
              appropriate image data specified, otherwise it will always fail to set the type.
        */
        DP_SG_CORE_API bool setTextureTarget( TextureTarget target );

            /*! \brief Tries to set the TextureTarget, returns false if unable to convert the image data
            \param target The TextureTarget defines how the texture is uploaded to the renderers
            \returns When the conversion was successful or texture's own target already matches, then \c true is returned.
              If the texture has a type other than \c TT_UNSPECIFIED_TEXTURE_TARGET or conversion is impossible \c false is returned.
            \remarks Renderers require the TextureTarget to be specified in advance. Changing the target can only be performed if
              the texture's target is TT_UNSPECIFIED_TEXTURE_TARGET.
        */
        DP_SG_CORE_API bool convertToTextureTarget( TextureTarget target );

        /*! \brief Sets creation flags to consider for further image processing. 
          * \param flags Creation flags to consider for further image processing.
          * \remarks
          * \remarks
          * This function lets you specify certain creation flags which affect subsequent 
          * image processing.\n
          * Possible creation flags are:\n
          * F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD\n 
          * F_NO_SCALING\n
          * F_IMAGE_STREAM\n
          * F_SCALE_FILTER_BOX\n
          * F_SCALE_FILTER_BELL\n
          * F_SCALE_FILTER_BSPLINE\n
          * F_SCALE_FILTER_LANCZOS3\n
          * F_SCALE_FILTER_MASK\n */
        DP_SG_CORE_API void setCreationFlags(unsigned int flags);

        //! Sets the textures GPU, or internal, format.
        /** This method sets flags that request that the image be converted to a 
         * particular format when uploaded to the GPU.  Compressed formats can be 
         * selected through this method, and Integer formats can be selected by
         * this method as well.
         * The default is TGF_DEFAULT which is OpenGL's default of using either
         * fixed or floating point format for textures, using the same methods
         * that were used in previous versions of SceniX.  Please see the extended
         * documentation for those defaults.
         * \param fmt The format from the TextureGPUFormat enum to use.
         * \remarks This is a suggestion only - the GPU is free to use a compatible
         * format if the requested format is not available.
         * \sa TextureHost::TextureGPUFormat, TextureHost::getTextureGPUFormat
         */
        DP_SG_CORE_API void setTextureGPUFormat( TextureGPUFormat fmt );

        //! Gets the textures GPU, or internal, format.
        /** This method returns the GPU format that was requested by the user.
         * \remarks Note that this is not necessarily the format that the texture
         * will reside in on the GPU, just what the users selection was.
         * \sa TextureHost::TextureGPUFormat, TextureHost::setTextureGPUFormat
         */
        DP_SG_CORE_API TextureGPUFormat getTextureGPUFormat( void ) const;

        /*! \brief Returns the name of the file from which the TextureHost was created. 
         * \returns The function returns the name of the file from which the TextureHost was created.
         * If the TextureHost was not created from a file, an empty string will be returned.
         * \sa createFromFile, createImage */
        DP_SG_CORE_API const std::string& getFileName() const;

        //! Creates an image.
        /** Use this function to create a single image of the specified dimensions, pixel format,
          * and pixel data type.
          * 
          * If the F_IMAGE_STREAM flag was not specified at object instantiation, the pixel data
          * pointed to by \a pixels will be copied to the internal data structures. 
          * If, on the other hand, the F_IMAGE_STREAM flag was specified at object instantiation, 
          * no pixel data will be copied to the internal structures but only the address specified
          * by \a pixels will be stored. The image will be treated as an image stream in the latter case.
          *
          * Note, that the pixel data pointed to by \a pixels always must match the specified dimensions,
          * pixel format, and data type. Otherwise this would result in undefined behavior.
          *
          * If pixels is 0 uninitialized memory will be allocated for pixel data.
          *
          * The behavior also is undefined if an invalid pointer will be passed through \a pixels.
          *
          * In addition to the level 0 image, additional mipmap levels can be specified through createImage.
          * The user can do so by passing the pointers to the raw pixel data of the additional mipmap levels
          * through \a mipmaps. If an incomplete mipmap chain is specified, the missing levels are added 
          * using the current filter settings. 
          * Note that, specifying an incomplete chain with F_IMAGE_STREAM set is an error; no mipmaps are 
          * created in this case.
          *
          * The passed mipmap levels are assumed to be sorted from level 1 to level \a n. The dimensions
          * of the mipmaps are assumed to be according to the particular level, that is - level 1 mipmap
          * is assumed of dimension (with(level 0) / 2) x (height(level 0) / 2) x (depth(level 0)/2) pixel, level 2 
          * mipmap is assumed of dimension (with(level 1) / 2) x (height(level 1) / 2) x (depth(level 1)/2) pixels,
          * and so on, down to 1 x 1 x 1 pixels for the last level. 
          *
          * The additional mipmap levels are assumed to have the same bytes per pixel as the level 0 image.
          * The additional mipmap levels are also assumed to be of the same pixel format, and data type as
          * the level 0 image.
          *
          * \returns The zero based index of the image if creation was successful.
          * Otherwise the function returns -1 indicating that the image creation failed.
          * \sa Image */
        DP_SG_CORE_API unsigned int createImage( unsigned int           width      //!< width in pixels
                                         , unsigned int           height     //!< height in pixels
                                         , unsigned int           depth      //!< depth in pixels 
                                         , Image::PixelFormat     format     //!< format of image data
                                         , Image::PixelDataType   type       //!< type of image data
                                         , const void           * pixels = 0 //!< raw pixel data 
                                         , const std::vector<const void*>& mipmaps = std::vector<const void*>() /**!< 
                                                                           additional mipmap levels (optional)*/
                                         );

        /*! \brief Add an Image structure without pixel data to the TextureHost.
         *  \param width The width of the Image to add.
         *  \param height The height of the Image to add.
         *  \param depth The depth of the Image to add.
         *  \param format The PixelFormat of the Image to add.
         *  \param type The PixelDataType of the Image to add.
         *  \return The index of the Image in this TextureHost.
         *  \remarks This function adds an Image structure to the TextureHost, but does not allocate
         *  memory for the pixel data.
         *  \sa createImage, setImageData, Image */
        DP_SG_CORE_API unsigned int addImage( unsigned int width
                                      , unsigned int height
                                      , unsigned int depth
                                      , Image::PixelFormat format
                                      , Image::PixelDataType type );

        /*! \brief Set the pixel data of an Image.
         *  \param image The index of the Image to set the data.
         *  \param pixels A pointer to the pixel data to set.
         *  \param mipmaps An optional vector of pointers to pixel data for the mipmaps.
         *  \remarks If the TextureHost is a streaming image (F_IMAGE_STREAM is set), the pointer to
         *  the pixel data of the Image at index \a image just points to \a pixels. Any data in
         *  \a mipmaps is ignored then.\n
         *  Otherwise the memory for the pixels are allocated if needed and copied to the Image at
         *  index \a image. If there are mipmaps provided, the data is also copied.
         *  \note The behavior is undefined if \a image is greater or equal to the number of images
         *  in this TextureHost. Furthermore, it is assumed that \a pixels point to memory that is
         *  large enough to hold the complete data of the Image. If \a mipmaps are provided, it is
         *  assumed that each of the pointers in \a mipmaps point to memory that is large enough to
         *  hold the complete data of the corresponding mipmap level of the Image. If an incomplete mipmap chain 
         *  is specified, the missing levels are added using the current filter settings. 
         *  Note that, specifying an incomplete chain with F_IMAGE_STREAM set is an error; no mipmaps are 
         *  created in this case.
         *  \sa createImage, addImage, Image */
        DP_SG_CORE_API void setImageData( unsigned int image
                                  , const void * pixels
                                  , const std::vector<const void *> & mipmaps = std::vector<const void *>() );

        /*! \brief Set the pixel data of an Image.
         *  \param image The index of the Image to set the data.
         *  \param pixels A Buffer to the pixel data to set.
         *  \param mipmaps An optional vector of Buffers to pixel data for the mipmaps.
         *  \remarks TextureHost will keep a reference to the passed in buffers. It'll not do a copy. Thus
         *  this could be used to stream data into textures.
         *  \note The behavior is undefined if \a image is greater or equal to the number of images
         *  in this TextureHost. Furthermore, it is assumed that \a pixels point to memory that is
         *  large enough to hold the complete data of the Image. If \a mipmaps are provided, it is
         *  assumed that each of the pointers in \a mipmaps point to memory that is large enough to
         *  hold the complete data of the corresponding mipmap level of the Image. If an incomplete mipmap chain 
         *  is specified, the missing levels are added using the current filter settings. 
         *  Note that, specifying an incomplete chain with F_IMAGE_STREAM set is an error; no mipmaps are 
         *  created in this case.
         *  \sa createImage, addImage, Image */
        DP_SG_CORE_API void setImageData( unsigned int image
                                  , const BufferSharedPtr & pixels
                                  , const std::vector<BufferSharedPtr> & mipmaps = std::vector<BufferSharedPtr>() );

        /*! \brief Creates all mipmap levels for all included images.
         *  \return \c true if all levels have been created successfully, \c false otherwise.
         *  \remarks This function explicitly creates all mipmap levels for all images included.
         *  Previously created mipmap levels will be released.\n
         *  If the function is called for a TextureHost with the F_IMAGE_STREAM flag specified,
         *  no mipmap levels are created, and the function returns \c false. Also, if the
         *  function is called for a TextureHost containing images of a color index format,
         *  any compressed formats, depth component format, or depth stencil format, no mipmap
         *  levels are created, and the function returns \c false.\n
         *  The behavior is undefined if any of the images is not a power-of-two image or if
         *  any of the images has overlaps. */
        DP_SG_CORE_API bool createMipmaps();

        /*! \brief Get the number of Images of the TextureHost (e.g. 6 for cubemaps).
         *  \return The number of images of the corresponding TextureHost.
         *  \sa getNumberOfMipmaps */
        DP_SG_CORE_API unsigned int getNumberOfImages() const; 
    
        /*! \brief Get the number of mipmap levels for the specified image.
         *  \param image An optional index of the image to get the number of mipmap levels for.
         *  \return The number of mipmap levels of the \a image.
         *  \note The behavior is undefined if \a image is larger than or equal to the number of
         *  images this TextureHost holds.
         *  \sa getNumberOfImages */
        DP_SG_CORE_API unsigned int getNumberOfMipmaps( unsigned int image=0 //!< Zero-based image index.
                                                ) const;

        //! Returns the PixelFormat of the specified mipmap level for a given image.
        /** The function returns the PixelFormat of the pixel data stored at mipmap level \a mipmap 
          * of the image specified by \a image. See the PixelFormat enumeration type for known pixel formats.
          * A mipmap level of 0 always identifies the image itself.
          *
          * Passing an invalid image index as well as an invalid mipmap level results in undefined behavior! 
          * A valid image index ranges from 0 to getNumberOfImages()-1.
          * A valid mipmap level ranges from 0 to getNumberOfMipmaps(image)-1.
          *
          * \returns A PixelFormat enum specifying the pixel format. */
        DP_SG_CORE_API Image::PixelFormat getFormat( unsigned int image=0  //!< Zero-based image index.
                                             , unsigned int mipmap=0 //!< Zero-based mipmap level.
                                             ) const;

        //! Returns the PixelDataType of the specified image.
        /** The function returns the PixelDataType of the pixel data stored at mipmap level \a mipmap 
          * of the image specified by \a image. See the PixelDataType enumeration type for known types. 
          * A mipmap level of 0 always identifies the image itself.
          *
          * Passing an invalid image index as well as an invalid mipmap level results in undefined behavior! 
          * A valid image index ranges from 0 to getNumberOfImages()-1.
          * A valid mipmap level ranges from 0 to getNumberOfMipmaps(image)-1.
          *
          * \returns A PixelDataType enum specifying the pixel data type. */
        DP_SG_CORE_API Image::PixelDataType getType( unsigned int image=0  //!< Zero-based image index.
                                             , unsigned int mipmap=0 //!< Zero-based mipmap level.
                                             ) const;

        /*! \brief Get the width of the specified image in pixels.
         *  \param image Optional index of the image to get width of. Default is zero.
         *  \param mipmap Optional index of the mipmap level of the specified image to get width of.
         *  Default is zero, that is the top level image.
         *  \return The width of the specified image in pixels.
         *  \note The behavior is undefined if \a image is larger or equal to the number of images in
         *  this TextureHost, or if \a mipmap is larger or equal to the number of mipmap levels of
         *  the specified image.
         *  \note The returned value is the width over the complete image, including overlaps.
         *  \sa getHeight, getDepth, getOverlapSize */
        DP_SG_CORE_API unsigned int getWidth( unsigned int image=0
                                      , unsigned int mipmap=0
                                      ) const;

        /*! \brief Get the height of the specified image in pixels.
         *  \param image Optional index of the image to get width of. Default is zero.
         *  \param mipmap Optional index of the mipmap level of the specified image to get width of.
         *  Default is zero, that is the top level image.
         *  \return The height of the specified image in pixels.
         *  \note The behavior is undefined if \a image is larger or equal to the number of images in
         *  this TextureHost, or if \a mipmap is larger or equal to the number of mipmap levels of
         *  the specified image.
         *  \note The returned value is the height over the complete image, including overlaps.
         *  \sa getWidth, getDepth, getOverlapSize */
        DP_SG_CORE_API unsigned int getHeight( unsigned int image=0
                                       , unsigned int mipmap=0
                                       ) const;

        /*! \brief Get the depth of the specified image in pixels.
         *  \param image Optional index of the image to get width of. Default is zero.
         *  \param mipmap Optional index of the mipmap level of the specified image to get width of.
         *  Default is zero, that is the top level image.
         *  \return The depth of the specified image in pixels.
         *  \note The behavior is undefined if \a image is larger or equal to the number of images in
         *  this TextureHost, or if \a mipmap is larger or equal to the number of mipmap levels of
         *  the specified image.
         *  \note The returned value is the depth over the complete image, including overlaps.
         *  \sa getWidth, getHeight, getOverlapSize */
        DP_SG_CORE_API unsigned int getDepth( unsigned int image=0
                                      , unsigned int mipmap=0
                                      ) const;

        //! Returns the number of bytes per pixel of the specified mipmap.
        /** The function returns the number of bytes per pixel of the mipmap at level \a mipmap of the image 
          * specified by \a image. A mipmap level of 0 always identifies the image itself. 
          *
          * Passing an invalid image index as well as an invalid mipmap level results in undefined behavior! 
          * A valid image index ranges from 0 to getNumberOfImages()-1.
          * A valid mipmap level ranges from 0 to getNumberOfMipmaps(image)-1.
          *
          * \returns The number of bytes per pixel of the specified mipmap. */
        DP_SG_CORE_API unsigned int getBytesPerPixel( unsigned int image=0  //!< Zero-based image index.
                                              , unsigned int mipmap=0 //!< Zero-based mipmap level.
                                              ) const;

        /*! \brief Get the number of bytes per line of the specified mipmap level of the specified image.
         *  \param image An optional index of the image to get information from. The default value
         *  is zero.
         *  \param mipmap An optional index of the mipmap level to get information from. The
         *  default value is zero.
         *  \return The number of bytes per line of the Image at mipmap level \a mipmap at the
         *  index \a image.
         *  \remarks A mipmap level of zero identifies the image itself.
         *  \note The behavior is undefined if \a image is larger than or equal to the number of
         *  images or if \a mipmap is larger than or equal to the number of mipmap levels of the
         *  Image \a image this TextureHost holds information for.
         *  \sa getNumberOfImages, getNumberOfMipmaps, getBytesPerPixel, getBytesPerPlane,
         *  getNumberOfBytes */
        DP_SG_CORE_API unsigned int getBytesPerScanLine( unsigned int image=0, unsigned int mipmap=0 ) const;
    
        /*! \brief Get the number of bytes per plane (slice) of the specified mipmap level of the specified image.
         *  \param image An optional index of the image to get information from. The default value
         *  is zero.
         *  \param mipmap An optional index of the mipmap level to get information from. The
         *  default value is zero.
         *  \return The number of bytes per plane (slice) of the Image at mipmap level \a mipmap at
         *  the index \a image.
         *  \remarks A mipmap level of zero identifies the image itself.
         *  \note The behavior is undefined if \a image is larger than or equal to the number of
         *  images or if \a mipmap is larger than or equal to the number of mipmap levels of the
         *  Image \a image this TextureHost holds information for.
         *  \sa getNumberOfImages, getNumberOfMipmaps, getBytesPerPixel, getBytesPerScanLine,
         *  getNumberOfBytes */
        DP_SG_CORE_API unsigned int getBytesPerPlane( unsigned int image=0, unsigned int mipmap=0 ) const;

        /*! \brief Get the number of bytes of the specified mipmap level of the specified image.
         *  \param image An optional index of the image to get information from. The default value
         *  is zero.
         *  \param mipmap An optional index of the mipmap level to get information from. The
         *  default value is zero.
         *  \return The number of bytes of the Image at mipmap level \a mipmap at the index \a image.
         *  \remarks A mipmap level of zero identifies the image itself.
         *  \note The behavior is undefined if \a image is larger than or equal to the number of
         *  images or if \a mipmap is larger than or equal to the number of mipmap levels of the
         *  Image \a image this TextureHost holds information for.
         *  \sa getNumberOfImages, getNumberOfMipmaps, getBytesPerPixel, getBytesPerScanLine,
         *  getBytesPerPlane */
        DP_SG_CORE_API unsigned int getNumberOfBytes( unsigned int image=0, unsigned int mipmap=0 ) const;

        /*! \brief Returns the number of bytes used for all included images and mipmap levels.
         * \return The number of bytes used for all included images and mipmap levels. */
        DP_SG_CORE_API unsigned int getTotalNumberOfBytes() const;

    #if 0
        //! Returns a pointer to the raw pixel data of the specified mipmap.
        /** The function returns a pointer to the raw pixel data of the mipmap at level \a mipmap of the image
          * specified by \a image. A mipmap level of 0 always identifies the image itself. 
          *
          * Passing an invalid image index as well as an invalid mipmap level results in undefined behavior! 
          * A valid image index ranges from 0 to getNumberOfImages()-1.
          * A valid mipmap level ranges from 0 to getNumberOfMipmaps(image)-1.
          *
          * \returns A pointer to the raw pixel data of the specified mipmap. */
        DP_SG_CORE_API const void * getPixels( unsigned int image = 0  //!< Zero-based image index.
                                       , unsigned int mipmap = 0 //!< Zero-based mipmap level.
                                       ) const;
    #endif
        //! Returns a Buffer containing the the raw pixel data of the specified mipmap.
        /** The function returns a Buffer of the raw pixel data of the mipmap at level \a mipmap of the image
          * specified by \a image. A mipmap level of 0 always identifies the image itself. 
          *
          * Passing an invalid image index as well as an invalid mipmap level results in undefined behavior! 
          * A valid image index ranges from 0 to getNumberOfImages()-1.
          * A valid mipmap level ranges from 0 to getNumberOfMipmaps(image)-1.
          *
          * \returns A Buffer of the raw pixel data of the specified mipmap. */
        DP_SG_CORE_API BufferSharedPtr getPixels( unsigned int image = 0  //!< Zero-based image index.
                                       , unsigned int mipmap = 0 //!< Zero-based mipmap level.
                                       ) const;
    
        /*! \brief getSubImagePixels copies a portion of an existing 1-, 2-, or 3-dimensional image.
          * \param 
          * image Index into vector of images.
          * \param 
          * mipmap Zero-based mipmap level
          * \param 
          * x_offset Pixel offset in the x direction of the image
          * \param 
          * y_offset Pixel offset in the y direction of the image
          * \param 
          * z_offset Pixel offset in the z direction of the image
          * \param
          * width Width of the sub image
          * \param
          * height Height of the sub image
          * \param
          * depth Depth of the sub image
          * \param
          * subPixels Target address to copy data to.
          * \return The function returns \c true on success. 
          * Otherwise it returns \c false (for example if the image does not exist (e.g. after upload),
          * or if the sub image boundary exceeds the existing image)
          * \remarks
          * The function copies \a width x \a height x \a depth x getBytesPerPixel() raw pixel data from
          * the specified offset of the existing image to the buffer pointed to by \a subPixels.
          * \n \n
          * The user is responsible for providing a buffer large enough to hold the requested data.
          * \sa setSubImagePixels
          */
        DP_SG_CORE_API bool getSubImagePixels( unsigned int image
                                       , unsigned int mipmap
                                       , unsigned int x_offset
                                       , unsigned int y_offset
                                       , unsigned int z_offset
                                       , unsigned int width
                                       , unsigned int height
                                       , unsigned int depth
                                       , void * subPixels
                                       ) const;

        /*! \brief getSubImagePixels copies a portion of an existing 1-, 2-, or 3-dimensional image.
          * \param 
          * image Index into vector of images.
          * \param 
          * mipmap Zero-based mipmap level
          * \param 
          * x_offset Pixel offset in the x direction of the image
          * \param 
          * y_offset Pixel offset in the y direction of the image
          * \param 
          * z_offset Pixel offset in the z direction of the image
          * \param
          * width Width of the sub image
          * \param
          * height Height of the sub image
          * \param
          * depth Depth of the sub image
          * \param
          * subPixels Buffer to copy data to. The buffer will be resized to fit the image.
          * \return The function returns \c true on success. 
          * Otherwise it returns \c false (for example if the image does not exist (e.g. after upload),
          * or if the sub image boundary exceeds the existing image)
          * \remarks
          * The function copies \a width x \a height x \a depth x getBytesPerPixel() raw pixel data from
          * the specified offset of the existing image to the buffer pointed to by \a subPixels.
          * \n \n
          * The user is responsible for providing a buffer large enough to hold the requested data.
          * \sa setSubImagePixels
          */
        DP_SG_CORE_API bool getSubImagePixels( unsigned int image
                                       , unsigned int mipmap
                                       , unsigned int x_offset
                                       , unsigned int y_offset
                                       , unsigned int z_offset
                                       , unsigned int width
                                       , unsigned int height
                                       , unsigned int depth
                                       , const BufferSharedPtr &buffer
                                       ) const;

        /*! \brief setSubImagePixels sets a portion of an existing 1-, 2-, or 3-dimensional image.
         * \param 
         * image Index into vector of images
         * \param 
         * mipmap Zero-based mipmap level
         * \param 
         * x_offset Pixel offset in the x direction of the image
         * \param 
         * y_offset Pixel offset in the y direction of the image
         * \param 
         * z_offset Pixel offset in the z direction of the image
         * \param
         * width Width of the sub image
         * \param
         * height Height of the sub image
         * \param
         * depth Depth of the sub image
         * \param
         * subPixels Source address to copy data from.
         * \return The function returns \c true on success. 
         * Otherwise it returns \c false (for example if the image does not exist (e.g. after upload),
         * or if the sub image boundary exceeds the existing image)
         * \remarks
         * The function sets \a width x \a height x \a depth x getBytesPerPixel() pixel data from
         * the specified offset of the existing image to that of the raw data in the buffer pointed to by \a subPixels.
         * \n \n
         * The PixelDataType of the data pointed to by subPixels must match the PixelDataType 
         * of the existing image. Behavior would be undefined otherwise.
         * \sa getSubImagePixels
         */
        DP_SG_CORE_API bool setSubImagePixels( unsigned int image
                                       , unsigned int mipmap
                                       , unsigned int x_offset
                                       , unsigned int y_offset
                                       , unsigned int z_offset
                                       , unsigned int width
                                       , unsigned int height
                                       , unsigned int depth
                                       , const void * subPixels
                                       );

        /*! \brief setSubImagePixels sets a portion of an existing 1-, 2-, or 3-dimensional image.
         * \param 
         * image Index into vector of images
         * \param 
         * mipmap Zero-based mipmap level
         * \param 
         * x_offset Pixel offset in the x direction of the image
         * \param 
         * y_offset Pixel offset in the y direction of the image
         * \param 
         * z_offset Pixel offset in the z direction of the image
         * \param
         * width Width of the sub image
         * \param
         * height Height of the sub image
         * \param
         * depth Depth of the sub image
         * \param
         * subPixels Buffer to copy data from.
         * \return The function returns \c true on success. 
         * Otherwise it returns \c false (for example if the image does not exist (e.g. after upload),
         * or if the sub image boundary exceeds the existing image)
         * \remarks
         * The function sets \a width x \a height x \a depth x getBytesPerPixel() pixel data from
         * the specified offset of the existing image to that of the raw data in the buffer pointed to by \a subPixels.
         * \n \n
         * The PixelDataType of the data pointed to by subPixels must match the PixelDataType 
         * of the existing image. Behavior would be undefined otherwise.
         * \sa getSubImagePixels
         */
        DP_SG_CORE_API bool setSubImagePixels( unsigned int image
                                       , unsigned int mipmap
                                       , unsigned int x_offset
                                       , unsigned int y_offset
                                       , unsigned int z_offset
                                       , unsigned int width
                                       , unsigned int height
                                       , unsigned int depth
                                       , const BufferSharedPtr &pixelBuffer
                                       );

        //! Returns the alignment for the start of each pixel row in memory. 
        DP_SG_CORE_API unsigned int getUnpackAlignment() const;

        /*! \brief Scale the specified image
         *  \param image The index of the image to scale.
         *  \param width The desired width of the scaled image.
         *  \param height The optional height of the scaled image. The default is 1.
         *  \param depth The optional depth of the scaled image. The default is 1.
         *  \return \c true if the scaling was successful, \c false otherwise.
         *  \remarks The function scales the image specified by \a image according to the input
         *  dimensions passed. The filter to use is specified by the creation flags.
         *  \note The behavior is undefined if \a image is larger than or equal to the number of
         *  images in this TextureHost, of if that image has overlaps. */
        DP_SG_CORE_API bool scale( unsigned int image      //!< Zero-based index of the image to scale.  
                           , unsigned int width      //!< Desired width of the scaled image.
                           , unsigned int height = 1 //!< Desired height of the scaled image.
                           , unsigned int depth = 1  //!< Desired depth of the scaled image.
                           );

        /*! \brief Change all images within this TextureHost to the given format.
         *  \param newFormat The new pixel format for the entire TextureHost.
         *  \remarks This function changes the pixel format of all images and all
         *           mipmaps contained in this TextureHost to the format supplied.
         *           If a format that doesn't contain an alpha channel is converted to
         *           one that does (ie: RGB -> BGRA) then the alpha channel is set to
         *           0xff. */
        DP_SG_CORE_API bool convertPixelFormat( Image::PixelFormat newFormat);      //!< the new pixel format.  
    
        /*! \brief Scales containing images to power-of-two
         * \param sizeLimit 
         * Specifies the platform dependent size limit that must not be exceeded during scaling.
         * \return 
         * \c true if scaling was successful, \c false otherwise. 
         *
         * The function scales all containing images to become power of two in each dimension. 
         * By default, the function always scales to the nearest power of two. 
         * This behavior can be changed by specifying a certain creation flag at TextureHost 
         * instantiation. If you specified F_SCALE_POT_BELOW, the single dimensions will always 
         * be scaled down to the highest power of two below the actual size. If you specified 
         * F_SCALE_POT_ABOVE, the single dimensions always will be scaled up to the very next 
         * power of two. If both flags were specified at instantiation, the function falls back
         * to the default algorithm. Independent of the algorithm used for scaling, the function 
         * always clamps each dimension to the specified \a sizeLimit.
         * 
         * The behavior is undefined if called for a TextureHost that was instantiated with the
         * F_IMAGE_STREAM flag specified. The behavior also is undefined if called for a TextureHost
         * that does not contain any images or for which all images already have been released 
         * (e.g. after upload).
         *
         * \note 
         * If the scaling failed, the function left behind the pixel data of containing images 
         * undefined!*/
        DP_SG_CORE_API bool scaleToPowerOfTwo(unsigned int sizeLimit);

        //! Returns whether the texture image represents a 1-dimensional texture image.
        /** \returns \c true if the texture image represents a 1-dimensional texture image, \c false otherwise. */
        DP_SG_CORE_API bool is1D()            const;
    
        //! Returns whether the texture image represents a 2-dimensional texture image.
        /** \returns \c true if the texture image represents a 2-dimensional texture image, \c false otherwise. */
        DP_SG_CORE_API bool is2D()            const;
    
        //! Returns whether the texture image represents a 2-dimensional texture image array.
        /** \returns \c true if the texture image represents a 2-dimensional texture image array, \c false otherwise. */
        DP_SG_CORE_API bool is2DArray()            const;
    
        //! Returns whether the texture image represents a 1-dimensional texture image array.
        /** \returns \c true if the texture image represents a 1-dimensional texture image array, \c false otherwise. */
        DP_SG_CORE_API bool is1DArray()            const;
    
        //! Returns whether the texture image represents a 3-dimensional texture image.
        /** \returns \c true if the texture image represents a 3-dimensional texture image, \c false otherwise. */
        DP_SG_CORE_API bool is3D()            const;
    
        //! Returns whether the texture image represents a cube map.
        /** A cube map consists of 6 images of dimension 2.
          * \returns \c true if the texture image represents a cube map, \c false otherwise. */
        DP_SG_CORE_API bool isCubeMap()       const;

        //! Returns whether the texture image represents a cube map array.
        /** A cube map array consists of n * 6 images of dimension 2.
          * \returns \c true if the texture image represents a cube map array, \c false otherwise. */
        DP_SG_CORE_API bool isCubeMapArray()       const;
    
        //! Returns whether the texture image represents a floating point texture image.
        /** \returns \c true if the texture image represents a floating point texture image, \c false otherwise. */
        DP_SG_CORE_API bool isFloatingPoint() const;

        //! Returns whether the texture image represents a fixed point texture image.
        /** \returns \c true if the texture image represents a fixed point texture image, \c false otherwise. */
        DP_SG_CORE_API bool isFixedPoint() const;

        //! Returns whether the texture image represents a compressed texture image.
        /** \returns \c true if the texture image represents a compressed texture image, \c false otherwise. */
        DP_SG_CORE_API bool isCompressed() const;

        //! Returns whether the texture image is treated as an image stream.
        /** Image streams are images for which the image data is not managed inside the TextureHost object. 
          * \returns \c true if the image is treated as a image stream, \c false otherwise. */
        DP_SG_CORE_API bool isImageStream() const;

        //! Will convert a 2D image to 6 cubemap faces.
        /** The 2D image must contain all faces in the vertical cross format,
          * as it can be exported e.g. from HDRShop.
          * Thus, the height of the 2D image must be 4/3 of the width. 
          * This function does not work with image streams. */    
        DP_SG_CORE_API void convert2DToCubeMap();

        //! Forces a re-upload of the texture image.
        /** The function marks the texture image as outdated, and hence, forces a re-upload with the
          * very next render step.
          *
          * The function in particular is useful for stream images, that is - images for which the image data
          * is not managed inside the TextureHost object. For these kind of images the TextureHost object
          * does not automatically realize any changes made to the image data but needs a hint from the user.
          * Whenever the image data have changed and, as a consequence, a re-upload of the texture image is
          * required, the user should call demandUpload to force the re-upload with the very next render step.
          *
          * Note, that a re-upload of the texture image will not allocate a new platform dependent texture
          * resource object but will reuse the former resource. */
        DP_SG_CORE_API void demandUpload();

        //! Returns whether a re-upload of the texture image is required.
        /** \returns \c true if a re-upload is required, \c false otherwise. */ 
        DP_SG_CORE_API bool isUploadDemanded() const;

        //! Finishes an image upload.
        /** The function should be called after the image upload succeed. It marks the image as being uploaded. 
          *  \param preserve If preserve is \c true, then the state of the F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD flag is ignored and
          *                  the image is always preserved.  If preserve is \c false, then the state of F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD
          *                  flag is used to determine whether data is deleted.  SceneRenderers such as SceneRendererGL and SceneRendererRT
          *                  can be configured to set this \c preserve flag to \c true and therefore to preserve all image data.  This can
          *                  be important rasterizing and ray tracing the same scene is desired.
          * If neither the F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD nor the F_IMAGE_STREAM flag was specified at TextureHost object 
          * instantiation (and \c preserve parameter is \c false), the function releases all internal image data. 
          * \note A re-upload is not possible after the internal image data was released. */
        DP_SG_CORE_API void finishUpload( );

        //! Returns the creation flag status of the TextureHost object.
        /** \returns An unsigned integer value representing the actual creation flag status of this object. */
        DP_SG_CORE_API unsigned int getCreationFlags() const;

        //! Assigns new contents to a TextureHost object.
        /** The contents of the object specified by \a rhs will be assigned to this object. */
        DP_SG_CORE_API TextureHost& operator=(const TextureHost& rhs);

        //! Mirror the image over the x-axis
        /** This is equivalent to reversing the rows, thus converting between origin upper left and lower left. */
        DP_SG_CORE_API void mirrorX( unsigned int image      //!< Zero-based index of the image to mirror.
          );

        //! Mirror the image over the y-axis
        /** This is equivalent to reversing the columns. */
        DP_SG_CORE_API void mirrorY( unsigned int image      //!< Zero-based index of the image to mirror.
          );

        /*! \brief Overrides Texture::isEquivalent.
          * \param p Pointer to the Texture to test for equivalence with this TextureHost object.
          * \param deepCompare The function performs a deep-compare instead of a shallow compare if this is \c true.
          * \return The function returns \c true if the Texture pointed to by \a p is detected to be equivalent
          * to this TextureHost object.
          * \sa Texture::isEquivalent */
        DP_SG_CORE_API virtual bool isEquivalent( TextureSharedPtr const& texture, bool deepCompare ) const;

      protected:
        /*! \brief Default-constructs a TextureHost.
         *  \param filename The name of the texture file.
         * A default created TextureHost uses a triangle filter for scaling and mipmap creation.
         * Image data later specified through createImage will be copied to the internal data 
         * structures and released after image upload. */
        DP_SG_CORE_API TextureHost( const std::string & filename );

        /*! \brief Constructs a TextureHost as a copy of another TextureHost. */
        DP_SG_CORE_API TextureHost(const TextureHost&);

        /*! \brief Feed the data of this object into the provied HashGenerator.
         *  \param hg The HashGenerator to update with the data of this object.
         *  \sa getHashKey */
        DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

      private:
        // internal flags
        enum
        {
          F_UPLOAD_DEMANDED = BIT0
        };

        // helper with copying images. in particular pixel data
        void copyImages( const std::vector<std::vector<Image> > & srcImages );

        // mipmap management helpers
        bool createMipmaps( std::vector<Image> &images, const std::vector<const void*>& mipmaps );
        bool createMipmaps( std::vector<Image> &images, const std::vector<BufferSharedPtr>& mipmaps );
        bool createMipmaps( std::vector<Image> &images );
        void releaseMipmaps();
        void releaseMipmaps( std::vector<Image> &images );

        // private helper to recalculate the total amount of image bytes 
        void recalcTotalNumberOfBytes();
    
        //  scaling helper functions
        void filter( const Image * src, Image * dst, ScaleFilter * sf );

        unsigned int                      m_creationFlags;
        unsigned int                      m_internalFlags;
        unsigned int                      m_totalNumBytes; 
        std::string                       m_filename; // if created from file
        std::vector<std::vector<Image> >  m_images;
        TextureGPUFormat                  m_gpuFormat;
      };

      TextureHostSharedPtr createStandardTexture();


      inline const std::string& TextureHost::getFileName() const
      {
        return m_filename;
      }

      inline bool TextureHost::isUploadDemanded() const
      {
        return m_internalFlags & F_UPLOAD_DEMANDED;
        // no need to invalidateHashKey on change of m_internalFlags
      }

      inline unsigned int TextureHost::getCreationFlags() const
      {
        return m_creationFlags;
      }

      inline unsigned int TextureHost::getNumberOfImages() const
      {
        return( dp::checked_cast<unsigned int>(m_images.size()) );
      }

      inline unsigned int TextureHost::getNumberOfMipmaps(unsigned int image) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( 0 < m_images[image].size() ) );
        return( dp::checked_cast<unsigned int>(m_images[image].size() - 1) );
      }

      inline Image::PixelFormat TextureHost::getFormat(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_format );
      }

      inline Image::PixelDataType TextureHost::getType(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_type );
      }

      inline TextureHost::TextureGPUFormat TextureHost::getTextureGPUFormat() const
      {
        return( m_gpuFormat );
      }

      inline unsigned int TextureHost::getWidth(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_width );
      }

      inline unsigned int TextureHost::getHeight(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_height );
      }

      inline unsigned int TextureHost::getDepth(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_depth );
      }

      inline unsigned int TextureHost::getBytesPerPixel(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_bpp );
      }

      inline unsigned int TextureHost::getBytesPerScanLine(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_bpl );
      }

      inline unsigned int TextureHost::getBytesPerPlane(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_bps );
      }

      inline unsigned int TextureHost::getNumberOfBytes(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_nob );
      }

      inline unsigned int TextureHost::getTotalNumberOfBytes() const
      {
    #if !defined(NDEBUG)
        unsigned int dbgTotalNumBytes = 0;
        for ( size_t i=0; i<m_images.size(); ++i )
        {
          for ( size_t j=0; j<m_images[i].size(); ++j )
          {
            dbgTotalNumBytes += m_images[i][j].m_nob;
          }
        }
        DP_ASSERT(dbgTotalNumBytes==m_totalNumBytes);
    #endif
        return m_totalNumBytes;
      }

      inline BufferSharedPtr TextureHost::getPixels(unsigned int image, unsigned int mipmap) const
      {
        DP_ASSERT( ( image < m_images.size() ) && ( mipmap < m_images[image].size() ) );
        return( m_images[image][mipmap].m_pixels );
      }

      inline unsigned int TextureHost::getUnpackAlignment() const
      {
        return getBytesPerPixel()==4 ? 4 : 1;
      }

      inline bool TextureHost::is1D() const
      {
        return (  1==m_images.size() 
                && 1==m_images[0][0].m_depth 
                && 1==m_images[0][0].m_height
                && 1<=m_images[0][0].m_width );
      }

      inline bool TextureHost::is2D() const
      {
        return (  1==m_images.size() 
                && 1==m_images[0][0].m_depth 
                && 1<=m_images[0][0].m_width
                && 1<=m_images[0][0].m_height );
      }

      inline bool TextureHost::is3D() const
      {
        return (  1==m_images.size() 
                && 1<=m_images[0][0].m_depth
                && 1<=m_images[0][0].m_width
                && 1<=m_images[0][0].m_height );
      }

      inline bool TextureHost::is2DArray() const
      {
        // Will conflict with cube map if you have the user
        // has defined an array of 6 square textures.  Not sure
        // how to handle this.


        // must have at least 2 images to identify as array.
        if( 1 < m_images.size() 
            && 1==m_images[0][0].m_depth
            && 1<=m_images[0][0].m_width
            && 1<=m_images[0][0].m_height )
        {
          unsigned int width = m_images[0][0].m_width;
          unsigned int height = m_images[0][0].m_height;

          for(unsigned int i=1; i<m_images.size(); i++ )
          {
            if( m_images[i][0].m_width != width ||
                m_images[i][0].m_height != height ||
                m_images[i][0].m_depth != 1 )
            {
              return false;
            }
          }

          return true;
        }
        else
        {
          return false;
        }
      }

      inline bool TextureHost::is1DArray() const
      {
        // must have at least 2 images to identify as array.
        if( 1 < m_images.size() 
            && 1==m_images[0][0].m_depth
            && 1<=m_images[0][0].m_width
            && 1==m_images[0][0].m_height )
        {
          unsigned int width = m_images[0][0].m_width;

          for(unsigned int i=1; i<m_images.size(); i++ )
          {
            if( m_images[i][0].m_width != width ||
                m_images[i][0].m_height != 1 ||
                m_images[i][0].m_depth != 1 )
            {
              return false;
            }
          }

          return true;
        }
        else
        {
          return false;
        }
      }

      inline bool TextureHost::isCubeMap() const
      {
        if (  6==m_images.size() 
              && 1==m_images[0][0].m_depth
              && 1<=m_images[0][0].m_width
              && 1<=m_images[0][0].m_height
              // must be square
              && m_images[0][0].m_width == m_images[0][0].m_height )
        {
          unsigned int width = m_images[0][0].m_width;

          for(unsigned int i=1; i<m_images.size(); i++ )
          {
            if( m_images[i][0].m_width != width ||
                m_images[i][0].m_height != width ||
                m_images[i][0].m_depth != 1 )
            {
              return false;
            }
          }

          return true;
        }
        else
        {
          return false;
        }

      }

      inline bool TextureHost::isCubeMapArray() const
      {
        bool result = ( m_images.size() > 0
          && m_images.size() % 6 == 0 
          && 1==m_images[0][0].m_depth
          && 1<=m_images[0][0].m_width
          && 1<=m_images[0][0].m_height
          // must be square
          && m_images[0][0].m_width == m_images[0][0].m_height );

        if (result)
        {
          unsigned int width = m_images[0][0].m_width;

          for(unsigned int i=1; i<m_images.size(); i++ )
          {
            if( m_images[i][0].m_width != width ||
              m_images[i][0].m_height != width ||
              m_images[i][0].m_depth != 1 )
            {
              result = false;
              break;
            }
          }
        }
        return result;
      }

      inline bool TextureHost::isImageStream() const
      {
        return !!(m_creationFlags & F_IMAGE_STREAM);
      }

      inline void TextureHost::demandUpload()
      {
        m_internalFlags |= F_UPLOAD_DEMANDED;
        // no need to invalidateHashKey on change of m_internalFlags
      }

      //! Scale Filter Functor
      /** Functor base class used in texture image scaling.\n
        * When scaling a TextureHost, a ScaleFilter is used to determine the weight for a source texel in a given
        * distance to the destination texel.\n
        * With these ScaleFilters, the image is resampled in two passes, stretching or shrinking the image first
        * horizontally and then vertically (or vice versa) with potentially different scale factors. The two-pass
        * approach has a significantly lower run-time cost, \b O(imageWidth*imageHeight*(filterWidth+filterHeight)),
        * than straightforward 2-D filtering, \b O(imageWidth*imageHeight*filterWidth*filterHeight)). The performance
        * gain is even higher in 3-D filtering.\n
        * For magnification, the ScaleFilter is centered around each source texel and scaled to the height of that
        * texel. For each destination texel, the corresponding location in the source image is calculated. The sum
        * of the weighted filter functions at this point determines the value of the destination texel.\n
        * For minification, to eliminate all frequency components above the resampling Nyquist frequency, the
        * ScaleFilter is stretched by the image reduction factor.\n
        * Because the scaling process for each row or column is identical, the placement and area of the ScaleFilter
        * is fixed. Therefore the contributors to each destination texel and the corresponding filter weight are
        * precomputed once, using the calling operator, and then applied to each destination texel. */
      class ScaleFilter
      {
        public:
          //! Get the support width of this scale filter.
          /** \return the maximal distance between a source and a destination texel, that is used to determine
          * the source texels that contribute to the destination texel. */
          DP_SG_CORE_API virtual float getSupportWidth( void ) const = 0;

          //! Weight calculation operator.
          /** \return the weight for a source texel that is \a t units from the destination texel to compute. */
          DP_SG_CORE_API virtual float operator()( float t         //!< distance between source and destination texel
                                           ) const = 0;
      };

      //! ScaleFilter functor providing box filter functionality
      /** A BoxFilter has a support width of 0.5. For distances between source and destination texel
        * of up to 0.5, the weight is 1.0, otherwise 0.0. */
      class BoxFilter : public ScaleFilter
      {
        public:
          //! Get the support width of the BoxFilter.
          /** \return the support width of the BoxFilter is 0.5. */
          DP_SG_CORE_API float getSupportWidth( void ) const;

          //! Weight calculation operator.
          /** \return 1.0, if \a t is less than or equal to 0.5, otherwise 0.0. */
          DP_SG_CORE_API float operator()( float t ) const;
      };

      inline float BoxFilter::getSupportWidth( void ) const
      {
        return( 0.5f );
      }

      inline float BoxFilter::operator()( float t ) const
      {
        DP_ASSERT( 0.0f <= t );
        return( ( t <= 0.5f ) ? 1.0f : 0.0f );
      }

      //! ScaleFilter functor providing triangle filter functionality
      /** A TriangleFilter has a support width of 1.0. For distances \a t between source and destination
        * texel of up to 1.0, the weight is <tt>( 1.0 - t )</tt>, otherwise 0.0. */
      class TriangleFilter : public ScaleFilter
      {
        public:
          //! Get the support width of the TriangleFilter.
          /** \return the support width of the TriangleFilter is 1.0. */
          DP_SG_CORE_API float getSupportWidth( void ) const;

          //! Weight calculation operator.
          /** \return <tt>( 1.0 - t )</tt>, if \a t is less than or equal to 1.0, otherwise 0.0. */
          DP_SG_CORE_API float operator()( float t ) const;
      };

      inline float TriangleFilter::getSupportWidth( void ) const
      {
        return( 1.0f );
      }

      inline float TriangleFilter::operator()( float t ) const
      {
        DP_ASSERT( 0.0f <= t );
        return( ( t <= 1.0f ) ? 1.0f - t : 0.0f );
      }

      //! ScaleFilter functor providing bell filter functionality
      /** A BellFilter has a support width of 1.5. For distances \a t between source and destination
        * texel in the half open interval [0.0,0.5), the weight is <tt>( 0.75 - t^2 )</tt>, if \a t
        * is in the half open interval [0.5,1.5), the weight is <tt>( 0.5 * ( 1.5 - t )^2 )</tt>,
        * otherwise 0.0. */
      class BellFilter : public ScaleFilter
      {
        public:
          //! Get the support width of the BellFilter.
          /** \return the support width of the BellFilter is 1.5. */
          DP_SG_CORE_API float getSupportWidth( void ) const;

          //! Weight calculation operator.
          /** \return <tt>( 0.75 - t^2 )</tt>, if \a t is in the interval [0.0,0.5),
            * <tt>( 0.5 * ( 1.5 - \a t )^2 )</tt> if \a t is in the interval [0.5,1.5), otherwise 0.0. */
          DP_SG_CORE_API float operator()( float t ) const;
      };

      inline float BellFilter::getSupportWidth( void ) const
      {
        return( 1.5f );
      }

      //! ScaleFilter functor providing BSpline filter functionality
      /** A BSplineFilter has a support width of 2.0. For distances \a t between source and destination
        * texel in the half open interval [0.0,1.0), the weight is <tt>( 2.0/3.0 - t^2 + 0.5 * t^3 )</tt>,
        * if \a t is in the half open interval [1.0,2.0), the weight is <tt>( 1.0/6.0 * ( 2.0 - t )^3 )</tt>,
        * otherwise 0.0. */
      class BSplineFilter : public ScaleFilter
      {
        public:
          //! Get the support width of the BSplineFilter.
          /** \return the support width of the BSplineFilter is 2.0. */
          DP_SG_CORE_API float getSupportWidth( void ) const;

          //! Weight calculation operator.
          /** \return <tt>( 2.0/3.0 - t^2 + 0.5 * t^3 )</tt>, if \a t is in the interval [0.0,1.0),
            * <tt>( 1.0/6.0 * ( 2.0 - t )^3 )</tt> if \a t is in the interval [1.0,2.0), otherwise 0.0. */
          DP_SG_CORE_API float operator()( float t ) const;
      };

      inline float BSplineFilter::getSupportWidth( void ) const
      {
        return( 2.0f );
      }

      //! ScaleFilter functor providing lanczos3 filter functionality
      /** A Lanczos3Filter has a support width of 3.0. For distances \a t between source and destination
        * texel in the half open interval [0.0,3.0), the weight is <tt>( sinc(x) * sinc(x/3.0) )</tt>,
        * with <tt>x = t * PI</tt>, otherwise 0.0. */
      class Lanczos3Filter : public ScaleFilter
      {
        public:
          //! Get the support width of the Lanczos3Filter.
          /** \return the support width of the Lanczos3Filter is 3.0. */
          DP_SG_CORE_API float getSupportWidth( void ) const;

          //! Weight calculation operator.
          /** \return <tt>( sinc(x) * sinc(x/3.0) )</tt>, with <tt>x=t * PI</tt>, if \a t is in the
            * interval [0.0,3.0), otherwise 0.0. */
          DP_SG_CORE_API float operator()( float t ) const;

        private:
          float sinc( float x ) const;
      };

      inline float Lanczos3Filter::getSupportWidth( void ) const
      {
        return( 3.0f );
      }

      inline float Lanczos3Filter::operator()( float t ) const
      {
        DP_ASSERT( 0.0f <= t );
        return( ( t < 3.0f ) ? sinc(t) * sinc(t/3.0f) : 0.0f );
      }

      inline float Lanczos3Filter::sinc( float x ) const
      {
        x *= dp::math::PI;
        return( ( FLT_EPSILON < x ) ? sin(x) / x : 1.0f );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
