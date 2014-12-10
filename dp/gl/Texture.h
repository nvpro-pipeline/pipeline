// Copyright NVIDIA Corporation 2010
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

#include <dp/gl/Config.h>
#include <dp/gl/Buffer.h>
#include <dp/gl/Object.h>
#include <dp/math/math.h>

// TODO Image capturing is currently not implemented. The code sequences using 
// TextureHost have been disabled with #if 0.

namespace dp
{
  namespace gl
  {
    DP_GL_API size_t getCompressedSize( GLenum format, GLsizei w, GLsizei h, GLsizei d, GLsizei layers = 0 );
    DP_GL_API size_t getImageDataSize( GLenum format, GLenum type, GLsizei w, GLsizei h, GLsizei d, GLsizei layers = 0 );
    DP_GL_API GLenum getTargetForSamplerType( GLenum samplerType );
    DP_GL_API bool isCompressedFormat( GLenum format );
    DP_GL_API bool isIntegerInternalFormat( GLenum format );
    DP_GL_API bool isValidInternalFormat( GLenum format );
    DP_GL_API bool isLayeredTarget( GLenum target );
    DP_GL_API bool isSamplerType( GLenum type );
    DP_GL_API bool isImageType( GLenum type );

    /*! \brief Base class to represent an OpenGL texture.
     *  \remarks Various sub classes exist that represent specific OpenGL textures.
     */
    class Texture : public Object
    {
    public:
      /*! \brief Binds texture and stores previous binding.
      **/
      DP_GL_API void bind() const;

      /*! \brief Generates all mipmaps based on level of detail 0 using hardware functionality and sets all mipmap levels as defined.
       *  \note Only supported when EXT/ARB_framebuffer_object exist and if the texture has m_maxLevel > 0.
       *  The function also sets the texture base level to 0.
      **/
      DP_GL_API void generateMipMap();

     /*! \brief Returns the OpenGL format of the pixel data (e.g. GL_RGBA).
         \note The value is initially set in the constructor and is overwritten in setData functions.
         Use this function if you want to retrieve data in the format, which client data was
         last specified.
      **/
      DP_GL_API GLenum getFormat() const;

      /*! \brief Returns the OpenGL internal format of the texture (e.g. GL_RGBA8).
      **/
      DP_GL_API GLenum getInternalFormat() const;

      /*! \brief Get the highest level the texture can have, based on its current size.
      **/
      DP_GL_API GLuint getMaxLevel() const;

      /*! \brief Returns the OpenGL target of the texture (e.g. GL_TEXTURE_2D).
      **/
      DP_GL_API GLenum getTarget() const;

     /*! \brief Returns the OpenGL data type of the pixel data (e.g. GL_UNSGINED_BYTE).
         \note The value is initially set in the constructor and is overwritten in setData functions.
         Use this function if you want to retrieve data in the type, which client data was
         last specified.
      **/
      DP_GL_API GLenum getType() const;

      /*! \brief Returns whether all necessary mipmap levels have been supplied. Once complete, mipmapped texture filtering can be used.
      **/
      DP_GL_API bool isMipMapComplete() const;

      /*! \brief Returns whether the texture level has been defined.
       *  \note The levels can be defined through setData, create from TextureHost or generateMipMap.<br> Level 0 is also defined through resize operations.
      **/
      DP_GL_API bool isMipMapLevelDefined( GLuint level ) const;

      /*! \brief Returns whether the texture level can be used at all.
          \note This operation is the same as level <= getMaxLevel
      **/
      DP_GL_API bool isMipMapLevelValid( GLuint level ) const;

      DP_GL_API void setBorderColor( float color[4] );
      DP_GL_API void setBorderColor( unsigned int color[4] );
      DP_GL_API void setBorderColor( int color[4] );

      DP_GL_API void setCompareParameters( GLenum mode, GLenum func );

      DP_GL_API void setFormat( GLenum format );

      /*! \brief Set filtering modes for this texture
       *  \param minFilter The filter to use for minification
       *  \param magFilter The filter to use for magnification
      **/
      DP_GL_API void setFilterParameters( GLenum minFilter, GLenum magFilter );

      DP_GL_API void setLODParameters( float minLOD, float maxLOD, float LODBias );

      DP_GL_API void setMaxAnisotropy( float anisotropy );

      DP_GL_API void setType( GLenum type );

      DP_GL_API void setWrapParameters( GLenum wrapS, GLenum wrapT, GLenum wrapR );

      /*! \brief Restores previous binding stored at bind.
      **/
      DP_GL_API void unbind() const;

    protected:
      /*! \brief Constructor for base class.
       *  \param target The OpenGL target of the texture (e.g. GL_TEXTURE_2D).
       *  \param internalFormat The GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param format The OpenGL texture client format used in resize operations (e.g. GL_RGBA).
       *  \param type The OpenGL texture client type used in resize operations (e.g. GL_UNSIGNED_BYTE).
       *  \param border The texture border size.
       *  \note By default all textures are set to nearest filtering without mipmaps to be renderable
       *  once first texture level is specified independent of all internal formats. Any other texture
       *  states, such as wrap modes, are in their original default state defined by OpenGL.
       */
      DP_GL_API Texture( GLenum target, GLenum internalFormat, GLenum format, GLenum type, GLsizei border = 0 );

      DP_GL_API virtual ~Texture();

      DP_GL_API void addDefinedLevel( GLuint level );
      DP_GL_API void resetDefinedLevels();
      DP_GL_API void setMaxLevel( GLuint level );

    private:
      GLenum      m_target;           //!< The OpenGL target of the texture (e.g. GL_TEXTURE_2D)
      GLenum      m_internalFormat;   //!< The GL format used to store the data on the OpenGL server (e.g. GL_RGBA8)
      GLenum      m_format;           //!< The OpenGL format of the pixel data (e.g. GL_RGBA)
      GLenum      m_type;             //!< The OpenGL data type of the pixel data (e.g. GL_UNSGINED_BYTE)
      GLbitfield  m_definedLevels;    //!< bitfield encoding which levels have been created already
      GLuint      m_maxLevel;         //!< must be specified when size of texture is changed
    };

    inline GLenum Texture::getTarget() const
    {
      return m_target;
    }

    inline GLenum Texture::getInternalFormat() const
    {
      return m_internalFormat;
    }

    inline GLenum Texture::getFormat() const
    {
      return m_format;
    }

    inline GLenum Texture::getType() const
    {
      return m_type;
    }

    inline GLuint Texture::getMaxLevel() const
    {
      return m_maxLevel;
    }

    inline bool Texture::isMipMapLevelValid( GLuint level ) const
    {
      return level <= m_maxLevel;
    }

    inline bool Texture::isMipMapLevelDefined( GLuint level ) const
    {
      return !!( m_definedLevels & ( 1 << level ));
    }

    inline bool Texture::isMipMapComplete() const
    {
      return (( 1u << (m_maxLevel + 1u)) - 1u) == m_definedLevels;
    }


    /*! \brief Class for 1D OpenGL textures.
     */
    class Texture1D : public Texture
    {
    public:
      /*! \brief Creates a texture based on the given GL parameters.
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param format The OpenGL texture client format used in resize operations (e.g. GL_RGBA).
       *  \param type The OpenGL texture client type used in resize operations (e.g. GL_UNSIGNED_BYTE).
       *  \param width The texture width.
       *  \sa setData, resize */
      DP_GL_API static Texture1DSharedPtr create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width = 0 );

    public:
      /*! \brief Transfers data to the OpenGL texture, keeping current internal format and size.
       *  \param data A pointer to the data to transfer.
       *  \param mipLevel The texture level (0 being base-level) the data is stored to.
       *  \sa getData */
      DP_GL_API void setData( const void *data, GLuint mipLevel = 0 );

      /*! \brief Retrieves the OpenGL texture data.
       *  \param data A pointer to the area to transfer the texture data to.
       *  \param mipLevel Optional texture level (0 being base level) that data is transfered from. Default is 0.
       *  \note It is assumed that there's enough memory allocated at \a data to hold the transfered data.
       *  \sa setData */
      DP_GL_API void getData( void *data, GLuint mipLevel = 0 ) const;

      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize( GLsizei width );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_TEXTURE_SIZE for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

    protected:
      Texture1D( GLenum internalFormat, GLenum format, GLenum type, GLsizei width );

    private:
      GLsizei m_width;
    };

    inline GLsizei Texture1D::getWidth() const
    {
      return m_width;
    }


    /*! \brief Class for 1D array OpenGL textures.
     *  \remarks This texture type requires additional hardware support.
     *  Array textures store multiple 1D textures in stacked layers.
     *  Texture interpolation is done within layers, but not between.
     *  \sa isSupported
     */
    class Texture1DArray : public Texture
    {
    public:
      /*! \brief Creates a texture based on the given GL parameters.
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param format The OpenGL texture client format used in resize operations (e.g. GL_RGBA).
       *  \param type The OpenGL texture client type used in resize operations (e.g. GL_UNSIGNED_BYTE).
       *  \param width The texture width.
       *  \param layers The amount of texture layers.
       *  \sa setData, resize */
      DP_GL_API static Texture1DArraySharedPtr create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width = 0, GLsizei layers = 0 );

    public:
      /*! \brief Transfers data to the OpenGL texture, keeping current internal format and size.
       *  \param data A pointer to the data that is read.
       *  \param layer The layer that is updated.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa getData */
      DP_GL_API void setData( const void *data, GLint layer, GLuint mipLevel = 0);

      /*! \brief Retrieves all layers of the OpenGL texture data.
       *  \param data A pointer for data storage.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa setData */
      DP_GL_API void getData( void *data, GLuint mipLevel = 0 ) const;

      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize( GLsizei width, GLsizei numLayers );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getLayers() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_TEXTURE_SIZE for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

      /*! \brief Returns the maximum texture layers allowed in the current OpenGL context.
      **/
      DP_GL_API static GLsizei getMaximumLayers();

      /*! \brief Returns \c true if this texture class is supported in the current OpenGL context.
      **/
      DP_GL_API static bool isSupported();

    protected:
      Texture1DArray( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei layers );

    private:
      int m_width;
      int m_layers;
    };

    inline GLsizei Texture1DArray::getWidth() const
    {
      return m_width;
    }

    inline GLsizei Texture1DArray::getLayers() const
    {
      return m_layers;
    }


    /*! \brief Class for 2D OpenGL textures.
     */
    class Texture2D : public Texture
    {
    public:
      /*! \brief Creates a texture based on the given GL parameters.
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param format The OpenGL texture client format used in resize operations (e.g. GL_RGBA).
       *  \param type The OpenGL texture client type used in resize operations (e.g. GL_UNSIGNED_BYTE).
       *  \param width The texture width.
       *  \param height The texture height.
       *  \sa setData, resize */
      DP_GL_API static Texture2DSharedPtr create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width = 0, GLsizei height = 0 );

    public:
      /*! \brief Transfers the buffer data to the OpenGL texture, keeping current internal format and size.
       *  \param data A pointer to the data that is read.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa getData */
      DP_GL_API void setData( const void *data, GLuint mipLevel = 0);

      /*! \brief Retrieves the OpenGL texture data.
       *  \param data A pointer for data storage.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa setData */
      DP_GL_API void getData( void *data, GLuint mipLevel = 0 ) const;

      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize( GLsizei width, GLsizei height );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the texture height.
      **/
      DP_GL_API GLsizei getHeight() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_TEXTURE_SIZE for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

    protected:
      Texture2D( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height );

    private:
      int m_width;
      int m_height;
    };

    inline GLsizei Texture2D::getWidth() const
    {
      return m_width;
    }

    inline GLsizei Texture2D::getHeight() const
    {
      return m_height;
    }


    /*! \brief Class for rectangle OpenGL textures.
     *  \remarks Rectangle textures do not support borders or mip map levels and
     *  are sampled using unnormalized texture coordinates.
     */
    class TextureRectangle : public Texture
    {
    public:
      /*! \brief Creates a texture based on the given GL parameters.
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param format The OpenGL texture client format used in resize operations (e.g. GL_RGBA).
       *  \param type The OpenGL texture client type used in resize operations (e.g. GL_UNSIGNED_BYTE).
       *  \param width The texture width.
       *  \param height The texture height.
       *  \sa setData, resize */
      DP_GL_API static TextureRectangleSharedPtr create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width = 0, GLsizei height = 0 );

    public:
      /*! \brief Transfers data to the OpenGL texture, keeping current internal format and size.
       *  \param data A pointer to the data that is read.
       *  \sa getData */
      DP_GL_API void setData( const void *data );

      /*! \brief Retrieves the OpenGL texture data.
       *  \param type Specifies the OpenGL client type of the data (e.g. GL_FLOAT).
       *  \sa setData */
      DP_GL_API void getData( void *data ) const;

      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize( GLsizei width, GLsizei height );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the texture height.
      **/
      DP_GL_API GLsizei getHeight() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_RECTANGLE_TEXTURE_SIZE_ARB for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

    protected:
      TextureRectangle( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height );

    private:
      int m_width;
      int m_height;
    };

    inline GLsizei TextureRectangle::getWidth() const
    {
      return m_width;
    }

    inline GLsizei TextureRectangle::getHeight() const
    {
      return m_height;
    }


    /*! \brief Class for 2D array OpenGL textures.
     *  \remarks This texture type requires additional hardware support.
     *  Array textures store multiple 2D textures in stacked layers.
     *  Texture interpolation is done within layers, but not between.
     *  \sa isSupported
     */
    class Texture2DArray : public Texture
    {
    public:
      /*! \brief Creates a texture based on the given GL parameters.
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param format The OpenGL texture client format used in resize operations (e.g. GL_RGBA).
       *  \param type The OpenGL texture client type used in resize operations (e.g. GL_UNSIGNED_BYTE).
       *  \param width The texture width.
       *  \param height The texture height.
       *  \param layers The amount of texture layers.
       *  \sa setData, resize */
      DP_GL_API static Texture2DArraySharedPtr create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width = 0, GLsizei height = 0, GLsizei layers = 0 );

    public:
      /*! \brief Transfers data to the OpenGL texture, keeping current internal format and size.
       *  \param data A pointer to the data that is read.
       *  \param layer The layer that is updated.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa getData */
      DP_GL_API void setData( const void *data, GLint layer, GLuint mipLevel = 0 );

      /*! \brief Retrieves all layers of the OpenGL texture data.
       *  \param data A pointer for data storage.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa setData */
      DP_GL_API void getData( void *data, GLuint mipLevel = 0 ) const;

      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize( GLsizei width, GLsizei height, GLsizei layers );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the texture height.
      **/
      DP_GL_API GLsizei getHeight() const;

      /*! \brief Returns the texture layers.
      **/
      DP_GL_API GLsizei getLayers() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_TEXTURE_SIZE for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

      /*! \brief Returns the maximum texture layers allowed in the current OpenGL context.
      **/
      DP_GL_API static GLsizei getMaximumLayers();

      /*! \brief Returns \c true if this texture class is supported in the current OpenGL context.
      **/
      DP_GL_API static bool isSupported();

    protected:
      Texture2DArray( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height, GLsizei layers );

    private:
      int m_width;
      int m_height;
      int m_layers;
    };

    inline GLsizei Texture2DArray::getWidth() const
    {
      return m_width;
    }

    inline GLsizei Texture2DArray::getHeight() const
    {
      return m_height;
    }

    inline GLsizei Texture2DArray::getLayers() const
    {
      return m_layers;
    }


    /*! \brief Class for 3D OpenGL textures.
     *  \remarks This type is mostly used for volume rendering.
     */
    class Texture3D : public Texture
    {
    public:
      /*! \brief Creates a texture based on the given GL parameters.
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param format The OpenGL texture client format used in resize operations (e.g. GL_RGBA).
       *  \param type The OpenGL texture client type used in resize operations (e.g. GL_UNSIGNED_BYTE).
       *  \param width The texture width.
       *  \param height The texture height.
       *  \param depth The texture depth.
       *  \sa setData, resize */
      DP_GL_API static Texture3DSharedPtr create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width = 0, GLsizei height = 0, GLsizei depth = 0 );

    public:
      /*! \brief Transfers data to the OpenGL texture, keeping current internal format and size.
       *  \param data A pointer to the data that is read.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa getData */
      DP_GL_API void setData( const void *data, GLuint mipLevel = 0);

      /*! \brief Retrieves the OpenGL texture data.
       *  \param data A pointer for data storage.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa setData */
      DP_GL_API void getData( void *data, GLuint mipLevel = 0 ) const;

      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize( GLsizei width, GLsizei height, GLsizei depth );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the texture height.
      **/
      DP_GL_API GLsizei getHeight() const;

      /*! \brief Returns the texture depth.
      **/
      DP_GL_API GLsizei getDepth() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_3D_TEXTURE_SIZE for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

    protected:
      Texture3D( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height, GLsizei depth );

    private:
      int m_width;
      int m_height;
      int m_depth;
    };

    inline GLsizei Texture3D::getWidth() const
    {
      return m_width;
    }

    inline GLsizei Texture3D::getHeight() const
    {
      return m_height;
    }

    inline GLsizei Texture3D::getDepth() const
    {
      return m_depth;
    }


    /*! \brief Class for cubemap OpenGL textures.
     *  \remarks Cubemaps represent the six inner faces of a cube and are mostly used
     *  for environment effects (e.g. reflections).
     *  The cube faces are square and stored in the order +X,-X,+Y,-Y,+Z,-Z.
     */
    class TextureCubemap : public Texture
    {
    public:
      /*! \brief Creates a texture based on the given GL parameters.
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param format The OpenGL texture client format used in resize operations (e.g. GL_RGBA).
       *  \param type The OpenGL texture client type used in resize operations (e.g. GL_UNSIGNED_BYTE).
       *  \sa setData, resize */
      DP_GL_API static TextureCubemapSharedPtr create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width = 0, GLsizei height = 0 );

    public:
      /*! \brief Transfers data to the OpenGL texture, keeping current internal format and size.
       *  \param data A pointer to the data that is read.
       *  \param face Specifies the face of the cube from 0 to 5 (+X to -Z).
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa getData */
      DP_GL_API void setData( const void *data, int face, GLuint mipLevel = 0 );

      /*! \brief Retrieves the OpenGL texture data.
       *  \param data A pointer for data storage.
       *  \param face Specifies the face of the cube from 0 to 5 (+X to -Z).
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa setData */
      DP_GL_API void getData( void *data, int face, GLuint mipLevel = 0 ) const;

      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize( GLsizei width, GLsizei height );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the texture height.
      **/
      DP_GL_API GLsizei getHeight() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_CUBE_MAP_TEXTURE_SIZE for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

    protected:
      TextureCubemap( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height );

    private:
      int m_width;
      int m_height;
    };

    inline GLsizei TextureCubemap::getWidth() const
    {
      return m_width;
    }

    inline GLsizei TextureCubemap::getHeight() const
    {
      return m_height;
    }


    /*! \brief Class for cubemap array OpenGL textures.
     *  \remarks Cubemaps represent the six inner faces of a cube and are mostly used
     *  for environment effects (e.g. reflections).
     *  The cube faces are stored in the order +X,-X,+Y,-Y,+Z,-Z.
     *  Each layer encodes one cube face, so every six layers a new cubemap
     *  starts. This texture class requires additional hardware support.
     * \sa isSupported
     */
    class TextureCubemapArray : public Texture
    {
    public:
      /*! \brief Creates a texture based on the given GL parameters.
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param format The OpenGL texture client format used in resize operations (e.g. GL_RGBA).
       *  \param type The OpenGL texture client type used in resize operations (e.g. GL_UNSIGNED_BYTE).
       *  \param width The texture width. Width and height must match.
       *  \param height The texture height. Width and height must match.
       *  \param layers The amount of texture layers.
       *  \sa setData, resize */
      DP_GL_API static TextureCubemapArraySharedPtr create( GLenum internalFormat, GLenum format, GLenum type, GLsizei width = 0, GLsizei height = 0, GLsizei layers = 0 );

    public:
      /*! \brief Transfers data to the OpenGL texture, keeping current internal format and size.
       *  \param data A pointer to the data that is read.
       *  \param layer The amount of texture layers.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa getData */
      DP_GL_API void setData( const void *data, GLint layer, GLuint mipLevel = 0);

      /*! \brief Retrieves all layers of the OpenGL texture data.
       *  \param data A pointer for data storage.
       *  \param mipLevel Specifies the texture level (0 being base-level) the data is stored to.
       *  \sa setData */
      DP_GL_API void getData( void *data, GLuint mipLevel = 0 ) const;

      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize(GLsizei width, GLsizei height, GLsizei layers );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the texture height.
      **/
      DP_GL_API GLsizei getHeight() const;

      /*! \brief Returns the texture layers.
      **/
      DP_GL_API GLsizei getLayers() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_CUBE_MAP_TEXTURE_SIZE for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

      /*! \brief Returns the maximum texture layers allowed in the current OpenGL context.
      **/
      DP_GL_API static GLsizei getMaximumLayers();

      /*! \brief Returns \c true if this texture class is supported in the current OpenGL context.
      **/
      DP_GL_API static bool isSupported();

    protected:
      TextureCubemapArray( GLenum internalFormat, GLenum format, GLenum type, GLsizei width, GLsizei height, GLsizei layers );

    private:
      int m_width;
      int m_height;
      int m_layers;
    };

    inline GLsizei TextureCubemapArray::getWidth() const
    {
      return m_width;
    }

    inline GLsizei TextureCubemapArray::getHeight() const
    {
      return m_height;
    }

    inline GLsizei TextureCubemapArray::getLayers() const
    {
      return m_layers;
    }


    /*! \brief Class for 2D multisample OpenGL textures.
     *  \remarks This texture type requires additional hardware support.
     *  Multisample textures store multiple samples per texel and
     *  are used as attachments for RenderTargetFBO.
     *  There is no support mipmapping nor texture filtering,
     *  however shaders can directly access samples individually.
     *  \sa isSupported
     */
    class Texture2DMultisample : public Texture
    {
    public:
      /*! \brief Creates a multi-sampled texture based on the given GL parameters 
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param samples The number of samples per pixel
       *  \param width The texture width.
       *  \param height The texture height.
       *  \sa resize, setSamples */
      DP_GL_API static Texture2DMultisampleSharedPtr create( GLenum internalFormat, GLsizei samples = 1, GLsizei width = 0, GLsizei height = 0, bool fixedLocations = true );

    public:
      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize( GLsizei width, GLsizei height );

      /*! \brief Changes the sample count of the texture. All content and mipmap levels are lost if the sample count is different from current state.
      **/
      DP_GL_API void setSamples( GLsizei samples );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the texture height.
      **/
      DP_GL_API GLsizei getHeight() const;

      /*! \brief Returns the number of samples per pixel.
      **/
      DP_GL_API GLsizei getSamples() const;

      /*! \brief Returns whether the texture uses fixed sample locations when being rendered to.
      **/
      DP_GL_API bool getFixedLocations() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_TEXTURE_SIZE for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

      /*! \brief Returns the maximum texture samples allowed in the current OpenGL context.
      **/
      DP_GL_API static GLsizei getMaximumSamples();

      /*! \brief Returns the maximum texture samples allowed in the current OpenGL context, when a integer internal format is used.
      **/
      DP_GL_API static GLsizei getMaximumIntegerSamples();

      /*! \brief Returns the maximum texture samples allowed in the current OpenGL context, when a color-renderable internal format is used.
      **/
      DP_GL_API static GLsizei getMaximumColorSamples();

      /*! \brief Returns the maximum texture samples allowed in the current OpenGL context, when a depth-renderable internal format is used.
      **/
      DP_GL_API static GLsizei getMaximumDepthSamples();

      /*! \brief Returns \c true if this texture class is supported in the current OpenGL context.
      **/
      DP_GL_API static bool isSupported();

    protected:
      Texture2DMultisample( GLenum internalFormat, GLsizei samples, GLsizei width, GLsizei height, bool fixedLocations );

    private:
      int m_width;
      int m_height;
      int m_samples;
      bool m_fixedLocations;
    };

    inline GLsizei Texture2DMultisample::getWidth() const
    {
      return m_width;
    }

    inline GLsizei Texture2DMultisample::getHeight() const
    {
      return m_height;
    }

    inline GLsizei Texture2DMultisample::getSamples() const
    {
      return m_samples;
    }

    inline bool Texture2DMultisample::getFixedLocations() const
    {
      return m_fixedLocations;
    }


    /*! \brief Class for 2D multisample array OpenGL textures.
     *  \remarks This texture type requires additional hardware support.
     *  Multisample textures store multiple samples per texel and
     *  are used as attachments for RenderTargetFBO.
     *  There is no support mipmapping nor texture filtering,
     *  however shaders can directly access samples individually.
     *  Array textures store multiple textures in stacked layers.
     *  \sa isSupported
     */
    class Texture2DMultisampleArray : public Texture
    {
    public:
      /*! \brief Creates a multi-sampled texture based on the given GL parameters 
       *  \param internalFormat Specifies the GL format used to store the data on the OpenGL server (e.g. GL_RGBA8).
       *  \param samples The number of samples per pixel
       *  \param width The texture width.
       *  \param height The texture height.
       *  \param layers The amount of texture layers.
       *  \param fixedLocations When set to true, the location of the samples are the same for all internalFormats and depend only on sample count.
       *  Otherwise they can vary with each internalFormat.
       *  \sa resize, setSamples */
      DP_GL_API static Texture2DMultisampleArraySharedPtr create( GLenum internalFormat, GLsizei samples = 1, GLsizei width = 0, GLsizei height = 0, GLsizei layers = 0, bool fixedLocations = true );

    public:
      /*! \brief Resizes the texture. All content and mipmap levels are lost if the size is different from current state.
      **/
      DP_GL_API void resize( GLsizei width, GLsizei height, GLsizei layers );

      /*! \brief Changes the sample count of the texture. All content and mipmap levels are lost if the sample count is different from current state.
      **/
      DP_GL_API void setSamples( GLsizei samples );

      /*! \brief Returns the texture width.
      **/
      DP_GL_API GLsizei getWidth() const;

      /*! \brief Returns the texture height.
      **/
      DP_GL_API GLsizei getHeight() const;

      /*! \brief Returns the texture layers.
      **/
      DP_GL_API GLsizei getLayers() const;

      /*! \brief Returns the number of samples per pixel.
      **/
      DP_GL_API GLsizei getSamples() const;

      /*! \brief Returns whether the texture uses fixed sample locations when being rendered to.
      **/
      DP_GL_API bool getFixedLocations() const;

      /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
       *  \note Uses GL_MAX_TEXTURE_SIZE for the query.
      **/
      DP_GL_API static GLsizei getMaximumSize();

      /*! \brief Returns the maximum texture layers allowed in the current OpenGL context.
      **/
      DP_GL_API static GLsizei getMaximumLayers();

      /*! \brief Returns the maximum texture samples allowed in the current OpenGL context.
      **/
      DP_GL_API static GLsizei getMaximumSamples();

      /*! \brief Returns the maximum texture samples allowed in the current OpenGL context, when a integer internal format is used.
      **/
      DP_GL_API static GLsizei getMaximumIntegerSamples();

      /*! \brief Returns the maximum texture samples allowed in the current OpenGL context, when a color-renderable internal format is used.
      **/
      DP_GL_API static GLsizei getMaximumColorSamples();

      /*! \brief Returns the maximum texture samples allowed in the current OpenGL context, when a depth-renderable internal format is used.
      **/
      DP_GL_API static GLsizei getMaximumDepthSamples();

      /*! \brief Returns \c true if this texture class is supported in the current OpenGL context.
      **/
      DP_GL_API static bool isSupported();

    protected:
      Texture2DMultisampleArray( GLenum internalFormat, GLsizei samples, GLsizei width, GLsizei height, GLsizei layers, bool fixedLocations );

    private:
      int m_width;
      int m_height;
      int m_samples;
      int m_layers;
      bool m_fixedLocations;
    };

    inline GLsizei Texture2DMultisampleArray::getWidth() const
    {
      return m_width;
    }

    inline GLsizei Texture2DMultisampleArray::getHeight() const
    {
      return m_height;
    }

    inline GLsizei Texture2DMultisampleArray::getLayers() const
    {
      return m_layers;
    }

    inline GLsizei Texture2DMultisampleArray::getSamples() const
    {
      return m_samples;
    }

    inline bool Texture2DMultisampleArray::getFixedLocations() const
    {
      return m_fixedLocations;
    }


    class TextureBuffer : public Texture
    {
      public:
        DP_GL_API static TextureBufferSharedPtr create( GLenum internalFormat, BufferSharedPtr const& buffer );
        DP_GL_API static TextureBufferSharedPtr create( GLenum internalFormat, unsigned int size = 0, GLvoid const* data = nullptr, GLenum usage = GL_DYNAMIC_COPY );
        DP_GL_API ~TextureBuffer();

      public:
        DP_GL_API BufferSharedPtr const& getBuffer() const;
        DP_GL_API void setBuffer( BufferSharedPtr const& buffer );

        /*! \brief Returns the maximum texture size allowed in the current OpenGL context.
         *  \note Uses GL_MAX_TEXTURE_SIZE for the query.
        **/
        DP_GL_API static GLsizei getMaximumSize();

        /*! \brief Returns \c true if this texture class is supported in the current OpenGL context.
        **/
        DP_GL_API static bool isSupported();

      protected:
        DP_GL_API TextureBuffer( GLenum internalFormat, BufferSharedPtr const& buffer );

      private:
        BufferSharedPtr  m_buffer;
    };

  } // namespace gl
} // namespace dp