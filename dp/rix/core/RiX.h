// Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/Types.h>
#include <dp/rix/core/RendererConfig.h>
#include <dp/rix/core/HandledObject.h>
#include <cassert>
#include <stddef.h>
#include <stdint.h>

namespace dp
{
  namespace rix
  {
    namespace core
    {

      // renderer creation
      class Renderer;
      typedef Renderer* (*PFNCREATERENDERER)( const char *options );

    #define DEFINE_RIX_HANDLE( name ) \
      class name : public dp::rix::core::HandledObject \
      { \
      protected: \
        name() {} \
      }; \
      typedef dp::rix::core::HandleTrait<name>::Type name##Handle; \
      typedef dp::rix::core::HandleTrait<name>::Type name##WeakHandle; \
      typedef dp::rix::core::SmartHandle<name> name##SharedHandle;

      // keep () tight for visual assist
      DEFINE_RIX_HANDLE(Buffer);
      DEFINE_RIX_HANDLE(Container);
      DEFINE_RIX_HANDLE(ContainerDescriptor);
      DEFINE_RIX_HANDLE(Drawable);
      DEFINE_RIX_HANDLE(Geometry);
      DEFINE_RIX_HANDLE(GeometryDescription);
      DEFINE_RIX_HANDLE(GeometryEffect);
      DEFINE_RIX_HANDLE(GeometryInstance);
      DEFINE_RIX_HANDLE(Indices);
      DEFINE_RIX_HANDLE(Program);
      DEFINE_RIX_HANDLE(ProgramPipeline);
      DEFINE_RIX_HANDLE(RenderGroup);
      DEFINE_RIX_HANDLE(Sampler);
      DEFINE_RIX_HANDLE(SamplerState);
      DEFINE_RIX_HANDLE(Texture);
      DEFINE_RIX_HANDLE(VertexAttributes);
      DEFINE_RIX_HANDLE(VertexData);
      DEFINE_RIX_HANDLE(VertexFormat);

      enum class BufferFormat
      {
          UNKNOWN
        , FLOAT // DAR FIXME Change these to BF_FLOAT_32
        , FLOAT2
        , FLOAT3
        , FLOAT4
        , INT_8
        , INT2_8
        , INT3_8
        , INT4_8
        , INT_16
        , INT2_16
        , INT3_16
        , INT4_16
        , INT_32
        , INT2_32
        , INT3_32
        , INT4_32
        , UINT_8
        , UINT2_8
        , UINT3_8
        , UINT4_8
        , UINT_16
        , UINT2_16
        , UINT3_16
        , UINT4_16
        , UINT_32
        , UINT2_32
        , UINT3_32
        , UINT4_32
        , NATIVE
        , NUM_BUFFERFORMATS
      };

      enum class InternalTextureFormat
      {
          R8
        , R16
        , RG8
        , RG16
        , RGB8
        , RGB16
        , RGBA8
        , RGBA16
        , R16F
        , RG16F
        , RGB16F
        , RGBA16F
        , R32F
        , RG32F
        , RGB32F
        , RGBA32F
        , R8I
        , R8UI
        , R16I
        , R16UI
        , R32I
        , R32UI
        , RG8I
        , RG8UI
        , RG16I
        , RG16UI
        , RG32I
        , RG32UI
        , RGB8I
        , RGB8UI
        , RGB16I
        , RGB16UI
        , RGB32I
        , RGB32UI
        , RGBA8I
        , RGBA8UI
        , RGBA16I
        , RGBA16UI
        , RGBA32I
        , RGBA32UI
        , COMPRESSED_R
        , COMPRESSED_RG
        , COMPRESSED_RGB
        , COMPRESSED_RGBA
        , COMPRESSED_SRGB
        , COMPRESSED_SRGB_ALPHA
        , ALPHA
        , LUMINANCE
        , LUMINANCE_ALPHA
        , RGB
        , RGBA
        , NATIVE
        , NUM_INTERNALTEXTUREFORMATS
      };

      // for cube maps, either use a native data type
      // or pass in the faces via TextureDataPtr
      // with 6 layers: +x,-x,+y,-y,+z,-z
      enum class TextureType
      {
          _1D
        , _1D_ARRAY
        , _2D
        , _2D_RECTANGLE
        , _2D_ARRAY
        , _3D
        , BUFFER
        , CUBEMAP
        , CUBEMAP_ARRAY
        , NATIVE
        , NUM_TEXTURETYPES
      };

      enum class TextureDataType
      {
          POINTER
        , BUFFER
        , NATIVE
        , NUM_TEXTUREDATATYPES
      };

      enum class GeometryInstanceDescriptionType
      {
          COMMON
        , NATIVE
        , NUM_GEOMETRYINSTANCEDESCRIPTIONTYPES
      };

      enum class ContainerParameterType
      {
          FLOAT
        , FLOAT2
        , FLOAT3
        , FLOAT4
        , INT_8
        , INT2_8
        , INT3_8
        , INT4_8
        , INT_16
        , INT2_16
        , INT3_16
        , INT4_16
        , INT_32
        , INT2_32
        , INT3_32
        , INT4_32
        , INT_64
        , INT2_64
        , INT3_64
        , INT4_64
        , UINT_8
        , UINT2_8
        , UINT3_8
        , UINT4_8
        , UINT_16
        , UINT2_16
        , UINT3_16
        , UINT4_16
        , UINT_32
        , UINT2_32
        , UINT3_32
        , UINT4_32
        , UINT_64
        , UINT2_64
        , UINT3_64
        , UINT4_64
        , BOOL
        , BOOL2
        , BOOL3
        , BOOL4
        , MAT2X2
        , MAT2X3
        , MAT2X4
        , MAT3X2
        , MAT3X3
        , MAT3X4
        , MAT4X2
        , MAT4X3
        , MAT4X4
        , SAMPLER
        , IMAGE
        , BUFFER_ADDRESS
        , BUFFER
        , CALLBACK_
        , NATIVE
        , NUM_PARAMETERTYPES
      };

      enum class ContainerDataType
      {
          RAW
        , BUFFER
        , SAMPLER
        , IMAGE
        , NATIVE
        , NUM_CONTAINERDATATYPES
      };

      enum class BufferReferenceType
      {
          BUFFER
        , SAMPLER
        , NATIVE
        , NUM_BUFFERREFERENCETYPES
      };

      enum class ProgramParameterDescriptorType
      {
          COMMON
        , NATIVE
        , NUM_CONTAINERDATATYPES
      };

      enum class ProgramShaderType
      {
          CODE
        , NATIVE
        , NUM_PROGRAMSHADERTYPES
      };

      enum class ShaderType
      {
        VERTEX_SHADER,
        TESS_CONTROL_SHADER,
        TESS_EVALUATION_SHADER,
        GEOMETRY_SHADER,
        FRAGMENT_SHADER,
        NUM_SHADERTYPES
      };

      enum class SamplerStateCompareMode
      {
          NONE
        , R_TO_TEXTURE
      };

      enum class SamplerStateDataType
      {
          COMMON
        , NATIVE
        , NUM_SAMPLERSTATEDATATYPES
      };

      enum class SamplerStateFilterMode
      {
          NEAREST
        , LINEAR
        , NEAREST_MIPMAP_NEAREST
        , LINEAR_MIPMAP_NEAREST
        , NEAREST_MIPMAP_LINEAR
        , LINEAR_MIPMAP_LINEAR
        , NUM_SAMPLERSTATEFILTERMODES
      };

      enum class SamplerStateWrapMode
      {
          CLAMP
        , CLAMP_TO_BORDER
        , CLAMP_TO_EDGE
        , MIRRORED_REPEAT
        , REPEAT
      };

      enum class AccessType
      {
          NONE
        , READ_ONLY
        , WRITE_ONLY
        , READ_WRITE
      };

      enum class BufferDescriptionType
      {
          COMMON
        , NATIVE
      };

      struct SamplerStateData
      {
        SamplerStateDataType getSamplerStateDataType() const { return m_type; }
        virtual ~SamplerStateData() {}

      protected:
        SamplerStateData( SamplerStateDataType type )
          : m_type( type )
        {}

      private:
        SamplerStateDataType m_type;
      };

      struct SamplerStateDataCommon : public SamplerStateData
      {
        SamplerStateDataCommon( SamplerStateFilterMode minFilterMode = SamplerStateFilterMode::NEAREST
                              , SamplerStateFilterMode magFilterMode = SamplerStateFilterMode::NEAREST
                              , SamplerStateWrapMode wrapSMode = SamplerStateWrapMode::CLAMP_TO_EDGE
                              , SamplerStateWrapMode wrapTMode = SamplerStateWrapMode::CLAMP_TO_EDGE
                              , SamplerStateWrapMode wrapRMode = SamplerStateWrapMode::CLAMP_TO_EDGE
                              , SamplerStateCompareMode compareMode = SamplerStateCompareMode::NONE)
          : SamplerStateData( SamplerStateDataType::COMMON )
          , m_minFilterMode( minFilterMode )
          , m_magFilterMode( magFilterMode )
          , m_wrapSMode( wrapSMode )
          , m_wrapTMode( wrapTMode )
          , m_wrapRMode( wrapRMode )
          , m_compareMode( compareMode )
        {}

        SamplerStateFilterMode  m_minFilterMode;
        SamplerStateFilterMode  m_magFilterMode;
        SamplerStateWrapMode    m_wrapSMode;
        SamplerStateWrapMode    m_wrapTMode;
        SamplerStateWrapMode    m_wrapRMode;
        SamplerStateCompareMode m_compareMode;
      };

      struct BufferDescription
      {
        RIX_CORE_API BufferDescription( BufferDescriptionType type = BufferDescriptionType::COMMON );
        RIX_CORE_API virtual ~BufferDescription();

        BufferDescriptionType m_type;
      };

      struct TextureDescription
      {
        RIX_CORE_API TextureDescription( TextureType type, InternalTextureFormat internalFormat,
                                         dp::PixelFormat pixelFormat, dp::DataType dataType,
                                         size_t width = 0, size_t height = 0, size_t depth = 0,
                                         size_t layers = 0, bool mipmaps = false );
        RIX_CORE_API virtual ~TextureDescription();

        TextureType           m_type;
        InternalTextureFormat m_internalFormat;
        dp::PixelFormat       m_pixelFormat;
        dp::DataType          m_dataType;
        size_t                m_width;
        size_t                m_height;
        size_t                m_depth;
        size_t                m_layers;
        bool                  m_mipmaps;
      };

      struct TextureData
      {
        RIX_CORE_API TextureData( TextureDataType type );
        RIX_CORE_API virtual ~TextureData();

        TextureDataType getTextureDataType() const { return m_type; }

      private:
        TextureDataType m_type;
      };

      /** \brief Texture Data struct to pass references to previously constructed Buffers. Sets m_type to TextureDataType::BUFFER.
          \remarks This struct can only be used in conjunction with BUFFER texture types.
       **/
      struct TextureDataBuffer : public TextureData
      {
        /** \brief Provide a buffer as source for a BUFFER texture
            \param buffer The handle of the buffer
         **/
        RIX_CORE_API TextureDataBuffer( BufferSharedHandle const & buffer );

        BufferSharedHandle m_buffer;
      };

      /** \brief Texture Data struct to pass raw data pointers into the Renderer API. Sets m_type to TextureDataType::POINTER.
       **/
      struct TextureDataPtr : public TextureData
      {
        /** \brief Provide a single image without mipmaps
            \param data          Pointer to pixel data
            \param pixelFormat   The format of a pixel in data
            \param pixelDataType The data type of a pixel in data
            \remarks This convenience constructor uses the intermediate pointer m_pData to construct the void** m_data, so
            only a void* has to be passed into the struct
        **/
        RIX_CORE_API TextureDataPtr( void const * data, dp::PixelFormat pixelFormat, dp::DataType pixelDataType );

        /** \brief Provide data with MipMapLevels. Each provider is for one MipMap level.
            \param data Pointer to pixel data
            \param numMipMapLevels Use 0 to automatically generate mipmaps
            \param pixelFormat   The format of a pixel in data
            \param pixelDataType The data type of a pixel in data
        **/
        RIX_CORE_API TextureDataPtr( void const * const *  data, unsigned int numMipMapLevels, dp::PixelFormat pixelFormat, dp::DataType pixelDataType );

        /** \brief Provide data with MipMapLevels and Number of layers. Data is organized as Layer0: MipMap0...n, Layer1: MipMap 0...n, ....
            \param data Pointer to pixel data
            \param numMipMapLevels Use 0 to automatically generate mipmaps
            \param numLayers Number of Layers for an array
            \param pixelFormat   The format of a pixel in data
            \param pixelDataType The data type of a pixel in data
        **/
        RIX_CORE_API TextureDataPtr( void const * const * data, unsigned int numMipMapLevels, unsigned int numLayers, dp::PixelFormat pixelFormat, dp::DataType pixelDataType );

        void const *          m_pData; // intermediate pointer to data for TextureDataPtr( void const * data )
        void const * const *  m_data;
        unsigned int          m_numLayers;
        unsigned int          m_numMipMapLevels;
        dp::PixelFormat       m_pixelFormat;
        dp::DataType          m_pixelDataType;
      };

      struct GeometryInstanceDescription
      {
        RIX_CORE_API GeometryInstanceDescription( GeometryInstanceDescriptionType type = GeometryInstanceDescriptionType::COMMON );
        RIX_CORE_API virtual ~GeometryInstanceDescription();

        GeometryInstanceDescriptionType m_type;
      };

      struct BufferObject
      {
        RIX_CORE_API BufferObject() {};
        RIX_CORE_API virtual ~BufferObject();
      };

      struct CallbackObject
      {
        typedef void (*RENDERCALLBACK)( const void * );
        RENDERCALLBACK m_func;
        void *         m_data;
      };

      typedef unsigned int ContainerEntry;

      struct ProgramParameter
      {
        RIX_CORE_API ProgramParameter( const char* name, ContainerParameterType type, unsigned int arraySize = 0 );
        RIX_CORE_API virtual ~ProgramParameter();

        const char* m_name;
        ContainerParameterType m_type;
        unsigned int m_arraySize;
      };

      struct ProgramParameterDescriptor
      {
        ProgramParameterDescriptor( ProgramParameterDescriptorType type )
          : m_type( type )
        {
        }

        RIX_CORE_API virtual ~ProgramParameterDescriptor();

        ProgramParameterDescriptorType getType() const { return m_type; }
      private:
        ProgramParameterDescriptorType m_type;
      };

      struct ProgramParameterDescriptorCommon : public ProgramParameterDescriptor
      {
        ProgramParameterDescriptorCommon( ProgramParameter* parameters, size_t numParameters, bool multicast = false )
          : ProgramParameterDescriptor( ProgramParameterDescriptorType::COMMON )
          , m_parameters( parameters )
          , m_numParameters( numParameters )
          , m_multicast(multicast)
        {
        }

        ProgramParameter* m_parameters;
        size_t            m_numParameters;
        bool              m_multicast;
      };

      struct ProgramShader
      {
      protected:
        ProgramShader( ProgramShaderType type )
          : m_type( type )
        {
        }

      public:
        virtual ~ProgramShader()
        {
        }

        const ProgramShaderType m_type;
      };

      struct ProgramShaderCode : public ProgramShader
      {
        RIX_CORE_API ProgramShaderCode( const char* code, ShaderType shaderType );
        RIX_CORE_API ProgramShaderCode( size_t numShaders, const char** codes, ShaderType* shaderTypes );
        RIX_CORE_API virtual ~ProgramShaderCode();

        const char*  m_codeData;       // intermediate data
        ShaderType   m_shaderTypeData; // for special case of one shader

        size_t       m_numShaders;
        const char** m_codes;
        ShaderType*  m_shaderTypes;
      };

      struct ProgramDescription
      {
        RIX_CORE_API ProgramDescription( const ProgramShader& programShader, ContainerDescriptorSharedHandle* descriptors, size_t numDescriptors);
        RIX_CORE_API virtual ~ProgramDescription();

        const ProgramShader&             m_shader; // reference to keep polymorphy across ProgramDescription object
        ContainerDescriptorSharedHandle* m_descriptors;
        size_t                           m_numDescriptors;
      };

      struct ContainerData
      {
        virtual ~ContainerData() {}

        ContainerDataType getContainerDataType() const
        {
          return m_containerDataType;
        }

      protected:
        ContainerData( ContainerDataType containerDataType )
          : m_containerDataType( containerDataType )
        {
        }

      private:
        ContainerDataType m_containerDataType;
      };

      struct ContainerDataRaw : public ContainerData
      {
        ContainerDataRaw(size_t offset, const void *data, size_t size, uint32_t gpuId = 0)
          : ContainerData(ContainerDataType::RAW)
          , m_offset(offset)
          , m_data(data)
          , m_size(size)
          , m_gpuId(gpuId)
        {
        }

        size_t      m_offset;
        const void* m_data;
        size_t      m_size;
        uint32_t    m_gpuId;
      };

      struct ContainerDataBuffer : public ContainerData
      {
        ContainerDataBuffer( BufferSharedHandle const & bufferHandle, size_t offset = 0, size_t length = ~0)
          : ContainerData( ContainerDataType::BUFFER )
          , m_bufferHandle( bufferHandle )
          , m_offset( offset )
          , m_length( length )
        {
        }

        BufferSharedHandle m_bufferHandle;
        size_t             m_offset;
        size_t             m_length;
      };

      struct ContainerDataSampler : public ContainerData
      {
        ContainerDataSampler( SamplerSharedHandle const & sampler )
          : ContainerData( ContainerDataType::SAMPLER )
          , m_samplerHandle( sampler )
        {}

        SamplerSharedHandle m_samplerHandle;
      };

      struct ContainerDataImage : public ContainerData
      {
        ContainerDataImage( TextureSharedHandle const & textureHandle, int level, bool layered, int layer, AccessType access )
          : ContainerData( ContainerDataType::IMAGE )
          , m_textureHandle( textureHandle )
          , m_level( level )
          , m_layered( layered )
          , m_layer( layer )
          , m_access( access )
        {}

        TextureSharedHandle m_textureHandle;
        int                 m_level;
        bool                m_layered;
        int                 m_layer;
        AccessType          m_access;
      };

      struct BufferReferences
      {
        virtual ~BufferReferences() {}

        BufferReferenceType getBufferReferenceType() const
        {
          return m_bufferReferenceType;
        }

      protected:
        BufferReferences( BufferReferenceType bufferReferenceType )
          : m_bufferReferenceType( bufferReferenceType )
        {
        }

      private:
        BufferReferenceType m_bufferReferenceType;
      };

      struct BufferReferencesSampler : public BufferReferences
      {
        BufferReferencesSampler( size_t slots)
          : BufferReferences( BufferReferenceType::SAMPLER )
          , m_numSlots( slots )
        {}

        size_t m_numSlots;
      };

      struct BufferReferencesBuffer : public BufferReferences
      {
        BufferReferencesBuffer( size_t slots)
          : BufferReferences( BufferReferenceType::BUFFER )
          , m_numSlots( slots )
        {
        }

        size_t m_numSlots;
      };

      struct BufferStoredReference {
        virtual ~BufferStoredReference() {}
      };

      struct VertexFormatInfo
      {
        VertexFormatInfo()
          : m_attributeIndex(~0)
        {
        }

        /* \brief attributeIndex attribute location in shader
        */
        VertexFormatInfo( uint8_t attributeIndex, dp::DataType dataType, uint8_t numberOfComponents, bool normalized, uint8_t streamId, size_t offset, size_t stride )
          : m_attributeIndex( attributeIndex )
          , m_numComponents( numberOfComponents )
          , m_streamId( streamId )
          , m_normalized( normalized )
          , m_dataType( dataType )
          , m_offset( offset )
          , m_stride( stride )
        {
        }

        uint8_t       m_attributeIndex;
        uint8_t       m_numComponents;
        uint8_t       m_streamId;
        bool          m_normalized;
        dp::DataType  m_dataType;
        size_t        m_offset;
        size_t        m_stride;
      };

      struct VertexFormatDescription
      {
        VertexFormatDescription()
          : m_vertexFormatInfos( nullptr )
          , m_numVertexFormatInfos( 0 )
        {
        }

        VertexFormatDescription(VertexFormatInfo* infos, size_t numInfos)
          : m_vertexFormatInfos( infos )
          , m_numVertexFormatInfos( numInfos )
        {
        }

        VertexFormatInfo* m_vertexFormatInfos; // DAR FIXME Danger! Local pointer would persist when assigning a VertexFormatDescription to another!
        size_t            m_numVertexFormatInfos;
      };

      class RenderOptions // Per render call data.
      {
      public:
        RenderOptions()
          : m_numInstances( 1 )
        {
        }

        ~RenderOptions()
        {
        }

        void setNumberOfInstances(size_t numInstances = 1)
        {
          m_numInstances = numInstances;
        }

      public:
        size_t       m_numInstances;      // OpenGL instanced rendering.
      };


      // Helper Functions
      RIX_CORE_API size_t getSizeOf( dp::DataType dataType );

      class Renderer
      {
      public:
        RIX_CORE_API virtual ~Renderer();
        RIX_CORE_API virtual void deleteThis() = 0;

        RIX_CORE_API virtual void update() = 0;
        RIX_CORE_API virtual void beginRender() = 0;
        RIX_CORE_API virtual void render( dp::rix::core::RenderGroupSharedHandle const & group, dp::rix::core::RenderOptions const & renderOptions = dp::rix::core::RenderOptions() ) = 0;
        RIX_CORE_API virtual void render( dp::rix::core::RenderGroupSharedHandle const & group, dp::rix::core::GeometryInstanceSharedHandle const * gis, size_t numGIs, dp::rix::core::RenderOptions const & renderOptions = dp::rix::core::RenderOptions() ) = 0;
        RIX_CORE_API virtual void endRender() = 0;
        /** Offset is in bytes always **/

        /** VertexFormat **/
        RIX_CORE_API virtual dp::rix::core::VertexFormatSharedHandle vertexFormatCreate( dp::rix::core::VertexFormatDescription const & vertexFormatDescription ) = 0;

        /** VertexData **/
        RIX_CORE_API virtual dp::rix::core::VertexDataSharedHandle vertexDataCreate() = 0;
        RIX_CORE_API virtual void vertexDataSet( dp::rix::core::VertexDataSharedHandle const & handle, unsigned int index, dp::rix::core::BufferSharedHandle const & bufferHandle, size_t offset, size_t numberOfVertices ) = 0;

        /** VertexAttributes **/
        RIX_CORE_API virtual dp::rix::core::VertexAttributesSharedHandle vertexAttributesCreate() = 0;
        RIX_CORE_API virtual void vertexAttributesSet( dp::rix::core::VertexAttributesSharedHandle const & handle, dp::rix::core::VertexDataSharedHandle const & vertexData, dp::rix::core::VertexFormatSharedHandle const & vertexFormat ) = 0;

        /** Indices **/
        RIX_CORE_API virtual dp::rix::core::IndicesSharedHandle indicesCreate() = 0;
        RIX_CORE_API virtual void indicesSetData( dp::rix::core::IndicesSharedHandle const & handle, dp::DataType dataType, dp::rix::core::BufferSharedHandle const & bufferHandle, size_t offset, size_t count ) = 0;

        /** Buffer **/
        RIX_CORE_API virtual dp::rix::core::BufferSharedHandle bufferCreate( dp::rix::core::BufferDescription const & bufferDescription = dp::rix::core::BufferDescription() ) = 0;
        RIX_CORE_API virtual void bufferSetSize( dp::rix::core::BufferSharedHandle const & handle, size_t width, size_t height = 0, size_t depth = 0 ) = 0;
        RIX_CORE_API virtual void bufferSetElementSize( dp::rix::core::BufferSharedHandle const & handle, size_t elementSize ) = 0;
        RIX_CORE_API virtual void bufferSetFormat( dp::rix::core::BufferSharedHandle const & handle, dp::rix::core::BufferFormat bufferFormat ) = 0;
        RIX_CORE_API virtual void bufferUpdateData( dp::rix::core::BufferSharedHandle const & handle, size_t offset, void const * data, size_t size ) = 0;
        RIX_CORE_API virtual void bufferInitReferences( dp::rix::core::BufferSharedHandle const & handle, dp::rix::core::BufferReferences const & refinfo ) = 0;
        RIX_CORE_API virtual void bufferSetReference( dp::rix::core::BufferSharedHandle const & handle, size_t slot, dp::rix::core::ContainerData const & data, dp::rix::core::BufferStoredReference & stored ) = 0;
        RIX_CORE_API virtual void* bufferMap( dp::rix::core::BufferSharedHandle const & handle, dp::rix::core::AccessType accessType ) = 0;
        RIX_CORE_API virtual bool  bufferUnmap( dp::rix::core::BufferSharedHandle const & handle ) = 0;

        /** Texture **/
        RIX_CORE_API virtual dp::rix::core::TextureSharedHandle textureCreate( dp::rix::core::TextureDescription const & description ) = 0;
        RIX_CORE_API virtual void textureSetData( dp::rix::core::TextureSharedHandle const & texture, dp::rix::core::TextureData const & data ) = 0;
        RIX_CORE_API virtual void textureSetDefaultSamplerState( dp::rix::core::TextureSharedHandle const & texture, dp::rix::core::SamplerStateSharedHandle const & samplerState ) = 0;

        /** Sampler **/
        RIX_CORE_API virtual dp::rix::core::SamplerSharedHandle samplerCreate( ) = 0;
        RIX_CORE_API virtual void samplerSetSamplerState( dp::rix::core::SamplerSharedHandle const & sampler, dp::rix::core::SamplerStateSharedHandle const & samplerState ) = 0;
        RIX_CORE_API virtual void samplerSetTexture( dp::rix::core::SamplerSharedHandle const & sampler, dp::rix::core::TextureSharedHandle const & texture) = 0;

        /** SamplerState **/
        // sampler state is an immutable object, pass its data into the create function
        RIX_CORE_API virtual dp::rix::core::SamplerStateSharedHandle samplerStateCreate( dp::rix::core::SamplerStateData const & data ) = 0;

        /** GeometryDescription **/
        RIX_CORE_API virtual dp::rix::core::GeometryDescriptionSharedHandle geometryDescriptionCreate() = 0;
        RIX_CORE_API virtual void geometryDescriptionSet( dp::rix::core::GeometryDescriptionSharedHandle const & geometryDescription, GeometryPrimitiveType type, unsigned int primitiveRestartIndex = ~0 ) = 0;
        RIX_CORE_API virtual void geometryDescriptionSetBaseVertex( dp::rix::core::GeometryDescriptionSharedHandle const & geometryDescription, unsigned int baseVertex ) = 0;
        RIX_CORE_API virtual void geometryDescriptionSetIndexRange( dp::rix::core::GeometryDescriptionSharedHandle const & geometryDescription, unsigned int first, unsigned int count ) = 0;

        /** Geometry **/
        RIX_CORE_API virtual dp::rix::core::GeometrySharedHandle geometryCreate() = 0;
        RIX_CORE_API virtual void geometrySetData( dp::rix::core::GeometrySharedHandle const & geometry, dp::rix::core::GeometryDescriptionSharedHandle const & geometryDescription
          , dp::rix::core::VertexAttributesSharedHandle const & vertexAttributes, dp::rix::core::IndicesSharedHandle const & indices = 0 ) = 0;

        /** GeometryInstance **/
        RIX_CORE_API virtual dp::rix::core::GeometryInstanceSharedHandle geometryInstanceCreate( dp::rix::core::GeometryInstanceDescription const & geometryInstanceDescription = dp::rix::core::GeometryInstanceDescription() ) = 0;
        RIX_CORE_API virtual bool geometryInstanceUseContainer( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::ContainerSharedHandle const & containerHandle ) = 0;
        RIX_CORE_API virtual void geometryInstanceSetGeometry( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::GeometrySharedHandle const & geometry ) = 0;
        RIX_CORE_API virtual void geometryInstanceSetProgramPipeline( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::ProgramPipelineSharedHandle const & programPipelineHandle ) = 0;
        RIX_CORE_API virtual void geometryInstanceSetVisible( dp::rix::core::GeometryInstanceSharedHandle const & handle, bool visible ) = 0;

        /** Program **/
        RIX_CORE_API virtual dp::rix::core::ProgramSharedHandle programCreate( dp::rix::core::ProgramDescription const & description ) = 0;

        /** ProgramPipeline **/
        RIX_CORE_API virtual dp::rix::core::ProgramPipelineSharedHandle programPipelineCreate( dp::rix::core::ProgramSharedHandle const * programs, unsigned int numPrograms ) = 0;

        /** Container **/
        RIX_CORE_API virtual dp::rix::core::ContainerSharedHandle containerCreate( dp::rix::core::ContainerDescriptorSharedHandle const & desc ) = 0;
        RIX_CORE_API virtual void containerSetData(dp::rix::core::ContainerSharedHandle const & containerHandle, dp::rix::core::ContainerEntry entry, dp::rix::core::ContainerData const & containerData ) = 0;

        /** ContainerDescriptor **/
        RIX_CORE_API virtual dp::rix::core::ContainerDescriptorSharedHandle containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptor const & programParameterDescriptor ) = 0;
        RIX_CORE_API virtual unsigned int containerDescriptorGetNumberOfEntries( dp::rix::core::ContainerDescriptorSharedHandle const & desc ) = 0;
        RIX_CORE_API virtual dp::rix::core::ContainerEntry containerDescriptorGetEntry( dp::rix::core::ContainerDescriptorSharedHandle const & desc, unsigned int index ) = 0;
        RIX_CORE_API virtual dp::rix::core::ContainerEntry containerDescriptorGetEntry( dp::rix::core::ContainerDescriptorSharedHandle const & desc, char const * name ) = 0;

        /** RenderGroup **/
        RIX_CORE_API virtual dp::rix::core::RenderGroupSharedHandle renderGroupCreate() = 0;
        RIX_CORE_API virtual void renderGroupAddGeometryInstance( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const & geometryHandle ) = 0;
        RIX_CORE_API virtual void renderGroupRemoveGeometryInstance( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const & geometryHandle ) = 0;
        RIX_CORE_API virtual void renderGroupSetProgramPipeline( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::ProgramPipelineSharedHandle const & programPipelineHandle ) = 0;
        RIX_CORE_API virtual void renderGroupUseContainer( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::ContainerSharedHandle const & containerHandle ) = 0;
      };

    } // core
  } // rix
} // dp
