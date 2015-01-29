// Copyright NVIDIA Corporation 2011-2012
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

#include <dp/rix/core/RendererConfig.h>
#include <dp/rix/core/HandledObject.h>
#include <dp/util/Types.h>
#include <cassert>
#include <stddef.h>

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

      enum BufferFormat
      {
          BF_UNKNOWN
        , BF_FLOAT // DAR FIXME Change these to BF_FLOAT_32
        , BF_FLOAT2
        , BF_FLOAT3
        , BF_FLOAT4
        , BF_INT_8
        , BF_INT2_8
        , BF_INT3_8
        , BF_INT4_8
        , BF_INT_16
        , BF_INT2_16
        , BF_INT3_16
        , BF_INT4_16
        , BF_INT_32
        , BF_INT2_32
        , BF_INT3_32
        , BF_INT4_32
        , BF_UINT_8
        , BF_UINT2_8
        , BF_UINT3_8
        , BF_UINT4_8
        , BF_UINT_16
        , BF_UINT2_16
        , BF_UINT3_16
        , BF_UINT4_16
        , BF_UINT_32
        , BF_UINT2_32
        , BF_UINT3_32
        , BF_UINT4_32
        , BF_NATIVE
        , BF_NUM_BUFFERFORMATS
      };

      enum InternalTextureFormat
      {
          ITF_R8
        , ITF_R16
        , ITF_RG8
        , ITF_RG16
        , ITF_RGB8
        , ITF_RGB16
        , ITF_RGBA8
        , ITF_RGBA16
        , ITF_R16F
        , ITF_RG16F
        , ITF_RGB16F
        , ITF_RGBA16F
        , ITF_R32F
        , ITF_RG32F
        , ITF_RGB32F
        , ITF_RGBA32F
        , ITF_R8I
        , ITF_R8UI
        , ITF_R16I
        , ITF_R16UI
        , ITF_R32I
        , ITF_R32UI
        , ITF_RG8I
        , ITF_RG8UI
        , ITF_RG16I
        , ITF_RG16UI
        , ITF_RG32I
        , ITF_RG32UI
        , ITF_RGB8I
        , ITF_RGB8UI
        , ITF_RGB16I
        , ITF_RGB16UI
        , ITF_RGB32I
        , ITF_RGB32UI
        , ITF_RGBA8I
        , ITF_RGBA8UI
        , ITF_RGBA16I
        , ITF_RGBA16UI
        , ITF_RGBA32I
        , ITF_RGBA32UI
        , ITF_COMPRESSED_R
        , ITF_COMPRESSED_RG
        , ITF_COMPRESSED_RGB
        , ITF_COMPRESSED_RGBA
        , ITF_COMPRESSED_SRGB
        , ITF_COMPRESSED_SRGB_ALPHA
        , ITF_ALPHA
        , ITF_LUMINANCE
        , ITF_LUMINANCE_ALPHA
        , ITF_RGB
        , ITF_RGBA
        , ITF_NATIVE
        , ITF_NUM_INTERNALTEXTUREFORMATS
      };

      // for cube maps, either use a native data type
      // or pass in the faces via TextureDataPtr
      // with 6 layers: +x,-x,+y,-y,+z,-z
      enum TextureType
      {
          TT_1D
        , TT_1D_ARRAY
        , TT_2D
        , TT_2D_RECTANGLE
        , TT_2D_ARRAY
        , TT_3D
        , TT_BUFFER
        , TT_CUBEMAP
        , TT_CUBEMAP_ARRAY
        , TT_NATIVE
        , TT_NUM_TEXTURETYPES
      };

      enum TextureDataType
      {
          TDT_POINTER
        , TDT_BUFFER
        , TDT_NATIVE
        , TDT_NUM_TEXTUREDATATYPES
      };

      enum GeometryInstanceDescriptionType
      {
          GIDT_COMMON
        , GIDT_NATIVE
        , GIDT_NUM_GEOMETRYINSTANCEDESCRIPTIONTYPES
      };

      enum ContainerParameterType
      {
          CPT_FLOAT
        , CPT_FLOAT2
        , CPT_FLOAT3
        , CPT_FLOAT4
        , CPT_INT_8
        , CPT_INT2_8
        , CPT_INT3_8
        , CPT_INT4_8
        , CPT_INT_16
        , CPT_INT2_16
        , CPT_INT3_16
        , CPT_INT4_16
        , CPT_INT_32
        , CPT_INT2_32
        , CPT_INT3_32
        , CPT_INT4_32
        , CPT_INT_64
        , CPT_INT2_64
        , CPT_INT3_64
        , CPT_INT4_64
        , CPT_UINT_8
        , CPT_UINT2_8
        , CPT_UINT3_8
        , CPT_UINT4_8
        , CPT_UINT_16
        , CPT_UINT2_16
        , CPT_UINT3_16
        , CPT_UINT4_16
        , CPT_UINT_32
        , CPT_UINT2_32
        , CPT_UINT3_32
        , CPT_UINT4_32
        , CPT_UINT_64
        , CPT_UINT2_64
        , CPT_UINT3_64
        , CPT_UINT4_64
        , CPT_BOOL
        , CPT_BOOL2
        , CPT_BOOL3
        , CPT_BOOL4
        , CPT_MAT2X2
        , CPT_MAT2X3
        , CPT_MAT2X4
        , CPT_MAT3X2
        , CPT_MAT3X3
        , CPT_MAT3X4
        , CPT_MAT4X2
        , CPT_MAT4X3
        , CPT_MAT4X4
        , CPT_SAMPLER
        , CPT_IMAGE
        , CPT_BUFFER_ADDRESS
        , CPT_BUFFER
        , CPT_CALLBACK
        , CPT_NATIVE
        , CPT_NUM_PARAMETERTYPES
      };

      enum ContainerDataType
      {
          CDT_RAW
        , CDT_BUFFER
        , CDT_SAMPLER
        , CDT_IMAGE
        , CDT_NATIVE
        , CDT_NUM_CONTAINERDATATYPES
      };

      enum BufferReferenceType
      {
          BRT_BUFFER
        , BRT_SAMPLER
        , BRT_NATIVE
        , BRT_NUM_BUFFERREFERENCETYPES
      };

      enum ProgramParameterDescriptorType
      {
          PPDT_COMMON
        , PPDT_NATIVE
        , PPDT_NUM_CONTAINERDATATYPES
      };

      enum ProgramShaderType
      {
          PST_CODE
        , PST_NATIVE
        , PST_NUM_PROGRAMSHADERTYPES
      };

      enum ShaderType
      {
        ST_VERTEX_SHADER,
        ST_TESS_CONTROL_SHADER,
        ST_TESS_EVALUATION_SHADER,
        ST_GEOMETRY_SHADER,
        ST_FRAGMENT_SHADER,
        ST_NUM_SHADERTYPES
      };

      enum SamplerStateCompareMode
      {
          SSCM_NONE
        , SSCM_R_TO_TEXTURE
      };

      enum SamplerStateDataType
      {
          SSDT_COMMON
        , SSDT_NATIVE
        , SSDT_NUM_SAMPLERSTATEDATATYPES
      };

      enum SamplerStateFilterMode
      {
          SSFM_NEAREST
        , SSFM_LINEAR
        , SSFM_NEAREST_MIPMAP_NEAREST
        , SSFM_LINEAR_MIPMAP_NEAREST
        , SSFM_NEAREST_MIPMAP_LINEAR
        , SSFM_LINEAR_MIPMAP_LINEAR
        , SSFM_NUM_SAMPLERSTATEFILTERMODES
      };

      enum SamplerStateWrapMode
      {
          SSWM_CLAMP
        , SSWM_CLAMP_TO_BORDER
        , SSWM_CLAMP_TO_EDGE
        , SSWM_MIRRORED_REPEAT
        , SSWM_REPEAT
      };

      enum AccessType
      {
          AT_NONE
        , AT_READ_ONLY
        , AT_WRITE_ONLY
        , AT_READ_WRITE
      };

      enum BufferDescriptionType
      {
          BDT_COMMON
        , BDT_NATIVE
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
        SamplerStateDataCommon( SamplerStateFilterMode minFilterMode = SSFM_NEAREST
                              , SamplerStateFilterMode magFilterMode = SSFM_NEAREST
                              , SamplerStateWrapMode wrapSMode = SSWM_CLAMP_TO_EDGE
                              , SamplerStateWrapMode wrapTMode = SSWM_CLAMP_TO_EDGE
                              , SamplerStateWrapMode wrapRMode = SSWM_CLAMP_TO_EDGE
                              , SamplerStateCompareMode compareMode = SSCM_NONE)
          : SamplerStateData( SSDT_COMMON )
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
        RIX_CORE_API BufferDescription( BufferDescriptionType type = BDT_COMMON );
        RIX_CORE_API virtual ~BufferDescription();

        BufferDescriptionType m_type;
      };

      struct TextureDescription
      {
        RIX_CORE_API TextureDescription( TextureType type, InternalTextureFormat internalFormat,
                                         dp::util::PixelFormat pixelFormat, dp::util::DataType dataType,
                                         size_t width = 0, size_t height = 0, size_t depth = 0,
                                         size_t layers = 0, bool mipmaps = false );
        RIX_CORE_API virtual ~TextureDescription();

        TextureType           m_type;
        InternalTextureFormat m_internalFormat;
        dp::util::PixelFormat m_pixelFormat;
        dp::util::DataType    m_dataType;
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

      /** \brief Texture Data struct to pass references to previously constructed Buffers. Sets m_type to TDT_BUFFER. 
          \remarks This struct can only be used in conjunction with TT_BUFFER texture types.
       **/
      struct TextureDataBuffer : public TextureData
      {
        /** \brief Provide a buffer as source for a TT_BUFFER texture
            \param buffer The handle of the buffer
         **/
        RIX_CORE_API TextureDataBuffer( BufferSharedHandle const & buffer );

        BufferSharedHandle m_buffer;
      };

      /** \brief Texture Data struct to pass raw data pointers into the Renderer API. Sets m_type to TDT_POINTER.
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
        RIX_CORE_API TextureDataPtr( void const * data, dp::util::PixelFormat pixelFormat, dp::util::DataType pixelDataType );

        /** \brief Provide data with MipMapLevels. Each provider is for one MipMap level.
            \param data Pointer to pixel data
            \param numMipMapLevels Use 0 to automatically generate mipmaps
            \param pixelFormat   The format of a pixel in data
            \param pixelDataType The data type of a pixel in data
        **/
        RIX_CORE_API TextureDataPtr( void const * const *  data, unsigned int numMipMapLevels, dp::util::PixelFormat pixelFormat, dp::util::DataType pixelDataType );

        /** \brief Provide data with MipMapLevels and Number of layers. Data is organized as Layer0: MipMap0...n, Layer1: MipMap 0...n, ....
            \param data Pointer to pixel data
            \param numMipMapLevels Use 0 to automatically generate mipmaps
            \param numLayers Number of Layers for an array
            \param pixelFormat   The format of a pixel in data
            \param pixelDataType The data type of a pixel in data
        **/
        RIX_CORE_API TextureDataPtr( void const * const * data, unsigned int numMipMapLevels, unsigned int numLayers, dp::util::PixelFormat pixelFormat, dp::util::DataType pixelDataType );

        void const *          m_pData; // intermediate pointer to data for TextureDataPtr( void const * data )
        void const * const *  m_data;
        unsigned int          m_numLayers;
        unsigned int          m_numMipMapLevels;
        dp::util::PixelFormat m_pixelFormat;
        dp::util::DataType    m_pixelDataType;
      };

      struct GeometryInstanceDescription
      {
        RIX_CORE_API GeometryInstanceDescription( GeometryInstanceDescriptionType type = GIDT_COMMON );
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
        ProgramParameterDescriptorCommon( ProgramParameter* parameters, size_t numParameters )
          : ProgramParameterDescriptor( PPDT_COMMON )
          , m_parameters( parameters )
          , m_numParameters( numParameters )
        {
        }

        ProgramParameter* m_parameters;
        size_t            m_numParameters;
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
        ContainerDataRaw( size_t offset, const void *data, size_t size ) 
          : ContainerData( CDT_RAW )
          , m_offset( offset )
          , m_data( data )
          , m_size( size )
        {
        }

        size_t      m_offset;
        const void* m_data;
        size_t      m_size;
      };

      struct ContainerDataBuffer : public ContainerData
      {
        ContainerDataBuffer( BufferSharedHandle const & bufferHandle, size_t offset = 0, size_t length = ~0)
          : ContainerData( CDT_BUFFER )
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
          : ContainerData( CDT_SAMPLER )
          , m_samplerHandle( sampler )
        {}
    
        SamplerSharedHandle m_samplerHandle;
      };

      struct ContainerDataImage : public ContainerData
      {
        ContainerDataImage( TextureSharedHandle const & textureHandle, int level, bool layered, int layer, AccessType access )
          : ContainerData( CDT_IMAGE )
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
          : BufferReferences( BRT_SAMPLER )
          , m_numSlots( slots )
        {}

        size_t m_numSlots;
      };

      struct BufferReferencesBuffer : public BufferReferences
      {
        BufferReferencesBuffer( size_t slots)
          : BufferReferences( BRT_BUFFER )
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
        VertexFormatInfo( dp::util::Uint8 attributeIndex, dp::util::DataType dataType, dp::util::Uint8 numberOfComponents, bool normalized, dp::util::Uint8 streamId, size_t offset, size_t stride )
          : m_attributeIndex( attributeIndex )
          , m_numComponents( numberOfComponents )
          , m_streamId( streamId )
          , m_normalized( normalized )
          , m_dataType( dataType )
          , m_offset( offset )
          , m_stride( stride )
        {
        }

        dp::util::Uint8     m_attributeIndex;
        dp::util::Uint8     m_numComponents;
        dp::util::Uint8     m_streamId;
        bool                m_normalized;
        dp::util::DataType  m_dataType;
        size_t              m_offset;
        size_t              m_stride;
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
      RIX_CORE_API size_t getSizeOf( dp::util::DataType dataType );

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
        RIX_CORE_API virtual void indicesSetData( dp::rix::core::IndicesSharedHandle const & handle, dp::util::DataType dataType, dp::rix::core::BufferSharedHandle const & bufferHandle, size_t offset, size_t count ) = 0;

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
