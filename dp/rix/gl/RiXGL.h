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

#include <dp/gl/Buffer.h>
#include <dp/gl/Texture.h>
#include <dp/util/Types.h>
#include <dp/rix/core/RiX.h>
#include <dp/rix/gl/Config.h>

#include <set>
#include <string>
#include <vector>

extern "C"
{
  RIX_GL_API dp::rix::core::Renderer* createRenderer( const char *options );
};

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      class RenderEngineGL;

#define DEFINE_RIX_GL_HANDLE( name ) \
      class name; \
      typedef name*  name##Handle; \
      typedef name*  name##WeakHandle; \
      typedef dp::rix::core::SmartHandle<name> name##SharedHandle;

      // keep () tight for visual assist
      DEFINE_RIX_GL_HANDLE(BufferGL);
      DEFINE_RIX_GL_HANDLE(CameraGL);
      DEFINE_RIX_GL_HANDLE(ContainerDescriptorGL);
      DEFINE_RIX_GL_HANDLE(ContainerGL);
      DEFINE_RIX_GL_HANDLE(GeometryDescriptionGL);
      DEFINE_RIX_GL_HANDLE(GeometryGL);
      DEFINE_RIX_GL_HANDLE(GeometryInstanceGL);
      DEFINE_RIX_GL_HANDLE(IndicesGL);
      DEFINE_RIX_GL_HANDLE(ProgramGL);
      DEFINE_RIX_GL_HANDLE(ProgramPipelineGL);
      DEFINE_RIX_GL_HANDLE(RenderGroupGL);
      DEFINE_RIX_GL_HANDLE(Sampler);
      DEFINE_RIX_GL_HANDLE(SamplerStateGL);
      DEFINE_RIX_GL_HANDLE(TextureGL);
      DEFINE_RIX_GL_HANDLE(TransformGL);
      DEFINE_RIX_GL_HANDLE(VertexFormatGL);
      DEFINE_RIX_GL_HANDLE(VertexDataGL);
      DEFINE_RIX_GL_HANDLE(VertexAttributesGL);
      DEFINE_RIX_GL_HANDLE(VertexDataGL);
      DEFINE_RIX_GL_HANDLE(VertexFormatGL);

#undef DEFINE_RIX_GL_HANDLE

      enum SamplerBorderColorDataType
      {
          SBCDT_FLOAT
        , SBCDT_UINT
        , SBCDT_INT
      };

      struct SamplerStateDataGL : public dp::rix::core::SamplerStateData
      {
        SamplerStateDataGL( unsigned int minFilterModeGL, unsigned int magFilterMode, unsigned int wrapMode, 
                            unsigned int compareModeGL, unsigned int compareFuncGL )
          : dp::rix::core::SamplerStateData( dp::rix::core::SSDT_NATIVE )
        {
          // argh!
          m_borderColorDataType = SBCDT_FLOAT;
          m_borderColor.f[0]  = 0.0f;
          m_borderColor.f[1]  = 0.0f;
          m_borderColor.f[2]  = 0.0f;
          m_borderColor.f[3]  = 0.0f;

          m_minFilterModeGL = minFilterModeGL;
          m_magFilterModeGL = magFilterMode;
          m_wrapSModeGL     = wrapMode;
          m_wrapTModeGL     = wrapMode;
          m_wrapRModeGL     = wrapMode;

          m_minLOD          = -1000.0f;
          m_maxLOD          =  1000.0f;
          m_LODBias         =  0.0f;

          m_compareModeGL   = compareModeGL;
          m_compareFuncGL   = compareFuncGL;

          m_maxAnisotropy   = 1.0f;
        }

        SamplerBorderColorDataType m_borderColorDataType;
        union
        {
          float        f[4];
          unsigned int ui[4];
          int          i[4];
        } m_borderColor;


        unsigned int m_minFilterModeGL;
        unsigned int m_magFilterModeGL;
        unsigned int m_wrapSModeGL;
        unsigned int m_wrapTModeGL;
        unsigned int m_wrapRModeGL;    

        float        m_minLOD;
        float        m_maxLOD;
        float        m_LODBias;

        unsigned int m_compareModeGL;
        unsigned int m_compareFuncGL;

        float        m_maxAnisotropy;
      };

      struct TextureDescriptionGL : public dp::rix::core::TextureDescription
      {
        // same constructor as in the base class to lower danger of parameter confusion when changing from base class
        // to GL class. the additional info has to be set explicitly, after base info was set to native.
        TextureDescriptionGL( dp::rix::core::TextureType type, dp::rix::core::InternalTextureFormat internalFormat,
          dp::util::PixelFormat pixelFormat, dp::util::DataType dataType,
          size_t width = 0, size_t height = 0, size_t depth = 0, size_t layers = 0, bool mipmaps = false )
          : TextureDescription( type, internalFormat, pixelFormat, dataType, width, height, depth, layers, mipmaps )
          , m_typeGL( 0 )
          , m_internalFormatGL( 0 )
        {}

        unsigned int m_typeGL;           // GL enum of the texture type (set type to TT_NATIVE to use it)
        unsigned int m_internalFormatGL; // GL enum of the internal format (set internalFormat to ITF_NATIVE to use it)
      };

      /** \brief Texture Data struct to pass in the GL id of a previously generated texture. Sets m_type to TDT_NATIVE.
       **/ 
      struct TextureDataGLTexture : public dp::rix::core::TextureData
      {      
        /** \brief Provide a texture id as source for the texture data
            \param id The GL id of the previously generated and prepared texture
         **/
        TextureDataGLTexture( dp::gl::SharedTexture const& texture )
          : dp::rix::core::TextureData( dp::rix::core::TDT_NATIVE )
          , m_texture( texture )
        {}

        dp::gl::SharedTexture  m_texture;
      };

      enum UsageHint
      {
        UH_STREAM_DRAW, 
        UH_STREAM_READ,
        UH_STREAM_COPY,
        UH_STATIC_DRAW,
        UH_STATIC_READ,
        UH_STATIC_COPY,
        UH_DYNAMIC_DRAW,
        UH_DYNAMIC_READ,
        UH_DYNAMIC_COPY
      };

      inline GLenum getGLUsage( UsageHint usageHint )
      {
        GLenum result = GL_STATIC_DRAW;

        switch ( usageHint )
        {
          case UH_STREAM_DRAW:  result = GL_STREAM_DRAW;  break;
          case UH_STREAM_READ:  result = GL_STREAM_READ;  break;
          case UH_STREAM_COPY:  result = GL_STREAM_COPY;  break;
          case UH_STATIC_DRAW:  result = GL_STATIC_DRAW;  break;
          case UH_STATIC_READ:  result = GL_STATIC_READ;  break;
          case UH_STATIC_COPY:  result = GL_STATIC_COPY;  break;
          case UH_DYNAMIC_DRAW: result = GL_DYNAMIC_DRAW; break;
          case UH_DYNAMIC_READ: result = GL_DYNAMIC_READ; break;
          case UH_DYNAMIC_COPY: result = GL_DYNAMIC_COPY; break;
        }
        return result;
      }

      struct BufferDescriptionGL : public dp::rix::core::BufferDescription
      {
        BufferDescriptionGL( UsageHint usageHint = UH_STATIC_DRAW, dp::gl::SharedBuffer const& buffer = nullptr )
          : dp::rix::core::BufferDescription( dp::rix::core::BDT_NATIVE )
          , m_buffer( buffer )
          , m_usageHint( usageHint )
        {
        }

        virtual ~BufferDescriptionGL()
        {
        }

        dp::gl::SharedBuffer  m_buffer;
        UsageHint             m_usageHint;
      };

      struct ProgramDescriptionGL : public dp::rix::core::ProgramDescription
      {
      };

      struct BufferStoredReferenceGL : public dp::rix::core::BufferStoredReference 
      {
        dp::util::Uint64  address;
      };

      class RiXGL : public dp::rix::core::Renderer
      {
      protected:
        virtual ~RiXGL();
        RiXGL( const char *renderEngine );

      public:
        friend RIX_GL_API dp::rix::core::Renderer* ::createRenderer( char const * );

        // delete the renderer
        RIX_GL_API virtual void deleteThis( void );

        RIX_GL_API virtual void update();
        RIX_GL_API virtual void beginRender();
        RIX_GL_API virtual void render( dp::rix::core::RenderGroupSharedHandle const & group, dp::rix::core::RenderOptions const & renderOptions = dp::rix::core::RenderOptions() );
        RIX_GL_API virtual void render( dp::rix::core::RenderGroupSharedHandle const & group, dp::rix::core::GeometryInstanceSharedHandle const * gis, size_t numGIs, dp::rix::core::RenderOptions const & renderOptions = dp::rix::core::RenderOptions() );
        RIX_GL_API virtual void endRender();
        /** Offset is in bytes always **/

        /** VertexFormat **/
        RIX_GL_API virtual dp::rix::core::VertexFormatSharedHandle vertexFormatCreate( dp::rix::core::VertexFormatDescription const & vertexFormatDescription );

        /** VertexData **/
        RIX_GL_API virtual dp::rix::core::VertexDataSharedHandle vertexDataCreate();
        RIX_GL_API virtual void vertexDataSet( dp::rix::core::VertexDataSharedHandle const & handle, unsigned int index, dp::rix::core::BufferSharedHandle const & bufferHandle, size_t offset, size_t numberOfVertices );

        /** VertexAttributes **/
        RIX_GL_API virtual dp::rix::core::VertexAttributesSharedHandle vertexAttributesCreate();
        RIX_GL_API virtual void vertexAttributesSet( dp::rix::core::VertexAttributesSharedHandle const & handle, dp::rix::core::VertexDataSharedHandle const & vertexData, dp::rix::core::VertexFormatSharedHandle const & vertexFormat );

        /** Indices **/
        RIX_GL_API virtual dp::rix::core::IndicesSharedHandle indicesCreate();
        RIX_GL_API virtual void indicesSetData( dp::rix::core::IndicesSharedHandle const & handle, dp::util::DataType dataType, dp::rix::core::BufferSharedHandle const & bufferHandle, size_t offset, size_t count );

        /** Buffer **/
        RIX_GL_API virtual dp::rix::core::BufferSharedHandle bufferCreate( dp::rix::core::BufferDescription const & bufferDescription = dp::rix::core::BufferDescription() );
        RIX_GL_API virtual void bufferSetSize( dp::rix::core::BufferSharedHandle const & handle, size_t width, size_t height = 0, size_t depth = 0 );
        RIX_GL_API virtual void bufferSetElementSize( dp::rix::core::BufferSharedHandle const & handle, size_t elementSize );
        RIX_GL_API virtual void bufferSetFormat( dp::rix::core::BufferSharedHandle const & handle, dp::rix::core::BufferFormat bufferFormat );
        RIX_GL_API virtual void bufferUpdateData( dp::rix::core::BufferSharedHandle const & handle, size_t offset, void const * data, size_t size );
        RIX_GL_API virtual void bufferInitReferences( dp::rix::core::BufferSharedHandle const & handle, dp::rix::core::BufferReferences const & refinfo );
        RIX_GL_API virtual void bufferSetReference( dp::rix::core::BufferSharedHandle const & handle, size_t slot, dp::rix::core::ContainerData const & data, dp::rix::core::BufferStoredReference & stored );
        RIX_GL_API virtual void* bufferMap( dp::rix::core::BufferSharedHandle const & handle, dp::rix::core::AccessType accessType );
        RIX_GL_API virtual bool  bufferUnmap( dp::rix::core::BufferSharedHandle const & handle );

        /** Texture **/
        RIX_GL_API virtual dp::rix::core::TextureSharedHandle textureCreate( dp::rix::core::TextureDescription const & description );
        RIX_GL_API virtual void textureSetData( dp::rix::core::TextureSharedHandle const & texture, dp::rix::core::TextureData const & data );
        RIX_GL_API virtual void textureSetDefaultSamplerState( dp::rix::core::TextureSharedHandle const & texture, dp::rix::core::SamplerStateSharedHandle const & samplerState );

        /** Sampler **/
        RIX_GL_API virtual dp::rix::core::SamplerSharedHandle samplerCreate( );
        RIX_GL_API virtual void samplerSetSamplerState( dp::rix::core::SamplerSharedHandle const & sampler, dp::rix::core::SamplerStateSharedHandle const & samplerState );
        RIX_GL_API virtual void samplerSetTexture( dp::rix::core::SamplerSharedHandle const & sampler, dp::rix::core::TextureSharedHandle const & texture);

        /** SamplerState **/
        // sampler state is an immutable object, pass its data into the create function
        RIX_GL_API virtual dp::rix::core::SamplerStateSharedHandle samplerStateCreate( dp::rix::core::SamplerStateData const & data );

        /** GeometryDescription **/
        RIX_GL_API virtual dp::rix::core::GeometryDescriptionSharedHandle geometryDescriptionCreate();
        RIX_GL_API virtual void geometryDescriptionSet( dp::rix::core::GeometryDescriptionSharedHandle const & geometryDescription, GeometryPrimitiveType type, unsigned int primitiveRestartIndex = ~0 );
        RIX_GL_API virtual void geometryDescriptionSetBaseVertex( dp::rix::core::GeometryDescriptionSharedHandle const & geometryDescription, unsigned int baseVertex );
        RIX_GL_API virtual void geometryDescriptionSetIndexRange( dp::rix::core::GeometryDescriptionSharedHandle const & geometryDescription, unsigned int first, unsigned int count );

        /** Geometry **/
        RIX_GL_API virtual dp::rix::core::GeometrySharedHandle geometryCreate();
        RIX_GL_API virtual void geometrySetData( dp::rix::core::GeometrySharedHandle const & geometry, dp::rix::core::GeometryDescriptionSharedHandle const & geometryDescription
                                               , dp::rix::core::VertexAttributesSharedHandle const & vertexAttributes, dp::rix::core::IndicesSharedHandle const & indices = 0 );

        /** GeometryInstance **/
        RIX_GL_API virtual dp::rix::core::GeometryInstanceSharedHandle geometryInstanceCreate( dp::rix::core::GeometryInstanceDescription const & geometryInstanceDescription = dp::rix::core::GeometryInstanceDescription() );
        RIX_GL_API virtual bool geometryInstanceUseContainer( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::ContainerSharedHandle const & containerHandle );
        RIX_GL_API virtual void geometryInstanceSetGeometry( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::GeometrySharedHandle const & geometry );
        RIX_GL_API virtual void geometryInstanceSetProgramPipeline( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::ProgramPipelineSharedHandle const & programPipelineHandle );
        RIX_GL_API virtual void geometryInstanceSetVisible( dp::rix::core::GeometryInstanceSharedHandle const & handle, bool visible );

        /** Program **/
        RIX_GL_API virtual dp::rix::core::ProgramSharedHandle programCreate( dp::rix::core::ProgramDescription const & description );

        /** ProgramPipeline **/
        RIX_GL_API virtual dp::rix::core::ProgramPipelineSharedHandle programPipelineCreate( dp::rix::core::ProgramSharedHandle const * programs, unsigned int numPrograms );

        /** Container **/
        RIX_GL_API virtual dp::rix::core::ContainerSharedHandle containerCreate( dp::rix::core::ContainerDescriptorSharedHandle const & desc );
        RIX_GL_API virtual void containerSetData(dp::rix::core::ContainerSharedHandle const & containerHandle, dp::rix::core::ContainerEntry entry, dp::rix::core::ContainerData const & containerData );

        /** ContainerDescriptor **/
        RIX_GL_API virtual dp::rix::core::ContainerDescriptorSharedHandle containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptor const & programParameterDescriptor );
        RIX_GL_API virtual unsigned int containerDescriptorGetNumberOfEntries( dp::rix::core::ContainerDescriptorSharedHandle const & desc );
        RIX_GL_API virtual dp::rix::core::ContainerEntry containerDescriptorGetEntry( dp::rix::core::ContainerDescriptorSharedHandle const & desc, unsigned int index );
        RIX_GL_API virtual dp::rix::core::ContainerEntry containerDescriptorGetEntry( dp::rix::core::ContainerDescriptorSharedHandle const & desc, char const * name );

        /** RenderGroup **/
        RIX_GL_API virtual dp::rix::core::RenderGroupSharedHandle renderGroupCreate();
        RIX_GL_API virtual void renderGroupAddGeometryInstance( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const & geometryHandle );
        RIX_GL_API virtual void renderGroupRemoveGeometryInstance( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const & geometryHandle );
        RIX_GL_API virtual void renderGroupSetProgramPipeline( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::ProgramPipelineSharedHandle const & programPipelineHandle );
        RIX_GL_API virtual void renderGroupUseContainer( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::ContainerSharedHandle const & containerHandle );

        // TODO: handle context registration differently. this forces the app to include RiXGL.h
        RIX_GL_API virtual void registerContext(); // register the current active context. Currently this may be called only once.

      private:
        RenderEngineGL* m_renderEngine;
        std::string     m_renderEngineName;

        bool m_isRendering;
      };

    } // namespace gl
  } // namespace rix
} // namespace dp

// keep compability for some time
namespace RiX
{
  namespace GL
  {
    using namespace dp::rix::gl;
  }
}

