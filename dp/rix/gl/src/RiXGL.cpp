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


#include <GL/glew.h>

#include <dp/rix/gl/RiXGL.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include "Vector.h"
#include "Matrix.h"
#include "RenderGroupGL.h"
#include "BufferGL.h"
#include "ContainerGL.h"
#include "GeometryDescriptionGL.h"
#include "GeometryGL.h"
#include "GeometryInstanceGL.h"
#include "IndicesGL.h"
#include "ProgramGL.h"
#include "ProgramPipelineGL.h"
#include "RenderEngineGL.h"
#include "TextureGL.h"
#include "Sampler.h"
#include "SamplerStateGL.h"
#include "VertexAttributesGL.h"

#include <iostream>

#if defined( _WIN32 )
#include <windows.h>
#endif

#include <GL/gl.h>

dp::rix::core::Renderer* createRenderer( const char *options )
{
  return new dp::rix::gl::RiXGL( options );
}


using namespace RiX;

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      /**************/
      /* RiXGL */
      /**************/
      RiXGL::RiXGL( const char *renderEngine )
      {
        m_renderEngineName = renderEngine ? renderEngine : "Bindless";
        m_renderEngine = getRenderEngine( m_renderEngineName.c_str() );
        m_isRendering = false;
      }

      RiXGL::~RiXGL()
      {
        delete m_renderEngine;
      }

      void RiXGL::deleteThis()
      {
        delete this;
      }

      inline void RiXGL::update()
      {
      }

      void RiXGL::beginRender()
      {
        DP_ASSERT( !m_isRendering );

        m_isRendering = true;
        m_renderEngine->beginRender();
      }

      void RiXGL::render( RenderGroupSharedHandle const & groupHandle, dp::rix::core::RenderOptions const & renderOptions )
      {
        DP_ASSERT( handleIsTypeOf<RenderGroupGL>( groupHandle ) );

        update();

        if ( !m_isRendering )
        {
          beginRender();
          m_renderEngine->render( handleCast<RenderGroupGL>(groupHandle), renderOptions );
          endRender();
        }
        else
        {
          m_renderEngine->render( handleCast<RenderGroupGL>(groupHandle), renderOptions );
        }
      }

      void RiXGL::render( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const * gis, size_t numGIs, dp::rix::core::RenderOptions const & renderOptions )
      {
        DP_ASSERT( handleIsTypeOf<RenderGroupGL>( groupHandle ) );

        update();

        if ( !m_isRendering )
        {
          beginRender();
          m_renderEngine->render( handleCast<RenderGroupGL>(groupHandle), gis, numGIs, renderOptions );
          endRender();
        }
        else
        {
          m_renderEngine->render( handleCast<RenderGroupGL>(groupHandle), gis, numGIs, renderOptions );
        }
      }

      void RiXGL::endRender()
      {
        DP_ASSERT( m_isRendering );

        m_isRendering = false;
        m_renderEngine->endRender();
      }

      ContainerSharedHandle RiXGL::containerCreate( ContainerDescriptorSharedHandle const & desc )
      {
        DP_ASSERT( handleIsTypeOf<ContainerDescriptorGL>( desc ) );

        ContainerDescriptorGLSharedHandle cdgl = handleCast< ContainerDescriptorGL >( desc );

        return new ContainerGL( cdgl );
      }

      void RiXGL::containerSetData(ContainerSharedHandle const & c, ContainerEntry entry, ContainerData const & containerData )
      {
         DP_ASSERT( handleIsTypeOf<ContainerGL>( c ) );

         ContainerGLSharedHandle cgl = handleCast< ContainerGL >( c );

         cgl->setData( entry, containerData );
      }

      ContainerDescriptorSharedHandle RiXGL::containerDescriptorCreate( ProgramParameterDescriptor const & programParameterDescriptor )
      {
        switch ( programParameterDescriptor.getType() )
        {
        case PPDT_COMMON:
        {
          DP_ASSERT( dynamic_cast<const ProgramParameterDescriptorCommon*>(&programParameterDescriptor) );
          ProgramParameterDescriptorCommon const & descriptor = static_cast<ProgramParameterDescriptorCommon const &>(programParameterDescriptor);
          return new ContainerDescriptorGL( this, descriptor.m_numParameters, descriptor.m_parameters );
        }
        default:
          DP_ASSERT( !"unsupported parameter descriptor" );
          return nullptr;
        }
      }

      unsigned int RiXGL::containerDescriptorGetNumberOfEntries( ContainerDescriptorSharedHandle const & desc )
      {
        DP_ASSERT( handleIsTypeOf<ContainerDescriptorGL>(desc) );
        ContainerDescriptorGLSharedHandle descriptor = handleCast<ContainerDescriptorGL>(desc);

        return static_cast<unsigned int>(descriptor->m_parameterInfos.size());
      }

      ContainerEntry RiXGL::containerDescriptorGetEntry( ContainerDescriptorSharedHandle const & desc, unsigned int index )
      {
        DP_ASSERT( handleIsTypeOf<ContainerDescriptorGL>(desc) );
        ContainerDescriptorGLSharedHandle descriptor = handleCast<ContainerDescriptorGL>(desc);

        DP_ASSERT( index < descriptor->m_parameterInfos.size() );
        return descriptor->generateEntry( index );
      }

      ContainerEntry RiXGL::containerDescriptorGetEntry( ContainerDescriptorSharedHandle const & desc, const char* name )
      {
        DP_ASSERT( handleIsTypeOf<ContainerDescriptorGL>(desc) );
        ContainerDescriptorGLSharedHandle descriptor = handleCast<ContainerDescriptorGL>(desc);

        return descriptor->getEntry( name );
      }

      /** VertexFormat **/
      dp::rix::core::VertexFormatSharedHandle RiXGL::vertexFormatCreate( VertexFormatDescription const & vertexFormatDescription )
      {
        return VertexFormatGL::create( vertexFormatDescription );
      }

      /** VertexData **/
      dp::rix::core::VertexDataSharedHandle RiXGL::vertexDataCreate()
      {
        return new VertexDataGL();
      }

      void RiXGL::vertexDataSet( VertexDataSharedHandle const & handle, unsigned int index, BufferSharedHandle const & bufferHandle, size_t offset, size_t numberOfVertices )
      {
        DP_ASSERT( handleIsTypeOf<VertexDataGL>( handle ) );
        DP_ASSERT( index <= RIX_GL_MAX_ATTRIBUTES );
        DP_ASSERT( handleIsTypeOf<BufferGL>( bufferHandle ) );

        VertexDataGLSharedHandle formatData = handleCast<VertexDataGL>(handle);
        VertexDataGL::Data &data = formatData->m_data[index];
        handleAssign( data.m_buffer, handleCast<BufferGL>(bufferHandle) );
        data.m_offset = offset;
        formatData->m_numberOfVertices = numberOfVertices;
      }


      /** VertexAttributes **/
      dp::rix::core::VertexAttributesSharedHandle RiXGL::vertexAttributesCreate()
      {         
        return new VertexAttributesGL();
      }

      void RiXGL::vertexAttributesSet( dp::rix::core::VertexAttributesSharedHandle const & handle, dp::rix::core::VertexDataSharedHandle const & vertexData, dp::rix::core::VertexFormatSharedHandle const & vertexFormat )
      {
        DP_ASSERT( handleIsTypeOf<VertexAttributesGL>( handle ) );
        DP_ASSERT( handleIsTypeOf<VertexDataGL>( vertexData ) );
        DP_ASSERT( handleIsTypeOf<VertexFormatGL>( vertexFormat ) );

        VertexAttributesGLSharedHandle attributes = handleCast<VertexAttributesGL>(handle);
        attributes->setVertexFormatGLHandle( handleCast<VertexFormatGL>(vertexFormat.get()) );
        attributes->setVertexDataGLHandle( handleCast<VertexDataGL>(vertexData.get()) );
      }

      /** Indices **/
      dp::rix::core::IndicesSharedHandle RiXGL::indicesCreate()
      {
        return new IndicesGL();
      }

      void RiXGL::indicesSetData( IndicesSharedHandle const & handle, dp::util::DataType dataType, BufferSharedHandle const & bufferHandle, size_t offset, size_t count )
      {
        DP_ASSERT( handleIsTypeOf<IndicesGL>( handle ) );
        DP_ASSERT( handleIsTypeOf<BufferGL>( bufferHandle ) );

        handleCast<IndicesGL>(handle)->setData( dataType, bufferHandle.get(), offset, count );
      }

      /** Buffer **/
      dp::rix::core::BufferSharedHandle RiXGL::bufferCreate( dp::rix::core::BufferDescription const & bufferDescription )
      {
        return BufferGL::create( bufferDescription );
      }

      void RiXGL::bufferSetSize( dp::rix::core::BufferSharedHandle const & handle, size_t width, size_t height /*= 0*/, size_t depth /*= 0 */ )
      {
        DP_ASSERT( handleIsTypeOf<BufferGL>( handle ) );

        handleCast<BufferGL>( handle )->setSize( width, height, depth );
      }

      void RiXGL::bufferSetElementSize( dp::rix::core::BufferSharedHandle const & handle, size_t elementSize )
      {
        DP_ASSERT( handleIsTypeOf<BufferGL>( handle ) );

        handleCast<BufferGL>( handle )->setElementSize( elementSize );
      }

      void RiXGL::bufferSetFormat( dp::rix::core::BufferSharedHandle const & handle, dp::rix::core::BufferFormat bufferFormat )
      {
        DP_ASSERT( handleIsTypeOf<BufferGL>( handle ) );

        handleCast<BufferGL>( handle )->setFormat( bufferFormat );
      }

      void RiXGL::bufferUpdateData( dp::rix::core::BufferSharedHandle const & handle, size_t offset, void const * data, size_t size )
      {
        DP_ASSERT( handleIsTypeOf<BufferGL>( handle ) );
        DP_ASSERT( data && size );

        handleCast<BufferGL>( handle )->updateData( offset, data, size );
      }

#if 1
      void RiXGL::bufferInitReferences ( dp::rix::core::BufferSharedHandle const & /*handle*/, BufferReferences const & /*refinfo*/ )
      {
        DP_ASSERT(!"not supported");
#else
      void RiXGL::bufferInitReferences ( dp::rix::core::BufferHandle handle, BufferReferences const & refinfo )
      {
        DP_ASSERT( handleIsTypeOf<BufferGL>( handle ) );

        BufferGLHandle bufferHandle = handleCast<BufferGL>( handle );
        bufferHandle->initReferences( refinfo );
#endif
      }

#if 1
      void RiXGL::bufferSetReference  ( dp::rix::core::BufferSharedHandle const & /*handle*/, size_t /*slot*/, ContainerData const & /*data*/,  BufferStoredReference & /*ref*/ )
      {
        DP_ASSERT(!"not supported");
#else
      void RiXGL::bufferSetReference  ( dp::rix::core::BufferHandle handle, size_t slot, ContainerData const & data,  BufferStoredReference & ref )
      {
        DP_ASSERT( handleIsTypeOf<BufferGL>( handle ) );
        DP_ASSERT( dynamic_cast<BufferStoredReferenceGL*>(&ref) );

        BufferGLHandle bufferHandle = handleCast<BufferGL>( handle );
        bufferHandle->setReference( slot, data, static_cast<BufferStoredReferenceGL&>(ref) );
#endif
      }

      void* RiXGL::bufferMap( dp::rix::core::BufferSharedHandle const & handle, dp::rix::core::AccessType accessType )
      {
        DP_ASSERT( handleIsTypeOf<BufferGL>( handle ) );

        return handleCast<BufferGL>( handle )->map( accessType );
      }

      bool RiXGL::bufferUnmap( dp::rix::core::BufferSharedHandle const & handle )
      {
        DP_ASSERT( handleIsTypeOf<BufferGL>( handle ) );

        return handleCast<BufferGL>( handle )->unmap();
      }

      /** Texture **/
      TextureSharedHandle RiXGL::textureCreate( TextureDescription const & description )
      {
        return TextureGL::create( description );
      }

      void RiXGL::textureSetData( TextureSharedHandle const & texture, TextureData const & data )
      {
        DP_ASSERT( handleIsTypeOf<TextureGL>(texture) );

        handleCast<TextureGL>( texture )->setData( data );
      }

      void RiXGL::textureSetDefaultSamplerState( TextureSharedHandle const & texture, SamplerStateSharedHandle const & samplerState )
      {
        DP_ASSERT( handleIsTypeOf<TextureGL>(texture) );
        DP_ASSERT( handleIsTypeOf<SamplerStateGL>(samplerState) );

        handleCast<TextureGL>( texture )->setDefaultSamplerState( handleCast<SamplerStateGL>( samplerState.get() ) );
      }

      /** Sampler **/
      dp::rix::core::SamplerSharedHandle RiXGL::samplerCreate( )
      {
        return Sampler::create( );
      }

      void RiXGL::samplerSetSamplerState( dp::rix::core::SamplerSharedHandle const & sampler, dp::rix::core::SamplerStateSharedHandle const & samplerState )
      {
        DP_ASSERT( handleIsTypeOf<Sampler>(sampler) );
        DP_ASSERT( handleIsTypeOf<SamplerState>(samplerState) );

        handleCast<Sampler>( sampler )->setSamplerState( handleCast<SamplerStateGL>( samplerState ) );
      }

      void RiXGL::samplerSetTexture(dp::rix::core::SamplerSharedHandle const & sampler, dp::rix::core::TextureSharedHandle const & texture)
      {
        DP_ASSERT( handleIsTypeOf<Sampler>(sampler) );
        DP_ASSERT( !texture || handleIsTypeOf<Texture>(texture) );

        handleCast<Sampler>( sampler )->setTexture( texture ? handleCast<TextureGL>( texture ) : TextureGLSharedHandle() );
      }


      /** SamplerState **/
      SamplerStateSharedHandle RiXGL::samplerStateCreate( SamplerStateData const & data )
      {
        return SamplerStateGL::create( data );
      }

      /** GeometryDescription **/
      dp::rix::core::GeometryDescriptionSharedHandle RiXGL::geometryDescriptionCreate()
      {
        return new GeometryDescriptionGL();
      }

      void RiXGL::geometryDescriptionSet( dp::rix::core::GeometryDescriptionSharedHandle const & handle, GeometryPrimitiveType primitiveType, unsigned int primitiveRestartIndex )
      {
        DP_ASSERT( handleIsTypeOf<GeometryDescriptionGL>( handle ) );

        GeometryDescriptionGLSharedHandle geometryDescription = handleCast<GeometryDescriptionGL>(handle);
        geometryDescription->setPrimitiveType( primitiveType );
        geometryDescription->setPrimitiveRestartIndex( primitiveRestartIndex );
      }

      void RiXGL::geometryDescriptionSetBaseVertex( dp::rix::core::GeometryDescriptionSharedHandle const & handle, unsigned int baseVertex )
      {
        DP_ASSERT( handleIsTypeOf<GeometryDescriptionGL>( handle ) );

        GeometryDescriptionGLSharedHandle geometryDescription = handleCast<GeometryDescriptionGL>(handle);
        geometryDescription->setBaseVertex( baseVertex );
      }

      void RiXGL::geometryDescriptionSetIndexRange( dp::rix::core::GeometryDescriptionSharedHandle const & handle, unsigned int first, unsigned int count )
      {
        DP_ASSERT( handleIsTypeOf<GeometryDescriptionGL>( handle ) );

        GeometryDescriptionGLSharedHandle geometryDescription = handleCast<GeometryDescriptionGL>(handle);
        geometryDescription->setIndexRange( first, count );
      }

      /** Geometry **/
      dp::rix::core::GeometrySharedHandle RiXGL::geometryCreate()
      {
        return new GeometryGL();
      }

      void RiXGL::geometrySetData( dp::rix::core::GeometrySharedHandle const & handle, dp::rix::core::GeometryDescriptionSharedHandle const & geometryDescriptionHandle
                                 , dp::rix::core::VertexAttributesSharedHandle const & vertexAttributesHandle, dp::rix::core::IndicesSharedHandle const & indicesHandle )
      {
        DP_ASSERT( handleIsTypeOf<GeometryGL>( handle ) );
        DP_ASSERT( handleIsTypeOf<GeometryDescriptionGL>( geometryDescriptionHandle ) );
        DP_ASSERT( handleIsTypeOf<VertexAttributesGL>( vertexAttributesHandle ) );
        DP_ASSERT( !indicesHandle || handleIsTypeOf<IndicesGL>( indicesHandle ) );

        GeometryGLSharedHandle geometryHandle = handleCast<GeometryGL>(handle);

        geometryHandle->setGeometryDescription( handleCast<GeometryDescriptionGL>(geometryDescriptionHandle) );
        geometryHandle->setVertexAttributes( handleCast<VertexAttributesGL>(vertexAttributesHandle) );
        geometryHandle->setIndices( indicesHandle ? handleCast<IndicesGL>(indicesHandle) : IndicesGLSharedHandle() );
      }

      /** GeometryInstance **/
      dp::rix::core::GeometryInstanceSharedHandle RiXGL::geometryInstanceCreate( dp::rix::core::GeometryInstanceDescription const & /*geometryInstanceDescription*/ )
      {
        return new GeometryInstanceGL();
      }

      bool RiXGL::geometryInstanceUseContainer( GeometryInstanceSharedHandle const & handle, ContainerSharedHandle const & containerHandle )
      {
        DP_ASSERT( handleIsTypeOf<GeometryInstanceGL>( handle ) );
        DP_ASSERT( handleIsTypeOf<ContainerGL>( containerHandle ) );

        if (    !handleIsTypeOf<GeometryInstanceGL>( handle )
             || !handleIsTypeOf<ContainerGL>( containerHandle ) )
        {
          return false;
        }

        GeometryInstanceGLSharedHandle geometryInstance = handleCast<GeometryInstanceGL>(handle);

        if ( !geometryInstance->m_programPipeline )
        {
          return false;
        }

        ContainerGLSharedHandle cgl = handleCast<ContainerGL>(containerHandle);
        if ( !geometryInstance->useContainer( cgl ) )
        {
          return false;
        }

        if ( geometryInstance->m_renderGroup )
        {
          geometryInstance->m_renderGroup->markDirty( geometryInstance.get() );
        }

        return true;
      };

      void RiXGL::geometryInstanceSetGeometry( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::GeometrySharedHandle const & geometry )
      {
        DP_ASSERT( handleIsTypeOf<GeometryInstanceGL>( handle ) );
        DP_ASSERT( handleIsTypeOf<GeometryGL>( geometry ) );

        handleCast<GeometryInstanceGL>(handle)->setGeometry( handleCast<GeometryGL>(geometry) );
      }

      void RiXGL::geometryInstanceSetProgramPipeline( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::ProgramPipelineSharedHandle const & programPipelineHandle )
      {
        DP_ASSERT( handleIsTypeOf<GeometryInstanceGL>( handle ) );
        DP_ASSERT( handleIsTypeOf<ProgramPipelineGL>( programPipelineHandle ) );

        handleCast<GeometryInstanceGL>(handle)->setProgramPipeline( handleCast<ProgramPipelineGL>( programPipelineHandle ) );
      }
  
      void RiXGL::geometryInstanceSetVisible( GeometryInstanceSharedHandle const & handle, bool visible )
      {
        DP_ASSERT( handleIsTypeOf<GeometryInstanceGL>( handle ) );
        handleCast<GeometryInstanceGL>( handle )->setVisible( visible );
      }

      /** Program **/
      dp::rix::core::ProgramSharedHandle RiXGL::programCreate( ProgramDescription const & description )
      {
        return new ProgramGL( description );
      }

      /** ProgramPipeline **/
      ProgramPipelineSharedHandle RiXGL::programPipelineCreate( ProgramSharedHandle const * programs, unsigned int numPrograms )
      {
        return new ProgramPipelineGL( programs, numPrograms );
      }

      /** RenderGroup **/
      RenderGroupSharedHandle RiXGL::renderGroupCreate()
      {
        return new RenderGroupGL( m_renderEngine );
      }

      void RiXGL::renderGroupAddGeometryInstance( RenderGroupSharedHandle const & groupHandle, GeometryInstanceSharedHandle const & geometryHandle )
      {
        DP_ASSERT( handleIsTypeOf<RenderGroupGL>( groupHandle ) );
        DP_ASSERT( handleIsTypeOf<GeometryInstanceGL>( geometryHandle ) );

        RenderGroupGLSharedHandle group = handleCast<RenderGroupGL>( groupHandle );
        GeometryInstanceGLSharedHandle geometry = handleCast<GeometryInstanceGL>( geometryHandle );

        group->addGeometryInstance( geometry.get() );
      }

      void RiXGL::renderGroupRemoveGeometryInstance( RenderGroupSharedHandle const & groupHandle, GeometryInstanceSharedHandle const & geometryHandle )
      {
        DP_ASSERT( handleIsTypeOf<RenderGroupGL>( groupHandle ) );
        DP_ASSERT( handleIsTypeOf<GeometryInstanceGL>( geometryHandle ) );

        RenderGroupGLSharedHandle group = handleCast<RenderGroupGL>( groupHandle );
        GeometryInstanceGLSharedHandle geometry = handleCast<GeometryInstanceGL>( geometryHandle );

        group->removeGeometryInstance( geometry.get() );
      }

      void RiXGL::renderGroupSetProgramPipeline( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::ProgramPipelineSharedHandle const & programPipelineHandle )
      {
        DP_ASSERT( handleIsTypeOf<RenderGroup>( groupHandle ) );
        DP_ASSERT( handleIsTypeOf<ProgramPipeline>( programPipelineHandle ) );

        RenderGroupGLSharedHandle group = handleCast<RenderGroupGL>( groupHandle );
        ProgramPipelineGLSharedHandle programPipeline = handleCast<ProgramPipelineGL>( programPipelineHandle );

        group->setProgramPipeline( programPipeline.get() );
      }

      void RiXGL::renderGroupUseContainer( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::ContainerSharedHandle const & containerHandle )
      {
        DP_ASSERT( handleIsTypeOf<RenderGroupGL>( groupHandle ) );
        DP_ASSERT( handleIsTypeOf<ContainerGL>( containerHandle ) );

        RenderGroupGLHandle group = handleCast<RenderGroupGL>( groupHandle.get() );
        ContainerGLHandle container = handleCast<ContainerGL>( containerHandle.get() );

        group->useContainer( container );
      }

  #if defined(WIN32)
  #define CALLCONVENTION  WINAPI
  #else
  #define CALLCONVENTION
  #endif
      static void CALLCONVENTION debugMessageCallback( unsigned int /*source*/, unsigned int type, unsigned int /*id*/, unsigned int severity, int /*length*/, const char* message, void* /*userParam*/ )
      {
        std::string header = "OpenGL ";

        switch ( type )
        {
        case GL_DEBUG_TYPE_ERROR_ARB:
          header += "ERROR ";
          break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
          header += "Deprecated Behavior ";
          break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
          header += "Undefined Behavior ";
          break;
        case GL_DEBUG_TYPE_PORTABILITY_ARB:
          header += "Portability ";
          break;
        case GL_DEBUG_TYPE_PERFORMANCE_ARB:
          header += "Performance ";
          break;
        case GL_DEBUG_TYPE_OTHER_ARB:
          header += "Other ";
          break;
        }

        switch ( severity )
        {
        case GL_DEBUG_SEVERITY_HIGH_ARB:
          header += "(high): ";
          break;
        case GL_DEBUG_SEVERITY_MEDIUM_ARB:
          header += "(medium): ";
          break;
        case GL_DEBUG_SEVERITY_LOW_ARB:
          header += "(low): ";
          break;
        }

        std::cerr << header << message << std::endl;

        if ( type == GL_DEBUG_TYPE_ERROR_ARB ) 
        {
          // DAR FIXME The OpenGL driver reports an error when a texture sampler is unassigned although it's not used, that is not fatal: DP_ASSERT( 0 && "OpenGL Error" );
        }
      }

      void RiXGL::registerContext()
      {
        GLenum init = glewInit();
        DP_ASSERT( init == GLEW_OK );
        if ( init == GLEW_OK )
        {
          if ( !!GLEW_ARB_debug_output )
          {
            glDebugMessageCallbackARB( debugMessageCallback, nullptr );
          }

          initBufferBinding();
        }
        else
        {
          /* TODO throw exception */
        }
      }
    } // namespace gl
  } // namespace rix
} // namespace dp
