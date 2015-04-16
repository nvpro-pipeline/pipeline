// Copyright NVIDIA Corporation 2011-2015
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
#include <dp/rix/gl/inc/RenderGroupGL.h>
#include <dp/rix/gl/inc/BufferGL.h>
#include <dp/rix/gl/inc/ContainerGL.h>
#include <dp/rix/gl/inc/GeometryDescriptionGL.h>
#include <dp/rix/gl/inc/GeometryGL.h>
#include <dp/rix/gl/inc/GeometryInstanceGL.h>
#include <dp/rix/gl/inc/IndicesGL.h>
#include <dp/rix/gl/inc/ProgramGL.h>
#include <dp/rix/gl/inc/ProgramPipelineGL.h>
#include <dp/rix/gl/inc/RenderEngineGL.h>
#include <dp/rix/gl/inc/TextureGL.h>
#include <dp/rix/gl/inc/Sampler.h>
#include <dp/rix/gl/inc/SamplerStateGL.h>
#include <dp/rix/gl/inc/VertexAttributesGL.h>

#include <iostream>

#if defined( _WIN32 )
#include <windows.h>
#endif

#include <GL/gl.h>

dp::rix::core::Renderer* createRenderer( const char *options )
{
  return new dp::rix::gl::RiXGL( options );
}


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

      void RiXGL::render( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::RenderOptions const & renderOptions )
      {
        DP_ASSERT( dp::rix::core::handleIsTypeOf<RenderGroupGL>( groupHandle ) );

        update();

        if ( !m_isRendering )
        {
          beginRender();
          m_renderEngine->render( dp::rix::core::handleCast<RenderGroupGL>(groupHandle), renderOptions );
          endRender();
        }
        else
        {
          m_renderEngine->render( dp::rix::core::handleCast<RenderGroupGL>(groupHandle), renderOptions );
        }
      }

      void RiXGL::render( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const * gis, size_t numGIs, dp::rix::core::RenderOptions const & renderOptions )
      {
        DP_ASSERT( dp::rix::core::handleIsTypeOf<RenderGroupGL>( groupHandle ) );

        update();

        if ( !m_isRendering )
        {
          beginRender();
          m_renderEngine->render( dp::rix::core::handleCast<RenderGroupGL>(groupHandle), gis, numGIs, renderOptions );
          endRender();
        }
        else
        {
          m_renderEngine->render( dp::rix::core::handleCast<RenderGroupGL>(groupHandle), gis, numGIs, renderOptions );
        }
      }

      void RiXGL::endRender()
      {
        DP_ASSERT( m_isRendering );

        m_isRendering = false;
        m_renderEngine->endRender();
      }

      dp::rix::core::ContainerSharedHandle RiXGL::containerCreate( dp::rix::core::ContainerDescriptorSharedHandle const & desc )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<ContainerDescriptorGL>(desc));

        ContainerDescriptorGLSharedHandle cdgl = dp::rix::core::handleCast<ContainerDescriptorGL>(desc);

        return new ContainerGL(this, cdgl);
      }

      void RiXGL::containerSetData(dp::rix::core::ContainerSharedHandle const & c, dp::rix::core::ContainerEntry entry, dp::rix::core::ContainerData const & containerData )
      {
         DP_ASSERT(dp::rix::core::handleIsTypeOf<ContainerGL>( c ));

         ContainerGLSharedHandle cgl = dp::rix::core::handleCast< ContainerGL >( c );

         cgl->setData( entry, containerData );
      }

      dp::rix::core::ContainerDescriptorSharedHandle RiXGL::containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptor const & programParameterDescriptor )
      {
        switch ( programParameterDescriptor.getType() )
        {
        case dp::rix::core::PPDT_COMMON:
        {
          DP_ASSERT( dynamic_cast<const dp::rix::core::ProgramParameterDescriptorCommon*>(&programParameterDescriptor) );
          dp::rix::core::ProgramParameterDescriptorCommon const & descriptor = static_cast<dp::rix::core::ProgramParameterDescriptorCommon const &>(programParameterDescriptor);
          return new ContainerDescriptorGL( this, descriptor.m_numParameters, descriptor.m_parameters );
        }
        default:
          DP_ASSERT( !"unsupported parameter descriptor" );
          return nullptr;
        }
      }

      unsigned int RiXGL::containerDescriptorGetNumberOfEntries( dp::rix::core::ContainerDescriptorSharedHandle const & desc )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<ContainerDescriptorGL>(desc));
        ContainerDescriptorGLSharedHandle descriptor = dp::rix::core::handleCast<ContainerDescriptorGL>(desc);

        return static_cast<unsigned int>(descriptor->m_parameterInfos.size());
      }

      dp::rix::core::ContainerEntry RiXGL::containerDescriptorGetEntry( dp::rix::core::ContainerDescriptorSharedHandle const & desc, unsigned int index )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<ContainerDescriptorGL>(desc));
        ContainerDescriptorGLSharedHandle descriptor = dp::rix::core::handleCast<ContainerDescriptorGL>(desc);

        DP_ASSERT( index < descriptor->m_parameterInfos.size() );
        return descriptor->generateEntry( index );
      }

      dp::rix::core::ContainerEntry RiXGL::containerDescriptorGetEntry( dp::rix::core::ContainerDescriptorSharedHandle const & desc, const char* name )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<ContainerDescriptorGL>(desc));
        ContainerDescriptorGLSharedHandle descriptor = dp::rix::core::handleCast<ContainerDescriptorGL>(desc);

        return descriptor->getEntry( name );
      }

      /** VertexFormat **/
      dp::rix::core::VertexFormatSharedHandle RiXGL::vertexFormatCreate( dp::rix::core::VertexFormatDescription const & vertexFormatDescription )
      {
        return VertexFormatGL::create( vertexFormatDescription );
      }

      /** VertexData **/
      dp::rix::core::VertexDataSharedHandle RiXGL::vertexDataCreate()
      {
        return new VertexDataGL();
      }

      void RiXGL::vertexDataSet( dp::rix::core::VertexDataSharedHandle const & handle, unsigned int index, dp::rix::core::BufferSharedHandle const & bufferHandle, size_t offset, size_t numberOfVertices )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<VertexDataGL>( handle ));
        DP_ASSERT(index <= RIX_GL_MAX_ATTRIBUTES);
        DP_ASSERT(dp::rix::core::handleIsTypeOf<BufferGL>( bufferHandle ));

        VertexDataGLSharedHandle formatData = dp::rix::core::handleCast<VertexDataGL>(handle);
        VertexDataGL::Data &data = formatData->m_data[index];
        dp::rix::core::handleAssign( data.m_buffer, dp::rix::core::handleCast<BufferGL>(bufferHandle) );
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
        DP_ASSERT(dp::rix::core::handleIsTypeOf<VertexAttributesGL>( handle ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<VertexDataGL>( vertexData ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<VertexFormatGL>( vertexFormat ));

        VertexAttributesGLSharedHandle attributes = dp::rix::core::handleCast<VertexAttributesGL>(handle);
        attributes->setVertexFormatGLHandle( dp::rix::core::handleCast<VertexFormatGL>(vertexFormat.get()) );
        attributes->setVertexDataGLHandle( dp::rix::core::handleCast<VertexDataGL>(vertexData.get()) );
      }

      /** Indices **/
      dp::rix::core::IndicesSharedHandle RiXGL::indicesCreate()
      {
        return new IndicesGL();
      }

      void RiXGL::indicesSetData( dp::rix::core::IndicesSharedHandle const & handle, dp::DataType dataType, dp::rix::core::BufferSharedHandle const & bufferHandle, size_t offset, size_t count )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<IndicesGL>( handle ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<BufferGL>( bufferHandle ));

        dp::rix::core::handleCast<IndicesGL>(handle)->setData( dataType, bufferHandle.get(), offset, count );
      }

      /** Buffer **/
      dp::rix::core::BufferSharedHandle RiXGL::bufferCreate( dp::rix::core::BufferDescription const & bufferDescription )
      {
        return BufferGL::create( bufferDescription );
      }

      void RiXGL::bufferSetSize( dp::rix::core::BufferSharedHandle const & handle, size_t width, size_t height /*= 0*/, size_t depth /*= 0 */ )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<BufferGL>( handle ));

        dp::rix::core::handleCast<BufferGL>( handle )->setSize( width, height, depth );
      }

      void RiXGL::bufferSetElementSize( dp::rix::core::BufferSharedHandle const & handle, size_t elementSize )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<BufferGL>( handle ));

        dp::rix::core::handleCast<BufferGL>( handle )->setElementSize( elementSize );
      }

      void RiXGL::bufferSetFormat( dp::rix::core::BufferSharedHandle const & handle, dp::rix::core::BufferFormat bufferFormat )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<BufferGL>( handle ));

        dp::rix::core::handleCast<BufferGL>( handle )->setFormat( bufferFormat );
      }

      void RiXGL::bufferUpdateData( dp::rix::core::BufferSharedHandle const & handle, size_t offset, void const * data, size_t size )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<BufferGL>( handle ));
        DP_ASSERT( data && size );

        dp::rix::core::handleCast<BufferGL>( handle )->updateData( offset, data, size );
      }

#if 1
      void RiXGL::bufferInitReferences ( dp::rix::core::BufferSharedHandle const & /*handle*/, dp::rix::core::BufferReferences const & /*refinfo*/ )
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
      void RiXGL::bufferSetReference  ( dp::rix::core::BufferSharedHandle const & /*handle*/, size_t /*slot*/, dp::rix::core::ContainerData const & /*data*/, dp::rix::core::BufferStoredReference & /*ref*/ )
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
        DP_ASSERT(dp::rix::core::handleIsTypeOf<BufferGL>( handle ));

        return dp::rix::core::handleCast<BufferGL>( handle )->map( accessType );
      }

      bool RiXGL::bufferUnmap( dp::rix::core::BufferSharedHandle const & handle )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<BufferGL>( handle ));

        return dp::rix::core::handleCast<BufferGL>( handle )->unmap();
      }

      /** Texture **/
      dp::rix::core::TextureSharedHandle RiXGL::textureCreate( dp::rix::core::TextureDescription const & description )
      {
        return TextureGL::create( description );
      }

      void RiXGL::textureSetData( dp::rix::core::TextureSharedHandle const & texture, dp::rix::core::TextureData const & data )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<TextureGL>(texture));

       dp::rix::core:: handleCast<TextureGL>( texture )->setData( data );
      }

      void RiXGL::textureSetDefaultSamplerState( dp::rix::core::TextureSharedHandle const & texture, dp::rix::core::SamplerStateSharedHandle const & samplerState )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<TextureGL>(texture));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<SamplerStateGL>(samplerState));

        dp::rix::core::handleCast<TextureGL>( texture )->setDefaultSamplerState( dp::rix::core::handleCast<SamplerStateGL>( samplerState.get() ) );
      }

      /** Sampler **/
      dp::rix::core::SamplerSharedHandle RiXGL::samplerCreate( )
      {
        return Sampler::create( );
      }

      void RiXGL::samplerSetSamplerState( dp::rix::core::SamplerSharedHandle const & sampler, dp::rix::core::SamplerStateSharedHandle const & samplerState )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<Sampler>(sampler));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<SamplerStateGL>(samplerState));

        dp::rix::core::handleCast<Sampler>( sampler )->setSamplerState( dp::rix::core::handleCast<SamplerStateGL>( samplerState ) );
      }

      void RiXGL::samplerSetTexture(dp::rix::core::SamplerSharedHandle const & sampler, dp::rix::core::TextureSharedHandle const & texture)
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<Sampler>(sampler));
        DP_ASSERT(!texture || dp::rix::core::handleIsTypeOf<TextureGL>(texture));

        dp::rix::core::handleCast<Sampler>( sampler )->setTexture( texture ? dp::rix::core::handleCast<TextureGL>( texture ) : TextureGLSharedHandle() );
      }


      /** SamplerState **/
      dp::rix::core::SamplerStateSharedHandle RiXGL::samplerStateCreate( dp::rix::core::SamplerStateData const & data )
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
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryDescriptionGL>( handle ));

        GeometryDescriptionGLSharedHandle geometryDescription = dp::rix::core::handleCast<GeometryDescriptionGL>(handle);
        geometryDescription->setPrimitiveType( primitiveType );
        geometryDescription->setPrimitiveRestartIndex( primitiveRestartIndex );
      }

      void RiXGL::geometryDescriptionSetBaseVertex( dp::rix::core::GeometryDescriptionSharedHandle const & handle, unsigned int baseVertex )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryDescriptionGL>( handle ));

        GeometryDescriptionGLSharedHandle geometryDescription = dp::rix::core::handleCast<GeometryDescriptionGL>(handle);
        geometryDescription->setBaseVertex( baseVertex );
      }

      void RiXGL::geometryDescriptionSetIndexRange( dp::rix::core::GeometryDescriptionSharedHandle const & handle, unsigned int first, unsigned int count )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryDescriptionGL>( handle ));

        GeometryDescriptionGLSharedHandle geometryDescription = dp::rix::core::handleCast<GeometryDescriptionGL>(handle);
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
        DP_ASSERT(                  dp::rix::core::handleIsTypeOf<GeometryGL>(handle));
        DP_ASSERT(                  dp::rix::core::handleIsTypeOf<GeometryDescriptionGL>(geometryDescriptionHandle));
        DP_ASSERT(                  dp::rix::core::handleIsTypeOf<VertexAttributesGL>(vertexAttributesHandle));
        DP_ASSERT(!indicesHandle || dp::rix::core::handleIsTypeOf<IndicesGL>(indicesHandle));

        GeometryGLSharedHandle geometryHandle = dp::rix::core::handleCast<GeometryGL>(handle);

        geometryHandle->setGeometryDescription( dp::rix::core::handleCast<GeometryDescriptionGL>(geometryDescriptionHandle) );
        geometryHandle->setVertexAttributes( dp::rix::core::handleCast<VertexAttributesGL>(vertexAttributesHandle) );
        geometryHandle->setIndices( indicesHandle ? dp::rix::core::handleCast<IndicesGL>(indicesHandle) : IndicesGLSharedHandle() );
      }

      /** GeometryInstance **/
      dp::rix::core::GeometryInstanceSharedHandle RiXGL::geometryInstanceCreate( dp::rix::core::GeometryInstanceDescription const & /*geometryInstanceDescription*/ )
      {
        return new GeometryInstanceGL();
      }

      bool RiXGL::geometryInstanceUseContainer( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::ContainerSharedHandle const & containerHandle )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryInstanceGL>( handle ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<ContainerGL>( containerHandle ));

        if (    !dp::rix::core::handleIsTypeOf<GeometryInstanceGL>( handle )
             || !dp::rix::core::handleIsTypeOf<ContainerGL>( containerHandle ) )
        {
          return false;
        }

        GeometryInstanceGLSharedHandle geometryInstance = dp::rix::core::handleCast<GeometryInstanceGL>(handle);

        if ( !geometryInstance->m_programPipeline )
        {
          return false;
        }

        ContainerGLSharedHandle cgl = dp::rix::core::handleCast<ContainerGL>(containerHandle);
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
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryInstanceGL>( handle ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryGL>( geometry ));

        dp::rix::core::handleCast<GeometryInstanceGL>(handle)->setGeometry( dp::rix::core::handleCast<GeometryGL>(geometry) );
      }

      void RiXGL::geometryInstanceSetProgramPipeline( dp::rix::core::GeometryInstanceSharedHandle const & handle, dp::rix::core::ProgramPipelineSharedHandle const & programPipelineHandle )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryInstanceGL>( handle ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<ProgramPipelineGL>( programPipelineHandle ));

        dp::rix::core::handleCast<GeometryInstanceGL>(handle)->setProgramPipeline( dp::rix::core::handleCast<ProgramPipelineGL>( programPipelineHandle ) );
      }
  
      void RiXGL::geometryInstanceSetVisible( dp::rix::core::GeometryInstanceSharedHandle const & handle, bool visible )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryInstanceGL>( handle ));
        dp::rix::core::handleCast<GeometryInstanceGL>( handle )->setVisible( visible );
      }

      /** Program **/
      dp::rix::core::ProgramSharedHandle RiXGL::programCreate( dp::rix::core::ProgramDescription const & description )
      {
        return new ProgramGL( description );
      }

      /** ProgramPipeline **/
      dp::rix::core::ProgramPipelineSharedHandle RiXGL::programPipelineCreate( dp::rix::core::ProgramSharedHandle const * programs, unsigned int numPrograms )
      {
        return new ProgramPipelineGL( programs, numPrograms );
      }

      /** RenderGroup **/
      dp::rix::core::RenderGroupSharedHandle RiXGL::renderGroupCreate()
      {
        return new RenderGroupGL( m_renderEngine );
      }

      void RiXGL::renderGroupAddGeometryInstance( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const & geometryHandle )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<RenderGroupGL>( groupHandle ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryInstanceGL>( geometryHandle ));

        RenderGroupGLSharedHandle group =dp::rix::core:: handleCast<RenderGroupGL>( groupHandle );
        GeometryInstanceGLSharedHandle geometry = dp::rix::core::handleCast<GeometryInstanceGL>( geometryHandle );

        group->addGeometryInstance( geometry.get() );
      }

      void RiXGL::renderGroupRemoveGeometryInstance( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const & geometryHandle )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<RenderGroupGL>( groupHandle ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<GeometryInstanceGL>( geometryHandle ));

        RenderGroupGLSharedHandle group = dp::rix::core::handleCast<RenderGroupGL>( groupHandle );
        GeometryInstanceGLSharedHandle geometry = dp::rix::core::handleCast<GeometryInstanceGL>( geometryHandle );

        group->removeGeometryInstance( geometry.get() );
      }

      void RiXGL::renderGroupSetProgramPipeline( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::ProgramPipelineSharedHandle const & programPipelineHandle )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<RenderGroupGL>( groupHandle ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<ProgramPipelineGL>( programPipelineHandle ));

        RenderGroupGLSharedHandle group = dp::rix::core::handleCast<RenderGroupGL>( groupHandle );
        ProgramPipelineGLSharedHandle programPipeline = dp::rix::core::handleCast<ProgramPipelineGL>( programPipelineHandle );

        group->setProgramPipeline( programPipeline.get() );
      }

      void RiXGL::renderGroupUseContainer( dp::rix::core::RenderGroupSharedHandle const & groupHandle, dp::rix::core::ContainerSharedHandle const & containerHandle )
      {
        DP_ASSERT(dp::rix::core::handleIsTypeOf<RenderGroupGL>( groupHandle ));
        DP_ASSERT(dp::rix::core::handleIsTypeOf<ContainerGL>( containerHandle ));

        RenderGroupGLHandle group = dp::rix::core::handleCast<RenderGroupGL>( groupHandle.get() );
        ContainerGLHandle container = dp::rix::core::handleCast<ContainerGL>( containerHandle.get() );

        group->useContainer( container );
      }

      ID RiXGL::aquireContainerID()
      {
        size_t freeID = m_containerFreeIDs.countLeadingZeroes();
        if (freeID == m_containerFreeIDs.getSize()) 
        {
          freeID = m_containerFreeIDs.getSize();
          // no free ids left, enlarge buffer
          m_containerFreeIDs.resize( m_containerFreeIDs.getSize() + 65536, true);
        }
        m_containerFreeIDs.disableBit(freeID);
        return dp::checked_cast<ID>(freeID);
      }

      void RiXGL::releaseUniqueContainerID(ID uniqueId)
      {
        DP_ASSERT(uniqueId < m_containerFreeIDs.getSize());
        DP_ASSERT(m_containerFreeIDs.getBit(uniqueId) == false);
        m_containerFreeIDs.enableBit(uniqueId);
      }

#if !defined(NDEBUG)
  #if defined(DP_OS_WINDOWS)
  #define CALLCONVENTION  WINAPI
  #else
  #define CALLCONVENTION
  #endif
      static void CALLCONVENTION debugMessageCallback( unsigned int /*source*/, unsigned int type, unsigned int /*id*/, unsigned int severity, int /*length*/, const char* message, const void* /*userParam*/ )
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
#endif

      void RiXGL::registerContext()
      {
        GLenum init = glewInit();
        DP_ASSERT( init == GLEW_OK );
        if ( init == GLEW_OK )
        {
#if !defined(NDEBUG)
          if ( !!GLEW_ARB_debug_output )
          {
            // the interface of the debugprocarb changed in OpenGL 4.5. Use a cast to make it work in all cases.
            glDebugMessageCallbackARB( reinterpret_cast<GLDEBUGPROCARB>(debugMessageCallback), nullptr );
            glEnable(GL_DEBUG_OUTPUT);
          }
#endif

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
