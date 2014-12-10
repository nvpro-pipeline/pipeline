// Copyright NVIDIA Corporation 2011
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


#include "RenderEngineGLImpl.hpp"
#include "RenderEngineGLBindlessVAO.h"
#include "GeometryGL.h"
#include "GeometryDescriptionGL.h"
#include "GeometryInstanceGL.h"
#include "VertexAttributesGL.h"
#include "ProgramGL.h"
#include "ProgramPipelineGL.h"
#include "IndicesGL.h"
#include "VertexFormatGL.h"
#include "DataTypeConversionGL.h"

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      void VertexAttributeCache<VertexCacheBindlessVAO>::beginFrame()
      {
      }

      void VertexAttributeCache<VertexCacheBindlessVAO>::endFrame()
      {
      }

      void VertexAttributeCache<VertexCacheBindlessVAO>::updateGeometryInstanceCacheEntry( GeometryInstanceGLHandle gi, GeometryInstanceCacheEntry& geometryInstanceCacheEntry, AttributeCacheEntry* /*attributeCacheEntry*/ )
      {
        // TODO implement direct conversion
        VertexDataCache *vertexCache = new VertexDataCache;
        gi->m_vertexCache = vertexCache;

        if ( !vertexCache->m_vao )
        {
          glGenVertexArrays( 1, &vertexCache->m_vao );
        }

        glBindVertexArray( vertexCache->m_vao );
        glEnableClientState(GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV);
        glEnableClientState(GL_ELEMENT_ARRAY_UNIFIED_NV );

        VertexAttributesGLSharedHandle const& vertexAttributes = gi->getGeometry()->getVertexAttributes();
        VertexFormatGLHandle formatHandle = vertexAttributes->getVertexFormatGLHandle();
        unsigned int vertexAttributeEnabledMask = 0;
        for ( unsigned int index = 0; index < RIX_GL_MAX_ATTRIBUTES; ++index )
        {
          //assert( ( vertexCache.m_vertexAttributeEnabledMask & gi->m_program->getActiveAttributeMask() ) == gi->m_program->getActiveAttributeMask()  );
          // TODO use vertex mask by shader! gi->m_program->getActiveAttributeMask()
          if ( formatHandle->m_format[index].m_enabled && ((1 << index) & gi->m_programPipeline->m_activeAttributeMask) )
          {
            glEnableVertexAttribArray( index );
            vertexAttributeEnabledMask |= 1 << index;
          }
          else
          {
            glDisableVertexAttribArray( index );
          }
        }

        VertexDataGLHandle vertexData = vertexAttributes->getVertexDataGLHandle();

        // TODO temporary
        size_t vertexFormatId = registerVertexFormat( vertexAttributes.get() );

        VertexFormat &format = m_vertexFormats[vertexFormatId];
        for ( unsigned int index = 0; index < format.m_numAttributes; ++index )
        {
          VertexAttribute &formatEntry = format.m_attributes[index];
          glVertexAttribFormatNV( formatEntry.index, formatEntry.numComponents, formatEntry.type, formatEntry.normalized, formatEntry.stride );
        }

        for ( unsigned int index = 0; index < RIX_GL_MAX_ATTRIBUTES; ++index )
        {
          dp::rix::gl::VertexFormatGL::Format &format = formatHandle->m_format[index];
          if ( format.m_enabled && ((1 << index) & gi->m_programPipeline->m_activeAttributeMask) )
          {
            dp::rix::gl::VertexFormatGL::Format &format = formatHandle->m_format[index];
            DP_ASSERT( vertexData->m_data[format.m_streamId].m_buffer );
            dp::gl::BufferSharedPtr const& buffer = vertexData->m_data[format.m_streamId].m_buffer->getBuffer();
            GLuint64EXT address = buffer->getAddress();
            size_t offset = vertexData->m_data[format.m_streamId].m_offset + format.m_offset;
            glBufferAddressRangeNV( GL_VERTEX_ATTRIB_ARRAY_ADDRESS_NV, index, address + offset, buffer->getSize() - offset );
          }
        }

        // update indices data
        IndicesGLSharedHandle const& indices = gi->getGeometry()->getIndices();
        if ( indices )
        {
          DP_ASSERT( indices->getBufferHandle() && indices->getBufferHandle()->getBuffer() );
          dp::gl::BufferSharedPtr const& buffer = indices->getBufferHandle()->getBuffer();
          GLuint indexBuffer = buffer->getGLId();

          GLuint64EXT address;
          glGetNamedBufferParameterui64vNV( indexBuffer, GL_BUFFER_GPU_ADDRESS_NV, &address );
          if ( !glIsNamedBufferResidentNV( indexBuffer ) )
          {
            glMakeNamedBufferResidentNV( indexBuffer, GL_READ_ONLY );
          }

          glBufferAddressRangeNV( GL_ELEMENT_ARRAY_ADDRESS_NV, 0, address, buffer->getSize() );
        }

        glBindVertexArray( 0 );

        geometryInstanceCacheEntry.m_vao = vertexCache->m_vao;
        geometryInstanceCacheEntry.m_drawCall.updateDrawCall( gi );
      }

      void VertexAttributeCache<VertexCacheBindlessVAO>::renderGeometryInstance( GeometryInstanceCacheEntry const& giCacheEntry )
      {
        //if ( m_currentVAO != giCacheEntry.m_vao ) // TODO will not help, different always. implement VAO cache?
        {
          glBindVertexArray( giCacheEntry.m_vao );
        }
        giCacheEntry.m_drawCall.draw( this );
      }

      template class RenderEngineGLImpl<VertexCacheBindlessVAO>;
      static bool initializedBindlessVAO = registerRenderEngine( "BindlessVAO", &renderEngineCreate<VertexCacheBindlessVAO> );

    } // namespace gl
  } // namespace rix
} // namespace dp
