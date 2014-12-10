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


#pragma once

#include "VertexCacheGL.h"
#include <dp/rix/gl/RiXGL.h>
#include "RendererGLConfig.h"
#include "VertexStateGL.h"
#include "RenderEngineGLDrawCall.h"
#include "GeometryInstanceGL.h"
#include "ProgramPipelineGL.h"
#include <GL/glew.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      class IndicesGL;
      class VertexAttributesGL;
      template <typename VertexCache> class RenderEngineGLImpl;

      struct VertexAttributeStateBindless
      {
        GLuint64EXT m_address;
        GLsizeiptr  m_length;
      };

      template <> struct VertexAttributeCache<VertexCacheBindless> : public VertexStateGL
      {
        struct AttributeCacheEntry
        {
          GLuint64EXT m_address;
          GLsizeiptr  m_length;
        };

        struct GeometryInstanceCacheEntry : public GeometryInstanceGL::Cache
        {
          VertexFormatId           m_formatId;
          VertexAttributesGLHandle m_vertexAttributesHandle;
          AttributeCacheEntry*     m_attributeCacheEntry;
          IndicesGLHandle          m_indicesHandle;
          AttributeCacheEntry      m_attributeCacheIndices;

          RenderEngineGLDrawCall   m_drawCall;
        };


        void beginFrame();
        void endFrame();

        void updateGeometryInstanceCacheEntry( GeometryInstanceGLHandle gi, GeometryInstanceCacheEntry& geometryInstanceCacheEntry, AttributeCacheEntry *attributeCacheEntry );
        void renderGeometryInstance( GeometryInstanceCacheEntry const& giCacheEntry );

        VertexFormat *m_currentVertexFormat;

        // Format information required by the renderGeometryInstance function. This one is as small as possible to reduce the number of accessed cache-lines
        struct AttributeLoopInfo
        {
          unsigned short index;
          unsigned short streamId;
          unsigned int   offset;
        };
        // iterator to element after the last used attribute. avoids indexing within the update-loop.
        // The iterator is before the attribute array because it's being read before the first attributeLoopInfo.
        AttributeLoopInfo* m_endAttribute;  
        AttributeLoopInfo attributeLoopInfo[16];
      };

      inline void VertexAttributeCache<VertexCacheBindless>::updateGeometryInstanceCacheEntry( GeometryInstanceGLHandle gi, GeometryInstanceCacheEntry& geometryInstanceCacheEntry, AttributeCacheEntry *attributeCacheEntry )
      {
        VertexAttributesGLSharedHandle const& vertexAttributes = gi->getGeometry()->getVertexAttributes();

        geometryInstanceCacheEntry.m_vertexAttributesHandle = vertexAttributes.get();

        VertexDataGLHandle vertexData = vertexAttributes->getVertexDataGLHandle();
        geometryInstanceCacheEntry.m_formatId = registerVertexFormat( vertexAttributes.get(), gi->m_programPipeline->m_activeAttributeMask );
        const VertexFormat &vertexFormat = m_vertexFormats[geometryInstanceCacheEntry.m_formatId];

        for ( unsigned int index = 0; index < vertexFormat.m_numStreams; ++index )
        {
          unsigned int streamId = vertexFormat.m_streamIds[index];
          AttributeCacheEntry& cacheEntry = attributeCacheEntry[index];
          BufferGL *buffer = vertexAttributes->getVertexDataGLHandle()->m_data[streamId].m_buffer;
          if ( buffer )
          {
            DP_ASSERT( buffer->getBuffer() );
            dp::gl::BufferSharedPtr const& b = buffer->getBuffer();
            cacheEntry.m_address = b->getAddress() + vertexData->m_data[streamId].m_offset;
            cacheEntry.m_length =  b->getSize() - vertexData->m_data[streamId].m_offset;
          }
          else
          {
            cacheEntry.m_address = 0;
            cacheEntry.m_length = 0;
          }
        }

        geometryInstanceCacheEntry.m_attributeCacheEntry = attributeCacheEntry;

        // update indices data
        IndicesGLSharedHandle const& indices = gi->getGeometry()->getIndices();
        geometryInstanceCacheEntry.m_indicesHandle = indices.get();
        if ( indices )
        {
          DP_ASSERT( indices->getBufferHandle() && indices->getBufferHandle()->getBuffer() );
          dp::gl::BufferSharedPtr const& b = indices->getBufferHandle()->getBuffer();
          geometryInstanceCacheEntry.m_attributeCacheIndices.m_address = b->getAddress();
          geometryInstanceCacheEntry.m_attributeCacheIndices.m_length = b->getSize();
        }
        else
        {
          geometryInstanceCacheEntry.m_attributeCacheIndices.m_address = 0;
          geometryInstanceCacheEntry.m_attributeCacheIndices.m_length = 0;
        }

        // update draw call
        geometryInstanceCacheEntry.m_drawCall.updateDrawCall( gi );
      }

      inline void VertexAttributeCache<VertexCacheBindless>::beginFrame()
      {
        // bindless
        glEnableClientState(GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV);
        glEnableClientState(GL_ELEMENT_ARRAY_UNIFIED_NV );

        m_currentIS = nullptr;
        m_currentVA = nullptr;
      }

      inline void VertexAttributeCache<VertexCacheBindless>::endFrame()
      {
        // bindless
        glDisableClientState(GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV);
        glDisableClientState(GL_ELEMENT_ARRAY_UNIFIED_NV );
      }

      inline void VertexAttributeCache<VertexCacheBindless>::renderGeometryInstance( GeometryInstanceCacheEntry const& giCacheEntry )
      {
        // update format/attributes/indices
        if ( setVertexFormat( giCacheEntry.m_formatId) )
        {
          m_currentVertexFormat = &m_vertexFormats[giCacheEntry.m_formatId];
          m_endAttribute = attributeLoopInfo + m_currentVertexFormat->m_numAttributes;
          setVertexFormatMask( m_currentVertexFormat->m_enabledMask );
          for ( unsigned int index = 0; index < m_currentVertexFormat->m_numAttributes; ++index )
          {
            VertexAttribute &entry = m_currentVertexFormat->m_attributes[index];
            glVertexAttribFormatNV( entry.index, entry.numComponents, entry.type, entry.normalized, entry.stride );

            // update local data to avoid cache trashing with the m_currentVertexFormat pointer
            attributeLoopInfo[index].index = entry.index;
            attributeLoopInfo[index].streamId = entry.streamId;
            attributeLoopInfo[index].offset = entry.offset;
          }
        }
        if ( m_currentVA != giCacheEntry.m_vertexAttributesHandle )
        {
          m_currentVA = giCacheEntry.m_vertexAttributesHandle;

          AttributeCacheEntry const* attributeCacheEntry = giCacheEntry.m_attributeCacheEntry;

          AttributeLoopInfo const* attribute = attributeLoopInfo;
          while ( attribute != m_endAttribute )
          {
            AttributeCacheEntry const& entry = attributeCacheEntry[attribute->streamId];
            glBufferAddressRangeNV( GL_VERTEX_ATTRIB_ARRAY_ADDRESS_NV, attribute->index, entry.m_address + attribute->offset, entry.m_length - attribute->offset );
            ++attribute;
          }
        }
        if ( giCacheEntry.m_indicesHandle )
        {
          if ( m_currentIS != giCacheEntry.m_indicesHandle )
          {
            m_currentIS = giCacheEntry.m_indicesHandle;
            AttributeCacheEntry const& entry = giCacheEntry.m_attributeCacheIndices;
            glBufferAddressRangeNV( GL_ELEMENT_ARRAY_ADDRESS_NV, 0, entry.m_address, entry.m_length );
          }
        }

        giCacheEntry.m_drawCall.draw( this );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
