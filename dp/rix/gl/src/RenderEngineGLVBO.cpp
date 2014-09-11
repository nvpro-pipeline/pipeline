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
#include "RenderEngineGLVBO.h"
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

      void VertexAttributeCache<VertexCacheVBO>::beginFrame()
      {
        m_currentIS = nullptr;
        m_currentVA = nullptr;
      }

      void VertexAttributeCache<VertexCacheVBO>::endFrame()
      {
      }

      void VertexAttributeCache<VertexCacheVBO>::updateGeometryInstanceCacheEntry( GeometryInstanceGLHandle gi, GeometryInstanceCacheEntry& geometryInstanceCacheEntry, AttributeCacheEntry* attributeCacheEntry )
      {
        VertexAttributesGLSharedHandle const& vertexAttributes = gi->getGeometry()->getVertexAttributes();

        VertexDataGLHandle vertexData = vertexAttributes->getVertexDataGLHandle();
        geometryInstanceCacheEntry.m_formatId = registerVertexFormat( vertexAttributes.get(), gi->m_programPipeline->m_activeAttributeMask );
        const VertexFormat &vertexFormat = m_vertexFormats[geometryInstanceCacheEntry.m_formatId];

        for ( unsigned int index = 0; index < vertexFormat.m_numStreams; ++index )
        {
          unsigned int streamId = vertexFormat.m_streamIds[index];
          AttributeCacheEntry& cacheEntry = attributeCacheEntry[index];
          BufferGL *buffer = vertexData->m_data[streamId].m_buffer;

          if ( buffer )
          {
            DP_ASSERT( buffer->getBuffer() );
            cacheEntry.vbo  = buffer->getBuffer()->getGLId();
            cacheEntry.offset = reinterpret_cast<GLvoid *>(vertexData->m_data[streamId].m_offset);
          }
          else
          {
            cacheEntry.vbo = 0;
            cacheEntry.offset = 0;
          }
        }

        // update indices data
        IndicesGLSharedHandle const& indices = gi->getGeometry()->getIndices();
        if ( indices && indices->getBufferHandle() )
        {
          DP_ASSERT( indices->getBufferHandle()->getBuffer() );
          geometryInstanceCacheEntry.m_attributeCacheIndices.vbo = indices->getBufferHandle()->getBuffer()->getGLId();
          geometryInstanceCacheEntry.m_attributeCacheIndices.offset = 0;
        }
        else
        {
          geometryInstanceCacheEntry.m_attributeCacheIndices.vbo = 0;
          geometryInstanceCacheEntry.m_attributeCacheIndices.offset = 0;
        }
        geometryInstanceCacheEntry.m_drawCall.updateDrawCall( gi );
        geometryInstanceCacheEntry.m_attributeCacheEntry = attributeCacheEntry;
        geometryInstanceCacheEntry.m_vertexAttributesHandle = vertexAttributes.get();
        geometryInstanceCacheEntry.m_indicesHandle = indices.get();
      }

      void VertexAttributeCache<VertexCacheVBO>::renderGeometryInstance( GeometryInstanceCacheEntry const& giCacheEntry )
      {
        // update format/attributes/indices
        if ( setVertexFormat( giCacheEntry.m_formatId) )
        {
          m_currentVertexFormat = &m_vertexFormats[giCacheEntry.m_formatId];
          setVertexFormatMask( m_currentVertexFormat->m_enabledMask );
        }
        if ( m_currentVA != giCacheEntry.m_vertexAttributesHandle )
        {
          m_currentVA = giCacheEntry.m_vertexAttributesHandle;
          for ( unsigned int index = 0;index < m_currentVertexFormat->m_numAttributes;++index )
          {
            const VertexAttribute& attribute = m_currentVertexFormat->m_attributes[index];
            const AttributeCacheEntry& entry = giCacheEntry.m_attributeCacheEntry[attribute.streamId];
            if ( m_currentArrayBuffer != entry.vbo )
            {
              setArrayBuffer( entry.vbo );
            }
            glVertexAttribPointer( attribute.index, attribute.numComponents, attribute.type, attribute.normalized, attribute.stride, (GLvoid*)((size_t)entry.offset + attribute.offset));
          }
        }
        if ( giCacheEntry.m_indicesHandle )
        {
          if ( m_currentIS != giCacheEntry.m_indicesHandle )
          {
            m_currentIS = giCacheEntry.m_indicesHandle;
            setElementBuffer( giCacheEntry.m_attributeCacheIndices.vbo );

          }
        }

        giCacheEntry.m_drawCall.draw( this );
      }


      template class RenderEngineGLImpl<VertexCacheVBO>;
      static bool initializedVBO = registerRenderEngine( "VBO", &renderEngineCreate<VertexCacheVBO> );

    } // namespace gl
  } // namespace rix
} // namespace dp
