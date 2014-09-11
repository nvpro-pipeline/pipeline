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
#include "RenderEngineGLVBOVAO.h"
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

      void VertexAttributeCache<VertexCacheVBOVAO>::beginFrame()
      {
      }

      void VertexAttributeCache<VertexCacheVBOVAO>::endFrame()
      {
        glBindVertexArray( 0 );
      }

      void VertexAttributeCache<VertexCacheVBOVAO>::updateGeometryInstanceCacheEntry( GeometryInstanceGLHandle gi, GeometryInstanceCacheEntry& geometryInstanceCacheEntry, AttributeCacheEntry * /*attributeCacheEntry*/ )
      {
        VertexDataCache* vertexCache = new VertexDataCache();
        gi->m_vertexCache = vertexCache;

        if ( !vertexCache->m_vao )
        {
          glGenVertexArrays( 1, &vertexCache->m_vao );
        }

        glBindVertexArray( vertexCache->m_vao );

        VertexAttributesGLSharedHandle const& vertexAttributes = gi->getGeometry()->getVertexAttributes();
        VertexFormatGLHandle formatHandle = vertexAttributes->getVertexFormatGLHandle();
        unsigned int vertexAttributeEnabledMask = 0;
        for ( unsigned int index = 0; index < RIX_GL_MAX_ATTRIBUTES; ++index )
        {
          dp::rix::gl::VertexFormatGL::Format &format = formatHandle->m_format[index];
          if ( format.m_enabled && ((1 << index) & gi->m_programPipeline->m_activeAttributeMask) )
          {
            const VertexDataGL::Data& data = vertexAttributes->getVertexDataGLHandle()->m_data[format.m_streamId];
            dp::gl::bind( GL_ARRAY_BUFFER, data.m_buffer->getBuffer() );
            glEnableVertexAttribArray( index );
            glVertexAttribPointer( index, format.m_numComponents, getGLDataType(format.m_dataType), format.m_numComponents, format.m_stride, (GLvoid*)(data.m_offset + format.m_offset) );
            vertexAttributeEnabledMask |= 1 << index;
          }
          else
          {
            glDisableVertexAttribArray( index );
          }
        }

        // update indices data
        IndicesGLSharedHandle const& indices = gi->getGeometry()->getIndices();
        if ( indices )
        {
          dp::gl::bind( GL_ELEMENT_ARRAY_BUFFER, indices->getBufferHandle()->getBuffer() );
        }

        glBindVertexArray( 0 );

        geometryInstanceCacheEntry.m_vao = vertexCache->m_vao;
        geometryInstanceCacheEntry.m_drawCall.updateDrawCall(gi);
      }

      void VertexAttributeCache<VertexCacheVBOVAO>::renderGeometryInstance( GeometryInstanceCacheEntry const& giCacheEntry )
      {
        //if ( m_currentVAO != giCacheEntry.m_vao ) // TODO will not help, different always. implement VAO cache?
        {
          glBindVertexArray( giCacheEntry.m_vao );
        }
        giCacheEntry.m_drawCall.draw( this );
      }


      template class RenderEngineGLImpl<VertexCacheVBOVAO>;
      static bool initializedBindlessVAO = registerRenderEngine( "VBOVAO", &renderEngineCreate<VertexCacheVBOVAO> );

    } // namespace gl
  } // namespace rix
} // namespace dp
