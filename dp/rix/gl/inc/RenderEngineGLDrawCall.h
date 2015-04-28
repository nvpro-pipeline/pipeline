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


#pragma once

#include <GeometryGL.h>
#include <GeometryInstanceGL.h>
#include <GeometryDescriptionGL.h>
#include <IndicesGL.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>
#include <dp/rix/gl/inc/VertexAttributesGL.h>


namespace dp
{
  namespace rix
  {
    namespace gl
    {
      class RenderEngineGLDrawCall
      {
      public:
        RenderEngineGLDrawCall()
        {
          m_drawCall = &RenderEngineGLDrawCall::drawDummy;
        }

        void updateDrawCall( GeometryInstanceGLHandle gi )
        {
          IndicesGLHandle indices = gi->getGeometry()->getIndices().get();
          GeometryDescriptionGLHandle geometryDescription = gi->getGeometry()->getGeometryDescription().get();

          if ( indices )
          {
            m_drawCall = &RenderEngineGLDrawCall::drawElements;
            m_drawElements.indexPrimitiveRestart = geometryDescription->getPrimitiveRestartIndex();
            m_drawElements.mode = getGLPrimitiveType( geometryDescription->getPrimitiveType() );
            m_drawElements.count = geometryDescription->getIndexCount();
            m_drawElements.type = getGLDataType( indices->getDataType() );
            m_drawElements.baseVertex = geometryDescription->getBaseVertex();
            m_drawElements.indices = (void*)(geometryDescription->getIndexFirst() * getSizeOf(indices->getDataType()));
          }
          else
          {
            m_drawCall = &RenderEngineGLDrawCall::drawArrays;
            m_drawArrays.mode = getGLPrimitiveType( geometryDescription->getPrimitiveType() );
            m_drawArrays.first = geometryDescription->getBaseVertex();
            m_drawArrays.count = static_cast<unsigned int>( gi->getGeometry()->getVertexAttributes()->getVertexDataGLHandle()->m_numberOfVertices );
          }
        }

        void draw( VertexStateGL* vertexStateGL ) const { (this->*m_drawCall)( vertexStateGL ); }

      protected:
        typedef void (RenderEngineGLDrawCall::*DrawCall)(  VertexStateGL *vertexStateGL ) const;
        DrawCall m_drawCall;

        void drawDummy( VertexStateGL * /*vertexStateGL*/ ) const
        {
          assert( 0 && "drawDummy");
        }

        void drawElements( VertexStateGL *vertexStateGL ) const
        {
          vertexStateGL->setPrimitiveRestartIndex( m_drawElements.indexPrimitiveRestart );
          glDrawElementsInstancedBaseVertex( m_drawElements.mode, m_drawElements.count, m_drawElements.type, m_drawElements.indices, vertexStateGL->getNumberOfInstances(), m_drawElements.baseVertex );
        }

        void drawArrays( VertexStateGL *vertexStateGL ) const
        {
          glDrawArraysInstanced( m_drawArrays.mode, m_drawArrays.first, m_drawArrays.count, vertexStateGL->getNumberOfInstances() );
        }

        struct DrawElements
        {
          unsigned int  indexPrimitiveRestart;
          GLenum        mode;
          GLsizei       count;
          GLenum        type;
          unsigned int  baseVertex;
          const GLvoid* indices;
        };

        struct DrawArrays
        {
          GLenum  mode;
          GLint   first;
          GLsizei count;
        };

        union
        {
          DrawElements m_drawElements;
          DrawArrays   m_drawArrays;
        };

      };

    } // namespace gl
  } // namespace rix
} // namespace dp
