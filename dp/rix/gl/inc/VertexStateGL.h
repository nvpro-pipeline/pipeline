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

#include <GL/glew.h>
#include "RendererGLConfig.h"
#include <vector>
#include "ProgramGL.h"


namespace dp
{
  namespace rix
  {
    namespace gl
    {

      struct VertexAttribute
      {
        GLuint    index;
        GLuint    streamId;
        GLuint    numComponents;
        GLenum    type;
        GLuint    normalized;
        GLuint    offset;
        GLuint    stride;
      };

      struct VertexFormat
      {
        unsigned int m_enabledMask;
        unsigned int m_numAttributes;
        unsigned int m_numStreams;
        unsigned int m_streamStrides[RIX_GL_MAX_ATTRIBUTES];
        unsigned int m_streamIds[RIX_GL_MAX_ATTRIBUTES];      // map from compact streamId to VertexData streamId
        VertexAttribute m_attributes[RIX_GL_MAX_ATTRIBUTES];
      };

      typedef size_t VertexFormatId;

      class VertexAttributesGL;
      typedef VertexAttributesGL* VertexAttributesGLHandle;

      class IndicesGL;
      typedef IndicesGL* IndicesGLHandle;

      class VertexStateGL
      {
      public:
        VertexStateGL();
        virtual ~VertexStateGL();

        VertexFormatId registerVertexFormat( VertexAttributesGLHandle handle, unsigned int attributeMask = ~0 );
        void setVertexFormatMask( unsigned int formatMask );
        void setPrimitiveRestartIndex( unsigned int primitiveRestartIndex );
        void setArrayBuffer( GLuint arrayBuffer );
        void setElementBuffer( GLuint elementBuffer );

        unsigned int getNumberOfInstances() const;

        /** \brief Choose the given vertex format
            \param return true if format has changed, false otherwise.
        **/
        bool setVertexFormat( VertexFormatId vertexFormat );
      protected:
        unsigned int m_currentVertexFormatMask;
        unsigned int m_currentPrimitiveRestartIndex;
        unsigned int m_currentArrayBuffer;
        unsigned int m_currentElementBuffer;

        std::vector<VertexFormat> m_vertexFormats;

        size_t m_currentVertexFormatId; // for bindless only

        VertexAttributesGLHandle m_currentVA;
        IndicesGLHandle          m_currentIS;

        unsigned int             m_numInstances;
      };

      inline bool VertexStateGL::setVertexFormat( VertexFormatId vertexFormatId )
      {
        if ( m_currentVertexFormatId != vertexFormatId )
        {
          m_currentVertexFormatId = vertexFormatId;
          // If the current vertex format has a different set of attributes enabled than the new mask and
          // the next draw call uses the same VA, it will not rebind those attributes since the VA hasn't changed.
          // Force update of the bindings by setting m_currentVA to nullptr.
          m_currentVA = nullptr;
          return true;
        }
        return false;
      }

      inline void VertexStateGL::setPrimitiveRestartIndex( unsigned int primitiveRestartIndex )
      {
        if ( m_currentPrimitiveRestartIndex != primitiveRestartIndex )
        {
          glPrimitiveRestartIndex( primitiveRestartIndex );
          m_currentPrimitiveRestartIndex = primitiveRestartIndex;
        }
      }

      inline unsigned int VertexStateGL::getNumberOfInstances() const
      {
        return m_numInstances;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
