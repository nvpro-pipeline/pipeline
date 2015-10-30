// Copyright (c) 2011-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <cstring>

#include <dp/rix/gl/inc/VertexStateGL.h>
#include <dp/rix/gl/inc/VertexFormatGL.h>
#include <dp/rix/gl/inc/VertexAttributesGL.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      VertexStateGL::VertexStateGL()
        : m_currentVertexFormatMask( 0 )
        , m_currentPrimitiveRestartIndex( 0 )
        , m_currentArrayBuffer( 0 )
        , m_currentElementBuffer( 0 )
        , m_currentVertexFormatId( ~0 )
        , m_currentVA( nullptr )
        , m_currentIS( nullptr )
        , m_numInstances( 0 )
      {

      }

      VertexStateGL::~VertexStateGL()
      {

      }

      void VertexStateGL::setArrayBuffer( GLuint arrayBuffer )
      {
        glBindBuffer( GL_ARRAY_BUFFER, arrayBuffer );
        m_currentArrayBuffer = arrayBuffer;
      }

      void VertexStateGL::setElementBuffer( GLuint elementBuffer )
      {
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, elementBuffer );
        m_currentElementBuffer = elementBuffer;
      }

      void VertexStateGL::setVertexFormatMask( unsigned int formatMask )
      {
        unsigned int delta = m_currentVertexFormatMask ^ formatMask;
        if ( delta )
        {
          m_currentVertexFormatMask = formatMask;
          for ( unsigned int i = 0; i < RIX_GL_MAX_ATTRIBUTES; ++i )
          {
            unsigned int mask = 1 << i;
            if ( delta & mask )
            {
              if ( formatMask & mask )
              {
                //enableVertexAttribute( i );
                glEnableVertexAttribArray( i );
              }
              else
              {
                //disableVertexAttribute( i );
                glDisableVertexAttribArray( i );
              }
            }
          }
        }
      }

      size_t VertexStateGL::registerVertexFormat( VertexAttributesGLHandle handle, unsigned int attributeMask )
      {
        VertexFormat newFormat;
        memset( &newFormat, 0, sizeof( newFormat ) );

        int8_t streamMap[RIX_GL_MAX_ATTRIBUTES];
        memset( streamMap, -1, RIX_GL_MAX_ATTRIBUTES );
        int numStreams = 0;

        for ( size_t attribIndex = 0; attribIndex < RIX_GL_MAX_ATTRIBUTES; ++attribIndex )
        {
          VertexFormatGL::Format &format = handle->getVertexFormatGLHandle()->m_format[attribIndex];
          if ( format.m_enabled && ((1 << attribIndex) & attributeMask) )
          {
            newFormat.m_enabledMask |= (1 << attribIndex);
            VertexAttribute &newAttribute = newFormat.m_attributes[newFormat.m_numAttributes];

            if ( streamMap[format.m_streamId] == -1 )
            {
              newFormat.m_streamIds[numStreams] = format.m_streamId;
              streamMap[format.m_streamId] = numStreams++;
            }

            newAttribute.index = (GLuint)attribIndex;
            newAttribute.streamId = streamMap[format.m_streamId];
            newAttribute.numComponents = format.m_numComponents;
            newAttribute.type = getGLDataType(format.m_dataType);
            newAttribute.normalized = format.m_normalized;
            newAttribute.offset = format.m_offset;
            newAttribute.stride = format.m_stride;

            ++newFormat.m_numAttributes;

            if ( newFormat.m_streamStrides[ newAttribute.streamId ] == 0 )
            {
              newFormat.m_streamStrides[ newAttribute.streamId ] = newAttribute.stride;
            }
            else
            {
              assert( newFormat.m_streamStrides[ newAttribute.streamId ] == newAttribute.stride );
            }
          }
        }
        newFormat.m_numStreams = numStreams;

        // look if format already exists
        size_t idx = 0;
        while (idx < m_vertexFormats.size() )
        {
          if ( memcmp( &m_vertexFormats[idx], &newFormat, sizeof( newFormat ) ) == 0 )
          {
            return idx;
          }
          ++idx;
        }
        m_vertexFormats.push_back( newFormat );
        return idx;
      }


    } // namespace gl
  } // namespace rix
} // namespace dp
