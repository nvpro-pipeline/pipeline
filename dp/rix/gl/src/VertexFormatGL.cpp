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


#include "VertexFormatGL.h"
#include <iostream>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

      bool VertexFormatGL::Format::operator==( const VertexFormatGL::Format& rhs ) const
      { 
        if ( m_enabled && rhs.m_enabled )
        {
          return   ( m_dataType == rhs.m_dataType )
                && ( m_streamId == rhs.m_streamId )
                && ( m_numComponents == rhs.m_numComponents )
                && ( m_normalized == rhs.m_normalized )
                && ( m_offset == rhs.m_offset )
                && ( m_stride == rhs.m_stride );
        }
        else
        {
          return (m_enabled == rhs.m_enabled);
        }
      }

      VertexFormatGL::VertexFormatGL( const VertexFormatDescription &vertexFormatDescription )
      {
        // Format is disabled by default. Update information for passed in format indices
        for (size_t index = 0;index < vertexFormatDescription.m_numVertexFormatInfos; ++index )
        {
          const VertexFormatInfo& info = vertexFormatDescription.m_vertexFormatInfos[index];
          Format& format = m_format[info.m_attributeIndex];
          format.m_enabled = true;
          format.m_streamId = info.m_streamId;
          format.m_dataType = info.m_dataType;
          format.m_numComponents = info.m_numComponents;
          format.m_offset = (unsigned int)(info.m_offset);
          format.m_stride = (unsigned int)(info.m_stride);
          format.m_normalized = info.m_normalized;
        }
      }

      VertexFormatGLHandle VertexFormatGL::create( const VertexFormatDescription &vertexFormatDescription )
      {
        static std::vector<VertexFormatGLSharedHandle> vertexFormats;

        // TODO this can be implemented more efficient
        VertexFormatGLHandle newFormat = new VertexFormatGL( vertexFormatDescription );
        for ( size_t idx = 0; idx < vertexFormats.size(); ++idx )
        {
          VertexFormatGLHandle format = vertexFormats[idx].get();
          if ( *format == *newFormat )
          {
            delete newFormat;
            return format;
          }
        }
        vertexFormats.push_back(newFormat);
        return newFormat;
      }

      size_t VertexFormatGL::getVertexSize( unsigned int attributeIndex )
      {
        size_t size = 0;
        Format &format = m_format[attributeIndex];
        if ( format.m_enabled )
        {
          size += dp::getSizeOf( format.m_dataType ) * format.m_numComponents;
        }

        return size;
      }

      bool VertexFormatGL::operator==( const VertexFormatGL& rhs ) const
      {
        bool equal = true;
        for ( size_t index = 0;index < RIX_GL_MAX_ATTRIBUTES && equal; ++index )
        {
          equal = (m_format[index] == rhs.m_format[index]);
        }
        return equal;
      }


    } // namespace gl
  } // namespace rix
} // namespace dp
