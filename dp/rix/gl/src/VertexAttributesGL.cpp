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


#include <dp/rix/gl/inc/VertexAttributesGL.h>
#include <dp/rix/gl/inc/BufferGL.h>

#include <vector>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      /** \brief Copy elements from a strided source to a strided destination
       *  \param dst Destination for data
       *  \param dstOffset Offset in bytes relative to dst of the first element to write
       *  \param dstStride Stride between two elements in destination data
       *  \param src Source of data
       *  \param srcOffset Offset in bytes relative to src of the first element to read
       *  \param srcStride Stride between two elements in source data
       *  \param elementSize Size of a single data element to copy
       *  \param elementCount Number of elements to copy
       *  \remarks The result is undefined if src and dst overlap.
       **/
      void stridedMemcpy( void *dst, size_t dstOffset, size_t dstStride, const void *src, size_t srcOffset, size_t srcStride, size_t elementSize, size_t elementCount );

      VertexAttributesGL::Event::Event( VertexAttributesGL* vertexAttributes)
        : m_vertexAttributes( vertexAttributes )
      {
      }

      VertexAttributesGL::~VertexAttributesGL()
      {
        detachObservers();

        handleReset( m_vertexData );
        handleReset( m_vertexFormat );
      }

      void VertexAttributesGL::setVertexDataGLHandle( VertexDataGLHandle vertexData )
      {
        if ( vertexData != m_vertexData )
        {
          detachObservers( );

          // attach observers
          for ( unsigned int index = 0; index < RIX_GL_MAX_ATTRIBUTES; ++index )
          {
            if ( vertexData->m_data[index].m_buffer )
            {
              vertexData->m_data[index].m_buffer->attach( this );
            }
          }
          handleAssign(m_vertexData, vertexData );
          notify( Event( this ) );
        }
      }

      void VertexAttributesGL::setVertexFormatGLHandle( VertexFormatGLHandle vertexFormat )
      {
        if ( m_vertexFormat != vertexFormat )
        {
          handleAssign(m_vertexFormat, vertexFormat);
          notify( Event( this ) );
        }
      }

      void VertexAttributesGL::onNotify( const dp::util::Event& /*event*/, dp::util::Payload* /*payload*/ )
      {
        // currently only buffers are being observed and it's not of interest which buffer has changed
        notify( Event( this ) );
      }

      void VertexAttributesGL::onDestroyed( const dp::util::Subject& /*subject*/, dp::util::Payload* /*payload*/ )
      {
        DP_ASSERT( !"need to detach from something?" );
      }

      void VertexAttributesGL::detachObservers()
      {
        if ( m_vertexData )
        {
          for ( unsigned int index = 0; index < RIX_GL_MAX_ATTRIBUTES; ++index )
          {
            if ( m_vertexData->m_data[index].m_buffer )
            {
              m_vertexData->m_data[index].m_buffer->detach( this );
            }
          }
        }
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
