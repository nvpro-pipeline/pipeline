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


#include "GeometryGL.h"

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      GeometryGL::~GeometryGL()
      {
        if ( m_vertexAttributes )
        {
          m_vertexAttributes->detach( this );
        }

        if ( m_geometryDescription )
        {
          m_geometryDescription->detach( this );
        }

        if ( m_indices )
        {
          m_indices->detach( this );
        }
      }

      void GeometryGL::upload()
      {
      }

      void GeometryGL::onNotify( const dp::util::Event& event, dp::util::Payload* /*payload*/ )
      {
        notify( event );
      }

      void GeometryGL::onDestroyed( const dp::util::Subject& /*subject*/, dp::util::Payload* /*payload*/ )
      {
        DP_ASSERT( !"need to detach from something?" );
      }

      void GeometryGL::setVertexAttributes( VertexAttributesGLSharedHandle const & vertexAttributes )
      {
        if ( m_vertexAttributes )
        {
          m_vertexAttributes->detach( this );
        }

        m_vertexAttributes =  vertexAttributes;

        if ( m_vertexAttributes )
        {
          m_vertexAttributes->attach( this );
        }

        notify( dp::util::Event() );
      }

      void GeometryGL::setIndices( IndicesGLSharedHandle const & indices )
      {
        if ( m_indices )
        {
          m_indices->detach( this );
        }

        m_indices =indices;

        if ( m_indices )
        {
          m_indices->attach( this );
        }

        notify( dp::util::Event() );
      }

      void GeometryGL::setGeometryDescription( GeometryDescriptionGLSharedHandle const & geometryDescription )
      {
        if ( m_geometryDescription )
        {
          m_geometryDescription->detach( this );
        }

        m_geometryDescription = geometryDescription;

        if ( m_geometryDescription )
        {
          m_geometryDescription->attach( this );
        }

        notify( dp::util::Event() );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
