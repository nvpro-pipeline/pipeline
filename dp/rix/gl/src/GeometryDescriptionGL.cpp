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


#include <dp/rix/gl/inc/GeometryDescriptionGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

      void GeometryDescriptionGL::setPrimitiveType( GeometryPrimitiveType primitiveType )
      {
        if ( m_primitiveType != primitiveType )
        {
          m_primitiveType = primitiveType;
          notify( dp::util::Event() );
        }
      }

      void GeometryDescriptionGL::setPrimitiveRestartIndex( unsigned int primitiveRestartIndex )
      {
        if ( m_primitiveRestartIndex != primitiveRestartIndex )
        {
          m_primitiveRestartIndex = primitiveRestartIndex;
          notify( dp::util::Event() );
        }
      }

      void GeometryDescriptionGL::setBaseVertex( unsigned int baseVertex )
      {
        if ( m_baseVertex != baseVertex) {
          m_baseVertex = baseVertex;
          notify( dp::util::Event() );
        }
      }

      void GeometryDescriptionGL::setIndexRange( unsigned int first, unsigned int count )
      {
        if ( first != m_indexFirst || count != m_indexCount ) {
          m_indexFirst = first;
          m_indexCount = count;
          notify( dp::util::Event() );
        }
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
