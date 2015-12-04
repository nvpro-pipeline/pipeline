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

#include <dp/rix/gl/RiXGL.h>

#include <dp/util/Observer.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      class GeometryDescriptionGL : public dp::rix::core::GeometryDescription, public dp::util::Subject
      {
      public:
        GeometryDescriptionGL()
          : m_primitiveType( GeometryPrimitiveType::TRIANGLES )
          , m_primitiveRestartIndex( ~0 )
          , m_baseVertex( 0 )
        {
        }

        ~GeometryDescriptionGL()
        {
        }

        void setPrimitiveType( GeometryPrimitiveType primitiveType );
        GeometryPrimitiveType getPrimitiveType() const { return m_primitiveType; }

        void setPrimitiveRestartIndex( unsigned int primitiveRestartIndex );
        unsigned int getPrimitiveRestartIndex() const { return m_primitiveRestartIndex; }

        void setBaseVertex( unsigned int baseVertex );
        unsigned int getBaseVertex() { return m_baseVertex; }

        void setIndexRange( unsigned int first, unsigned int count );
        unsigned int getIndexFirst() { return m_indexFirst; }
        unsigned int getIndexCount() { return m_indexCount; }

        // TODO implement notify

      protected:
        GeometryPrimitiveType m_primitiveType;
        unsigned int m_primitiveRestartIndex;
        unsigned int m_baseVertex;
        unsigned int m_indexFirst;
        unsigned int m_indexCount;
      };
    } // namespace gl
  } // namespace rix
} // namespace dp
