// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <test/rix/core/helpers/GeometryHelper.h>

using namespace std;
using namespace dp::math;

namespace dp
{
  namespace rix
  {
    using namespace core;

    namespace util
    {

      GeometrySharedHandle generateGeometry( dp::rix::util::GeometryDataSharedPtr& meshIn, dp::rix::core::Renderer* m_rix )
      {
        AttributeMask attrMask;

        for( map<dp::rix::util::AttributeID, dp::rix::util::AttributeData>::iterator it = meshIn->m_attributes.begin(); it != meshIn->m_attributes.end(); ++it )
        {
          attrMask |= it->first;
        }

        map<dp::rix::util::AttributeID, BufferSharedHandle> vbuffers;
        vector<VertexFormatInfo> vfis;

        BufferSharedHandle ibuffer;
        if( !meshIn->m_indices.empty() )
        {
          ibuffer = m_rix->bufferCreate();
          size_t bufferSize = meshIn->m_indices.size() * sizeof(unsigned int);
          m_rix->bufferSetSize( ibuffer, bufferSize );
          m_rix->bufferUpdateData( ibuffer, 0, &meshIn->m_indices[0], bufferSize );
        }

        VertexDataSharedHandle vertexData = m_rix->vertexDataCreate();
        for(unsigned int i = 0; i < NUM_ATTRIBS; i++)
        {
          dp::rix::util::AttributeID curAttr = (dp::rix::util::AttributeID)(1 << i);
          if ( attrMask & curAttr )
          {
            vbuffers[curAttr] = m_rix->bufferCreate();
            m_rix->bufferSetSize( vbuffers[curAttr], meshIn->m_attributes[curAttr].m_data.size() * sizeof(float) );
            m_rix->bufferUpdateData( vbuffers[curAttr], 0, &meshIn->m_attributes[curAttr].m_data[0], meshIn->m_attributes[curAttr].m_data.size() * sizeof(float) );

            uint8_t streamId = dp::checked_cast<uint8_t>( vfis.size() );

            vfis.push_back( VertexFormatInfo( i, dp::DataType::FLOAT_32, meshIn->m_attributes[curAttr].m_dimensionality, false,
                                              streamId, 0, meshIn->m_attributes[curAttr].m_dimensionality * sizeof(float)) );

            m_rix->vertexDataSet( vertexData, streamId, vbuffers[curAttr], 0, meshIn->m_attributes[curAttr].m_data.size() / meshIn->m_attributes[curAttr].m_dimensionality );
          }
        }

        VertexFormatDescription vertexFormatDescription( &vfis[0], vfis.size() );
        VertexFormatSharedHandle vertexFormat = m_rix->vertexFormatCreate( vertexFormatDescription );

        VertexAttributesSharedHandle vertexAttributes = m_rix->vertexAttributesCreate();
        m_rix->vertexAttributesSet( vertexAttributes, vertexData, vertexFormat );

        IndicesSharedHandle indices = 0;
        if( !meshIn->m_indices.empty() )
        {
          indices = m_rix->indicesCreate();
          m_rix->indicesSetData( indices, dp::DataType::UNSIGNED_INT_32, ibuffer, 0, meshIn->m_indices.size() );
        }

        GeometryDescriptionSharedHandle geometryDescription = m_rix->geometryDescriptionCreate();
        m_rix->geometryDescriptionSet( geometryDescription, meshIn->m_gpt );

        GeometrySharedHandle geometry = m_rix->geometryCreate();
        m_rix->geometrySetData( geometry, geometryDescription, vertexAttributes, indices );

        return geometry;

      }

    } // namespace util
  } // namespace rix
} // namespace dp
