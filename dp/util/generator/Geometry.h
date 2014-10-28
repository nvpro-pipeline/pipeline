// Copyright NVIDIA Corporation 2012
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

#include <vector>
#include <map>

#include <dp/util/Config.h>
#include <dp/util/SharedPtr.h>
#include <dp/util/Types.h>
#include <dp/math/Vecnt.h>

namespace dp
{
  namespace util
  {
    namespace generator
    {
      #define NUM_ATTRIBS 16
      enum AttributeID
      {
        ATTRIB_POSITION        =1<<0,
        ATTRIB_VERTEX_WEIGHT   =1<<1,
        ATTRIB_NORMAL          =1<<2,
        ATTRIB_COLOR           =1<<3,
        ATTRIB_SECONDARY_COLOR =1<<4,
        ATTRIB_FOG_COORD       =1<<5,
        ATTRIB_UNUSED_1        =1<<6,
        ATTRIB_UNUSED_2        =1<<7,
        ATTRIB_TEXCOORD0       =1<<8,
        ATTRIB_TEXCOORD1       =1<<9,
        ATTRIB_TEXCOORD2       =1<<10,
        ATTRIB_TEXCOORD3       =1<<11,
        ATTRIB_TEXCOORD4       =1<<12,
        ATTRIB_TEXCOORD5       =1<<13,
        ATTRIB_TANGENT         =1<<14,
        ATTRIB_BINORMAL        =1<<15,
      };

      static const bool ATTRIB_POS_DEFAULTS[NUM_ATTRIBS] = 
      {
        true,
        true,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        false
      };

      struct AttributeData
      {
        unsigned int m_dimensionality;
        std::vector<float> m_data;
      };

      SMART_TYPES( GeometryData );

      class GeometryData
      {
      public:
        DP_UTIL_API static SmartGeometryData create( GeometryPrimitiveType gpt );
        DP_UTIL_API static SmartGeometryData create( SmartGeometryData const& rhs );
        DP_UTIL_API ~GeometryData();

        std::vector<unsigned int> m_indices;
        std::map<AttributeID,AttributeData> m_attributes;
        GeometryPrimitiveType m_gpt;
#if !defined(NDEBUG)
        DP_UTIL_API bool checkConsistency() const;
#endif

      protected:
        DP_UTIL_API GeometryData( GeometryPrimitiveType gpt );
        DP_UTIL_API GeometryData( SmartGeometryData const& rhs);
      };

      class AttributeFeed
      {
      public:
        AttributeFeed(SmartGeometryData& data, AttributeID attributeId, unsigned int attrMask, unsigned int dimensionality, size_t numElements = 0)
        {
          if( m_enabled = !!(attrMask & attributeId) )
          {
            m_data = &data->m_attributes[attributeId];
            m_data->m_dimensionality = dimensionality;
            m_data->m_data.reserve(numElements * dimensionality);
          }
        }

        inline void add( float element )
        {
          if(m_enabled)
          {
            DP_ASSERT(m_data->m_dimensionality == 1);
            m_data->m_data.push_back(element);
          }
        }

        inline void add( dp::math::Vec2f element )
        {
          if(m_enabled)
          {
            DP_ASSERT(m_data->m_dimensionality == 2);
            m_data->m_data.push_back(element[0]);
            m_data->m_data.push_back(element[1]);
          }
        }

        inline void add( dp::math::Vec3f element )
        {
          if(m_enabled)
          {
            DP_ASSERT(m_data->m_dimensionality == 3);
            m_data->m_data.push_back(element[0]);
            m_data->m_data.push_back(element[1]);
            m_data->m_data.push_back(element[2]);
          }
        }

        inline void add( dp::math::Vec4f element )
        {
          if(m_enabled)
          {
            DP_ASSERT(m_data->m_dimensionality == 4);
            m_data->m_data.push_back(element[0]);
            m_data->m_data.push_back(element[1]);
            m_data->m_data.push_back(element[2]);
            m_data->m_data.push_back(element[3]);
          }
        }

      private:
        AttributeData* m_data;
        bool m_enabled;
      };

      class IndexFeed
      {
      public:
        IndexFeed( SmartGeometryData& data, size_t numElements = 0 )
                 : m_indices(data->m_indices)
        {
          m_indices.reserve(numElements);
        }

        void inline add(unsigned int index)
        {
          m_indices.push_back(index);
        }

      private:
        std::vector<unsigned int>& m_indices;
      };

      template <unsigned int dim>
      SmartGeometryData createPoint( const math::Vecnt<dim,float>& v0 )
      {
        SmartGeometryData meshOut = new GeometryData(GPT_POINTS);

        AttributeFeed positions(meshOut, ATTRIB_POSITION, ATTRIB_POSITION, dim, 1);
        positions.add( v0 );

        DP_ASSERT( meshOut->checkConsistency() );

        return meshOut;
      }

      template <unsigned int dim>
      SmartGeometryData createPoints( const std::vector< math::Vecnt<dim,float> >& v, bool indexed = false )
      {
        SmartGeometryData meshOut = new GeometryData(GPT_POINTS);

        AttributeFeed positions(meshOut, ATTRIB_POSITION, ATTRIB_POSITION, dim, v.size() );
        for( typename std::vector< math::Vecnt<dim,float> >::const_iterator it = v.begin(); it != v.end(); ++it )
        {
          positions.add( *it );
        }

        if( indexed )
        {
          IndexFeed indices( meshOut, v.size() );
          for( unsigned int i = 0; i < v.size(); ++i )
          {
            indices.add(i);
          }
        }

        DP_ASSERT( meshOut->checkConsistency() );

        return meshOut;
      }

      DP_UTIL_API SmartGeometryData createQuad( unsigned int attrMask
                                              , math::Vec3f v0 = math::Vec3f(0.0f, 0.0f, 0.0f)
                                              , math::Vec3f v1 = math::Vec3f(1.0f, 0.0f, 0.0f)
                                              , math::Vec3f v2 = math::Vec3f(0.0f, 1.0f, 0.0f)
                                              , math::Vec2f t0 = math::Vec2f(0.0f, 0.0f)
                                              , math::Vec2f t1 = math::Vec2f(1.0f, 0.0f)
                                              , math::Vec2f t2 = math::Vec2f(0.0f, 1.0f) );

      DP_UTIL_API SmartGeometryData createQuadIndexed( unsigned int attrMask
                                                     , math::Vec2f t0 = math::Vec2f(0.0f, 0.0f)
                                                     , math::Vec2f t1 = math::Vec2f(1.0f, 0.0f)
                                                     , math::Vec2f t2 = math::Vec2f(0.0f, 1.0f) );

      DP_UTIL_API SmartGeometryData createTriangle( unsigned int attrMask
                                                  , math::Vec3f v0 = math::Vec3f(0.0f, 0.0f, 0.0f)
                                                  , math::Vec3f v1 = math::Vec3f(1.0f, 0.0f, 0.0f) 
                                                  , math::Vec3f v2 = math::Vec3f(0.0f, 1.0f, 0.0f)
                                                  , math::Vec2f t0 = math::Vec2f(0.0f, 0.0f)
                                                  , math::Vec2f t1 = math::Vec2f(1.0f, 0.0f)
                                                  , math::Vec2f t2 = math::Vec2f(0.0f, 1.0f) );

      //TODO: The float t{Left|Top|Right|Bottom} needs to be adapted to Vec4f tRect
      DP_UTIL_API SmartGeometryData createRectangle( unsigned int attrMask
                                                   , float left, float top, float right, float bottom
                                                   , float tLeft = 0.0f, float tTop = 1.0f
                                                   , float tRight = 1.0f, float tBottom = 0.0f);

      DP_UTIL_API SmartGeometryData createCube( unsigned int attrMask
                                              , math::Vec2f t0 = math::Vec2f(0.0f, 0.0f)
                                              , math::Vec2f t1 = math::Vec2f(1.0f, 0.0f)
                                              , math::Vec2f t2 = math::Vec2f(0.0f, 1.0f) );

      DP_UTIL_API SmartGeometryData createCylinder( unsigned int attrMask
                                                  , unsigned int longitudeDivs                   //Number of times to subdivide the circular cross-section
                                                  , unsigned int heightDivs = 2                  //Number of times to subdivide the height span of the cylinder
                                                  , float longitudeEnd = 0.0f                    //Optionally set the ending angle of the circular cross-section
                                                  , float innerRadius = 0.0f );                  //Optionally set an inner Radius, thus making the shape a solid tube

      DP_UTIL_API SmartGeometryData createSphere( unsigned int attrMask
                                                , unsigned int longitudeDivs                     //Number of axial subdivisions (y-axis)  
                                                , unsigned int latitudeDivs                      //Number of meridional subdivisions
                                                , float longitudeEnd = 0.0f                      //Optionally set the ending longitudinal angle
                                                , float latitudeEnd = math::PI                   //Optionally set the ending latitudinal angle
                                                , float latitudeBegin = 0.0f );                  //Optionall set the starting latitude angle


      template<unsigned int n>
      SmartGeometryData transformAttribute( math::Matmnt<n,n,float> matrix, AttributeID attribute
                                          , const SmartGeometryData& meshIn, bool bPositional = true
                                          , SmartGeometryData meshOut = SmartGeometryData::null )
      {
        DP_ASSERT( meshIn->checkConsistency() );

        if( meshOut == SmartGeometryData::null )
        {
          meshOut = GeometryData::create(meshIn);
        }
        else
        {
          DP_ASSERT( meshOut->checkConsistency() );
        }

        std::map<AttributeID,AttributeData>::const_iterator curAttr = meshIn->m_attributes.find(attribute);
        DP_ASSERT( curAttr != meshIn->m_attributes.end() );

        unsigned int dimensionality = curAttr->second.m_dimensionality;
        if( !((dimensionality == n) || (dimensionality == n-1)) )
        {
          DP_ASSERT(!"Invalid dimensions");
          return false;
        }

        float coeffHomo = bPositional ? 1.0f : 0.0f;
        std::vector<float>& outv = meshOut->m_attributes[attribute].m_data;

        if( !outv.empty() )
        {
          meshOut->m_attributes[attribute].m_data.clear();
        }


        for( std::vector<float>::const_iterator it = curAttr->second.m_data.begin(); it != curAttr->second.m_data.end(); it += dimensionality )
        {
          math::Vecnt<n,float> vec;
          for(int i = 0; i < n-1; i++)
          {
            vec[i] = *(it+i);
          }
          vec[n-1] = dimensionality == n-1 ? coeffHomo : *(it+n-1);

          math::Vecnt<n,float> vecResult = matrix*vec;
          for(int i = 0; i < n-1; i++)
          {
            outv.push_back(vecResult[i]);
          }
          if( dimensionality == n)
          {
            outv.push_back(vecResult[n-1]);
          }

        }

        return meshOut;

      }

      template<unsigned int n>
      SmartGeometryData transformAttributes( math::Matmnt<n,n,float> matrix, unsigned int attributes, const SmartGeometryData& meshIn, SmartGeometryData meshOut = SmartGeometryData::null )
      {
        for(unsigned int attr = 0; attr < NUM_ATTRIBS; attr++)
        {
          AttributeID curAttr = (AttributeID)(1<<attr);
          if(attributes & curAttr)
          {
            meshOut = transformAttribute(matrix, curAttr, meshIn, ATTRIB_POS_DEFAULTS[attr], meshOut);
          }
        }
        return meshOut;
      }

    } // namespace generator
  } // namespace util
} // namespace dp
