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


#pragma once

#include <vector>
#include <map>
#include <dp/Types.h>
#include <dp/math/math.h>
#include <dp/math/Matmnt.h>
#include <dp/math/Vecnt.h>
#include <dp/util/Flags.h>
#include <dp/util/PointerTypes.h>
#include <test/rix/core/helpers/inc/Config.h>

namespace dp
{
  namespace rix
  {
    namespace util
    {
      #define NUM_ATTRIBS 16
      enum class AttributeID
      {
        POSITION        =1<<0,
        VERTEX_WEIGHT   =1<<1,
        NORMAL          =1<<2,
        COLOR           =1<<3,
        SECONDARY_COLOR =1<<4,
        FOG_COORD       =1<<5,
        UNUSED_1        =1<<6,
        UNUSED_2        =1<<7,
        TEXCOORD0       =1<<8,
        TEXCOORD1       =1<<9,
        TEXCOORD2       =1<<10,
        TEXCOORD3       =1<<11,
        TEXCOORD4       =1<<12,
        TEXCOORD5       =1<<13,
        TANGENT         =1<<14,
        BINORMAL        =1<<15,
      };

      typedef dp::util::Flags<AttributeID>  AttributeMask;

      inline AttributeMask operator|( AttributeID bit0, AttributeID bit1 )
      {
        return AttributeMask( bit0 ) | bit1;
      }

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

      DEFINE_PTR_TYPES( GeometryData );

      class GeometryData
      {
      public:
        DPHELPERS_API static GeometryDataSharedPtr create( GeometryPrimitiveType gpt );
        DPHELPERS_API static GeometryDataSharedPtr create( GeometryDataSharedPtr const& rhs );
        DPHELPERS_API ~GeometryData();

        std::vector<unsigned int> m_indices;
        std::map<AttributeID,AttributeData> m_attributes;
        GeometryPrimitiveType m_gpt;
#if !defined(NDEBUG)
        DPHELPERS_API bool checkConsistency() const;
#endif

      protected:
        DPHELPERS_API GeometryData( GeometryPrimitiveType gpt );
        DPHELPERS_API GeometryData( GeometryDataSharedPtr const& rhs);
      };

      class AttributeFeed
      {
      public:
        AttributeFeed(GeometryDataSharedPtr& data, AttributeID attributeId, AttributeMask attrMask, unsigned int dimensionality, size_t numElements = 0)
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
        IndexFeed( GeometryDataSharedPtr& data, size_t numElements = 0 )
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
      GeometryDataSharedPtr createPoint( const math::Vecnt<dim,float>& v0 )
      {
        GeometryDataSharedPtr meshOut = new GeometryData(GeometryPrimitiveType::POINTS);

        AttributeFeed positions(meshOut, ATTRIB_POSITION, ATTRIB_POSITION, dim, 1);
        positions.add( v0 );

        DP_ASSERT( meshOut->checkConsistency() );

        return meshOut;
      }

      template <unsigned int dim>
      GeometryDataSharedPtr createPoints( const std::vector< math::Vecnt<dim,float> >& v, bool indexed = false )
      {
        GeometryDataSharedPtr meshOut = new GeometryData(GeometryPrimitiveType::POINTS);

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

      DPHELPERS_API GeometryDataSharedPtr createQuad( AttributeMask attrMask
                                                  , math::Vec3f v0 = math::Vec3f(0.0f, 0.0f, 0.0f)
                                                  , math::Vec3f v1 = math::Vec3f(1.0f, 0.0f, 0.0f)
                                                  , math::Vec3f v2 = math::Vec3f(0.0f, 1.0f, 0.0f)
                                                  , math::Vec2f t0 = math::Vec2f(0.0f, 0.0f)
                                                  , math::Vec2f t1 = math::Vec2f(1.0f, 0.0f)
                                                  , math::Vec2f t2 = math::Vec2f(0.0f, 1.0f) );

      DPHELPERS_API GeometryDataSharedPtr createQuadIndexed( AttributeMask attrMask
                                                         , math::Vec2f t0 = math::Vec2f(0.0f, 0.0f)
                                                         , math::Vec2f t1 = math::Vec2f(1.0f, 0.0f)
                                                         , math::Vec2f t2 = math::Vec2f(0.0f, 1.0f) );

      DPHELPERS_API GeometryDataSharedPtr createTriangle( AttributeMask attrMask
                                                      , math::Vec3f v0 = math::Vec3f(0.0f, 0.0f, 0.0f)
                                                      , math::Vec3f v1 = math::Vec3f(1.0f, 0.0f, 0.0f)
                                                      , math::Vec3f v2 = math::Vec3f(0.0f, 1.0f, 0.0f)
                                                      , math::Vec2f t0 = math::Vec2f(0.0f, 0.0f)
                                                      , math::Vec2f t1 = math::Vec2f(1.0f, 0.0f)
                                                      , math::Vec2f t2 = math::Vec2f(0.0f, 1.0f) );

      //TODO: The float t{Left|Top|Right|Bottom} needs to be adapted to Vec4f tRect
      DPHELPERS_API GeometryDataSharedPtr createRectangle( AttributeMask attrMask
                                                       , float left, float top, float right, float bottom
                                                       , float tLeft = 0.0f, float tTop = 1.0f
                                                       , float tRight = 1.0f, float tBottom = 0.0f);

      DPHELPERS_API GeometryDataSharedPtr createCube( AttributeMask attrMask
                                                  , math::Vec2f t0 = math::Vec2f(0.0f, 0.0f)
                                                  , math::Vec2f t1 = math::Vec2f(1.0f, 0.0f)
                                                  , math::Vec2f t2 = math::Vec2f(0.0f, 1.0f) );

      DPHELPERS_API GeometryDataSharedPtr createCylinder( AttributeMask attrMask
                                                      , unsigned int longitudeDivs                   //Number of times to subdivide the circular cross-section
                                                      , unsigned int heightDivs = 2                  //Number of times to subdivide the height span of the cylinder
                                                      , float longitudeEnd = 0.0f                    //Optionally set the ending angle of the circular cross-section
                                                      , float innerRadius = 0.0f );                  //Optionally set an inner Radius, thus making the shape a solid tube

      DPHELPERS_API GeometryDataSharedPtr createSphere( AttributeMask attrMask
                                                    , unsigned int longitudeDivs                     //Number of axial subdivisions (y-axis)
                                                    , unsigned int latitudeDivs                      //Number of meridional subdivisions
                                                    , float longitudeEnd = 0.0f                      //Optionally set the ending longitudinal angle
                                                    , float latitudeEnd = math::PI                   //Optionally set the ending latitudinal angle
                                                    , float latitudeBegin = 0.0f );                  //Optionall set the starting latitude angle


      template<unsigned int n>
      GeometryDataSharedPtr transformAttribute( math::Matmnt<n,n,float> matrix, AttributeID attribute
                                              , const GeometryDataSharedPtr& meshIn, bool bPositional = true
                                              , GeometryDataSharedPtr meshOut = GeometryDataSharedPtr() )
      {
        DP_ASSERT( meshIn->checkConsistency() );

        if( meshOut == GeometryDataSharedPtr() )
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
      GeometryDataSharedPtr transformAttributes( math::Matmnt<n,n,float> matrix, unsigned int attributes, const GeometryDataSharedPtr& meshIn, GeometryDataSharedPtr meshOut = GeometryDataSharedPtr::null )
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
