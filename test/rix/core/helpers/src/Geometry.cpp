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


#include <dp/Assert.h>
#include <dp/math/math.h>
#include <dp/math/Matmnt.h>
#include <test/rix/core/helpers/Geometry.h>

namespace dp
{
  namespace rix
  {
    using namespace std;
    using namespace math;

    namespace util
    {
      GeometryDataSharedPtr GeometryData::create( GeometryPrimitiveType gpt )
      {
        return( std::shared_ptr<GeometryData>( new GeometryData( gpt ) ) );
      }

      GeometryDataSharedPtr GeometryData::create( GeometryDataSharedPtr const& rhs )
      {
        return( std::shared_ptr<GeometryData>( new GeometryData( rhs ) ) );
      }

      GeometryData::GeometryData( GeometryPrimitiveType gpt )
      {
        m_gpt = gpt;
      }

      GeometryData::~GeometryData()
      {
      }

      GeometryData::GeometryData( GeometryDataSharedPtr const& rhs )
      {
        m_attributes = rhs->m_attributes;
        m_indices = rhs->m_indices;
      }

#if !defined(NDEBUG)
      bool GeometryData::checkConsistency() const
      {
        size_t numElements = 0;
        for(map<AttributeID,AttributeData>::const_iterator it = m_attributes.begin(); it != m_attributes.end(); it++)
        {
          if( it->second.m_data.empty() )
          {
            return false;
          }

          if(!numElements)
          {
            numElements = it->second.m_data.size() / it->second.m_dimensionality;
          }
          else
          {
            if( numElements != it->second.m_data.size() / it->second.m_dimensionality || !!(it->second.m_data.size()%it->second.m_dimensionality) )
            {
              return false;
            }
          }
        }

        return true;
      }

#endif

      void calculateTBN( const Vec3f & v0, const Vec3f & v1, const Vec3f & v2
                       , const Vec2f & t0, const Vec2f & t1, const Vec2f & t2
                       , Vec3f & t, Vec3f & b, Vec3f & n )
      {
        // normal is cross product of edges
        Vec3f e0 = v1 - v0;
        Vec3f e1 = v2 - v0;
        n = e0 ^ e1;
        n.normalize();

        // tangent is determined by texture and edges
        Vec2f dt = t1 - t0;
        t = dt[0] * e0 + dt[1] * e1;
        t.normalize();

        // binormal is cross product of normal and tangent
        // (normalization might not be needed)
        b = n ^ t;
        b.normalize();
      }

      GeometryDataSharedPtr createQuad( unsigned int attrMask
                                      , Vec3f v0 /*= math::Vec3f(0.0f, 0.0f, 0.0f)*/
                                      , Vec3f v1 /*= math::Vec3f(1.0f, 0.0f, 0.0f)*/
                                      , Vec3f v2 /*= math::Vec3f(0.0f, 1.0f, 0.0f)*/
                                      , Vec2f t0 /*= math::Vec2f(0.0f, 0.0f)*/
                                      , Vec2f t1 /*= math::Vec2f(1.0f, 0.0f)*/
                                      , Vec2f t2 /*= math::Vec2f(0.0f, 1.0f)*/ )
      {

        GeometryDataSharedPtr meshOut = GeometryData::create(GeometryPrimitiveType::TRIANGLE_STRIP);

        AttributeFeed positions(meshOut, ATTRIB_POSITION, attrMask, 3, 4);
        AttributeFeed texCoords(meshOut, ATTRIB_TEXCOORD0, attrMask, 2, 4);
        AttributeFeed normals(meshOut, ATTRIB_NORMAL, attrMask, 3, 4);
        AttributeFeed tangents(meshOut, ATTRIB_TANGENT, attrMask, 3, 4);
        AttributeFeed binormals(meshOut, ATTRIB_BINORMAL, attrMask, 3, 4);

        Vec3f n0, tg0, bn0;
        calculateTBN( v0, v1, v2, t0, t1, t2, tg0, bn0, n0 );

        positions.add( v0 );
        texCoords.add( t0 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        positions.add( v1 );
        texCoords.add( t1 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        positions.add( v2 );
        texCoords.add( t2 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        positions.add( v2 + v1 - v0 );
        texCoords.add( t2 + t1 - t0 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        DP_ASSERT( meshOut->checkConsistency() );

        return meshOut;
      }

      GeometryDataSharedPtr createQuadIndexed( unsigned int attrMask
                                             , Vec3f v0 /*= math::Vec3f(0.0f, 0.0f, 0.0f)*/
                                             , Vec3f v1 /*= math::Vec3f(1.0f, 0.0f, 0.0f)*/
                                             , Vec3f v2 /*= math::Vec3f(0.0f, 1.0f, 0.0f)*/
                                             , Vec2f t0 /*= math::Vec2f(0.0f, 0.0f)*/
                                             , Vec2f t1 /*= math::Vec2f(1.0f, 0.0f)*/
                                             , Vec2f t2 /*= math::Vec2f(0.0f, 1.0f)*/ )
      {

        GeometryDataSharedPtr meshOut = GeometryData::create(GeometryPrimitiveType::TRIANGLE_STRIP);

        AttributeFeed positions(meshOut, ATTRIB_POSITION, attrMask, 3, 4);
        AttributeFeed texCoords(meshOut, ATTRIB_TEXCOORD0, attrMask, 2, 4);
        AttributeFeed normals(meshOut, ATTRIB_NORMAL, attrMask, 3, 4);
        AttributeFeed tangents(meshOut, ATTRIB_TANGENT, attrMask, 3, 4);
        AttributeFeed binormals(meshOut, ATTRIB_BINORMAL, attrMask, 3, 4);

        Vec3f n0, tg0, bn0;
        calculateTBN( v0, v1, v2, t0, t1, t2, tg0, bn0, n0 );

        positions.add( v0 );
        texCoords.add( t0 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        positions.add( v1 );
        texCoords.add( t1 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        positions.add( v2 );
        texCoords.add( t2 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        positions.add( v2 + v1 - v0 );
        texCoords.add( t2 + t1 - t0 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        IndexFeed indices(meshOut, 4);
        indices.add(0);
        indices.add(1);
        indices.add(2);
        indices.add(3);

        DP_ASSERT( meshOut->checkConsistency() );

        return meshOut;
      }


      GeometryDataSharedPtr createTriangle( unsigned int attrMask
                                          , Vec3f v0 /*= math::Vec3f(0.0f, 0.0f, 0.0f)*/
                                          , Vec3f v1 /*= math::Vec3f(1.0f, 0.0f, 0.0f)*/
                                          , Vec3f v2 /*= math::Vec3f(0.0f, 1.0f, 0.0f)*/
                                          , Vec2f t0 /*= math::Vec2f(0.0f, 0.0f)*/
                                          , Vec2f t1 /*= math::Vec2f(1.0f, 0.0f)*/
                                          , Vec2f t2 /*math::Vec2f(0.0f, 1.0f)*/ )
      {

        GeometryDataSharedPtr meshOut = GeometryData::create(GeometryPrimitiveType::TRIANGLES);

        AttributeFeed positions(meshOut, ATTRIB_POSITION, attrMask, 3, 3);
        AttributeFeed texCoord(meshOut, ATTRIB_TEXCOORD0, attrMask, 2, 3);
        AttributeFeed normals(meshOut, ATTRIB_NORMAL, attrMask, 3, 3);
        AttributeFeed tangents(meshOut, ATTRIB_TANGENT, attrMask, 3, 3);
        AttributeFeed binormals(meshOut, ATTRIB_BINORMAL, attrMask, 3, 3);

        Vec3f n0, tg0, bn0;
        calculateTBN( v0, v1, v2, t0, t1, t2, tg0, bn0, n0 );

        positions.add( v0 );
        texCoord.add( t0 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        positions.add( v1 );
        texCoord.add( t1 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        positions.add( v2 );
        texCoord.add( t2 );
        normals.add( n0 );
        tangents.add( tg0 );
        binormals.add( bn0 );

        DP_ASSERT( meshOut->checkConsistency() );

        return meshOut;
      }

      //TODO: The float t{Left|Top|Right|Bottom} needs to be adapted to Vec4f tRect
      GeometryDataSharedPtr createRectangle( unsigned int attrMask
                                           , float left, float top, float right, float bottom
                                           , float tLeft /*= 0.0f*/
                                           , float tTop /*= 1.0f*/
                                           , float tRight /*= 1.0f*/
                                           , float tBottom /*= 0.0f*/)
      {
        return createQuadIndexed(attrMask, Vec3f(left, bottom, 0.0f) 
                                         , Vec3f(right, bottom, 0.0f)
                                         , Vec3f(left, top, 0.0f)
                                         , Vec2f(tLeft, tBottom)
                                         , Vec2f(tRight, tBottom)
                                         , Vec2f(tLeft, tTop) );
      }

      GeometryDataSharedPtr createCube( unsigned int attrMask
                                      , Vec2f t0 /*= math::Vec2f(0.0f, 0.0f)*/
                                      , Vec2f t1 /*= math::Vec2f(1.0f, 0.0f)*/
                                      , Vec2f t2 /*= math::Vec2f(0.0f, 1.0f)*/ )
      {
        const int numVerts = 24;

        GeometryDataSharedPtr meshOut = GeometryData::create(GeometryPrimitiveType::TRIANGLES);

        AttributeFeed positions(meshOut, ATTRIB_POSITION, attrMask, 3, numVerts);
        AttributeFeed texCoords(meshOut, ATTRIB_TEXCOORD0, attrMask, 2, numVerts);
        AttributeFeed normals(meshOut, ATTRIB_NORMAL, attrMask, 3, numVerts);
        AttributeFeed tangents(meshOut, ATTRIB_TANGENT, attrMask, 3, numVerts);
        AttributeFeed binormals(meshOut, ATTRIB_BINORMAL, attrMask, 3, numVerts);


        positions.add( Vec3f(-1.0f, -1.0f, 1.0f) );
        texCoords.add( t0 );
        normals.add( Vec3f(0.0f, 0.0f, 1.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(1.0f, -1.0f, 1.0f) );
        texCoords.add( Vec2f(t1[0], t1[1]) );
        normals.add( Vec3f(0.0f, 0.0f, 1.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(1.0f, 1.0f, 1.0f) );
        texCoords.add( t1 + t2 - 2.0f*t0 );
        normals.add( Vec3f(0.0f, 0.0f, 1.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(-1.0f, 1.0f, 1.0f) );
        texCoords.add( t2 );
        normals.add( Vec3f(0.0f, 0.0f, 1.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );


        positions.add( Vec3f(1.0f, -1.0f, 1.0f) );
        texCoords.add( t0 );
        normals.add( Vec3f(1.0f, 0.0f, 0.0f) );
        tangents.add( Vec3f(0.0f, 0.0f, -1.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(1.0f, -1.0f, -1.0f) );
        texCoords.add( Vec2f(t1[0], t1[1]) );
        normals.add( Vec3f(1.0f, 0.0f, 0.0f) );
        tangents.add( Vec3f(0.0f, 0.0f, -1.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(1.0f, 1.0f, -1.0f) );
        texCoords.add( t1 + t2 - 2.0f*t0 );
        normals.add( Vec3f(1.0f, 0.0f, 0.0f) );
        tangents.add( Vec3f(0.0f, 0.0f, -1.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(1.0f, 1.0f, 1.0f) );
        texCoords.add( t2 );
        normals.add( Vec3f(1.0f, 0.0f, 0.0f) );
        tangents.add( Vec3f(0.0f, 0.0f, -1.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );


        positions.add( Vec3f(1.0f, -1.0f, -1.0f) );
        texCoords.add( t0 );
        normals.add( Vec3f(0.0f, 0.0f, -1.0f) );
        tangents.add( Vec3f(-1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(-1.0f, -1.0f, -1.0f) );
        texCoords.add( Vec2f(t1[0], t1[1]) );
        normals.add( Vec3f(0.0f, 0.0f, -1.0f) );
        tangents.add( Vec3f(-1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(-1.0f, 1.0f, -1.0f) );
        texCoords.add( t1 + t2 - 2.0f*t0 );
        normals.add( Vec3f(0.0f, 0.0f, -1.0f) );
        tangents.add( Vec3f(-1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(1.0f, 1.0f, -1.0f) );
        texCoords.add( t2 );
        normals.add( Vec3f(0.0f, 0.0f, -1.0f) );
        tangents.add( Vec3f(-1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );


        positions.add( Vec3f(-1.0f, -1.0f, -1.0f) );
        texCoords.add( t0 );
        normals.add( Vec3f(-1.0f, 0.0f, 0.0f) );
        tangents.add( Vec3f(0.0f, 0.0f, 1.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(-1.0f, -1.0f, 1.0f) );
        texCoords.add( Vec2f(t1[0], t1[1]) );
        normals.add( Vec3f(-1.0f, 0.0f, 0.0f) );
        tangents.add( Vec3f(0.0f, 0.0f, 1.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(-1.0f, 1.0f, 1.0f) );
        texCoords.add( t1 + t2 - 2.0f*t0 );
        normals.add( Vec3f(-1.0f, 0.0f, 0.0f) );
        tangents.add( Vec3f(0.0f, 0.0f, 1.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );

        positions.add( Vec3f(-1.0f, 1.0f, -1.0f) );
        texCoords.add( t2 );
        normals.add( Vec3f(-1.0f, 0.0f, 0.0f) );
        tangents.add( Vec3f(0.0f, 0.0f, 1.0f) );
        binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );


        positions.add( Vec3f(-1.0f, 1.0f, 1.0f) );
        texCoords.add( t0 );
        normals.add( Vec3f(0.0f, 1.0f, 0.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 0.0f, -1.0f) );

        positions.add( Vec3f(-1.0f, 1.0f, -1.0f) );
        texCoords.add( Vec2f(t1[0], t1[1]) );
        normals.add( Vec3f(0.0f, 1.0f, 0.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 0.0f, -1.0f) );

        positions.add( Vec3f(1.0f, 1.0f, -1.0f) );
        texCoords.add( t1 + t2 - 2.0f*t0 );
        normals.add( Vec3f(0.0f, 1.0f, 0.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 0.0f, -1.0f) );

        positions.add( Vec3f(1.0f, 1.0f, 1.0f) );
        texCoords.add( t2 );
        normals.add( Vec3f(0.0f, 1.0f, 0.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 0.0f, -1.0f) );


        positions.add( Vec3f(1.0f, -1.0f, -1.0f) );
        texCoords.add( t0 );
        normals.add( Vec3f(0.0f, -1.0f, 0.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 0.0f, 1.0f) );

        positions.add( Vec3f(-1.0f, -1.0f, -1.0f) );
        texCoords.add( Vec2f(t1[0], t1[1]) );
        normals.add( Vec3f(0.0f, -1.0f, 0.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 0.0f, 1.0f) );

        positions.add( Vec3f(-1.0f, -1.0f, 1.0f) );
        texCoords.add( t1 + t2 - 2.0f*t0 );
        normals.add( Vec3f(0.0f, -1.0f, 0.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 0.0f, 1.0f) );

        positions.add( Vec3f(1.0f, -1.0f, 1.0f) );
        texCoords.add( t2 );
        normals.add( Vec3f(0.0f, -1.0f, 0.0f) );
        tangents.add( Vec3f(1.0f, 0.0f, 0.0f) );
        binormals.add( Vec3f(0.0f, 0.0f, 1.0f) );

        const unsigned int numIndices = 36;

        IndexFeed indices(meshOut, numIndices);

        for( unsigned int i = 0; i < 6; ++i )
        {
          indices.add( 4*i );
          indices.add( 4*i + 1 );
          indices.add( 4*i + 2 );

          indices.add( 4*i + 2 );
          indices.add( 4*i + 3 );
          indices.add( 4*i );
        }

        DP_ASSERT( meshOut->checkConsistency() );

        return meshOut;
      }

      GeometryDataSharedPtr createCylinder( unsigned int attrMask
                                          , unsigned int longitudeDivs
                                          , unsigned int heightDivs /*= 2*/
                                          , float longitudeEnd /*= 0.0f*/
                                          , float innerRadius /*= 0.0f*/ )
      {
        DP_ASSERT(heightDivs > 1);
        DP_ASSERT(longitudeDivs > 2);
        DP_ASSERT(innerRadius < 1.0f);
        DP_ASSERT(longitudeEnd < 2.0f * PI);
        DP_ASSERT(!(longitudeEnd < 0.0f));

        GeometryDataSharedPtr meshOut = GeometryData::create(GeometryPrimitiveType::TRIANGLES);

        bool bLongEndSplit = longitudeEnd > 0.0f;
        bool bTube = innerRadius > 0.0f;
        unsigned int numVertsPerLongitude = longitudeDivs + 1;
        unsigned int numVerts = (bTube ? 2 : 1) * heightDivs * numVertsPerLongitude        // If we have an inner radius then it's a tube and the circular prism must be repeated

                              + (bTube ? 4 * numVertsPerLongitude : 2 * numVertsPerLongitude)  // If we don't have an inner radius we just need two sets of vertices for the circular 
                                                                                                  // caps. If we do have an inner radius, wee have four circles, and so we need four such sets.

                              + (bLongEndSplit ? 8 : 0) + 2;                        // If our longitude angle is sort of the full circle we need to fill the two resulting
                                                                                                  // rectangular cross sections, and so wee need eight more vertices.

        AttributeFeed positions(meshOut, ATTRIB_POSITION, attrMask, 3, numVerts);
        AttributeFeed texCoords(meshOut, ATTRIB_TEXCOORD0, attrMask, 2, numVerts);
        AttributeFeed normals(meshOut, ATTRIB_NORMAL, attrMask, 3, numVerts);
        AttributeFeed tangents(meshOut, ATTRIB_TANGENT, attrMask, 3, numVerts);
        AttributeFeed binormals(meshOut, ATTRIB_BINORMAL, attrMask, 3, numVerts);

        float dtheta = (bLongEndSplit ? longitudeEnd : 2.0f * PI) / longitudeDivs;
        float dh = 2.0f / (heightDivs - 1);

        //Two vertices at the centers of both circular caps
        if(!bTube)
        {
          positions.add( Vec3f(0.0f, -1.0f, 0.0f) );
          texCoords.add( Vec2f(0.5f, 0.5f) );
          normals.add( Vec3f(0.0f, -1.0f, 0.0f) );
          tangents.add( Vec3f(-1.0f, 0.0f, 0.0f) );
          binormals.add( Vec3f(0.0f, 0.0f, -1.0f) );

          positions.add( Vec3f(0.0f, 1.0f, 0.0f) );
          texCoords.add( Vec2f(0.5f, 0.5f) );
          normals.add( Vec3f(0.0f, 1.0f, 0.0f) );
          tangents.add( Vec3f(-1.0f, 0.0f, 0.0f) );
          binormals.add( Vec3f(0.0f, 0.0f, 1.0f) );
        }

        //Create the round circular prism; twice if we have an inner radius
        for(int k = 0; k < 1 + (innerRadius>0.0f); k++)
        {
          float r = k ? innerRadius : 1.0f;
          float inv = k ? -1.0f : 1.0f;
          for(unsigned int i = 0; i < heightDivs; i++)
          {
            float y = i * dh - 1.0f;

            float texUDiv = 1.0f / longitudeDivs;
            float texV = (float)i / (heightDivs - 1);
            for(unsigned int j = 0; j <= longitudeDivs; j++)
            {
              float curTheta = j * dtheta;
              float x = -sin(curTheta);
              float z = -cos(curTheta);

              positions.add( Vec3f(r * x, y, r * z) );
              texCoords.add( Vec2f(texUDiv * j, texV) );
              normals.add( Vec3f(inv * x, 0.0f, inv * z) );
              tangents.add( Vec3f(inv * z, 0.0f, -inv * x) );
              binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );
            }
          }
        }

        //Create vertices along the circular caps, twice as may if we have an inner radius
        float halfInnerRadius = 0.5f * innerRadius;
        for(unsigned int k = 0; k < 2; k++)
        {
          float y = -1.0f + 2.0f * k;
          float inv = k ? 1.0f : -1.0f;
          for(int l = 0; l < 1 + (innerRadius>0.0f); l++ )
          {
            float r = l ? innerRadius : 1.0f;
            for(unsigned int j = 0; j <= longitudeDivs; j++)
            {
              float curTheta = j * dtheta;
              float x = -sin(curTheta);
              float z = -cos(curTheta);

              positions.add( Vec3f(r * x, y, r * z) );
              texCoords.add( Vec2f(0.5f + (l ? halfInnerRadius : 0.5f) * x, 0.5f + (l ? halfInnerRadius : 0.5f) * z ) );
              normals.add( Vec3f(0.0f, inv, 0.0f) );
              tangents.add( Vec3f( -inv, 0.0f, 0.0f) );
              binormals.add( Vec3f(0.0f, 0.0f, 1.0f) );
            }
          }
        }

        //If our longitudinal angle cuts short of the full circle, we must fill the two resulting rectangular cross-sections.
        if(bLongEndSplit)
        {
          for(unsigned int k = 0; k < 2; k++)
          {
            float curAngle = k * longitudeEnd;
            float x = -sin(curAngle);
            float z = -cos(curAngle);
            float inv = k ? 1.0f : -1.0f;
            
            for(unsigned int i = 0; i < heightDivs; i++)
            {
              float y = i * dh - 1.0f;
              float texV = float(i) / (heightDivs - 1);


              positions.add( Vec3f(x, y, z) );
              texCoords.add( Vec2f( 1.0f, texV ) );
              normals.add( Vec3f(inv * z, 0.0f, -inv * x) );
              tangents.add( Vec3f(-inv * x, 0.0f, -inv * z) );
              binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );


              positions.add( Vec3f(innerRadius * x, y, innerRadius * z) );
              texCoords.add( Vec2f( innerRadius, texV ) );
              normals.add( Vec3f(inv * z, 0.0f, -inv * x) );
              tangents.add( Vec3f(-inv * x, 0.0f, -inv * z) );
              binormals.add( Vec3f(0.0f, 1.0f, 0.0f) );
            }
          }
        }

        //That's it for the vertices, now the indices:

        unsigned int numIndices = (bTube ? 2 : 1) * (heightDivs - 1) * longitudeDivs * 6  //Round circular prism
                                + (bTube ? 2 : 1) * 6 * longitudeDivs                     //Caps
                                + (bLongEndSplit ? 12 : 0);                               //Rectangular cross sections if the longitudinal end angle is short of the full circle

        IndexFeed indices(meshOut, numIndices);

        unsigned int curVertex = bTube ? 0 : 2;

        for(int k = 0; k < 1 + (innerRadius>0.0f); k++)
        {
          for(unsigned int i = 0; i < heightDivs - 1; i++)
          {
            for(unsigned int j = 0; j < longitudeDivs; j++)
            {
              indices.add( curVertex + j + (k ? 1 : numVertsPerLongitude) );
              indices.add( curVertex + j );
              indices.add( curVertex + j + (k ? numVertsPerLongitude : 1) );

              indices.add( curVertex + j + (k ? numVertsPerLongitude : 1) );
              indices.add( curVertex + j + 1 + numVertsPerLongitude );
              indices.add( curVertex + j + (k ? 1 : numVertsPerLongitude ) );
            }
            curVertex += numVertsPerLongitude;
          }

          curVertex += numVertsPerLongitude;
        }

        if(bTube)
        {
          for(unsigned int k = 0; k < 2; k++)
          {
            for(unsigned int j = 0; j < longitudeDivs; j++)
            {
              indices.add( curVertex + j + (k ? numVertsPerLongitude : 1) );
              indices.add( curVertex + j );
              indices.add( curVertex + j + (k ? 1 : numVertsPerLongitude) );

              indices.add( curVertex + j + (k ? 1 : numVertsPerLongitude) );
              indices.add( curVertex + j + 1 + numVertsPerLongitude );
              indices.add( curVertex + j + (k ? numVertsPerLongitude : 1) );
            }
            curVertex += 2 * numVertsPerLongitude;
          }
        }
        else
        {
          for(unsigned int k = 0; k < 2; k++)
          {
            for(unsigned int j = 0; j < longitudeDivs; j++)
            {
              indices.add( k );
              indices.add( curVertex + j + !k );
              indices.add( curVertex + j + k );
            }
            curVertex += numVertsPerLongitude;
          }
        }

        if(bLongEndSplit)
        {
          for(unsigned int i = 0; i < heightDivs - 1; i++)
          {
            indices.add( curVertex );
            indices.add( curVertex + 1 );
            indices.add( curVertex + 3 );

            indices.add( curVertex + 3 );
            indices.add( curVertex + 2 );
            indices.add( curVertex + 0 );

            curVertex +=2;
          }
          curVertex +=2;

          for(unsigned int i = 0; i < heightDivs - 1; i++)
          {
            indices.add( curVertex );
            indices.add( curVertex + 2 );
            indices.add( curVertex + 3 );

            indices.add( curVertex + 3 );
            indices.add( curVertex + 1 );
            indices.add( curVertex + 0 );

            curVertex +=2;
          }
          curVertex +=2;
        }

        DP_ASSERT( meshOut->checkConsistency() );

        return meshOut;
      }

      GeometryDataSharedPtr createSphere( unsigned int attrMask
                                        , unsigned int longitudeDivs
                                        , unsigned int latitudeDivs
                                        , float longitudeEnd /*= 0.0f*/
                                        , float latitudeEnd /*= math::PI*/
                                        , float latitudeBegin /*= 0.0f*/ )
      {
        DP_ASSERT(latitudeDivs > 2);
        DP_ASSERT(longitudeDivs > 2);
        DP_ASSERT(!(longitudeEnd < 0.0f));
        DP_ASSERT(longitudeEnd < 2.0f * PI);
        DP_ASSERT(!(latitudeEnd > PI));
        DP_ASSERT(latitudeEnd > 0.0f);
        DP_ASSERT(!(latitudeBegin < 0.0f));
        DP_ASSERT(latitudeBegin < PI);
        DP_ASSERT(latitudeBegin < latitudeEnd);

        GeometryDataSharedPtr meshOut = GeometryData::create(GeometryPrimitiveType::TRIANGLES);

        bool bLongEndSplit = longitudeEnd > 0.0f;
        bool bLatEndSplit = latitudeEnd < PI;
        bool bLatBeginSplit = latitudeBegin > 0.0f;
        unsigned int numVertsPerLongitude = longitudeDivs + 1;
        unsigned int numVerts = latitudeDivs * numVertsPerLongitude          //Spherical surface
                              + (bLongEndSplit ? 2 + 2 * latitudeDivs        //Two semi-circular cross sections in case the longitudinal angle cuts short of a full circle
                              + (bLatEndSplit ? 2 : 0)                       //Two support vertices for the semi-circular cross sections that get cut off
                              + (bLatBeginSplit ? 2 : 0) : 0 )               //Two support vertices for the semi-circular cross sections that get cut off
                              + (bLatEndSplit ? longitudeDivs + 4 : 0)       //Cap for circular cross-section in case the ending latitudinal cut-off angle is short of PI
                              + (bLatBeginSplit ? longitudeDivs + 4 : 0);    //Cap for circular cross-section in case the starting latitudinal cut-off angle is greater than 0

        AttributeFeed positions(meshOut, ATTRIB_POSITION, attrMask, 3, numVerts);
        AttributeFeed texCoords(meshOut, ATTRIB_TEXCOORD0, attrMask, 2, numVerts);
        AttributeFeed normals(meshOut, ATTRIB_NORMAL, attrMask, 3, numVerts);
        AttributeFeed tangents(meshOut, ATTRIB_TANGENT, attrMask, 3, numVerts);
        AttributeFeed binormals(meshOut, ATTRIB_BINORMAL, attrMask, 3, numVerts);

        float dphi = (latitudeEnd - latitudeBegin) / (latitudeDivs - 1);
        float phi0 = latitudeBegin;
        float dtheta = (bLongEndSplit ? longitudeEnd : 2.0f * PI) / (longitudeDivs);

        //Generate spherical surface
        for(unsigned int i = 0; i < latitudeDivs; i++)
        {
          float curPhi = phi0 + dphi * i;
          float y = -cos( curPhi );
          float r = sin( curPhi );

          unsigned int j;
          float x;
          float z;
          float texU = (float)i / latitudeDivs;
          float texVDiv = 1.0f / longitudeDivs;
          for(j = 0; j <= longitudeDivs; j++)
          {
            float curTheta = dtheta * j;
            x = r * sin( curTheta );
            z = -r * cos( curTheta );

            positions.add( Vec3f(x, y, z) );
            texCoords.add( Vec2f(texU, texVDiv * j) );
            normals.add( Vec3f(x, y, z) );
            tangents.add( Vec3f( -cos( curTheta ), 0.0f, -sin( curTheta ) ) );
            binormals.add( Vec3f( -y * sin( curTheta ), r, y * cos( curTheta ) ) );
          }
        }

        //Generate two semi-circular cross section if longitudeEnd cuts short of a full circle
        if(bLongEndSplit)
        {
          float y;
          float r;
          float x;
          float z;
          float phi;
          float ncosLatBegin = -cos(latitudeBegin);
          float ncosLatEnd = -cos(latitudeEnd);

          {
            float centerY = (ncosLatBegin <= 0.0f) && (ncosLatEnd >= 0.0f) ? 0.0f : 0.5f * (ncosLatBegin + ncosLatEnd);
            positions.add( Vec3f(0.0f, centerY, 0.0f) );
            texCoords.add( Vec2f(0.5f, 0.5f + centerY) );
            normals.add( Vec3f(-1.0f, 0.0f, 0.0f) );
            tangents.add( Vec3f( 0.0f, 0.0f, 1.0f) );
            binormals.add( Vec3f( 0.0f, 1.0f, 0.0f) );
          }
          if( bLatBeginSplit )
          {
            positions.add( Vec3f(0.0f, ncosLatBegin, 0.0f) );
            texCoords.add( Vec2f(0.5f, 0.5f + 0.5f * ncosLatBegin) );
            normals.add( Vec3f(-1.0f, 0.0f, 0.0f) );
            tangents.add( Vec3f( 0.0f, 0.0f, 1.0f) );
            binormals.add( Vec3f( 0.0f, 1.0f, 0.0f) );
          }
          for(unsigned int i = 0; i < latitudeDivs; i++)
          {
            phi = phi0 + dphi * i;
            y = -cos( phi );
            r = sin( phi );
            x = 0.0f; // = r * sin( 0.0f )
            z = -r; // = -r * cos( 0.0f )

            positions.add( Vec3f(x, y, z) );
            texCoords.add( Vec2f(0.5f + 0.5f * r, 0.5f + 0.5f * y) );
            normals.add( Vec3f(-1.0f, 0.0f, 0.0f) );
            tangents.add( Vec3f( 0.0f, z, -y) );
            binormals.add( Vec3f(-x, -y, -z) );
          }
          if( bLatEndSplit )
          {
            positions.add( Vec3f(0.0f, ncosLatEnd, 0.0f) );
            texCoords.add( Vec2f(0.5f, 0.5f + 0.5f * ncosLatEnd) );
            normals.add( Vec3f(-1.0f, 0.0f, 0.0f) );
            tangents.add( Vec3f( 0.0f, 0.0f, -1.0f) );
            binormals.add( Vec3f( 0.0f, -1.0f, 0.0f) );
          }

          float theta = longitudeEnd;
          float ncosTheta = -cos(theta);
          float sinTheta = sin(theta);
          {
            float centerY = (ncosLatBegin <= 0.0f) && (ncosLatEnd >= 0.0f) ? 0.0f : 0.5f * (ncosLatBegin + ncosLatEnd);
            positions.add( Vec3f(0.0f, centerY, 0.0f) );
            texCoords.add( Vec2f(0.5f, 0.5f + centerY) );
            normals.add( Vec3f(ncosTheta, 0.0f, sinTheta ) );
            tangents.add( Vec3f( sinTheta, 0.0f, -ncosTheta) );
            binormals.add( Vec3f( 0.0f, 1.0f, 0.0f) );
          }
          if( bLatBeginSplit )
          {
            positions.add( Vec3f(0.0f, ncosLatBegin, 0.0f) );
            texCoords.add( Vec2f(0.5f, 0.5f + 0.5f * ncosLatBegin) );
            normals.add( Vec3f(ncosTheta, 0.0f, sinTheta) );
            tangents.add( Vec3f( sinTheta, 0.0f, -ncosTheta) );
            binormals.add( Vec3f( 0.0f, 1.0f, 0.0f) );
          }
          for(unsigned int i = 0; i < latitudeDivs; i++)
          {
            phi = phi0 + dphi * i;
            y = -cos( phi );
            r = sin( phi );
            x = r * sinTheta;
            z = r * ncosTheta;

            positions.add( Vec3f(x, y, z) );
            texCoords.add( Vec2f(0.5f + 0.5f * sin(phi), 0.5f + 0.5f * cos(phi)) );
            normals.add( Vec3f(ncosTheta, 0.0f, sinTheta) );
            tangents.add( Vec3f( x, 0.0f, -z) );
            binormals.add( Vec3f( 0.0f, 1.0f, 0.0f) );
          }
          if( bLatEndSplit )
          {
            positions.add( Vec3f(0.0f, ncosLatEnd, 0.0f) );
            texCoords.add( Vec2f(0.5f, 0.5f + 0.5f * ncosLatEnd) );
            normals.add( Vec3f(ncosTheta, 0.0f, sinTheta) );
            tangents.add( Vec3f( sinTheta, 0.0f, -ncosTheta) );
            binormals.add( Vec3f( 0.0f, 1.0f, 0.0f) );
          }
        }

        //Generate a cap for the circular cross-section in case latitudeEnd cuts short of PI
        if( bLatEndSplit )
        {
          float y = -cos( latitudeEnd );
          float r = sin( latitudeEnd );

          {
            positions.add( Vec3f(0.0f, y, 0.0f) );
            texCoords.add( Vec2f(0.5f, 0.5f) );
            normals.add( Vec3f(0.0f, 1.0f, 0.0f) );
            tangents.add( Vec3f( 0.0f, 0.0f, 1.0f) );
            binormals.add( Vec3f( 1.0f, 0.0f, 0.0f) );
          }

          for(unsigned int i = 0; i <= longitudeDivs; i++ )
          {
            float curTheta = dtheta * i;
            float s = sin( curTheta );
            float c = cos( curTheta );
            float x = r * s;
            float z = -r * c;

            positions.add( Vec3f(x, y, z) );
            texCoords.add( Vec2f(0.5f + 0.5f * x, 0.5f + 0.5f * z) );
            normals.add( Vec3f(0.0f, 1.0f, 0.0f) );
            tangents.add( Vec3f( 0.0f, 0.0f, 1.0f) );
            binormals.add( Vec3f( 1.0f, 0.0f, 0.0f) );
          }
        }

        //Generate a cap for the circular cross-section in case latitudeBegin is create than 0
        if( bLatBeginSplit )
        {
          float y = -cos( latitudeBegin );
          float r = sin( latitudeBegin );

          {
            positions.add( Vec3f(0.0f, y, 0.0f) );
            texCoords.add( Vec2f(0.5f, 0.5f) );
            normals.add( Vec3f(0.0f, 1.0f, 0.0f) );
            tangents.add( Vec3f( 1.0f, 0.0f, 0.0f) );
            binormals.add( Vec3f( 0.0f, 0.0f, 1.0f) );
          }

          for(unsigned int i = 0; i <= longitudeDivs; i++ )
          {
            float curTheta = dtheta * i;
            float s = sin( curTheta );
            float c = cos( curTheta );
            float x = r * s;
            float z = -r * c;

            positions.add( Vec3f(x, y, z) );
            texCoords.add( Vec2f(0.5f + 0.5f * x, 0.5f + 0.5f * z) );
            normals.add( Vec3f(0.0f, 1.0f, 0.0f) );
            tangents.add( Vec3f( 1.0f, 0.0f, 0.0f) );
            binormals.add( Vec3f( 0.0f, 0.0f, 1.0f) );
          }
        }

        unsigned int numIndices = 6 * (latitudeDivs - 1) * longitudeDivs
                                + 3 * (bLongEndSplit ? (latitudeDivs + bLatBeginSplit + bLatEndSplit) * 2 : 0)
                                + 3 * (bLatEndSplit ? longitudeDivs : 0)
                                + 3 * (bLatBeginSplit ? longitudeDivs : 0);

        IndexFeed indices(meshOut, numIndices);

        unsigned int currentVert = 0;

        for(unsigned int i = 0; i < latitudeDivs - 1; i++)
        {

          for(unsigned int j = 0; j < longitudeDivs; j++)
          {
            indices.add(currentVert + j);
            indices.add(currentVert + j + numVertsPerLongitude);
            indices.add(currentVert + j + 1 + numVertsPerLongitude);

            indices.add(currentVert + j + 1 + numVertsPerLongitude);
            indices.add(currentVert + j + 1);
            indices.add(currentVert + j);
          }

          currentVert += numVertsPerLongitude;
        }

        currentVert += numVertsPerLongitude;

        if(bLongEndSplit)
        {
          for(unsigned int j = 1; j <= latitudeDivs + bLatBeginSplit + bLatEndSplit; j++)
          {
            indices.add( currentVert );
            indices.add( currentVert + j + 1 );
            indices.add( currentVert + j );
          }

          currentVert += 1 + latitudeDivs + bLatBeginSplit + bLatEndSplit;

          for(unsigned int j = 1; j <= latitudeDivs + bLatBeginSplit + bLatEndSplit; j++)
          {
            indices.add( currentVert );
            indices.add( currentVert + j );
            indices.add( currentVert + j + 1 );
          }

          currentVert += 1 + latitudeDivs + bLatBeginSplit + bLatEndSplit;
        }

        if(bLatEndSplit)
        {

          for(unsigned int j = 1; j <= longitudeDivs; j++)
          {
            indices.add( currentVert );
            indices.add( currentVert + j + 1 );
            indices.add( currentVert + j );
          }
          currentVert += longitudeDivs + 2;
        }

        if(bLatBeginSplit)
        {

          for(unsigned int j = 1; j <= longitudeDivs; j++)
          {
            indices.add( currentVert );
            indices.add( currentVert + j );
            indices.add( currentVert + j + 1 );
          }
        }

        DP_ASSERT( meshOut->checkConsistency() );

        return meshOut;
      }

    } // namespace generator
  } // namespace util
} // namespace dp
