// Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/generator/PreviewScene.h>

#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/Node.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Transform.h>

#include <dp/sg/generator/MeshGenerator.h>

#include <dp/math/math.h>
#include <dp/math/Vecnt.h>

#include <dp/fx/EffectLibrary.h>

#include <dp/util/Array.h>

using namespace dp::math;
using namespace dp::sg::core;

using namespace std;


typedef map<pair<int, int>, int> MapEdgeToIndex;


// Constants used to control the look and layout of the preview scene parts.
// Not all values will work. I spend considerable amount of time to make them work together.
static const float g_closureOuter  = 0.85f; // A value < 1.0 which defines the maximum theta of the main outer section and thereby defines the size of the hole at the top of the main object.
static const float g_radiusInner   = 0.75f; // A value < 1.0 which defines the radius of the inner section. Carefully picked to not let the dent intersect with the inner surface!
static const float g_offsetScale   = 0.75f; // A value < 1.0 which shifts the inner main section upwards (depending on the inner radius) to have a thinner bevel and thicker bottom.
static const float g_thicknessBowl = 0.7f;  // A value < 1.0 which scales the radius of the bowl smaller to give the inner half sphere, this defines the edge thickness of the bowl.
static const float g_gapScale      = 0.975f; // Scaling between adjacent radii to prevent touching geometry.
static const float g_dentAngle     = degToRad(22.5f); // Defines the size of the dent in the main object. Note the dependence to g_radiusInner!


static void calculateTextureCoordinates( const vector< Vec3f >& vertices, vector< float >& texcoords )
{
  // Calculate the texture v-coordinate by wrapping the range [0, 1] around the length of the outline.
  size_t size = vertices.size();
  float len = 0.0f;
  for (size_t i = 0; i < size - 1; ++i)
  {
    len += length(vertices[i + 1] - vertices[i]);
  }
  // Now again for the wrapping of the v-coordinate:
  float texV = 0.0f;
  for (size_t i = 0; i < size - 1; ++i)
  {
    texcoords.push_back( texV );
    texV += length(vertices[i + 1] - vertices[i]) / len;
  }
  texcoords.push_back( 1.0f );
}


static void lathe( int m, float angle, const Vec3f& offset,
                   const vector< Vec3f >& verticesIn,
                   const vector< Vec3f >& tangentsIn,
                   const vector< Vec3f >& binormalsIn,
                   const vector< Vec3f >& normalsIn,
                   const vector< float >& texcoordsIn,
                   vector< Vec3f >& verticesOut,
                   vector< Vec3f >& tangentsOut,
                   vector< Vec3f >& binormalsOut,
                   vector< Vec3f >& normalsOut,
                   vector< Vec2f >& texcoordsOut )
{
  float phiStep = angle / (float) m;
  float texUStep = 1.0f / (float) m;

  for ( size_t i = 0; i < verticesIn.size(); i++ )
  {
    // Copy the input data locally to ease adressing of the components in the inner loop.
    Vec3f v( verticesIn[i] + offset );
    Vec3f t( tangentsIn[i] );
    Vec3f b( binormalsIn[i] );
    Vec3f n( normalsIn[i] );
    float texV( texcoordsIn[i] );

    for ( int longitude = 0; longitude <= m; longitude++ )
    {
      float phi = (float) longitude * phiStep;
      float sinPhi = sin( phi );
      float cosPhi = cos( phi );

      float texU = (float) longitude * texUStep;

      verticesOut.push_back(  Vec3f( cosPhi * v[0] + sinPhi * v[2], v[1], -sinPhi * v[0] + cosPhi * v[2] ) );
      tangentsOut.push_back(  Vec3f( cosPhi * t[0] + sinPhi * t[2], t[1], -sinPhi * t[0] + cosPhi * t[2] ) );
      binormalsOut.push_back( Vec3f( cosPhi * b[0] + sinPhi * b[2], b[1], -sinPhi * b[0] + cosPhi * b[2] ) );
      normalsOut.push_back(   Vec3f( cosPhi * n[0] + sinPhi * n[2], n[1], -sinPhi * n[0] + cosPhi * n[2] ) );
      texcoordsOut.push_back(  Vec2f( texU, texV ) );
    }
  }
}


// Main object, a spherical bowl with a hole at the top with perfect round bevel.
// When offset is != 0 the bowl is thinner at the upper edge than at the bottom
// Parameters:
// n Number of section points on the very outside section.
//   The number of points on the bevel and inner section are determined from that
//   depending on the section length to result in even spacing.
// closureOuter A value < 1.0 defining the maximum theta angle of the outer section.
//              The smaller the bigger the hole at the top. 0.85f is a good value.
static void createMainObject( int nOuter,
                              vector< Vec3f >& vertices,
                              vector< Vec3f >& tangents,
                              vector< Vec3f >& binormals,
                              vector< Vec3f >& normals,
                              vector< float >& texcoords )
{
  float angleOuter = PI * g_closureOuter; // Maximum theta for the outer sphere arc.
  float thetaStepOuter = angleOuter / (float) (nOuter - 1);

  float offsetInner = (1.0f - g_radiusInner) * g_offsetScale;

  // Calculate the maximum theta angle of the inner sphere to allow a 180 degree bevel at the upper hole.
  Vec2f pOuter( sin(angleOuter), -cos(angleOuter) );
  Vec2f v = pOuter - Vec2f( 0.0f, offsetInner );
  normalize( v ); // Normalized vector from inner arc center to outer arc bevel edge.

  // The angle between the x-axis and this vector is:
  float angleInner = atan(v[1] / v[0]) + PI_HALF;  // max theta for the inner sphere.
  Vec2f pInner = Vec2f( sin(angleInner) * g_radiusInner,
                       -cos(angleInner) * g_radiusInner + offsetInner );
  int nInner = nOuter;
  float thetaStepInner = angleInner / (float) ( nInner - 1 );

  // Center of th bevel half circle.
  Vec2f centerBevel = ( pInner + pOuter ) * 0.5f;
  float radiusBevel = length( pOuter - pInner ) * 0.5f;
  int nBevel = nOuter / 2; // The length of the bevel is a lot less than the rest of th emain object, use fewer points.
  float thetaStepBevel = (dp::math::PI - (angleOuter - angleInner)) / (float) ( nBevel - 1 ); // Due to the offsetInner, the radius of the bevel is smaller than 180 degrees.

  // OUTER SPHERE ARC
  // Start points of the latitudinal rings at phi == 0 which are going to be lathed in a second step to generate the volume.
  // Starting at the south pole going upwards.
  for ( int latitude = 0; latitude < nOuter; latitude++ )
  {
    float theta = (float) latitude * thetaStepOuter;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    vertices.push_back( Vec3f(sinTheta, -cosTheta, 0.0f) ); // - y to start at the south pole.
    tangents.push_back( Vec3f(0.0f, 0.0f, -1.0f) );
    binormals.push_back( Vec3f(cosTheta, sinTheta, 0.0f) );
    normals.push_back( Vec3f(sinTheta, -cosTheta, 0.0f) );
  }

  // BEVEL
  // Neither generate the first nor the last ring, because those are part of the sphere arcs' latitudinal rings.
  for ( int latitude = 1; latitude < nBevel - 1; latitude++ ) // theta angle.
  {
    float theta = angleOuter + (float) latitude * thetaStepBevel;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    vertices.push_back( Vec3f( centerBevel[0] + sinTheta * radiusBevel,
                               centerBevel[1] - cosTheta * radiusBevel,
                               0.0f ) );
    tangents.push_back( Vec3f(0.0f, 0.0f, -1.0f) );
    binormals.push_back( Vec3f(cosTheta, sinTheta, 0.0f) );
    normals.push_back( Vec3f(sinTheta, -cosTheta, 0.0f) );
  }

  // INNER SPHERE ARC
  for ( int latitude = nInner - 1; latitude >= 0; latitude-- ) // from top down
  {
    float theta = (float) latitude * thetaStepInner;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    vertices.push_back( Vec3f( sinTheta * g_radiusInner,
                              -cosTheta * g_radiusInner + offsetInner,
                               0.0f) );
    // Normal space is flipped to the inside on this surface!
    // Rotate 180 degrees about tangent, means binormals and normals negated.
    tangents.push_back( Vec3f( 0.0f, 0.0f, -1.0f ) );
    binormals.push_back( Vec3f( -cosTheta, -sinTheta, 0.0f ) );
    normals.push_back( Vec3f( -sinTheta, cosTheta, 0.0f ) );
  }

  calculateTextureCoordinates(vertices, texcoords);
}


// Macro to ease the lerping and duplication of the vertices along the outer circle of the dent.
#define INTERPOLATE(from, to) \
  float a = -d[from] / (d[to] - d[from]); \
  DP_ASSERT(0.0f <= a && a <= 1.0f); \
  Vec3f V = lerp( a, v[from], v[to] ); \
  Vec3f T = lerp( a, t[from], t[to] ); \
  Vec3f B = lerp( a, b[from], b[to] ); \
  Vec3f N = lerp( a, n[from], n[to] ); \
  Vec2f UV = lerp( a, uv[from], uv[to] ); \
  vertices.push_back( V ); \
  vertices.push_back( V ); \
  tangents.push_back( T ); \
  tangents.push_back( T ); \
  binormals.push_back( B ); \
  binormals.push_back( B ); \
  normals.push_back( N ); \
  normals.push_back( N ); \
  texcoords.push_back( UV ); \
  texcoords.push_back( UV );

// Marching Quads algorithm to re-tessellate the dent's outer circle onto the main object's outer sphere.
// This results in a perfectly sharp watertight crease.
static void tessellateDent( int i0, int i1, int i2, int i3,
                            vector<unsigned int>& indices,
                            vector<Vec3f>& vertices,
                            vector<Vec3f>& tangents,
                            vector<Vec3f>& binormals,
                            vector<Vec3f>& normals,
                            vector<Vec2f>& texcoords,
                            MapEdgeToIndex& mapEdgeToIndex,
                            std::set<int>& mirrorVertices,
                            std::set<int>& mirrorAttributes )
{
  Vec3f v[4] = { vertices[i0],   // lower left
                 vertices[i1],   // lower right
                 vertices[i2],   // upper right
                 vertices[i3] }; // upper left

  Vec3f t[4] = { tangents[i0],
                 tangents[i1],
                 tangents[i2],
                 tangents[i3] };

  Vec3f b[4] = { binormals[i0],
                 binormals[i1],
                 binormals[i2],
                 binormals[i3] };

  Vec3f n[4] = { normals[i0],
                 normals[i1],
                 normals[i2],
                 normals[i3] };

  Vec2f uv[4] = { texcoords[i0],
                  texcoords[i1],
                  texcoords[i2],
                  texcoords[i3] };


  float d[4]; // distance values needed twice.

  float threshold = (g_radiusInner + 1.0f) * 0.5f; // Only need to consider these vertices because
  DP_ASSERT( threshold < cos(g_dentAngle) ); // Make sure the threshold test is

  float radius = sin(g_dentAngle);

  int bits = 0; // The marching quad case, 0 - 15
  int mask = 1;
  for (int i = 0; i < 4; ++i)
  {
    // Need the distance to the cylinder along x for inside and outside vertices to calculate the interpolant.
    d[i] = sqrtf( v[i][1] * v[i][1] + v[i][2] * v[i][2] ) - radius; // negative is inside, positive is outside (distance field).

    // Limit the vertices looked at to the outside main sphere with this simple threshold check.
    if (threshold < v[i][0] && d[i] <= 0.0f ) // "Equal" is included because that needs a new normal.
    {
      bits |= mask; // vertex inside.
    }
    mask <<= 1;
  }

  // Indices of newly generated vertices.
  int i4; // on i0, i1 edge
  int i5; // on i1, i2 edge
  int i6; // on i2, i3 edge
  int i7; // on i3, i0 edge

  pair<int, int> edge01(i0, i1);
  pair<int, int> edge12(i1, i2);
  pair<int, int> edge23(i2, i3);
  pair<int, int> edge30(i3, i0);

  MapEdgeToIndex::const_iterator it;

  switch (bits)
  {
  case 0: // all ouside
    indices.push_back(i0);
    indices.push_back(i1);
    indices.push_back(i2);

    indices.push_back(i2);
    indices.push_back(i3);
    indices.push_back(i0);
    break;

  case 1:
    {
      mirrorVertices.insert( i0 );

      it = mapEdgeToIndex.find( edge01 );
      if (it == mapEdgeToIndex.end())
      {
        i4 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(0, 1);
        mirrorAttributes.insert( i4 + 1 ); // Only the normal space is mirrored to keep the mesh gap free.
        mapEdgeToIndex[edge01] = i4;
      }
      else
      {
        i4 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge30 );
      if (it == mapEdgeToIndex.end())
      {
        i7 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(0, 3);
        mirrorAttributes.insert( i7 + 1 ); // Only the normal space is mirrored to keep the mehs gap free.
        mapEdgeToIndex[edge30] = i7;
      }
      else
      {
        i7 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i4 + 1);
      indices.push_back(i7 + 1);

      indices.push_back(i4);
      indices.push_back(i1);
      indices.push_back(i2);

      indices.push_back(i2);
      indices.push_back(i3);
      indices.push_back(i7);

      indices.push_back(i7);
      indices.push_back(i4);
      indices.push_back(i2);
    }
    break;

  case 2:
    {
      mirrorVertices.insert( i1 );

      it = mapEdgeToIndex.find( edge01 );
      if (it == mapEdgeToIndex.end())
      {
        i4 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(1, 0);
        mirrorAttributes.insert( i4 + 1 );
        mapEdgeToIndex[edge01] = i4;
      }
      else
      {
        i4 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge12 );
      if (it == mapEdgeToIndex.end())
      {
        i5 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(1, 2);
        mirrorAttributes.insert( i5 + 1 );
        mapEdgeToIndex[edge12] = i5;
      }
      else
      {
        i5 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i4);
      indices.push_back(i3);

      indices.push_back(i4 + 1);
      indices.push_back(i1);
      indices.push_back(i5 + 1);

      indices.push_back(i5);
      indices.push_back(i2);
      indices.push_back(i3);

      indices.push_back(i3);
      indices.push_back(i4);
      indices.push_back(i5);
    }
    break;

  case 3:
    {
      mirrorVertices.insert( i0 );
      mirrorVertices.insert( i1 );

      it = mapEdgeToIndex.find( edge12 );
      if (it == mapEdgeToIndex.end())
      {
        i5 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(1, 2);
        mirrorAttributes.insert( i5 + 1 );
        mapEdgeToIndex[edge12] = i5;
      }
      else
      {
        i5 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge30 );
      if (it == mapEdgeToIndex.end())
      {
        i7 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(0, 3);
        mirrorAttributes.insert( i7 + 1 );
        mapEdgeToIndex[edge30] = i7;
      }
      else
      {
        i7 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i1);
      indices.push_back(i5 + 1);

      indices.push_back(i5 + 1);
      indices.push_back(i7 + 1);
      indices.push_back(i0);

      indices.push_back(i2);
      indices.push_back(i3);
      indices.push_back(i7);

      indices.push_back(i7);
      indices.push_back(i5);
      indices.push_back(i2);
    }
    break;

  case 4:
    {
      mirrorVertices.insert( i2 );

      it = mapEdgeToIndex.find( edge12 );
      if (it == mapEdgeToIndex.end())
      {
        i5 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(2, 1);
        mirrorAttributes.insert( i5 + 1 );
        mapEdgeToIndex[edge12] = i5;
      }
      else
      {
        i5 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge23 );
      if (it == mapEdgeToIndex.end())
      {
        i6 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(2, 3);
        mirrorAttributes.insert( i6 + 1 );
        mapEdgeToIndex[edge23] = i6;
      }
      else
      {
        i6 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i1);
      indices.push_back(i5);

      indices.push_back(i5 + 1);
      indices.push_back(i2);
      indices.push_back(i6 + 1);

      indices.push_back(i6);
      indices.push_back(i3);
      indices.push_back(i0);

      indices.push_back(i0);
      indices.push_back(i5);
      indices.push_back(i6);
    }
    break;

  case 5: // Doesn't happen with the usual small grid size.
    {
      mirrorVertices.insert( i0 );
      mirrorVertices.insert( i2 );

      it = mapEdgeToIndex.find( edge01 );
      if (it == mapEdgeToIndex.end())
      {
        i4 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(0, 1);
        mirrorAttributes.insert( i4 + 1 );
        mapEdgeToIndex[edge01] = i4;
      }
      else
      {
        i4 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge30 );
      if (it == mapEdgeToIndex.end())
      {
        i7 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(0, 3);
        mirrorAttributes.insert( i7 + 1 );
        mapEdgeToIndex[edge30] = i7;
      }
      else
      {
        i7 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge12 );
      if (it == mapEdgeToIndex.end())
      {
        i5 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(2, 1);
        mirrorAttributes.insert( i5 + 1 );
        mapEdgeToIndex[edge12] = i5;
      }
      else
      {
        i5 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge23 );
      if (it == mapEdgeToIndex.end())
      {
        i6 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(2, 3);
        mirrorAttributes.insert( i6 + 1 );
        mapEdgeToIndex[edge23] = i6;
      }
      else
      {
        i6 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i4 + 1);
      indices.push_back(i7 + 1);

      indices.push_back(i4);
      indices.push_back(i1);
      indices.push_back(i5);

      indices.push_back(i5);
      indices.push_back(i7);
      indices.push_back(i4);

      indices.push_back(i5 + 1);
      indices.push_back(i2);
      indices.push_back(i6 + 1);

      indices.push_back(i6);
      indices.push_back(i3);
      indices.push_back(i7);

      indices.push_back(i7);
      indices.push_back(i5);
      indices.push_back(i6);
    }
    break;

  case 6:
    {
      mirrorVertices.insert( i1 );
      mirrorVertices.insert( i2 );

      it = mapEdgeToIndex.find( edge01 );
      if (it == mapEdgeToIndex.end())
      {
        i4 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(1, 0);
        mirrorAttributes.insert( i4 + 1 );

        mapEdgeToIndex[edge01] = i4;
      }
      else
      {
        i4 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge23 );
      if (it == mapEdgeToIndex.end())
      {
        i6 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(2, 3);
        mirrorAttributes.insert( i6 + 1 );
        mapEdgeToIndex[edge23] = i6;
      }
      else
      {
        i6 = (*it).second;
      }

      indices.push_back(i1);
      indices.push_back(i2);
      indices.push_back(i6 + 1);

      indices.push_back(i6 + 1);
      indices.push_back(i4 + 1);
      indices.push_back(i1);

      indices.push_back(i6);
      indices.push_back(i3);
      indices.push_back(i4);

      indices.push_back(i3);
      indices.push_back(i0);
      indices.push_back(i4);
    }
    break;

  case 7:
    {
      mirrorVertices.insert( i0 );
      mirrorVertices.insert( i1 );
      mirrorVertices.insert( i2 );

      it = mapEdgeToIndex.find( edge23 );
      if (it == mapEdgeToIndex.end())
      {
        i6 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(2, 3);
        mirrorAttributes.insert( i6 + 1 );
        mapEdgeToIndex[edge23] = i6;
      }
      else
      {
        i6 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge30 );
      if (it == mapEdgeToIndex.end())
      {
        i7 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(0, 3);
        mirrorAttributes.insert( i7 + 1 );
        mapEdgeToIndex[edge30] = i7;
      }
      else
      {
        i7 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i1);
      indices.push_back(i7 + 1);

      indices.push_back(i1);
      indices.push_back(i2);
      indices.push_back(i6 + 1);

      indices.push_back(i6);
      indices.push_back(i3);
      indices.push_back(i7);

      indices.push_back(i1);
      indices.push_back(i6 + 1);
      indices.push_back(i7 + 1);
    }
    break;

  case 8:
    {
      mirrorVertices.insert( i3 );

      it = mapEdgeToIndex.find( edge23 );
      if (it == mapEdgeToIndex.end())
      {
        i6 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(3, 2);
        mirrorAttributes.insert( i6 + 1 );
        mapEdgeToIndex[edge23] = i6;
      }
      else
      {
        i6 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge30 );
      if (it == mapEdgeToIndex.end())
      {
        i7 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(3, 0);
        mirrorAttributes.insert( i7 + 1 );
        mapEdgeToIndex[edge30] = i7;
      }
      else
      {
        i7 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i1);
      indices.push_back(i7);

      indices.push_back(i1);
      indices.push_back(i2);
      indices.push_back(i6);

      indices.push_back(i6 + 1);
      indices.push_back(i3);
      indices.push_back(i7 + 1);

      indices.push_back(i1);
      indices.push_back(i6);
      indices.push_back(i7);
    }
    break;

  case 9:
    {
      mirrorVertices.insert( i0 );
      mirrorVertices.insert( i3 );

      it = mapEdgeToIndex.find( edge12 );
      if (it == mapEdgeToIndex.end())
      {
        i4 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(0, 1);
        mirrorAttributes.insert( i4 + 1 );
        mapEdgeToIndex[edge12] = i4;
      }
      else
      {
        i4 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge23 );
      if (it == mapEdgeToIndex.end())
      {
        i6 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(3, 2);
        mirrorAttributes.insert( i6 + 1 );
        mapEdgeToIndex[edge23] = i6;
      }
      else
      {
        i6 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i4 + 1);
      indices.push_back(i3);

      indices.push_back(i4 + 1);
      indices.push_back(i6 + 1);
      indices.push_back(i3);

      indices.push_back(i4);
      indices.push_back(i1);
      indices.push_back(i6);

      indices.push_back(i1);
      indices.push_back(i2);
      indices.push_back(i6);
    }
    break;

  case 10: // Doesn't happen with the usual small grid size.
    {
      mirrorVertices.insert( i1 );
      mirrorVertices.insert( i3 );

      it = mapEdgeToIndex.find( edge01 );
      if (it == mapEdgeToIndex.end())
      {
        i4 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(1, 0);
        mirrorAttributes.insert( i4 + 1 );
        mapEdgeToIndex[edge01] = i4;
      }
      else
      {
        i4 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge12 );
      if (it == mapEdgeToIndex.end())
      {
        i5 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(1, 2);
        mirrorAttributes.insert( i5 + 1 );
        mapEdgeToIndex[edge12] = i5;
      }
      else
      {
        i5 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge23 );
      if (it == mapEdgeToIndex.end())
      {
        i6 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(3, 2);
        mirrorAttributes.insert( i6 + 1 );
        mapEdgeToIndex[edge23] = i6;
      }
      else
      {
        i6 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge30 );
      if (it == mapEdgeToIndex.end())
      {
        i7 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(3, 0);
        mirrorAttributes.insert( i7 + 1 );
        mapEdgeToIndex[edge30] = i7;
      }
      else
      {
        i7 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i4);
      indices.push_back(i7);

      indices.push_back(i4 + 1);
      indices.push_back(i1);
      indices.push_back(i5 + 1);

      indices.push_back(i5);
      indices.push_back(i7);
      indices.push_back(i4);

      indices.push_back(i5);
      indices.push_back(i2);
      indices.push_back(i6);

      indices.push_back(i6 + 1);
      indices.push_back(i3);
      indices.push_back(i7 + 1);

      indices.push_back(i7);
      indices.push_back(i5);
      indices.push_back(i6);
    }
    break;

  case 11:
    {
      mirrorVertices.insert( i0 );
      mirrorVertices.insert( i1 );
      mirrorVertices.insert( i3 );

      it = mapEdgeToIndex.find( edge12 );
      if (it == mapEdgeToIndex.end())
      {
        i5 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(1, 2);
        mirrorAttributes.insert( i5 + 1 );
        mapEdgeToIndex[edge12] = i5;
      }
      else
      {
        i5 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge23 );
      if (it == mapEdgeToIndex.end())
      {
        i6 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(3, 2);
        mirrorAttributes.insert( i6 + 1 );
        mapEdgeToIndex[edge23] = i6;
      }
      else
      {
        i6 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i1);
      indices.push_back(i5 + 1);

      indices.push_back(i5);
      indices.push_back(i2);
      indices.push_back(i6);

      indices.push_back(i6 + 1);
      indices.push_back(i3);
      indices.push_back(i0);

      indices.push_back(i0);
      indices.push_back(i5 + 1);
      indices.push_back(i6 + 1);
    }
    break;

  case 12:
    {
      mirrorVertices.insert( i2 );
      mirrorVertices.insert( i3 );

      it = mapEdgeToIndex.find( edge12 );
      if (it == mapEdgeToIndex.end())
      {
        i5 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(2, 1);
        mirrorAttributes.insert( i5 + 1 );
        mapEdgeToIndex[edge12] = i5;
      }
      else
      {
        i5 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge30 );
      if (it == mapEdgeToIndex.end())
      {
        i7 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(3, 0);
        mirrorAttributes.insert( i7 + 1 );
        mapEdgeToIndex[edge30] = i7;
      }
      else
      {
        i7 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i1);
      indices.push_back(i5);

      indices.push_back(i5);
      indices.push_back(i7);
      indices.push_back(i0);

      indices.push_back(i2);
      indices.push_back(i3);
      indices.push_back(i7 + 1);

      indices.push_back(i7 + 1);
      indices.push_back(i5 + 1);
      indices.push_back(i2);
    }
    break;

  case 13:
    {
      mirrorVertices.insert( i0 );
      mirrorVertices.insert( i2 );
      mirrorVertices.insert( i3 );

      it = mapEdgeToIndex.find( edge01 );
      if (it == mapEdgeToIndex.end())
      {
        i4 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(0, 1);
        mirrorAttributes.insert( i4 + 1 );
        mapEdgeToIndex[edge01] = i4;
      }
      else
      {
        i4 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge12 );
      if (it == mapEdgeToIndex.end())
      {
        i5 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(2, 1);
        mirrorAttributes.insert( i5 + 1 );
        mapEdgeToIndex[edge12] = i5;
      }
      else
      {
        i5 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i4 + 1);
      indices.push_back(i3);

      indices.push_back(i4);
      indices.push_back(i1);
      indices.push_back(i5);

      indices.push_back(i5 + 1);
      indices.push_back(i2);
      indices.push_back(i3);

      indices.push_back(i3);
      indices.push_back(i4 + 1);
      indices.push_back(i5 + 1);
    }
    break;

  case 14:
    {
      mirrorVertices.insert( i1 );
      mirrorVertices.insert( i2 );
      mirrorVertices.insert( i3 );

      it = mapEdgeToIndex.find( edge01 );
      if (it == mapEdgeToIndex.end())
      {
        i4 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(1, 0);
        mirrorAttributes.insert( i4 + 1 );
        mapEdgeToIndex[edge01] = i4;
      }
      else
      {
        i4 = (*it).second;
      }

      it = mapEdgeToIndex.find( edge30 );
      if (it == mapEdgeToIndex.end())
      {
        i7 = dp::checked_cast<int>( vertices.size() );
        INTERPOLATE(3, 0);
        mirrorAttributes.insert( i7 + 1 );
        mapEdgeToIndex[edge30] = i7;
      }
      else
      {
        i7 = (*it).second;
      }

      indices.push_back(i0);
      indices.push_back(i4);
      indices.push_back(i7);

      indices.push_back(i4 + 1);
      indices.push_back(i1);
      indices.push_back(i2);

      indices.push_back(i2);
      indices.push_back(i3);
      indices.push_back(i7 + 1);

      indices.push_back(i4 + 1);
      indices.push_back(i2);
      indices.push_back(i7 + 1);
    }
    break;

  case 15: // all inside
    mirrorVertices.insert( i0 );
    mirrorVertices.insert( i1 );
    mirrorVertices.insert( i2 );
    mirrorVertices.insert( i3 );

    indices.push_back(i0);
    indices.push_back(i1);
    indices.push_back(i2);

    indices.push_back(i2);
    indices.push_back(i3);
    indices.push_back(i0);
    break;

  }
}


// m is the number of longitudes along the latitudinal rings of the outer most sphere arc.
// Everything else is dependent on that.
static PrimitiveSharedPtr createPreviewMainObject( int m )
{
  // The input data used to lathe the individual objects.
  vector< Vec3f > verticesIn;
  vector< Vec3f > tangentsIn;
  vector< Vec3f > binormalsIn;
  vector< Vec3f > normalsIn;
  vector< float > texcoordsIn;

  vector< Vec3f > verticesOut;
  vector< Vec3f > tangentsOut;
  vector< Vec3f > binormalsOut;
  vector< Vec3f > normalsOut;
  vector< Vec2f > texcoordsOut;

  // First create the main outer object.
  int nOuter = m / 2;  // Use half the number of longitudes for the number of latitudes.
  createMainObject( nOuter,
                    verticesIn,
                    tangentsIn,
                    binormalsIn,
                    normalsIn,
                    texcoordsIn );

  Vec3f vertexOffset( 0.0f, 1.375f, 0.0f ); // Don't apply during lathing. The dent is applied while the sphere is placed at the origin.

  lathe( m, 2.0f * PI, Vec3f(0.0f, 0.0f, 0.0f),
         verticesIn,  tangentsIn,  binormalsIn,  normalsIn,  texcoordsIn,
         verticesOut, tangentsOut, binormalsOut, normalsOut, texcoordsOut );


  // Calculate indices
  vector<unsigned int> indices;

  // We have generated m + 1 vertices per latitude.
  int columns = m + 1;
  int n = dp::checked_cast<int>( verticesIn.size() ); // The number of latitudinal rings.

  // Track vertex indices created during dent tessellation to reuse them on adjacent cells.
  MapEdgeToIndex mapEdgeToIndex;
  std::set<int> mirrorVertices;   // Set of vertex indices which need to be moved and inversed.
  std::set<int> mirrorAttributes; // Set of indices exactly on the crease where only the normal space needs to be mirrored to not generate gaps.

  for( int latitude = 0; latitude < n - 1; latitude++ )
  {
    for( int longitude = 0; longitude < m; longitude++ )
    {
      int i0 =  latitude      * columns + longitude    ;  // lower left
      int i1 =  latitude      * columns + longitude + 1;  // lower right
      int i2 = (latitude + 1) * columns + longitude + 1;  // upper right
      int i3 = (latitude + 1) * columns + longitude    ;  // upper left

      tessellateDent( i0, i1, i2, i3, indices,
                      verticesOut, tangentsOut, binormalsOut, normalsOut, texcoordsOut,
                      mapEdgeToIndex, mirrorVertices, mirrorAttributes );
    }
  }

  // Now mirror the nornal space on the vertices inside the dent.
  float radius = cos( g_dentAngle );
  std::set<int>::const_iterator it     = mirrorVertices.begin();
  std::set<int>::const_iterator it_end = mirrorVertices.end();
  while (it != it_end)
  {
    int i = *it;

    verticesOut[i][0] = radius - (verticesOut[i][0] - radius);

    tangentsOut[i][0] *= -1.0f;
    tangentsOut[i][1] *= -1.0f;

    binormalsOut[i][0] *= -1.0f;
    binormalsOut[i][2] *= -1.0f;

    normalsOut[i][1] *= -1.0f;
    normalsOut[i][2] *= -1.0f;

    it++;
  }

  it     = mirrorAttributes.begin();
  it_end = mirrorAttributes.end();
  while (it != it_end)
  {
    int i = *it;

    tangentsOut[i][0] *= -1.0f;
    tangentsOut[i][1] *= -1.0f;

    binormalsOut[i][0] *= -1.0f;
    binormalsOut[i][2] *= -1.0f;

    normalsOut[i][1] *= -1.0f;
    normalsOut[i][2] *= -1.0f;

    it++;
  }


  unsigned int numVertices = dp::checked_cast<unsigned int>( verticesOut.size() );

  // Finally place it above the platform. TODO Could use transforms per sub-object.
  for (unsigned int i = 0; i < numVertices; ++i)
  {
    verticesOut[i] += vertexOffset;
  }

  // Create a VertexAttributeSet with vertices, normals and texcoords
  VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
  vasPtr->setVertices( &verticesOut[0], numVertices );
  vasPtr->setNormals( &normalsOut[0], numVertices );
  vasPtr->setTexCoords( 0, &texcoordsOut[0], numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_TANGENT  - VertexAttributeSet::DP_SG_TEXCOORD0, &tangentsOut[0],  numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_BINORMAL - VertexAttributeSet::DP_SG_TEXCOORD0, &binormalsOut[0], numVertices );

  IndexSetSharedPtr indexSet = IndexSet::create();
  indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

  PrimitiveSharedPtr primitivePtr = Primitive::create( PRIMITIVE_TRIANGLES );
  primitivePtr->setVertexAttributeSet( vasPtr );
  primitivePtr->setIndexSet( indexSet );

  return primitivePtr;
}


static void createBowl( int nBowl, /* float radiusInner, float thicknessBowl, */
                        vector< Vec3f >& vertices,
                        vector< Vec3f >& tangents,
                        vector< Vec3f >& binormals,
                        vector< Vec3f >& normals,
                        vector< float >& texcoords )
{
  float angleBowl = PI_HALF; // Maximum theta for the outer sphere arc.
  float radiusBowlOuter = g_radiusInner * g_gapScale;
  float thetaStepBowl = angleBowl / (float) (nBowl - 1);

  float lengthBowlOuter = radiusBowlOuter * angleBowl;
  float gridSize = lengthBowlOuter / (float) nBowl;

  float x0 =  sin(angleBowl) * radiusBowlOuter;
  float y0 = -cos(angleBowl) * radiusBowlOuter;

  float radiusBowlInner = radiusBowlOuter * g_thicknessBowl; // scales smaller

  float x1 = sin(angleBowl) * radiusBowlInner;

  float lengthBowlEdge = x0 - x1; // outer edge x - inner edge x. Flat top because both sphere arcs are exactly 90 degrees.
  int nBowlEdge = (int) ceil(lengthBowlEdge / gridSize);
  float stepBowlEdge = lengthBowlEdge / (float) (nBowlEdge - 1);

  // Starting at the south pole going upwards.
  for ( int latitude = 0; latitude < nBowl; latitude++ )
  {
    float theta = (float) latitude * thetaStepBowl;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    vertices.push_back( Vec3f( sinTheta * radiusBowlOuter,
                              -cosTheta * radiusBowlOuter,
                               0.0f) );
    tangents.push_back( Vec3f(0.0f, 0.0f, -1.0f) );
    binormals.push_back( Vec3f(cosTheta, sinTheta, 0.0f) );
    normals.push_back( Vec3f(sinTheta, -cosTheta, 0.0f) );
  }

  // Special case the outside point of the edge as bevel point with normal pointing in 45 degrees to the upper right.
  vertices.push_back( Vec3f( x0 - gridSize * 0.3f, y0 + gridSize * 0.7f, 0.0f ) );
  tangents.push_back( Vec3f( 0.0f, 0.0f, -1.0f ) );
  binormals.push_back( Vec3f( -sin(PI_QUARTER), cos(PI_QUARTER), 0.0f) );
  normals.push_back( Vec3f( cos(PI_QUARTER), sin(PI_QUARTER), 0.0f) );

  for (int latitude = 1; latitude < nBowlEdge - 1; latitude++) // flat top
  {
    float x = (float) latitude * stepBowlEdge;
    vertices.push_back( Vec3f( x0 - x, y0 + gridSize, 0.0f ) ); // Push this a little upwards to make room for the bevel points
    tangents.push_back( Vec3f(0.0f, 0.0f, -1.0f) );
    binormals.push_back( Vec3f(-1.0f, 0.0, 0.0f) );
    normals.push_back( Vec3f(0.0f, 1.0f, 0.0f) );
  }

  // Special case the inside point of the edge as bevel point with normal pointing in 45 degrees to the upper left.
  vertices.push_back( Vec3f( x1 + gridSize * 0.3f , y0 + gridSize * 0.7f, 0.0f ) );
  tangents.push_back( Vec3f( 0.0f, 0.0f, -1.0f ) );
  binormals.push_back( Vec3f( -sin(PI_QUARTER), -cos(PI_QUARTER), 0.0f) );
  normals.push_back( Vec3f( -cos(PI_QUARTER), sin(PI_QUARTER), 0.0f) );

  for ( int latitude = nBowl - 1; latitude >= 0; latitude-- )
  {
    float theta = (float) latitude * thetaStepBowl;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    vertices.push_back( Vec3f( sinTheta * radiusBowlInner,
                              -cosTheta * radiusBowlInner,
                               0.0f) );
    // Normal space is flipped to the inside on this surface!
    // Rotate 180 degrees about tangent, means binormals and normals negated.
    tangents.push_back( Vec3f( 0.0f, 0.0f, -1.0f ) );
    binormals.push_back( Vec3f( -cosTheta, -sinTheta, 0.0f ) );
    normals.push_back( Vec3f( -sinTheta, cosTheta, 0.0f ) );
  }

  calculateTextureCoordinates( vertices, texcoords );
}

// m is the number of points along the latitudinal rings of the outer most sphere arc.
// Everything else is dependent on that.
static PrimitiveSharedPtr createPreviewBowl( int m )
{
  // The input data used to lathe the individual objects.
  vector< Vec3f > verticesIn;
  vector< Vec3f > tangentsIn;
  vector< Vec3f > binormalsIn;
  vector< Vec3f > normalsIn;
  vector< float > texcoordsIn;

  vector< Vec3f > verticesOut;
  vector< Vec3f > tangentsOut;
  vector< Vec3f > binormalsOut;
  vector< Vec3f > normalsOut;
  vector< Vec2f > texcoordsOut;

  // First create the main outer object.

  createBowl( m / 2, // nBowl
              verticesIn,
              tangentsIn,
              binormalsIn,
              normalsIn,
              texcoordsIn );

  float offsetInner = (1.0f - g_radiusInner) * g_offsetScale;

  lathe( m, 2.0f * PI, Vec3f( 0.0f, 1.375f + offsetInner, 0.0f ),
         verticesIn,  tangentsIn,  binormalsIn,  normalsIn,  texcoordsIn,
         verticesOut, tangentsOut, binormalsOut, normalsOut, texcoordsOut );

  vector<unsigned int> indices;

  // Calculate indices

  // We have generated m + 1 vertices per latitude.
  int columns = m + 1;
  int n = dp::checked_cast<int>( verticesIn.size() ); // The number of latitudinal rings.

  for( int latitude = 0; latitude < n - 1; latitude++ )
  {
    for( int longitude = 0; longitude < m; longitude++ )
    {
      indices.push_back(  latitude      * columns + longitude     );  // lower left
      indices.push_back(  latitude      * columns + longitude + 1 );  // lower right
      indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right

      indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right
      indices.push_back( (latitude + 1) * columns + longitude     );  // upper left
      indices.push_back(  latitude      * columns + longitude     );  // lower left
    }
  }

  unsigned int numVertices = dp::checked_cast<unsigned int>( verticesOut.size() );

  // Create a VertexAttributeSet with vertices, normals and texcoords
  VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
  vasPtr->setVertices( &verticesOut[0], numVertices );
  vasPtr->setNormals( &normalsOut[0], numVertices );
  vasPtr->setTexCoords( 0, &texcoordsOut[0], numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_TANGENT  - VertexAttributeSet::DP_SG_TEXCOORD0, &tangentsOut[0],  numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_BINORMAL - VertexAttributeSet::DP_SG_TEXCOORD0, &binormalsOut[0], numVertices );

  IndexSetSharedPtr indexSet( IndexSet::create() );
  indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

  // create pointer to return
  PrimitiveSharedPtr primitivePtr = Primitive::create( PRIMITIVE_TRIANGLES );
  primitivePtr->setVertexAttributeSet( vasPtr );
  primitivePtr->setIndexSet( indexSet );

  return primitivePtr;
}


static void createCenterSphere( int nSphere, float radiusSphere,
                                vector< Vec3f >& vertices,
                                vector< Vec3f >& tangents,
                                vector< Vec3f >& binormals,
                                vector< Vec3f >& normals,
                                vector< float >& texcoords )
{
  float angleSphere = (float) PI; // Maximum theta for the outer sphere arc.
  float thetaStepSphere = angleSphere / (float) (nSphere - 1);

  float texVStep = 1.0f / (float) (nSphere - 1);  // Texture v coordinate from 0.0 at the south pole to 1.0 at the north pole.

  // Starting at the south pole going upwards.
  for ( int latitude = 0; latitude < nSphere; latitude++ )
  {
    float theta = (float) latitude * thetaStepSphere;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    float texV = (float) latitude * texVStep;

    vertices.push_back( Vec3f( sinTheta * radiusSphere,
                              -cosTheta * radiusSphere,
                               0.0f) );
    tangents.push_back( Vec3f(0.0f, 0.0f, -1.0f) );
    binormals.push_back( Vec3f(cosTheta, sinTheta, 0.0f) );
    normals.push_back( Vec3f(sinTheta, -cosTheta, 0.0f) );
    texcoords.push_back( texV );
  }
}


// m is the number of points along the latitudinal rings of the outer most sphere arc.
// Everything else is dependent on that.
static PrimitiveSharedPtr createPreviewSphere( int m )
{
  // The input data used to lathe the individual objects.
  vector< Vec3f > verticesIn;
  vector< Vec3f > tangentsIn;
  vector< Vec3f > binormalsIn;
  vector< Vec3f > normalsIn;
  vector< float > texcoordsIn;

  vector< Vec3f > verticesOut;
  vector< Vec3f > tangentsOut;
  vector< Vec3f > binormalsOut;
  vector< Vec3f > normalsOut;
  vector< Vec2f > texcoordsOut;

  // First create the main outer object.
  int nSphere = m / 2; // The number of latitudinal rings on the very outer arc. The returned vertexIn.size() is bigger!

  float radiusBowl   = g_radiusInner * g_gapScale * g_thicknessBowl; // Smaller than the inner sphere
  float radiusSphere = radiusBowl * 0.85f;

  float offsetInner = (1.0f - g_radiusInner) * g_offsetScale;
  float offsetSphere = offsetInner - (radiusBowl - radiusSphere) * 0.9f;

  createCenterSphere( nSphere, radiusSphere,
                      verticesIn,
                      tangentsIn,
                      binormalsIn,
                      normalsIn,
                      texcoordsIn );

  lathe( m, 2.0f * PI, Vec3f( 0.0f, 1.375f + offsetSphere, 0.0f ),
         verticesIn,  tangentsIn,  binormalsIn,  normalsIn,  texcoordsIn,
         verticesOut, tangentsOut, binormalsOut, normalsOut, texcoordsOut );

  vector<unsigned int> indices;

  // Calculate indices

  // We have generated m + 1 vertices per latitude.
  int columns = m + 1;
  int n = dp::checked_cast<int>( verticesIn.size() ); // The number of latitudinal rings.

  for( int latitude = 0; latitude < n - 1; latitude++ )
  {
    for( int longitude = 0; longitude < m; longitude++ )
    {
      indices.push_back(  latitude      * columns + longitude     );  // lower left
      indices.push_back(  latitude      * columns + longitude + 1 );  // lower right
      indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right

      indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right
      indices.push_back( (latitude + 1) * columns + longitude     );  // upper left
      indices.push_back(  latitude      * columns + longitude     );  // lower left
    }
  }

  unsigned int numVertices = dp::checked_cast<unsigned int>( verticesOut.size() );

  // Create a VertexAttributeSet with vertices, normals and texcoords
  VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
  vasPtr->setVertices( &verticesOut[0], numVertices );
  vasPtr->setNormals( &normalsOut[0], numVertices );
  vasPtr->setTexCoords( 0, &texcoordsOut[0], numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_TANGENT  - VertexAttributeSet::DP_SG_TEXCOORD0, &tangentsOut[0],  numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_BINORMAL - VertexAttributeSet::DP_SG_TEXCOORD0, &binormalsOut[0], numVertices );

  IndexSetSharedPtr indexSet( IndexSet::create() );
  indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

  // create pointer to return
  PrimitiveSharedPtr primitivePtr = Primitive::create( PRIMITIVE_TRIANGLES );
  primitivePtr->setVertexAttributeSet( vasPtr );
  primitivePtr->setIndexSet( indexSet );

  return primitivePtr;
}



// The profile is just a number of straight lines with bevel edges at the corners.
static void createPlatform( vector< Vec3f >& vertices,
                            vector< Vec3f >& tangents,
                            vector< Vec3f >& binormals,
                            vector< Vec3f >& normals,
                            vector< float >& texcoords )
{
  // Predefine the eight normals.

  float sc = sin(PI_QUARTER); // == cos(PI_QUARTER)

  Vec3f nR ( 1.0f,  0.0f, 0.0f); // right
  Vec3f nUR(   sc,    sc, 0.0f); // upper right
  Vec3f nU ( 0.0f,  1.0f, 0.0f); // up
  Vec3f nUL(  -sc,    sc, 0.0f); // upper left
  Vec3f nL (-1.0f,  0.0f, 0.0f); // left
  Vec3f nBL(  -sc,   -sc, 0.0f); // bottom left
  Vec3f nB ( 0.0f, -1.0f, 0.0f); // bottom
  Vec3f nBR(   sc,   -sc, 0.0f); // bottom right

  Vec3f t(0.0f, 0.0f, -1.0f);

  float g = 0.0125f; // offset for the corner surrounding points.
  float fit = 0.85f - 0.0025f;
  float radius = 1.0f;

  vertices.push_back( Vec3f( 0.0f, 0.0f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nR);
  normals.push_back(nB);

  vertices.push_back( Vec3f( 0.5f, 0.0f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nR);
  normals.push_back(nB);

  vertices.push_back( Vec3f( fit, 0.0f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nR);
  normals.push_back(nB);

  vertices.push_back( Vec3f( radius - g, 0.0f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nR);
  normals.push_back(nB);

  vertices.push_back( Vec3f( radius - 0.3f * g, 0.3f * g, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nUR);
  normals.push_back(nBR);

  vertices.push_back( Vec3f( radius, g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nU);
  normals.push_back(nR);

  vertices.push_back( Vec3f( radius, 0.125f - g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nU);
  normals.push_back(nR);

  vertices.push_back( Vec3f( radius - 0.3f * g, 0.125f - 0.3f * g, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nUL);
  normals.push_back(nUR);

  vertices.push_back( Vec3f( radius - g, 0.125f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( fit + g, 0.125f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( fit + 0.3f * g, 0.125f + 0.3f * g, 0.0f ) ); // knee
  tangents.push_back(t);
  binormals.push_back(nUL);
  normals.push_back(nUR);

  vertices.push_back( Vec3f( fit, 0.125f + g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nU);
  normals.push_back(nR);

  vertices.push_back( Vec3f( fit, 0.375f - g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nU);
  normals.push_back(nR);

  vertices.push_back( Vec3f( fit - 0.3f * g, 0.375f - 0.3f * g, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nUL);
  normals.push_back(nUR);

  vertices.push_back( Vec3f( fit - g, 0.375f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( 0.5f, 0.375f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( 0.0f, 0.375f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  calculateTextureCoordinates( vertices, texcoords );
}


// The ring object around the dais of the preview scene is kept more edgy
// to have some sharper creases in the scene and to ease the tessellation of the cut planes.
static PrimitiveSharedPtr createPreviewPlatform( int m )
{
  // The input data used to lathe the individual objects.
  vector< Vec3f > verticesIn;
  vector< Vec3f > tangentsIn;
  vector< Vec3f > binormalsIn;
  vector< Vec3f > normalsIn;
  vector< float > texcoordsIn;

  vector< Vec3f > verticesOut;
  vector< Vec3f > tangentsOut;
  vector< Vec3f > binormalsOut;
  vector< Vec3f > normalsOut;
  vector< Vec2f > texcoordsOut;

  createPlatform( verticesIn, tangentsIn, binormalsIn, normalsIn, texcoordsIn );

  lathe( m, 2.0f * PI, Vec3f( 0.0f, 0.0f, 0.0f ),
         verticesIn,  tangentsIn,  binormalsIn,  normalsIn,  texcoordsIn,
         verticesOut, tangentsOut, binormalsOut, normalsOut, texcoordsOut );

  vector<unsigned int> indices;

  // Calculate indices

  // We have generated m + 1 vertices per latitude.
  int columns = m + 1;
  int n = dp::checked_cast<int>( verticesIn.size() ); // The number of latitudinal rings.

  for( int latitude = 0; latitude < n - 1; latitude++ )
  {
    for( int longitude = 0; longitude < m; longitude++ )
    {
      indices.push_back(  latitude      * columns + longitude     );  // lower left
      indices.push_back(  latitude      * columns + longitude + 1 );  // lower right
      indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right

      indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right
      indices.push_back( (latitude + 1) * columns + longitude     );  // upper left
      indices.push_back(  latitude      * columns + longitude     );  // lower left
    }
  }

  unsigned int numVertices = dp::checked_cast<unsigned int>( verticesOut.size() );

  // Create a VertexAttributeSet with vertices, normals and texcoords
  VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
  vasPtr->setVertices( &verticesOut[0], numVertices );
  vasPtr->setNormals( &normalsOut[0], numVertices );
  vasPtr->setTexCoords( 0, &texcoordsOut[0], numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_TANGENT  - VertexAttributeSet::DP_SG_TEXCOORD0, &tangentsOut[0],  numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_BINORMAL - VertexAttributeSet::DP_SG_TEXCOORD0, &binormalsOut[0], numVertices );

  IndexSetSharedPtr indexSet( IndexSet::create() );
  indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

  // Create a Primitive
  PrimitiveSharedPtr primitivePtr = Primitive::create( PRIMITIVE_TRIANGLES );
  primitivePtr->setVertexAttributeSet( vasPtr );
  primitivePtr->setIndexSet( indexSet );

  return primitivePtr;
}




// The profile is just a number of straight lines with bevel edges at the corners.
static void createRing( vector< Vec3f >& vertices,
                        vector< Vec3f >& tangents,
                        vector< Vec3f >& binormals,
                        vector< Vec3f >& normals,
                        vector< float >& texcoords )
{
  // Predefine the eight normals.

  float sc = sin(PI_QUARTER); // == cos(PI_QUARTER)

  Vec3f nR ( 1.0f,  0.0f, 0.0f); // right
  Vec3f nUR(   sc,    sc, 0.0f); // upper right
  Vec3f nU ( 0.0f,  1.0f, 0.0f); // up
  Vec3f nUL(  -sc,    sc, 0.0f); // upper left
  Vec3f nL (-1.0f,  0.0f, 0.0f); // left
  Vec3f nBL(  -sc,   -sc, 0.0f); // bottom left
  Vec3f nB ( 0.0f, -1.0f, 0.0f); // bottom
  Vec3f nBR(   sc,   -sc, 0.0f); // bottom right

  Vec3f t(0.0f, 0.0f, -1.0f);

  float d = 0.125f; // Grid for the tesselation of the cutplane
  float g = 0.0125f; // offset for the corner surrounding points.

  vertices.push_back( Vec3f( 0.3f * g, 0.3f * g, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nBR);
  normals.push_back(nBL);

  vertices.push_back( Vec3f( 1.0f * g, 0.0f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nR);
  normals.push_back(nB);

  vertices.push_back( Vec3f( 1.0f * d, 0.0f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nR);
  normals.push_back(nB);

  vertices.push_back( Vec3f( 2.0f * d, 0.0f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nR);
  normals.push_back(nB);

  vertices.push_back( Vec3f( 3.0f * d - g, 0.0f, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nR);
  normals.push_back(nB);

  vertices.push_back( Vec3f( 3.0f * d, 0.0f, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nUR);
  normals.push_back(nBR);

  vertices.push_back( Vec3f( 4.0f * d, d, 0.0f ) );    // corner
  tangents.push_back(t);
  binormals.push_back(nUR);
  normals.push_back(nBR);

  vertices.push_back( Vec3f( 4.0f * d, d + g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nU);
  normals.push_back(nR);

  vertices.push_back( Vec3f( 4.0f * d, 2.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nU);
  normals.push_back(nR);

  vertices.push_back( Vec3f( 4.0f * d, 3.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nU);
  normals.push_back(nR);

  vertices.push_back( Vec3f( 4.0f * d, 3.5f * d - g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nU);
  normals.push_back(nR);

  vertices.push_back( Vec3f( 4.0f * d, 3.5f * d, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nUL);
  normals.push_back(nUR);

  vertices.push_back( Vec3f( 3.5f * d, 4.0f * d, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nUL);
  normals.push_back(nUR);

  vertices.push_back( Vec3f( 3.5f * d - g, 4.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( 3.0f * d, 4.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( 2.5f * d + g, 4.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( 2.5f * d, 4.0f * d, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nBL);
  normals.push_back(nUL);

  vertices.push_back( Vec3f( 2.0f * d, 3.5f * d, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nBL);
  normals.push_back(nUL);

  vertices.push_back( Vec3f( 2.0f * d, 3.5f * d - g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nB);
  normals.push_back(nL);

  vertices.push_back( Vec3f( 2.0f * d, 3.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nB);
  normals.push_back(nL);

  vertices.push_back( Vec3f( 2.0f * d, 2.0f * d + g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nB);
  normals.push_back(nL);

  vertices.push_back( Vec3f( 2.0f * d - 0.3f * g, 2.0f * d + 0.3f * g, 0.0f ) ); // knee
  tangents.push_back(t);
  binormals.push_back(nBL);
  normals.push_back(nUL);

  vertices.push_back( Vec3f( 2.0f * d - g, 2.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( 1.0f * d, 2.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( g, 2.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nL);
  normals.push_back(nU);

  vertices.push_back( Vec3f( 0.3f * g, 2.0f * d - 0.3f * g, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nBL);
  normals.push_back(nUL);

  vertices.push_back( Vec3f( 0.0f, 2.0f * d - g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nB);
  normals.push_back(nL);

  vertices.push_back( Vec3f( 0.0f, 1.0f * d, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nB);
  normals.push_back(nL);

  vertices.push_back( Vec3f( 0.0f, g, 0.0f ) );
  tangents.push_back(t);
  binormals.push_back(nB);
  normals.push_back(nL);

  // Close it with the initial corner:
  vertices.push_back( Vec3f( 0.3f * g, 0.3f * g, 0.0f ) ); // corner
  tangents.push_back(t);
  binormals.push_back(nBR);
  normals.push_back(nBL);

  calculateTextureCoordinates( vertices, texcoords );
}


// The ring object around the dais of the preview scene is kept more edgy
// to have some sharper creases in the scene and to ease the tessellation of the cut planes.
static PrimitiveSharedPtr createPreviewRing( int m )
{
  // The input data used to lathe the individual objects.
  vector< Vec3f > verticesIn;
  vector< Vec3f > tangentsIn;
  vector< Vec3f > binormalsIn;
  vector< Vec3f > normalsIn;
  vector< float > texcoordsIn;

  vector< Vec3f > verticesOut;
  vector< Vec3f > tangentsOut;
  vector< Vec3f > binormalsOut;
  vector< Vec3f > normalsOut;
  vector< Vec2f > texcoordsOut;

  createRing( verticesIn, tangentsIn, binormalsIn, normalsIn, texcoordsIn );

  Vec3f offset( 0.85f, 0.125f + 0.0025f, 0.0f);

  lathe( m, 1.5f * PI, offset, // 0 - 270 degrees
         verticesIn,  tangentsIn,  binormalsIn,  normalsIn,  texcoordsIn,
         verticesOut, tangentsOut, binormalsOut, normalsOut, texcoordsOut );

  vector<unsigned int> indices;

  // Calculate indices

  // We have generated m + 1 vertices per latitude.
  int columns = m + 1;
  int n = dp::checked_cast<int>( verticesIn.size() ); // The number of latitudinal rings.

  for( int latitude = 0; latitude < n - 1; latitude++ )
  {
    for( int longitude = 0; longitude < m; longitude++ )
    {
      indices.push_back(  latitude      * columns + longitude     );  // lower left
      indices.push_back(  latitude      * columns + longitude + 1 );  // lower right
      indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right

      indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right
      indices.push_back( (latitude + 1) * columns + longitude     );  // upper left
      indices.push_back(  latitude      * columns + longitude     );  // lower left
    }
  }

  // Cut planes closing the ring object.

  // Get the vertex indices of the five helper points.
  int A = dp::checked_cast<int>( verticesOut.size() );
  int B = A + 1;
  int C = A + 2;
  int D = A + 3;
  int E = A + 4;

  float d = 0.125f; // Grid for the tesselation of the cutplane

  // On the xy-plane
  Vec3f ta0(1.0f, 0.0f, 0.0f);
  Vec3f bi0(0.0f, 1.0f, 0.0f);
  Vec3f no0(0.0f, 0.0f, 1.0f);

  verticesOut.push_back( Vec3f( d, d, 0.0f ) + offset ); // A
  tangentsOut.push_back( ta0 );
  binormalsOut.push_back( bi0 );
  normalsOut.push_back( no0 );
  texcoordsOut.push_back( Vec2f( d, d ) );

  verticesOut.push_back( Vec3f( 2.0f * d, d, 0.0f ) + offset ); // B
  tangentsOut.push_back( ta0 );
  binormalsOut.push_back( bi0 );
  normalsOut.push_back( no0 );
  texcoordsOut.push_back( Vec2f( 2.0f * d, d ) );

  verticesOut.push_back( Vec3f( 3.0f * d, d, 0.0f ) + offset ); // C
  tangentsOut.push_back( ta0 );
  binormalsOut.push_back( bi0 );
  normalsOut.push_back( no0 );
  texcoordsOut.push_back( Vec2f( 3.0f * d, d ) );

  verticesOut.push_back( Vec3f( 3.0f * d, 2.0f * d, 0.0f ) + offset ); // D
  tangentsOut.push_back( ta0 );
  binormalsOut.push_back( bi0 );
  normalsOut.push_back( no0 );
  texcoordsOut.push_back( Vec2f( 3.0f * d, 2.0f * d ) );

  verticesOut.push_back( Vec3f( 3.0f * d, 3.0f * d, 0.0f ) + offset ); // E
  tangentsOut.push_back( ta0 );
  binormalsOut.push_back( bi0 );
  normalsOut.push_back( no0 );
  texcoordsOut.push_back( Vec2f( 3.0f * d, 3.0f * d ) );

  // On the yz-plane
  Vec3f ta1(0.0f, 0.0f, -1.0f);
  Vec3f bi1(0.0f, 1.0f,  0.0f);
  Vec3f no1(1.0f, 0.0f,  0.0f);

  Vec3f offsetYZ = Vec3f(0.0, offset[1], offset[0]); // Swap x- and z-offsets for the yz-cutplane coordinates.

  verticesOut.push_back( Vec3f( 0.0f, d, d ) + offsetYZ ); // A
  tangentsOut.push_back( ta1 );
  binormalsOut.push_back( bi1 );
  normalsOut.push_back( no1 );
  texcoordsOut.push_back( Vec2f( d, d ) );

  verticesOut.push_back( Vec3f( 0.0f, d, 2.0f * d ) + offsetYZ ); // B
  tangentsOut.push_back( ta1 );
  binormalsOut.push_back( bi1 );
  normalsOut.push_back( no1 );
  texcoordsOut.push_back( Vec2f( 2.0f * d, d ) );

  verticesOut.push_back( Vec3f( 0.0f, d, 3.0f * d ) + offsetYZ ); // C
  tangentsOut.push_back( ta1 );
  binormalsOut.push_back( bi1 );
  normalsOut.push_back( no1 );
  texcoordsOut.push_back( Vec2f( 3.0f * d, d ) );

  verticesOut.push_back( Vec3f( 0.0f, 2.0f * d, 3.0f * d ) + offsetYZ ); // D
  tangentsOut.push_back( ta1 );
  binormalsOut.push_back( bi1 );
  normalsOut.push_back( no1 );
  texcoordsOut.push_back( Vec2f( 3.0f * d, 2.0f * d ) );

  verticesOut.push_back( Vec3f( 0.0f, 3.0f * d, 3.0f * d ) + offsetYZ ); // E
  tangentsOut.push_back( ta1 );
  binormalsOut.push_back( bi1 );
  normalsOut.push_back( no1 );
  texcoordsOut.push_back( Vec2f( 3.0f * d, 3.0f * d ) );

  int idxBaseXY = dp::checked_cast<int>( verticesOut.size() );

  // The cut planes need to duplicate the vertices because the normals and texture coordinates must not be shared.
  // Copy the first longitudinal ring in the xy-plane and append it with new attributes.
  for ( int latitude = 0; latitude < n; latitude++ )
  {
    Vec3f v(verticesOut[latitude * columns]);

    verticesOut.push_back( v );
    tangentsOut.push_back( ta0 );
    binormalsOut.push_back( bi0 );
    normalsOut.push_back( no0 );
    texcoordsOut.push_back( Vec2f(v[0] - offset[0], v[1] - offset[1]) );
  }

  int idxBaseYZ = dp::checked_cast<int>( verticesOut.size() );

  // Copy the last longitudinal ring in the yz-plane and append it with new attributes.
  for ( int latitude = 0; latitude < n; latitude++ )
  {
    Vec3f v(verticesOut[latitude * columns + m]);

    verticesOut.push_back( v );
    tangentsOut.push_back( ta1 );
    binormalsOut.push_back( bi1 );
    normalsOut.push_back( no1 );
    texcoordsOut.push_back( Vec2f(v[2] - offsetYZ[2], v[1] - offsetYZ[1]) );
  }

   // Indices in this list need to be multiplied by the colums.
  const static int cutPlane[] =
  {
    0, 1, 29,   // lower left corner
    25, 26, 27, // upper left corner

    A, 29, 1,
    A, 1, 2,
    A, 2, 3,
    A, 3, B,
    A, B, 22,
    A, 22, 23,
    A, 23, 24,
    A, 24, 25,
    A, 25, 27,
    A, 27, 28,
    A, 28, 29,

    B, 3, 4,
    B, 4, 5,
    B, 5, C,
    B, C, D,
    B, D, 21,
    B, 21, 22,

    C, 5, 6,
    C, 6, 7,
    C, 7, 8,
    C, 8, D,

    D, 8, 9,
    D, 9, E,
    D, E, 19,
    D, 19, 20,
    D, 20, 21,

    E, 9, 10,
    E, 10, 11,
    E, 11, 12,
    E, 12, 13,
    E, 13, 14,
    E, 14, 15,
    E, 15, 16,
    E, 16, 17,
    E, 17, 18,
    E, 18, 19
  };

  for (size_t i = 0; i < sizeof(cutPlane) / sizeof(int); i++ )
  {
    int idx = cutPlane[i];
    // Each latitudinal ring has columns vertices in the array,
    // Adjust the idx to the ones which are on the first longitude!
    if (idx < A)
    {
      idx += idxBaseXY;
    }
    indices.push_back(idx);
  }

  for (size_t i = 0; i < sizeof(cutPlane) / sizeof(int); i += 3 )
  {
    int idx0 = cutPlane[i    ];
    int idx1 = cutPlane[i + 1];
    int idx2 = cutPlane[i + 2];

    // The five helper point indices on the yz-plane are directly behind the A-E ones.
    idx0 += (A <= idx0) ? 5 : idxBaseYZ;
    idx1 += (A <= idx1) ? 5 : idxBaseYZ;
    idx2 += (A <= idx2) ? 5 : idxBaseYZ;

    // Invert the winding!
    indices.push_back(idx2);
    indices.push_back(idx1);
    indices.push_back(idx0);
  }

  unsigned int numVertices = dp::checked_cast<unsigned int>( verticesOut.size() );

  // Create a VertexAttributeSet with vertices, normals and texcoords
  VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
  vasPtr->setVertices( &verticesOut[0], numVertices );
  vasPtr->setNormals( &normalsOut[0], numVertices );
  vasPtr->setTexCoords( 0, &texcoordsOut[0], numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_TANGENT  - VertexAttributeSet::DP_SG_TEXCOORD0, &tangentsOut[0],  numVertices );
  vasPtr->setTexCoords( VertexAttributeSet::DP_SG_BINORMAL - VertexAttributeSet::DP_SG_TEXCOORD0, &binormalsOut[0], numVertices );

  IndexSetSharedPtr indexSet( IndexSet::create() );
  indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

  // Create a Primitive
  PrimitiveSharedPtr primitivePtr = Primitive::create( PRIMITIVE_TRIANGLES );
  primitivePtr->setVertexAttributeSet( vasPtr );
  primitivePtr->setIndexSet( indexSet );

  return primitivePtr;
}


PreviewScene::PreviewScene()
{
  m_primitive[0] = createPreviewMainObject( 90 );
  m_primitive[1] = createPreviewSphere( 90 );
  m_primitive[2] = createPreviewBowl( 90 );

  m_primitive[3] = createPreviewRing( 90 );
  m_primitive[4] = createPreviewPlatform( 90 );


  dp::fx::EffectLibrary::instance()->loadEffects( "PreviewScene.xml");

  char const* const names[] =
  {
      "Main Object"
    , "Sphere Object"
    , "Bowl Object"
    , "Ring Object"
    , "Platform Object"
  };

  char const* const materials[] =
  {
      "phong_red"
    , "phong_green"
    , "phong_white"
    , "phong_yellow"
    , "phong_white"
  };

  for (int i = 0; i < 5; i++)
  {
    m_geoNodeHandle[i] = GeoNode::create();
    m_geoNodeHandle[i]->setPrimitive( m_primitive[i] );
    m_geoNodeHandle[i]->setName( names[i] );
    setEffectData( i, materials[i] );
  }

  m_transformHandle = Transform::create();
  Trafo trafo;
  trafo.setCenter( Vec3f( 0.0f, 1.375f, 0.0f ) );
  trafo.setOrientation( Quatf( Vec3f(0.0f, 0.0f, 1.0f), -PI_QUARTER * 0.65f ) *
                        Quatf( Vec3f(0.0f, 1.0f, 0.0f), -PI_QUARTER ) );

  // Create a Transform
  m_transformHandle->setTrafo( trafo );
  m_transformHandle->addChild( m_geoNodeHandle[0] );
  m_transformHandle->addChild( m_geoNodeHandle[1] );
  m_transformHandle->addChild( m_geoNodeHandle[2] );

  // Create the root
  GroupSharedPtr groupHdl = Group::create();
  groupHdl->addChild( m_transformHandle );
  groupHdl->addChild( m_geoNodeHandle[3] );
  groupHdl->addChild( m_geoNodeHandle[4] );
  groupHdl->setName( "Root Node" );

  m_sceneHandle = Scene::create();
  m_sceneHandle->setBackColor(  Vec4f( 71.0f / 255.0f, 111.0f / 255.0f, 0.0f, 1.0f ) );
  m_sceneHandle->setRootNode( groupHdl );
}

PreviewScene::~PreviewScene()
{
}

void PreviewScene::setEffectData( size_t index, const std::string& effectData )
{
  DP_ASSERT( index < sizeof dp::util::array( m_effectHandle ) );

  dp::fx::EffectDataSharedPtr fxEffectData = dp::fx::EffectLibrary::instance()->getEffectData( effectData );
  DP_ASSERT( fxEffectData );

  m_effectHandle[index] = EffectData::create( fxEffectData );
  m_geoNodeHandle[index]->setMaterialEffect( m_effectHandle[index] );
}
