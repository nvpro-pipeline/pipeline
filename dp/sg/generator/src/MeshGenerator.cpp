// Copyright (c) 2009-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/generator/MeshGenerator.h>

#include <dp/math/math.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Node.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/io/PlugInterface.h>
#include <dp/sg/io/PlugInterfaceID.h>
#include <dp/util/PlugIn.h>

#include <vector>

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using namespace std;

namespace dp
{
  namespace sg
  {
    namespace generator
    {
      namespace
      {
        //! Helper function to calculate the face normal of the face a, b, c
        Vec3f calculateFaceNormal( const Vec3f &a, const Vec3f &b, const Vec3f &c )
        {
          Vec3f d1 = b - a;
          Vec3f d2 = c - a;
          Vec3f n  = d1 ^ d2;
          n.normalize();
          return n;
        }

        //! Helper function to setup the faces, vertices, normals and texccords of a tessellated plane
        //with \a subdiv subdivisions and a transformation-matrix transf
        void setupTessellatedPlane( unsigned int subdiv, const Mat44f &transf,
                                    vector < Vec3f > &vertices,
                                    vector < Vec3f > &normals, vector < Vec2f > &texcoords,
                                    vector <unsigned int> &indices )
        {
          float step = 2.0f/(float)(subdiv + 1);
          unsigned int row = subdiv + 2;
          unsigned int offset = dp::checked_cast<unsigned int>(vertices.size());


          // This is expensive do it once outside the loops!
          Mat44f transfIT;
          bool validInverse = invert( transf, transfIT );
          if (validInverse)
          {
            transfIT = ~transfIT;
          }

          float y = -1.0f;
          for ( unsigned int sY = 0; sY < row; sY++ )
          {
            float x = -1.0f;
            for ( unsigned int sX = 0; sX < row; sX++ )
            {
              vertices.push_back( Vec3f(Vec4f( x, y, 0.0f, 1.0f ) * transf ) );
              Vec3f normal(0.0f, 0.0f, 1.0f); // Initialize to some valid normal in case the transf matrix cannot be inverted.
              if ( validInverse )
              {
                normal = Vec3f(Vec4f( 0.0f, 0.0f, 1.0f, 0.0f) * transfIT);
                normal.normalize();
              }
              normals.push_back( normal );
              texcoords.push_back( Vec2f( x * 0.5f + 0.5f, y * 0.5f + 0.5f ) );
              x += step;
            }
            y += step;
          }

          for ( unsigned int sY = 0; sY <= subdiv; sY++ )
          {
            for ( unsigned int sX = 0; sX <= subdiv; sX++ )
            {
              indices.push_back( offset + sX + sY * row );
              indices.push_back( offset + sX + 1 + sY * row );
              indices.push_back( offset + sX + 1 + (sY+1) * row );

              indices.push_back( offset + sX + 1 + (sY+1) * row );
              indices.push_back( offset + sX + (sY+1) * row );
              indices.push_back( offset + sX + sY * row );
            }
          }
        }
      }

      // ===========================================================================

      PrimitiveSharedPtr createQuadSet( unsigned int m, unsigned int n, const float size, const float gap )
      {
        DP_ASSERT( m >= 1 && n >= 1 && "createQuadSet(): m and n both have to be at least 1." );

        // create pointer to return
        PrimitiveSharedPtr primitivePtr;

        // setup vertices, normals and faces for n X m tiles
        vector<Vec3f> vertices;
        vector<Vec3f> normals;
        vector<unsigned int> indices;

        const int size_v = 4 * m * n;
        vertices.reserve( size_v );
        normals.reserve( size_v );
        indices.reserve( size_v );

        // lower-left corner of the current tile
        float dy = 0.0f;

        // n tiles in x-direction
        for( unsigned int i = 0; i < m; ++i )
        {
          // reset x-offset
          float dx = 0.0f;

          // n tiles in x-direction
          for( unsigned int j = 0; j < n; ++j )
          {
            // add 4 vertices
            Vec3f a = Vec3f( dx       , dy       ,              0.0f );
            Vec3f b = Vec3f( dx + size, dy       , (float)j/(float)n );
            Vec3f c = Vec3f( dx + size, dy + size, (float)j/(float)n );
            Vec3f d = Vec3f( dx       , dy + size,              0.0f );
            vertices.push_back( a );
            vertices.push_back( b );
            vertices.push_back( c );
            vertices.push_back( d );

            unsigned int first_index = ( i * n + j ) * 4;
            Vec3f n = calculateFaceNormal( a, b, d );

            // Setup normals and faces
            for( unsigned int k = 0; k < 4; ++k )
            {
              normals.push_back(n);
              indices.push_back( first_index + k );
            }

            dx += size + gap;
          }
          dy += size + gap;
        }

        // Create a VertexAttributeSet with vertices and normals
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], size_v );
        vasPtr->setNormals( &normals[0], size_v );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // Create a Primitive
        primitivePtr = Primitive::create( PrimitiveType::QUADS );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createQuadStrip( unsigned int n, float height, float radius )
      {
        DP_ASSERT( n >= 1 && "createQuadStrip(): n has to be at least 1." );

        // create pointer to return
        PrimitiveSharedPtr primitivePtr;

        // setup vertices, normals and indices
        vector<Vec3f> vertices;
        vector<Vec3f> normals;
        vector<unsigned int> indices;

        const int size_v = 2*(n+1);
        vertices.reserve( size_v );
        normals.reserve( size_v );
        indices.reserve( size_v );

        // for n quads, we need 2*(n+1) vertices, so run to including n
        for( unsigned int i = 0; i <= n; ++i )
        {
          float phi = PI * ( 1.0f - (float)i/(float)n );
          float x = cos(phi);
          float z = sin(phi);
          Vec3f v( x, 0.0f, z );

          normals.push_back( v );
          normals.push_back( v );
          vertices.push_back( v * radius + Vec3f( 0.0f, height, 0.0f ) );
          vertices.push_back( v * radius );

          indices.push_back( i * 2 );
          indices.push_back( i * 2 + 1 );

        }

        // Create a VertexAttributeSet with vertices and normals
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], size_v );
        vasPtr->setNormals( &normals[0], size_v );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // Create a Primitive
        primitivePtr = Primitive::create( PrimitiveType::QUAD_STRIP );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createTriSet( unsigned int m, unsigned int n, const float size, const float gap )
      {
        DP_ASSERT( m >= 1 && n >= 1 && "createTriSet(): m and n both have to be at least 1." );

        // setup vertices, normals and faces for n X m tiles
        vector<Vec3f> vertices;
        vector<Vec3f> normals;
        vector<unsigned int> indices;

        const int size_v = 3 * m * n;
        vertices.reserve( size_v );
        normals.reserve( size_v );
        indices.reserve( size_v );

        // lower-left corner of the current tile
        float dy = 0.0f;

        // m tiles in y-direction
        for( unsigned int i = 0; i < m; ++i )
        {
          // reset x-offset
          float dx = 0.0f;

          // n tiles in x-direction
          for( unsigned int j = 0; j < n; ++j )
          {
            // add 3 vertices
            Vec3f a = Vec3f( dx       , dy       ,              0.0f );
            Vec3f b = Vec3f( dx + size, dy       , (float)j/(float)n );
            Vec3f c = Vec3f( dx       , dy + size,              0.0f );
            vertices.push_back( a );
            vertices.push_back( b );
            vertices.push_back( c );

            // Setup faces and normals
            unsigned int first_index = ( i * n + j ) * 3;
            Vec3f n = calculateFaceNormal( a, b, c );

            for( unsigned int k = 0; k < 3; ++k )
            {
              normals.push_back(n);
              indices.push_back( first_index + k );
            }

            dx += size + gap;
          }
          dy += size + gap;
        }

        // Create a VertexAttributeSet with vertices and normals
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], size_v );
        vasPtr->setNormals( &normals[0], size_v );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createTriFan( unsigned int n, const float radius, const float elevation )
      {
        DP_ASSERT( n > 1 && "createTriFan(): n has to be at least 2." );

        // setup vertices and faces for n triangles
        vector<Vec3f> vertices;
        vector<unsigned int> indices;

        vertices.reserve( n + 2 );
        indices.reserve( n + 2 );

        vertices.push_back( Vec3f( 0.0f, 0.0f, elevation ) );
        vertices.push_back( Vec3f( radius, 0.0f, 0.0f ) );

        indices.push_back(0);
        indices.push_back(1);

        // n triangles ( i from 1 to n ! )
        for( unsigned int i = 1; i <= n; ++i )
        {
          float phi = PI * ((float)i/(float)n);
          float x = radius * cos(phi);
          float y = radius * sin(phi);
          vertices.push_back( Vec3f(x, y, 0.f) );
          indices.push_back( i + 1 );
        }

        // Calculate normals
        vector<Vec3f> normals;
        normals.resize(vertices.size());
        for( size_t i = 0; i < normals.size(); ++i )
        {
          normals[i] = Vec3f( 0.0f, 0.0f, 0.0f );
        }

        for( unsigned int i = 0; i < n; ++i )
        {
          // calculate face normal for face i
          Vec3f fn = calculateFaceNormal( vertices.at(0), vertices.at(i+1), vertices.at(i+2) );
          // accumulate in vertex normals
          normals.at(0)   += fn;
          normals.at(i+1) += fn;
          normals.at(i+2) += fn;
        }

        for( size_t i = 0; i < normals.size(); ++i )
        {
          normals[i].normalize();
        }


        // Create a VertexAttributeSet with vertices and normals
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], n + 2 );
        vasPtr->setNormals( &normals[0], n + 2 );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // create pointer to return
        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLE_FAN );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createTriStrip( unsigned int rows, unsigned int columns, float width, float height )
      {
        DP_ASSERT( rows >= 1 && columns >= 1 && "createTriStrip(): rows and columns both have to be at least 1." );

        // Setup vertices, normals and indices
        vector<Vec3f> vertices;
        vector<Vec3f> normals;
        vector<unsigned int> indices;

        unsigned int size_v = ( rows + 1 ) * ( columns + 1 );
        vertices.reserve( size_v );
        normals.reserve( size_v );
        indices.reserve( size_v );

        for( unsigned int i = 0; i <= rows; ++i )
        {
          for( unsigned int j = 0; j <= columns; ++j )
          {
            vertices.push_back( Vec3f( j * width, i * height, 0.0f ) );
            normals.push_back( Vec3f( 0.0f, 0.0f, 1.0f ) );
          }
        }

        for( unsigned int i = 0; i < rows; ++i )
        {
          for( unsigned int j = 0; j <= columns; ++j )
          {
            indices.push_back( i * ( columns + 1 ) + j + columns + 1 );
            indices.push_back( i * ( columns + 1 ) + j );
          }
        }

        // Create a VertexAttributeSet with vertices and normals
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], size_v );
        vasPtr->setNormals( &normals[0], size_v );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // create pointer to return
        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLE_STRIP );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      GeoNodeSharedPtr createTriPatches4( const std::vector<std::string> & searchPaths, unsigned int m, unsigned int n, const Vec3f & size, const Vec2f & offset )
      {
        // Set up the tri patch:
        //  9
        //  7 8
        //  4 5 6
        //  0 1 2 3
        //
        Vec3f verts[10] =
        {
          Vec3f(size[0]*0.0f/4.0f, size[1]*0.0f/4.0f, size[2]*0.0f/4.0f), // 0
          Vec3f(size[0]*1.0f/4.0f, size[1]*0.0f/4.0f, size[2]*1.0f/4.0f), // 1
          Vec3f(size[0]*2.0f/4.0f, size[1]*0.0f/4.0f, size[2]*1.0f/4.0f), // 2
          Vec3f(size[0]*3.0f/4.0f, size[1]*0.0f/4.0f, size[2]*0.0f/4.0f), // 3
          Vec3f(size[0]*0.0f/4.0f, size[1]*1.0f/4.0f, size[2]*1.0f/4.0f), // 4
          Vec3f(size[0]*1.0f/4.0f, size[1]*1.0f/4.0f, size[2]*2.0f/4.0f), // 5
          Vec3f(size[0]*2.0f/4.0f, size[1]*1.0f/4.0f, size[2]*1.0f/4.0f), // 6
          Vec3f(size[0]*0.0f/4.0f, size[1]*2.0f/4.0f, size[2]*1.0f/4.0f), // 7
          Vec3f(size[0]*1.0f/4.0f, size[1]*2.0f/4.0f, size[2]*1.0f/4.0f), // 8
          Vec3f(size[0]*0.0f/4.0f, size[1]*3.0f/4.0f, size[2]*0.0f/4.0f)  // 9
        };

        std::vector<Vec3f> vertices;

        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;

        for( unsigned int i = 0; i < m; ++i )
        {
          for( unsigned int j = 0; j < n; ++j )
          {
            for( unsigned int k = 0; k < 10; ++k )
            {
              vertices.push_back( verts[k] + Vec3f(x, y, z) );
            }
            x += offset[0];
          }
          y += offset[1];
          x = 0.0f;
        }

        // Create a VertexAttributeSet
        VertexAttributeSetSharedPtr vas = VertexAttributeSet::create();
        vas->setVertices( &vertices[0], dp::checked_cast<unsigned int>(vertices.size()) );

        // Create a Primitive as triangular patches with 10 vertices per patch
        PrimitiveSharedPtr triPatches = Primitive::create( PatchesType::CUBIC_BEZIER_TRIANGLES, PatchesMode::TRIANGLES );
        triPatches->setVertexAttributeSet( vas );

        // Create a GeoNode and add the geometry
        return GeoNodeSharedPtr::null;
      }

      // ===========================================================================

      GeoNodeSharedPtr createQuadPatches4x4( const std::vector<std::string> & searchPaths, unsigned int n, unsigned int m, const Vec2f & offset )
      {
        const float r = std::min( offset[0], offset[1] )/2.0f * 0.75f;
        const float dy = r;  // distance between rows of vertices

        // For a cylinder, we need four quad patches, each extruding an approximation
        // of a quarter of a circle

        // For a bezier curve to approximate a quarter of a circle, we need to use a
        // cubic bezier curve with inner points at a distance x of r*4(sqrt(2)-1)/3 from
        // the end points in the tangent direction at the end points:
        //
        //   0--d--1
        //   |      \
        //   r       2
        //   |       |
        //   *       3
        //

        // Distance of the boundary control points to the corner control points
        const float d = r * 4.0f * (sqrt(2.0f)-1.0f)/3.0f;

        // A zero for nice alignment
        const float o = 0.0f;

        // First construct the bezier curves to be extruded
        Vec3f bezierCurves[16];
        bezierCurves[ 0] = Vec3f( o, o, r ); //
        bezierCurves[ 1] = Vec3f( d, o, r ); //
        bezierCurves[ 2] = Vec3f( r, o, d ); //
        bezierCurves[ 3] = Vec3f( r, o, o ); //        |
        bezierCurves[ 4] = Vec3f( r, o, o ); //        |
        bezierCurves[ 5] = Vec3f( r, o,-d ); //      9 786
        bezierCurves[ 6] = Vec3f( d, o,-r ); //     A  |  5
        bezierCurves[ 7] = Vec3f( o, o,-r ); //  --BC-----34--x
        bezierCurves[ 8] = Vec3f( o, o,-r ); //     D  |  2
        bezierCurves[ 9] = Vec3f(-d, o,-r ); //      E F0 1
        bezierCurves[10] = Vec3f(-r, o,-d ); //        |
        bezierCurves[11] = Vec3f(-r, o, o ); //        z
        bezierCurves[12] = Vec3f(-r, o, o ); //
        bezierCurves[13] = Vec3f(-r, o, d ); //
        bezierCurves[14] = Vec3f(-d, o, r ); //
        bezierCurves[15] = Vec3f( o, o, r ); //

        // Extrude the curves to form the patches
        //            y  C D E F
        // one patch: |  8 9 A B
        //            |  4 5 6 7
        //            O  0 1 2 3
        vector< Vec3f > verts;
        for( unsigned int i = 0; i < 4; ++i )
        {
          // patch i
          for( unsigned int j = 0; j < 4; ++j )
          {
            // row j
            for( unsigned int k = 0; k < 4; ++k )
            {
              // column k
              verts.push_back( bezierCurves[i*4+k] + Vec3f( o, dy*j, o ));
            }
          }
        }

        // Generate n x m cylinders
        vector< Vec3f > vertices;
        for( unsigned int i = 0; i < n; ++i )
        {
          for( unsigned int j = 0; j < m; ++j )
          {
            float x = i * offset[0];
            float z = j * offset[1];
            for( unsigned int k = 0; k < verts.size(); ++k )
            {
              vertices.push_back( verts.at( k ) + Vec3f( x, o, z ) );
            }
          }
        }

        // Create a VertexAttributeSet
        VertexAttributeSetSharedPtr vas = VertexAttributeSet::create();
        vas->setVertices( &vertices[0], dp::checked_cast<unsigned int>( vertices.size() ) );

        // Create a Primitive as rectangular patches with 16 vertices per patch
        PrimitiveSharedPtr patches = Primitive::create( PatchesType::CUBIC_BEZIER_QUADS, PatchesMode::QUADS );
        patches->setVertexAttributeSet( vas );

        // Create a GeoNode and add the geometry
        return( GeoNodeSharedPtr::null );
      }

      // ===========================================================================

      PrimitiveSharedPtr createCube()
      {
        // Right handed model coordinates.
        // Vertex numbering and coordinate setup match the 3-bit pattern: (z << 2) | (y << 1) | x
        // ASCII art:
        /*
             y

             2--------------3
            /              /|
           / |            / |
          6--------------7  |
          |  |           |  |
          |              |  |
          |  |           |  |
          |  0 -  -  -  -| -1  x
          | /            | /
          |/             |/
          4--------------5
         z
        */

        // Setup vertices
        static const Vec3f vertices[8] =
        {
          Vec3f( -1.0f, -1.0f, -1.0f ), // 0
          Vec3f(  1.0f, -1.0f, -1.0f ), // 1
          Vec3f( -1.0f,  1.0f, -1.0f ), // 2
          Vec3f(  1.0f,  1.0f, -1.0f ), // 3
          Vec3f( -1.0f, -1.0f,  1.0f ), // 4
          Vec3f(  1.0f, -1.0f,  1.0f ), // 5
          Vec3f( -1.0f,  1.0f,  1.0f ), // 6
          Vec3f(  1.0f,  1.0f,  1.0f )  // 7
        };

        // Setup faces
        static const unsigned int faces[12*3] =
        {
          1, 0, 2, 2, 3, 1, // back
          0, 4, 6, 6, 2, 0, // left
          0, 1, 5, 5, 4, 0, // bottom
          4, 5, 7, 7, 6, 4, // front
          5, 1, 3, 3, 7, 5, // right
          3, 2, 6, 6, 7, 3  // top
        };

        // Setup texture coordinates
        static const Vec2f texcoords[4] =
        {
          Vec2f(0.0f, 0.0f),
          Vec2f(1.0f, 0.0f),
          Vec2f(1.0f, 1.0f),
          Vec2f(0.0f, 1.0f)
        };

        Vec3f v[36];
        Vec3f n[36];
        Vec2f tc[36];
        vector<unsigned int> indices;

        Vec3f v0, v1, v2, fn;

        for ( int kf = 0, kv = 0; kf < 12; kf++, kv += 3 )
        {
          v0 = vertices[faces[kv+0]];
          v1 = vertices[faces[kv+1]];
          v2 = vertices[faces[kv+2]];
          fn = calculateFaceNormal( v0, v1, v2 );

          indices.push_back(kv);
          indices.push_back(kv+1);
          indices.push_back(kv+2);

          v[kv]    = v0;
          v[kv+1]  = v1;
          v[kv+2]  = v2;

          n[kv]    = fn;
          n[kv+1]  = fn;
          n[kv+2]  = fn;

          // Assign texture coordinates
          if (kf & 1)
          { // odd faces
            tc[kv]    = texcoords[2];
            tc[kv+1]  = texcoords[3];
            tc[kv+2]  = texcoords[0];
          }
          else
          { // Even faces
            tc[kv]    = texcoords[0];
            tc[kv+1]  = texcoords[1];
            tc[kv+2]  = texcoords[2];
          }
        }

        // Create a VertexAttributeSet with vertices, normals and texture coordinates
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( v, 36 );
        vasPtr->setNormals( n, 36 );
        vasPtr->setTexCoords( 0, tc, 36 );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // Create a Primitive
        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createTetrahedron()
      {
        // create pointer to return
        PrimitiveSharedPtr primitivePtr;

        // Setup vertices
        static const Vec3f vertices[4] =
        {
          Vec3f( -1.0f, -1.0f, -1.0f ),
          Vec3f( 1.0f, 1.0f, -1.0f ),
          Vec3f( 1.0f, -1.0f, 1.0f ),
          Vec3f( -1.0f, 1.0f, 1.0f )
        };

        // Setup faces:
        static const unsigned int faces[4*3] =
        {
          0, 3, 1,
          0, 1, 2,
          0, 2, 3,
          1, 3, 2
        };

        // Setup texture coordinates
        static const Vec2f texcoords[4] =
        {
          Vec2f(0.0f, 0.0f),
          Vec2f(1.0f, 1.0f),
          Vec2f(1.0f, 0.0f),
          Vec2f(0.0f, 1.0f)
        };

        // Calculate normals
        Vec3f v[12];
        Vec3f n[12];
        Vec2f tc[12];

        Vec3f v0, v1, v2, fn;

        vector<unsigned int> indices;

        for ( int kf = 0, kv = 0; kf < 4; kf++, kv += 3 )
        {
          v0 = vertices[faces[kv+0]];
          v1 = vertices[faces[kv+1]];
          v2 = vertices[faces[kv+2]];

          fn = calculateFaceNormal( v0, v1, v2 );

          indices.push_back(kv);
          indices.push_back(kv+1);
          indices.push_back(kv+2);

          v[kv]    = v0;
          v[kv+1]  = v1;
          v[kv+2]  = v2;

          n[kv]    = fn;
          n[kv+1]  = fn;
          n[kv+2]  = fn;
        }

        // Assign texture coordinates
        tc[0]  = texcoords[0];
        tc[1]  = texcoords[1];
        tc[2]  = texcoords[2];
        tc[3]  = texcoords[0];
        tc[4]  = texcoords[3];
        tc[5]  = texcoords[1];
        tc[6]  = texcoords[0];
        tc[7]  = texcoords[2];
        tc[8]  = texcoords[3];
        tc[9]  = texcoords[1];
        tc[10] = texcoords[3];
        tc[11] = texcoords[2];


        // Create a VertexAttributeSet with vertices, normals and texture coordinates
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( v, 12 );
        vasPtr->setNormals( n, 12 );
        vasPtr->setTexCoords( 0, tc, 12 );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createOctahedron()
      {
        // Setup vertices
        static const Vec3f vertices[6] =
        {
          Vec3f( 0.0f, 1.0f, 0.0f ),
          Vec3f( 0.0f, -1.0f, 0.0f ),
          Vec3f( 1.0f, 0.0f, 0.0f ),
          Vec3f( -1.0f, 0.0f, 0.0f ),
          Vec3f( 0.0f, 0.0f, 1.0f ),
          Vec3f( 0.0f, 0.0f, -1.0f )
        };

        // Setup faces
        static const unsigned int faces[8*3] =
        {
          0, 4, 2,
          0, 2, 5,
          0, 5, 3,
          0, 3, 4,
          1, 2, 4,
          1, 5, 2,
          1, 3, 5,
          1, 4, 3
        };

        // Setup texture coordinates
        static const Vec2f texcoords[5] =
        {
          Vec2f(0.0f, 0.0f),  //0
          Vec2f(1.0f, 0.0f),  //1
          Vec2f(1.0f, 1.0f),  //2
          Vec2f(0.0f, 1.0f),  //3
          Vec2f(0.5f, 0.5f)   //4
        };

        // Calculate normals
        Vec3f v[24];
        Vec3f n[24];
        Vec2f tc[24];

        Vec3f v0, v1, v2, fn;

        vector<unsigned int> indices;

        for ( int kf = 0, kv = 0; kf < 8; kf++, kv += 3 )
        {
          v0 = vertices[faces[kv+0]];
          v1 = vertices[faces[kv+1]];
          v2 = vertices[faces[kv+2]];

          fn = calculateFaceNormal( v0, v1, v2 );

          indices.push_back(kv);
          indices.push_back(kv+1);
          indices.push_back(kv+2);

            v[kv]    = v0;
          v[kv+1]  = v1;
          v[kv+2]  = v2;

          n[kv]    = fn;
          n[kv+1]  = fn;
          n[kv+2]  = fn;
        }


        // Assign texture coordinates
        tc[0]  = texcoords[2];
        tc[1]  = texcoords[4];
        tc[2]  = texcoords[1];
        tc[3]  = texcoords[2];
        tc[4]  = texcoords[1];
        tc[5]  = texcoords[4];
        tc[6]  = texcoords[2];
        tc[7]  = texcoords[4];
        tc[8]  = texcoords[3];
        tc[9]  = texcoords[2];
        tc[10] = texcoords[3];
        tc[11] = texcoords[4];
        tc[12] = texcoords[0];
        tc[13] = texcoords[1];
        tc[14] = texcoords[4];
        tc[15] = texcoords[0];
        tc[16] = texcoords[4];
        tc[17] = texcoords[1];
        tc[18] = texcoords[0];
        tc[19] = texcoords[3];
        tc[20] = texcoords[4];
        tc[21] = texcoords[0];
        tc[22] = texcoords[4];
        tc[23] = texcoords[3];


        // Create a VertexAttributeSet with vertices, normals and texture coordinates
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( v, 24 );
        vasPtr->setNormals( n, 24 );
        vasPtr->setTexCoords( 0, tc, 24 );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // Create a Primitive
        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createDodecahedron()
      {
        //USE ICOSAHEDRON VERTICES AND FACES TO SETUP DODECAHEDRON
        //--------------------------------------------------------
        static const float x = 0.525731112119133606f;
        static const float z = 0.850650808352039932f;

        static const Vec3f icoVertices[12] =
        {
          Vec3f(   -x , 0.0f ,    z ),
          Vec3f(    x , 0.0f ,    z ),
          Vec3f(   -x , 0.0f ,   -z ),
          Vec3f(    x , 0.0f ,   -z ),
          Vec3f( 0.0f ,    z ,    x ),
          Vec3f( 0.0f ,    z ,   -x ),
          Vec3f( 0.0f ,   -z ,    x ),
          Vec3f( 0.0f ,   -z ,   -x ),
          Vec3f(    z ,    x , 0.0f ),
          Vec3f(   -z ,    x , 0.0f ),
          Vec3f(    z ,   -x , 0.0f ),
          Vec3f(   -z ,   -x , 0.0f )
        };

        static const unsigned int icoFaces[20*3] =
        {
           0,  1,  4, //  0
           0,  4,  9, //  1
           0,  9, 11, //  2
           0,  6,  1, //  3
           0, 11,  6, //  4
           1,  6, 10, //  5
           1, 10,  8, //  6
           1,  8,  4, //  7
           2,  3,  7, //  8
           2,  5,  3, //  9
           2,  9,  5, // 10
           2, 11,  9, // 11
           2,  7, 11, // 12
           3,  5,  8, // 13
           3,  8, 10, // 14
           3, 10,  7, // 15
           4,  5,  9, // 16
           4,  8,  5, // 17
           6,  7, 10, // 18
           6, 11,  7  // 19
        };

        //--------------------------------------------------------

        // Setup indices of dodecahedron
        static const int idxDode[12][5] =
        {
          { 0,  1,  2,  4,  3}, //  0
          { 0,  3,  5,  6,  7}, //  1
          { 9,  8, 12, 11, 10}, //  2
          { 9, 13, 14, 15,  8}, //  3
          { 0,  7, 17, 16,  1}, //  4
          { 9, 10, 16, 17, 13}, //  5
          {18,  5,  3,  4, 19}, //  6
          {15, 18, 19, 12,  8}, //  7
          { 6, 14, 13, 17,  7}, //  8
          { 1, 16, 10, 11,  2}, //  9
          { 5, 18, 15, 14,  6}, // 10
          { 2, 11, 12, 19,  4}  // 11
        };

        // Calculate vertices
        Vec3f vertices[20];
        Vec3f v;

        // The 20 vertices of the dodecahedron are the centers of the 20 icosahedron triangle faces
        // pushed out to unit sphere radius by normalization
        for ( int i = 0; i < 20; i++ )
        {
          v = icoVertices[icoFaces[3*i+0]] +  icoVertices[icoFaces[3*i+1]] +  icoVertices[icoFaces[3*i+2]];
          v.normalize();
          vertices[i] = v;
        }

        // Setup vertices, normals and faces
        vector<Vec3f> vv;
        vector<Vec3f> vn;

        vv.reserve( 72 );
        vn.reserve( 72 );

        vector<unsigned int> indices;
        indices.reserve( 180 );

        for( unsigned int i = 0 ; i < 12; ++i )
        {
          Vec3f v( 0.0f, 0.0f, 0.0f );
          for( unsigned int j = 0 ; j < 5; ++j )
          {
            v += vertices[idxDode[i][j]];
          }
          v /= 5.0f;

          // center point of face
          vv.push_back(v);
          vv.push_back(vertices[idxDode[i][0]]);
          vv.push_back(vertices[idxDode[i][1]]);
          vv.push_back(vertices[idxDode[i][2]]);
          vv.push_back(vertices[idxDode[i][3]]);
          vv.push_back(vertices[idxDode[i][4]]);

          v.normalize();
          for( unsigned int j=0 ; j<6; ++j )
          {
            vn.push_back(v);
          }

          unsigned int k = i * 6;

          indices.push_back(k);
          indices.push_back(k+1);
          indices.push_back(k+2);

          indices.push_back(k);
          indices.push_back(k+2);
          indices.push_back(k+3);

          indices.push_back(k);
          indices.push_back(k+3);
          indices.push_back(k+4);

          indices.push_back(k);
          indices.push_back(k+4);
          indices.push_back(k+5);

          indices.push_back(k);
          indices.push_back(k+5);
          indices.push_back(k+1);
        }

        // Create a VertexAttributeSet with vertices and normals
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vv[0], 72 );
        vasPtr->setNormals( &vn[0], 72 );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // create a Primitive
        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createIcosahedron()
      {
        static const float x = 0.525731112119133606f;
        static const float z = 0.850650808352039932f;

        // Setup vertices
        static const Vec3f vertices[12] =
        {
          Vec3f(   -x , 0.0f ,    z ),
          Vec3f(    x , 0.0f ,    z ),
          Vec3f(   -x , 0.0f ,   -z ),
          Vec3f(    x , 0.0f ,   -z ),
          Vec3f( 0.0f ,    z ,    x ),
          Vec3f( 0.0f ,    z ,   -x ),
          Vec3f( 0.0f ,   -z ,    x ),
          Vec3f( 0.0f ,   -z ,   -x ),
          Vec3f(    z ,    x , 0.0f ),
          Vec3f(   -z ,    x , 0.0f ),
          Vec3f(    z ,   -x , 0.0f ),
          Vec3f(   -z ,   -x , 0.0f )
        };

        // Setup faces
        static const unsigned int faces[20*3] =
        {
           0,  1,  4, //  0
           0,  4,  9, //  1
           0,  9, 11, //  2
           0,  6,  1, //  3
           0, 11,  6, //  4
           1,  6, 10, //  5
           1, 10,  8, //  6
           1,  8,  4, //  7
           2,  3,  7, //  8
           2,  5,  3, //  9
           2,  9,  5, // 10
           2, 11,  9, // 11
           2,  7, 11, // 12
           3,  5,  8, // 13
           3,  8, 10, // 14
           3, 10,  7, // 15
           4,  5,  9, // 16
           4,  8,  5, // 17
           6,  7, 10, // 18
           6, 11,  7  // 19
        };

        vector<Vec3f> vv;
        vector<Vec3f> vn;

        vv.reserve(60);
        vn.reserve(60);

        unsigned int j=0;

        vector<unsigned int> indices;
        indices.reserve( 60 );

        for( unsigned int i = 0 ; i < 20; ++i )
        {
          Vec3f a = vertices[faces[3*i+0]];
          Vec3f b = vertices[faces[3*i+1]];
          Vec3f c = vertices[faces[3*i+2]];
          vv.push_back(a);
          vv.push_back(b);
          vv.push_back(c);

          Vec3f n = calculateFaceNormal( a, b, c );
          vn.push_back(n);
          vn.push_back(n);
          vn.push_back(n);

          indices.push_back(j++);
          indices.push_back(j++);
          indices.push_back(j++);
        }

        // Create a VertexAttributeSet with vertices and normals
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vv[0], 60 );
        vasPtr->setNormals( &vn[0], 60 );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // Create a Primitive
        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createSphere( unsigned int m, unsigned int n, float radius )
      {
        DP_ASSERT( m >= 3 && n >= 3 && "createSphere(): m and n both have to be at least 3." );

        // setup vertices, normals, indices/faces and texture coordinates
        vector< Vec3f > vertices;
        vector< Vec3f > tangents;
        vector< Vec3f > binormals;
        vector< Vec3f > normals;
        vector< Vec2f > texcoords;
        vector<unsigned int> indices;

        const int size_v = ( m + 1 ) * n;
        vertices.reserve( size_v );
        tangents.reserve( size_v );
        binormals.reserve( size_v );
        normals.reserve( size_v );
        texcoords.reserve( size_v );
        indices.reserve( 6 * m * ( n - 1 ) );

        float phi_step = 2.0f * PI / (float) m;
        float theta_step = PI / (float) (n - 1);

        // Latitudinal rings.
        // Starting at the south pole going upwards.
        for( unsigned int latitude = 0 ; latitude < n ; latitude++ ) // theta angle
        {
          float theta = (float) latitude * theta_step;
          float sinTheta = sinf( theta );
          float cosTheta = cosf( theta );
          float texv = (float) latitude / (float) (n - 1); // Range [0.0f, 1.0f]

          // Generate vertices along the latitudinal rings.
          // On each latitude there are m + 1 vertices,
          // the last one and the first one are on identical positions but have different texture coordinates.
          for( unsigned int longitude = 0 ; longitude <= m ; longitude++ ) // phi angle
          {
            float phi = (float) longitude * phi_step;
            float sinPhi = sinf( phi );
            float cosPhi = cosf( phi );
            float texu = (float) longitude / (float) m; // Range [0.0f, 1.0f]

            // Unit sphere coordinates are the normals.
            Vec3f v = Vec3f( cosPhi * sinTheta,
                            -cosTheta,                 // -y to start at the south pole.
                            -sinPhi * sinTheta );

            vertices.push_back( v * radius );
            texcoords.push_back( Vec2f( texu , texv ) );
            normals.push_back( v );
            tangents.push_back( Vec3f( -sinPhi, 0.0f, -cosPhi ) );
            binormals.push_back( Vec3f( cosTheta * cosPhi, sinTheta, cosTheta * -sinPhi ) );
          }
        }

        // We have generated m + 1 vertices per latitude.
        const unsigned int columns = m + 1;

        // Calculate indices
        for( unsigned int latitude = 0 ; latitude < n - 1 ; latitude++ )
        {
          for( unsigned int longitude = 0 ; longitude < m ; longitude++ )
          {
            indices.push_back(  latitude      * columns + longitude     );  // lower left
            indices.push_back(  latitude      * columns + longitude + 1 );  // lower right
            indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right

            indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right
            indices.push_back( (latitude + 1) * columns + longitude     );  // upper left
            indices.push_back(  latitude      * columns + longitude     );  // lower left
          }
        }

        // Create a VertexAttributeSet with vertices, normals and texcoords
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], size_v );
        vasPtr->setNormals( &normals[0], size_v );
        vasPtr->setTexCoords( 0, &texcoords[0], size_v );
        vasPtr->setTexCoords( static_cast<unsigned int>(VertexAttributeSet::AttributeID::TANGENT)  - static_cast<unsigned int>(VertexAttributeSet::AttributeID::TEXCOORD0), &tangents[0], size_v );
        vasPtr->setTexCoords( static_cast<unsigned int>(VertexAttributeSet::AttributeID::BINORMAL) - static_cast<unsigned int>(VertexAttributeSet::AttributeID::TEXCOORD0), &binormals[0], size_v );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // Create a Primitive
        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

       return primitivePtr;
      }

      PrimitiveSharedPtr createCylinder( float r, float h, unsigned int hdivs, unsigned int phidivs, bool bOuter /*= true*/, bool bcaps /*= true*/ )
      {
        vector< Vec3f > vertices;
        vector< Vec3f > normals;
        vector< Vec2f > texcoords;
        vector<unsigned int> indices;

        unsigned int size_v = (hdivs+1)*phidivs + (bcaps ? 2*phidivs + 2 : 0);

        // ---Vertex order---
        // 1) 1                 bottom center vert
        // 2) thdivs            bottom rim
        // 3) thdivs*(hdivs+1)  all height division circles including bottom and top rim
        // 4) 1                 top center point
        // 5) thdivs            top rim

        float phi_step = 2.0f*PI / (float) phidivs;
        float h_step = h / (float) hdivs;

        //-------------------------------
        // Generate the vertices/normals
        //-------------------------------

        if( bcaps )
        {
          vertices.push_back( Vec3f( 0.0f, -h/2.0f, 0.0f ) );
          normals.push_back( Vec3f( 0.0f, bOuter ? -1.0f : 1.0f, 0.0f) );
          texcoords.push_back( Vec2f(0.5f, 0.5f) );

          for(unsigned int iphi = 0; iphi < phidivs; iphi++)
          {
            float curPhi = iphi*phi_step;
            vertices.push_back( Vec3f( r*cos(curPhi), -h/2.0f, -r*sin(curPhi) ) );
            normals.push_back( Vec3f( 0.0f, bOuter ? -1.0f : 1.0f, 0.0f) );
            texcoords.push_back( Vec2f( 0.5f*cos(curPhi) + 0.5f, -0.5f*sin(curPhi) + 0.5f) );
          }
        }

        for(unsigned int ih = 0; ih < hdivs+1; ih++)
        {
          float curH = -h/2.0f + ih*h_step;

          for(unsigned int iphi = 0; iphi < phidivs; iphi++)
          {
            float curPhi = iphi*phi_step;

            vertices.push_back( Vec3f( r*cos(curPhi), curH, -r*sin(curPhi)) );
            normals.push_back( Vec3f( (bOuter ? 1.0f : -1.0f)*cos(curPhi), 0.0f, -(bOuter ? 1.0f : -1.0f)*sin(curPhi)) );
            texcoords.push_back( Vec2f( 1.0f - 0.5f*curPhi/PI, curH/h ) );
          }
        }

        if( bcaps )
        {
          vertices.push_back( Vec3f( 0.0f, h/2.0f, 0.0f ) );
          normals.push_back( Vec3f( 0.0f, bOuter ? 1.0f : -1.0f, 0.0f) );
          texcoords.push_back( Vec2f(0.5f, 0.5f) );

          for(unsigned int iphi = 0; iphi < phidivs; iphi++)
          {
            float curPhi = iphi*phi_step;
            vertices.push_back( Vec3f( r*cos(curPhi), h/2.0f, -r*sin(curPhi) ) );
            normals.push_back( Vec3f( 0.0f, bOuter ? 1.0f : -1.0f, 0.0f ) );
            texcoords.push_back( Vec2f( 0.5f*cos(curPhi) + 0.5f, -0.5f*sin(curPhi) + 0.5f) );
          }
        }

        //-------------------------------
        // Generate the indices
        //-------------------------------

        int curvcount = 0;

        if( bcaps )
        {
          for(unsigned int iphi = 0; iphi < phidivs; iphi++)
          {
            indices.push_back( curvcount );

            if(bOuter)
            {
              indices.push_back( curvcount+2 + (iphi < phidivs-1 ? iphi : -1) );
              indices.push_back( curvcount+1 + iphi );
            }
            else
            {
              indices.push_back( curvcount+1 + iphi );
              indices.push_back( curvcount+2 + (iphi < phidivs-1 ? iphi : -1) );
            }
          }

          curvcount += 1 + phidivs;
        }

        for(unsigned int ih = 0; ih < hdivs; ih++)
        {
          for(unsigned int iphi = 0; iphi < phidivs; iphi++)
          {
            indices.push_back( curvcount+iphi );
            if(bOuter)
            {
              indices.push_back( curvcount+(iphi < phidivs-1 ? iphi+1 : 0) );
              indices.push_back( curvcount+iphi+phidivs );
            }
            else
            {
              indices.push_back( curvcount+iphi+phidivs );
              indices.push_back( curvcount+(iphi < phidivs-1 ? iphi+1 : 0) );
            }

            indices.push_back( curvcount+iphi+phidivs );
            if(bOuter)
            {
              indices.push_back( curvcount+(iphi < phidivs-1 ? iphi+1 : 0) );
              indices.push_back( curvcount+(iphi < phidivs-1 ? iphi+1 : 0) + phidivs );
            }
            else
            {
              indices.push_back( curvcount+(iphi < phidivs-1 ? iphi+1 : 0) + phidivs );
              indices.push_back( curvcount+(iphi < phidivs-1 ? iphi+1 : 0) );
            }
          }

          curvcount += phidivs;
        }

        curvcount += phidivs;

        if( bcaps )
        {
          for(unsigned int iphi = 0; iphi < phidivs; iphi++)
          {
            indices.push_back( curvcount );

            if(bOuter)
            {
              indices.push_back( curvcount+1 + iphi );
              indices.push_back( curvcount+2 + (iphi < phidivs-1 ? iphi : -1) );
            }
            else
            {
              indices.push_back( curvcount+2 + (iphi < phidivs-1 ? iphi : -1) );
              indices.push_back( curvcount+1 + iphi );
            }
          }
        }



        //-------------------------------
        // Register the primitive
        //-------------------------------

        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], size_v );
        vasPtr->setNormals( &normals[0], size_v );
        vasPtr->setTexCoords( 0, &texcoords[0], size_v );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createTorus( unsigned int m, unsigned int n, float innerRadius , float outerRadius )
      {
        // The torus is a ring with radius outerRadius rotated around the y-axis along the circle with innerRadius.

        /*           y
           ___       |       ___
         /     \           /     \
        |       |    |    |       |
        |       |         |       |
         \ ___ /     |     \ ___ /
                              <--->
                              outerRadius
                     <------->
                     innerRadius
        */

        DP_ASSERT( m >= 3 && n >= 3 && "createTorus(): m and n both have to be at least 3." );

        vector< Vec3f > vertices;
        vector< Vec3f > tangents;
        vector< Vec3f > binormals;
        vector< Vec3f > normals;
        vector< Vec2f > texcoords;
        vector<unsigned int> indices;

        unsigned int size_v = ( m + 1 ) * ( n + 1 );

        vertices.reserve( size_v );
        tangents.reserve( size_v );
        binormals.reserve( size_v );
        normals.reserve( size_v );
        texcoords.reserve( size_v );
        indices.reserve( 4 * m * n );

        float mf = (float) m;
        float nf = (float) n;

        float phi_step   = 2.0f * PI / mf;
        float theta_step = 2.0f * PI / nf;

        // Setup vertices and normals
        // Generate the Torus exactly like the sphere with rings around the origin along the latitudes.
        for ( unsigned int latitude = 0; latitude <= n; latitude++ ) // theta angle
        {
          float theta = (float) latitude * theta_step;
          float sinTheta = sinf(theta);
          float cosTheta = cosf(theta);

          float radius = innerRadius + outerRadius * cosTheta;

          for ( unsigned int longitude = 0; longitude <= m; longitude++ ) // phi angle
          {
            float phi = (float) longitude * phi_step;
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            vertices.push_back( Vec3f( radius      *  cosPhi,
                                       outerRadius *  sinTheta,
                                       radius      * -sinPhi ) );

            tangents.push_back( Vec3f( -sinPhi, 0.0f, -cosPhi ) );

            binormals.push_back( Vec3f( cosPhi * -sinTheta,
                                        cosTheta,
                                        sinPhi * sinTheta ) );

            normals.push_back( Vec3f( cosPhi * cosTheta,
                                      sinTheta,
                                     -sinPhi * cosTheta ) );

            texcoords.push_back( Vec2f( (float) longitude / mf , (float) latitude / nf ) );
          }
        }

        const unsigned int columns = m + 1;

        // Setup indices
        for( unsigned int latitude = 0 ; latitude < n ; latitude++ )
        {
          for( unsigned int longitude = 0 ; longitude < m ; longitude++ )
          {
            indices.push_back(  latitude      * columns + longitude     );  // lower left
            indices.push_back(  latitude      * columns + longitude + 1 );  // lower right
            indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right
            indices.push_back( (latitude + 1) * columns + longitude     );  // upper left
          }
        }

        // Create a VertexAttributeSet with vertices, normals and texture coordinates
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], size_v );
        vasPtr->setNormals( &normals[0], size_v );
        vasPtr->setTexCoords( 0, &texcoords[0], size_v );
        vasPtr->setTexCoords( static_cast<unsigned int>(VertexAttributeSet::AttributeID::TANGENT)  - static_cast<unsigned int>(VertexAttributeSet::AttributeID::TEXCOORD0), &tangents[0], size_v );
        vasPtr->setTexCoords( static_cast<unsigned int>(VertexAttributeSet::AttributeID::BINORMAL) - static_cast<unsigned int>(VertexAttributeSet::AttributeID::TEXCOORD0), &binormals[0], size_v );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::QUADS );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createTessellatedPlane( unsigned int subdiv, const Mat44f &transf )
      {
        // Setup vertices, normals, faces and texture coordinates (and indices for primitive creation mode)
        vector< Vec3f > vertices;
        vector< Vec3f > normals;
        vector< Vec2f > texcoords;
        vector<unsigned int> indices;

        const int size_v =  ( subdiv + 2 ) * ( subdiv + 2 );
        vertices.reserve( size_v );
        normals.reserve( size_v );
        texcoords.reserve( size_v );
        indices.reserve( size_v );

        // Setup tessellated plane
        setupTessellatedPlane( subdiv, transf, vertices, normals, texcoords, indices);

        // Create a VertexAttributeSet with vertices, normals and texture coordinates
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], size_v );
        vasPtr->setNormals( &normals[0], size_v );
        vasPtr->setTexCoords( 0, &texcoords[0], size_v );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createPlane( float x0, float y0,
        float width, float height,
        float wext, float hext)
      {
        vector< Vec3f > vertices;
        vector< Vec3f > normals;
        vector< Vec2f > texcoords;

        vertices.push_back( Vec3f(x0, y0, 0.0f) );
        normals.push_back( Vec3f(0.0f, 0.0f, 1.0f) );
        texcoords.push_back( Vec2f(0.0f, 0.0f) );

        vertices.push_back( Vec3f(x0 + width, y0, 0.0f) );
        normals.push_back( Vec3f(0.0f, 0.0f, 1.0f) );
        texcoords.push_back( Vec2f(wext, 0.0f) );

        vertices.push_back( Vec3f(x0 + width, y0 + height, 0.0f) );
        normals.push_back( Vec3f(0.0f, 0.0f, 1.0f) );
        texcoords.push_back( Vec2f(wext, hext) );

        /////

        vertices.push_back( Vec3f(x0 + width, y0 + height, 0.0f) );
        normals.push_back( Vec3f(0.0f, 0.0f, 1.0f) );
        texcoords.push_back( Vec2f(wext, hext) );

        vertices.push_back( Vec3f(x0, y0 + height, 0.0f) );
        normals.push_back( Vec3f(0.0f, 0.0f, 1.0f) );
        texcoords.push_back( Vec2f(0.0f, hext) );

        vertices.push_back( Vec3f(x0, y0, 0.0f) );
        normals.push_back( Vec3f(0.0f, 0.0f, 1.0f) );
        texcoords.push_back( Vec2f(0.0f, 0.0f) );


        //-------------------------------
        // Register the primitive
        //-------------------------------

        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], 6 );
        vasPtr->setNormals( &normals[0], 6 );
        vasPtr->setTexCoords( 0, &texcoords[0], 6 );
        vasPtr->setTexCoords( 1, &texcoords[0], 6 );

        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );

        return primitivePtr;
      }

      // ===========================================================================

      PrimitiveSharedPtr createTessellatedBox( unsigned int subdiv )
      {
        // Setup vertices, normals, faces and texture coordinates
        vector< Vec3f > vertices;
        vector< Vec3f > normals;
        vector< Vec2f > texcoords;
        vector<unsigned int> indices;

        const int size_v = 6 * ( subdiv + 2 ) * ( subdiv + 2 );
        vertices.reserve( size_v );
        normals.reserve( size_v );
        texcoords.reserve( size_v );
        indices.reserve( size_v );

        // Setup transformations for 6 box sides
        Mat44f transf[6];
        transf[0] = Mat44f( { 1.0f,  0.0f,  0.0f,  0.0f,
                              0.0f,  1.0f,  0.0f,  0.0f,
                              0.0f,  0.0f,  1.0f,  0.0f,
                              0.0f,  0.0f,  1.0f,  1.0f } ); // front

        transf[1] = Mat44f( { -1.0f,  0.0f,  0.0f,  0.0f,
                               0.0f,  1.0f,  0.0f,  0.0f,
                               0.0f,  0.0f, -1.0f,  0.0f,
                               0.0f,  0.0f, -1.0f,  1.0f } ); // back, 180 degrees around y-axis

        transf[2] = Mat44f( { 0.0f,  0.0f,  1.0f,  0.0f,
                              0.0f,  1.0f,  0.0f,  0.0f,
                             -1.0f,  0.0f,  0.0f,  0.0f,
                             -1.0f,  0.0f,  0.0f,  1.0f } ); // left, -90 degrees around y-axis

        transf[3] = Mat44f( { 0.0f,  0.0f, -1.0f,  0.0f,
                              0.0f,  1.0f,  0.0f,  0.0f,
                              1.0f,  0.0f,  0.0f,  0.0f,
                              1.0f,  0.0f,  0.0f,  1.0f } ); // right, 90 degrees around y-axis

        transf[4] = Mat44f( { 1.0f,  0.0f,  0.0f,  0.0f,
                              0.0f,  0.0f,  1.0f,  1.0f,
                              0.0f, -1.0f,  0.0f,  0.0f,
                              0.0f, -1.0f,  0.0f,  1.0f } ); // bottom, 90 degrees around x-axis

        transf[5] = Mat44f( { 1.0f,  0.0f,  0.0f,  0.0f,
                              0.0f,  0.0f, -1.0f,  1.0f,
                              0.0f,  1.0f,  0.0f,  0.0f,
                              0.0f,  1.0f,  0.0f,  1.0f } ); // top, -90 degrees around x-axis

        for ( unsigned int i=0; i<6; i++ )
        {
          setupTessellatedPlane( subdiv, transf[i], vertices, normals, texcoords, indices);
        }

        // Create a VertexAttributeSet with vertices, normals and texture coordinates
        VertexAttributeSetSharedPtr vasPtr = VertexAttributeSet::create();
        vasPtr->setVertices( &vertices[0], size_v );
        vasPtr->setNormals( &normals[0], size_v );
        vasPtr->setTexCoords( 0, &texcoords[0], size_v );

        IndexSetSharedPtr indexSet( IndexSet::create() );
        indexSet->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

        // Create a Primitive
        PrimitiveSharedPtr primitivePtr = Primitive::create( PrimitiveType::TRIANGLES );
        primitivePtr->setVertexAttributeSet( vasPtr );
        primitivePtr->setIndexSet( indexSet );

        return primitivePtr;
      }

      // ===========================================================================

      dp::sg::core::PipelineDataSharedPtr createTexture()
      {
        vector<Vec4f> tex;
        tex.resize(64);

        // Create pattern
        for( unsigned int i = 0; i < 8; ++i )
        {
          for( unsigned int j = 0; j < 8; ++j )
          {
            unsigned int pos = i * 8 + j;
            Vec4f col(float(( i ^ j ) & 1), float((( i ^ j ) & 2) / 2), float(((i  ^ j ) & 4) / 4), 1.0f);
            tex.at(pos) = col;
          }
        }

        TextureHostSharedPtr textureHost = TextureHost::create();
        textureHost->setCreationFlags( TextureHost::F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD );
        unsigned int index = textureHost->addImage( 8, 8, 1, Image::PixelFormat::RGBA, Image::PixelDataType::FLOAT32 );
        DP_ASSERT( index != -1 );
        textureHost->setImageData( index, (const void *) &tex[0] );
        textureHost->setTextureTarget( TextureTarget::TEXTURE_2D );
        textureHost->setTextureGPUFormat(TextureHost::TextureGPUFormat::FIXED8);

        SamplerSharedPtr sampler = Sampler::create( textureHost );
        sampler->setMagFilterMode( TextureMagFilterMode::NEAREST );
        sampler->setMinFilterMode( TextureMinFilterMode::NEAREST );

        ParameterGroupDataSharedPtr texture = createStandardTextureParameterData( sampler );

        dp::sg::core::PipelineDataSharedPtr pipelineData = dp::sg::core::PipelineData::create( getStandardMaterialSpec() );
        DP_VERIFY( pipelineData->setParameterGroupData( texture ) );

        return( pipelineData );
      }

      // ===========================================================================

      dp::sg::core::PipelineDataSharedPtr createAlphaTexture( unsigned int n /* =64 */ )
      {
        vector<Vec4f> tex;
        tex.resize(n*n);

        // Create pattern
        for( unsigned int i = 0; i < n; ++i )
        {
          for( unsigned int j = 0; j < n; ++j )
          {
            float dx = ( (float)i - (float)(n-1)/2.0f ) / ( (float)(n-1)/2.0f );
            float dy = ( (float)j - (float)(n-1)/2.0f ) / ( (float)(n-1)/2.0f );

            float val = max(0.0f, 1.0f - (dx*dx+dy*dy));

            unsigned int pos = i * n + j;
            Vec4f col( val, val, val, val );
            tex.at(pos) = col;
          }
        }

        TextureHostSharedPtr textureHost = TextureHost::create();
        textureHost->setCreationFlags( TextureHost::F_PRESERVE_IMAGE_DATA_AFTER_UPLOAD );
        unsigned int index = textureHost->addImage( n, n, 1, Image::PixelFormat::RGBA, Image::PixelDataType::FLOAT32 );
        DP_ASSERT( index != -1 );
        textureHost->setImageData( index, (const void *) &tex[0] );
        textureHost->setTextureTarget( TextureTarget::TEXTURE_2D );
        textureHost->setTextureGPUFormat(TextureHost::TextureGPUFormat::FIXED8);

        SamplerSharedPtr sampler = Sampler::create( textureHost );
        sampler->setMagFilterMode( TextureMagFilterMode::NEAREST );
        sampler->setMinFilterMode( TextureMinFilterMode::NEAREST );

        ParameterGroupDataSharedPtr texture = createStandardTextureParameterData( sampler );

        dp::sg::core::PipelineDataSharedPtr pipelineData = dp::sg::core::PipelineData::create( getStandardMaterialSpec() );
        DP_VERIFY( pipelineData->setParameterGroupData( texture ) );

        return( pipelineData );
      }

      // ===========================================================================

      GeoNodeSharedPtr createGeoNode( const PrimitiveSharedPtr &primitive )
      {
        //Create a GeoNode combining StateSet and Primitive
        GeoNodeSharedPtr geoNode = GeoNode::create();
        geoNode->setPrimitive( primitive );

        return geoNode;
      }

      // ===========================================================================

      GeoNodeSharedPtr createGeoNode( const PrimitiveSharedPtr &primitive, const dp::sg::core::PipelineDataSharedPtr & materialEffect )
      {
        //Create a GeoNode combining EffectData and Primitive
        GeoNodeSharedPtr geoNode = GeoNode::create();
        geoNode->setMaterialPipeline( materialEffect );
        geoNode->setPrimitive( primitive );

        return geoNode;
      }

      // ===========================================================================

      TransformSharedPtr createTransform( const NodeSharedPtr &node , const Vec3f &translation, const Quatf &orientation, const Vec3f &scaling )
      {
        // Make a Transformation
        Trafo trafo;
        trafo.setTranslation( translation );
        trafo.setOrientation( orientation );
        trafo.setScaling( scaling );

        // Create a Transform
        TransformSharedPtr transPtr = Transform::create();
        transPtr->setTrafo( trafo );
        transPtr->addChild( node );

        return transPtr;
      }

      // ===========================================================================

      TransformSharedPtr imitateRaster( const NodeSharedPtr &node, unsigned int width, unsigned int height )
      {
        Trafo trafo;

        trafo.setTranslation( Vec3f( -0.5f*width/height,  -0.5f,  0.0f ) );
        trafo.setOrientation( Quatf( Vec3f(0.0, 1.0, 0.0), 0.0) );
        trafo.setScaling( Vec3f( 1.0f/height, 1.0f/height, 1.0f ) );

        TransformSharedPtr transPtr = Transform::create();
        transPtr->setTrafo( trafo );
        transPtr->addChild( node );

        return transPtr;
      }

      // ===========================================================================

      TransformSharedPtr imitateRaster( unsigned int width, unsigned int height )
      {
        Trafo trafo;

        trafo.setTranslation( Vec3f( -0.5f*width/height,  -0.5f,  0.0f ) );
        trafo.setOrientation( Quatf( Vec3f(0.0, 1.0, 0.0), 0.0) );
        trafo.setScaling( Vec3f( 1.0f/height, 1.0f/height, 1.0f ) );


        TransformSharedPtr transPtr = Transform::create();
        transPtr->setTrafo( trafo );

        return transPtr;
      }

      // ===========================================================================

      void setCameraPOV(float x, float y, float z, dp::sg::ui::ViewStateSharedPtr const& viewState)
      {
        viewState->getCamera()->setPosition( Vec3f( x, y, z ) );
      }

      // ===========================================================================

      void setCameraDirNoRoll(float x, float y, float z, dp::sg::ui::ViewStateSharedPtr const& viewState)
      {
        CameraSharedPtr const& camera = viewState->getCamera();

        Vec3f dir(x, y, z);
        dir.normalize();

        camera->setDirection( dir );
        camera->setUpVector( Vec3f(0.0f, 1.0f, 0.0f) );
      }

      // ===========================================================================

      void setCameraDir(float x, float y, float z, dp::sg::ui::ViewStateSharedPtr const& viewState)
      {
        CameraSharedPtr const& camera = viewState->getCamera();

        Vec3f dir(x, y, z);
        dir.normalize();

        Vec3f up;

        if( ( dir != Vec3f(0.0f, 1.0f, 0.0f) ) && ( dir != Vec3f(0.0f, -1.0f, 0.0f) ) )
        {
          Vec3f yup(0.0f, 1.0f, 0.0f);

          up = yup - dir*(dir*yup);
          up.normalize();
        }
        else
        {
          up = Vec3f(0.0f, 0.0f, dir[1]);
        }

        camera->setUpVector( up );
        camera->setDirection( dir );
      }

      dp::sg::core::GroupSharedPtr replicate( dp::sg::core::NodeSharedPtr const& node, dp::math::Vec3ui const& gridSize, dp::math::Vec3f const& gridSpacing, bool clone )
      {
        DP_ASSERT( isPositive( node->getBoundingBox() ) );
        dp::math::Vec3f bboxSize = node->getBoundingBox().getSize();
        for ( int i=0 ; i<3 ; ++i )
        {
          bboxSize[i] *= gridSpacing[i];
        }

        dp::sg::core::GroupSharedPtr group = dp::sg::core::Group::create();
        dp::math::Trafo trafo;
        dp::math::Vec3f translation( 0.0f, 0.0f, 0.0f );
        for ( unsigned int x=0 ; x<gridSize[0] ; ++x )
        {
          translation[1] = 0.0f;
          for ( unsigned int y=0 ; y<gridSize[1] ; ++y )
          {
            translation[2] = 0.0f;
            for ( unsigned int z=0 ; z<gridSize[2] ; ++z )
            {
              trafo.setTranslation( translation );
              dp::sg::core::TransformSharedPtr transform = dp::sg::core::Transform::create();
              transform->setTrafo( trafo );
              transform->addChild( clone ? node->clone().inplaceCast<dp::sg::core::Node>() : node );
              group->addChild( transform );

              translation[2] += bboxSize[2];
            }
            translation[1] += bboxSize[1];
          }
          translation[0] += bboxSize[0];
        }
        return( group );
      }

    } // namespace generator
  } // namespace sg
} //namespace dp

