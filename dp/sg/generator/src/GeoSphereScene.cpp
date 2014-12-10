// Copyright NVIDIA Corporation 2013
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


#include <dp/sg/generator/GeoSphereScene.h>

#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/Node.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Transform.h>

#include <dp/sg/generator/MeshGenerator.h>

#include <dp/math/math.h>
#include <dp/math/Vecnt.h>

#include <dp/fx/EffectLibrary.h>

using namespace dp::math;
using namespace dp::sg::core;
using namespace std;

typedef std::pair<unsigned int, unsigned int> TriangleEdge;
typedef std::map<TriangleEdge, unsigned int> MapTriangleEdgeToIndex;

// Subdivision level of the shell
// (each subdivision results in four times the number of triangles starting at level 0 with 20 triangles of the icosahedron)
#define NUM_SUBDIVISIONS  5
// Hole radius on the outer shell as scale of the arc length between two adjacent base vertices.
// (Must be smaller than 0.5f and the working size depends on the shell tessellation. Triangles must not touch two holes!)
#define HOLE_RADIUS_SCALE (1.0f / 3.0f)
// Number of rings building the rim
// (excluding the vertices on the shells. This should be an odd number to get a latitude at the rim's "equator".)
#define NUM_LATITUDES     7
// Half wall thickness of the object, the radius of the rim
// (this affects the effective hole size)
#define THICKNESS_HALF    0.025f

static void doSubdivision( const std::vector<unsigned int>& indicesIn, 
                           std::vector<unsigned int>& indicesOut, std::vector<Vec3f>& vertices )
{
  MapTriangleEdgeToIndex edgeToIndex;
  
  unsigned int nextIndex    = dp::util::checked_cast<unsigned int>( vertices.size() );
  unsigned int numTriangles = dp::util::checked_cast<unsigned int>( indicesIn.size() ) / 3;
  
  indicesOut.clear();

  for ( unsigned int i = 0; i < numTriangles; ++i ) 
  {
    unsigned int idx[3]; // Original triangle indices.
    unsigned int abc[3]; // Newly generated indices on the edges.

    idx[0] = indicesIn[i * 3    ];
    idx[1] = indicesIn[i * 3 + 1];
    idx[2] = indicesIn[i * 3 + 2];
    
    for ( unsigned int j = 0; j < 3; ++j )
    {
      unsigned int k = (j + 1) % 3;
      unsigned int m = min( idx[j], idx[k] );
      unsigned int n = max( idx[j], idx[k] );

      TriangleEdge edge(m, n);

      MapTriangleEdgeToIndex::iterator it = edgeToIndex.find( edge );
      if ( it == edgeToIndex.end() ) // Edge not found?
      {
        // Generate new vertex in the center of the edge.
        Vec3f v = vertices[m] + vertices[n];
        normalize( v );
        vertices.push_back( v );
        edgeToIndex[edge] = nextIndex;
        abc[j] = nextIndex++;
      }
      else
      {
        abc[j] = it->second;
      }
    }

    // Here all vertices required to subdive the current triangle exist.

    // Lower left triangle.
    indicesOut.push_back( idx[0] );
    indicesOut.push_back( abc[0] );
    indicesOut.push_back( abc[2] );
    // Lower right triangle.
    indicesOut.push_back( abc[0] );
    indicesOut.push_back( idx[1] );
    indicesOut.push_back( abc[1] );
    // Upper triangle.
    indicesOut.push_back( abc[1] );
    indicesOut.push_back( idx[2] );
    indicesOut.push_back( abc[2] );
    // Center triangle.
    indicesOut.push_back( abc[2] );
    indicesOut.push_back( abc[0] );
    indicesOut.push_back( abc[1] );
  }
}


void doHole( const Vec3f& center, float radius,
             const std::vector<unsigned int>& indicesIn, std::vector<Vec3f>& vertices,
             std::vector<unsigned int>& indicesOut,
             std::set<unsigned int>& holeIndices, std::set<TriangleEdge>& holeEdges )
{
  indicesOut.clear();
  DP_ASSERT( holeIndices.empty() );
  DP_ASSERT( holeEdges.empty() );

  unsigned int numTriangles = dp::util::checked_cast<unsigned int>( indicesIn.size() ) / 3;

  for ( unsigned int i = 0; i < numTriangles; ++i ) 
  {
    unsigned int idx[3]; // Original triangle indices.
    unsigned int abc[3]; // Zero to three indices which are inside the radius.

    idx[0] = indicesIn[i * 3    ];
    idx[1] = indicesIn[i * 3 + 1];
    idx[2] = indicesIn[i * 3 + 2];
    
    unsigned int mask = 0;
    unsigned int count = 0; // Count the vertices which are visible.
    for ( unsigned int j = 0; j < 3; ++j )
    {
      unsigned int k = idx[j];
      const Vec3f& v = vertices[k];
      float d = dp::math::distance( center, v ) - radius;
      if ( d < 0.0f ) // Inside the radius.
      {
        mask |= 1 << j;
        abc[count++] = k;
      }
    }

    if ( mask < 7 ) // Some part of the triangle is visible. Keep the original topology!
    {
      indicesOut.push_back( idx[0] );
      indicesOut.push_back( idx[1] );
      indicesOut.push_back( idx[2] );
    }

    // Now add the indices which need to be pushed to the radius to build the hole edge.
    // Those are all inside indices of triangles which are not trivially in or out.
    if ( count == 1 )
    {
      holeIndices.insert( abc[0] );
    }
    else if ( count == 2 )
    {
      holeIndices.insert( abc[0] );
      holeIndices.insert( abc[1] );

      // Remember the hole's edge segments. (This is actually the complete unordered line strip!)
      unsigned int m = min( abc[0], abc[1] );
      unsigned int n = max( abc[0], abc[1] );
      
      TriangleEdge edge(m, n);

      holeEdges.insert( edge );
    }
  }
  
  // See if circleEdges contain all edges. That is the case if the circleIndices.size() == circleEdges.size(),
  // because then each vertex must be start point of one edge. That simplifies generating the rim a lot.
  DP_ASSERT( holeIndices.size() == holeEdges.size() );

  // Here the set holeIndices contains all indices of vertices 
  // which need to be pushed outside to lie on the circle radius.
  for ( std::set<unsigned int>::const_iterator it = holeIndices.begin(); it != holeIndices.end(); ++it ) 
  {
    Vec3f v = vertices[*it];
    Vec3f dir = v - center;
    float d = dp::math::length( dir ); 
    normalize( dir );
    v =  v + dir * (radius - d);
    normalize( v );
    vertices[*it] = v;
  }
}


void doRim( const Vec3f& center, float radius,
            const std::set<unsigned int>& holeIndices, const std::set<TriangleEdge>& holeEdges, unsigned int offset,
            std::vector<unsigned int>& indices, std::vector<Vec3f>& vertices, std::vector<Vec3f>& normals )
{
  // Each vector in the rim array contains the indices of one longitudinal ring. The correct ring index is found with a map.
  std::vector<unsigned int> rim[NUM_LATITUDES];
  std::map<unsigned int, unsigned int> mapHoleIndexToLongitudeIndex; // To find the longitudinal ring when having a start index on the circle.

  unsigned int iv = 0; // Index into rim array.
  for ( std::set<unsigned int>::const_iterator it = holeIndices.begin(); it != holeIndices.end(); ++it )
  {
    unsigned int k = *it; // The index on the outer shell.
    mapHoleIndexToLongitudeIndex[k] = iv++; // To find the longitude vertices in the rim array.
    
    // Generate a right handed orthonormal basis with the normal as x-axis, the y-axis on the plane spanned by v and center.
    Vec3f v = vertices[k];
    Vec3f xAxis = v; // This is normalized.
    Vec3f zAxis = v ^ center;
    normalize( zAxis );
    Vec3f yAxis = zAxis ^ xAxis; // Unit length implicit.

    // Now generate a longitudinal arc of vertices on the standard xyz-coordinate system and transform it into the orthonormal base.
    for ( unsigned int j = 0; j < NUM_LATITUDES; ++j )
    {
      float angle = float(j + 1) * dp::math::PI / float(NUM_LATITUDES + 1);
      float c = cos( angle );
      float s = sin( angle );

      Vec3f n = c * xAxis + s * yAxis;   // Normal.
      Vec3f p = (1.0f - THICKNESS_HALF) * v + THICKNESS_HALF * n; // Center of the rim plus scaled normal gives the point.

      rim[j].push_back( dp::util::checked_cast<unsigned int>( vertices.size() ) ); // Store the new index in the rim's longitude indices.

      normals.push_back( n );
      vertices.push_back( p );
    }
  }

  // Now generate the rim triangles.
  for ( std::set<TriangleEdge>::const_iterator it = holeEdges.begin(); it != holeEdges.end(); ++it )
  {
    // Get the edge indices.
    unsigned int j = it->first;
    unsigned int k = it->second;

    // Figure out how the winding is.
    Vec3f& a = vertices[j];
    Vec3f& b = vertices[k];

    Vec3f c = (a - center) ^ (b - center); // cross
    float d = center * c; // cosine between the vectors.
    if ( d < 0.0f ) // center vector and cross product point to opposite directions, swap edge indices.
    {
      swap(j, k);
    }

    // The final two indices on the inner shell's hole edge are these:
    unsigned int jInner = j + offset;
    unsigned int kInner = k + offset;

    // Find the indices on the longitudinal rings.
    std::map<unsigned int, unsigned int>::const_iterator itj = mapHoleIndexToLongitudeIndex.find( j );
    std::map<unsigned int, unsigned int>::const_iterator itk = mapHoleIndexToLongitudeIndex.find( k );
    
    for ( unsigned int i = 0; i < NUM_LATITUDES; ++i )
    {
      unsigned int m = rim[i][itj->second];
      unsigned int n = rim[i][itk->second];

      indices.push_back( j );
      indices.push_back( k );
      indices.push_back( n );

      indices.push_back( n );
      indices.push_back( m );
      indices.push_back( j );

      j = m;
      k = n;
    }

    // The final rim triangles close the gap to the inner shell hole edge.
    // The indices of that edge are the same as the ones on the outside plus the index offset.
    indices.push_back( j );
    indices.push_back( k );
    indices.push_back( kInner );

    indices.push_back( kInner );
    indices.push_back( jInner );
    indices.push_back( j );
  }
}

static PrimitiveSharedPtr createGeoSphereObject()
{
  static const float X = 0.525731112119133606f;
  static const float Z = 0.850650808352039932f;

  // Setup vertices. Initial vertex pool, subdivision level zero.
  static const Vec3f verticesBase[12] =
  { 
    Vec3f(   -X, 0.0f,    Z ),
    Vec3f(    X, 0.0f,    Z ),
    Vec3f(   -X, 0.0f,   -Z ),
    Vec3f(    X, 0.0f,   -Z ),
    Vec3f( 0.0f,    Z,    X ),
    Vec3f( 0.0f,    Z,   -X ),
    Vec3f( 0.0f,   -Z,    X ),
    Vec3f( 0.0f,   -Z,   -X ),
    Vec3f(    Z,    X, 0.0f ),
    Vec3f(   -Z,    X, 0.0f ),
    Vec3f(    Z,   -X, 0.0f ),
    Vec3f(   -Z,   -X, 0.0f )
  };

  // Setup triangles 
  static const unsigned int triangles[20 * 3] =
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

  unsigned int src = 0;
  unsigned int dst = 1;

  vector<Vec3f> vertices;

  vector<unsigned int> indices[2];
  vector<unsigned int> indicesPong;

  // Seed the initial input vectors.
  for ( unsigned int i = 0 ; i < 12; ++i )
  {
    vertices.push_back( verticesBase[i] );
  }

  for ( unsigned int i = 0 ; i < 20 * 3; ++i )
  {
    indices[src].push_back( triangles[i] );
  }

  // Indices are completely regenerated each time.
  // New vertices are appended to the input vertices.
  for (unsigned int subdivisions = 0; subdivisions < NUM_SUBDIVISIONS; ++subdivisions ) 
  {
    doSubdivision( indices[src], indices[dst], vertices );
    src ^= 1;
    dst ^= 1;
  }

  // Find the indices and edges building the twelve holes.
  // The number of vertices does NOT change during this step, which means all indices stay valid.
  std::set<unsigned int> holeIndices[12];
  std::set<TriangleEdge> holeEdges[12];

  // First generate the outer shell with all twelve holes with circular hole edges.
  float radius = HOLE_RADIUS_SCALE * acos( verticesBase[0] * verticesBase[1] );
  for ( unsigned int i = 0; i < 12; ++i )
  {
    doHole( verticesBase[i], radius, 
            indices[src], vertices,
            indices[dst], holeIndices[i], holeEdges[i] ); // Output.
    src ^= 1;
    dst ^= 1;
  }

  // Generate the inner shell vertices and normals.
  unsigned int countVertices = dp::util::checked_cast<unsigned int>( vertices.size() ); // Important, this is the offset for all inner triangle indices.
  std::vector<Vec3f> normals = vertices; // The unit vectors of the outer shell are the normals.
  for ( unsigned int i = 0; i < countVertices; ++i )
  {
    Vec3f v = vertices[i];
    normals.push_back( -v ); 
    vertices.push_back( v * (1.0f - THICKNESS_HALF - THICKNESS_HALF) );
  }
  
  // Now add the inner shell triangles.
  size_t countTriangles = indices[src].size() / 3;
  for ( size_t i = 0; i < countTriangles; ++i )
  {
    unsigned int a = indices[src][i * 3    ];
    unsigned int b = indices[src][i * 3 + 1];
    unsigned int c = indices[src][i * 3 + 2];

    // Inverse the winding for the inner shell and use the vertices behind the outer shell.
    indices[src].push_back( a + countVertices );
    indices[src].push_back( c + countVertices );
    indices[src].push_back( b + countVertices );
  }

  // Build the rims.
  for ( unsigned int i = 0; i < 12; ++i )
  {
    doRim( verticesBase[i], radius, 
           holeIndices[i], holeEdges[i], countVertices,
           indices[src], vertices, normals );
  }

  // Remove the unused vertices and normals from the array here. (At subdivision level 5 these are over 7200.)
  // The ones trivially inside the holes are still in the two arrays, but the triangles don't index any of them.
  std::vector<Vec3f> verticesFinal;
  std::vector<Vec3f> normalsFinal;
  std::map<unsigned int, unsigned int> remapper;
  indices[dst].clear();
  unsigned int k = 0; // New index.
  for ( size_t i = 0; i < indices[src].size(); ++i ) 
  {
    unsigned int j = indices[src][i]; // Old index.
    std::map<unsigned int, unsigned int>::const_iterator it = remapper.find( j );
    if ( it == remapper.end() ) 
    {
      // remap j to k
      verticesFinal.push_back( vertices[j] ); // This is the kth vertex now.
      normalsFinal.push_back( normals[j] );
      indices[dst].push_back( k );
      remapper[j] = k++;
    }
    else
    {
      indices[dst].push_back( it->second );
    }
  }
  src ^= 1;
  dst ^= 1;

  // Create a VertexAttributeSet with vertices and normals
  VertexAttributeSetSharedPtr vertexAttributeSet = VertexAttributeSet::create();
  vertexAttributeSet->setVertices( &verticesFinal[0], dp::util::checked_cast<unsigned int>(verticesFinal.size()) );
  vertexAttributeSet->setNormals( &normalsFinal[0], dp::util::checked_cast<unsigned int>(normalsFinal.size()) );

  // Create a Primitive
  IndexSetSharedPtr indexSet = IndexSet::create();
  indexSet->setData( &indices[src][0], dp::util::checked_cast<unsigned int>(indices[src].size()) );

  // create pointer to return
  PrimitiveSharedPtr primitive = Primitive::create( PRIMITIVE_TRIANGLES );
  primitive->setVertexAttributeSet( vertexAttributeSet );
  primitive->setIndexSet( indexSet );

  return primitive;
}


GeoSphereScene::GeoSphereScene()
{  
  m_primitive = createGeoSphereObject();
  
  dp::fx::EffectLibrary::instance()->loadEffects( "PreviewScene.xml");

  dp::fx::EffectDataSharedPtr phongEffectData = dp::fx::EffectLibrary::instance()->getEffectData( "phong_red" );
  DP_ASSERT( phongEffectData );

  m_effectHandle = EffectData::create( phongEffectData );

  m_geoNodeHandle = GeoNode::create();
  m_geoNodeHandle->setPrimitive( m_primitive );
  m_geoNodeHandle->setName( "GeoSphere");
  m_geoNodeHandle->setMaterialEffect( m_effectHandle );

  // Create a Transform
  m_transformHandle = Transform::create();
  Trafo trafo;
  //trafo.setOrientation( Quatf( Vec3f(0.0f, 0.0f, 1.0f), -PI_QUARTER * 0.65f ) * 
  //                      Quatf( Vec3f(0.0f, 1.0f, 0.0f), -PI_QUARTER ) );
  m_transformHandle->setTrafo( trafo );
  m_transformHandle->addChild( m_geoNodeHandle );

  // Create the root
  GroupSharedPtr groupHdl = Group::create();
  groupHdl->addChild( m_transformHandle );
  groupHdl->setName( "Root Node" );

  m_sceneHandle = Scene::create();
  // m_sceneHandle->setBackColor(  Vec4f( 71.0f / 255.0f, 111.0f / 255.0f, 0.0f, 1.0f ) );
  m_sceneHandle->setBackColor(  Vec4f( 0.3f, 0.3f, 0.3f, 0.0f ) );
  m_sceneHandle->setRootNode( groupHdl );
}

GeoSphereScene::~GeoSphereScene()
{
}

