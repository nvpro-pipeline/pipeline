// Copyright NVIDIA Corporation 2002-2012
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


#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/TextureFile.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>

#include "ExtractGeometryTraverser.h"

#include <vector>
#include <sstream>

using namespace dp::fx;
using namespace dp::math;
using namespace std;
using namespace dp::sg::core;

ExtractGeometryTraverser::ExtractGeometryTraverser()
{
  // construct a material to extract defaults 
  const EffectSpecSharedPtr & standardSpec = getStandardMaterialSpec();
  EffectSpec::iterator groupSpecIt = standardSpec->findParameterGroupSpec( string( "standardMaterialParameters" ) );
  DP_ASSERT( groupSpecIt != standardSpec->endParameterGroupSpecs() );
  dp::sg::core::ParameterGroupDataSharedPtr material = dp::sg::core::ParameterGroupData::create( *groupSpecIt );

  ParameterGroupSpecSharedPtr pgs = material->getParameterGroupSpec();
  CSFSGMaterial csfsgmaterial;
  csfsgmaterial.name  = std::string("default");
  csfsgmaterial.diffuse  = Vec4f( material->getParameter<Vec3f>( pgs->findParameterSpec( "frontDiffuseColor" ) ), 1.0f );

  m_materials.push_back(csfsgmaterial);
  m_materialIDX = 0;

  m_materialMap.insert(CSFSGMaterialHashPair(nullptr,m_materialIDX));

  CSFSGNode csfsgroot;
  csfsgroot.geometryIDX = -1;
  csfsgroot.objectTM = dp::math::cIdentity44f;

  m_nodes.push_back(csfsgroot);

  m_parentStack.push(  0);
  m_objectStack.push( -1);

  m_objectIDX = -1;
}

ExtractGeometryTraverser::~ExtractGeometryTraverser(void)
{
}


void ExtractGeometryTraverser::handleLOD( const LOD *p )
{
  if( p->getNumberOfRanges() )
  {
    // we only traverse the highest LOD 
    traverseObject( *p->beginChildren() );
  }
  else
  {
    SharedTraverser::handleLOD(p);
  }
}

void ExtractGeometryTraverser::handleTransform( const Transform *p )
{
  // multiply trafo on top of current matrices
  const Trafo & trafo = p->getTrafo();

  //  call the (overloadable) preTraverse() between stack adjustment and traversal
  if ( preTraverseTransform( &trafo ) )
  {
    SharedTraverser::handleTransform( p );

    //  call the (overloadable) postTraverse() between stack adjustment and traversal
    postTraverseTransform( &trafo );
  }
}

bool ExtractGeometryTraverser::preTraverseTransform( const Trafo *p )
{
  CSFSGNode node;

  node.objectTM = p->getMatrix();
  node.geometryIDX = -1;

  int nodeIDX = makeIDX( m_nodes.size() );

  m_nodes.push_back(node);
  
  if(m_parentStack.size() != 0 && m_nodes.size() != 0)
  {
    m_nodes[m_parentStack.top()].children.push_back( nodeIDX ); 
  }

  m_parentStack.push( nodeIDX );
  m_objectStack.push( -1 );

  return( true );
}

void  ExtractGeometryTraverser::postTraverseTransform( const Trafo *p )
{
  m_parentStack.pop();
  m_objectStack.pop();
}

void  ExtractGeometryTraverser::handleGeoNode( const GeoNode * p )
{
  if ( p->getMaterialEffect() )
  {
    dp::sg::core::EffectDataSharedPtr const& ed = p->getMaterialEffect();
    const dp::sg::core::ParameterGroupDataSharedPtr & smp = ed->findParameterGroupData( string( "standardMaterialParameters" ) );
    if ( smp )
    {
      CSFSGMaterialHashMap::const_iterator itSearch = m_materialMap.find(ed.getWeakPtr());
      if ( itSearch == m_materialMap.end() )
      {
        ParameterGroupSpecSharedPtr pgs = smp->getParameterGroupSpec();
        CSFSGMaterial material;

        // simply copy this material for later
        material.name = smp->getName();
        material.diffuse  = Vec4f(smp->getParameter<Vec3f>( pgs->findParameterSpec( "frontDiffuseColor" ) ), 1.0f );

        m_materialIDX = makeIDX( m_materials.size() );

        m_materialMap.insert(CSFSGMaterialHashPair(ed.getWeakPtr(),m_materialIDX));
        m_materials.push_back(material);
      }
      else{
        m_materialIDX = itSearch->second;
      }
    }
  }
  SharedTraverser::handleGeoNode( p );
}

int ExtractGeometryTraverser::addPrimitive(int geometryIDX, const Primitive* p)
{
  CSFSGGeometry& geometry = m_geometries[geometryIDX];
  bool valid = false;
  unsigned int numPrims = 0;

  VertexAttributeSetSharedPtr const& vas = p->getVertexAttributeSet();

  Buffer::ConstIterator<Vec3f>::Type vertices = vas->getVertices();

  vector<unsigned int> indices;

  unsigned int offset = p->getElementOffset();
  unsigned int count  = p->getElementCount();

  // Turn all filled primitive types into triangulated index lists.
  switch ( p->getPrimitiveType() )
  {
  case PRIMITIVE_QUADS:
    {
      valid = true;
      // just in case
      count -= (count & 3); // modulo 4
      // assume no primitive restarts in indices
      numPrims = 2 * (count / 4);

      if ( p->isIndexed() )
      {
        IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );

        // assume no primitive restarts in indices
        for ( unsigned int i = 0; i < count; i += 4 )
        {
          if (  dp::math::distance( vertices[iter[i+0]], vertices[iter[i+2]] ) <=
                dp::math::distance( vertices[iter[i+1]], vertices[iter[i+3]] ) )
          {
            indices.push_back( iter[i+0] );
            indices.push_back( iter[i+1] );
            indices.push_back( iter[i+2] );

            indices.push_back( iter[i+2] );
            indices.push_back( iter[i+3] );
            indices.push_back( iter[i+0] );
          }
          else
          {
            indices.push_back( iter[i+1] );
            indices.push_back( iter[i+2] );
            indices.push_back( iter[i+3] );

            indices.push_back( iter[i+3] );
            indices.push_back( iter[i+0] );
            indices.push_back( iter[i+1] );
          }
        }
      }
      else
      {
        // Zero based indices! Adjustment happens by only storing the referenced vertices.
        for ( unsigned int i = 0; i < count; i += 4 )
        {
          if (  dp::math::distance( vertices[offset + i + 0], vertices[offset + i + 2] ) <=
                dp::math::distance( vertices[offset + i + 1], vertices[offset + i + 3] ) )
          {
            indices.push_back( i+0 );
            indices.push_back( i+1 );
            indices.push_back( i+2 );

            indices.push_back( i+2 );
            indices.push_back( i+3 );
            indices.push_back( i+0 );
          }
          else
          {
            indices.push_back( i+1 );
            indices.push_back( i+2 );
            indices.push_back( i+3 );

            indices.push_back( i+3 );
            indices.push_back( i+0 );
            indices.push_back( i+1 );
          }
        }
      }
    }
    break;

  case PRIMITIVE_QUAD_STRIP:
    {
      valid = true;

      if ( p->isIndexed() )
      {
        IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );
        unsigned int prIdx = p->getIndexSet()->getPrimitiveRestartIndex();

        for ( unsigned int i = 3; i < count; i += 2 )
        {
          if( iter[i] == prIdx )
          {
            // increment thrice so that i will start at base + 3, after loop increment
            i += 3;
            continue;
          }

          if (  dp::math::distance( vertices[iter[i-3]], vertices[iter[i-0]] ) <=
                dp::math::distance( vertices[iter[i-2]], vertices[iter[i-1]] ) )
          {
            indices.push_back( iter[i-3] );
            indices.push_back( iter[i-2] );
            indices.push_back( iter[i-0] );

            indices.push_back( iter[i-0] );
            indices.push_back( iter[i-1] );
            indices.push_back( iter[i-3] );
          }
          else
          {
            indices.push_back( iter[i-2] );
            indices.push_back( iter[i-0] );
            indices.push_back( iter[i-1] );

            indices.push_back( iter[i-1] );
            indices.push_back( iter[i-3] );
            indices.push_back( iter[i-2] );
          }

          numPrims += 2;
        }
      }
      else
      {
        // numQuads = (count - 2) / 2, so numTris = 2 * ( (count - 2) / 2)
        numPrims = count - 2;
        // Zero based indices! Adjustment happens by only storing the referenced vertices.
        for ( unsigned int i = 3; i < count; i += 2 )
        {
          if (  dp::math::distance( vertices[offset + i - 3], vertices[offset + i - 0] ) <=
                dp::math::distance( vertices[offset + i - 2], vertices[offset + i - 1] ) )
          {
            indices.push_back( i-3 );
            indices.push_back( i-2 );
            indices.push_back( i-0 );

            indices.push_back( i-0 );
            indices.push_back( i-1 );
            indices.push_back( i-3 );
          }
          else
          {
            indices.push_back( i-2 );
            indices.push_back( i-0 );
            indices.push_back( i-1 );

            indices.push_back( i-1 );
            indices.push_back( i-3 );
            indices.push_back( i-2 );
          }
        }
      }
    }
    break;

  case PRIMITIVE_POLYGON:
  case PRIMITIVE_TRIANGLE_FAN:
    {
      valid = true;

      if ( p->isIndexed() )
      {
        IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );
        unsigned int prIdx = p->getIndexSet()->getPrimitiveRestartIndex();
        unsigned int startIdx = 0;

        for ( unsigned int i = 2; i < count; i++ )
        {
          if ( iter[i] == prIdx )
          {
            i++;
            startIdx = i; // set startIdx at next index in list

            // increment one more so that i will start at startIdx + 2, after 
            // loop increment
            i++;
            continue;
          }

          indices.push_back( iter[startIdx] );
          indices.push_back( iter[i-1] );
          indices.push_back( iter[i-0] );

          numPrims++;
        }
      }
      else
      {
        numPrims = count - 2;
        // Zero based indices! Adjustment happens by only storing the referenced vertices.
        for ( unsigned int j = 2; j < count; j++ )
        {
          indices.push_back( 0 );
          indices.push_back( j-1 );
          indices.push_back( j-0 );
        }
      }
    }
    break;

  case PRIMITIVE_TRIANGLE_STRIP:
    {
      valid = true;

      if ( p->isIndexed() )
      {
        IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );
        unsigned int prIdx = p->getIndexSet()->getPrimitiveRestartIndex();
        bool ccw = true;

        for( unsigned int i = 2; i < count; i++ )
        {
          if( iter[i] == prIdx )
          {
            // increment twice so that i will start at base + 2, after loop increment
            i += 2;
            ccw = true;  // reset winding
            continue;
          }

          if ( ccw )
          {
            indices.push_back( iter[i-2] );
            indices.push_back( iter[i-1] );
            indices.push_back( iter[i-0] );
          }
          else
          {
            indices.push_back( iter[i-2] );
            indices.push_back( iter[i-0] );
            indices.push_back( iter[i-1] );
          }

          ccw = !ccw;
          numPrims++;
        }
      }
      else
      {
        numPrims = count - 2;
        bool ccw = true;
        // Zero based indices! Adjustment happens by only storing the referenced vertices.
        for ( unsigned int j = 2; j < count; j++ )
        {
          if ( ccw )
          {
            indices.push_back( j-2 );
            indices.push_back( j-1 );
            indices.push_back( j-0 );
          }
          else
          {
            indices.push_back( j-2 );
            indices.push_back( j-0 );
            indices.push_back( j-1 );
          }
          ccw = !ccw;
        }
      }
    }
    break;

  case PRIMITIVE_TRIANGLES:
    {
      valid = true;
      // just in case
      count -= (count % 3);
      // assume no primitive restarts in data stream
      numPrims = count / 3;

      if ( p->isIndexed() )
      {
        IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );

        // assume no primitive restarts in data stream
        for( unsigned int i = 0; i < count; i++ )
        {
          indices.push_back( iter[i] );
        }
      }
      else
      {
        // Zero based indices! Adjustment happens by only storing the referenced vertices.
        for ( unsigned int j = 0; j < count; j++ )
        {
          indices.push_back( j );
        }
      }
    }
    break;

  default:
    // Support for this primitive type not implemented.
    break;
  }

  // If there was geometry for which we could generate indices
  // store the appropriate vertex data.
  if ( valid && numPrims )
  {
    size_t numVerts     = vas->getNumberOfVertices();
    size_t numNormals   = vas->getNumberOfNormals();
    size_t numTexCoords = vas->getNumberOfTexCoords(0);

    if ( !p->isIndexed() )
    {
      numVerts = count;  // Only store the vertices which are used.
    }
    else
    {
      offset = 0;  // Store the whole array.
    }

    size_t vertexSize = numVerts;
    size_t oldVertexSize = geometry.vertices.size();

    geometry.vertices. reserve(vertexSize + oldVertexSize);
    geometry.normals.  reserve(vertexSize + oldVertexSize);
    geometry.texcoords.reserve(vertexSize + oldVertexSize);


    for ( unsigned int i = 0; i < numVerts; i++ )
    {
      geometry.vertices.push_back(vertices[offset + i]);
    }

    if( numNormals )
    {
      Buffer::ConstIterator<Vec3f>::Type normals = vas->getNormals();

      for ( unsigned int i = 0; i < numVerts; i++ )
      {
        Vec3f normal = normals[offset + i];
        normal.normalize();
        geometry.normals.push_back(normal);
      }
    }
    else
    {
      for( unsigned int i = 0; i < numVerts; i++ )
      {
        geometry.normals.push_back( Vec3f( 0.0f, 0.0f, 1.0f ) );
      }
    }

    if ( numTexCoords )
    {
      Buffer::ConstIterator<Vec2f>::Type texCoords = vas->getTexCoords<Vec2f>(0);

      for( unsigned int i = 0; i < numVerts; i++ )
      {
        geometry.texcoords.push_back( texCoords[i] );
      }
    }
    else
    {
      for ( unsigned int i = 0; i < numVerts; i++ )
      {
        geometry.texcoords.push_back( Vec2f( 0.0f, 0.0f ) );
      }
    }

    size_t indexSize = indices.size();
    geometry.indices.reserve(geometry.indices.size() + indexSize);

    for (size_t i = 0; i < indexSize; i+=3)
    {
      geometry.indices.push_back( dp::checked_cast<unsigned int, size_t>(oldVertexSize) + indices[i+0] );
      geometry.indices.push_back( dp::checked_cast<unsigned int, size_t>(oldVertexSize) + indices[i+1] );
      geometry.indices.push_back( dp::checked_cast<unsigned int, size_t>(oldVertexSize) + indices[i+2] );
    }

    CSFGeometryPart part;
    part.indexWire  = 0;
    part.indexSolid = static_cast<unsigned int>(indexSize);
    part.vertex     = static_cast<unsigned int>(vertexSize);
    geometry.parts.push_back(part);
    geometry.primitives.push_back(p);

    CSFSGGeometryHashMap::iterator itSearch = m_geometryMap.find( p );
    if (itSearch != m_geometryMap.end()){
      // primitive might have been added to another geometry before
      m_geometries[itSearch->second.first].alternativePrimitiveGeometryExists = true;
      m_geometryMap.erase(itSearch);
    }

    int partIDX = makeIDX ( geometry.parts.size() - 1 );
    m_geometryMap.insert( CSFSGGeometryHashPair(p,CSFSGGeometryHashEntry( geometryIDX, partIDX)));

    return partIDX;
  }

  return -1;

}

void ExtractGeometryTraverser::traversePrimitive( const Primitive * p )
{
  // test for active object (node with geometry)
  int objectIDX = m_objectStack.top();
  if (objectIDX < 0){
    // create object
    CSFSGNode node;
    node.objectTM = dp::math::cIdentity44f;
    node.geometryIDX = -1;
    objectIDX = makeIDX( m_nodes.size() );
    m_nodes.push_back(node);
    // add as child
    m_nodes[m_parentStack.top()].children.push_back( objectIDX );

    m_objectStack.top() = objectIDX;
  }

  // check if geometry exists
  int geometryIDX = m_nodes[objectIDX].geometryIDX;
  int partIDX = -1;

  if ( geometryIDX < 0 ){
    CSFSGGeometryHashMap::iterator itSearch = m_geometryMap.find( p );
    if (itSearch != m_geometryMap.end()){
      geometryIDX = itSearch->second.first;
      partIDX     = itSearch->second.second;
    }
    else{
      CSFSGGeometry geometry;
      geometryIDX = makeIDX( m_geometries.size() );
      m_geometries.push_back( geometry );
    }

    m_nodes[objectIDX].geometryIDX = geometryIDX;
  }

  // check if inside geometry
  if (partIDX < 0){
    CSFSGGeometry& geometry = m_geometries[geometryIDX];
    for ( size_t i = 0; i < geometry.primitives.size(); i++ ){
      if (geometry.primitives[i] == p){
        partIDX = makeIDX(i);
      }
    }
  }

  if (partIDX < 0){
    // add to geometry & object
    partIDX = addPrimitive(geometryIDX,p);
    if (partIDX < 0)
      return;

    CSFNodePart filler;
    filler.linewidth    = 1.0;
    filler.active       = 0;
    filler.materialIDX  = 0;

    CSFSGGeometry& geometry = m_geometries[geometryIDX];
    m_nodes[objectIDX].parts.resize( geometry.parts.size(), filler);
  }

  // assign proper material to node part list
  m_nodes[objectIDX].parts[partIDX].active = 1;
  m_nodes[objectIDX].parts[partIDX].materialIDX = m_materialIDX < 0 ? 0 : m_materialIDX;


}

std::vector<CSFSGNode>& ExtractGeometryTraverser::getNodes()
{
  return m_nodes;
}

std::vector<CSFSGGeometry>& ExtractGeometryTraverser::getGeometries()
{
  return m_geometries;
}

std::vector<CSFSGMaterial>& ExtractGeometryTraverser::getMaterials()
{
  return m_materials;
}

