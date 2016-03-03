// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/TextureHost.h>

#include "ExtractGeometryTraverser.h"

#include <vector>
#include <sstream>

using namespace dp::sg::core;
using namespace dp::math;
using namespace std;

ExtractGeometryTraverser::ExtractGeometryTraverser()
{
  m_transformStack.setWorldToView( cIdentity44f, cIdentity44f );

  // construct a material to extract defaults
  const dp::fx::EffectSpecSharedPtr & standardSpec = getStandardMaterialSpec();
 dp::fx:: EffectSpec::iterator groupSpecIt = standardSpec->findParameterGroupSpec( string( "standardMaterialParameters" ) );
  DP_ASSERT( groupSpecIt != standardSpec->endParameterGroupSpecs() );
  ParameterGroupDataSharedPtr material = ParameterGroupData::create( *groupSpecIt );
  dp::fx::ParameterGroupSpecSharedPtr pgs = material->getParameterGroupSpec();

  m_material.isTexture = false;
  m_material.isMaterial = false;
  m_material.ambient  = material->getParameter<Vec3f>( pgs->findParameterSpec( "frontAmbientColor" ) );
  m_material.diffuse  = material->getParameter<Vec3f>( pgs->findParameterSpec( "frontDiffuseColor" ) );
  m_material.specular = material->getParameter<Vec3f>( pgs->findParameterSpec( "frontSpecularColor" ) );
  m_material.exponent = material->getParameter<float>( pgs->findParameterSpec( "frontSpecularExponent" ) );
  m_material.opacity  = material->getParameter<float>( pgs->findParameterSpec( "frontOpacity" ) );
}

ExtractGeometryTraverser::~ExtractGeometryTraverser(void)
{
  DP_ASSERT( m_transformStack.getStackDepth() == 1 );
}

void  ExtractGeometryTraverser::handleBillboard( const Billboard *p )
{
  // ignore billboards for the moment
}

void  ExtractGeometryTraverser::handleSwitch( const Switch *p )
{
  // ignore switches for the moment
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
  m_transformStack.pushModelToWorld( trafo.getMatrix(), trafo.getInverse() );

  //  call the (overloadable) preTraverse() between stack adjustment and traversal
  if ( preTraverseTransform( &trafo ) )
  {
    SharedTraverser::handleTransform( p );

    //  call the (overloadable) postTraverse() between stack adjustment and traversal
    postTraverseTransform( &trafo );
  }

  // pop off view matrices after proceeding
  m_transformStack.popModelToWorld();
}

bool ExtractGeometryTraverser::preTraverseTransform( const Trafo *p )
{
  return( true );
}

void  ExtractGeometryTraverser::postTraverseTransform( const Trafo *p )
{
}

void  ExtractGeometryTraverser::handleGeoNode( const GeoNode * p )
{
  if ( p->getMaterialPipeline() )
  {
    dp::sg::core::PipelineDataSharedPtr const& ed = p->getMaterialPipeline();
    const ParameterGroupDataSharedPtr & smp = ed->findParameterGroupData( string( "standardMaterialParameters" ) );
    if ( smp )
    {
      dp::fx::ParameterGroupSpecSharedPtr pgs = smp->getParameterGroupSpec();

      // simply copy this material for later
      m_material.isMaterial = true;
      m_material.ambient  = smp->getParameter<Vec3f>( pgs->findParameterSpec( "frontAmbientColor" ) );
      m_material.diffuse  = smp->getParameter<Vec3f>( pgs->findParameterSpec( "frontDiffuseColor" ) );
      m_material.specular = smp->getParameter<Vec3f>( pgs->findParameterSpec( "frontSpecularColor" ) );
      m_material.exponent = smp->getParameter<float>( pgs->findParameterSpec( "frontSpecularExponent" ) );
      m_material.opacity  = smp->getParameter<float>( pgs->findParameterSpec( "frontOpacity" ) );
    }

    const ParameterGroupDataSharedPtr & stp = ed->findParameterGroupData( string( "standardTextureParameters" ) );
    if ( stp )
    {
      dp::fx::ParameterGroupSpecSharedPtr pgs = stp->getParameterGroupSpec();

      m_material.isTexture = true;
      const SamplerSharedPtr & sampler = stp->getParameter<SamplerSharedPtr>( pgs->findParameterSpec( "sampler" ) );
      if ( sampler )
      {
        const TextureSharedPtr & texture = sampler->getTexture();
        if ( texture && std::dynamic_pointer_cast<TextureHost>(texture) )
        {
          m_material.filename = std::static_pointer_cast<TextureHost>(texture)->getFileName();
        }
      }
    }
  }
  SharedTraverser::handleGeoNode( p );
}

//
// Here is where most of the work gets done
//
void
ExtractGeometryTraverser::traversePrimitive( const Primitive * p )
{
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
    case PrimitiveType::QUADS:
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
          if ( dp::math::distance( vertices[iter[i+0]], vertices[iter[i+2]] ) <=
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
          if ( dp::math::distance( vertices[offset + i + 0], vertices[offset + i + 2] ) <=
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

    case PrimitiveType::QUAD_STRIP:
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

          if ( dp::math::distance( vertices[iter[i-3]], vertices[iter[i-0]] ) <=
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
          if ( dp::math::distance( vertices[offset + i - 3], vertices[offset + i - 0] ) <=
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

    case PrimitiveType::POLYGON:
    case PrimitiveType::TRIANGLE_FAN:
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

    case PrimitiveType::TRIANGLE_STRIP:
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

    case PrimitiveType::TRIANGLES:
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
    const Mat44f modelToWorld    =  m_transformStack.getModelToWorld(); // model to world for the vertices.
    const Mat44f modelToWorld_IT = ~m_transformStack.getWorldToModel(); // inverse transpose for the normals.

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

    vector<Vec3f> transformedVerts( numVerts );
    vector<Vec3f> transformedNormals( numVerts );
    vector<Vec2f> transformedTexCoords( numVerts );

    for ( unsigned int i = 0; i < numVerts; i++ )
    {
      Vec4f v4f = Vec4f( vertices[offset + i], 1.0f ) * modelToWorld;
      transformedVerts[i] = Vec3f( v4f );
    }

    if( numNormals )
    {
      Buffer::ConstIterator<Vec3f>::Type normals = vas->getNormals();

      for ( unsigned int i = 0; i < numVerts; i++ )
      {
        Vec4f v4f = Vec4f( normals[offset + i], 0.0f ) * modelToWorld_IT;
        Vec3f v3f( v4f );
        v3f.normalize();
        transformedNormals[i] = v3f;
      }
    }
    else
    {
      for( unsigned int i = 0; i < numVerts; i++ )
      {
        // Hack: Use the opposite of the default lighting direction.
        transformedNormals[i] = Vec3f( 0.0f, 0.0f, 1.0f );
      }
    }

    if ( numTexCoords )
    {
      Buffer::ConstIterator<Vec2f>::Type texCoords = vas->getTexCoords<Vec2f>(0);

      for( unsigned int i = 0; i < numVerts; i++ )
      {
        transformedTexCoords[i] = texCoords[i];
      }
    }
    else
    {
      for ( unsigned int i = 0; i < numVerts; i++ )
      {
        transformedTexCoords[i] = Vec2f( 0.0f, 0.0f ); // kind of a hack for now
      }
    }

    submitIndexedTriangleSet( indices,
                              transformedVerts, transformedNormals, transformedTexCoords,
                              m_material );
  }
}

