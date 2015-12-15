// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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


#include "DisplayNormalsTraverser.h"
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/Scene.h>

DisplayNormalsTraverser::DisplayNormalsTraverser()
: m_normalLength(10.0f)
{
  m_trafo.setMatrix(dp::math::cIdentity44f);

  m_material = dp::sg::core::createStandardMaterialData();
  m_material->setName("SceniX Normals");
}

DisplayNormalsTraverser::~DisplayNormalsTraverser(void)
{
}

void DisplayNormalsTraverser::doApply( const dp::sg::core::NodeSharedPtr & root )
{
  DP_ASSERT( m_scene->getRootNode() == root );

  m_normalsGeoNodes.push( std::vector<std::pair<dp::sg::core::GeoNodeSharedPtr,dp::sg::core::GeoNodeSharedPtr> >() );

  ExclusiveTraverser::doApply( root );

  if ( !m_normalsGeoNodes.top().empty() )
  {
    DP_ASSERT( root.isPtrTo<dp::sg::core::GeoNode>() );
    DP_ASSERT( root->getName() != NORMALS_NAME );
    DP_ASSERT( m_normalsGeoNodes.top().size() == 1 );
    DP_ASSERT( m_normalsGeoNodes.top()[0].first == root );

    dp::sg::core::GroupSharedPtr newGroup = dp::sg::core::Group::create();
    newGroup->setName( NORMALS_NAME );
    newGroup->addChild( m_normalsGeoNodes.top()[0].first );   // original child
    newGroup->addChild( m_normalsGeoNodes.top()[0].second );  // normals

    m_scene->setRootNode( newGroup );
  }
  m_normalsGeoNodes.pop();
  DP_ASSERT( m_normalsGeoNodes.empty() );
}

void DisplayNormalsTraverser::setNormalColor( dp::math::Vec3f & color )
{
  const dp::sg::core::ParameterGroupDataSharedPtr & parameterGroupData = m_material->findParameterGroupData( std::string( "standardMaterialParameters" ) );
  DP_ASSERT( parameterGroupData );
  dp::math::Vec3f black( 0.0f, 0.0f, 0.0f );
  DP_VERIFY( parameterGroupData->setParameter( "frontEmissiveColor", color ) );
  DP_VERIFY( parameterGroupData->setParameter( "frontAmbientColor", black ) );
  DP_VERIFY( parameterGroupData->setParameter( "frontDiffuseColor", black ) );
  DP_VERIFY( parameterGroupData->setParameter( "backEmissiveColor", color ) );
  DP_VERIFY( parameterGroupData->setParameter( "backAmbientColor", black ) );
  DP_VERIFY( parameterGroupData->setParameter( "backDiffuseColor", black ) );
}

void DisplayNormalsTraverser::handleTransform( dp::sg::core::Transform * p )
{
  DP_ASSERT( m_normalsPrimitives.empty() );

  m_normalsGeoNodes.push( std::vector<std::pair<dp::sg::core::GeoNodeSharedPtr,dp::sg::core::GeoNodeSharedPtr> >() );
  m_normalsGroups.push( std::vector<dp::sg::core::GroupSharedPtr>() );

  // append this transform's matrix to the global trafo
  dp::math::Mat44f temp = m_trafo.getMatrix();
  m_trafo.setMatrix(p->getTrafo().getMatrix() * temp);

  // continue traversing down the tree with the appended trafo
  ExclusiveTraverser::handleTransform( p );

  checkNormals( p );

  // std::set the global trafo back to what is was on the way back up
  m_trafo.setMatrix(temp);
}

void DisplayNormalsTraverser::handleGroup( dp::sg::core::Group * p )
{
  DP_ASSERT( m_normalsPrimitives.empty() );
  m_normalsGeoNodes.push( std::vector<std::pair<dp::sg::core::GeoNodeSharedPtr,dp::sg::core::GeoNodeSharedPtr> >() );
  m_normalsGroups.push( std::vector<dp::sg::core::GroupSharedPtr>() );
  ExclusiveTraverser::handleGroup( p );
  checkNormals( p );
}

void DisplayNormalsTraverser::handleSwitch( dp::sg::core::Switch * p )
{
  DP_ASSERT( m_normalsPrimitives.empty() );
  m_normalsGeoNodes.push( std::vector<std::pair<dp::sg::core::GeoNodeSharedPtr,dp::sg::core::GeoNodeSharedPtr> >() );
  m_normalsGroups.push( std::vector<dp::sg::core::GroupSharedPtr>() );
  ExclusiveTraverser::handleSwitch( p );
  checkNormals( p );
}

void DisplayNormalsTraverser::handleLOD( dp::sg::core::LOD * p )
{
  DP_ASSERT( m_normalsPrimitives.empty() );
  m_normalsGeoNodes.push( std::vector<std::pair<dp::sg::core::GeoNodeSharedPtr,dp::sg::core::GeoNodeSharedPtr> >() );
  m_normalsGroups.push( std::vector<dp::sg::core::GroupSharedPtr>() );
  ExclusiveTraverser::handleLOD( p );
  checkNormals( p );
}

void DisplayNormalsTraverser::checkNormals( dp::sg::core::Group * p )
{
  if ( p->getName() == NORMALS_NAME )
  {
    // This is an inserted dp::sg::core::Group, holding an original dp::sg::core::GeoNode and a corresponding normals dp::sg::core::GeoNode
    DP_ASSERT( p->getNumberOfChildren() == 2 );
    DP_ASSERT( (*(++p->beginChildren()))->getName() == NORMALS_NAME );
    DP_ASSERT( m_normalsPrimitives.size() <= 1 );
    DP_ASSERT( m_normalsGroups.top().empty() );
    m_normalsGroups.pop();    // pop the m_normalsGroups before potentially adding some for the level above

    if ( std::numeric_limits<float>::epsilon() < m_normalLength )
    {
      // Normals are to be used
      if ( ! m_normalsGeoNodes.top().empty() )
      {
        // a potentially newer version of the normals dp::sg::core::GeoNode has been created in handleGeoNode
        DP_ASSERT( m_normalsGeoNodes.top().size() == 1 );
        DP_ASSERT( *(p->beginChildren()) == m_normalsGeoNodes.top()[0].first );
        dp::sg::core::NodeSharedPtr oldNormals = *(++p->beginChildren());
        DP_VERIFY( p->replaceChild( m_normalsGeoNodes.top()[0].second, oldNormals ) );
      }
      else
      {
        // there is no new normals dp::sg::core::GeoNode -> remove the dp::sg::core::Primitive of the current one as outdated
        (++p->beginChildren())->staticCast<dp::sg::core::GeoNode>()->setPrimitive( dp::sg::core::PrimitiveSharedPtr() );
      }
    }
    else
    {
      // no normals to use -> store this dp::sg::core::Group as a candidate to remove
      m_normalsGroups.top().push_back( p->getSharedPtr<dp::sg::core::Group>() );
    }
  }
  else
  {
    // This is an original dp::sg::core::Group -> replace any dp::sg::core::GeoNode by a dp::sg::core::Group holding that dp::sg::core::GeoNode and a
    // corresponding normals dp::sg::core::GeoNode
    if ( std::numeric_limits<float>::epsilon() < m_normalLength )
    {
      // Normals are supposed to be used -> do the replacements determined in handleGeoNode
      for ( size_t i=0 ; i<m_normalsGeoNodes.top().size() ; i++ )
      {
        dp::sg::core::GroupSharedPtr newGroup = dp::sg::core::Group::create();
        newGroup->setName( NORMALS_NAME );
        newGroup->addChild( m_normalsGeoNodes.top()[i].first );   // original child
        newGroup->addChild( m_normalsGeoNodes.top()[i].second );  // normals
        DP_VERIFY( p->replaceChild( newGroup, m_normalsGeoNodes.top()[i].first ) );
      }
    }
    else
    {
      // no normals to use -> replace any normals groups by their first child (being the original dp::sg::core::GeoNode)
      for ( std::vector<dp::sg::core::GroupSharedPtr>::const_iterator it = m_normalsGroups.top().begin() ; it != m_normalsGroups.top().end() ; ++it )
      {
        dp::sg::core::NodeSharedPtr originalNode = *(*it)->beginChildren();
        DP_VERIFY( p->replaceChild( originalNode, *it ) );
      }
    }
    m_normalsGroups.pop();    // the m_normalsGroups has been handled -> pop to level above
  }
  m_normalsGeoNodes.pop();
}

void DisplayNormalsTraverser::handleGeoNode( dp::sg::core::GeoNode *p )
{
  if ( p->getName() != NORMALS_NAME )
  {
    if ( std::numeric_limits<float>::epsilon() < m_normalLength )
    {
      m_indices.clear();

      ExclusiveTraverser::handleGeoNode( p );

      // if this dp::sg::core::GeoNode has any vertices, we will display their normals
      std::vector<dp::math::Vec3f> linesVerts;
      if ( m_indices.size() )
      {
        // figure out the total number of vertices in this dp::sg::core::GeoNode
        unsigned int totalVerts = 0;
        std::map<dp::sg::core::VertexAttributeSetSharedPtr,std::set<unsigned int> >::iterator iter;
        for ( iter = m_indices.begin() ; iter != m_indices.end() ; ++iter )
        {
          DP_ASSERT( iter->first->getNumberOfVertices() && ( iter->first->getNumberOfVertices() == iter->first->getNumberOfNormals() ) );
          totalVerts += iter->first->getNumberOfVertices();
        }

        // check if there is scaling in the transform above
        bool scaled = false;
        dp::math::Vec3f scaling = m_trafo.getScaling();
        if ( scaling[0] != 1.0f || scaling[1] != 1.0f || scaling[2] != 1.0f )
        {
          scaled = true;

          // invert the scaling std::vector
          for ( unsigned int i=0 ; i<3 ; i++ )
          {
            if ( scaling[i] != 0.0f )
            {
              scaling[i] = 1.0f / scaling[i];
            }
          }
        }

        // a list storing all vertices for the visualized normals
        linesVerts.reserve( 2 * totalVerts );

        // add each std::set of vertices to the list
        for ( iter = m_indices.begin(); iter != m_indices.end(); ++iter )
        {
          dp::sg::core::Buffer::ConstIterator<dp::math::Vec3f>::Type verts = iter->first->getVertices();
          unsigned int nVertices = iter->first->getNumberOfVertices();

          dp::sg::core::Buffer::ConstIterator<dp::math::Vec3f>::Type norms = iter->first->getNormals();
          unsigned int nNormals = iter->first->getNumberOfNormals();

          // calculate the normal line for each vertex and store it in the Segment list
          for ( std::set<unsigned int>::iterator verIt(iter->second.begin()); verIt != iter->second.end(); ++verIt )
          {
            dp::math::Vec3f v = verts[*verIt];
            dp::math::Vec3f n = norms[*verIt];
            n.normalize();

            // don't add this normal if it's null
            if ( isNull( n ) )
            {
              continue;
            }

            // store the root vertex of the normal line in the list
            linesVerts.push_back( v );

            dp::math::Vec3f offset = m_normalLength * n;
            // if the transform above is scaled, apply the inverse transform
            if(scaled)
            {
              offset[0] *= scaling[0];
              offset[1] *= scaling[1];
              offset[2] *= scaling[2];
            }

            // calculate the other point of the normal line, which is m_normalLength away from the vertex
            linesVerts.push_back( v + offset );
          }
        }
      }

      // if we've added any vertices, create the Lines object to represent the normals
      if ( ! linesVerts.empty() )
      {
        // create a dp::sg::core::VertexAttributeSet to contain the vertices for the Lines
        dp::sg::core::VertexAttributeSetSharedPtr normVas = dp::sg::core::VertexAttributeSet::create();
        normVas->setVertices( &linesVerts[0], dp::checked_cast<unsigned int>( linesVerts.size() ) );

        // create a Lines dp::sg::core::Primitive to hold the normals
        dp::sg::core::PrimitiveSharedPtr normals = dp::sg::core::Primitive::create( dp::sg::core::PrimitiveType::LINES );
        normals->setName(NORMALS_NAME);
        normals->setVertexAttributeSet( normVas );

        dp::sg::core::GeoNodeSharedPtr newGeoNode = dp::sg::core::GeoNode::create();
        newGeoNode->setName( NORMALS_NAME );
        newGeoNode->setMaterialPipeline( m_material );
        newGeoNode->setPrimitive( normals );

        m_normalsGeoNodes.top().push_back( std::make_pair( p->getSharedPtr<dp::sg::core::GeoNode>(), newGeoNode ) );
      }
    }
  }
}

void DisplayNormalsTraverser::handlePrimitive( dp::sg::core::Primitive * p )
{
  DP_ASSERT( p->getName() != NORMALS_NAME );
  DP_ASSERT( std::numeric_limits<float>::epsilon() < m_normalLength );

  dp::sg::core::VertexAttributeSetSharedPtr const& vas = p->getVertexAttributeSet();
  if ( vas->getNumberOfVertices() && ( vas->getNumberOfVertices() == vas->getNumberOfNormals() ) )
  {
    // check if we already have a VAS, if so, use that usedVertex std::set as base
    std::set<unsigned int> emptySet;
    std::map<dp::sg::core::VertexAttributeSetSharedPtr,std::set<unsigned int> >::iterator it = m_indices.find( p->getVertexAttributeSet() );
    std::set<unsigned int> & usedVertices = ( it != m_indices.end() ) ? it->second : emptySet;
    if ( usedVertices.size() == vas->getNumberOfVertices() )
    {
      //no need to handle this any further
      return;
    }

    unsigned int offset = p->getElementOffset();
    unsigned int count = p->getElementCount();
    if ( p->getIndexSet() )
    {
      dp::sg::core::IndexSet::ConstIterator<unsigned int> isit(p->getIndexSet(),offset);
      for ( unsigned int i=0; i<count; ++i )
      {
        usedVertices.insert( isit[i] );
      }
      //removing primitive restart index entry
      usedVertices.erase( p->getIndexSet()->getPrimitiveRestartIndex() );
    }
    else
    {
      for ( unsigned int i=offset; i<offset+count; ++i )
      {
        usedVertices.insert( i );
      }
    }

    // add the Lines' vertex std::set to the global list so their normals can be displayed
    m_indices.insert(std::make_pair( p->getVertexAttributeSet(), usedVertices ));
  }
}
