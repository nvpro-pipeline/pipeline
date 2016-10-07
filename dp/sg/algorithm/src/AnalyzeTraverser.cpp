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


#include <set>
#include <sstream>
#include <vector>
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/algorithm/AnalyzeTraverser.h>
#include <dp/sg/algorithm/CombineTraverser.h>

using namespace dp::math;
using namespace dp::sg::core;
using namespace dp::util;
using std::map;
using std::pair;
using std::set;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      std::ostream& operator<<( std::ostream& os, const CombinableResult& r )
      {
        os << "CombinableResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of combinable objects: " << r.combinableCount << "\n";
        os << "\tnumber of objects if combined: " << r.reducedCount << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const DenormalizedNormalsResult& r )
      {
        os << "DenormalizedNormalsResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of objects with denormalized normals: " << r.denormalizedNormalsCount << "\n";
        os << "\tnumber of denormalized normals in those objects: " << r.denormalizedNormalsNumber << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const EmptyResult& r )
      {
        os << "EmptyResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of empty objects: " << r.emptyCount << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const EquivalenceResult& r )
      {
        os << "EquivalenceResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of equivalent objects: " << r.equivalentCount <<"\n";
        os << "\tnumber of objects if combined: " << r.reducedCount << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const IdentityResult& r )
      {
        os << "IdentityResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of identity Transforms: " << r.identityCount << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const MissingResult& r )
      {
        os << "MissingResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of incomplete objects: " << r.missingCount << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const NullNormalsResult& r )
      {
        os << "NullNormalsResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of objects with null normals: " << r.nullNormalsCount << "\n";
        os << "\tnumber of null normals in those objects: " << r.nullNormalsNumber << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const RedundantIndexSetResult& r )
      {
        os << "RedundantIndexSetResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of Primitives with redundant IndexSet: " << r.redundantIndexSetCount << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const RedundantLODRangesResult& r )
      {
        os << "RedundantLODRangesResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of LODs with redundant ranges: " << r.redundantLODs << "\n";
        os << "\tnumber of ranges: " << r.lodRanges << "\n";
        os << "\tnumber of redundant ranges: " << r.redundantLODRanges << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const SingleChildResult& r )
      {
        os << "SingleChildResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of objects with a single child: " << r.singleChildCount << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const ShortStrippedResult& r )
      {
        os << "ShortStrippedResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of objects with short strips: " << r.shortStrippedCount << "\n";
        os << "\tnumber of short strips in those objects: " << r.shortStrippedNumber << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const UnusedVerticesResult& r )
      {
        os << "UnusedVerticesResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        os << "\tnumber of unused vertices: " << r.unusedVerticesCount << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const AnalyzeResult& r )
      {
        os << "AnalyzeResult:\n";
        os << "\tobject code: " << objectCodeToName( r.objectCode ) << "\n";
        os << "\tnumber of objects: " << r.objectCount << "\n";
        return( os );
      }

      std::ostream& operator<<( std::ostream& os, const AnalyzeTraverser& at )
      {
        std::vector<AnalyzeResult *> results;
        unsigned int numberOfResults = at.getAnalysis( results );
        for ( unsigned int i=0 ; i<numberOfResults ; i++ )
        {
          if ( dynamic_cast<CombinableResult *>( results[i] ) != NULL )
          {
            os << *static_cast<CombinableResult*>( results[i] );
          }
          else if ( dynamic_cast<DenormalizedNormalsResult *>( results[i] ) != NULL )
          {
            os << *static_cast<DenormalizedNormalsResult*>( results[i] );
          }
          else if ( dynamic_cast<EmptyResult *>( results[i] ) != NULL )
          {
            os << *static_cast<EmptyResult*>( results[i] );
          }
          else if ( dynamic_cast<EquivalenceResult *>( results[i] ) != NULL )
          {
            os << *static_cast<EquivalenceResult*>( results[i] );
          }
          else if ( dynamic_cast<IdentityResult *>( results[i] ) != NULL )
          {
            os << *static_cast<IdentityResult*>( results[i] );
          }
          else if ( dynamic_cast<MissingResult *>( results[i] ) != NULL )
          {
            os << *static_cast<MissingResult*>( results[i] );
          }
          else if ( dynamic_cast<NullNormalsResult *>( results[i] ) != NULL )
          {
            os << *static_cast<NullNormalsResult*>( results[i] );
          }
          else if ( dynamic_cast<RedundantIndexSetResult *>( results[i] ) != NULL )
          {
            os << *static_cast<RedundantIndexSetResult*>( results[i] );
          }
          else if ( dynamic_cast<RedundantLODRangesResult *>( results[i] ) != NULL )
          {
            os << *static_cast<RedundantLODRangesResult*>( results[i] );
          }
          else if ( dynamic_cast<SingleChildResult *>( results[i] ) != NULL )
          {
            os << *static_cast<SingleChildResult*>( results[i] );
          }
          else if ( dynamic_cast<ShortStrippedResult *>( results[i] ) != NULL )
          {
            os << *static_cast<ShortStrippedResult*>( results[i] );
          }
          else if ( dynamic_cast<UnusedVerticesResult *>( results[i] ) != NULL )
          {
            os << *static_cast<UnusedVerticesResult*>( results[i] );
          }
          else if ( dynamic_cast<UnusedVerticesResult *>( results[i] ) != NULL )
          {
            os << *static_cast<UnusedVerticesResult*>( results[i] );
          }
          else
          {
            os << *results[i];
          }
          // delete the result after interpretation
          delete results[i];
        }
        return( os );
      }

      AnalyzeTraverser::AnalyzeTraverser( void )
      : m_currentLOD(NULL)
      {
      }

      AnalyzeTraverser::~AnalyzeTraverser( void )
      {
      }

      void AnalyzeTraverser::doApply( const NodeSharedPtr & root )
      {
        m_countMap.clear();
        m_emptyMap.clear();
        m_equivalenceMap.clear();
        m_lodRanges = 0;
        m_identityCount = 0;
        m_redundantIndexSets = 0;
        m_redundantLODs = 0;
        m_redundantLODRanges = 0;
        m_sharedObjects.clear();
        m_vertexUseMap.clear();

        SharedTraverser::doApply( root );
      }

      unsigned int AnalyzeTraverser::getAnalysis( vector<AnalyzeResult*> &results ) const
      {
        for ( map<dp::sg::core::ObjectCode,pair<unsigned int,unsigned int> >::const_iterator ci = m_combinableInfo.begin()
            ; ci != m_combinableInfo.end()
            ; ++ci )
        {
          CombinableResult * cr = new CombinableResult;
          cr->objectCode = ci->first;
          cr->objectCount = m_countMap.find( ci->first )->second;

          cr->combinableCount = ci->second.first + ci->second.second;
          cr->reducedCount = cr->objectCount - ci->second.second;
          results.push_back( cr );
        }

        if ( m_denormalizedNormalsVAS.first )
        {
          DenormalizedNormalsResult * dnr = new DenormalizedNormalsResult;
          dnr->objectCode = ObjectCode::VERTEX_ATTRIBUTE_SET;
          dnr->objectCount = m_countMap.find( dnr->objectCode )->second;

          dnr->denormalizedNormalsCount = m_denormalizedNormalsVAS.first;
          dnr->denormalizedNormalsNumber = m_denormalizedNormalsVAS.second;
          results.push_back( dnr );
        }

        for ( map<dp::sg::core::ObjectCode, unsigned int>::const_iterator ci = m_emptyMap.begin();
              ci != m_emptyMap.end();
              ++ci )
        {
          EmptyResult * er = new EmptyResult;
          er->objectCode  = ci->first;
          DP_ASSERT( m_countMap.find( ci->first ) != m_countMap.end() );
          er->objectCount = m_countMap.find( ci->first )->second;

          er->emptyCount = ci->second;
          results.push_back( er );
        }

        for ( map<dp::sg::core::ObjectCode, EquivalenceInfo>::const_iterator ci = m_equivalenceMap.begin();
              ci != m_equivalenceMap.end();
              ++ci )
        {
          if ( ! ci->second.equivalentObjects.empty() )
          {
            EquivalenceResult * er = new EquivalenceResult;
            er->objectCode  = ci->first;
            DP_ASSERT( m_countMap.find( ci->first ) != m_countMap.end() );
            er->objectCount = m_countMap.find( ci->first )->second;

            er->equivalentCount = dp::checked_cast<unsigned int>(ci->second.equivalentObjects.size());
            er->reducedCount    = dp::checked_cast<unsigned int>(ci->second.uniqueObjects.size());
            results.push_back( er );
          }
        }

        if ( m_identityCount )
        {
          IdentityResult * ir = new IdentityResult;
          ir->objectCode  = ObjectCode::TRANSFORM;
          DP_ASSERT( m_countMap.find( ObjectCode::TRANSFORM ) != m_countMap.end() );
          ir->objectCount = m_countMap.find( ObjectCode::TRANSFORM )->second;

          ir->identityCount = m_identityCount;
          results.push_back( ir );
        }

        for ( map<dp::sg::core::ObjectCode, unsigned int>::const_iterator ci = m_missingMap.begin();
              ci != m_missingMap.end();
              ++ci )
        {
          MissingResult * mr = new MissingResult;
          mr->objectCode  = ci->first;
          DP_ASSERT( m_countMap.find( ci->first ) != m_countMap.end() );
          mr->objectCount = m_countMap.find( ci->first )->second;

          mr->missingCount = ci->second;
          results.push_back( mr );
        }

        if ( m_nullNormalsVAS.first )
        {
          NullNormalsResult * nnr = new NullNormalsResult;
          nnr->objectCode = ObjectCode::VERTEX_ATTRIBUTE_SET;
          nnr->objectCount = m_countMap.find( nnr->objectCode )->second;

          nnr->nullNormalsCount = m_nullNormalsVAS.first;
          nnr->nullNormalsNumber = m_nullNormalsVAS.second;
          results.push_back( nnr );
        }

        if ( m_redundantIndexSets )
        {
          RedundantIndexSetResult * risr = new RedundantIndexSetResult;
          risr->objectCode = ObjectCode::PRIMITIVE;
          risr->objectCount = m_countMap.find( ObjectCode::PRIMITIVE )->second;
          risr->redundantIndexSetCount = m_redundantIndexSets;
          results.push_back( risr );
        }

        if ( m_redundantLODs )
        {
          RedundantLODRangesResult * rlodrr = new RedundantLODRangesResult;
          rlodrr->objectCode  = ObjectCode::LOD;
          rlodrr->objectCount = m_countMap.find( ObjectCode::LOD )->second;

          rlodrr->lodRanges           = m_lodRanges;
          rlodrr->redundantLODs       = m_redundantLODs;
          rlodrr->redundantLODRanges  = m_redundantLODRanges;
          results.push_back( rlodrr );
        }

        for ( map<dp::sg::core::ObjectCode, unsigned int>::const_iterator ci = m_singleChildMap.begin();
              ci != m_singleChildMap.end();
              ++ci )
        {
          SingleChildResult * scr = new SingleChildResult;
          scr->objectCode   = ci->first;
          DP_ASSERT( m_countMap.find( ci->first ) != m_countMap.end() );
          scr->objectCount  = m_countMap.find( ci->first )->second;

          scr->singleChildCount = ci->second;
          results.push_back( scr );
        }

        for ( map<dp::sg::core::ObjectCode, pair<unsigned int, unsigned int> >::const_iterator ci = m_shortStripped.begin();
              ci != m_shortStripped.end();
              ++ci )
        {
          ShortStrippedResult * ssr = new ShortStrippedResult;
          ssr->objectCode = ci->first;
          DP_ASSERT( m_countMap.find( ci->first ) != m_countMap.end() );
          ssr->objectCount  = m_countMap.find( ci->first )->second;

          ssr->shortStrippedCount = ci->second.first;
          ssr->shortStrippedNumber = ci->second.second;
          results.push_back( ssr );
        }

        UnusedVerticesResult * uvr = nullptr;
        for ( VertexUseMap::const_iterator ci = m_vertexUseMap.begin() ; ci != m_vertexUseMap.end() ; ++ci )
        {
          if ( ! ci->second.empty() )
          {
            if ( ! uvr )
            {
              uvr = new UnusedVerticesResult;
            }
            uvr->objectCount++;
            uvr->unusedVerticesCount += dp::checked_cast<unsigned int>(ci->second.size());
          }
        }
        if ( uvr != nullptr )
        {
          results.push_back( uvr );
        }

        return( dp::checked_cast<unsigned int>(results.size()) );
      }

      void AnalyzeTraverser::handleBillboard( const Billboard *p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[p->getObjectCode()]++;
          analyzeCombinable( p );
          analyzeEmpty( p, p->getNumberOfChildren() );
          analyzeEquivalent( p );
          SharedTraverser::handleBillboard( p );
        }
      }

      void AnalyzeTraverser::handleGeoNode( const GeoNode * p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[p->getObjectCode()]++;
          analyzeEmpty( p, p->getPrimitive() ? 1 : 0 );
          analyzeEquivalent( p );
          SharedTraverser::handleGeoNode( p );
        }
      }

      void AnalyzeTraverser::handleGroup( const Group *p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[p->getObjectCode()]++;
          analyzeCombinable( p );
          analyzeEmpty( p, p->getNumberOfChildren() );
          analyzeEquivalent( p );
          analyzeSingleChild( p );
          SharedTraverser::handleGroup( p );
        }
      }

      void AnalyzeTraverser::handleIndexSet( const IndexSet * p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[p->getObjectCode()]++;
          SharedTraverser::handleIndexSet( p );
          analyzeEmpty( p, p->getNumberOfIndices() );
          analyzeEquivalent( p );
        }
      }

      void AnalyzeTraverser::handleLOD( const LOD *p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[p->getObjectCode()]++;
          analyzeEmpty( p, p->getNumberOfChildren() );
          analyzeEquivalent( p );
          analyzeRedundantLODRanges( p );
          analyzeSingleChild( p );
          m_currentLOD = p;
          SharedTraverser::handleLOD( p );
          m_currentLOD = NULL;
        }
      }

      void AnalyzeTraverser::handleParameterGroupData( const ParameterGroupData * p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[p->getObjectCode()]++;
          analyzeEquivalentParameterGroupSpec( p->getParameterGroupSpec() );
          analyzeEquivalent( p );
          SharedTraverser::handleParameterGroupData( p );
        }
      }

      void AnalyzeTraverser::handlePipelineData( const dp::sg::core::PipelineData * p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[p->getObjectCode()]++;
          analyzeEquivalentEffectSpec( p->getEffectSpec() );
          analyzeEquivalent( p );
          SharedTraverser::handlePipelineData( p );
        }
      }

      void AnalyzeTraverser::handleSampler( const Sampler * p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[p->getObjectCode()]++;
          analyzeEquivalent( p );
          analyzeMissing( p, p->getTexture() );
          SharedTraverser::handleSampler( p );
        }
      }

      void AnalyzeTraverser::handleSwitch( const Switch * p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[p->getObjectCode()]++;
          analyzeEmpty( p, p->getNumberOfChildren() );
          analyzeEquivalent( p );
          SharedTraverser::handleSwitch( p );
        }
      }

      void AnalyzeTraverser::handleTransform( const Transform * p )
      {
        if ( isToBeHandled( p ) )
        {
          analyzeCombinable( p );
          m_countMap[p->getObjectCode()]++;
          if ( ! p->isJoint() )
          {
            // only non-joints can be considered to be empty
            analyzeEmpty( p, p->getNumberOfChildren() );
          }
          analyzeEquivalent( p );
          if ( isIdentity( p->getTrafo().getMatrix() ) )
          {
            m_identityCount++;
          }
          SharedTraverser::handleTransform( p );
        }
      }

      void AnalyzeTraverser::handleVertexAttributeSet( const VertexAttributeSet * p )
      {
        if ( isToBeHandled( p ) )
        {
          testVertexAttributeSet( p );
          SharedTraverser::handleVertexAttributeSet( p );
        }
      }

      void AnalyzeTraverser::traversePrimitive( const Primitive * p )
      {
        if ( isToBeHandled( p ) )
        {
          m_countMap[ObjectCode::PRIMITIVE]++;
          analyzeEmpty( p, p->getElementCount() );
          analyzeEquivalent( p );
          analyzeRedundantIndexSet( p->getIndexSet(), p->getElementOffset(), p->getElementCount() );
          analyzeUnusedVertices( p->getIndexSet(), p->getVertexAttributeSet(), p->getElementOffset(), p->getElementCount() );
          SharedTraverser::traversePrimitive( p );
        }
      }

      void AnalyzeTraverser::analyzeCombinable( const Group * p )
      {
        map<GeoNodeSharedPtr,vector<GeoNodeSharedPtr> >     geoNodes;
        map<LODSharedPtr,vector<LODSharedPtr> >             lods;
        map<TransformSharedPtr,vector<TransformSharedPtr> > transforms;

        for ( Group::ChildrenConstIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( std::dynamic_pointer_cast<GeoNode>(*gci) )
          {
            GeoNodeSharedPtr geoNode = std::static_pointer_cast<GeoNode>(*gci);

            map<GeoNodeSharedPtr,vector<GeoNodeSharedPtr> >::iterator it;
            for ( it = geoNodes.begin() ; it != geoNodes.end() ; ++it )
            {
              if ( CombineTraverser::areCombinable( geoNode, it->first, true ) )
              {
                DP_ASSERT( *gci != it->first );
                it->second.push_back( geoNode );
                break;
              }
            }
            if ( it == geoNodes.end() )
            {
              geoNodes[geoNode];
            }
          }
          else if ( std::dynamic_pointer_cast<LOD>(*gci) )
          {
            LODSharedPtr lod = std::static_pointer_cast<LOD>(*gci);

            map<LODSharedPtr,vector<LODSharedPtr> >::iterator it;
            for ( it = lods.begin() ; it != lods.end() ; ++it )
            {
              if ( CombineTraverser::areCombinable( lod, it->first, true ) )
              {
                DP_ASSERT( *gci != it->first );
                it->second.push_back( lod );
                break;
              }
            }
            if ( it == lods.end() )
            {
              lods[lod];
            }
          }
          else if ( std::dynamic_pointer_cast<Transform>(*gci) )
          {
            TransformSharedPtr transform = std::static_pointer_cast<Transform>(*gci);

            map<TransformSharedPtr,vector<TransformSharedPtr> >::iterator it;
            for ( it = transforms.begin() ; it != transforms.end() ; ++it )
            {
              if ( CombineTraverser::areCombinable( transform, it->first, true ) )
              {
                DP_ASSERT( *gci != it->first );
                it->second.push_back( transform );
                break;
              }
            }
            if ( it == transforms.end() )
            {
              transforms[transform];
            }
          }
        }

        for ( map<GeoNodeSharedPtr,vector<GeoNodeSharedPtr> >::const_iterator it = geoNodes.begin() ; it != geoNodes.end() ; ++it )
        {
          if ( ! it->second.empty() )
          {
            m_combinableInfo[ObjectCode::GEO_NODE].first++;
            m_combinableInfo[ObjectCode::GEO_NODE].second += dp::checked_cast<unsigned int>(it->second.size());
          }
        }
        for ( map<LODSharedPtr,vector<LODSharedPtr> >::const_iterator it = lods.begin() ; it != lods.end() ; ++it )
        {
          if ( ! it->second.empty() )
          {
            m_combinableInfo[ObjectCode::LOD].first++;
            m_combinableInfo[ObjectCode::LOD].second += dp::checked_cast<unsigned int>(it->second.size());
          }
        }
        for ( map<TransformSharedPtr,vector<TransformSharedPtr> >::const_iterator it = transforms.begin() ; it != transforms.end() ; ++it )
        {
          if ( ! it->second.empty() )
          {
            m_combinableInfo[ObjectCode::TRANSFORM].first++;
            m_combinableInfo[ObjectCode::TRANSFORM].second += dp::checked_cast<unsigned int>(it->second.size());
          }
        }
      }

      void AnalyzeTraverser::analyzeEmpty( const Object * p, unsigned int numberOfElements )
      {
        if ( ( 0 == numberOfElements ) && !isChildOfCurrentLOD( p ) )
        {
          m_emptyMap[p->getObjectCode()]++;
        }
      }

      void AnalyzeTraverser::analyzeEquivalent( const Object *p )
      {
        EquivalenceInfo & ei = m_equivalenceMap[p->getObjectCode()];
        bool foundEquivalent = false;
        for ( size_t i=0 ; i<ei.uniqueObjects.size() ; i++ )
        {
          foundEquivalent =   ( p->getSharedPtr<dp::sg::core::Object>() != ei.uniqueObjects[i] )
                          &&  p->isEquivalent( ei.uniqueObjects[i] );
          if ( foundEquivalent )
          {
            ei.equivalentObjects.insert( p->getSharedPtr<dp::sg::core::Object>() );
            ei.equivalentObjects.insert( ei.uniqueObjects[i] );
            break;
          }
        }
        if ( ! foundEquivalent )
        {
          ei.uniqueObjects.push_back( p->getSharedPtr<dp::sg::core::Object>() );
        }
      }

      void AnalyzeTraverser::analyzeEquivalentEffectSpec( const dp::fx::EffectSpecSharedPtr & p )
      {
        EffectSpecEquivalenceInfo & ei = m_effectSpecEquivalenceInfo;
        bool foundEquivalent = false;
        for ( size_t i=0 ; i<ei.uniqueSpecs.size() ; i++ )
        {
          foundEquivalent =   ( p != ei.uniqueSpecs[i] )
                          &&  p->isEquivalent( ei.uniqueSpecs[i], false, false );
          if ( foundEquivalent )
          {
            ei.equivalentSpecs.insert( p );
            ei.equivalentSpecs.insert( ei.uniqueSpecs[i] );
            break;
          }
        }
        if ( ! foundEquivalent )
        {
          ei.uniqueSpecs.push_back( p );
        }
      }

      void AnalyzeTraverser::analyzeEquivalentParameterGroupSpec( const dp::fx::ParameterGroupSpecSharedPtr & p )
      {
        ParameterGroupSpecEquivalenceInfo & ei = m_parameterGroupSpecEquivalenceInfo;
        bool foundEquivalent = false;
        for ( size_t i=0 ; i<ei.uniqueSpecs.size() ; i++ )
        {
          foundEquivalent =   ( p != ei.uniqueSpecs[i] )
                          &&  p->isEquivalent( ei.uniqueSpecs[i], false, false );
          if ( foundEquivalent )
          {
            ei.equivalentSpecs.insert( p );
            ei.equivalentSpecs.insert( ei.uniqueSpecs[i] );
            break;
          }
        }
        if ( ! foundEquivalent )
        {
          ei.uniqueSpecs.push_back( p );
        }
      }

      void AnalyzeTraverser::analyzeMissing( const Object * p, dp::sg::core::TextureSharedPtr const& ptr )
      {
        if ( ! ptr )
        {
          m_missingMap[p->getObjectCode()]++;
        }
      }

      void AnalyzeTraverser::analyzeNormalsNormalized( Buffer::ConstIterator<Vec3f>::Type normals, unsigned int non
                                                     , unsigned int &nullNormals, unsigned int &denormalizedNormals )
      {
        for ( unsigned int i=0 ; i<non ; i++ )
        {
          if ( ! isNormalized( normals[i] ) )
          {
            if ( isNull( normals[i] ) )
            {
              nullNormals++;
            }
            else
            {
              denormalizedNormals++;
            }
          }
        }
      }

      void AnalyzeTraverser::analyzeRedundantIndexSet( const IndexSetSharedPtr & p, unsigned int offset, unsigned int count )
      {
        if ( p )
        {
          DP_ASSERT( offset + count <= p->getNumberOfIndices() );
          IndexSet::ConstIterator<unsigned int> indices( p, offset );
          set<unsigned int> checkedIndices;
          bool redundant = true;
          for ( unsigned int i=0 ; i<count && redundant ; i++ )
          {
            pair<set<unsigned int>::iterator,bool> r = checkedIndices.insert( indices[i] );
            redundant = r.second;
          }
          if ( redundant && ( 1 < checkedIndices.size() ) )
          {
            set<unsigned int>::const_iterator next = checkedIndices.begin();
            for ( set<unsigned int>::const_iterator curr = next++ ; next != checkedIndices.end() && redundant ; curr++, next++ )
            {
              redundant = ( *curr + 1 == *next );
            }
            if ( redundant )
            {
              m_redundantIndexSets++;
            }
          }
        }
      }

      void AnalyzeTraverser::analyzeRedundantLODRanges( const LOD * p )
      {
        unsigned int redundant = 0;
        if ( 1 < p->getNumberOfChildren() )
        {
          for ( Group::ChildrenConstIterator gcciPrev = p->beginChildren(), gcci = ++p->beginChildren()
              ; gcci != p->endChildren()
              ; ++gcciPrev, ++gcci )
          {
            if ( *gcciPrev == *gcci )
            {
              redundant++;
            }
          }
        }
        if ( redundant )
        {
          m_redundantLODs++;
          m_redundantLODRanges += redundant;
        }
        m_lodRanges += p->getNumberOfChildren();
      }

      void AnalyzeTraverser::analyzeSingleChild( const Group * p )
      {
        // A Group with just one child is considered to be a single-child-group, and thus candidate for
        // optimizations, if
        //  - it holds no light references
        //  - it holds no clip planes
        //  - it's single child is not a joint
        if (    ( 1 == p->getNumberOfChildren() )
            &&  ( 0 == p->getNumberOfClipPlanes() )
            &&  (   std::dynamic_pointer_cast<Transform>(*p->beginChildren())
                &&  !std::static_pointer_cast<Transform>(*p->beginChildren())->isJoint() ) )
        {
          m_singleChildMap[p->getObjectCode()]++;
        }
      }

      void AnalyzeTraverser::analyzeUnusedVertices( const IndexSetSharedPtr & isSP
                                                  , const VertexAttributeSetSharedPtr & vasSP
                                                  , unsigned int offset, unsigned int count )
      {
        DP_ASSERT( vasSP );

        unsigned int nov = vasSP->getNumberOfVertices();

        if ( isSP )
        {
          VertexUseMap::iterator it = m_vertexUseMap.find( vasSP );
          if ( it == m_vertexUseMap.end() )
          {
            // if this VertexAttributeSet hasn't been encounterd before, put it in the map and initialize the set with
            // all indices from 0 to vas->getNumberOfVertices()
            pair<VertexUseMap::iterator,bool> pitb = m_vertexUseMap.insert( make_pair( vasSP, set<unsigned int>() ) );
            DP_ASSERT( pitb.second );
            it = pitb.first;
            for ( unsigned int i=0 ; i<nov ; i++)
            {
              it->second.insert( i );
            }
          }
          if ( ! it->second.empty() )
          {
            // The VertexAttributeSet hasn't been completely used, yet
            IndexSet::ConstIterator<unsigned int> isci( isSP, offset );
            for ( unsigned int i=0 ; i<count ; i++ )
            {
              it->second.erase( isci[i] );
            }
          }
        }
        else if ( ( offset != 0 ) || ( count != nov ) )
        {
          VertexUseMap::iterator it = m_vertexUseMap.find( vasSP );
          if ( it == m_vertexUseMap.end() )
          {
            // if this VertexAttributeSet hasn't been encounterd before, put it in the map and initialize the set with
            // all indices from 0 to offset, and from offset + count to nov
            pair<VertexUseMap::iterator,bool> pitb = m_vertexUseMap.insert( make_pair( vasSP, set<unsigned int>() ) );
            DP_ASSERT( pitb.second );
            it = pitb.first;
            for ( unsigned int i=0 ; i<offset ; i++)
            {
              it->second.insert( i );
            }
            for ( unsigned int i=offset+count ; i<nov ; i++ )
            {
              it->second.insert( i );
            }
          }
          if ( ! it->second.empty() )
          {
            // The VertexAttributeSet hasn't been completely used, yet
            DP_ASSERT( offset + count <= nov );
            for ( unsigned int i=0 ; i<count ; i++ )
            {
              it->second.erase( offset+i );
            }
          }
        }
        else
        {
          // when there is no IndexSet, and no offset/count, the complete VertexAttributeSet is used -> mark it as clear
          m_vertexUseMap[vasSP].clear();
        }
      }

      bool AnalyzeTraverser::isChildOfCurrentLOD( const Object * p )
      {
        if ( m_currentLOD )
        {
          for ( Group::ChildrenConstIterator gcci = m_currentLOD->beginChildren() ; gcci != m_currentLOD->endChildren() ; ++gcci )
          {
            if ( *gcci == p->getSharedPtr<dp::sg::core::Node>() )
            {
              return( true );
            }
          }
        }
        return( false );
      }

      bool AnalyzeTraverser::isToBeHandled( const Object *p )
      {
        return( m_sharedObjects.insert( p ).second );
      }

      void AnalyzeTraverser::testVertexAttributeSet( const VertexAttributeSet * p )
      {
        m_countMap[p->getObjectCode()]++;
        analyzeEmpty( p, p->getNumberOfVertices() );
        analyzeEquivalent( p );
        if ( p->getNumberOfNormals() )
        {
          unsigned int nullNormals(0), denormalizedNormals(0);

          analyzeNormalsNormalized( p->getNormals(), p->getNumberOfNormals(), nullNormals
            , denormalizedNormals );
          if ( nullNormals )
          {
            m_nullNormalsVAS.first++;
            m_nullNormalsVAS.second += nullNormals;
          }
          if ( denormalizedNormals )
          {
            m_denormalizedNormalsVAS.first++;
            m_denormalizedNormalsVAS.second += denormalizedNormals;
          }
        }
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
