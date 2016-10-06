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
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/algorithm/Search.h>
#include <dp/sg/algorithm/UnifyTraverser.h>
#include <dp/util/Memory.h>

#include <thread>

#define CHECK_HASH_RESULTS  0

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using std::set;
using std::vector;
using std::list;
using std::map;
using std::multimap;
using std::pair;
using std::make_pair;
using std::string;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {
      DEFINE_STATIC_PROPERTY( UnifyTraverser, UnifyTargets );
      DEFINE_STATIC_PROPERTY( UnifyTraverser, Epsilon );

      BEGIN_REFLECTION_INFO( UnifyTraverser )
        DERIVE_STATIC_PROPERTIES( UnifyTraverser, OptimizeTraverser );
        INIT_STATIC_PROPERTY_RW( UnifyTraverser, UnifyTargets, TargetMask,  Semantic::VALUE, value, value );
        INIT_STATIC_PROPERTY_RW( UnifyTraverser, Epsilon,      float,       Semantic::VALUE, value, value );
      END_REFLECTION_INFO

      UnifyTraverser::UnifyTraverser( void )
      : m_epsilon(std::numeric_limits<float>::epsilon())
      , m_unifyTargets(Target::ALL)
      {
      }

      UnifyTraverser::~UnifyTraverser( void )
      {
      }

      void UnifyTraverser::doApply( const NodeSharedPtr & root )
      {
        DP_ASSERT(   m_pipelineData.empty() && m_geoNodes.empty() && m_groups.empty() && m_indexSets.empty()
                    && m_LODs.empty() && m_parameterGroupData.empty() && m_primitives.empty() && m_samplers.empty()
                    && m_textures.empty() && m_vertexAttributeSets.empty() );

        if (m_unifyTargets & Target::VERTICES)
        {
          std::vector<dp::sg::core::ObjectSharedPtr> results = dp::sg::algorithm::searchClass(root, "class dp::sg::core::VertexAttributeSet");
          m_unifyVerticesIndex = 0;

          unsigned int threadCount = std::min<unsigned int>(std::thread::hardware_concurrency(), dp::checked_cast<unsigned int>(results.size()));
          std::vector<std::thread> threads;
          for (unsigned int i = 0; i < threadCount; i++)
          {
            threads.push_back(std::thread(&UnifyTraverser::unifyVerticesThreadFunction, this, results));
          }
          for (unsigned int i = 0; i < threadCount; i++)
          {
            DP_ASSERT(threads[i].joinable());
            threads[i].join();
          }
        }

        OptimizeTraverser::doApply( root );

        m_geoNodes.clear();
        m_groups.clear();
        m_indexSets.clear();
        m_parameterGroupData.clear();
        m_pipelineData.clear();
        m_primitives.clear();
        m_LODs.clear();
        m_objects.clear();
        m_samplers.clear();
        m_textures.clear();
        m_vertexAttributeSets.clear();
      }

      void UnifyTraverser::handleBillboard( Billboard *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleBillboard( p );
          unifyChildren( p );
        }
      }

      void UnifyTraverser::handleGeoNode( GeoNode *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleGeoNode( p );

          if ( optimizationAllowed( p->getSharedPtr<GeoNode>() ) )
          {
            if ( ( m_unifyTargets & Target::PIPELINE_DATA ) && p->getMaterialPipeline() )
            {
              const dp::sg::core::PipelineDataSharedPtr & replacement = unifyPipelineData( p->getMaterialPipeline() );
              if ( replacement )
              {
                p->setMaterialPipeline( replacement );
              }
            }
            if ( ( m_unifyTargets & Target::PRIMITIVE ) && m_replacementPrimitive )
            {
              p->setPrimitive( m_replacementPrimitive );
              m_replacementPrimitive.reset();
            }
          }
        }
      }

      void UnifyTraverser::handlePipelineData( dp::sg::core::PipelineData * p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handlePipelineData( p );
          if ( optimizationAllowed( p->getSharedPtr<dp::sg::core::PipelineData>() ) && ( m_unifyTargets & Target::PARAMETER_GROUP_DATA ) )
          {
            const dp::fx::EffectSpecSharedPtr & es = p->getEffectSpec();
            for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
            {
              const ParameterGroupDataSharedPtr & parameterGroupData = p->getParameterGroupData( it );
              if ( parameterGroupData )
              {
                typedef multimap<HashKey,ParameterGroupDataSharedPtr>::const_iterator I;
                I pgdit;
                HashKey hashKey;
                bool found = false;
                {
                  hashKey = parameterGroupData->getHashKey();

                  pair<I,I> itp = m_parameterGroupData.equal_range( hashKey );
                  for ( pgdit = itp.first ; pgdit != itp.second ; ++pgdit )
                  {
                    if (    ( parameterGroupData == pgdit->second )
                        ||  ( parameterGroupData->isEquivalent( pgdit->second, getIgnoreNames(), false ) ) )
                    {
                      found = true;
                      break;
                    }
                  }
                }
                if ( found )
                {
                  if ( p->getParameterGroupData( it ) != pgdit->second )
                  {
                    p->setParameterGroupData( it, pgdit->second );
                    setTreeModified();
                  }
                }
                else
                {
                  m_parameterGroupData.insert( make_pair( hashKey, parameterGroupData ) );
                }
              }
            }
          }
        }
      }

      dp::sg::core::PipelineDataSharedPtr UnifyTraverser::unifyPipelineData( dp::sg::core::PipelineDataSharedPtr const& pipelineData )
      {
        DP_ASSERT( ( m_unifyTargets & Target::PIPELINE_DATA ) && pipelineData );
        typedef multimap<HashKey,dp::sg::core::PipelineDataSharedPtr>::const_iterator I;
        I it;
        HashKey hashKey;
        bool found = false;
        {
          hashKey = pipelineData->getHashKey();

          pair<I,I> itp = m_pipelineData.equal_range( hashKey );
          for ( it = itp.first ; it != itp.second ; ++it )
          {
            if (    ( pipelineData == it->second )
                ||  ( pipelineData->isEquivalent( it->second, getIgnoreNames(), false ) ) )
            {
              found = true;
              break;
            }
          }
        }
        if ( found )
        {
          return( it->second );
        }
        else
        {
          m_pipelineData.insert( make_pair( hashKey, pipelineData ) );
          return( dp::sg::core::PipelineDataSharedPtr() );
        }
      }

      void UnifyTraverser::handleGroup( Group *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleGroup( p );
          unifyChildren( p );
        }
      }

      void UnifyTraverser::handleLightSource( LightSource * p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleLightSource( p );
          if ( optimizationAllowed( p->getSharedPtr<LightSource>() ) && ( m_unifyTargets & Target::PIPELINE_DATA ) && p->getLightPipeline() )
          {
            const dp::sg::core::PipelineDataSharedPtr & replacement = unifyPipelineData( p->getLightPipeline() );
            if ( replacement )
            {
              p->setLightPipeline( replacement );
            }
          }
        }
      }

      void UnifyTraverser::handleIndexSet(IndexSet *p)
      {
        pair<set<const void*>::iterator, bool> pitb = m_objects.insert(p);
        if (pitb.second)
        {
          OptimizeTraverser::handleIndexSet(p);
          if (optimizationAllowed(p->getSharedPtr<IndexSet>()) && (m_unifyTargets & Target::BUFFER))
          {
            unifyBuffers(p);
          }
        }
      }

      void UnifyTraverser::handleLOD( LOD *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleGroup( p );  // traverse as a Group here (all children!)
          unifyChildren( p );
        }
      }

      void UnifyTraverser::handleParameterGroupData( ParameterGroupData * p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleParameterGroupData( p );
          if ( optimizationAllowed( p->getSharedPtr<ParameterGroupData>() ) && ( m_unifyTargets & Target::SAMPLER ) )
          {
            const dp::fx::ParameterGroupSpecSharedPtr & pgs = p->getParameterGroupSpec();
            for ( dp::fx::ParameterGroupSpec::iterator it = pgs->beginParameterSpecs() ; it != pgs->endParameterSpecs() ; ++it )
            {
              if ( ( it->first.getType() & dp::fx::PT_POINTER_TYPE_MASK ) == dp::fx::PT_SAMPLER_PTR )
              {
                typedef multimap<HashKey,SamplerSharedPtr>::const_iterator I;
                I sit;
                HashKey hashKey;

                const SamplerSharedPtr & sampler = p->getParameter<SamplerSharedPtr>( it );
                bool found = false;
                {
                  hashKey = sampler->getHashKey();

                  pair<I,I> itp = m_samplers.equal_range( hashKey );
                  for ( sit = itp.first ; sit != itp.second ; ++sit )
                  {
                    if (    ( sampler == sit->second )
                        ||  ( sampler->isEquivalent( sit->second, getIgnoreNames(), false ) ) )
                    {
                      found = true;
                      break;
                    }
                  }
                }
                if ( found )
                {
                  p->setParameter( it, sit->second );
                }
                else
                {
                  m_samplers.insert( make_pair( hashKey, sampler ) );
                }
              }
            }
          }
        }
      }

      void UnifyTraverser::handlePrimitive( Primitive *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handlePrimitive( p );

          // replace the VertexAttributeSet, in case it has been optimized
          VASReplacementMap::iterator it = m_vasReplacements.find( p->getVertexAttributeSet() );
          if ( it != m_vasReplacements.end() )
          {
            IndexSetSharedPtr newIndexSet( IndexSet::create() );
            if ( p->isIndexed() )
            {
              IndexSetSharedPtr oldIndexSet = p->getIndexSet();
              unsigned int pri = oldIndexSet->getPrimitiveRestartIndex();
              vector<unsigned int> newIndices( p->getElementCount() );
              IndexSet::ConstIterator<unsigned int> oldIndices( oldIndexSet, p->getElementOffset() );
              for ( size_t i=0 ; i<newIndices.size() ; i++ )
              {
                DP_ASSERT((oldIndices[i] == pri) || (oldIndices[i] < it->second.m_indexMap.size()));
                newIndices[i] = ( oldIndices[i] == pri ) ? pri : it->second.m_indexMap[oldIndices[i]];
              }
              newIndexSet->setData( &newIndices[0], dp::checked_cast<unsigned int>(newIndices.size()) );
            }
            else
            {
              newIndexSet->setData( it->second.m_indexMap.data(), dp::checked_cast<unsigned int>(it->second.m_indexMap.size()) );
            }
            p->setIndexSet( newIndexSet );
            p->setVertexAttributeSet( it->second.m_vas );
            p->setElementRange( 0, ~0 );
          }

          if ( optimizationAllowed( p->getSharedPtr<Primitive>() ) )
          {
            if ( m_unifyTargets & Target::INDEX_SET )
            {
              unifyIndexSets( p );
            }
            if ( m_unifyTargets & Target::VERTEX_ATTRIBUTE_SET )
            {
              unifyVertexAttributeSet( p );
            }
            if ( m_unifyTargets & Target::PRIMITIVE )
            {
              checkPrimitive( m_primitives[p->getPrimitiveType()], p );
            }
          }
        }
      }

      void UnifyTraverser::handleSampler( Sampler * p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleSampler( p );
          if ( optimizationAllowed( p->getSharedPtr<Sampler>() ) )
          {
            if ( ( m_unifyTargets & Target::TEXTURE ) && p->getTexture() )
            {
              typedef multimap<HashKey,TextureSharedPtr>::const_iterator I;
              I it;
              HashKey hashKey;

              const TextureSharedPtr & texture = p->getTexture();
              bool found = false;
              hashKey = texture->getHashKey();

              pair<I,I> itp = m_textures.equal_range( hashKey );
              for ( it = itp.first ; it != itp.second ; ++it )
              {
                if (    ( texture == it->second )
                    ||  texture->isEquivalent( it->second, false ) )
                {
                  found = true;
                  break;
                }
              }
              if ( found )
              {
                p->setTexture( it->second );
              }
              else
              {
                m_textures.insert( make_pair( hashKey, texture ) );
              }
            }
          }
        }
      }

      void UnifyTraverser::handleSwitch( Switch *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleGroup( p );  // traverse as a Group here (all children!)
          unifyChildren( p );
        }
      }

      void UnifyTraverser::handleTransform( Transform *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleTransform( p );
          unifyChildren( p );
        }
      }

      void UnifyTraverser::handleVertexAttributeSet( VertexAttributeSet * p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleVertexAttributeSet( p );
          // Check if optimization is allowed
          if (optimizationAllowed(p->getSharedPtr<VertexAttributeSet>()))
          {
            if (m_unifyTargets & Target::BUFFER)
            {
              unifyBuffers(p);
            }
          }
        }
      }

      void UnifyTraverser::unifyBuffers(IndexSet *p)
      {
        BufferSharedPtr buffer = p->getBuffer();
        if (buffer)
        {
          HashKey hashKey = buffer->getHashKey();
          typedef multimap<HashKey, BufferSharedPtr>::const_iterator I;
          pair<I, I> itp = m_indexBuffers.equal_range(hashKey);

          bool found = false;
          for (I it = itp.first; it != itp.second && !found; ++it)
          {
            found = (buffer == it->second) || buffer->isEquivalent(it->second, getIgnoreNames(), false);
            if (found && (buffer != it->second))
            {
              p->setBuffer(it->second, p->getNumberOfIndices(), p->getIndexDataType(), p->getPrimitiveRestartIndex());
            }
          }
#if CHECK_HASH_RESULTS
          bool checkFound = false;
          for (I it = m_indexBuffers.begin(); it != m_indexBuffers.end() && !checkFound; ++it)
          {
            checkFound = (buffer == it->second) || buffer->isEquivalent(it->second, getIgnoreNames(), false);
            DP_ASSERT(!checkFound || (buffer == it->second));
          }
          DP_ASSERT(found == checkFound);
#endif
          if (!found)
          {
            m_indexBuffers.insert(make_pair(hashKey, buffer));
          }
        }
      }

      void UnifyTraverser::unifyBuffers(VertexAttributeSet *p)
      {
        for (unsigned int i = 0; i < static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT); i++)
        {
          VertexAttributeSet::AttributeID attribute = static_cast<VertexAttributeSet::AttributeID>(i);
          if (p->getNumberOfVertexData(attribute))
          {
            VertexAttribute va = p->getVertexAttribute(attribute);

            BufferSharedPtr buffer = va.getBuffer();
            HashKey hashKey = buffer->getHashKey();
            typedef multimap<HashKey, BufferSharedPtr>::const_iterator I;
            pair<I, I> itp = m_vertexBuffers.equal_range(hashKey);

            bool found = false;
            for (I it = itp.first; it != itp.second && !found; ++it)
            {
              found = (buffer == it->second) || buffer->isEquivalent(it->second, getIgnoreNames(), false);
              if (found && (buffer != it->second))
              {
                va.setData(va.getVertexDataSize(), va.getVertexDataType(), it->second, va.getVertexDataOffsetInBytes(), va.getVertexDataStrideInBytes(), va.getVertexDataCount());
                p->swapVertexData(attribute, va);
              }
            }
#if CHECK_HASH_RESULTS
            bool checkFound = false;
            for (I it = m_vertexBuffers.begin(); it != m_vertexBuffers.end() && !checkFound; ++it)
            {
              checkFound = (buffer == it->second) || buffer->isEquivalent(it->second, getIgnoreNames(), false);
              DP_ASSERT(!checkFound || (buffer == it->second));
            }
            DP_ASSERT(found == checkFound);
#endif
            if (!found)
            {
              m_vertexBuffers.insert(make_pair(hashKey, buffer));
            }
          }
        }
      }

      void UnifyTraverser::unifyChildren( Group *p )
      {
        // make sure we can optimize the children of this group
        if( optimizationAllowed( p->getSharedPtr<Group>() ) )
        {
          if ( m_unifyTargets & Target::GEONODE )
          {
            unifyGeoNodes( p );
          }
          if ( m_unifyTargets & Target::GROUP )
          {
            unifyGroups( p );
          }
          if ( m_unifyTargets & Target::LOD )
          {
            unifyLODs( p );
          }
        }
      }

      void UnifyTraverser::unifyGeoNodes( Group *p )
      {
        DP_ASSERT( m_unifyTargets & Target::GEONODE );

        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( std::dynamic_pointer_cast<GeoNode>(*gci) )
          {
            GeoNodeSharedPtr geoNode = std::static_pointer_cast<GeoNode>(*gci);
            {
              if ( optimizationAllowed( geoNode ) )
              {
                HashKey hashKey = geoNode->getHashKey();
                typedef multimap<HashKey,GeoNodeSharedPtr>::const_iterator I;
                pair<I,I> itp = m_geoNodes.equal_range( hashKey );

                bool found = false;
                for ( I it=itp.first ; it!= itp.second && !found ; ++it )
                {
                  found =  ( geoNode == it->second )
                        || geoNode->isEquivalent( it->second, getIgnoreNames(), false );
                  if ( found && ( geoNode != it->second ) )
                  {
                    p->replaceChild( it->second, gci );
                  }
                }
#if CHECK_HASH_RESULTS
                bool checkFound = false;
                for ( I it = m_geoNodes.begin() ; it != m_geoNodes.end() && !checkFound ; ++it )
                {
                  checkFound = ( geoNode == it->second )
                            || geoNode->isEquivalent( it->second, getIgnoreNames(), false );
                  DP_ASSERT( !checkFound || ( geoNode == it->second ) );
                }
                DP_ASSERT( found == checkFound );
#endif
                if ( ! found )
                {
                  m_geoNodes.insert( make_pair( hashKey, geoNode ) );
                }
              }
            }
          }
        }
      }

      void UnifyTraverser::unifyGroups( Group *p )
      {
        DP_ASSERT( m_unifyTargets & Target::GROUP );

        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( std::dynamic_pointer_cast<Group>(*gci) )
          {
            GroupSharedPtr group = std::static_pointer_cast<Group>(*gci);
            {
              if ( optimizationAllowed( group ) )
              {
                HashKey hashKey = group->getHashKey();
                typedef multimap<HashKey,GroupSharedPtr>::const_iterator I;
                pair<I,I> itp = m_groups.equal_range( hashKey );

                bool found = false;
                for ( I it=itp.first ; it!= itp.second && !found ; ++it )
                {
                  found =  ( group == it->second )
                        || group->isEquivalent( it->second, getIgnoreNames(), false );
                  if ( found && ( group != it->second ) )
                  {
                    p->replaceChild( it->second, gci );
                  }
                }
#if CHECK_HASH_RESULTS
                bool checkFound = false;
                for ( I it = m_groups.begin() ; it != m_groups.end() && !checkFound ; ++it )
                {
                  checkFound = ( group == it->second )
                            || group->isEquivalent( it->second, getIgnoreNames(), false );
                  DP_ASSERT( !checkFound || ( group == it->second ) );
                }
                DP_ASSERT( found == checkFound );
#endif
                if ( ! found )
                {
                  m_groups.insert( make_pair( hashKey, group ) );
                }
              }
            }
          }
        }
      }

      void UnifyTraverser::unifyIndexSets( Primitive *p )
      {
        DP_ASSERT( m_unifyTargets & Target::INDEX_SET );
        if ( p->isIndexed() )
        {
          const IndexSetSharedPtr & iset = p->getIndexSet();
          IndexSetSharedPtr const& is = p->getIndexSet();
          if ( optimizationAllowed( is ) )
          {
            HashKey hashKey = is->getHashKey();
            typedef multimap<HashKey,IndexSetSharedPtr>::const_iterator I;
            pair<I,I> itp = m_indexSets.equal_range( hashKey );

            bool found = false;
            for ( I it=itp.first ; it!= itp.second && !found ; ++it )
            {
              found =  ( iset == it->second )
                    || is->isEquivalent( it->second, getIgnoreNames(), false );
              if ( found && ( iset != it->second ) )
              {
                p->setIndexSet( it->second );
              }
            }
#if CHECK_HASH_RESULTS
            bool checkFound = false;
            for ( I it = m_indexSets.begin() ; it != m_indexSets.end() && !checkFound ; ++it )
            {
              checkFound = ( iset == it->second )
                        || is->isEquivalent( it->second, getIgnoreNames(), false );
              DP_ASSERT( !checkFound || ( iset == it->second ) );
            }
            DP_ASSERT( found == checkFound );
#endif
            if ( !found )
            {
              m_indexSets.insert( make_pair( hashKey, iset ) );
            }
          }
        }
      }

      void UnifyTraverser::unifyLODs( Group *p )
      {
        DP_ASSERT( m_unifyTargets & Target::LOD );

        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( std::dynamic_pointer_cast<LOD>(*gci) )
          {
            bool optimizable = false, found = false;
            LODSharedPtr lod = std::static_pointer_cast<LOD>(*gci);
            {
              optimizable = optimizationAllowed( lod );
              if( optimizable )
              {
                for ( size_t j=0 ; j<m_LODs.size() && !found ; j++ )
                {
                  if ( ( lod == m_LODs[j] ) || lod->isEquivalent( m_LODs[j], getIgnoreNames(), false ) )
                  {
                    found = true;
                    if ( lod != m_LODs[j] )
                    {
                      p->replaceChild( m_LODs[j], gci );
                    }
                  }
                }
              }
            }
            // do not allow things to be merged INTO a dynamic node.
            if ( ! found && optimizable )
            {
              m_LODs.push_back( lod );
            }
          }
        }
      }

      void UnifyTraverser::checkPrimitive( multimap<HashKey,PrimitiveSharedPtr> &v, Primitive * p )
      {
        // Unify Primitives of each type
        DP_ASSERT( m_unifyTargets & Target::PRIMITIVE );

        if( !optimizationAllowed( p->getSharedPtr<Primitive>() ) )
        {
          return;
        }

#if CHECK_HASH_RESULTS
        PrimitiveWeakPtr foundPrimitive = nullptr;
#endif

        // look for all Primitives of the same type already encountered, with the same hash String (should not be too many!)
        bool found = false;
        HashKey hashKey = p->getHashKey();
        typedef multimap<HashKey,PrimitiveSharedPtr>::const_iterator I;
        pair<I,I> itp = v.equal_range( hashKey );
        PrimitiveSharedPtr primitive = p->getSharedPtr<Primitive>();
        for ( I it = itp.first ; it != itp.second && !found ; ++it )
        {
          // check if any of those Primitives is equal or equivalent to the currently handled
          found =   ( primitive == it->second )
                ||  p->isEquivalent( it->second, getIgnoreNames(), false );
          if ( found && ( primitive != it->second ) )
          {
            // there is an equivalent Primitive, that's not the same as the currently handled -> store as replacement
            m_replacementPrimitive = it->second;
          }
#if CHECK_HASH_RESULTS
          if ( found )
          {
            foundPrimitive = it->second;
          }
#endif
        }

#if CHECK_HASH_RESULTS
        // just to make sure, that we find the same equivalent Primitive with exhaustive search (no hash string usage)
        bool checkFound = false;
        for ( I it = mm.begin() ; it != mm.end() && !checkFound ; ++it )
        {
          checkFound = ( primitive == it->second )
                    || pT->isEquivalent( SharedHandle<T>::Lock(it->second), getIgnoreNames(), false );
          DP_ASSERT( !checkFound || ( primitive == it->second ) || ( foundPrimitive == it->second ) );
        }
        DP_ASSERT( found == checkFound );
#endif

        // if we did not found that Primitive (or an equivalent one) before -> store it for later searches
        if ( ! found )
        {
          v.insert( make_pair( hashKey, primitive ) );
        }
      }

      void UnifyTraverser::unifyVertices(VertexAttributeSetSharedPtr const& vas)
      {
        unsigned int n = vas->getNumberOfVertices();

        //  handle VAS with more than one vertex only
        if (1 < n)
        {
          // ***************************************************************
          // the algorithm currently only works for float-typed vertex data!
          // ***************************************************************
          for (unsigned int i = 0; i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT); i++)
          {
            dp::DataType type = vas->getTypeOfVertexData(static_cast<VertexAttributeSet::AttributeID>(i));
            if (type != dp::DataType::UNKNOWN // no data is ok!
              && type != dp::DataType::FLOAT_32)
            {
              DP_ASSERT(!"This algorithm currently only works for float-typed vertex data!");
              return;
            }
          }
          // ***************************************************************
          // ***************************************************************

          //  count the dimension of the VertexAttributeSet
          unsigned int  dimension = 0;
          for (unsigned int i = 0; i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT); i++)
          {
            VertexAttributeSet::AttributeID id = static_cast<VertexAttributeSet::AttributeID>(i);
            if (vas->getNumberOfVertexData(id))
            {
              dimension += vas->getSizeOfVertexData(id);
            }
          }

          //  fill valuesIn with the vertex attribute data
          vector<float> valuesIn(n * dimension);
          for (unsigned int i = 0, j = 0; i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT); i++)
          {
            VertexAttributeSet::AttributeID id = static_cast<VertexAttributeSet::AttributeID>(i);
            if (vas->getNumberOfVertexData(id) != 0)
            {
              unsigned int dim = vas->getSizeOfVertexData(id);
              Buffer::ConstIterator<float>::Type vad = vas->getVertexData<float>(id);
              dp::util::stridedMemcpy(valuesIn.data(), j * sizeof(float), dimension * sizeof(float), &vad[0], 0, dim * sizeof(float), dim * sizeof(float), n);
              j += dim;
            }
          }

          // initialize the indices vector to sort
          std::vector<unsigned int> indices(n);
          unsigned int i = 0;
          std::generate(indices.begin(), indices.end(), [&i] { return i++; });

          // just sort for first component of values (x-coordinate of position)
          std::sort(indices.begin(), indices.end(), [&valuesIn, dimension](unsigned int const& a, unsigned int const& b){ return valuesIn[a*dimension] < valuesIn[b*dimension]; });

          // reserve enough space for the output
          std::vector<float> valuesOut;
          valuesOut.reserve(n * dimension);
          std::vector<float> currentVertex(dimension);
          unsigned int outIndex = 0;
          std::vector<unsigned int> indexMap(n, ~0);

          // gather the reduced stuff, if any
          for (unsigned int i = 0; i < n; i++)
          {
            if (indices[i] != ~0)
            {
              memcpy(currentVertex.data(), &valuesIn[indices[i] * dimension], dimension * sizeof(float));
              indexMap[indices[i]] = outIndex;
              unsigned int count = 1;
              for (unsigned int j = i + 1; j < n; j++)
              {
                if (indices[j] != ~0)
                {
                  if (valuesIn[indices[j] * dimension] - valuesIn[indices[i] * dimension] <= m_epsilon)
                  {
                    bool similar = true;
                    for (unsigned int k = 1; k < dimension && similar; k++)
                    {
                      similar = (abs(valuesIn[indices[j] * dimension + k] - valuesIn[indices[i] * dimension + k]) <= m_epsilon);
                    }
                    if (similar)
                    {
                      for (unsigned int k = 0; k < dimension; k++)
                      {
                        currentVertex[k] += valuesIn[indices[j] * dimension + k];
                      }
                      indexMap[indices[j]] = outIndex;
                      indices[j] = ~0;
                      count++;
                    }
                  }
                  else
                  {
                    // due to sorting, all further components have a delta larger than m_epsilon
                    // -> break inner loop as vertices "equal" to valuesIn[indices[i]] are identified, handle those equals and continue with next vertex
                    break;
                  }
                }
              }
              indices[i] = ~0;
              if (1 < count)
              {
                for (unsigned int k = 0; k < dimension; k++)
                {
                  valuesOut.push_back(currentVertex[k] / count);
                }
              }
              else
              {
                for (unsigned int k = 0; k < dimension; k++)
                {
                  valuesOut.push_back(currentVertex[k]);
                }
              }
              outIndex++;
            }
          }
#if !defined(NDEBUG)
          // assert that all indices have been remapped
          for (unsigned int i = 0; i < n; i++)
          {
            DP_ASSERT(indexMap[i] != ~0);
            DP_ASSERT(indices[i] == ~0);
          }
#endif

          // if at least one point was reduced...
          if (outIndex < n)
          {
            //  create a new VertexAttributeSet with the condensed data
            VertexAttributeSetSharedPtr newVAS = VertexAttributeSet::create();
            for (unsigned int i = 0, j = 0; i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT); i++)
            {
              VertexAttributeSet::AttributeID id = static_cast<VertexAttributeSet::AttributeID>(i);
              if (vas->getNumberOfVertexData(id))
              {
                unsigned int dim = vas->getSizeOfVertexData(id);
                vector<float> vad(dim * outIndex);
                for (size_t k = 0; k<outIndex; k++)
                {
                  for (unsigned int l = 0; l<dim; l++)
                  {
                    vad[dim*k + l] = valuesOut[k*dimension + j + l];
                  }
                }
                newVAS->setVertexData(id, dim, dp::DataType::FLOAT_32, &vad[0], 0, dp::checked_cast<unsigned int>(vad.size() / dim));

                // inherit enable states from source attrib
                // normalize-enable state only meaningful for generic aliases!
                newVAS->setEnabled(id, vas->isEnabled(id)); // conventional

                id = static_cast<VertexAttributeSet::AttributeID>(i + 16);    // generic
                newVAS->setEnabled(id, vas->isEnabled(id));
                newVAS->setNormalizeEnabled(id, vas->isNormalizeEnabled(id));
                j += dim;
              }
            }

            DP_ASSERT(m_vasReplacements.find(vas) == m_vasReplacements.end());
            m_vasReplacements[vas] = VASReplacement(newVAS, indexMap);
          }
        }
      }

      void UnifyTraverser::unifyVerticesThreadFunction(std::vector<dp::sg::core::ObjectSharedPtr> const& results)
      {
        for (unsigned int i = m_unifyVerticesIndex.fetch_add(1); i < results.size(); i = m_unifyVerticesIndex.fetch_add(1))
        {
          unifyVertices(std::static_pointer_cast<dp::sg::core::VertexAttributeSet>(results[i]));
        }
      }

      void UnifyTraverser::unifyVertexAttributeSet(Primitive *p)
      {
        DP_ASSERT( m_unifyTargets & Target::VERTEX_ATTRIBUTE_SET );
        DP_ASSERT( p && p->getVertexAttributeSet() );

        if( !optimizationAllowed( p->getSharedPtr<Primitive>() ) )
        {
          return;
        }

        bool found = false;
        HashKey hashKey;
        VertexAttributeSetSharedPtr vertexAttributeSet = p->getVertexAttributeSet();    // get a share count, in case it's deleted below
        {
          hashKey = vertexAttributeSet->getHashKey();

          typedef multimap<HashKey,VertexAttributeSetSharedPtr>::const_iterator I;
          pair<I,I> itp = m_vertexAttributeSets.equal_range( hashKey );
          for ( I it=itp.first ; it!= itp.second && !found ; ++it )
          {
            found =  ( vertexAttributeSet == it->second )
                  || vertexAttributeSet->isEquivalent( it->second, getIgnoreNames(), false );
            if ( found && ( vertexAttributeSet != it->second ) )
            {
              p->setVertexAttributeSet( it->second );
            }
          }
#if CHECK_HASH_RESULTS
          bool checkFound = false;
          for ( I it = m_vertexAttributeSets.begin() ; it != m_vertexAttributeSets.end() && !checkFound ; ++it )
          {
            checkFound = ( vertexAttributeSet == it->second )
                      || vertexAttributeSet->isEquivalent( it->second, getIgnoreNames(), false );
            DP_ASSERT( !checkFound || ( vertexAttributeSet == it->second ) || ( p->getVertexAttributeSet() == it->second ) );
          }
          DP_ASSERT( found == checkFound );
#endif
        }
        if ( ! found )
        {
          m_vertexAttributeSets.insert( make_pair( hashKey, vertexAttributeSet ) );
        }
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
