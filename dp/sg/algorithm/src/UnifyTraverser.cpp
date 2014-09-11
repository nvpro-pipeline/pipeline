// Copyright NVIDIA Corporation 2002-2011
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
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/algorithm/UnifyTraverser.h>

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

      bool areSimilarFirst( const float * v0, const float * v1, float eps );
      bool areSimilarLast( const float * v0, const float * v1, unsigned int dimension, float eps );

      class VUTOctreeIndices;
      class VUTOctreeSubNodes;

      class VUTOctreeNode
      {
        public :
          ~VUTOctreeNode();

        public:
          void addIndexList( const vector<vector<unsigned int> > & indexList );
          void addVertex( const vector<float*> &vertices, unsigned int index, unsigned int dimension, float tol, unsigned int level = 0 );
          void distributeData( const vector<float *> & vertices, float tol );
          unsigned int getNumberOfPoints( void ) const;
          void init( const Box3f & bbox, unsigned int maxIndicesPerOctel );
          void mapValues( const vector<float *> &valuesIn, vector<float *> & valuesOut, unsigned int dimension, vector<unsigned int> &indexMap, unsigned int &index );

        private :
          //  the octel encloses the m_box, m_center caches the center of m_box.
          //  an octel holds either m_data with point informations or eight pointers to the sub-octels
          Box3f               m_box;
          Vec3f               m_center;
          VUTOctreeSubNodes * m_nodes;
          VUTOctreeIndices  * m_data;           //  pointer to the data of this octel
      };

      class VUTOctreeIndices
      {
        public:
          typedef list<vector<vector<unsigned int> > > IndexContainer;

        public:
          VUTOctreeIndices( unsigned int maxIndices );
          ~VUTOctreeIndices();

        public:
          void addIndexList( const vector<vector<unsigned int> > & indexList );
          bool addVertex( const vector<float*> &vertices, unsigned int index, unsigned int dimension, float tol );
          void clear( void );    // clear the indices, without deletion!
          IndexContainer::const_iterator getIndexListBegin() const;
          IndexContainer::const_iterator getIndexListEnd() const;
          unsigned int  getMaxIndices( void ) const;
          unsigned int  getNumberOfPoints( void ) const;
          void mapValues( const vector<float *> & valuesIn, vector<float *> & valuesOut, unsigned int dimension, vector<unsigned int> &indexMap, unsigned int &index );

        private:
          //  indices is a vector that is allowed to grow up to m_maxIndices
          //  each element of m_indices[i][j] is the index of a vertex in an valuesIn buffer
          //  each element of m_indices[i] holds the indices of all vertices with the same position, but maybe
          //    different attributes
          //  each element of m_indices holds the indices of all vertices in that octel
          IndexContainer  m_indices;
          unsigned int    m_maxIndices;
      };

      class VUTOctreeSubNodes
      {
        public:
          VUTOctreeSubNodes( const Box3f & box, const Vec3f & center, unsigned int maxIndicesPerOctel );

        public:
          void addIndexList( const vector<vector<unsigned int> > & indexList, const vector<float*> &vertices, const Vec3f & center );
          void addVertex( const vector<float*> &vertices, unsigned int index, unsigned int dimension, float tol, unsigned int level, const Vec3f & center );
          unsigned int getNumberOfPoints() const;
          void mapValues( const vector<float *> & valuesIn, vector<float *> & valuesOut, unsigned int dimension, vector<unsigned int> &indexMap, unsigned int &index );

        private:
          VUTOctreeNode nodes[2][2][2];
      };

      DEFINE_STATIC_PROPERTY( UnifyTraverser, UnifyTargets );
      DEFINE_STATIC_PROPERTY( UnifyTraverser, Epsilon );

      BEGIN_REFLECTION_INFO( UnifyTraverser )
        DERIVE_STATIC_PROPERTIES( UnifyTraverser, OptimizeTraverser );
        INIT_STATIC_PROPERTY_RW( UnifyTraverser, UnifyTargets, unsigned int, SEMANTIC_VALUE, value, value );
        INIT_STATIC_PROPERTY_RW( UnifyTraverser, Epsilon,      float,        SEMANTIC_VALUE, value, value );
      END_REFLECTION_INFO

      UnifyTraverser::UnifyTraverser( void )
      : m_epsilon(std::numeric_limits<float>::epsilon())
      , m_unifyTargets(UT_ALL_TARGETS_MASK)
      {
      }

      UnifyTraverser::~UnifyTraverser( void )
      {
      }

      void UnifyTraverser::doApply( const NodeSharedPtr & root )
      {
        DP_ASSERT(   m_effectData.empty() && m_geoNodes.empty() && m_groups.empty() && m_indexSets.empty()
                    && m_LODs.empty() && m_parameterGroupData.empty() && m_primitives.empty() && m_samplers.empty()
                    && m_textures.empty() && m_vertexAttributeSets.empty() );
        DP_ASSERT( m_multiOwnedHandledVAS.empty() && m_removedVAS.empty() );

        OptimizeTraverser::doApply( root );

        m_effectData.clear();
        m_geoNodes.clear();
        m_groups.clear();
        m_indexSets.clear();
        m_parameterGroupData.clear();
        m_primitives.clear();
        m_LODs.clear();
        m_objects.clear();
        m_samplers.clear();
        m_textures.clear();
        m_vertexAttributeSets.clear();
        m_multiOwnedHandledVAS.clear();
        m_removedVAS.clear();
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

      void UnifyTraverser::handleEffectData( EffectData * p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleEffectData( p );
          if ( optimizationAllowed( p->getSharedPtr<EffectData>() ) && ( m_unifyTargets & UT_PARAMETER_GROUP_DATA ) )
          {
            const dp::fx::SmartEffectSpec & es = p->getEffectSpec();
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

      void UnifyTraverser::handleGeoNode( GeoNode *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          OptimizeTraverser::handleGeoNode( p );

          if ( optimizationAllowed( p->getSharedPtr<GeoNode>() ) )
          {
            if ( ( m_unifyTargets & UT_EFFECT_DATA ) && p->getMaterialEffect() )
            {
              const EffectDataSharedPtr & replacement = unifyEffectData( p->getMaterialEffect() );
              if ( replacement )
              {
                p->setMaterialEffect( replacement );
              }
            }
            if ( ( m_unifyTargets & UT_PRIMITIVE ) && m_replacementPrimitive )
            {
              p->setPrimitive( m_replacementPrimitive );
              m_replacementPrimitive.reset();
            }
          }
        }
      }

      const EffectDataSharedPtr & UnifyTraverser::unifyEffectData( const EffectDataSharedPtr & effectData )
      {
        DP_ASSERT( ( m_unifyTargets & UT_EFFECT_DATA ) && effectData );
        typedef multimap<HashKey,EffectDataSharedPtr>::const_iterator I;
        I it;
        HashKey hashKey;
        bool found = false;
        {
          hashKey = effectData->getHashKey();

          pair<I,I> itp = m_effectData.equal_range( hashKey );
          for ( it = itp.first ; it != itp.second ; ++it )
          {
            if (    ( effectData == it->second )
                ||  ( effectData->isEquivalent( it->second, getIgnoreNames(), false ) ) )
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
          static EffectDataSharedPtr dummy;
          m_effectData.insert( make_pair( hashKey, effectData ) );
          return( dummy );
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
          if ( optimizationAllowed( p->getSharedPtr<LightSource>() ) && ( m_unifyTargets & UT_EFFECT_DATA ) && p->getLightEffect() )
          {
            const EffectDataSharedPtr & replacement = unifyEffectData( p->getLightEffect() );
            if ( replacement )
            {
              p->setLightEffect( replacement );
            }
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
          if ( optimizationAllowed( p->getSharedPtr<ParameterGroupData>() ) && ( m_unifyTargets & UT_SAMPLER ) )
          {
            const dp::fx::SmartParameterGroupSpec & pgs = p->getParameterGroupSpec();
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
          if ( optimizationAllowed( p->getSharedPtr<Primitive>() ) )
          {
            if ( m_unifyTargets & UT_INDEX_SET )
            {
              unifyIndexSets( p );
            }
            if ( m_unifyTargets & UT_VERTEX_ATTRIBUTE_SET )
            {
              unifyVertexAttributeSet( p );
            }
            if ( m_unifyTargets & UT_PRIMITIVE ) 
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
            if ( ( m_unifyTargets & UT_TEXTURE ) && p->getTexture() )
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
          if ( optimizationAllowed( p->getSharedPtr<VertexAttributeSet>() ) && (m_unifyTargets & UT_VERTICES) )
          {
            VertexAttributeSetSharedPtr vas = p->getSharedPtr<VertexAttributeSet>();
            unsigned int n = p->getNumberOfVertices();

            //  multiple used VertexAttributeSets are only handled once
            if (    (     ( 1 == p->getNumberOfOwners() )
                      ||  ( m_multiOwnedHandledVAS.find( vas ) == m_multiOwnedHandledVAS.end() ) )
                &&  ( n > 1 ) )
            {

              // ***************************************************************
              // the algorithm currently only works for float-typed vertex data!
              // ***************************************************************
              for ( unsigned int i=0 ; i<VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT ; i++ )
              {
                unsigned int type = p->getTypeOfVertexData(i);
                if (  type != dp::util::DT_UNKNOWN // no data is ok!
                   && type != dp::util::DT_FLOAT_32 )  
                {
                  if ( p->getNumberOfOwners() > 1 )
                  {
                    // don't consider again
                    m_multiOwnedHandledVAS.insert( vas );
                  }
                  DP_ASSERT( !"This algorithm currently only works for float-typed vertex data!" );
                  return; 
                }
              }
              // ***************************************************************
              // ***************************************************************


              //  count the dimension of the VertexAttributeSet
              unsigned int  dimension = 0;
              for ( unsigned int i=0 ; i<VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT ; i++ )
              {
                if ( p->getNumberOfVertexData( i ) )
                {
                  dimension += p->getSizeOfVertexData( i );
                }
              }

              vector<float> valueDataIn( n * dimension );
              vector<float*> valuesIn( n );
              for ( size_t i=0, j=0 ; i<valuesIn.size() ; i++, j+=dimension )
              {
                valuesIn[i] = &valueDataIn[j];
              }

              //  fill valuesIn with the vertex attribute data
              for ( unsigned int i=0, j=0 ; i<VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT ; i++ )
              {
                if ( p->getNumberOfVertexData( i ) != 0 )
                {
                  unsigned int dim = p->getSizeOfVertexData( i );
                  Buffer::ConstIterator<float>::Type vad = p->getVertexData<float>( i );
                  for ( unsigned int k=0 ; k<n ; k++ )
                  {
                    const float *value = &vad[k];
                    for ( unsigned int l=0 ; l<dim ; l++ )
                    {
                      valuesIn[k][j+l] = value[l];
                    }
                  }
                  j += dim;
                }
              }

              VUTOctreeNode * octree = new VUTOctreeNode();
              octree->init( boundingBox<3, float, Buffer::ConstIterator<Vec3f>::Type >( p->getVertices(), p->getNumberOfVertices() )
                          , std::max( (unsigned int)32, (unsigned int)pow( p->getNumberOfVertices(), 0.25 ) ) );
              unsigned int count = checked_cast<unsigned int>(valuesIn.size());
              for ( unsigned int i=0 ; i<count ; i++ )
              {
                octree->addVertex( valuesIn, i, dimension, m_epsilon );
              }
              unsigned int pointCount = octree->getNumberOfPoints();

              //  if there are less points only
              if ( pointCount < n )
              {
                //  the VertexAttributeSet p will be removed, so hold a ref on it here
                m_removedVAS.push_back( vas );

                //  initialize the index mapping to undefined
                vector<unsigned int>  indexMap( n );
                for ( unsigned int i=0 ; i<n ; i++ )
                {
                  indexMap[i] = ~0;
                }

                //  create the vector of vertex attribute valuesIn
                vector<float> valueDataOut( pointCount * dimension );
                vector<float*> valuesOut( pointCount );
                for ( size_t i=0, j=0 ; i<valuesOut.size() ; i++, j+=dimension )
                {
                  valuesOut[i] = &valueDataOut[j];
                }

                // fill valuesOut and the indexMap
                unsigned int index = 0;
                octree->mapValues( valuesIn, valuesOut, dimension, indexMap, index );
                delete octree;

                //  create a new VertexAttributeSet with the condensed data
                VertexAttributeSetSharedPtr newVASH = VertexAttributeSet::create();
                for ( unsigned int i=0, j=0 ; i<VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT ; i++ )
                {
                  if ( p->getNumberOfVertexData( i ) )
                  {
                    unsigned int dim = p->getSizeOfVertexData( i );
                    vector<float> vad( dim * valuesOut.size() );
                    for ( size_t k=0 ; k<valuesOut.size() ; k++ )
                    {
                      for ( unsigned int l=0 ; l<dim ; l++ )
                      {
                        vad[dim*k+l] = valuesOut[k][j+l];
                      }
                    }
                    newVASH->setVertexData( i, dim, dp::util::DT_FLOAT_32, &vad[0], 0, checked_cast<unsigned int>(vad.size()/dim) );

                    // inherit enable states from source attrib
                    // normalize-enable state only meaningful for generic aliases!
                    newVASH->setEnabled(i, p->isEnabled(i)); // conventional
                    newVASH->setEnabled(i+16, p->isEnabled(i+16)); // generic
                    newVASH->setNormalizeEnabled(i+16, p->isNormalizeEnabled(i+16)); // generic only!
                    j += dim;
                  }
                }

                //  now replace the VertexAttributeSet of all it's owners by the new one and adjust their index information
                while ( 0 < p->getNumberOfOwners() )
                {
                  DP_ASSERT( isPtrTo<Primitive>(p->getOwner(p->ownersBegin())) );
                  PrimitiveSharedPtr primitive = p->getOwner( p->ownersBegin() )->getSharedPtr<Primitive>();

                  IndexSetSharedPtr newIndexSet( IndexSet::create() );
                  if ( primitive->isIndexed() )
                  {
                    IndexSetSharedPtr oldIndexSet = primitive->getIndexSet();
                    unsigned int pri = oldIndexSet->getPrimitiveRestartIndex();
                    vector<unsigned int> newIndices( primitive->getElementCount() );
                    IndexSet::ConstIterator<unsigned int> oldIndices( oldIndexSet, primitive->getElementOffset() );
                    for ( size_t i=0 ; i<newIndices.size() ; i++ )
                    {
                      newIndices[i] = ( oldIndices[i] == pri ) ? pri : indexMap[oldIndices[i]];
                    }
                    newIndexSet->setData( &newIndices[0], checked_cast<unsigned int>(newIndices.size()) );
                  }
                  else
                  {
                    newIndexSet->setData( &indexMap[0], checked_cast<unsigned int>(indexMap.size()) );
                  }
                  primitive->setIndexSet( newIndexSet );
                  primitive->setVertexAttributeSet( newVASH );
                  primitive->setElementRange( 0, ~0 );
                }

                //  store the new VertexAttributeSet as already handled (if it's multiply owned)
                if ( 1 < newVASH->getNumberOfOwners() )
                {
                  m_multiOwnedHandledVAS.insert( newVASH );
                }
              }
              else
              {
                delete octree;

                //  store the old VertexAttributeSet as already handled (if it's multiply owned)
                if ( 1 < p->getNumberOfOwners() )
                {
                  m_multiOwnedHandledVAS.insert( vas );
                }
              }
            }
          }
        }
      }

      void UnifyTraverser::unifyChildren( Group *p )
      {
        // make sure we can optimize the children of this group
        if( optimizationAllowed( p->getSharedPtr<Group>() ) )
        {
          if ( m_unifyTargets & UT_GEONODE )
          {
            unifyGeoNodes( p );
          }
          if ( m_unifyTargets & UT_GROUP )
          {
            unifyGroups( p );
          }
          if ( m_unifyTargets & UT_LOD )
          {
            unifyLODs( p );
          }
        }
      }

      void UnifyTraverser::unifyGeoNodes( Group *p )
      {
        DP_ASSERT( m_unifyTargets & UT_GEONODE );

        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( gci->isPtrTo<GeoNode>() )
          {
            GeoNodeSharedPtr const& geoNode = gci->staticCast<GeoNode>();
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
        DP_ASSERT( m_unifyTargets & UT_GROUP );

        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( gci->isPtrTo<Group>() )
          {
            GroupSharedPtr const& group = gci->staticCast<Group>();
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
        DP_ASSERT( m_unifyTargets & UT_INDEX_SET );
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
        DP_ASSERT( m_unifyTargets & UT_LOD );

        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( gci->isPtrTo<LOD>() )
          {
            bool optimizable = false, found = false;
            LODSharedPtr const& lod = gci->staticCast<LOD>();
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
        DP_ASSERT( m_unifyTargets & UT_PRIMITIVE );

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
        PrimitiveWeakPtr pwp = getWeakPtr<Primitive>( p );
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
            foundPrimitive = it->second.getWeakPtr();
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
          DP_ASSERT( !checkFound || ( primitive == it->second ) || ( foundPrimitive == it->second.getWeakPtr() ) );
        }
        DP_ASSERT( found == checkFound );
#endif

        // if we did not found that Primitive (or an equivalent one) before -> store it for later searches
        if ( ! found )
        {
          v.insert( make_pair( hashKey, primitive ) );
        }
      }

      void UnifyTraverser::unifyVertexAttributeSet( Primitive *p )
      {
        DP_ASSERT( m_unifyTargets & UT_VERTEX_ATTRIBUTE_SET );
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
            DP_ASSERT( !checkFound || ( vertexAttributeSet == it->second ) || ( p->getVertexAttributeSet() == it->second.getWeakPtr() ) );
          }
          DP_ASSERT( found == checkFound );
#endif
        }
        if ( ! found )
        {
          m_vertexAttributeSets.insert( make_pair( hashKey, vertexAttributeSet ) );
        }
      }

      //  Compare the first three components of arrays of float
      //  These are the position of a vertex
      bool areSimilarFirst( const float * v0, const float * v1, float eps )
      {
        for ( unsigned int i=0 ; i<3 ; i++ )
        {
          if ( eps < fabsf( v0[i] - v1[i] ) )
          {
            return( false );
          }
        }
        return( true );
      }

      //  Compare all but the first three components of two arrays of float
      //  These are the vertex attributes without position
      bool areSimilarLast( const float * v0, const float * v1, unsigned int n, float eps )
      {
        DP_ASSERT( 3 <= n );
        for ( unsigned int i=3 ; i<n ; i++ )
        {
          if ( eps < fabsf( v0[i] - v1[i] ) )
          {
            return( false );
          }
        }
        return( true );
      }

      VUTOctreeIndices::VUTOctreeIndices( unsigned int maxIndices )
        : m_maxIndices(maxIndices)
      {
      }

      VUTOctreeIndices::~VUTOctreeIndices( void )
      {
      }

      bool  VUTOctreeIndices::addVertex( const vector<float *> &vertices, unsigned int index, unsigned int dimension, float tol )
      {
        const float * v = vertices[index];
        bool inserted = false;
        for ( IndexContainer::iterator it = m_indices.begin() ; it != m_indices.end() ; ++it )
        {
          //  determine if the vertex v is already represented in this octel
          if ( areSimilarFirst( v, vertices[(*it)[0][0]], tol ) )
          {
            //  the vertex v has a similar position as a previously encountered vertex
            for ( size_t j=0 ; j<it->size() ; j++ )
            {
              //  determine if one of those vertices with the same position also has the same attributes
              if ( areSimilarLast( v, vertices[(*it)[j][0]], dimension, tol ) )
              {
                //  the vertex v also has similar attributes as a previously encountered vertex
                //  => push the index of v and break the inner loop
                (*it)[j].push_back( index );
                inserted = true;
                break;
              }
            }
            if ( ! inserted )
            {
              //  there is no previously encountered vertex with similar attributes
              //  => push the index of v into a new vector of indices and break the loop
              it->push_back( vector<unsigned int>() );
              it->back().push_back( index );
              inserted = true;
              break;
            }
          }
        }
        if ( ! inserted && ( m_indices.size() < m_maxIndices ) )
        {
          //  there is no previously encountered vertex with the same position and there is still space for a new
          //  set of indices => push the index of v into a new vector of indices...
          m_indices.push_back( vector<vector<unsigned int> >() );
          m_indices.back().push_back( vector<unsigned int>() );
          m_indices.back().back().push_back( index );
          inserted = true;
        }
        return( inserted );
      }

      void VUTOctreeIndices::addIndexList( const vector<vector<unsigned int> > & indexList )
      {
        //  push the new indexList
        DP_ASSERT( m_indices.size() < m_maxIndices );
        m_indices.push_back( indexList );
      }

      void VUTOctreeIndices::clear( void )
      {
        //  clear the indices without deleting all the contents; these should already be pushed into other
        //  VUTOctreeIndices elements
        m_indices.clear();
      }

      VUTOctreeIndices::IndexContainer::const_iterator VUTOctreeIndices::getIndexListBegin() const
      {
        return( m_indices.begin() );
      }

      VUTOctreeIndices::IndexContainer::const_iterator VUTOctreeIndices::getIndexListEnd() const
      {
        return( m_indices.end() );
      }

      unsigned int VUTOctreeIndices::getMaxIndices( void ) const
      {
        return( m_maxIndices );
      }

      unsigned int VUTOctreeIndices::getNumberOfPoints( void ) const
      {
        //  get the number of different points in that octel
        size_t nop = 0;
        for ( IndexContainer::const_iterator it = m_indices.begin() ; it != m_indices.end() ; ++it )
        {
          nop += it->size();
        }
        return( checked_cast<unsigned int>(nop) );
      }

      void VUTOctreeIndices::mapValues( const vector<float *> & valuesIn, vector<float *> & valuesOut, unsigned int dimension, vector<unsigned int> &indexMap, unsigned int &index )
      {
        // loop over all vertices in the octel
        for ( IndexContainer::const_iterator it = m_indices.begin() ; it != m_indices.end() ; ++it )
        {
          DP_ASSERT( it->size() );
          // loop over all vertices with equal position (at least one)
          for ( size_t j=0 ; j<it->size() ; j++ )
          {
            DP_ASSERT( (*it)[j].size() );
            //  get the index of the first vertex in the set of similar vertices
            size_t first = (*it)[j][0];
            //  get the first vertex into the valuesOut vector
            memcpy( valuesOut[index], valuesIn[first], dimension * sizeof(float) );
            //  and store it's new index into the indexMap
            indexMap[first] = index;
            if ( 1 < (*it)[j].size() )
            {
              //  there are more than one similar vertices
              //  => calculate the representation as the average of all of them
              for ( size_t k=1 ; k<(*it)[j].size() ; k++ )
              {
                size_t next = (*it)[j][k];
                for ( unsigned int d=0 ; d<dimension ; d++ )
                {
                  valuesOut[index][d] += valuesIn[next][d];
                }
                indexMap[next] = index;
              }
              for ( unsigned int d=0 ; d<dimension ; d++ )
              {
                valuesOut[index][d] /= (float)((*it)[j].size());
              }
            }
            index++;
          }
        }
      }

      VUTOctreeSubNodes::VUTOctreeSubNodes( const Box3f & box, const Vec3f & center, unsigned int maxIndicesPerOctel )
      {
        const Vec3f & lower = box.getLower();
        const Vec3f & upper = box.getUpper();

        nodes[0][0][0].init( Box3f( Vec3f( lower[0], lower[1], lower[2] ), Vec3f( center[0], center[1], center[2] ) ), maxIndicesPerOctel );
        nodes[0][0][1].init( Box3f( Vec3f( lower[0], lower[1], center[2] ), Vec3f( center[0], center[1], upper[2] ) ), maxIndicesPerOctel );
        nodes[0][1][0].init( Box3f( Vec3f( lower[0], center[1], lower[2] ), Vec3f( center[0], upper[1], center[2] ) ), maxIndicesPerOctel );
        nodes[0][1][1].init( Box3f( Vec3f( lower[0], center[1], center[2] ), Vec3f( center[0], upper[1], upper[2] ) ), maxIndicesPerOctel );
        nodes[1][0][0].init( Box3f( Vec3f( center[0], lower[1], lower[2] ), Vec3f( upper[0], center[1], center[2] ) ), maxIndicesPerOctel );
        nodes[1][0][1].init( Box3f( Vec3f( center[0], lower[1], center[2] ), Vec3f( upper[0], center[1], upper[2] ) ), maxIndicesPerOctel );
        nodes[1][1][0].init( Box3f( Vec3f( center[0], center[1], lower[2] ), Vec3f( upper[0], upper[1], center[2] ) ), maxIndicesPerOctel );
        nodes[1][1][1].init( Box3f( Vec3f( center[0], center[1], center[2] ), Vec3f( upper[0], upper[1], upper[2] ) ), maxIndicesPerOctel );
      }

      void VUTOctreeSubNodes::addIndexList( const vector<vector<unsigned int> > & indexList, const vector<float*> &vertices, const Vec3f & center )
      {
        const float * v = vertices[indexList[0][0]];
        nodes[center[0]<=v[0]][center[1]<=v[1]][center[2]<=v[2]].addIndexList( indexList );
      }

      void VUTOctreeSubNodes::addVertex( const vector<float*> &vertices, unsigned int index, unsigned int dimension, float tol, unsigned int level, const Vec3f & center )
      {
        const float * v = vertices[index];
        nodes[center[0]<=v[0]][center[1]<=v[1]][center[2]<=v[2]].addVertex( vertices, index, dimension, tol, level );
      }

      unsigned int VUTOctreeSubNodes::getNumberOfPoints() const
      {
        return( nodes[0][0][0].getNumberOfPoints()
              + nodes[0][0][1].getNumberOfPoints()
              + nodes[0][1][0].getNumberOfPoints()
              + nodes[0][1][1].getNumberOfPoints()
              + nodes[1][0][0].getNumberOfPoints()
              + nodes[1][0][1].getNumberOfPoints()
              + nodes[1][1][0].getNumberOfPoints()
              + nodes[1][1][1].getNumberOfPoints() );
      }

      void VUTOctreeSubNodes::mapValues( const vector<float *> & valuesIn, vector<float *> & valuesOut, unsigned int dimension, vector<unsigned int> &indexMap, unsigned int &index )
      {
        nodes[0][0][0].mapValues( valuesIn, valuesOut, dimension, indexMap, index );
        nodes[0][0][1].mapValues( valuesIn, valuesOut, dimension, indexMap, index );
        nodes[0][1][0].mapValues( valuesIn, valuesOut, dimension, indexMap, index );
        nodes[0][1][1].mapValues( valuesIn, valuesOut, dimension, indexMap, index );
        nodes[1][0][0].mapValues( valuesIn, valuesOut, dimension, indexMap, index );
        nodes[1][0][1].mapValues( valuesIn, valuesOut, dimension, indexMap, index );
        nodes[1][1][0].mapValues( valuesIn, valuesOut, dimension, indexMap, index );
        nodes[1][1][1].mapValues( valuesIn, valuesOut, dimension, indexMap, index );
      }

      VUTOctreeNode::~VUTOctreeNode( void )
      {
        if ( m_data )
        {
          delete m_data;
        }
        else
        {
          DP_ASSERT( m_nodes );
          delete m_nodes;
        }
      }

      void VUTOctreeNode::addVertex( const vector<float *> &vertices, unsigned int index, unsigned int dimension, float tol, unsigned int level )
      {
        if ( m_data )
        {
          //  the octel holds data, so try to add the vertex here
          if ( ! m_data->addVertex( vertices, index, dimension, tol ) )
          {
            //  the vertex couldn't be added, so distribute the data of this octel
            distributeData( vertices, tol );
            //  and add the vertex to the appropriate sub-octel
            m_nodes->addVertex( vertices, index, dimension, tol, level+1, m_center );
          }
        }
        else
        {
          //  the octel is not a leave, so add the vertex to the appropriat sub-octel
          m_nodes->addVertex( vertices, index, dimension, tol, level+1, m_center );
        }
      }

      void VUTOctreeNode::addIndexList( const vector<vector<unsigned int> > & indexList )
      {
        DP_ASSERT( m_data );
        m_data->addIndexList( indexList );
      }

      void VUTOctreeNode::distributeData( const vector<float *> & vertices, float tol )
      {
        unsigned int maxIndicesPerOctel = m_data->getMaxIndices();

        //  create the eight sub-octels with half the size
        m_nodes = new VUTOctreeSubNodes( m_box, m_center, maxIndicesPerOctel );

        //  distribute the index lists into the sub-octels
        for ( VUTOctreeIndices::IndexContainer::const_iterator it = m_data->getIndexListBegin() ; it != m_data->getIndexListEnd() ; ++it )
        {
          m_nodes->addIndexList( *it, vertices, m_center );
        }

        //  clear the m_data to make clear, this octel is not a leave anymore
        m_data->clear();
        delete m_data;
        m_data = NULL;
      }

      unsigned int VUTOctreeNode::getNumberOfPoints( void ) const
      {
        return( m_data ? m_data->getNumberOfPoints() : m_nodes->getNumberOfPoints() );
      }

      void VUTOctreeNode::init( const Box3f & bbox, unsigned int maxIndicesPerOctel )
      {
        m_box = bbox;
        m_center = bbox.getCenter();
        m_data = new VUTOctreeIndices( maxIndicesPerOctel );
        m_nodes = NULL;
      }

      void VUTOctreeNode::mapValues( const vector<float *> & valuesIn, vector<float *> & valuesOut, unsigned int dimension, vector<unsigned int> &indexMap, unsigned int &index )
      {
        if ( m_data )
        {
          m_data->mapValues( valuesIn, valuesOut, dimension, indexMap, index );
        }
        else
        {
          m_nodes->mapValues( valuesIn, valuesOut, dimension, indexMap, index );
        }
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
