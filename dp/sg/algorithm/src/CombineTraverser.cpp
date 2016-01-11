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


#include <iostream>
#include <vector>
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/BufferHost.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/algorithm/CombineTraverser.h>
#include <dp/util/HashGeneratorMurMur.h>

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using std::map;
using std::multimap;
using std::pair;
using std::set;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      DEFINE_STATIC_PROPERTY( CombineTraverser, CombineTargets );
      DEFINE_STATIC_PROPERTY( CombineTraverser, IgnoreAccelerationBuilderHints );

      BEGIN_REFLECTION_INFO( CombineTraverser )
        DERIVE_STATIC_PROPERTIES( CombineTraverser, OptimizeTraverser );
        INIT_STATIC_PROPERTY_RW( CombineTraverser, CombineTargets,                 CombineTraverser::TargetMask, Semantic::VALUE, value, value );
        INIT_STATIC_PROPERTY_RW( CombineTraverser, IgnoreAccelerationBuilderHints, bool                        , Semantic::VALUE, value, value );
      END_REFLECTION_INFO

      /** \brief Combines two vertex attributes. Copies data in a new host buffer
      **/
      static void combine( VertexAttribute &v0, const VertexAttribute &v1)
      {
        DP_ASSERT( v0.getVertexDataBytes() == v1.getVertexDataBytes() );
        DP_ASSERT( v0.getVertexDataSize() == v1.getVertexDataSize() );
        DP_ASSERT( v0.getVertexDataType() == v1.getVertexDataType() );

        BufferSharedPtr buffer = BufferHost::create();
        buffer->setSize( ( v0.getVertexDataCount() + v1.getVertexDataCount() ) * v0.getVertexDataBytes() );

        Buffer::Iterator<char>::Type itDst = buffer->getIterator<char>( Buffer::MAP_WRITE, v0.getVertexDataBytes() );
        Buffer::ConstIterator<char>::Type itSrc0 = v0.beginRead<char>();
        Buffer::ConstIterator<char>::Type itSrc1 = v1.beginRead<char>();

        size_t vertexDataBytes = v0.getVertexDataBytes();
        for ( size_t index = 0;index < v0.getVertexDataCount(); ++index)
        {
          memcpy( &(*itDst), &(*itSrc0), vertexDataBytes );
          ++itDst;
          ++itSrc0;
        }

        for ( size_t index = 0;index < v1.getVertexDataCount(); ++index)
        {
          memcpy( &(*itDst), &(*itSrc1), vertexDataBytes );
          ++itDst;
          ++itSrc1;
        }

        v0.setData( v0.getVertexDataSize(), v0.getVertexDataType(), buffer, 0, (unsigned int)vertexDataBytes, v0.getVertexDataCount() + v1.getVertexDataCount() );
      }

      static bool needsPrimitiveRestartIndex( PrimitiveType type )
      {
        return(   type == PrimitiveType::LINE_STRIP
               || type == PrimitiveType::LINE_LOOP
               || type == PrimitiveType::TRIANGLE_STRIP
               || type == PrimitiveType::TRIANGLE_FAN
               || type == PrimitiveType::QUAD_STRIP
               || type == PrimitiveType::POLYGON
               || type == PrimitiveType::TRIANGLE_STRIP_ADJACENCY
               || type == PrimitiveType::LINE_STRIP_ADJACENCY );
      }

      CombineTraverser::CombineTraverser( void )
      : m_combineTargets(Target::ALL)
      {
      }

      CombineTraverser::~CombineTraverser( void )
      {
      }

      void CombineTraverser::postApply( const NodeSharedPtr & root )
      {
        OptimizeTraverser::postApply( root );
        m_objects.clear();
      }

      void CombineTraverser::handleBillboard( Billboard *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          filterMultiple( p );
          OptimizeTraverser::handleBillboard( p );
          if ( m_combineTargets & Target::GEONODE )
          {
            combineGeoNodes( p );
          }
          if ( m_combineTargets & Target::LOD )
          {
            combineLODs( p );
          }
          if ( m_combineTargets & Target::TRANSFORM )
          {
            combineTransforms( p );
          }
        }
      }

      void CombineTraverser::handleGroup( Group *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          filterMultiple( p );
          OptimizeTraverser::handleGroup( p );
          if ( m_combineTargets & Target::GEONODE )
          {
            combineGeoNodes( p );
          }
          if ( m_combineTargets & Target::LOD )
          {
            combineLODs( p );
          }
          if ( m_combineTargets & Target::TRANSFORM )
          {
            combineTransforms( p );
          }
        }
      }

      void CombineTraverser::handleLOD( LOD * p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          if ( m_combineTargets & Target::LOD_RANGES )
          {
            combineLODRanges( p );
          }
          OptimizeTraverser::handleLOD( p );
        }
      }

      void CombineTraverser::handleTransform( Transform *p )
      {
        pair<set<const void *>::iterator,bool> pitb = m_objects.insert( p );
        if ( pitb.second )
        {
          filterMultiple( p );
          OptimizeTraverser::handleTransform( p );
          if ( m_combineTargets & Target::GEONODE )
          {
            combineGeoNodes( p );
          }
          if ( m_combineTargets & Target::LOD )
          {
            combineLODs( p );
          }
          if ( m_combineTargets & Target::TRANSFORM )
          {
            combineTransforms( p );
          }
        }
      }

      bool CombineTraverser::areCombinable( GeoNodeSharedPtr const& p0, GeoNodeSharedPtr const& p1, bool ignoreNames )
      {
        bool combinable =   areCombinable( ObjectSharedPtr( p0 ), ObjectSharedPtr( p1 ), ignoreNames )
                        &&  ( p0->getMaterialPipeline() == p1->getMaterialPipeline() )
                        &&  ( !!p0->getPrimitive() == !!p1->getPrimitive() )
                        &&  (   !p0->getPrimitive()
                            ||  ( p0->getPrimitive()->getObjectCode() ==  p1->getPrimitive()->getObjectCode() ) );
        if ( combinable && p0->getPrimitive() )
        {
          combinable = areCombinable( p0->getPrimitive(), p1->getPrimitive(), ignoreNames );
        }
        return( combinable );
      }

      bool CombineTraverser::areCombinable( GroupSharedPtr const& p0, GroupSharedPtr const& p1, bool ignoreNames )
      {
        bool combinable =   areCombinable( ObjectSharedPtr( p0 ), ObjectSharedPtr( p1 ), ignoreNames )
                        && ( p0->getNumberOfClipPlanes() == p1->getNumberOfClipPlanes() );

        if ( combinable )
        {
          // test if each clip plane in p0 also is in p1
          // no test in other direction needed, because we know the number of clip planes is the same
          for ( Group::ClipPlaneIterator gcpci = p0->beginClipPlanes() ; gcpci != p0->endClipPlanes() ; ++gcpci )
          {
            if ( p1->findClipPlane( *gcpci ) == p1->endClipPlanes() )
            {
              return( false );
            }
          }
        }
        return( combinable );
      }

      bool CombineTraverser::areCombinable( LODSharedPtr const& p0, LODSharedPtr const& p1, bool ignoreNames )
      {
        bool combinable =   areCombinable( GroupSharedPtr( p0 ), GroupSharedPtr( p1 ), ignoreNames )
                        &&  ( p0->getNumberOfRanges() == p1->getNumberOfRanges() )
                        &&  ( length( p0->getCenter() - p1->getCenter() ) < FLT_EPSILON );

        // test if the ranges differ by less than epsilon
        for ( unsigned int i=0 ; combinable && i<p0->getNumberOfRanges() ; i++ )
        {
          combinable = ( fabsf( p1->getRanges()[i] - p0->getRanges()[i] ) < FLT_EPSILON );
        }

        return( combinable );
      }

      bool CombineTraverser::areCombinable( ObjectSharedPtr const& p0, ObjectSharedPtr const& p1, bool ignoreNames )
      {
        return(   ( ignoreNames || ( p0->getName() == p1->getName() ) )
              &&  ( p0->getAnnotation() == p1->getAnnotation() )      // we can't combine objects with (different) annotations
              &&  ( p0->getUserData() == p1->getUserData() )
              &&  ( p0->getHints() == p1->getHints() ) );
      }

      bool CombineTraverser::areCombinable( PrimitiveSharedPtr const& p0, PrimitiveSharedPtr const& p1, bool ignoreNames )
      {
        DP_ASSERT( p0->getVertexAttributeSet() && p1->getVertexAttributeSet() );
        return(   areCombinable( ObjectSharedPtr( p0 ), ObjectSharedPtr( p1 ), ignoreNames )
              &&  ( p0->getInstanceCount() == p1->getInstanceCount() )
              &&  ( p0->getPrimitiveType() == p1->getPrimitiveType() )
              &&  ( p0->getPatchesType() == p1->getPatchesType() )
              &&  areCombinable( p0->getVertexAttributeSet(), p1->getVertexAttributeSet(), ignoreNames ) );
      }

      bool CombineTraverser::areCombinable( TransformSharedPtr const& p0, TransformSharedPtr const& p1, bool ignoreNames )
      {
        return(   areCombinable( GroupSharedPtr( p0 ), GroupSharedPtr( p1 ), ignoreNames )
              &&  ! ( p0->isJoint() || p1->isJoint() )
              &&  ( p0->getTrafo() == p1->getTrafo() ) );
      }

      bool CombineTraverser::areCombinable( VertexAttributeSetSharedPtr const& p0, VertexAttributeSetSharedPtr const& p1, bool ignoreNames )
      {
        bool combinable =   areCombinable( ObjectSharedPtr( p0 ), ObjectSharedPtr( p1 ), ignoreNames );
        for ( unsigned int i=0 ; combinable && i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT) ; i++ )
        {
          VertexAttributeSet::AttributeID attribute = static_cast<VertexAttributeSet::AttributeID>(i);
          combinable =  ( p0->getSizeOfVertexData( attribute ) == p1->getSizeOfVertexData( attribute ) )
                    &&  ( p0->getTypeOfVertexData( attribute ) == p1->getTypeOfVertexData( attribute ) )
                    &&  ( p0->isEnabled( attribute )           == p1->isEnabled( attribute ) )
                    &&  ( p0->isNormalizeEnabled( attribute )  == p1->isNormalizeEnabled( attribute ) );
        }
        return( combinable );
      }

      void CombineTraverser::combineGeoNodes( Group *p )
      {
        DP_ASSERT( m_combineTargets & Target::GEONODE );

        if ( optimizationAllowed( p->getSharedPtr<Group>() ) )
        {
          // get the combinable GeoNodes in bins
          map<GeoNodeSharedPtr,vector<GeoNodeSharedPtr> >  geoNodes;

          for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
          {
            if ( gci->isPtrTo<GeoNode>() && optimizationAllowed( *gci ) )
            {
              GeoNodeSharedPtr const& geoNode = gci->staticCast<GeoNode>();

              map<GeoNodeSharedPtr,vector<GeoNodeSharedPtr> >::iterator it;
              for ( it = geoNodes.begin() ; it != geoNodes.end() ; ++it )
              {
                if ( areCombinable( geoNode, it->first, getIgnoreNames() ) )
                {
                  DP_ASSERT( *gci != it->first );
                  if ( geoNode->getPrimitive() == it->first->getPrimitive() )
                  {
                    gci = p->removeChild( gci );
                    --gci;
                  }
                  else
                  {
                    it->second.push_back( geoNode );
                  }
                  break;
                }
              }
              if ( it == geoNodes.end() )
              {
                geoNodes[geoNode];
              }
            }
          }

          // combine combinable GeoNodes
          for ( map<GeoNodeSharedPtr,vector<GeoNodeSharedPtr> >::const_iterator it = geoNodes.begin() ; it != geoNodes.end() ; ++it )
          {
            if ( ! it->second.empty() )
            {
              // reduce Primitives to those data that are really referenced, and get the number of indices and vertices
              unsigned int noi, nov;
              const PrimitiveSharedPtr & basePrimitive = it->first->getPrimitive();
              reduceVertexAttributeSet( basePrimitive );
              noi = basePrimitive->getElementCount();
              if ( needsPrimitiveRestartIndex( basePrimitive->getPrimitiveType() ) )
              {
                noi += dp::checked_cast<unsigned int>( it->second.size() );
              }
              nov = basePrimitive->getVertexAttributeSet()->getNumberOfVertexData( VertexAttributeSet::AttributeID::NORMAL );

              for ( size_t i=0 ; i<it->second.size() ; i++ )
              {
                const PrimitiveSharedPtr & primitive = it->second[i]->getPrimitive();
                reduceVertexAttributeSet( primitive );
                DP_ASSERT( primitive->getPrimitiveType() == basePrimitive->getPrimitiveType() );
                noi += primitive->getElementCount();
                nov += primitive->getVertexAttributeSet()->getNumberOfVertexData( VertexAttributeSet::AttributeID::NORMAL );
              }

              // create a clone of the first Primitive
              {
                VertexAttributeSetSharedPtr const& newVAS = basePrimitive->getVertexAttributeSet();

                // reserve new vertex data set accordingly
                for ( unsigned int j=0 ; j<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT) ; j++ )
                {
                  VertexAttributeSet::AttributeID attribute = static_cast<VertexAttributeSet::AttributeID>(j);
                  if ( newVAS->getSizeOfVertexData( attribute ) )
                  {
                    newVAS->reserveVertexData( attribute, newVAS->getSizeOfVertexData( attribute ), newVAS->getTypeOfVertexData( attribute ), nov );
                  }
                }
              }

              // gather every combinable primitive into the basePrimitive, gathering indices into combinedIndices
              vector<unsigned int> combinedIndices( noi );
              unsigned int indexOffset = gatherIndices( basePrimitive, &combinedIndices[0], 0 );
              for ( size_t i=0 ; i<it->second.size() ; i++ )
              {
                indexOffset = combine( basePrimitive, combinedIndices, indexOffset, it->second[i]->getPrimitive() );
                p->removeChild( it->second[i] );
              }
              DP_ASSERT( indexOffset == noi );

              // create an IndexSet with the combinedIndices and put it into basePrimitive
              IndexSetSharedPtr newIndexSet = IndexSet::create();
              newIndexSet->setData( &combinedIndices[0], indexOffset );
              basePrimitive->setIndexSet( newIndexSet );
              basePrimitive->setElementRange( 0, ~0 );     // make sure, the complete IndexSet is used !
              setTreeModified();
            }
          }
        }
      }

      void CombineTraverser::combineLODs( Group *p )
      {
        DP_ASSERT( m_combineTargets & Target::LOD );

        if( !optimizationAllowed( p->getSharedPtr<Group>() ) )
        {
          return;
        }

        //  first look for all LOD children
        vector<vector<LODSharedPtr> >  lods;
        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if( gci->isPtrTo<LOD>() )
          {
            LODSharedPtr const& lod = gci->staticCast<LOD>();
            if( optimizationAllowed( lod ) )
            {
              unsigned int j = 0;
              for ( /**/ ; j<lods.size() ; j++ )
              {
                if ( areCombinable( lod, lods[j][0], getIgnoreNames() ) )
                {
                  break;
                }
              }
              if ( j == lods.size() )
              {
                lods.resize( j + 1 );
              }
              lods[j].push_back( lod );
            }
          }
        }

        for ( size_t i=0 ; i<lods.size() ; i++ )
        {
          if ( 1 < lods[i].size() )
          {
            //  If there are more than one compatible LOD children, create a new LOD with a Group on each level
            //  that gathers the children of the compatible LODs, and replace them all by this new one.
            LODSharedPtr newLOD = LOD::create();
            LODSharedPtr const& lod = lods[i][0];
            newLOD->setName( lod->getName() );
            for ( Group::ClipPlaneIterator gcpci = lod->beginClipPlanes() ; gcpci != lod->endClipPlanes() ; ++gcpci )
            {
              newLOD->addClipPlane( *gcpci );
            }
            newLOD->setRanges( lod->getRanges(), lod->getNumberOfRanges() );
            for ( unsigned int j=0 ; j<=newLOD->getNumberOfRanges() ; j++ )
            {
              newLOD->addChild( Group::create() );
            }

            for ( size_t j=0 ; j<lods[i].size() ; j++ )
            {
              LODSharedPtr const& lod = lods[i][0];
              Group::ChildrenIterator gcciLOD = lod->beginChildren();
              Group::ChildrenIterator gciNewLOD = newLOD->beginChildren();
              for (
                  ; ( gcciLOD != lod->endChildren() ) && ( gciNewLOD != newLOD->endChildren() )
                  ; ++gcciLOD, ++gciNewLOD )
              {
              }
              for ( --gcciLOD ; gciNewLOD != newLOD->endChildren() ; ++gciNewLOD )
              {
              }
              p->removeChild( lods[i][j] );
            }

            p->addChild( newLOD );
            setTreeModified();
          }
        }
      }

      void CombineTraverser::combineLODRanges( LOD *p )
      {
        DP_ASSERT( m_combineTargets & Target::LOD_RANGES );

        if( !optimizationAllowed( p->getSharedPtr<LOD>() ) )
        {
          return;
        }

        bool doCombine = false;
        if ( 1 < p->getNumberOfChildren() )
        {
          for ( Group::ChildrenIterator gciPrev = p->beginChildren(), gci = ++(p->beginChildren())
              ; gci != p->endChildren()
              ; ++gci, ++gciPrev )
          {
            // if the children are equal, AND can be optimized
            doCombine = ( *gciPrev == *gci ) && optimizationAllowed( *gci );
          }
        }

        if ( doCombine )
        {
          vector<float> combinedRanges;
          const float * ranges = p->getRanges();
          Group::ChildrenIterator gciPrev = p->beginChildren();
          Group::ChildrenIterator gci = ++(p->beginChildren());
          for ( unsigned int r = 0 ; gci != p->endChildren() ; ++r )
          {
            if ( *gciPrev == *gci )
            {
              gci = p->removeChild( gci );   // gciPrev unchanged !
            }
            else
            {
              if ( r < p->getNumberOfRanges() )
              {
                combinedRanges.push_back( ranges[r] );
              }
              gciPrev = gci;
              ++gci;
            }
          }
          if ( p->setRanges( combinedRanges.empty() ? NULL : &combinedRanges[0]
                           , dp::checked_cast<unsigned int>(combinedRanges.size()) ) )
          {
            setTreeModified();
          }
        }
      }

      void CombineTraverser::combineTransforms( Group *p )
      {
        DP_ASSERT( m_combineTargets & Target::TRANSFORM );

        if( !optimizationAllowed( p->getSharedPtr<Group>() ) )
        {
          return;
        }

        //  first look for all Transform children
        typedef multimap<HashKey,vector<vector<TransformSharedPtr> > >  TransformMap;
        typedef TransformMap::iterator I;

        TransformMap transforms;
        for ( Group::ChildrenIterator gci = p->beginChildren() ; gci != p->endChildren() ; ++gci )
        {
          if ( ((*gci)->getObjectCode() == ObjectCode::TRANSFORM) && optimizationAllowed( *gci ) )
          {
            //  gather Transforms in compatible bins
            TransformSharedPtr const& transform = gci->staticCast<Transform>();

            HashKey hashKey;
            HashGeneratorMurMur hg;
            Trafo trafo = transform->getTrafo();
            hg.update( reinterpret_cast<const unsigned char *>(&trafo), sizeof(Trafo) );
            hg.finalize( &hashKey );

            pair<I,I> itp = transforms.equal_range( hashKey );

            bool found = false;
            for ( I it=itp.first ; it!= itp.second && !found ; ++it )
            {
              for ( unsigned int j=0 ; j<it->second.size() && !found ; j++ )
              {
                found =   ( *gci == it->second[j][0] )
                      ||  areCombinable( transform, it->second[j][0], getIgnoreNames() );
                if ( found && ( *gci != it->second[j][0] ) )
                {
                  it->second[j].push_back( transform );
                }
              }
            }
            if ( ! found )
            {
              I it = transforms.insert( make_pair( hashKey, vector<vector<TransformSharedPtr> >() ) );
              it->second.push_back( vector<TransformSharedPtr>() );
              it->second.back().push_back( transform );
            }
          }
        }

        for ( I it = transforms.begin() ; it != transforms.end() ; ++it )
        {
          for ( size_t i=0 ; i<it->second.size() ; i++ )
          {
            if ( 1 < it->second[i].size() )
            {
              TransformSharedPtr newTransform = Transform::create();
              newTransform->setName( it->second[i][0]->getName() );
              newTransform->setHints( it->second[i][0]->getHints() );
              for ( Group::ClipPlaneIterator gcpci = it->second[i][0]->beginClipPlanes() ; gcpci != it->second[i][0]->endClipPlanes() ; ++gcpci )
              {
                newTransform->addClipPlane( *gcpci );
              }
              newTransform->setTrafo( it->second[i][0]->getTrafo() );
              DP_ASSERT( !it->second[i][0]->isJoint() );
              for ( size_t j=0 ; j<it->second[i].size() ; j++ )
              {
                for ( Group::ChildrenIterator gcci = it->second[i][0]->beginChildren() ; gcci != it->second[i][0]->endChildren() ; ++gcci )
                {
                  newTransform->addChild( *gcci );
                }
                p->removeChild( it->second[i][j] );
              }
              p->addChild( newTransform );
              setTreeModified();
            }
          }
        }
      }

      void CombineTraverser::filterMultiple( Group * p )
      {
        if( optimizationAllowed( p->getSharedPtr<Group>() ) )
        {
          set<NodeSharedPtr>  children;
          for ( Group::ChildrenIterator it = p->beginChildren() ; it != p->endChildren() ; ++it )
          {
            children.insert( *it );
          }
          if ( children.size() < p->getNumberOfChildren() )
          {
            p->clearChildren();
            for ( set<NodeSharedPtr>::const_iterator it = children.begin() ; it != children.end() ; ++it )
            {
              p->addChild( *it );
            }
            setTreeModified();
          }
        }
      }

      void CombineTraverser::combine( const VertexAttributeSetSharedPtr & vash0
                                    , const VertexAttributeSetSharedPtr & vash1 )
      {
        DP_ASSERT( vash0->getObjectCode() == vash1->getObjectCode() );
        for ( unsigned int i=0 ; i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT) ; i++ )
        {
          VertexAttributeSet::AttributeID attribute = static_cast<VertexAttributeSet::AttributeID>(i);
          unsigned int size = vash1->getSizeOfVertexData(attribute);
          if ( size )
          {
            DP_ASSERT( vash0->getVertexBuffer(attribute) != vash1->getVertexBuffer(attribute) && "shared buffers not supported" );
            DP_ASSERT( !vash0->getOffsetOfVertexData(attribute) && !vash1->getOffsetOfVertexData(attribute) && "buffer offsets not supported" );
            DP_ASSERT( vash0->isContiguousVertexData(attribute) && vash1->isContiguousVertexData(attribute) && "interleaved not supported" );
            Buffer::DataReadLock lock = vash1->getVertexData(attribute);
            unsigned int stride = vash1->getStrideOfVertexData(attribute);
            // append data
            vash0->setVertexData( attribute, ~0, size, vash1->getTypeOfVertexData(attribute)
                                , lock.getPtr(), stride, vash1->getNumberOfVertexData(attribute) );
            setTreeModified();
          }
        }
      }

      unsigned int CombineTraverser::combine( const PrimitiveSharedPtr & p0, vector<unsigned int> & combinedIndices
                                            , unsigned int indexOffset, const PrimitiveSharedPtr & p1 )
      {
        DP_ASSERT( m_combineTargets & Target::GEONODE );
        DP_ASSERT( p0 != p1 );

        DP_ASSERT( p0->getPrimitiveType() == p1->getPrimitiveType() );

        unsigned int offset = 0;
        if ( p0->getVertexAttributeSet() != p1->getVertexAttributeSet() )
        {
          // in case of different VertexAttributeSets, offset the indices of p1 by the number of vertices currently in p0 (before combining)
          offset = p0->getVertexAttributeSet()->getNumberOfVertices();
          combine( p0->getVertexAttributeSet(), p1->getVertexAttributeSet() );
        }

        if ( needsPrimitiveRestartIndex( p1->getPrimitiveType() ) )
        {
          combinedIndices[indexOffset] = ~0;
          indexOffset++;
        }
        indexOffset += gatherIndices( p1, &combinedIndices[indexOffset], offset );
        return( indexOffset );
      }

      unsigned int CombineTraverser::gatherIndices( const PrimitiveSharedPtr & primitive, unsigned int * newIndices, unsigned int offset )
      {
        unsigned int elementCount = primitive->getElementCount();
        if ( primitive->isIndexed() )
        {
          unsigned int pri = primitive->getIndexSet()->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<unsigned int> isci( primitive->getIndexSet(), primitive->getElementOffset() );
          for ( unsigned int i=0 ; i<elementCount ; i++ )
          {
            newIndices[i] = ( isci[i] == pri ) ? ~0 : isci[i] + offset;
          }
        }
        else
        {
          for ( unsigned int i=0 ; i<elementCount ; i++ )
          {
            newIndices[i] = i + offset;
          }
        }
        return( elementCount );
      }

      VertexAttributeSetSharedPtr CombineTraverser::reduceVertexAttributeSet( const VertexAttributeSetSharedPtr & p
                                                                            , const vector<unsigned int> & indexMap
                                                                            , unsigned int foundIndices )
      {
        DP_ASSERT( m_combineTargets & Target::GEONODE );

        // copy from/to indices to plain vectors
        vector<unsigned int> from; from.reserve(foundIndices);
        vector<unsigned int> to;   to.reserve(foundIndices);

        for ( unsigned int i=0 ; i<indexMap.size() ; i++ )
        {
          if ( indexMap[i] != ~0 )
          {
            from.push_back( i );
            to.push_back( indexMap[i] );
          }
        }
        DP_ASSERT( ( from.size() == foundIndices ) && ( to.size() == foundIndices ) );

        VertexAttributeSetSharedPtr newVASH = VertexAttributeSet::create();
        for ( unsigned int i=0 ; i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT) ; i++ )
        {
          VertexAttributeSet::AttributeID attribute = static_cast<VertexAttributeSet::AttributeID>(i);
          if ( p->getNumberOfVertexData( attribute ) )
          {
            Buffer::DataReadLock oldData = p->getVertexData(attribute);
            if ( oldData.getPtr() )
            {
              unsigned int size = p->getSizeOfVertexData(attribute);
              dp::DataType type = p->getTypeOfVertexData(attribute);

              newVASH->setVertexData( attribute, &to[0], &from[0], size, type, oldData.getPtr(), p->getStrideOfVertexData(attribute), foundIndices );

              // inherit enable states from source attrib
              // normalize-enable state only meaningful for generic aliases!
              newVASH->setEnabled(attribute, p->isEnabled(attribute)); // conventional

              attribute = static_cast<VertexAttributeSet::AttributeID>(i+16);   // generic
              newVASH->setEnabled(attribute, p->isEnabled(attribute));
              newVASH->setNormalizeEnabled(attribute, p->isNormalizeEnabled(attribute));
            }
          }
        }
        return( newVASH );
      }

      VertexAttributeSetSharedPtr CombineTraverser::reduceVertexAttributeSet( const VertexAttributeSetSharedPtr & p
                                                                            , unsigned int offset, unsigned int count )
      {
        DP_ASSERT( m_combineTargets & Target::GEONODE );

        VertexAttributeSetSharedPtr newVASH = VertexAttributeSet::create();
        for ( unsigned int i=0 ; i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT) ; i++ )
        {
          VertexAttributeSet::AttributeID attribute = static_cast<VertexAttributeSet::AttributeID>(i);
          if ( p->getNumberOfVertexData( attribute ) )
          {
            unsigned int size = p->getVertexAttribute( attribute ).getVertexDataBytes();
            if ( size )
            {
              Buffer::DataReadLock oldData( p->getVertexBuffer(attribute), offset * size, count * size );
              DP_ASSERT( oldData.getPtr() );
              newVASH->setVertexData( attribute, p->getSizeOfVertexData( attribute ), p->getTypeOfVertexData( attribute )
                                   , oldData.getPtr(), p->getStrideOfVertexData( attribute ), count );

              // inherit enable states from source attrib
              // normalize-enable state only meaningful for generic aliases!
              newVASH->setEnabled(attribute, p->isEnabled(attribute)); // conventional

              attribute = static_cast<VertexAttributeSet::AttributeID>(i+16);   // generic
              newVASH->setEnabled(attribute, p->isEnabled(attribute));
              newVASH->setNormalizeEnabled(attribute, p->isNormalizeEnabled(attribute));
            }
          }
        }
        return( newVASH );
      }

      unsigned int CombineTraverser::reduceVertexAttributeSet( const PrimitiveSharedPtr & p )
      {
        DP_ASSERT( m_combineTargets & Target::GEONODE );

        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();
        unsigned int nov = p->getVertexAttributeSet()->getNumberOfVertices();

        if ( p->isIndexed() )
        {
          //  determine the map from old indices to new indices (using only parts of the VertexAttributeSet)
          unsigned int pri = p->getIndexSet()->getPrimitiveRestartIndex();
          vector<unsigned int> indexMap( nov, ~0 );  // map from old indices to new ones
          unsigned int foundIndices = 0;
          {
            IndexSet::ConstIterator<unsigned int> indices( p->getIndexSet(), offset );
            for ( unsigned int i=0 ; i<count ; ++i )
            {
              unsigned int idx = indices[i];
              DP_ASSERT( ( idx == pri ) || ( idx < nov ) );
              if ( ( idx != pri ) && ( indexMap[idx] == ~0 ) )
              {
                indexMap[idx] = foundIndices++;
              }
            }
          }

          //  we only have to reduce the data, if there are less data used than available
          if ( foundIndices < nov )
          {
            VertexAttributeSetSharedPtr newVASH = reduceVertexAttributeSet( p->getVertexAttributeSet(), indexMap, foundIndices );

            vector<unsigned int> newIndices( count );
            {
              IndexSet::ConstIterator<unsigned int> indices( p->getIndexSet(), offset );
              for ( unsigned int i=0 ; i<count ; i++ )
              {
                unsigned int idx = indices[i];
                newIndices[i] = ( idx == pri ) ? pri : indexMap[idx];
              }
            }

            IndexSetSharedPtr newIndexSet( IndexSet::create() );
            newIndexSet->setData( &newIndices[0], dp::checked_cast<unsigned int>(newIndices.size()) );
            newIndexSet->setPrimitiveRestartIndex( pri );

            p->setVertexAttributeSet( newVASH ); // this replaces the former VAS
            p->setIndexSet( newIndexSet );
            p->setElementRange( 0, ~0 );
            setTreeModified();
          }
        }
        else if ( ( offset != 0 ) || ( count  != nov ) )
        {
          // no IndexSet, but not the complete VAS is to be used
          VertexAttributeSetSharedPtr newVASH = reduceVertexAttributeSet( p->getVertexAttributeSet(), offset, count );
          p->setVertexAttributeSet( newVASH );
          p->setElementRange( 0, ~0 );
          setTreeModified();
        }
        return( p->getVertexAttributeSet()->getNumberOfVertices() );
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
