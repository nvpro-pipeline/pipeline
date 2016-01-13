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
#include <dp/math/math.h>
#include <dp/sg/core/BufferHost.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/algorithm/DestrippingTraverser.h>
#include <dp/sg/algorithm/SmoothTraverser.h>

using namespace dp::math;
using namespace dp::sg::core;

using std::vector;
using std::pair;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      DEFINE_STATIC_PROPERTY( SmoothTraverser, CreaseAngle );

      BEGIN_REFLECTION_INFO( SmoothTraverser )
        DERIVE_STATIC_PROPERTIES( SmoothTraverser, ExclusiveTraverser );
        INIT_STATIC_PROPERTY_RW(SmoothTraverser, CreaseAngle, float, Semantic::VALUE, value, value );
      END_REFLECTION_INFO

      SmoothTraverser::SmoothTraverser(void)
      : m_creaseAngle(PI_QUARTER)
      , m_destrippingTraverser( new DestrippingTraverser )
      {
      }

      SmoothTraverser::~SmoothTraverser(void)
      {
      }

      void SmoothTraverser::setCreaseAngle( float creaseAngle )
      {
        m_creaseAngle = creaseAngle;
      }

      float SmoothTraverser::getCreaseAngle( ) const
      {
        return m_creaseAngle;
      }


      void SmoothTraverser::doApply( const NodeSharedPtr & root )
      {
        m_destrippingTraverser->apply( root );   //  perform a destripping operation first!

        ExclusiveTraverser::doApply( root );
      }

      bool SmoothTraverser::optimizationAllowed( PrimitiveSharedPtr const& p )
      {
        DP_ASSERT( ( p != NULL ) && ( p->getVertexAttributeSet() != NULL ) );
        return(   ( p->getHints( Object::DP_SG_HINT_DYNAMIC ) == 0 )
              &&  ( p->getVertexAttributeSet()->getHints( Object::DP_SG_HINT_DYNAMIC ) == 0 ) );
      }

      inline void buildBuffers( const VertexAttributeSetSharedPtr & vash, 
                                std::vector< Vec3f > & vertices, std::vector< Vec3f > & normals, 
                                size_t & index )
      {
        unsigned int nov = vash->getNumberOfVertices();
        Buffer::ConstIterator<Vec3f>::Type itVertices = vash->getVertices();
        vertices.insert( vertices.begin() + index, itVertices, itVertices + nov );

        Buffer::ConstIterator<Vec3f>::Type itNormals = vash->getNormals();
        normals.insert( normals.begin() + index, itNormals, itNormals + nov );
        index += nov;
      }

      void  SmoothTraverser::handleGeoNode( GeoNode *p )
      {
        //  traverse the geometries and store the supported GeoSets
        ExclusiveTraverser::handleGeoNode( p );

        //  get the number of vertices in the vector of supported GeoSets
        unsigned int numVertices = 0;
        for ( size_t i=0 ; i<m_primitives.size() ; i++ )
        {
          DP_ASSERT( optimizationAllowed( m_primitives[i] ) );
          numVertices += m_primitives[i]->getVertexAttributeSet()->getNumberOfVertices();
        }

        if ( numVertices )
        {
          //  get the vertices and face normals (one per vertex) of the GeoSets
          vector<Vec3f> vertices, normals( numVertices );    
          vertices.reserve( numVertices );
          size_t index = 0;
          for ( size_t i=0; i<m_primitives.size() ; i++ )
          {
            DP_ASSERT( optimizationAllowed( m_primitives[i] ) );
            buildBuffers(m_primitives[i]->getVertexAttributeSet(), vertices, normals, index);
          }

          DP_ASSERT( isValid( p->getBoundingSphere() ) );
          smoothNormals( vertices, p->getBoundingSphere(), m_creaseAngle, normals );

          index = 0;
          for ( size_t i=0; i<m_primitives.size() ; i++ )
          {
            DP_ASSERT( optimizationAllowed( m_primitives[i] ) );
            VertexAttributeSetSharedPtr const& vas = m_primitives[i]->getVertexAttributeSet();

            unsigned int size = vas->getNumberOfVertices();
            vas->setNormals( &normals[index], size );
            index += size;
          }
          setTreeModified();
        }

        //  finally clear the primitiveSets vector again
        m_primitives.clear();
      }

      void SmoothTraverser::handlePrimitive( Primitive *p )
      {
        // we only work for certain types at the moment
        switch( p->getPrimitiveType() )
        {
          case PrimitiveType::QUADS:
          case PrimitiveType::TRIANGLES:
          case PrimitiveType::TRIANGLES_ADJACENCY:
          case PrimitiveType::PATCHES:
            flattenPrimitive( p );
            break;
          case PrimitiveType::POINTS:
          case PrimitiveType::LINE_STRIP:
          case PrimitiveType::LINE_LOOP:
          case PrimitiveType::LINES:
          case PrimitiveType::TRIANGLE_STRIP:
          case PrimitiveType::TRIANGLE_FAN:
          case PrimitiveType::QUAD_STRIP:
          case PrimitiveType::POLYGON:
          case PrimitiveType::TRIANGLE_STRIP_ADJACENCY:
          case PrimitiveType::LINES_ADJACENCY:
          case PrimitiveType::LINE_STRIP_ADJACENCY:
            break;                  // no support of other primitive types (yet)
          default:
            DP_ASSERT( false );
            break;
        }
      }

      template<typename T>
      bool checkFlat( const T * indices, unsigned int count )
      {
        std::set<T>  tmpIndexSet;
        for ( unsigned int i=0 ; i<count ; i++ )
        {
          pair<typename std::set<T>::iterator,bool> p = tmpIndexSet.insert( indices[i] );
          if ( ! p.second )
          {
            return( false );
          }
        }
        return( true );
      }

      template<typename T>
      bool checkTrivial( const T * indices, unsigned int count )
      {
        if ( count <= std::numeric_limits<T>::max() )
        {
          for ( unsigned int i=0 ; i<count ; i++ )
          {
            if ( indices[i] != i )
            {
              return( false );
            }
          }
          return( true );
        }
        return( false );
      }

      template<typename T>
      void deIndexPrimitiveT( Primitive * p )
      {
        Buffer::DataReadLock indexBuffer( p->getIndexSet()->getBuffer() );

        VertexAttributeSetSharedPtr const& vas = p->getVertexAttributeSet();
        unsigned int nov = vas->getNumberOfVertices();
        unsigned int noi = p->getIndexSet()->getNumberOfIndices();
        if (    ( nov != noi )
            ||  ! checkFlat( (const T *)indexBuffer.getPtr(), noi )
            ||  ! checkTrivial( (const T *)indexBuffer.getPtr(), noi ) )
        {
          VertexAttributeSetSharedPtr newVASH = VertexAttributeSet::create();
          for ( unsigned int i=0 ; i<static_cast<unsigned int>(VertexAttributeSet::AttributeID::VERTEX_ATTRIB_COUNT) ; i++ )
          {
            VertexAttributeSet::AttributeID id = static_cast<VertexAttributeSet::AttributeID>(i);
            if ( vas->getSizeOfVertexData(id) )
            {
              Buffer::DataReadLock indexBufferLock( p->getIndexSet()->getBuffer() );
              const T * idxPtr = indexBufferLock.getPtr<T>();
              dp::DataType type = vas->getTypeOfVertexData(id);
              unsigned int size = dp::checked_cast<unsigned int>(vas->getSizeOfVertexData(id) * dp::getSizeOf( type ));
              unsigned int stride = vas->getStrideOfVertexData(id);
              DP_ASSERT( size <= stride );
              unsigned int numIndices = p->getIndexSet()->getNumberOfIndices();

              BufferHostSharedPtr newVertexBuffer = BufferHost::create();
              newVertexBuffer->setSize( numIndices * size );
              {
                Buffer::DataReadLock oldBufferLock( vas->getVertexData( id ) );
                const unsigned char * oldBuffPtr = oldBufferLock.getPtr<unsigned char>();

                Buffer::DataWriteLock newBufferLock( newVertexBuffer, Buffer::MapMode::WRITE );
                unsigned char * newBuffPtr = newBufferLock.getPtr<unsigned char>();

                for ( unsigned int idx=0 ; idx < numIndices ; idx++ )
                {
                  memcpy( newBuffPtr, oldBuffPtr + stride * idxPtr[idx], size );
                  newBuffPtr += size;
                }
              }

              newVASH->setVertexData( id, vas->getSizeOfVertexData(id), type, newVertexBuffer, 0, stride, numIndices );

              // inherit enable states from source attrib
              // normalize-enable state only meaningful for generic aliases!
              newVASH->setEnabled(id, vas->isEnabled(id)); // conventional

              id = static_cast<VertexAttributeSet::AttributeID>(i+16);    // generic
              newVASH->setEnabled(id, vas->isEnabled(id));
              newVASH->setNormalizeEnabled(id, vas->isNormalizeEnabled(id));
            }
          }
          p->setVertexAttributeSet( newVASH );
        }
      }

      void  SmoothTraverser::flattenPrimitive( Primitive *p )
      {
        if ( optimizationAllowed( p->getSharedPtr<Primitive>() ) )
        {
          if ( p->isIndexed() )
          {
            switch( p->getIndexSet()->getIndexDataType() )
            {
              case dp::DataType::UNSIGNED_INT_8 :
                deIndexPrimitiveT<unsigned char>( p );
                break;
              case dp::DataType::UNSIGNED_INT_16 :
                deIndexPrimitiveT<unsigned short>( p );
                break;
              case dp::DataType::UNSIGNED_INT_32 :
                deIndexPrimitiveT<unsigned int>( p );
                break;
              default :
                DP_ASSERT( false );
                break;
            }
            p->setIndexSet( IndexSetSharedPtr::null );
            DP_ASSERT( !p->isIndexed() );
          }

          p->generateNormals();
          m_primitives.push_back( p->getSharedPtr<Primitive>() );
          setTreeModified();
        }
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
