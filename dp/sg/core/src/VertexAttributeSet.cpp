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


#include <dp/math/math.h>
#include <dp/util/Memory.h>
#include <dp/sg/core/VertexAttributeSet.h>
#include <dp/sg/core/BufferHost.h>
#include <dp/sg/core/OwnedObject.hpp>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      static void generateTexCoordsCylinder( VertexAttributeSetSharedPtr const& vas, const Sphere3f &sphere, unsigned int tu );
      static void generateTexCoordsPlane( VertexAttributeSetSharedPtr const& vas, const Sphere3f &sphere, unsigned int tu );
      static void generateTexCoordsSphere( VertexAttributeSetSharedPtr const& vas, const Sphere3f &sphere, unsigned int tu );
      static void getSortedAxes( Buffer::ConstIterator<Vec3f>::Type vertices, unsigned int numberOfVertices, unsigned int & u, unsigned int & v, unsigned int & w );

      VertexAttribute VertexAttributeSet::m_emptyAttribute;

      BEGIN_REFLECTION_INFO( VertexAttributeSet )
        DERIVE_STATIC_PROPERTIES( VertexAttributeSet, Object );
      END_REFLECTION_INFO

      VertexAttributeSetSharedPtr VertexAttributeSet::create()
      {
        return( std::shared_ptr<VertexAttributeSet>( new VertexAttributeSet() ) );
      }

      HandledObjectSharedPtr VertexAttributeSet::clone() const
      {
        return( std::shared_ptr<VertexAttributeSet>( new VertexAttributeSet( *this ) ) );
      }

      VertexAttributeSet::VertexAttributeSet()
      : OwnedObject<Object>()
      , m_enableFlags(0)
      , m_normalizeEnableFlags(0)
      {
        m_bufferObserver.setVertexAttributeSet( this );
        m_objectCode = OC_VERTEX_ATTRIBUTE_SET;
      }

      VertexAttributeSet::VertexAttributeSet(const VertexAttributeSet& rhs)
      : OwnedObject<Object>(rhs)
      , m_enableFlags(rhs.m_enableFlags)
      , m_normalizeEnableFlags(rhs.m_normalizeEnableFlags)
      , m_vattribs(rhs.m_vattribs)
      {
        m_bufferObserver.setVertexAttributeSet( this );
        m_objectCode = OC_VERTEX_ATTRIBUTE_SET;
      }

      VertexAttributeSet::~VertexAttributeSet()
      {
        // detach position buffer
        unsubscribeBuffer( m_vattribs.begin() );
      }


      void VertexAttributeSet::swapVertexData(unsigned int attrib, VertexAttribute& rhs)
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) ); // undefined behavior on invalid attrib

        // we alias generic and conventional vertex attributes, that is - 
        // pairs of generic and conventional vertex attributes are sharing the same storage
        unsigned int attrIndex = attribIndex(attrib);

        AttributeContainer::iterator it = m_vattribs.find( attrIndex );
        if ( it == m_vattribs.end() )
        {
          std::pair<AttributeContainer::iterator,bool> pacib = m_vattribs.insert( std::make_pair( attrIndex, VertexAttribute() ) );
          DP_ASSERT( pacib.second );
          it = pacib.first;
        }
        else
        {
          unsubscribeBuffer( it );
        }
        it->second.swapData( rhs );
        if ( it->second.getBuffer() )
        {
          subscribeBuffer( it );
        }
        else
        {
          m_vattribs.erase( it );
        }

        if ( attrib == NVSG_POSITION )
        {
          markDirty( NVSG_BOUNDING_VOLUMES );
        }
        notify( Event(this ) );
      }

      void VertexAttributeSet::reserveVertexData( unsigned int attrib, unsigned int size
                                                , dp::util::DataType type, unsigned int count )
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) );
        DP_ASSERT( ( dp::util::DT_INT_8 <= type ) && ( type <= dp::util::DT_NUM_DATATYPES ) );

        // we alias generic and conventional vertex attributes, that is - 
        // pairs of generic and conventional vertex attributes are sharing the same storage
        unsigned int attrIndex = attribIndex(attrib);

        // if data is appended, it must be of same size and type as previously specified data!
        // assert this!
        AttributeContainer::iterator it = m_vattribs.find( attrIndex );
        if ( it == m_vattribs.end() )
        {
          std::pair<AttributeContainer::iterator,bool> pacib = m_vattribs.insert( std::make_pair( attrIndex, VertexAttribute() ) );
          DP_ASSERT( pacib.second );
          it = pacib.first;
        }
        else
        {
          unsubscribeBuffer( it );
        }
        DP_ASSERT(    !it->second.getBuffer()
                    ||  (   ( it->second.getVertexDataSize() == size )
                        &&  ( it->second.getVertexDataType() == type ) ) );
        it->second.reserveData( size, type, count );
        subscribeBuffer( it );
      }

      void VertexAttributeSet::setVertexData( unsigned int attrib, unsigned int size, dp::util::DataType type
                                            , const void * data, unsigned int stride, unsigned int count
                                            , bool enable )
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) );
        DP_ASSERT( ( dp::util::DT_INT_8 <= type ) && ( type <= dp::util::DT_NUM_DATATYPES ) );
        DP_ASSERT( data );

        // we alias generic and conventional vertex attributes, that is - 
        // pairs of generic and conventional vertex attributes are sharing the same storage
        unsigned int attrIndex = attribIndex(attrib);

        AttributeContainer::iterator it = m_vattribs.find( attrIndex );
        if ( it == m_vattribs.end() )
        {
          std::pair<AttributeContainer::iterator,bool> pacib = m_vattribs.insert( std::make_pair( attrIndex, VertexAttribute() ) );
          DP_ASSERT( pacib.second );
          it = pacib.first;
        }
        else
        {
          unsubscribeBuffer( it );
        }
        it->second.setData( size, type, data, stride, count );
        subscribeBuffer( it );
        setEnabled( attrib, enable );

        if ( NVSG_POSITION == attrIndex )
        {
          markDirty( NVSG_BOUNDING_VOLUMES );
        }
        notify( Event(this ) );
      }

      void VertexAttributeSet::setVertexData( unsigned int attrib, unsigned int pos, unsigned int size
                                            , dp::util::DataType type, const void * data, unsigned int stride
                                            , unsigned int count, bool enable )
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) );
        DP_ASSERT( ( dp::util::DT_INT_8 <= type ) && ( type <= dp::util::DT_NUM_DATATYPES ) );
        DP_ASSERT( data );

        // we alias generic and conventional vertex attributes, that is - 
        // pairs of generic and conventional vertex attributes are sharing the same storage
        unsigned int attrIndex = attribIndex(attrib);

        // if data is appended, it must be of same size and type as previously specified data!
        // assert this!
        AttributeContainer::iterator it = m_vattribs.find( attrIndex );
        if ( it == m_vattribs.end() )
        {
          std::pair<AttributeContainer::iterator,bool> pacib = m_vattribs.insert( std::make_pair( attrIndex, VertexAttribute() ) );
          DP_ASSERT( pacib.second );
          it = pacib.first;
        }
        else
        {
          unsubscribeBuffer( it );
        }
        DP_ASSERT(    !it->second.getBuffer()
                    ||  (   ( it->second.getVertexDataSize() == size )
                        &&  ( it->second.getVertexDataType() == type ) ) );

        // re-specify data starting at 'pos'
        it->second.setData( pos, size, type, data, stride, count );
        subscribeBuffer( it );
        setEnabled( attrib, enable );

        if ( attrIndex == NVSG_POSITION )
        {
          markDirty( NVSG_BOUNDING_VOLUMES );
        }
        notify( Event(this ) );
      }

      void VertexAttributeSet::setVertexData(  unsigned int attrib, unsigned int size, dp::util::DataType type
                                             , const BufferSharedPtr &buffer, unsigned int offset
                                             , unsigned int strideInBytes, unsigned int count
                                             , bool enable )
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) );
        DP_ASSERT( ( dp::util::DT_INT_8 <= type ) && ( type <= dp::util::DT_NUM_DATATYPES ) );
        DP_ASSERT( buffer );

        // we alias generic and conventional vertex attributes, that is - 
        // pairs of generic and conventional vertex attributes are sharing the same storage
        unsigned int attrIndex = attribIndex(attrib);

        // re-specify data starting at 'pos'
        AttributeContainer::iterator it = m_vattribs.find( attrIndex );
        if ( it == m_vattribs.end() )
        {
          std::pair<AttributeContainer::iterator,bool> pacib = m_vattribs.insert( std::make_pair( attrIndex, VertexAttribute() ) );
          DP_ASSERT( pacib.second );
          it = pacib.first;
        }
        else
        {
          unsubscribeBuffer( it );
        }
        it->second.setData( size, type, buffer, offset, strideInBytes, count );
        subscribeBuffer( it );
        setEnabled( attrib, enable );

        if ( NVSG_POSITION == attrIndex )
        {
          markDirty( NVSG_BOUNDING_VOLUMES );
        }
        notify( Event(this ) );
      }


      // FIXME This will run extremly slow due to a lot of setData calls.
      void VertexAttributeSet::setVertexData( unsigned int attrib, const unsigned int * to
                                            , const unsigned int * from, unsigned int size
                                            , dp::util::DataType type, const void * data, unsigned int stride
                                            , unsigned int count, bool enable )
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) );
        DP_ASSERT( from );
        DP_ASSERT( ( dp::util::DT_INT_8 <= type ) && ( type <= dp::util::DT_NUM_DATATYPES ) );
        DP_ASSERT( data );
  
        // we alias generic and conventional vertex attributes, that is - 
        // pairs of generic and conventional vertex attributes are sharing the same storage
        unsigned int attrIndex = attribIndex(attrib);

        AttributeContainer::iterator it = m_vattribs.find( attrIndex );
        if ( it == m_vattribs.end() )
        {
          std::pair<AttributeContainer::iterator,bool> pacib = m_vattribs.insert( std::make_pair( attrIndex, VertexAttribute() ) );
          DP_ASSERT( pacib.second );
          it = pacib.first;
        }
        else
        {
          unsubscribeBuffer( it );
        }
        // if data is appended, it must be of same size and type as previously specified data!
        // assert this!
        DP_ASSERT(    !it->second.getBuffer()
                    ||  (   ( it->second.getVertexDataSize() == size )
                        &&  ( it->second.getVertexDataType() == type ) ) );

        it->second.reserveData( size, type, count );

        VertexAttribute::memptr_t vdata = (VertexAttribute::memptr_t)data;
        if ( !to  ) // to == NULL --> append
        {
          unsigned int pos = it->second.getVertexDataCount();

          // reference input vertex data through 'from' index array
          // -- requires to copy vertex by vertex
          unsigned int vdb = stride ? stride : it->second.getVertexDataBytes();    // call after (!) reserveData
          for ( unsigned int i=0; i<count; ++i )
          {
            // we need the byte-offset to dereference source data
            unsigned int offs = from[i] * vdb;
            it->second.setData(pos+i, size, type, vdata+offs, 0, 1);
          }
        }
        else
        {
          // now copy vertex by vertex
          unsigned int vdb = it->second.getVertexDataBytes();    // call after (!) reserveData
          for ( unsigned int i=0; i<count; ++i )
          {
            DP_ASSERT(to[i] < count);
            // from[i] can actually exceed count!

            // we need the byte-offset to dereference source data
            unsigned int offs = from[i] * vdb;
            it->second.setData(to[i], size, type, vdata+offs, 0, 1);
          }
        }
        subscribeBuffer( it );
        setEnabled( attrib, enable );

        if ( NVSG_POSITION == attrIndex )
        {
          markDirty( NVSG_BOUNDING_VOLUMES );
        }
        notify( Event( this ) );
      }

      void VertexAttributeSet::removeVertexData( unsigned int attrib, bool disable )
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) ); // undefined behavior on invalid attrib

        // we alias generic and conventional vertex attributes, that is - 
        // pairs of generic and conventional vertex attributes are sharing the same storage
        unsigned int attrIndex = attribIndex(attrib); 

        AttributeContainer::iterator it = m_vattribs.find( attrIndex );
        if ( it != m_vattribs.end() )
        {
          unsubscribeBuffer( it );
          DP_ASSERT( it->second.getVertexDataCount() );   // nothing to do if no data was specified before
          it->second.removeData();
          m_vattribs.erase( it );
          if ( NVSG_POSITION == attrIndex )
          {
            markDirty( NVSG_BOUNDING_VOLUMES );
          }
          notify( Event( this ) );
          setEnabled( attrib, !disable );
        }
      }

      Buffer::DataReadLock VertexAttributeSet::getVertexData(unsigned int attrib) const
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) ); // undefined behavior on invalid attrib

        // we alias generic and conventional vertex attributes, that is - 
        // pairs of generic and conventional vertex attributes are sharing the same storage
        unsigned int attrIndex = attribIndex(attrib); 
        AttributeContainer::const_iterator it = m_vattribs.find( attrIndex );
        DP_ASSERT( it != m_vattribs.end() );

        const BufferSharedPtr & buffer = it->second.getBuffer();
        DP_ASSERT( buffer );
        size_t length = it->second.getVertexDataCount() * it->second.getVertexDataStrideInBytes()
                      - it->second.getVertexDataOffsetInBytes() % it->second.getVertexDataStrideInBytes();
        return( Buffer::DataReadLock( buffer, it->second.getVertexDataOffsetInBytes(), length ) );
      }

      const BufferSharedPtr & VertexAttributeSet::getVertexBuffer(unsigned int attrib) const
      {
        unsigned int attrIndex = attribIndex(attrib); 
        AttributeContainer::const_iterator it = m_vattribs.find( attrIndex );
        DP_ASSERT( it != m_vattribs.end() );
        return( it->second.getBuffer() );
      }

      unsigned int VertexAttributeSet::getSizeOfVertexData(unsigned int attrib) const
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) ); // undefined behavior on invalid attrib
        AttributeContainer::const_iterator it = m_vattribs.find( attrib );
        return( ( it == m_vattribs.end() ) ? 0 : it->second.getVertexDataSize() );
      }

      dp::util::DataType VertexAttributeSet::getTypeOfVertexData(unsigned int attrib) const
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) ); // undefined behavior on invalid attrib
        AttributeContainer::const_iterator it = m_vattribs.find( attrib );
        return( ( it == m_vattribs.end() ) ? dp::util::DT_UNKNOWN : it->second.getVertexDataType() );
      }

      unsigned int VertexAttributeSet::getNumberOfVertexData(unsigned int attrib) const
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) ); // undefined behavior on invalid attrib
        AttributeContainer::const_iterator it = m_vattribs.find( attrib );
        return( ( it == m_vattribs.end() ) ? 0 : it->second.getVertexDataCount() );
      }

      unsigned int VertexAttributeSet::getStrideOfVertexData(unsigned int attrib) const
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) ); // undefined behavior on invalid attrib
        AttributeContainer::const_iterator it = m_vattribs.find( attrib );
        return( ( it == m_vattribs.end() ) ? 0 : it->second.getVertexDataStrideInBytes() );
      }

      unsigned int VertexAttributeSet::getOffsetOfVertexData(unsigned int attrib) const
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) ); // undefined behavior on invalid attrib
        AttributeContainer::const_iterator it = m_vattribs.find( attrib );
        return( ( it == m_vattribs.end() ) ? 0 : it->second.getVertexDataOffsetInBytes() );
      }

      bool VertexAttributeSet::isContiguousVertexData(unsigned int attrib) const
      {
        DP_ASSERT( ( NVSG_POSITION <= attrib ) && ( attrib <= NVSG_ATTR15 ) ); // undefined behavior on invalid attrib
        AttributeContainer::const_iterator it = m_vattribs.find( attrib );
        return( ( it == m_vattribs.end() ) ? false : it->second.isContiguous() );
      }

      VertexAttributeSet & VertexAttributeSet::operator=(const VertexAttributeSet & rhs)
      {
        if (&rhs != this)
        {
          OwnedObject<Object>::operator=(rhs);
          m_enableFlags = rhs.m_enableFlags;
          m_normalizeEnableFlags = rhs.m_normalizeEnableFlags;
          for ( AttributeContainer::iterator it = m_vattribs.begin() ; it != m_vattribs.end() ; ++it )
          {
            unsubscribeBuffer( it );
            it->second.removeData();
          }
          m_vattribs.clear();
          for ( AttributeContainer::const_iterator it = rhs.m_vattribs.begin() ; it != rhs.m_vattribs.end() ; ++it )
          {
            m_vattribs[it->first] = it->second;
            subscribeBuffer( it );
          }
        }
        return *this;
      }

      void copy( VertexAttributeSetSharedPtr const& src, VertexAttributeSetSharedPtr const& dst )
      {
        if ( src != dst )
        {
          dst->setName( src->getName() );
          dst->setAnnotation( src->getAnnotation() );
          dst->setHints( src->getHints() );
          dst->setUserData( src->getUserData() );

          for ( unsigned int i=0 ; i<2*VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT ; ++i )
          {
            if ( src->getSizeOfVertexData( i ) )
            {
              DP_ASSERT( ( src->getOffsetOfVertexData( i ) == 0 ) && src->isContiguousVertexData( i ) );
              dst->setVertexData( i, src->getSizeOfVertexData( i ), src->getTypeOfVertexData( i )
                                , src->getVertexBuffer( i ), src->getOffsetOfVertexData( i )
                                , src->getStrideOfVertexData( i ), src->getNumberOfVertexData( i ) );
              dst->setEnabled( i, src->isEnabled( i ) );
              if ( i >= VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT )
              {
                dst->setNormalizeEnabled( i, src->isNormalizeEnabled( i ) );
              }
            }
          }
        }
      }


      void generateTexCoords( VertexAttributeSetSharedPtr const& vas, TextureCoordType tct, const Sphere3f &sphere, unsigned int tc )
      {
        DP_ASSERT( vas );
        DP_ASSERT(VertexAttributeSet::NVSG_TEXCOORD0 <= tc && tc <= VertexAttributeSet::NVSG_TEXCOORD7);

        switch( tct )
        {
          case TCT_CYLINDRICAL :
            generateTexCoordsCylinder( vas, sphere, tc-VertexAttributeSet::NVSG_TEXCOORD0 ); // convert 'tc' to zero based unit
            break;
          case TCT_PLANAR :
            generateTexCoordsPlane( vas, sphere, tc-VertexAttributeSet::NVSG_TEXCOORD0 ); // convert 'tc' to zero based unit
            break;
          case TCT_SPHERICAL :
            generateTexCoordsSphere( vas, sphere, tc-VertexAttributeSet::NVSG_TEXCOORD0 ); // convert 'tc' to zero based unit
            break;
          default :
            DP_ASSERT( false );
            break;
        }
      }

      void generateTexCoordsSphere( VertexAttributeSetSharedPtr const& vas, const Sphere3f &sphere, unsigned int tu )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = vas->getVertices();
        std::vector<Vec2f> texCoords( vas->getNumberOfVertices() );
        Vec2f tp;

        for ( unsigned int i=0 ; i<vas->getNumberOfVertices() ; i++ )
        {
          Vec3f p = vertices[i] - sphere.getCenter();
          if ( fabsf( p[1] ) > FLT_EPSILON )
          {
            tp[0] = 0.5f * atan2f( p[0], -p[1] ) / (float) PI + 0.5f;
          }
          else
          {
            tp[0] = (float)( ( p[0] >= 0.0f ) ? 0.75f : 0.25f );
          }

          float d = sqrtf( square( p[0] ) + square( p[1] ) );
          if ( d > FLT_EPSILON )
          {
            tp[1] = atan2f( p[2], d ) / (float) PI + 0.5f;
          }
          else
          {
            tp[1] = (float)( ( p[2] >= 0.0f ) ? 1.0f : 0.0f );
          }

          texCoords[i] = tp;
        }

        vas->setTexCoords( tu, &texCoords[0], vas->getNumberOfVertices() );
      }

      void  generateTexCoordsCylinder( VertexAttributeSetSharedPtr const& vas, const Sphere3f &sphere, unsigned int tu )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = vas->getVertices();

        // get the bounding box of vas and determine the axes u,v to use to texture along
        unsigned int u, v, w;
        getSortedAxes( vertices, vas->getNumberOfVertices(), u, v, w );

        // create the texture coordinates
        std::vector<Vec2f> texCoords( vas->getNumberOfVertices() );
        Vec2f tp;
        float oneOverTwoR = 0.5f / sphere.getRadius();
        float tpuMin = 1.0f;
        float tpuMax = 0.0f;
        for ( unsigned int i=0 ; i<vas->getNumberOfVertices() ; i++ )
        {
          Vec3f p = vertices[i] - sphere.getCenter();

          // largest axis as the cylinder axis
          tp[0] = p[u] * oneOverTwoR + 0.5f;   //  0.0 <= tp[1] <= 1.0
          if ( tp[0] < tpuMin )
          {
            tpuMin = tp[0];
          }
          else if ( tpuMax < tp[0] )
          {
            tpuMax = tp[0];
          }

          if ( FLT_EPSILON < fabsf( p[v] ) )
          {
            tp[1] = 0.5f * atan2f( p[w], -p[v] ) / (float) PI + 0.5f;
          }
          else
          {
            tp[1] = (float)( ( 0.0f <= p[w] ) ? 0.75f : 0.25f );
          }

          texCoords[i] = tp;
        }

        // scale the texcoords to [0..1]
        DP_ASSERT( FLT_EPSILON < ( tpuMax - tpuMin ) );
        float tpuScale = 1.0f / ( tpuMax - tpuMin );
        for ( unsigned int i=0 ; i<vas->getNumberOfVertices() ; i++ )
        {
          texCoords[i][0] = ( texCoords[i][0] - tpuMin ) * tpuScale;
        }
        vas->setTexCoords( tu, &texCoords[0], vas->getNumberOfVertices() );
      }

      void  generateTexCoordsPlane( VertexAttributeSetSharedPtr const& vas, const Sphere3f &sphere, unsigned int tu )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = vas->getVertices();

        // get the bounding box of vas and determine the axes u,v to use to texture along
        unsigned int u, v, w;
        getSortedAxes( vertices, vas->getNumberOfVertices(), u, v, w );

        // create the texture coordinates
        std::vector<Vec2f> texCoords( vas->getNumberOfVertices() );
        Vec2f tp;
        float oneOverTwoR = 0.5f / sphere.getRadius();
        Vec2f tpMin( 1.0f, 1.0f );
        Vec2f tpMax( 0.0f, 0.0f );
        for ( unsigned int i=0 ; i<vas->getNumberOfVertices() ; i++ )
        {
          Vec3f p = vertices[i] - sphere.getCenter();
          tp[0] = p[u] * oneOverTwoR + 0.5f;  // 0.0 <= tp[0] <= 1.0
          if ( tp[0] < tpMin[0] )
          {
            tpMin[0] = tp[0];
          }
          else if ( tp[0] > tpMax[0] )
          {
            tpMax[0] = tp[0];
          }

          tp[1] = p[v] * oneOverTwoR + 0.5f;   //  0.0 <= tp[1] <= 1.0
          if ( tp[1] < tpMin[1] )
          {
            tpMin[1] = tp[1];
          }
          else if ( tp[1] > tpMax[1] )
          {
            tpMax[1] = tp[1];
          }

          texCoords[i] = tp;
        }

        // scale the texcoords to [0..1]
        DP_ASSERT( ( FLT_EPSILON < ( tpMax[0] - tpMin[0] ) ) && ( FLT_EPSILON < ( tpMax[1] - tpMin[1] ) ) );
        Vec2f tpScale( 1.0f / ( tpMax[0] - tpMin[0] ), 1.0f / ( tpMax[1] - tpMin[1] ) );
        for ( unsigned int i=0 ; i<vas->getNumberOfVertices() ; i++ )
        {
          texCoords[i] = Vec2f( ( texCoords[i][0] - tpMin[0] ) * tpScale[0],
                                ( texCoords[i][1] - tpMin[1] ) * tpScale[1] );
        }

        vas->setTexCoords( tu, &texCoords[0], vas->getNumberOfVertices() );
      }

      void getSortedAxes( Buffer::ConstIterator<Vec3f>::Type vertices, unsigned int numberOfVertices, unsigned int & u, unsigned int & v, unsigned int & w )
      {
        Vec3f boxMin, boxMax, size;
        boundingBox( vertices, numberOfVertices, boxMin, boxMax );
        size = boxMax - boxMin;

        if ( size[0] <= size[1] )
        {
          if ( size[1] <= size[2] )
          {
            //  size[0] <= size[1] <= size[2]
            u = 2; v = 1; w = 0;
          }
          else if ( size[0] <= size[2] )
          {
            //  size[0] <= size[2] < size[1]
            u = 1; v = 2; w = 0;
          }
          else
          {
            //  size[2] < size[0] <= size[1]
            u = 1; v = 0; w = 2;
          }
        }
        else
        {
          if ( size[0] <= size[2] )
          {
            //  size[1] < size[0] <= size[2]
            u = 2; v = 0; w = 1;
          }
          else if ( size[1] <= size[2] )
          {
            //  size[1] <= size[2] < size[0] )
            u = 0; v = 2; w = 1;
          }
          else
          {
            //  size[2] < size[1] < size[0]
            u = 0; v = 1; w = 2;
          }
        }
        DP_ASSERT( ( FLT_EPSILON < size[u] ) && ( FLT_EPSILON < size[v] ) );
      }

      bool VertexAttributeSet::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<VertexAttributeSet>() && OwnedObject<Object>::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          VertexAttributeSetSharedPtr const& vas = object.staticCast<VertexAttributeSet>();

          equi = ( m_vattribs.size() == vas->m_vattribs.size() );
          for ( AttributeContainer::const_iterator thisit = m_vattribs.begin(), thatit = vas->m_vattribs.begin()
              ; equi && ( thisit != m_vattribs.end() ) && ( thatit != vas->m_vattribs.end() )
              ; ++thisit, ++thatit )
          {
            equi =  ( thisit->first == thatit->first )
                &&  ( thisit->second.getVertexDataCount() == thatit->second.getVertexDataCount() )
                &&  ( thisit->second.getVertexDataSize()  == thatit->second.getVertexDataSize() )
                &&  ( thisit->second.getVertexDataType()  == thatit->second.getVertexDataType() );
          }
          if ( equi )
          {
            if ( !deepCompare )
            {
              for ( AttributeContainer::const_iterator thisit = m_vattribs.begin(), thatit = vas->m_vattribs.begin()
                  ; equi && thisit != m_vattribs.end() && thatit != vas->m_vattribs.end()
                  ; ++thisit, ++thatit )
              {
                equi =  ( thisit->second.getBuffer() == thatit->second.getBuffer() )
                    &&  ( thisit->second.getVertexDataStrideInBytes() == thatit->second.getVertexDataStrideInBytes() )
                    // FIMXE nyi && ( thisit->second.getVertexDataOffsetInBytes() == thatit->second.getVertexDataOffsetInBytes() )
                    ;
              }
            }
            else
            {
              for ( AttributeContainer::const_iterator thisit = m_vattribs.begin(), thatit = vas->m_vattribs.begin()
                  ; equi && thisit != m_vattribs.end() && thatit != vas->m_vattribs.end()
                  ; ++thisit, ++thatit )
              {
                dp::util::DataType type = thisit->second.getVertexDataType();
                size_t vertexDataCount = thisit->second.getVertexDataCount();
                size_t vertexDataBytes = thisit->second.getVertexDataBytes();

                if ( type != dp::util::DT_UNKNOWN )
                {
                  if ( isIntegerType(type) )
                  {
                    Buffer::ConstIterator<char>::Type lhsData = getVertexData<char>(thisit->first);
                    Buffer::ConstIterator<char>::Type rhsData = vas->getVertexData<char>(thisit->first);

                    for ( size_t i = 0; equi && i < vertexDataCount; ++i )
                    {
                      equi = !memcmp(&lhsData[0], &rhsData[0], vertexDataBytes );
                      ++lhsData;
                      ++rhsData;
                    }
                  }
                  else
                  {
                    unsigned int numComps = thisit->second.getVertexDataCount() * thisit->second.getVertexDataSize();
                    if ( type == dp::util::DT_FLOAT_32 )
                    {
                      Buffer::ConstIterator<float>::Type lhsData = getVertexData<float>(thisit->first);
                      Buffer::ConstIterator<float>::Type rhsData = vas->getVertexData<float>(thisit->first);

                      for ( size_t i = 0; equi && i < vertexDataCount && equi; ++i )
                      {
                        const float *lhs = &lhsData[0];
                        const float *rhs = &rhsData[0];
                        for ( unsigned int j = 0; equi && j < numComps && equi ; ++j )
                        {
                          equi = ( fabsf(lhs[j] - rhs[j]) <= FLT_EPSILON );
                        }
                        ++lhsData;
                        ++rhsData;
                      }
                    }
                    else
                    {
                      DP_ASSERT(type==dp::util::DT_FLOAT_64);
                      Buffer::ConstIterator<double>::Type lhsData = getVertexData<double>(thisit->first);
                      Buffer::ConstIterator<double>::Type rhsData = vas->getVertexData<double>(thisit->first);

                      for ( size_t i = 0; equi && i < vertexDataCount && equi; ++i )
                      {
                        const double *lhs = &lhsData[0];
                        const double *rhs = &rhsData[0];
                        for ( unsigned int j = 0; equi && j < numComps && equi ; ++j )
                        {
                          equi = ( fabs(lhs[j] - rhs[j]) <= DBL_EPSILON );
                        }
                        ++lhsData;
                        ++rhsData;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        return( equi );
      }

      void VertexAttributeSet::feedHashGenerator( util::HashGenerator & hg ) const
      {
        OwnedObject<Object>::feedHashGenerator( hg );
        hg.update( reinterpret_cast<const unsigned char *>(&m_enableFlags), sizeof(m_enableFlags) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_normalizeEnableFlags), sizeof(m_normalizeEnableFlags) );
        for ( AttributeContainer::const_iterator it = m_vattribs.begin() ; it != m_vattribs.end() ; ++it )
        {
          hg.update( reinterpret_cast<const unsigned char *>(&it->first), sizeof(it->first) );
          it->second.feedHashGenerator( hg );
        }
      }

      void copySelectedVertices( const VertexAttributeSetSharedPtr & from, const VertexAttributeSetSharedPtr & to
                               , std::vector<unsigned int> & indices )
      {
        // indexMap is intended to hold vertex indices, which should not exceed 32-bit precision by definition!
        std::vector<unsigned int> indexMap( from->getNumberOfVertices(), 0xFFFFFFFF );
        std::vector<unsigned int> iFrom; iFrom.reserve(from->getNumberOfVertices());
        for ( size_t i=0 ; i<indices.size() ; i++ )
        {
          if ( indexMap[indices[i]] == 0xFFFFFFFF )
          {
            indexMap[indices[i]] = util::checked_cast<unsigned int>(iFrom.size());
            iFrom.push_back(indices[i]);
          }
        }

        for ( unsigned int slot=0 ; slot<VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT ; slot++ )
        {
          if ( from->getSizeOfVertexData( slot ) )
          {
            BufferSharedPtr oldData = from->getVertexBuffer(slot);
            Buffer::DataReadLock lock( oldData );

            unsigned int size = from->getSizeOfVertexData( slot );
            dp::util::DataType type = from->getTypeOfVertexData( slot ); 

            to->setVertexData( slot, NULL, &iFrom[0], size, type, lock.getPtr()
                                , from->getStrideOfVertexData( slot ) 
                                , util::checked_cast<unsigned int>(iFrom.size()) );

            // inherit enable states from source attrib
            // normalize-enable state only meaningful for generic aliases!
            to->setEnabled(slot, from->isEnabled(slot)); // conventional
            to->setEnabled(slot+16, from->isEnabled(slot+16)); // generic
            to->setNormalizeEnabled(slot+16, from->isNormalizeEnabled(slot+16)); // only generic
          }
        }

        for ( size_t i=0 ; i<indices.size() ; i++ )
        {
          indices[i] = indexMap[indices[i]];
        }
      }

      void copySelectedVertices( const VertexAttributeSetSharedPtr & from, const VertexAttributeSetSharedPtr & to
                               , std::vector<std::vector<unsigned int> > &indices )
      {
        // indexMap below is intended to hold vertex indices, which should not exceed 32-bit precision by definition!
        std::vector<unsigned int> indexMap( from->getNumberOfVertices(), ~0 );
        std::vector<unsigned int> iFrom; iFrom.reserve(from->getNumberOfVertices());
        for ( size_t i=0 ; i<indices.size() ; i++ )
        {
          for ( size_t j=0 ; j<indices[i].size() ; j++ )
          {
            if ( indexMap[indices[i][j]] == ~0 )
            {
              indexMap[indices[i][j]] = util::checked_cast<unsigned int>(iFrom.size());
              iFrom.push_back(indices[i][j]);
            }
          }
        }

        for ( unsigned int slot=0 ; slot<VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT ; slot++ )
        {
          if ( from->getSizeOfVertexData( slot ) )
          {
            BufferSharedPtr oldData = from->getVertexBuffer(slot);
            Buffer::DataReadLock lock( oldData );

            unsigned int size = from->getSizeOfVertexData( slot );
            dp::util::DataType type = from->getTypeOfVertexData( slot ); 

            to->setVertexData( slot, NULL, &iFrom[0], size, type, lock.getPtr()
                             , from->getStrideOfVertexData( slot )
                             , util::checked_cast<unsigned int>(iFrom.size()) );

            // inherit enable states from source attrib
            // normalize-enable state only meaningful for generic aliases!
            to->setEnabled(slot, from->isEnabled(slot)); // conventional
            to->setEnabled(slot+16, from->isEnabled(slot+16)); // generic
            to->setNormalizeEnabled(slot+16, from->isNormalizeEnabled(slot+16)); // only generic
          }
        }

        for ( size_t i=0 ; i<indices.size() ; i++ )
        {
          for ( size_t j=0 ; j<indices[i].size() ; j++ )
          {
            indices[i][j] = indexMap[indices[i][j]];
          }
        }
      }

      void VertexAttributeSet::setVertexAttribute( unsigned int attrib, const VertexAttribute &vertexAttribute, bool enable )
      {
        int attribIdx = attribIndex( attrib );
        AttributeContainer::iterator it = m_vattribs.find( attribIdx );
        if ( it == m_vattribs.end() )
        {
          std::pair<AttributeContainer::iterator,bool> pacib = m_vattribs.insert( std::make_pair( attribIdx, vertexAttribute ) );
          DP_ASSERT( pacib.second );
          it = pacib.first;
        }
        else
        {
          unsubscribeBuffer( it );
          it->second = vertexAttribute;
        }
        subscribeBuffer( it );
        setEnabled( attrib, enable );
      }

      void VertexAttributeSet::subscribeBuffer( const AttributeContainer::const_iterator & it )
      {
        if ( it != m_vattribs.end() && it->first == 0 && it->second.getBuffer() )
        {
          it->second.getBuffer()->attach( &m_bufferObserver );
        }
      }

      void VertexAttributeSet::unsubscribeBuffer( const AttributeContainer::const_iterator & it )
      {
        if ( it != m_vattribs.end() && it->first == 0 && it->second.getBuffer() )
        {
          it->second.getBuffer()->detach( &m_bufferObserver );
        }
      }

      void VertexAttributeSet::combineBuffers()
      {
        // calculate stride & number of vertices
        unsigned int stride = 0;
        unsigned int numberOfVertices = 0;
        for ( unsigned int index = 0; index < VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT; ++index )
        {
          if ( isEnabled( index ) )
          {
            VertexAttribute attribute = getVertexAttribute( index );
            stride += attribute.getVertexDataBytes();

            if ( !numberOfVertices )
            {
              numberOfVertices = attribute.getVertexDataCount();
            }
            else
            {
              DP_ASSERT( numberOfVertices == attribute.getVertexDataCount() );
            }
          }
        }

        // create one big buffer with all data
        BufferSharedPtr newBufferSharedPtr = BufferHost::create();
        newBufferSharedPtr->setSize( numberOfVertices * stride );
        unsigned int offset = 0;
        for ( unsigned int index = 0; index < VertexAttributeSet::NVSG_VERTEX_ATTRIB_COUNT; ++index )
        {
          if ( isEnabled( index ) )
          {
            VertexAttribute attributeNew;
            const VertexAttribute& attributeOld = getVertexAttribute( index );
            attributeNew.setData( attributeOld.getVertexDataSize(), attributeOld.getVertexDataType(), newBufferSharedPtr, offset, stride, numberOfVertices );

            Buffer::DataReadLock drl(attributeOld.getBuffer());
            Buffer::DataWriteLock dwl(attributeNew.getBuffer(), Buffer::MAP_READWRITE );
            dp::util::stridedMemcpy( dwl.getPtr(), attributeNew.getVertexDataOffsetInBytes(), attributeNew.getVertexDataStrideInBytes(), 
              drl.getPtr(), attributeOld.getVertexDataOffsetInBytes(), attributeOld.getVertexDataStrideInBytes(),
              attributeOld.getVertexDataBytes(), numberOfVertices
              );
            offset += attributeOld.getVertexDataBytes();
            setVertexAttribute( index, attributeNew, true );
          }
        }
      }

      void VertexAttributeSet::onBufferChanged( )
      {
        // called only if position attribute buffer has been changed
        markDirty( NVSG_BOUNDING_VOLUMES );
      }

    } // namespace core
  } // namespace sg
} // namespace dp