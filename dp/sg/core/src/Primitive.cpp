// Copyright (c) 2010-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/Assert.h>
#include <dp/math/Spherent.h>
#include <limits>

#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/IndexSet.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      template <typename NormalOp>
      void dispatchCalculateNormals(Buffer::ConstIterator<Vec3f>::Type const& vertices,
                                    dp::sg::core::IndexSetSharedPtr const& indexSet,
                                    uint32_t count,
                                    uint32_t offset,
                                    std::vector<Vec3f> & normals,
                                    NormalOp op = {})
      {
        switch (indexSet->getIndexDataType())
        {
        case dp::DataType::INT_8:
          op.eval<uint8_t>(vertices, indexSet, count, offset, normals);
          break;
        case dp::DataType::INT_16:
          op.eval<int16_t>(vertices, indexSet, count, offset, normals);
          break;
        case dp::DataType::INT_32:
          op.eval<int32_t>(vertices, indexSet, count, offset, normals);
          break;
        case dp::DataType::UNSIGNED_INT_8:
          op.eval<uint8_t>(vertices, indexSet, count, offset, normals);
          break;
        case dp::DataType::UNSIGNED_INT_16:
          op.eval<uint16_t>(vertices, indexSet, count, offset, normals);
          break;
        case dp::DataType::UNSIGNED_INT_32:
          op.eval<uint32_t>(vertices, indexSet, count, offset, normals);
          break;
        default:
          DP_ASSERT(!"unsupported datatype");
        }
      }

      template <typename TangentOp>
      void dispatchCalculateTangents(VertexAttributeSetSharedPtr const& vas,
                                     VertexAttributeSet::AttributeID tc, dp::sg::core::IndexSetSharedPtr const& indexSet,
                                     uint32_t count,
                                     uint32_t offset,
                                     std::vector<Vec3f> & normals,
                                     TangentOp op = {})
      {
        switch (indexSet->getIndexDataType())
        {
        case dp::DataType::INT_8:
          op.eval<uint8_t>(vas, tc, indexSet, count, offset, normals);
          break;
        case dp::DataType::INT_16:
          op.eval<int16_t>(vas, tc, indexSet, count, offset, normals);
          break;
        case dp::DataType::INT_32:
          op.eval<int32_t>(vas, tc, indexSet, count, offset, normals);
          break;
        case dp::DataType::UNSIGNED_INT_8:
          op.eval<uint8_t>(vas, tc, indexSet, count, offset, normals);
          break;
        case dp::DataType::UNSIGNED_INT_16:
          op.eval<uint16_t>(vas, tc, indexSet, count, offset, normals);
          break;
        case dp::DataType::UNSIGNED_INT_32:
          op.eval<uint32_t>(vas, tc, indexSet, count, offset, normals);
          break;
        default:
          DP_ASSERT(!"unsupported datatype");
        }
      }

      BEGIN_REFLECTION_INFO( Primitive )
        DERIVE_STATIC_PROPERTIES( Primitive, Object );
      END_REFLECTION_INFO

      PrimitiveSharedPtr Primitive::create( PrimitiveType primitiveType )
      {
        DP_ASSERT( primitiveType != PrimitiveType::PATCHES );
        return( std::shared_ptr<Primitive>( new Primitive( primitiveType, PatchesType::NONE, PatchesMode::TRIANGLES ) ) );
      }

      PrimitiveSharedPtr Primitive::create( PatchesType patchesType, PatchesMode patchesMode )
      {
        return( std::shared_ptr<Primitive>( new Primitive( PrimitiveType::PATCHES, patchesType, patchesMode ) ) );
      }

      HandledObjectSharedPtr Primitive::clone() const
      {
        return( std::shared_ptr<Primitive>( new Primitive( *this ) ) );
      }

      Primitive::Primitive( PrimitiveType primitiveType, PatchesType patchesType, PatchesMode patchesMode )
        : m_primitiveType( primitiveType )
        , m_patchesType( patchesType )
        , m_patchesMode( patchesMode )
        , m_patchesSpacing( PatchesSpacing::EQUAL )
        , m_patchesOrdering( PatchesOrdering::CCW )
        , m_indexSet( 0 )
        , m_elementOffset( 0 )
        , m_elementCount( ~0 )
        , m_instanceCount( 1 )
        , m_vertexAttributeSet( 0 )
        , m_renderFlags( 0 )
        , m_cachedNumberOfPrimitives( ~0 )
        , m_cachedNumberOfFaces( ~0 )
        , m_cachedNumberOfPrimitiveRestarts( ~0 )
      {
        m_objectCode = ObjectCode::PRIMITIVE;
      }

      Primitive::Primitive(const Primitive& rhs)
      : BoundingVolumeObject(rhs) // copy base class part
      , m_boundingBox( rhs.m_boundingBox )
      , m_boundingSphere( rhs.m_boundingSphere )
      {
        m_objectCode = ObjectCode::PRIMITIVE;

        if ( rhs.m_indexSet )
        {
          m_indexSet = std::static_pointer_cast<dp::sg::core::IndexSet>(rhs.m_indexSet->clone());
          m_indexSet->attach( this );
        }
        if ( rhs.m_vertexAttributeSet )
        {
          m_vertexAttributeSet = std::static_pointer_cast<dp::sg::core::VertexAttributeSet>(rhs.m_vertexAttributeSet->clone());
          m_vertexAttributeSet->attach( this );
        }

        m_primitiveType    = rhs.m_primitiveType;
        m_patchesType      = rhs.m_patchesType;
        m_patchesMode      = rhs.m_patchesMode;
        m_patchesSpacing   = rhs.m_patchesSpacing;
        m_patchesOrdering  = rhs.m_patchesOrdering;
        m_renderFlags      = rhs.m_renderFlags;
        m_elementOffset    = rhs.m_elementOffset;
        m_elementCount     = rhs.m_elementCount;
        m_instanceCount    = rhs.m_instanceCount;

        m_cachedNumberOfPrimitives        = rhs.m_cachedNumberOfPrimitives;
        m_cachedNumberOfFaces             = rhs.m_cachedNumberOfFaces;
        m_cachedNumberOfPrimitiveRestarts = rhs.m_cachedNumberOfPrimitiveRestarts;
      }

      Primitive::~Primitive()
      {
        if ( m_vertexAttributeSet )
        {
          m_vertexAttributeSet->detach( this );
        }
        if ( m_indexSet )
        {
          m_indexSet->detach( this );
        }
      }

      Primitive & Primitive::operator=( const Primitive & rhs )
      {
        if (&rhs != this)
        {
          BoundingVolumeObject::operator=(rhs);
          setVertexAttributeSet( rhs.m_vertexAttributeSet );
          setIndexSet( rhs.m_indexSet );

          m_renderFlags      = rhs.m_renderFlags;
          m_primitiveType    = rhs.m_primitiveType;
          m_patchesType      = rhs.m_patchesType;
          m_patchesMode      = rhs.m_patchesMode;
          m_patchesSpacing   = rhs.m_patchesSpacing;
          m_patchesOrdering  = rhs.m_patchesOrdering;
          m_elementOffset    = rhs.m_elementOffset;
          m_elementCount     = rhs.m_elementCount;
          m_instanceCount    = rhs.m_instanceCount;
          m_boundingBox      = rhs.m_boundingBox;
          m_boundingSphere   = rhs.m_boundingSphere;

          m_cachedNumberOfPrimitives        = rhs.m_cachedNumberOfPrimitives;
          m_cachedNumberOfFaces             = rhs.m_cachedNumberOfFaces;
          m_cachedNumberOfPrimitiveRestarts = rhs.m_cachedNumberOfPrimitiveRestarts;
        }

        return *this;
      }

      void Primitive::setVertexAttributeSet( const VertexAttributeSetSharedPtr & vash )
      {
        if ( vash != m_vertexAttributeSet )
        {
          // see above
          if ( vash )
          {
            vash->attach( this );
          }
          if ( m_vertexAttributeSet )
          {
            m_vertexAttributeSet->detach( this );
          }
          m_vertexAttributeSet = vash;
          notify( Event(this ) );

          clearCachedCounts();
        }
      }

      void Primitive::setIndexSet( const IndexSetSharedPtr & iset )
      {
        if( m_indexSet != iset )
        {
          if ( iset )
          {
            iset->attach( this );
          }
          if ( m_indexSet )
          {
            m_indexSet->detach( this );
          }

          m_indexSet = iset;

          // set render flags if necessary
          if( m_indexSet )
          {
            // assure there is really index data attached
            m_renderFlags |= DRAW_INDEXED;
          }
          else
          {
            m_renderFlags &= ~DRAW_INDEXED;
          }

          clearCachedCounts();
          notify( Event( this ) );
        }
      }

      bool Primitive::makeIndexed()
      {
        if ( !isIndexed() )
        {
          // if there are no indices, fill in the trivial index set
          unsigned int elementCount = getElementCount();
          std::vector<unsigned int> indices( elementCount );
          for ( unsigned int i=0 ; i<elementCount ; i++ )
          {
            indices[i] = i + m_elementOffset;
          }

          IndexSetSharedPtr indexSet( IndexSet::create() );
          indexSet->setData( &indices[0], elementCount );

          setIndexSet( indexSet );
          setElementRange( 0, ~0 );
          return( true );
        }
        return( false );
      }

      void Primitive::setInstanceCount( unsigned int icount )
      {
        if( m_instanceCount != icount )
        {
          m_instanceCount = icount;

          if( m_instanceCount > 1 )
          {
            m_renderFlags |= DRAW_INSTANCED;
          }
          else
          {
            m_renderFlags &= ~DRAW_INSTANCED;
          }

          // will change BV, but we have no way to know how, since primitives are output in rasterizer

          // Fixme?  Cached counts don't change with more instances ATM
          //clearCachedCounts();
        }
      }

      void Primitive::setPatchesOrdering( PatchesOrdering po )
      {
        if ( m_patchesOrdering != po )
        {
          m_patchesOrdering = po;
          notify( Event( this ) );
        }
      }

      void Primitive::setPatchesSpacing( PatchesSpacing ps )
      {
        if ( m_patchesSpacing != ps )
        {
          m_patchesSpacing = ps;
          notify( Event( this ) );
        }
      }

      bool Primitive::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object.get() == this )
        {
          return( true );
        }

        bool equi = std::dynamic_pointer_cast<Primitive>(object) && BoundingVolumeObject::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          PrimitiveSharedPtr const& p = std::static_pointer_cast<Primitive>(object);

          equi = m_primitiveType        == p->m_primitiveType &&
                 m_elementOffset        == p->m_elementOffset &&
                 m_elementCount         == p->m_elementCount &&
                 m_instanceCount        == p->m_instanceCount &&
                 !!m_vertexAttributeSet == !!p->m_vertexAttributeSet &&
                 !!m_indexSet           == !!p->m_indexSet &&
                 ( !( m_primitiveType == PrimitiveType::PATCHES ) || (  ( m_patchesType == p->m_patchesType )
                                                                && ( m_patchesMode == p->m_patchesMode )
                                                                && ( m_patchesSpacing == p->m_patchesSpacing )
                                                                && ( m_patchesOrdering == p->m_patchesOrdering ) ) );
          if ( equi )
          {
            if ( deepCompare )
            {
              if ( m_vertexAttributeSet )
              {
                DP_ASSERT( p->m_vertexAttributeSet );
                equi = m_vertexAttributeSet->isEquivalent( p->m_vertexAttributeSet, ignoreNames, true );
              }
              if ( equi && m_indexSet )
              {
                DP_ASSERT( p->m_indexSet );
                equi = m_indexSet->isEquivalent( p->m_indexSet, ignoreNames, true );
              }
            }
            else
            {
              equi =    ( m_vertexAttributeSet  == p->m_vertexAttributeSet )
                    &&  ( m_indexSet            == p->m_indexSet );
            }
          }
        }
        return( equi );
      }


      template <typename T>
      unsigned int scanForPrimitiveRestart( const T * indices, unsigned int start, unsigned int count, unsigned int prIndex )
      {
        unsigned int pcount = 0;

        for( unsigned int i = start; i < count; i ++ )
        {
          // note, comparison is done as unsigned int, like in the GL
          if( indices[i] == prIndex )
          {
            pcount++;
          }
        }

        return pcount;
      }

      unsigned int Primitive::getNumberOfPrimitiveRestarts() const
      {
        if( isIndexed() && (m_cachedNumberOfPrimitiveRestarts == ~0) )
        {
          // we must scan the index list to determine if there are any Primitive Restart indices there in order to arrive at
          // the proper count
          unsigned int prIndex = m_indexSet->getPrimitiveRestartIndex();

          unsigned int elementOffset = getElementOffset();
          unsigned int elementCount  = getElementCount();

          Buffer::DataReadLock reader( m_indexSet->getBuffer() );

          switch( m_indexSet->getIndexDataType() )
          {
          case dp::DataType::INT_8:
            m_cachedNumberOfPrimitiveRestarts = scanForPrimitiveRestart<int8_t>(reader.getPtr<int8_t>(), elementOffset, elementCount, prIndex);
            break;

          case dp::DataType::INT_16:
            m_cachedNumberOfPrimitiveRestarts = scanForPrimitiveRestart<int16_t>(reader.getPtr<int16_t>(), elementOffset, elementCount, prIndex);
            break;

          case dp::DataType::INT_32:
            m_cachedNumberOfPrimitiveRestarts = scanForPrimitiveRestart<int32_t>(reader.getPtr<int32_t>(), elementOffset, elementCount, prIndex);
            break;

          case dp::DataType::UNSIGNED_INT_8:
            m_cachedNumberOfPrimitiveRestarts = scanForPrimitiveRestart<uint8_t>(reader.getPtr<uint8_t>(), elementOffset, elementCount, prIndex);
            break;

          case dp::DataType::UNSIGNED_INT_16:
            m_cachedNumberOfPrimitiveRestarts = scanForPrimitiveRestart<uint16_t>(reader.getPtr<uint16_t>(), elementOffset, elementCount, prIndex);
            break;

          case dp::DataType::UNSIGNED_INT_32:
            m_cachedNumberOfPrimitiveRestarts = scanForPrimitiveRestart<uint32_t>(reader.getPtr<uint32_t>(), elementOffset, elementCount, prIndex);
            break;

          default:
            DP_ASSERT(!"INVALID INDEX TYPE");
            return 0;
          }
        }
        else if(!isIndexed())
        {
          // if we aren't indexed, there can be no primitive restarts
          m_cachedNumberOfPrimitiveRestarts = 0;
        }

        return m_cachedNumberOfPrimitiveRestarts;
      }

      void Primitive::determinePrimitiveAndFaceCount() const
      {
        unsigned int elementOffset = getElementOffset();
        unsigned int elementCount  = getElementCount();
        // gets the number of PR in range..
        unsigned int numberOfPrimitiveRestarts = getNumberOfPrimitiveRestarts();

        unsigned int correctedCount = elementCount - numberOfPrimitiveRestarts;
        unsigned int vpe = getNumberOfVerticesPerPrimitive();

        switch( getPrimitiveType() )
        {
          case PrimitiveType::TRIANGLE_STRIP_ADJACENCY:
            m_cachedNumberOfPrimitives = numberOfPrimitiveRestarts + 1;
            // for N primitives, we need 2n + 4 indices.
            m_cachedNumberOfFaces = (correctedCount - 4 * m_cachedNumberOfPrimitives) / 2;
            break;

          case PrimitiveType::LINE_STRIP_ADJACENCY:
          case PrimitiveType::LINE_STRIP:
          case PrimitiveType::LINE_LOOP:
            m_cachedNumberOfPrimitives = numberOfPrimitiveRestarts + 1;
            m_cachedNumberOfFaces = 0;
            break;

          case PrimitiveType::TRIANGLE_STRIP:
          case PrimitiveType::TRIANGLE_FAN:
            m_cachedNumberOfPrimitives = numberOfPrimitiveRestarts + 1;
            // for N primitives, we need n + 2 indices.
            m_cachedNumberOfFaces = correctedCount - 2 * m_cachedNumberOfPrimitives;
            break;

          case PrimitiveType::QUAD_STRIP:
            m_cachedNumberOfPrimitives = numberOfPrimitiveRestarts + 1;
            // for N primitives, we need 2n + 2 indices.
            m_cachedNumberOfFaces = (correctedCount - 2 * m_cachedNumberOfPrimitives) / 2;
            break;

          case PrimitiveType::POLYGON:
            m_cachedNumberOfPrimitives = numberOfPrimitiveRestarts + 1;
            m_cachedNumberOfFaces = m_cachedNumberOfPrimitives;
            break;

          case PrimitiveType::POINTS:
            m_cachedNumberOfPrimitives = correctedCount;
            m_cachedNumberOfFaces = 0;
            break;

          case PrimitiveType::LINES_ADJACENCY:
          case PrimitiveType::LINES:
            m_cachedNumberOfPrimitives = correctedCount / vpe;
            m_cachedNumberOfFaces = 0;
            break;

          case PrimitiveType::QUADS:
          case PrimitiveType::TRIANGLES:
          case PrimitiveType::TRIANGLES_ADJACENCY:
          case PrimitiveType::PATCHES:
            m_cachedNumberOfFaces    = correctedCount / vpe;
            m_cachedNumberOfPrimitives = m_cachedNumberOfFaces;
            break;

          default:
          case PrimitiveType::UNINITIALIZED:
            m_cachedNumberOfPrimitives = 0;
            m_cachedNumberOfFaces    = 0;
            break;
        }
      }

      void Primitive::generateTexCoords( TextureCoordType type, VertexAttributeSet::AttributeID tc, bool overwrite )
      {
        DP_ASSERT( ( VertexAttributeSet::AttributeID::TEXCOORD0 <= tc ) && ( tc <= VertexAttributeSet::AttributeID::TEXCOORD7 ) );
        calculateTexCoords( type, tc, overwrite );
      }

      void Primitive::calculateTexCoords(TextureCoordType type, VertexAttributeSet::AttributeID tc, bool overwrite)
      {
        if ( m_vertexAttributeSet )
        {
          if ( overwrite || !m_vertexAttributeSet->getNumberOfVertexData(tc) )
          {
            core::generateTexCoords(m_vertexAttributeSet, type, getBoundingSphere(), tc);
          }
        }
      }

      void Primitive::feedHashGenerator( util::HashGenerator & hg ) const
      {
        BoundingVolumeObject::feedHashGenerator( hg );
        hg.update( reinterpret_cast<const unsigned char *>(&m_primitiveType), sizeof(m_primitiveType) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_elementOffset), sizeof(m_elementOffset) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_elementCount), sizeof(m_elementCount) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_instanceCount), sizeof(m_instanceCount) );
        if ( m_indexSet )
        {
          hg.update( m_indexSet );
        }
        if ( m_vertexAttributeSet )
        {
          hg.update( m_vertexAttributeSet );
        }
        if ( m_primitiveType == PrimitiveType::PATCHES )
        {
          hg.update( reinterpret_cast<const unsigned char *>(&m_patchesType), sizeof(m_patchesType) );
          hg.update( reinterpret_cast<const unsigned char *>(&m_patchesMode), sizeof(m_patchesMode) );
          hg.update( reinterpret_cast<const unsigned char *>(&m_patchesSpacing), sizeof(m_patchesSpacing) );
          hg.update( reinterpret_cast<const unsigned char *>(&m_patchesOrdering), sizeof(m_patchesOrdering) );
        }
      }

      template <typename IndexType>
      Box3f getBoundingBoxForIndices( const IndexSetSharedPtr &indexSet, unsigned int offset, unsigned int count
                                    , unsigned int primitiveRestartIndex, const Buffer::ConstIterator<Vec3f>::Type &points )
      {
        Box3f bbox;

        Buffer::DataReadLock drl( indexSet->getBuffer() );
        const IndexType* indices = drl.getPtr<IndexType>() + offset;

        for ( unsigned int i=0; i<count; ++i )
        {
          // make sure to skip primitive restart index
          if( indices[i] != primitiveRestartIndex )
          {
            bbox.update(points[indices[i]]);
          }
        }

        return bbox;
      }

      Box3f Primitive::calculateBoundingBox() const
      {
        Box3f bbox;

        unsigned int offset = getElementOffset();
        unsigned int count  = getElementCount();

        Buffer::ConstIterator<Vec3f>::Type points = m_vertexAttributeSet->getVertices();

        if( isIndexed() )
        {
          // The general purpose IndexSet::ConstIterator is slow since each indexed access is a virtual function call.
          // Instead use a templated version of the algorithm which is operation directly on the buffer of the IndexSet to gain speed.
          const IndexSetSharedPtr &indexSet = getIndexSet();
          unsigned int prIdx = indexSet->getPrimitiveRestartIndex();
          switch (indexSet->getIndexDataType() )
          {
          case dp::DataType::UNSIGNED_INT_32:
            bbox = getBoundingBoxForIndices<uint32_t>( indexSet, offset, count, prIdx, points );
            break;
          case dp::DataType::INT_32:
            bbox = getBoundingBoxForIndices<int32_t>( indexSet, offset, count, prIdx, points );
            break;
          case dp::DataType::UNSIGNED_INT_16:
            bbox = getBoundingBoxForIndices<uint16_t>( indexSet, offset, count, prIdx, points );
            break;
          case dp::DataType::INT_16:
            bbox = getBoundingBoxForIndices<int16_t>( indexSet, offset, count, prIdx, points );
            break;
          case dp::DataType::UNSIGNED_INT_8:
            bbox = getBoundingBoxForIndices<uint8_t>( indexSet, offset, count, prIdx, points );
            break;
          case dp::DataType::INT_8:
            bbox = getBoundingBoxForIndices<int8_t>( indexSet, offset, count, prIdx, points );
            break;
          default:
            DP_ASSERT( 0 && "unsupported datatype for indexset");
          }
        }
        else
        {
          DP_ASSERT( offset+count <= m_vertexAttributeSet->getNumberOfVertices() );
          for( unsigned int i=offset; i < offset+count; ++i )
          {
            bbox.update( points[i] );
          }
        }

        return bbox;
      }

      template <typename IndexType>
      float computeMinRadius(IndexSetSharedPtr const& indexSet,
                             uint32_t count,
                             uint32_t offset,
                             Buffer::ConstIterator<Vec3f>::Type const& points,
                             Vec3f const& center,
                             size_t numberOfVertices)
      {
        IndexSet::ConstIterator<IndexType> indices(indexSet, offset);
        float minRadius = 0.f;

        unsigned int primitiveRestartIndex = indexSet->getPrimitiveRestartIndex();

        bool invalidIndices = false;
        for (uint32_t i = 0; i < count; i++)
        {
          IndexType index = indices[i];

          if ((index != primitiveRestartIndex) && (index < numberOfVertices))
          {
            float d = lengthSquared(points[index] - center);
            if (minRadius < d)
            {
              minRadius = d;
            }
          }
          else
          {
            invalidIndices |= (index != primitiveRestartIndex);
          }
        }
        if (invalidIndices)
        {
          std::cerr << "Primitive contains out of range indices" << std::endl;
          DP_ASSERT(false);
        }

        return minRadius;
      }

      Sphere3f Primitive::calculateBoundingSphere() const
      {
        unsigned int offset = getElementOffset();
        unsigned int count  = getElementCount();

        Sphere3f bsphere;
        Box3f bbox = calculateBoundingBox();
        Vec3f center = bbox.getCenter();

        Buffer::ConstIterator<Vec3f>::Type points = m_vertexAttributeSet->getVertices();

        // now determine min radius
        float minRadius = 0.f;
        if( isIndexed() )
        {
          switch (m_indexSet->getIndexDataType())
          {
          case dp::DataType::INT_8:
            minRadius = computeMinRadius<int8_t>(m_indexSet, count, offset, points, center, m_vertexAttributeSet->getNumberOfVertices());
            break;
          case dp::DataType::INT_16:
            minRadius = computeMinRadius<int16_t>(m_indexSet, count, offset, points, center, m_vertexAttributeSet->getNumberOfVertices());
            break;
          case dp::DataType::INT_32:
            minRadius = computeMinRadius<int32_t>(m_indexSet, count, offset, points, center, m_vertexAttributeSet->getNumberOfVertices());
            break;
          case dp::DataType::UNSIGNED_INT_8:
            minRadius = computeMinRadius<uint8_t>(m_indexSet, count, offset, points, center, m_vertexAttributeSet->getNumberOfVertices());
            break;
          case dp::DataType::UNSIGNED_INT_16:
            minRadius = computeMinRadius<uint16_t>(m_indexSet, count, offset, points, center, m_vertexAttributeSet->getNumberOfVertices());
            break;
          case dp::DataType::UNSIGNED_INT_32:
            minRadius = computeMinRadius<uint32_t>(m_indexSet, count, offset, points, center, m_vertexAttributeSet->getNumberOfVertices());
            break;
          default:
            DP_ASSERT(!"unsupported datatype");
          }
        }
        else
        {
          unsigned int end = offset + count;
          if ( end > m_vertexAttributeSet->getNumberOfVertices() )
          {
            std::cerr << "Primitive " << getName() << " references out of range vertices" << std::endl;
            DP_ASSERT( false );
            end = m_vertexAttributeSet->getNumberOfVertices();
          }
          for( unsigned int i = offset; i < end; i ++ )
          {
            float d = lengthSquared( points[i] - center );
            if( minRadius < d )
            {
              minRadius = d;
            }
          }
        }

        return Sphere3f( center, sqrt(minRadius) );
      }

      void Primitive::setElementRange( unsigned int offset, unsigned int count )
      {
        if ( (m_elementOffset != offset) || (m_elementCount != count) )
        {
#ifndef NDEBUG
          unsigned int maxCount = getMaxElementCount();
          DP_ASSERT( offset <= maxCount );
          if (count != ~0)
          {
            DP_ASSERT( offset + count <= maxCount );
          }
#endif

          // Original user values. m_elementCount == ~0 is allowed.
          m_elementOffset = offset;
          m_elementCount  = count;

          notify( Event( this ) );

          clearCachedCounts();
        }
      }

      // Return the number of indices or vertices irrespective of the currently active m_elementOffset and m_elementCount.
      unsigned int Primitive::getMaxElementCount() const
      {
        return m_indexSet ?
               m_indexSet->getNumberOfIndices() :
               getVertexAttributeSet()->getNumberOfVertexData( VertexAttributeSet::AttributeID::POSITION );
      }

      unsigned int Primitive::getElementCount() const
      {
        return ( m_elementCount == ~0 ) ?               // Default case?
               getMaxElementCount() - m_elementOffset : // Use tail of the IndexSet or VertexAttributeSet.
               m_elementCount;                          // Use user defined count.
      }

      struct calculateNormalsPolygonWithIndices
      {
        template <typename IndexType>
        void eval(Buffer::ConstIterator<Vec3f>::Type const& vertices,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & normals)
        {
          // calculate an average normal on each face
          unsigned int pri = indexSet->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);

          Vec3f faceNormal = Vec3f(0.0f, 0.0f, 0.0f);
          DP_ASSERT((indices[0] != pri) && (indices[1] != pri));
          uint32_t i0 = 0;
          for (uint32_t i = 2; i < count; i++)
          {
            if (indices[i] == pri)
            {
              for (uint32_t j = i0; j < i; j++)
              {
                normals[indices[j]] += faceNormal;
              }
              faceNormal = Vec3f(0.0f, 0.0f, 0.0f);
              i0 = i + 1;
              i += 2;
            }
            else
            {
              faceNormal += (vertices[indices[i - 1]] - vertices[indices[i0]])
                          ^ (vertices[indices[i    ]] - vertices[indices[i0]]);
            }
          }
          if (indices[count - 1] != pri)
          {
            for (uint32_t j = i0; j < count; j++)
            {
              normals[indices[j]] += faceNormal;
            }
          }
        }
      };

      void Primitive::calculateNormalsPolygon( Buffer::ConstIterator<Vec3f>::Type & vertices, std::vector<Vec3f> & normals )
      {
        uint32_t count  = getElementCount();
        uint32_t offset = getElementOffset();

        if ( m_indexSet )
        {
          dispatchCalculateNormals<calculateNormalsPolygonWithIndices>(vertices, m_indexSet, count, offset, normals);
        }
        else
        {
          Vec3f faceNormal = Vec3f( 0.0f, 0.0f, 0.0f );
          for ( size_t i=offset+2 ; i<offset+count ; i++ )
          {
            faceNormal += ( vertices[i-1] - vertices[0] ) ^ ( vertices[i] - vertices[0] );
          }
          for ( size_t i=offset ; i<offset+count ; i++ )
          {
            normals[i] = faceNormal;
          }
        }
      }

      struct calculateNormalsQuadWithIndices {
        template <typename IndexType>
        void eval(Buffer::ConstIterator<Vec3f>::Type const& vertices,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & normals)
        {
          // calculate the normals for each facet
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);
          for (uint32_t i = 0; i < count; i += 4)
          {
            //  determine the face normal
            Vec3f faceNormal = (vertices[indices[i + 2]] - vertices[indices[i + 0]])
                             ^ (vertices[indices[i + 3]] - vertices[indices[i + 1]]);

            //  and accumulate it to the three vertices of the facet
            for (uint32_t j = 0; j < 4; j++)
            {
              normals[indices[i + j]] += faceNormal;
            }
          }
        }
      };


      void Primitive::calculateNormalsQuad( Buffer::ConstIterator<Vec3f>::Type & vertices, std::vector<Vec3f> & normals )
      {
        uint32_t count  = getElementCount();
        uint32_t offset = getElementOffset();
        DP_ASSERT( count % 4 == 0 );

        if ( m_indexSet )
        {
          dispatchCalculateNormals<calculateNormalsQuadWithIndices>(vertices, m_indexSet, count, offset, normals);
        }
        else
        {
          for ( size_t i=offset ; i<offset+count ; i+=4 )
          {
            // determine the face normal
            Vec3f faceNormal = ( vertices[i+2] - vertices[i+0] ) ^ ( vertices[i+3] - vertices[i+1] );

            // each vertex normal just gets the face normal
            for ( unsigned int j=0 ; j<4 ; j++ )
            {
              normals[i+j] = faceNormal;
            }
          }
        }
      }

      struct calculateNormalsQuadStripWithIndices
      {
        template <typename IndexType>
        void eval(Buffer::ConstIterator<Vec3f>::Type const& vertices,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & normals)
        {
          // calculate the normals for each facet
          unsigned int pri = indexSet->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);

          DP_ASSERT((indices[0] != pri) && (indices[1] != pri) && (indices[2] != pri));
          for (uint32_t i = 3; i < count; i += 2)
          {
            if (indices[i] == pri)
            {
              DP_ASSERT((indices[i + 1] != pri) && (indices[i + 2] != pri) && (indices[i + 3] != pri));
              i += 2;   // advance to end of first quad in next strip
            }
            else
            {
              //  determine the face normal
              Vec3f faceNormal = (vertices[indices[i - 0]] - vertices[indices[i - 3]])
                               ^ (vertices[indices[i - 1]] - vertices[indices[i - 2]]);

              //  and accumulate it to the four vertices of the facet
              for (uint32_t j = 0; j < 4; j++)
              {
                normals[indices[i - j]] += faceNormal;
              }
            }
          }
        }
      };

      void Primitive::calculateNormalsQuadStrip( Buffer::ConstIterator<Vec3f>::Type & vertices, std::vector<Vec3f> & normals )
      {
        unsigned int count  = getElementCount();
        unsigned int offset = getElementOffset();

        if ( m_indexSet )
        {
          dispatchCalculateNormals<calculateNormalsQuadStripWithIndices>(vertices, m_indexSet, count, offset, normals);
        }
        else
        {
          for ( size_t i=offset ; i<offset+count-2 ; i+=2 )
          {
            //  determine the face normal
            Vec3f faceNormal = ( vertices[i+3] - vertices[i+0] ) ^ ( vertices[i+2] - vertices[i+1] );

            //  and accumulate it to the four vertices of the facet
            for ( unsigned int j=0 ; j<4 ; j++ )
            {
              normals[i+j] += faceNormal;
            }
          }
        }
      }

      struct calculateNormalsTriangleWithIndices
      {
        template <typename IndexType>
        void eval(Buffer::ConstIterator<Vec3f>::Type const& vertices,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & normals)
        {
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);
          for (uint32_t i = 0; i < count; i += 3)
          {
            //  determine the face normal
            Vec3f faceNormal = (vertices[indices[i + 1]] - vertices[indices[i + 0]])
                             ^ (vertices[indices[i + 2]] - vertices[indices[i + 0]]);

            //  and accumulate it to the three vertices of the facet
            for (uint32_t j = 0; j < 3; j++)
            {
              normals[indices[i + j]] += faceNormal;
            }
          }
        }
      };

      void Primitive::calculateNormalsTriangle( Buffer::ConstIterator<Vec3f>::Type & vertices, std::vector<Vec3f> & normals )
      {
        unsigned int count  = getElementCount();
        unsigned int offset = getElementOffset();
        DP_ASSERT( count % 3 == 0 );

        if ( m_indexSet )
        {
          dispatchCalculateNormals<calculateNormalsTriangleWithIndices>(vertices, m_indexSet, count, offset, normals);
        }
        else
        {
          for ( size_t i=offset ; i<offset+count ; i+=3 )
          {
            // determine the face normal
            Vec3f faceNormal = ( vertices[i+1] - vertices[i+0] )
                             ^ ( vertices[i+2] - vertices[i+0] );

            // each vertex normal just gets the face normal
            for ( unsigned int j=0 ; j<3 ; j++ )
            {
              normals[i+j] = faceNormal;
            }
          }
        }
      }

      struct calculateNormalsTriFanWithIndices
      {
        template <typename IndexType>
        void eval(Buffer::ConstIterator<Vec3f>::Type const& vertices,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & normals)
        {
          // calculate the normals for each facet
          unsigned int pri = indexSet->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);

          unsigned int start = 0;
          DP_ASSERT((indices[start] != pri) && (indices[start + 1] != pri));
          Vec3f edge1 = vertices[indices[start + 1]] - vertices[indices[start]];
          for (uint32_t i = 2; i < count; i++)
          {
            if (indices[i] == pri)
            {
              DP_ASSERT(i + 2 < count);
              start = i + 1;
              DP_ASSERT((indices[start] != pri) && (indices[start + 1] != pri));
              edge1 = vertices[indices[start + 1]] - vertices[indices[start]];
              i += 2;   // advance to end of first triangle in next strip
            }
            else
            {
              //  determine the face normal
              Vec3f edge0 = edge1;
              edge1 = vertices[indices[i]] - vertices[indices[start]];
              Vec3f faceNormal = edge0 ^ edge1;

              //  and accumulate it to the three vertices of the facet
              normals[indices[start]] += faceNormal;
              normals[indices[i - 1]] += faceNormal;
              normals[indices[i]] += faceNormal;
            }
          }
        }
      };

      void Primitive::calculateNormalsTriFan( Buffer::ConstIterator<Vec3f>::Type & vertices, std::vector<Vec3f> & normals )
      {
        unsigned int count  = getElementCount();
        unsigned int offset = getElementOffset();

        if ( m_indexSet )
        {
          dispatchCalculateNormals<calculateNormalsTriFanWithIndices>(vertices, m_indexSet, count, offset, normals);
        }
        else
        {
          Vec3f edge1 = vertices[1] - vertices[0];
          for ( size_t i=offset+1 ; i<offset+count-1 ; i++ )
          {
            //  determine the face normal
            Vec3f edge0 = edge1;
            edge1 = vertices[i+1] - vertices[0];
            Vec3f faceNormal = edge0 ^ edge1;

            //  and accumulate it to the three vertices of the facet
            normals[0] += faceNormal;
            normals[i] += faceNormal;
            normals[i+1] += faceNormal;
          }
        }
      }

      struct calculateNormalsTriStripWithIndices
      {
        template <typename IndexType>
        void eval(Buffer::ConstIterator<Vec3f>::Type const& vertices,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & normals)
        {
          // calculate the normals for each facet
          unsigned int pri = indexSet->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);

          DP_ASSERT((indices[0] != pri) && (indices[1] != pri));
          for (uint32_t i = 2, j = 0; i < count; i++, j++)
          {
            if (indices[i] == pri)
            {
              DP_ASSERT((indices[i + 1] != pri) && (indices[i + 2] != pri));
              i += 2;   // advance to end of first triangle in next strip
              j = ~0;   // reset j such that it's zero on next iteration
            }
            else
            {
              Vec3f faceNormal;
              if (j & 1)
              { // odd normals
                faceNormal = (vertices[indices[i - 0]] - vertices[indices[i - 2]])
                           ^ (vertices[indices[i - 1]] - vertices[indices[i - 2]]);
              }
              else
              { // even normals
                faceNormal = (vertices[indices[i - 1]] - vertices[indices[i - 2]])
                           ^ (vertices[indices[i - 0]] - vertices[indices[i - 2]]);
              }
              for (uint32_t k = 0; k < 3; k++)
              {
                normals[indices[i - 2 + k]] += faceNormal;
              }
            }
          }
        }
      };

      void Primitive::calculateNormalsTriStrip( Buffer::ConstIterator<Vec3f>::Type & vertices, std::vector<Vec3f> & normals )
      {
        unsigned int count  = getElementCount();
        unsigned int offset = getElementOffset();

        if ( m_indexSet )
        {
          dispatchCalculateNormals<calculateNormalsTriStripWithIndices>(vertices, m_indexSet, count, offset, normals);
        }
        else
        {
          //  calculate even normals
          for ( size_t i=offset ; i<offset+count-2 ; i+=2 )
          {
            Vec3f faceNormal = ( vertices[i+1] - vertices[i] ) ^ ( vertices[i+2] - vertices[i] );
            for ( unsigned int j=0 ; j<3 ; j++ )
            {
              normals[i+j] += faceNormal;
            }
          }
          //  calculate odd normals
          for ( size_t i=offset+1 ; i<offset+count-2 ; i+=2 )
          {
            Vec3f faceNormal = ( vertices[i+2] - vertices[i] ) ^ ( vertices[i+1] - vertices[i] );
            for ( unsigned int j=0 ; j<3 ; j++ )
            {
              normals[i+j] += faceNormal;
            }
          }
        }
      }

      bool Primitive::generateNormals( bool overwrite )
      {
        return( calculateNormals( overwrite ) );
      }

      bool Primitive::calculateNormals( bool overwrite )
      {
        bool ok = calculateNormals( getVertexAttributeSet(), overwrite );

        return( ok );
      }

      bool Primitive::calculateNormals( const VertexAttributeSetSharedPtr& vassp, bool overwrite )
      {
        DP_ASSERT( vassp->getNumberOfVertices() );

        bool ok =   ( overwrite || !vassp->getNumberOfNormals() )
                &&  (   ( m_primitiveType == PrimitiveType::TRIANGLE_STRIP )
                    ||  ( m_primitiveType == PrimitiveType::TRIANGLE_FAN )
                    ||  ( m_primitiveType == PrimitiveType::TRIANGLES )
                    ||  ( m_primitiveType == PrimitiveType::QUAD_STRIP )
                    ||  ( m_primitiveType == PrimitiveType::QUADS )
                    ||  ( m_primitiveType == PrimitiveType::POLYGON ) );
        if ( ok )
        {
          // temporary normals buffer
          std::vector<Vec3f> normals;
          if ( vassp->getNumberOfNormals() )
          {
            DP_ASSERT( vassp->getNumberOfVertices() == vassp->getNumberOfNormals() );
            // get the current normals, to prevent modifying normals that are not used in this Primitive
            normals.assign( vassp->getNormals(), vassp->getNormals() + vassp->getNumberOfNormals() );
          }
          else
          {
            // just resize the std::vector of normals to hold the needed number of normals
            normals.resize( vassp->getNumberOfVertices() );
          }

          // initialize with zero
          memset( &normals[0], 0, normals.size() * sizeof(Vec3f) );

          DP_ASSERT( vassp->getVertexBuffer( VertexAttributeSet::AttributeID::POSITION ) );
          Buffer::ConstIterator<Vec3f>::Type vertices = vassp->getVertices();

          // calculate the normals, depending on primitive type
          switch( m_primitiveType )
          {
            case PrimitiveType::TRIANGLE_STRIP :
              calculateNormalsTriStrip( vertices, normals );
              break;
            case PrimitiveType::TRIANGLE_FAN :
              calculateNormalsTriFan( vertices, normals );
              break;
            case PrimitiveType::TRIANGLES :
              calculateNormalsTriangle( vertices, normals );
              break;
            case PrimitiveType::QUAD_STRIP :
              calculateNormalsQuadStrip( vertices, normals );
              break;
            case PrimitiveType::QUADS :
              calculateNormalsQuad( vertices, normals );
              break;
            case PrimitiveType::POLYGON :
              calculateNormalsPolygon( vertices, normals );
              break;
            default :
              DP_ASSERT( !"Normals calculation not implemented for this Primitive type" );
              break;
          }

          //  normalize the normals calculated above
          for ( size_t i=0 ; i<normals.size() ; i++ )
          {
            normals[i].normalize();
          }

          // and throw them in
          vassp->setNormals( &normals[0], dp::checked_cast<unsigned int>(normals.size()) );
        }
        return( ok );
      }

      void calculateTangent( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc, unsigned int i, unsigned int i1
                           , unsigned int i2, std::vector<Vec3f> & tangents )
      {
        //  determine the texture dependent plane weight
        DP_ASSERT( vas->getVertexBuffer( VertexAttributeSet::AttributeID::POSITION ) );
        DP_ASSERT( tc != VertexAttributeSet::AttributeID::POSITION );

        Buffer::ConstIterator<Vec3f>::Type vertices = vas->getVertices();
        Vec3f edge0 = vertices[i1] - vertices[i];
        Vec3f edge1 = vertices[i2] - vertices[i];

        Buffer::ConstIterator<Vec2f>::Type texCoords = vas->getVertexData<Vec2f>(tc);
        Vec2f dTex0 = texCoords[i1] - texCoords[i];
        Vec2f dTex1 = texCoords[i2] - texCoords[i];

        // accumulate the tangents on this vertex
        // will be normalized later on, resulting in some kind of area-weighted tangent direction
        tangents[i] += dTex1[1] * edge0 - dTex0[1] * edge1;
      }


      struct calculateTangentsQuadWithIndices
      {
        template <typename IndexType>
        void eval(VertexAttributeSetSharedPtr const& vas,
                  VertexAttributeSet::AttributeID tc,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & tangents)
        {
          unsigned int pri = indexSet->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);

          for (uint32_t i = 3; i < count; i += 4)
          {
            DP_ASSERT((indices[i - 3] != pri) && (indices[i - 2] != pri) && (indices[i - 1] != pri) && (indices[i] != pri));
            calculateTangent(vas, tc, indices[i - 3], indices[i - 2], indices[i], tangents);
            calculateTangent(vas, tc, indices[i - 2], indices[i - 1], indices[i - 3], tangents);
            calculateTangent(vas, tc, indices[i - 1], indices[i], indices[i - 2], tangents);
            calculateTangent(vas, tc, indices[i], indices[i - 3], indices[i - 1], tangents);
          }
        }
      };

      void Primitive::calculateTangentsQuad( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc, std::vector<Vec3f> & tangents )
      {
        unsigned int offset = getElementOffset();
        unsigned int count  = getElementCount();
        DP_ASSERT( count % 4 == 0 );

        if ( m_indexSet )
        {
          dispatchCalculateTangents<calculateTangentsQuadWithIndices>(vas, tc, m_indexSet, count, offset, tangents);
        }
        else
        {
          for ( unsigned int i=offset+3 ; i<offset+count ; i+=4 )
          {
            calculateTangent( vas, tc, i-3, i-2, i, tangents );
            calculateTangent( vas, tc, i-2, i-1, i-3, tangents );
            calculateTangent( vas, tc, i-1, i, i-2, tangents );
            calculateTangent( vas, tc, i, i-3, i-1, tangents );
          }
        }
      }


      struct calculateTangentsQuadStripWithIndices
      {
        template <typename IndexType>
        void eval(VertexAttributeSetSharedPtr const& vas,
                  VertexAttributeSet::AttributeID tc,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & tangents)
        {
          // calculate the tangents and binormals for each facet
          unsigned int pri = indexSet->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);

          for (uint32_t i = 3; i < count; i += 2)
          {
            DP_ASSERT((indices[i - 3] != pri) && (indices[i - 2] != pri) && (indices[i - 1] != pri) && (indices[i] != pri));
            calculateTangent(vas, tc, indices[i - 3], indices[i - 2], indices[i - 1], tangents);
            calculateTangent(vas, tc, indices[i - 2], indices[i], indices[i - 3], tangents);
            calculateTangent(vas, tc, indices[i - 1], indices[i - 3], indices[i], tangents);
            calculateTangent(vas, tc, indices[i], indices[i - 1], indices[i - 2], tangents);
            if ((i + 1 < count) && (indices[i + 1] == pri))
            {
              i += 3;
            }
          }
        }
      };

      void Primitive::calculateTangentsQuadStrip( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc, std::vector<Vec3f> & tangents )
      {
        unsigned int offset = getElementOffset();
        unsigned int count  = getElementCount();

        if ( m_indexSet )
        {
          dispatchCalculateTangents<calculateTangentsQuadStripWithIndices>(vas, tc, m_indexSet, count, offset, tangents);
        }
        else
        {
          for ( unsigned int i=offset+3 ; i<offset+count ; i+=2 )
          {
            calculateTangent( vas, tc, i-3, i-2, i-1, tangents );
            calculateTangent( vas, tc, i-2, i, i-3, tangents );
            calculateTangent( vas, tc, i-1, i-3, i, tangents );
            calculateTangent( vas, tc, i, i-1, i-2, tangents );
          }
        }
      }


      struct calculateTangentsTriangleWithIndices
      {
        template <typename IndexType>
        void eval(VertexAttributeSetSharedPtr const& vas,
                  VertexAttributeSet::AttributeID tc,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & tangents)
        {
          unsigned int pri = indexSet->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);

          for (uint32_t i = 2; i < count; i += 3)
          {
            DP_ASSERT((indices[i - 2] != pri) && (indices[i - 1] != pri) && (indices[i] != pri));
            calculateTangent(vas, tc, indices[i - 2], indices[i - 1], indices[i], tangents);
            calculateTangent(vas, tc, indices[i - 1], indices[i], indices[i - 2], tangents);
            calculateTangent(vas, tc, indices[i], indices[i - 2], indices[i - 1], tangents);
          }
        }
      };

      void Primitive::calculateTangentsTriangle( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc, std::vector<Vec3f> & tangents )
      {
        unsigned int offset = getElementOffset();
        unsigned int count  = getElementCount();
        DP_ASSERT( count % 3 == 0 );

        if ( m_indexSet )
        {
          dispatchCalculateTangents<calculateTangentsTriangleWithIndices>(vas, tc, m_indexSet, count, offset, tangents);
        }
        else
        {
          for ( unsigned int i=offset+2 ; i<offset+count ; i+=3 )
          {
            calculateTangent( vas, tc, i-2, i-1, i, tangents );
            calculateTangent( vas, tc, i-1, i, i-2, tangents );
            calculateTangent( vas, tc, i, i-2, i-1, tangents );
          }
        }
      }


      struct calculateTangentsTriFanWithIndices
      {
        template <typename IndexType>
        void eval(VertexAttributeSetSharedPtr const& vas,
                  VertexAttributeSet::AttributeID tc,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & tangents)
        {
          // calculate the tangents for each facet
          unsigned int pri = indexSet->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);

          for (uint32_t i = 2, j = 0; i < count; i++)
          {
            if (indices[i] == pri)
            {
              j = i + 1;
              i += 2;
            }
            else
            {
              DP_ASSERT((indices[j] != pri) && (indices[i - 1] != pri));
              calculateTangent(vas, tc, indices[j], indices[i - 1], indices[i], tangents);
              calculateTangent(vas, tc, indices[i - 1], indices[i], indices[j], tangents);
              calculateTangent(vas, tc, indices[i], indices[j], indices[i - 1], tangents);
            }
          }
        }
      };

      void Primitive::calculateTangentsTriFan( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc, std::vector<Vec3f> & tangents )
      {
        unsigned int offset = getElementOffset();
        unsigned int count  = getElementCount();

        if ( m_indexSet )
        {
          dispatchCalculateTangents<calculateTangentsTriFanWithIndices>(vas, tc, m_indexSet, count, offset, tangents);
        }
        else
        {
          for ( unsigned int i=offset+2 ; i<offset+count ; i++ )
          {
            calculateTangent( vas, tc, 0, i-1, i, tangents );
            calculateTangent( vas, tc, i-1, i, 0, tangents );
            calculateTangent( vas, tc, i, 0, i-1, tangents );
          }
        }
      }


      struct calculateTangentsTriStripWithIndices
      {
        template <typename IndexType>
        void eval(VertexAttributeSetSharedPtr const& vas,
                  VertexAttributeSet::AttributeID tc,
                  dp::sg::core::IndexSetSharedPtr const& indexSet,
                  uint32_t count,
                  uint32_t offset,
                  std::vector<Vec3f> & tangents)
        {
          // calculate the tangents for each facet
          unsigned int pri = indexSet->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<IndexType> indices(indexSet, offset);

          DP_ASSERT((indices[0] != pri) && (indices[1] != pri));
          for (uint32_t i = 2, j = 0; i < count; i++, j++)
          {
            if (indices[i] == pri)
            {
              i += 2;   // advance to end of first triangle in next strip
              j = ~0;   // reset j such that it's zero on next iteration
            }
            else
            {
              DP_ASSERT((indices[i - 2] != pri) && (indices[i - 1] != pri));
              if (j & 1)
              { // odd triangle
                calculateTangent(vas, tc, indices[i - 2], indices[i], indices[i - 1], tangents);
                calculateTangent(vas, tc, indices[i - 1], indices[i - 2], indices[i], tangents);
                calculateTangent(vas, tc, indices[i], indices[i - 1], indices[i - 2], tangents);
              }
              else
              { // even triangle
                calculateTangent(vas, tc, indices[i - 2], indices[i - 1], indices[i], tangents);
                calculateTangent(vas, tc, indices[i - 1], indices[i], indices[i - 2], tangents);
                calculateTangent(vas, tc, indices[i], indices[i - 2], indices[i - 1], tangents);
              }
            }
          }
        }
      };

      void Primitive::calculateTangentsTriStrip( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc
                                               , std::vector<Vec3f> & tangents )
      {
        unsigned int offset = getElementOffset();
        unsigned int count  = getElementCount();

        if ( m_indexSet )
        {
          dispatchCalculateTangents<calculateTangentsTriStripWithIndices>(vas, tc, m_indexSet, count, offset, tangents);
        }
        else
        {
          for ( unsigned int i=offset+2, j=0 ; i<count ; i++, j++ )
          {
            if ( j & 1 )
            { // odd triangle
              calculateTangent( vas, tc, i-2, i, i-1, tangents );
              calculateTangent( vas, tc, i-1, i-2, i, tangents );
              calculateTangent( vas, tc, i, i-1, i-2, tangents );
            }
            else
            { // even triangle
              calculateTangent( vas, tc, i-2, i-1, i, tangents );
              calculateTangent( vas, tc, i-1, i, i-2, tangents );
              calculateTangent( vas, tc, i, i-2, i-1, tangents );
            }
          }
        }
      }

      void Primitive::generateTangentSpace( VertexAttributeSet::AttributeID tc, VertexAttributeSet::AttributeID tg, VertexAttributeSet::AttributeID bn, bool overwrite )
      {
        calculateTangentSpace( tc, tg, bn, overwrite );
      }

      template <typename IndexType>
      void initTangentSpace(IndexSetSharedPtr const& indexSet,
                            uint32_t count,
                            uint32_t offset,
                            std::vector<dp::math::Vec3f> &tangents)
      {
        IndexSet::ConstIterator<IndexType> indices(indexSet, offset);
        for (uint32_t i = 0; i < count; i++)
        {
          tangents[indices[i]] = Vec3f(0.0f, 0.0f, 0.0f);
        }
      }

      template <typename IndexType>
      void normalizeTangentsAndComputeBinormals(IndexSetSharedPtr const& indexSet,
                                              uint32_t count,
                                              uint32_t offset,
                                              Buffer::ConstIterator<Vec3f>::Type normals,
                                              std::vector<dp::math::Vec3f> &tangents,
                                              std::vector<dp::math::Vec3f> &binormals)
      {
        IndexSet::ConstIterator<IndexType> indices(indexSet, offset);
        for (uint32_t i = 0; i < count; i++)
        {
          IndexType idx = indices[i];
          tangents[idx].normalize();
          tangents[idx].orthonormalize(normals[idx]);
          //  the binormal is orthogonal to the normal and the tangent
          binormals[idx] = normals[idx] ^ tangents[idx];
        }
      }

      void Primitive::calculateTangentSpace( VertexAttributeSet::AttributeID tc, VertexAttributeSet::AttributeID tg, VertexAttributeSet::AttributeID bn, bool overwrite )
      {
        DP_ASSERT( ( tc != tg ) && ( tc != bn ) && ( tg != bn ) );
        DP_ASSERT( getVertexAttributeSet() );
        DP_ASSERT( m_vertexAttributeSet->getNumberOfVertices() );
        DP_ASSERT( m_vertexAttributeSet->getNumberOfNormals() );
        DP_ASSERT( m_vertexAttributeSet->getNumberOfVertexData( tc ) );
        DP_ASSERT( m_vertexAttributeSet->getSizeOfVertexData( tc ) >= 2 );
        DP_ASSERT( m_vertexAttributeSet->getTypeOfVertexData( tc ) == dp::DataType::FLOAT_32 );
        DP_ASSERT(   ( m_vertexAttributeSet->getNumberOfVertices() == m_vertexAttributeSet->getNumberOfNormals() )
                 &&  ( m_vertexAttributeSet->getNumberOfVertices() == m_vertexAttributeSet->getNumberOfVertexData(tc) ) );

        bool ok =   ( overwrite || (!m_vertexAttributeSet->getNumberOfVertexData( tg ) && !m_vertexAttributeSet->getNumberOfVertexData( bn ) ) )
                &&  (   ( m_primitiveType == PrimitiveType::TRIANGLE_STRIP )
                    ||  ( m_primitiveType == PrimitiveType::TRIANGLE_FAN )
                    ||  ( m_primitiveType == PrimitiveType::TRIANGLES )
                    ||  ( m_primitiveType == PrimitiveType::QUAD_STRIP )
                    ||  ( m_primitiveType == PrimitiveType::QUADS ) );
        if ( ok )
        {
          // temporary tangents buffer
          std::vector<Vec3f> tangents;
          if ( m_vertexAttributeSet->getNumberOfVertexData( tg ) )
          {
            DP_ASSERT( m_vertexAttributeSet->getNumberOfVertices() == m_vertexAttributeSet->getNumberOfVertexData( tg ) );
            // get the current tangents, to prevent modifying tangents that are not used in this Primitive
            tangents.assign( (const Vec3f *) m_vertexAttributeSet->getVertexData( tg ).getPtr()
                           , (const Vec3f *) m_vertexAttributeSet->getVertexData( tg ).getPtr() + m_vertexAttributeSet->getNumberOfVertexData( tg ) );
          }
          else
          {
            // just resize the vector of tangents to hold the needed number of tangents
            tangents.resize( m_vertexAttributeSet->getNumberOfVertices() );
          }

          // initialize used tangents with zero
          unsigned int offset = getElementOffset();
          unsigned int count  = getElementCount();
          if ( m_indexSet )
          {
            switch (m_indexSet->getIndexDataType())
            {
            case dp::DataType::INT_8:
              initTangentSpace<int8_t>(m_indexSet, count, offset, tangents);
              break;
            case dp::DataType::INT_16:
              initTangentSpace<int16_t>(m_indexSet, count, offset, tangents);
              break;
            case dp::DataType::INT_32:
              initTangentSpace<int32_t>(m_indexSet, count, offset, tangents);
              break;
            case dp::DataType::UNSIGNED_INT_8:
              initTangentSpace<uint8_t>(m_indexSet, count, offset, tangents);
              break;
            case dp::DataType::UNSIGNED_INT_16:
              initTangentSpace<uint16_t>(m_indexSet, count, offset, tangents);
              break;
            case dp::DataType::UNSIGNED_INT_32:
              initTangentSpace<uint32_t>(m_indexSet, count, offset, tangents);
              break;
            default:
              DP_ASSERT(!"unsupported datatype");
            }
          }
          else
          {
            for ( unsigned int i=offset ; i<offset + count ; i++ )
            {
              tangents[i] = Vec3f( 0.0f, 0.0f, 0.0f );
            }
          }

          // calculate the normals, depending on primitive type
          switch( m_primitiveType )
          {
            case PrimitiveType::TRIANGLE_STRIP :
              calculateTangentsTriStrip( m_vertexAttributeSet, tc, tangents );
              break;
            case PrimitiveType::TRIANGLE_FAN :
              calculateTangentsTriFan( m_vertexAttributeSet, tc, tangents );
              break;
            case PrimitiveType::TRIANGLES :
              calculateTangentsTriangle( m_vertexAttributeSet, tc, tangents );
              break;
            case PrimitiveType::QUAD_STRIP :
              calculateTangentsQuadStrip( m_vertexAttributeSet, tc, tangents );
              break;
            case PrimitiveType::QUADS :
              calculateTangentsQuad( m_vertexAttributeSet, tc, tangents );
              break;
            default :
              DP_ASSERT( !"Tangent space calculation not implemented for this Primitive type" );
              break;
          }

          // temporary binormals buffer
          std::vector<Vec3f> binormals;
          if ( m_vertexAttributeSet->getNumberOfVertexData( bn ) )
          {
            DP_ASSERT( m_vertexAttributeSet->getNumberOfVertices() == m_vertexAttributeSet->getNumberOfVertexData( bn ) );
            // get the current binormals, to prevent modifying binormals that are not used in this Primitive
            binormals.assign( (const Vec3f *)m_vertexAttributeSet->getVertexData( bn ).getPtr()
                            , (const Vec3f *)m_vertexAttributeSet->getVertexData( bn ).getPtr() + m_vertexAttributeSet->getNumberOfVertexData( bn ) );
          }
          else
          {
            // just resize the vector of tangents to hold the needed number of tangents
            binormals.resize( m_vertexAttributeSet->getNumberOfVertices() );
          }

          //  normalize the tangents calculated above, orthonormalize them with the normals, calculate the binormals
          Buffer::ConstIterator<Vec3f>::Type normals = m_vertexAttributeSet->getNormals();
          if ( m_indexSet )
          {
            // calculate the normals, depending on primitive type
            switch (m_indexSet->getIndexDataType())
            {
            case dp::DataType::INT_8:
              normalizeTangentsAndComputeBinormals<int8_t>(m_indexSet, count, offset, normals, tangents, binormals);
              break;
            case dp::DataType::INT_16:
              normalizeTangentsAndComputeBinormals<int16_t>(m_indexSet, count, offset, normals, tangents, binormals);
              break;
            case dp::DataType::INT_32:
              normalizeTangentsAndComputeBinormals<int32_t>(m_indexSet, count, offset, normals, tangents, binormals);
              break;
            case dp::DataType::UNSIGNED_INT_8:
              normalizeTangentsAndComputeBinormals<uint8_t>(m_indexSet, count, offset, normals, tangents, binormals);
              break;
            case dp::DataType::UNSIGNED_INT_16:
              normalizeTangentsAndComputeBinormals<uint16_t>(m_indexSet, count, offset, normals, tangents, binormals);
              break;
            case dp::DataType::UNSIGNED_INT_32:
              normalizeTangentsAndComputeBinormals<uint32_t>(m_indexSet, count, offset, normals, tangents, binormals);
              break;
            default:
              DP_ASSERT(!"unsupported datatype");
            }
          }
          else
          {
            for ( size_t i=offset ; i<offset+count ; i++ )
            {
              tangents[i].normalize();
              tangents[i].orthonormalize( normals[i] );
              //  the binormal is orthogonal to the normal and the tangent
              binormals[i] = normals[i] ^ tangents[i];
            }
          }

          // put tangents and binormals into right slots
          m_vertexAttributeSet->setVertexData( tg, 3, dp::DataType::FLOAT_32, &tangents[0], 0, m_vertexAttributeSet->getNumberOfVertices() );
          m_vertexAttributeSet->setVertexData( bn, 3, dp::DataType::FLOAT_32, &binormals[0], 0, m_vertexAttributeSet->getNumberOfVertices() );
          // enable
          m_vertexAttributeSet->setEnabled( tg, true );
          m_vertexAttributeSet->setEnabled( bn, true );
        }
      }

      PrimitiveType primitiveNameToType( std::string const& name )
      {
        static const std::map<std::string, PrimitiveType> primitiveTypes =
        {
          { "Points",                 PrimitiveType::POINTS                    },
          { "LineStrip",              PrimitiveType::LINE_STRIP                },
          { "LineLoop",               PrimitiveType::LINE_LOOP                 },
          { "Lines",                  PrimitiveType::LINES                     },
          { "TriangleStrip",          PrimitiveType::TRIANGLE_STRIP            },
          { "TriangleFan",            PrimitiveType::TRIANGLE_FAN              },
          { "Triangles",              PrimitiveType::TRIANGLES                 },
          { "QuadStrip",              PrimitiveType::QUAD_STRIP                },
          { "Quads",                  PrimitiveType::QUADS                     },
          { "Polygon",                PrimitiveType::POLYGON                   },
          { "TrianglesAdjacency",     PrimitiveType::TRIANGLES_ADJACENCY       },
          { "TriangleStripAdjacency", PrimitiveType::TRIANGLE_STRIP_ADJACENCY  },
          { "LinesAdjacency",         PrimitiveType::LINES_ADJACENCY           },
          { "LineStripAdjacency",     PrimitiveType::LINE_STRIP_ADJACENCY      },
          { "Patches",                PrimitiveType::PATCHES                   }
        };

          std::map<std::string,PrimitiveType>::const_iterator it = primitiveTypes.find( name );
          DP_ASSERT( it != primitiveTypes.end() );
          return( it->second );
      }

      std::string primitiveTypeToName( PrimitiveType pt )
      {
        switch( pt )
        {
          case PrimitiveType::POINTS :                   return( "Points" );
          case PrimitiveType::LINE_STRIP :               return( "LineStrip" );
          case PrimitiveType::LINE_LOOP :                return( "LineLoop" );
          case PrimitiveType::LINES :                    return( "Lines" );
          case PrimitiveType::TRIANGLE_STRIP :           return( "TriangleStrip" );
          case PrimitiveType::TRIANGLE_FAN :             return( "TriangleFan" );
          case PrimitiveType::TRIANGLES :                return( "Triangles" );
          case PrimitiveType::QUAD_STRIP :               return( "QuadStrip" );
          case PrimitiveType::QUADS :                    return( "Quads" );
          case PrimitiveType::POLYGON :                  return( "Polygon" );
          case PrimitiveType::TRIANGLES_ADJACENCY :      return( "TrianglesAdjacency" );
          case PrimitiveType::TRIANGLE_STRIP_ADJACENCY : return( "TriangleStripAdjacency" );
          case PrimitiveType::LINES_ADJACENCY :          return( "LinesAdjacency" );
          case PrimitiveType::LINE_STRIP_ADJACENCY :     return( "LineStripAdjacency" );
          case PrimitiveType::PATCHES :                  return( "Patches" );
          default :
            DP_ASSERT( !"unknown primtive type" );
            return( "" );
        }
      }

      PatchesType patchesNameToType( std::string const& name )
      {
        static const std::map<std::string, PatchesType> patchesTypes =
        {
          { "NoPatches",            PatchesType::NONE                   },
          { "PNTriangles",          PatchesType::PN_TRIANGLES           },
          { "PNQuads",              PatchesType::PN_QUADS               },
          { "CubicBezierTriangles", PatchesType::CUBIC_BEZIER_TRIANGLES },
          { "CubicBezierQuads",     PatchesType::CUBIC_BEZIER_QUADS     }
        };

          std::map<std::string,PatchesType>::const_iterator it = patchesTypes.find( name );
          DP_ASSERT( it != patchesTypes.end() );
          return( it->second );
      }

      std::string patchesTypeToName( PatchesType pt )
      {
        switch( pt )
        {
          case PatchesType::NONE :                    return( "NoPatches" );
          case PatchesType::PN_TRIANGLES :            return( "PNTriangles" );
          case PatchesType::PN_QUADS :                return( "PNQuads" );
          case PatchesType::CUBIC_BEZIER_TRIANGLES :  return( "CubicBezierTriangles" );
          case PatchesType::CUBIC_BEZIER_QUADS :      return( "CubicBezierQuads" );
          default :
            DP_ASSERT( !"unknown patches type!" );
            return( "" );
        }
      }

      unsigned int verticesPerPatch( PatchesType pt )
      {
        switch( pt )
        {
          case PatchesType::NONE :                    return( 0 );
          case PatchesType::PN_TRIANGLES :            return( 3 );
          case PatchesType::PN_QUADS :                return( 4 );
          case PatchesType::CUBIC_BEZIER_TRIANGLES :  return( 10 );
          case PatchesType::CUBIC_BEZIER_QUADS :      return( 16 );
          default :
            DP_ASSERT( !"unknown patches type!" );
            return( 0 );
        }
      }

    } // namespace core
  } // namespace sg
} // namespace dp
