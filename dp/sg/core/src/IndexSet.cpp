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


#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/IndexSet.h>
#include <dp/sg/core/BufferHost.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      BEGIN_REFLECTION_INFO( IndexSet )
        DERIVE_STATIC_PROPERTIES( IndexSet, Object );
      END_REFLECTION_INFO

      IndexSetSharedPtr IndexSet::create()
      {
        return( std::shared_ptr<IndexSet>( new IndexSet() ) );
      }

      HandledObjectSharedPtr IndexSet::clone() const
      {
        return( std::shared_ptr<IndexSet>( new IndexSet( *this ) ) );
      }

      IndexSet::IndexSet()
      : m_dataType( dp::DT_UNSIGNED_INT_32 )
      , m_primitiveRestartIndex(~0)
      , m_numberOfIndices(0)
      {
        m_bufferObserver.setIndexSet( this );
        m_objectCode = OC_INDEX_SET;
      }

      IndexSet::IndexSet(const IndexSet& rhs)
      : m_dataType(rhs.m_dataType)
      , m_numberOfIndices(rhs.m_numberOfIndices)
      , m_primitiveRestartIndex(rhs.m_primitiveRestartIndex)
      , m_buffer(rhs.m_buffer)
      {
        m_bufferObserver.setIndexSet( this );
        m_objectCode = OC_INDEX_SET;
        if ( m_buffer )
        {
          m_buffer->attach( &m_bufferObserver );
        }
      }

      IndexSet::~IndexSet()
      {
        if ( m_buffer )
        {
          m_buffer->detach( &m_bufferObserver );
        }
      }

      IndexSet & IndexSet::operator=( const IndexSet & rhs )
      {
        if ( this != &rhs )
        {
          m_dataType = rhs.m_dataType;
          m_numberOfIndices = rhs.m_numberOfIndices;
          m_primitiveRestartIndex = rhs.m_primitiveRestartIndex;
          if( m_buffer )
          {
            m_buffer->detach( &m_bufferObserver );
          }
          m_buffer = rhs.m_buffer;
          if ( m_buffer )
          {
            m_buffer->attach( &m_bufferObserver );
          }
        }
        return( *this );
      }

      void IndexSet::setIndexDataType( dp::DataType type )
      {
        DP_ASSERT( (type == dp::DT_UNSIGNED_INT_32) || (type == dp::DT_UNSIGNED_INT_16) || (type == dp::DT_UNSIGNED_INT_8) );
        if ( m_dataType != type )
        {
          m_dataType = type;
          notify( Event( this ) );
        }
      }

      void IndexSet::setPrimitiveRestartIndex( unsigned int pri )
      {
        if ( m_primitiveRestartIndex != pri )
        {
          m_primitiveRestartIndex = pri;
          notify( Event( this ) );
        }
      }

      void IndexSet::copyDataToBuffer( const void * ptr, unsigned int count )
      {
        size_t typeScale = dp::getSizeOf( m_dataType );

        if( m_buffer == 0 )
        {
          m_buffer = BufferHost::create();
          m_buffer->attach( &m_bufferObserver );
        }

        DP_ASSERT( m_buffer );

        size_t byteSize = count * typeScale;

        {
          m_buffer->setSize( byteSize );
          m_buffer->setData( 0, byteSize, ptr );
        }
      }

      void IndexSet::setData( const void * indices, unsigned int count, dp::DataType dataType, unsigned int primitiveRestartIndex )
      {
        setIndexDataType( dataType );

        copyDataToBuffer( indices, count );

        m_numberOfIndices = count;
        setPrimitiveRestartIndex( primitiveRestartIndex );
      }

      void IndexSet::setData( const unsigned int   * indices, unsigned int count, unsigned int primitiveRestartIndex )
      {
        setData( indices, count, dp::DT_UNSIGNED_INT_32, primitiveRestartIndex );
      }

      void IndexSet::setData( const unsigned short * indices, unsigned int count, unsigned int primitiveRestartIndex )
      {
        setData( indices, count, dp::DT_UNSIGNED_INT_16, primitiveRestartIndex );
      }

      void IndexSet::setData( const unsigned char  * indices, unsigned int count, unsigned int primitiveRestartIndex )
      {
        setData( indices, count, dp::DT_UNSIGNED_INT_8, primitiveRestartIndex );
      }

      bool IndexSet::getData( void * destination ) const
      {
        DP_ASSERT( destination );

        bool result = false;

        if( m_buffer )
        {
          size_t sizeInBytes = ( m_dataType == dp::DT_UNSIGNED_INT_32 ? 4 : m_dataType == dp::DT_UNSIGNED_INT_16 ? 2 : 1 ) * getNumberOfIndices();

          m_buffer->getData( 0, sizeInBytes, destination );
          result = true;
        }

        return result;
      }

      void IndexSet::setBuffer( const BufferSharedPtr &buffer, unsigned int count, dp::DataType type, unsigned int primitiveRestartIndex )
      {
        m_numberOfIndices = count;
        setPrimitiveRestartIndex( primitiveRestartIndex );
        setIndexDataType( type );

        if ( m_buffer != buffer )
        {
          if ( m_buffer )
          {
            m_buffer->detach( &m_bufferObserver );
          }

          m_buffer = buffer;

          if ( m_buffer )
          {
            m_buffer->attach( &m_bufferObserver );
          }
          notify( Event( this ) );
        }
      }

      void IndexSet::onBufferChanged( )
      {
        notify( Event( this ) );
      }

      void IndexSet::feedHashGenerator( util::HashGenerator & hg ) const
      {
        hg.update( reinterpret_cast<const unsigned char *>(&m_dataType), sizeof(m_dataType) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_numberOfIndices), sizeof(m_numberOfIndices) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_primitiveRestartIndex), sizeof(m_primitiveRestartIndex) );
        // TODO: should we add hash key handling to Buffer ?
        if ( m_buffer )
        {
          hg.update( reinterpret_cast<const unsigned char *>(m_buffer.getWeakPtr()), sizeof(const Buffer *) );
        }
      }

      bool IndexSet::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<IndexSet>() && Object::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          IndexSetSharedPtr const& is = object.staticCast<IndexSet>();

          equi =    m_dataType              == is->m_dataType
                &&  m_numberOfIndices       == is->m_numberOfIndices
                &&  m_primitiveRestartIndex == is->m_primitiveRestartIndex
                &&  (!!m_buffer)            == (!!is->m_buffer);    // make sure buffer status of both is the same

          if ( equi && m_buffer )
          {
            equi = ( m_buffer == is->m_buffer );

            if ( !equi && deepCompare )
            {
              unsigned int numBytes = dp::checked_cast<unsigned int>( dp::getSizeOf( m_dataType ) * m_numberOfIndices );
              Buffer::DataReadLock rhsBuffer( m_buffer );
              Buffer::DataReadLock lhsBuffer( is->m_buffer );
              equi = ( memcmp( rhsBuffer.getPtr(), lhsBuffer.getPtr(), numBytes ) == 0 );
            }
          }
        }
        return( equi );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
