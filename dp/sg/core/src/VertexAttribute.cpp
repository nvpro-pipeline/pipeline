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


#include <dp/math/math.h>
#include <dp/math/Vecnt.h>
#include <dp/util/Memory.h>
#include <dp/sg/core/VertexAttribute.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/BufferHost.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      VertexAttribute::VertexAttribute()
      : m_size(0)
      , m_type(dp::DataType::UNKNOWN)
      , m_bytes(0)
      , m_strideInBytes(0)
      , m_offset(0)
      , m_count(0)
      {
      }

      VertexAttribute::VertexAttribute(const VertexAttribute& rhs)
      : m_size(rhs.m_size)
      , m_type(rhs.m_type)
      , m_bytes(rhs.m_bytes)
      , m_strideInBytes(rhs.m_strideInBytes)
      , m_offset(rhs.m_offset)
      , m_count(rhs.m_count)
      , m_buffer(rhs.m_buffer)
      {
      }

      VertexAttribute::~VertexAttribute()
      {
        removeData();
      }

      VertexAttribute & VertexAttribute::operator=( const VertexAttribute & rhs )
      {
        if ( this != &rhs )
        {
          m_count = rhs.m_count;
          m_size  = rhs.m_size;
          m_type  = rhs.m_type;
          m_bytes = rhs.m_bytes;
          m_strideInBytes = rhs.m_strideInBytes;
          m_offset = rhs.m_offset;
          m_buffer  = rhs.m_buffer;
        }
        return( *this );
      }

      void VertexAttribute::swapData(VertexAttribute& rhs)
      {
        if ( this != &rhs ) // never exchange same 
        {
          std::swap( m_buffer, rhs.m_buffer );
          std::swap( m_count, rhs.m_count );
          std::swap( m_size, rhs.m_size );
          std::swap( m_type, rhs.m_type );
          std::swap( m_bytes, rhs.m_bytes );
          std::swap( m_strideInBytes, rhs.m_strideInBytes );
          std::swap( m_offset, rhs.m_offset );
        }
      }

      void VertexAttribute::initData( unsigned int size, dp::DataType type )
      {
        m_size = size;
        m_type = type;
        m_bytes = dp::checked_cast<unsigned int>( m_size * dp::getSizeOf(type) );
      }

      void VertexAttribute::reserveData( unsigned int size, dp::DataType type, unsigned int count )
      {
        DP_ASSERT( size <= 4 );
        DP_ASSERT( ( dp::DataType::INT_8 <= type ) && ( type <= dp::DataType::NUM_DATATYPES ) );
        DP_ASSERT( !m_buffer || ( ( size == m_size ) && ( type == m_type ) ) );

        initData( size, type );
        m_strideInBytes = m_bytes;

        if ( !m_buffer )
        {
          m_buffer = BufferHost::create();
          m_offset = 0;
        }

        size_t newSize = m_strideInBytes * count + m_offset;
        if ( m_buffer->getSize() < newSize )
        {
          m_buffer->resize( newSize );
        }
      }

      void VertexAttribute::setData(unsigned int size, dp::DataType type, const void * data, unsigned int stride, unsigned int count)
      {
        DP_ASSERT( size <= 4 );
        DP_ASSERT( ( dp::DataType::INT_8 <= type ) && ( type <= dp::DataType::NUM_DATATYPES ) );
        DP_ASSERT( data );

        initData( size, type );
        m_count = count;

        m_strideInBytes = m_bytes;
        m_offset = 0;

        m_buffer = BufferHost::create();
        m_buffer->setSize( m_count * m_strideInBytes );
        Buffer::DataWriteLock lock( m_buffer, Buffer::MapMode::WRITE, m_offset, m_count * m_strideInBytes );
        dp::util::stridedMemcpy( lock.getPtr(), 0, m_strideInBytes, data, 0, stride ? stride : m_bytes, m_bytes, count );
      }

      void VertexAttribute::setData(unsigned int pos, unsigned int size, dp::DataType type, const void * data, unsigned int stride, unsigned int count)
      {
        DP_ASSERT(size <=4);
        DP_ASSERT(type>=dp::DataType::INT_8 && type <=dp::DataType::FLOAT_64);

        if ( !m_buffer )
        {
          m_buffer = BufferHost::create();
          initData( size, type );
          m_strideInBytes = m_bytes;
          m_offset = 0;
        }

        DP_ASSERT( ( size == m_size ) && ( type == m_type ) );

        unsigned int numBytes = count * m_bytes; // input byte count

        unsigned int dstOffset = 0;
        unsigned int newCount = m_count;
        if ( ( pos == ~0 ) || ( m_count == pos ) )
        {
          // this means: add vertex data to the very end
          dstOffset = m_count;
          newCount += count;
        }
        else if ( m_count < pos )
        {
          // copy outside of original range
          dstOffset = pos;
          newCount = pos + count;
        }
        else
        {
          // overwrite/copy the stuff, expand if necessary
          dstOffset = pos;
          newCount = std::max(pos + count, m_count);
        }

        if ( newCount != m_count )
        {
          size_t newSize = newCount * m_strideInBytes + m_offset;
          if ( m_buffer->getSize() < newSize )
          {
            m_buffer->resize( newSize );
          }
          m_count = newCount;
        }

        Buffer::DataWriteLock lock( m_buffer, Buffer::MapMode::WRITE, m_offset + dstOffset * m_strideInBytes, count * m_strideInBytes );
        util::stridedMemcpy( lock.getPtr(), 0, m_strideInBytes, data, 0, stride ? stride : m_bytes, m_bytes, count );
      }

      void VertexAttribute::setData(unsigned int size, dp::DataType type, const BufferSharedPtr &buffer, unsigned int offset, unsigned int strideInBytes, unsigned int count)
      {
        DP_ASSERT( size <= 4 );
        DP_ASSERT( ( dp::DataType::INT_8 <= type ) && ( type <= dp::DataType::NUM_DATATYPES ) );
        DP_ASSERT( buffer );

        initData( size, type );
        m_count = count;

        m_strideInBytes = strideInBytes ? strideInBytes : m_bytes;
        m_offset = offset;

        m_buffer = buffer;
      }

      void VertexAttribute::removeData()
      {
        m_buffer.reset();
        m_size = 0;
        m_type = dp::DataType::UNKNOWN;
        m_bytes = 0;
        m_count = 0;
        m_strideInBytes = 0;
        m_offset = 0;
      }

      void VertexAttribute::feedHashGenerator( util::HashGenerator & hg ) const
      {
        hg.update( reinterpret_cast<const unsigned char *>(&m_count), sizeof(m_count) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_size), sizeof(m_size) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_type), sizeof(m_type) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_bytes), sizeof(m_bytes) );
        hg.update(reinterpret_cast<const unsigned char *>(&m_offset), sizeof(m_offset));
        hg.update(reinterpret_cast<const unsigned char *>(&m_strideInBytes), sizeof(m_strideInBytes));
        if (m_buffer)
        {
          hg.update(m_buffer);
        }
      }

      template<typename T, unsigned char dim>
      void _normalize( VertexAttribute & va )
      {
        unsigned int n = va.getVertexDataCount();
        typename Buffer::Iterator<Vecnt<dim, T> >::Type it = va.begin< Vecnt<dim, T> >();
        for ( unsigned int i=0 ; i<n ; ++i )
        {
          normalize( *it );
          ++it;
        }
      }

      template<typename T>
      void _normalize( VertexAttribute & va )
      {
        switch( va.getVertexDataSize() )
        {
          case 1 :
            _normalize<T,1>( va );
            break;
          case 2 :
            _normalize<T,2>( va );
            break;
          case 3 :
            _normalize<T,3>( va );
            break;
          case 4 :
            _normalize<T,4>( va );
            break;
          default :
            DP_ASSERT( false );
            break;
        }
      }

      void normalize( VertexAttribute & va )
      {
        switch( va.getVertexDataType() )
        {
          case dp::DataType::FLOAT_32 :
            _normalize<float>( va );
            break;
          case dp::DataType::FLOAT_64 :
            _normalize<double>( va );
            break;
          default :
            DP_ASSERT( false );
            break;
        }
      }

    } // namespace core
  } // namespace sg

  namespace math
  {
    template<typename F>
    void __lerp( float alpha, const sg::core::VertexAttribute & va0, const sg::core::VertexAttribute & va1, sg::core::VertexAttribute & var )
    {
      unsigned int n = va0.getVertexDataCount();
      std::vector<F> vr(n);
      typename sg::core::Buffer::ConstIterator<F>::Type it0 = va0.beginRead<F>();
      typename sg::core::Buffer::ConstIterator<F>::Type it1 = va1.beginRead<F>();
      
      for ( unsigned int i=0 ; i<n ; ++i )
      {
        vr[i] = lerp<F>( alpha, *it0, *it1 );
        ++it0;
        ++it1;
      }

      var.setData( va0.getVertexDataSize(), va0.getVertexDataType(), &vr[0], 0, va0.getVertexDataCount() );
    }

    template <typename DataType>
    void _lerp( float alpha, const sg::core::VertexAttribute & va0, const sg::core::VertexAttribute & va1, sg::core::VertexAttribute & var )
    {
      switch ( va0.getVertexDataSize() )
      {
      case 1:
        __lerp<Vecnt<1, DataType> >( alpha, va0, va1, var );
      break;
      case 2:
        __lerp<Vecnt<2, DataType> >( alpha, va0, va1, var );
      break;
      case 3:
        __lerp<Vecnt<3, DataType> >( alpha, va0, va1, var );
      break;
      case 4:
        __lerp<Vecnt<4, DataType> >( alpha, va0, va1, var );
      break;
      default:
        DP_ASSERT( 0 && "invalid VertexDataSize" );
      }
    }

    void lerp( float alpha, const sg::core::VertexAttribute & va0, const sg::core::VertexAttribute & va1, sg::core::VertexAttribute & var )
    {
      DP_ASSERT( va0.getVertexDataCount() == va1.getVertexDataCount() );
      DP_ASSERT( va0.getVertexDataSize() == va1.getVertexDataSize() );
      DP_ASSERT( va0.getVertexDataType() == va1.getVertexDataType() );

      switch( va0.getVertexDataType() )
      {
        case dp::DataType::INT_8 :
          _lerp<char>( alpha, va0, va1, var );
          break;
        case dp::DataType::UNSIGNED_INT_8 :
          _lerp<unsigned char>( alpha, va0, va1, var );
          break;
        case dp::DataType::INT_16 :
          _lerp<short>( alpha, va0, va1, var );
          break;
        case dp::DataType::UNSIGNED_INT_16 :
          _lerp<unsigned short>( alpha, va0, va1, var );
          break;
        case dp::DataType::INT_32 :
          _lerp<int>( alpha, va0, va1, var );
          break;
        case dp::DataType::UNSIGNED_INT_32 :
          _lerp<unsigned int>( alpha, va0, va1, var );
          break;
        case dp::DataType::FLOAT_32 :
          _lerp<float>( alpha, va0, va1, var );
          break;
        case dp::DataType::FLOAT_64 :
          _lerp<double>( alpha, va0, va1, var );
          break;
        default :
          DP_ASSERT( false );
          break;
      }
    }

  } // namespace math
} // namespace dp

