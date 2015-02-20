// Copyright NVIDIA Corporation 2012-2015
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


#pragma once

#include <dp/Types.h>
#include <algorithm>
#include <cstring>
#include <memory>

namespace dp
{
  namespace util
  {

    /** \brief Visitor which forwards the visitor operator with a fixed offset **/
    template <typename Visitor>
    struct OffsetVisitor
    {
      inline OffsetVisitor( Visitor& visitor, size_t offset )
        : m_visitor( visitor )
        , m_offset( offset)
      {

      }

      inline void operator()( size_t index )
      {
        m_visitor( index + m_offset );
      }
    private:
      Visitor& m_visitor;
      size_t   m_offset;
    };


#if (defined(DP_ARCH_X86) || defined(DP_ARCH_X86_64)) && defined(_MSC_VER)
    template <typename Visitor> inline void bitTraverse( dp::Uint32 bits, Visitor& visitor )
    {
      unsigned long localIndex;
      while ( _BitScanForward( &localIndex, bits ) )
      {
        visitor( localIndex );
        bits ^= 1 << localIndex; // clear the current bit so that the next one is being found by the bitscan
      }
    }

    template <typename Visitor> inline void bitTraverse( dp::Uint64 bits, Visitor& visitor )
    {
      unsigned long localIndex;
      while ( _BitScanForward64( &localIndex, bits ) )
      {
        visitor( localIndex );
        bits ^= dp::Uint64(1) << localIndex; // clear the current bit so that the next one is being found by the bitscan
      }
    }

    inline size_t clz(dp::Uint64 bits)
    {
      unsigned long localIndex;
      return _BitScanForward64(&localIndex,bits) ? localIndex : 64;
    }

    inline size_t clz(dp::Uint32 bits)
    {
      unsigned long localIndex;
      return _BitScanForward(&localIndex,bits) ? localIndex : 32;
    }
#else
    // TODO implement GCC version!
    template <typename BitType, typename Visitor> inline void bitTraverse( BitType bits, Visitor& visitor )
    {
      size_t index = 0;
      while (bits)
      {
        if ( bits & 0xff ) // skip ifs if the byte is 0
        {
          if ( bits & 0x01 ) visitor( index + 0 );
          if ( bits & 0x02 ) visitor( index + 1 );
          if ( bits & 0x04 ) visitor( index + 2 );
          if ( bits & 0x08 ) visitor( index + 3 );
          if ( bits & 0x10 ) visitor( index + 4 );
          if ( bits & 0x20 ) visitor( index + 5 );
          if ( bits & 0x40 ) visitor( index + 6 );
          if ( bits & 0x80 ) visitor( index + 7 );
        } 
        bits >>= 8;
        index += 8;
      }
    }
#endif

    /** \brief Call visitor(index) for each bit set **/
    template <typename BitType, typename Visitor> inline void bitTraverse( BitType* elements, size_t numberOfElements, Visitor& visitor )
    {
      size_t baseIndex = 0;
      for ( size_t elementIndex = 0; elementIndex < numberOfElements; ++elementIndex )
      {
        OffsetVisitor<Visitor> offsetVisitor( visitor, baseIndex );
        bitTraverse( elements[elementIndex], offsetVisitor );
        baseIndex += sizeof(*elements) * 8;
      }

    }

    class BitArray
    {
    public:
      typedef size_t BitStorageType;
      enum { StorageBitsPerElement = sizeof(BitStorageType) * 8 };

    public:

      /** \brief Create a new BitVector with all bits set to false
          \param size Number of bits in this Array
      **/
      DP_UTIL_API BitArray();
      DP_UTIL_API BitArray( size_t size );
      DP_UTIL_API BitArray( const BitArray & rhs );
      DP_UTIL_API ~BitArray();

      DP_UTIL_API BitArray & operator=( const BitArray & rhs );
      DP_UTIL_API bool operator==( const BitArray & rhs );
      DP_UTIL_API BitArray operator^( BitArray const & rhs );
      DP_UTIL_API BitArray operator&( BitArray const & rhs );
      DP_UTIL_API BitArray operator|( BitArray const & rhs );
      DP_UTIL_API BitArray & operator^=( BitArray const & rhs );
      DP_UTIL_API BitArray & operator&=( BitArray const & rhs );
      DP_UTIL_API BitArray & operator|=( BitArray const & rhs );

      DP_UTIL_API void clear();
      DP_UTIL_API void fill();

      /** \brief Change the number of bits in this array. The state of remaining bits is being kept.
                 New bits will be initialized to false.
          \param size New number of bits in this array
          \param defaultValue The new default value for the new bits
      **/
      DP_UTIL_API void resize( size_t size, bool defaultValue = false );

      size_t getSize() const { return m_size; }

      // inline functions
      void enableBit( size_t index );
      void disableBit( size_t index);
      void setBit( size_t index, bool value );
      bool getBit( size_t index ) const;

      BitStorageType const* getBits() const;

      template <typename T>
      void setBits( T const* bits, size_t numberOfBits );


      /** \brief call Visitor( size_t index ) on all bits which are set. **/
      template <typename Visitor> void traverseBits( Visitor &visitor );

      DP_UTIL_API size_t countLeadingZeroes() const;
      
    private:
      /** \brief Determine the element / bit for the given index **/
      void determineBitPosition( size_t index, size_t& element, size_t& bit ) const;
      size_t determineNumberOfElements( ) const;

      /** \brief Clear the last unused bits in the last element.
          \remarks Clear bits whose number is >= m_size. those are traversed unconditional and would produce invalid results.
                   restrict shifting range to 0 to StorageBitsPerElement - 1 to handle the case usedBitsInLastElement==0
                   which would result in shifting StorageBitsPerElement which is undefined by the standard and not the desired operation.
      **/
      void clearUnusedBits();

      /** \brief Set the last unused bits in the last element.
          \remarks Set bits whose number is >= m_size. This is required when expanding the vector with the bits set to true.
      **/
      void setUnusedBits();

      size_t  m_size;
      std::unique_ptr<BitStorageType[]> m_bits;
    };

    /** \brief Determine the element / bit for the given index **/
    inline void BitArray::determineBitPosition( size_t index, size_t& element, size_t& bit ) const
    {
      element = index / StorageBitsPerElement;
      bit = index % StorageBitsPerElement;
    }

    inline size_t BitArray::determineNumberOfElements( ) const
    {
      return (m_size + StorageBitsPerElement - 1) / StorageBitsPerElement;
    }

    inline void BitArray::enableBit( size_t index )
    {
      DP_ASSERT( index < m_size );
      size_t element;
      size_t bit;
      determineBitPosition(index, element, bit );
      m_bits[element] |= BitStorageType(1) << bit;
    }

    inline void BitArray::disableBit( size_t index)
    {
      DP_ASSERT( index < m_size );

      size_t element;
      size_t bit;
      determineBitPosition(index, element, bit );
      m_bits[element] &= ~(BitStorageType(1) << bit);
    }

    inline void BitArray::setBit( size_t index, bool value )
    {
      DP_ASSERT( index < m_size );
      if ( value )
      {
        enableBit( index );
      }
      else
      {
        disableBit( index );
      }
    }

    inline BitArray::BitStorageType const* BitArray::getBits() const
    {
      return m_bits.get();
    }

    inline bool BitArray::getBit( size_t index ) const
    {
      DP_ASSERT( index < m_size );
      size_t element;
      size_t bit;
      determineBitPosition(index, element, bit );
      return !!(m_bits[element] & (BitStorageType(1) << bit));
    }

    template <typename T>
    inline void BitArray::setBits( T const * bits, size_t numberOfBits )
    {
      size_t newNumberOfElements = (numberOfBits + StorageBitsPerElement - 1) / StorageBitsPerElement;
      if ( determineNumberOfElements() != newNumberOfElements )
      {
        m_bits.reset( new BitStorageType[newNumberOfElements] );
        m_size = numberOfBits;
      }
      size_t numberOfSourceElements = (numberOfBits + sizeof(*bits) * 8 - 1) / (sizeof(*bits) * 8);
      size_t numberOfSourceBytes = numberOfSourceElements * sizeof(*bits);
      size_t numberOfDestinationBytes = determineNumberOfElements() * sizeof(BitStorageType);

      memset( m_bits.get(), 0xff, numberOfDestinationBytes );
      memcpy( m_bits.get(), bits, std::min( numberOfSourceBytes, numberOfDestinationBytes ) );

      clearUnusedBits();
    }

    /** \brief call Visitor( size_t index ) on all bits which are set. **/
    template <typename Visitor>
    inline void BitArray::traverseBits( Visitor &visitor )
    {
      size_t index = 0; // index of bit
      bitTraverse(m_bits.get(), determineNumberOfElements(), visitor );
    }

    inline void BitArray::clearUnusedBits()
    {
      if ( m_size )
      {
        size_t usedBitsInLastElement = m_size % StorageBitsPerElement;
        m_bits[determineNumberOfElements() - 1] &= ~BitStorageType(0) >> ((StorageBitsPerElement - usedBitsInLastElement) & StorageBitsPerElement - 1);
      }
    }

    inline void BitArray::setUnusedBits()
    {
      if ( m_size )
      {
        size_t usedBitsInLastElement = m_size % StorageBitsPerElement;
        m_bits[determineNumberOfElements() - 1] |= ~BitStorageType(0) << usedBitsInLastElement;
      }
    }


  } // namespace util
} // namespace dp
