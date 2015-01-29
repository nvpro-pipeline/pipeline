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


#include <dp/util/BitArray.h>
#include <stdexcept>

namespace dp
{
  namespace util
  {

    /** \brief Create a new BitVector.
    **/
    BitArray::BitArray()
      : m_size( 0 )
      , m_bits( nullptr )
    {
    }

    /** \brief Create a new BitVector with all bits set to false
        \param size Number of Bits in the Array
    **/
    BitArray::BitArray( size_t size )
      : m_size( size )
      , m_bits( new BitStorageType[determineNumberOfElements()] )
    {
      clear();
    }

    BitArray::BitArray( const BitArray &rhs )
      : m_size( rhs.m_size )
      , m_bits( new BitStorageType[determineNumberOfElements()] )
    {
      std::copy( rhs.m_bits.get(), rhs.m_bits.get() + determineNumberOfElements(), m_bits.get() );
    }

    BitArray::~BitArray()
    {
    }

    void BitArray::resize( size_t newSize, bool defaultValue)
    {
      // if the default value for the new bits is true enabled the unused bits in the last element.
      if (defaultValue) {
        setUnusedBits();
      }

      size_t oldNumberOfElements = determineNumberOfElements();
      size_t oldSize = m_size;
      m_size = newSize;
      size_t newNumberOfElements = determineNumberOfElements();

      // the number of elements has changed, reallocate array
      if ( oldNumberOfElements != newNumberOfElements )
      {
        std::unique_ptr<BitStorageType[]> newBits(new BitStorageType[newNumberOfElements]);
        if ( newNumberOfElements < oldNumberOfElements )
        {
          std::copy( m_bits.get(), m_bits.get() + newNumberOfElements, newBits.get() );
        }
        else
        {
          std::copy(m_bits.get(), m_bits.get() + oldNumberOfElements, newBits.get());
          std::fill(newBits.get() + oldNumberOfElements, newBits.get() + newNumberOfElements, defaultValue ? ~BitStorageType(0) : BitStorageType(0));
        }
        m_bits.swap( newBits );
      }
      clearUnusedBits();
    }

    BitArray& BitArray::operator=( const BitArray &rhs )
    {
      if ( m_size != rhs.m_size )
      {
        m_size = rhs.m_size;
        m_bits.reset(new BitStorageType[determineNumberOfElements()]);
      }
      std::copy( rhs.m_bits.get(), rhs.m_bits.get() + determineNumberOfElements(), m_bits.get() );

      return *this;
    }

    bool BitArray::operator==( const BitArray& rhs )
    {
      return (m_size == rhs.m_size) ? std::equal( m_bits.get(), m_bits.get() + determineNumberOfElements(), rhs.m_bits.get() ) : false;
    }
    
    BitArray BitArray::operator^( BitArray const & rhs )
    {
      if ( getSize() != rhs.getSize() )
      {
        throw std::runtime_error( "BitArrays have different size!" );
      }

      BitArray result( getSize() );
      for ( size_t index = 0;index < determineNumberOfElements(); ++index )
      {
        result.m_bits[index] = m_bits[index] ^ rhs.m_bits[index];
      }
      clearUnusedBits();

      return result;
    }

    BitArray BitArray::operator|( BitArray const & rhs )
    {
      if ( getSize() != rhs.getSize() )
      {
        throw std::runtime_error( "BitArrays have different size!" );
      }

      BitArray result( getSize() );
      for ( size_t index = 0;index < determineNumberOfElements(); ++index )
      {
        result.m_bits[index] = m_bits[index] | rhs.m_bits[index];
      }
      clearUnusedBits();

      return result;
    }

    BitArray BitArray::operator&( BitArray const & rhs )
    {
      if ( getSize() != rhs.getSize() )
      {
        throw std::runtime_error( "BitArrays have different size!" );
      }

      BitArray result( getSize() );
      for ( size_t index = 0;index < determineNumberOfElements(); ++index )
      {
        result.m_bits[index] = m_bits[index] & rhs.m_bits[index];
      }
      clearUnusedBits();

      return result;
    }

    BitArray & BitArray::operator^=( BitArray const & rhs )
    {
      if ( getSize() != rhs.getSize() )
      {
        throw std::runtime_error( "BitArrays have different size!" );
      }

      for ( size_t index = 0;index < determineNumberOfElements(); ++index )
      {
        m_bits[index] ^= rhs.m_bits[index];
      }
      clearUnusedBits();

      return *this;
    }

    BitArray & BitArray::operator|=( BitArray const & rhs )
    {
      if ( getSize() != rhs.getSize() )
      {
        throw std::runtime_error( "BitArrays have different size!" );
      }

      for ( size_t index = 0;index < determineNumberOfElements(); ++index )
      {
        m_bits[index] |= rhs.m_bits[index];
      }

      return *this;
    }

    BitArray & BitArray::operator&=( BitArray const & rhs )
    {
      if ( getSize() != rhs.getSize() )
      {
        throw std::runtime_error( "BitArrays have different size!" );
      }

      for ( size_t index = 0;index < determineNumberOfElements(); ++index )
      {
        m_bits[index] &= rhs.m_bits[index];
      }

      return *this;
    }

    void BitArray::clear()
    {
      std::fill( m_bits.get(), m_bits.get() + determineNumberOfElements(), 0 );
    }

    void BitArray::fill()
    {
      if ( determineNumberOfElements() )
      {
        std::fill( m_bits.get(), m_bits.get() + determineNumberOfElements(), ~0 );

        clearUnusedBits();
      }
    }

    size_t BitArray::countLeadingZeroes() const
    {
      size_t index = 0;
      
      // first count 
      while (index < determineNumberOfElements() && !m_bits[index])
      {
        ++index;
      }
      size_t leadingZeroes = index * StorageBitsPerElement;
      if (index < determineNumberOfElements())
      {
        leadingZeroes += clz(m_bits[index]);
      }

      return std::min(leadingZeroes, getSize());
    }

  } // namespace util
} // namespace dp
