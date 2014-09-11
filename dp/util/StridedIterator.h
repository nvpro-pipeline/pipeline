// Copyright NVIDIA Corporation 2010
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

#include <iterator>

namespace dp
{
  namespace util
  {

    struct DummyPayload {};

    /*******************/
    /* StridedIterator */
    /*******************/
    template <typename ValueType, typename Payload = DummyPayload>
    class StridedIterator : public std::iterator< std::input_iterator_tag, ValueType >
    {
    public:
      StridedIterator();
      StridedIterator( ValueType *basePtr, size_t strideInBytes, Payload = Payload() );
      StridedIterator( const StridedIterator &rhs );
      StridedIterator &operator=( const StridedIterator &rhs );
      virtual ~StridedIterator() {}

      ValueType &operator*();
      ValueType &operator[](size_t index);
      bool operator==(const StridedIterator &rhs) const; // compares only ptr!
      bool operator!=(const StridedIterator &rhs) const; // compares only ptr!
      StridedIterator operator+(int offset) const;
      StridedIterator &operator++();  // pre-increment
      StridedIterator operator++(int);  //post-increment

      typedef std::input_iterator_tag iterator_category;
      typedef ValueType value_type;
    protected:
      ValueType *m_basePtr;
      size_t     m_strideInBytes;
      Payload    m_payload;
    };

    template <typename ValueType, typename Payload>
    inline StridedIterator<ValueType, Payload>::StridedIterator( ) 
      : m_basePtr(0)
      , m_strideInBytes(0)
    {
    }

    template <typename ValueType, typename Payload>
    inline StridedIterator<ValueType, Payload>::StridedIterator( ValueType *basePtr, size_t strideInBytes, Payload payload )
      : m_basePtr( basePtr )
      , m_strideInBytes( strideInBytes )
      , m_payload( payload )
    {
    }

    template <typename ValueType, typename Payload>
    inline StridedIterator<ValueType,Payload>::StridedIterator( const StridedIterator &rhs )
      : m_basePtr( rhs.m_basePtr )
      , m_strideInBytes( rhs.m_strideInBytes )
      , m_payload( rhs.m_payload )
    {
    }

    template <typename ValueType, typename Payload>
    inline StridedIterator<ValueType,Payload> &StridedIterator<ValueType,Payload>::operator=( const StridedIterator &rhs )
    {
      m_basePtr = rhs.m_basePtr;
      m_strideInBytes = rhs.m_strideInBytes;
      m_payload = rhs.m_payload;

      return *this;
    }

    template <typename ValueType, typename Payload>
    ValueType &StridedIterator<ValueType,Payload>::operator*()
    {
      return *reinterpret_cast<ValueType *>(m_basePtr);
    }

    template <typename ValueType, typename Payload>
    ValueType &StridedIterator<ValueType,Payload>::operator[](size_t index)
    {
      char *ptr = reinterpret_cast<char *>(m_basePtr);
      ptr += m_strideInBytes * index;
      return *reinterpret_cast<ValueType *>(ptr);
    }

    template <typename ValueType, typename Payload>
    StridedIterator<ValueType,Payload> StridedIterator<ValueType,Payload>::operator+(int offset) const
    {
      char *ptr = reinterpret_cast<char *>(m_basePtr);
      ptr += offset * m_strideInBytes;
      return StridedIterator<ValueType, Payload>( ptr, m_strideInBytes, m_payload );
    }

    template <typename ValueType, typename Payload>
    StridedIterator<ValueType,Payload> &StridedIterator<ValueType,Payload>::operator++()
    {
      char *ptr = reinterpret_cast<char *>(m_basePtr);
      ptr += m_strideInBytes;
      m_basePtr = reinterpret_cast<ValueType *>(ptr);
      return *this;
    }

    template <typename ValueType, typename Payload>
    StridedIterator<ValueType,Payload> StridedIterator<ValueType,Payload>::operator++(int)
    {
      StridedIterator copy(*this);
      ++(*this);
      return copy;
    }

    template <typename ValueType, typename Payload>
    bool StridedIterator<ValueType,Payload>::operator==( const StridedIterator<ValueType, Payload> &rhs ) const
    {
      return m_basePtr == rhs.m_basePtr;
    }

    template <typename ValueType, typename Payload>
    bool StridedIterator<ValueType,Payload>::operator!=( const StridedIterator<ValueType, Payload> &rhs ) const
    {
      return m_basePtr != rhs.m_basePtr;
    }

    /************************/
    /* StridedConstIterator */
    /************************/
    template <typename ValueType, typename Payload = DummyPayload>
    class StridedConstIterator : public std::iterator< std::input_iterator_tag, ValueType >
    {
    public:
      StridedConstIterator( );
      StridedConstIterator( const ValueType *basePtr, size_t strideInBytes, Payload = Payload() );
      StridedConstIterator( const StridedConstIterator &rhs );
      StridedConstIterator &operator=( const StridedConstIterator &rhs );
      virtual ~StridedConstIterator() {}

      const ValueType &operator*() const;
      const ValueType &operator[](size_t index) const;
      bool operator==(const StridedConstIterator &rhs) const; // compares only ptr!
      bool operator!=(const StridedConstIterator &rhs) const; // compares only ptr!
      StridedConstIterator operator+(size_t offset) const;
      StridedConstIterator &operator++();  // pre-increment
      StridedConstIterator operator++(int);  //post-increment
    protected:
      const ValueType *m_basePtr;
      size_t           m_strideInBytes;
      Payload          m_payload;
    };

    template <typename ValueType, typename Payload>
    inline StridedConstIterator<ValueType, Payload>::StridedConstIterator( )
      : m_basePtr(0)
      , m_strideInBytes(0)
    {
    }

    template <typename ValueType, typename Payload>
    inline StridedConstIterator<ValueType, Payload>::StridedConstIterator( const ValueType *basePtr, size_t strideInBytes, Payload payload )
      : m_basePtr( basePtr )
      , m_strideInBytes( strideInBytes )
      , m_payload( payload )
    {
    }

    template <typename ValueType, typename Payload>
    inline StridedConstIterator<ValueType,Payload>::StridedConstIterator( const StridedConstIterator &rhs )
      : m_basePtr( rhs.m_basePtr )
      , m_strideInBytes( rhs.m_strideInBytes )
      , m_payload( rhs.m_payload )
    {
    }

    template <typename ValueType, typename Payload>
    inline StridedConstIterator<ValueType,Payload> &StridedConstIterator<ValueType,Payload>::operator=( const StridedConstIterator &rhs )
    {
      m_basePtr = rhs.m_basePtr;
      m_strideInBytes = rhs.m_strideInBytes;
      m_payload = rhs.m_payload;

      return *this;
    }

    template <typename ValueType, typename Payload>
    const ValueType &StridedConstIterator<ValueType,Payload>::operator*() const
    {
      return *reinterpret_cast<const ValueType *>(m_basePtr);
    }

    template <typename ValueType, typename Payload>
    const ValueType &StridedConstIterator<ValueType,Payload>::operator[](size_t index) const
    {
      const char *ptr = reinterpret_cast<const char *>(m_basePtr);
      ptr += m_strideInBytes * index;
      return *reinterpret_cast<const ValueType *>(ptr);
    }

    template <typename ValueType, typename Payload>
    StridedConstIterator<ValueType,Payload> StridedConstIterator<ValueType,Payload>::operator+( size_t offset ) const
    {
      const char *ptr = reinterpret_cast<const char *>(m_basePtr);
      ptr += offset * m_strideInBytes;
      return StridedConstIterator<ValueType, Payload>( reinterpret_cast<const ValueType *>(ptr), m_strideInBytes, m_payload );
    }

    template <typename ValueType, typename Payload>
    StridedConstIterator<ValueType,Payload> &StridedConstIterator<ValueType,Payload>::operator++()
    {
      const char *ptr = reinterpret_cast<const char *>(m_basePtr);
      ptr += m_strideInBytes;
      m_basePtr = reinterpret_cast<const ValueType *>(ptr);
      return *this;
    }

    template <typename ValueType, typename Payload>
    StridedConstIterator<ValueType,Payload> StridedConstIterator<ValueType,Payload>::operator++(int)
    {
      StridedConstIterator copy(*this);
      ++(*this);
      return copy;
    }

    template <typename ValueType, typename Payload>
    bool StridedConstIterator<ValueType,Payload>::operator==( const StridedConstIterator<ValueType, Payload> &rhs ) const
    {
      return m_basePtr == rhs.m_basePtr;
    }

    template <typename ValueType, typename Payload>
    bool StridedConstIterator<ValueType,Payload>::operator!=( const StridedConstIterator<ValueType, Payload> &rhs ) const
    {
      return m_basePtr != rhs.m_basePtr;
    }

  } //namespace util
} // namespace dp
