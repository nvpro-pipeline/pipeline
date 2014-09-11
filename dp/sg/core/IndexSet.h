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


#pragma once
/** @file */

#include <dp/sg/core/nvsgapi.h>
#include <dp/util/HashGenerator.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/util/Types.h>
#include "dp/util/RCObject.h" // base class definition
#include <dp/sg/core/Buffer.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/OwnedObject.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Provides an interface to handle indices to be attached to a Primitive
       *  \sa Primitive */
      class IndexSet : public OwnedObject<Primitive>
      {
        public:
          /*! \brief Creates a new IndexSet */
          static DP_SG_CORE_API IndexSetSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API ~IndexSet();

        public:
          /*! \brief Convenience function to set the indices.
           *  \param indices A pointer to the constant indices to use.
           *  \param count The number of indices.
           *  \param primitiveRestartIndex The primitiveRestartIndex - see setPrimitiveRestartIndex for info.
           *  \remarks Copies \a count elements from \a indices into this IndexSet, and sets the data type, primitive restart index, and number of
           *  indices.
           *  \note SceniX does not make a separate copy of the data, it stores the data in the attached buffer.  If no buffer exists,
           *  one will be created.  To extract the indices, the application must manipulate the buffer or call getData().
           *  \note The behavior is undefined if there are less than \a count * sizeof( *indices ) bytes of data located at \a indices.
           *  \note This is a convenience function to reference the attached Buffer.  If no buffer exists, one will be allocated, resized and
           *  the data will be written to the buffer.  Beware that if a buffer does exist, it will be resized and data will be written to the buffer.  
           *  See setBuffer for more details.
           *  \note Since this method makes assumptions about buffer sizes, it may be faster to manipulate the buffer directly.
           *  \sa getNumberOfIndices, appendIndexData, setBuffer 
           */
          DP_SG_CORE_API void setData( const unsigned int   * indices, unsigned int count, unsigned int primitiveRestartIndex = ~0 );
          DP_SG_CORE_API void setData( const unsigned short * indices, unsigned int count, unsigned int primitiveRestartIndex = ~0 );
          DP_SG_CORE_API void setData( const unsigned char  * indices, unsigned int count, unsigned int primitiveRestartIndex = ~0 );
          DP_SG_CORE_API void setData( const void * indices, unsigned int count, dp::util::DataType type = dp::util::DT_UNSIGNED_INT_32, 
                                                                           unsigned int primitiveRestartIndex = ~0 );

          /*! \brief set the buffer object to use for this IndexSet
           *  \param buffer The buffer to assign to this IndexSet.
           *  \param count The number of indices packed in the buffer.
           *  \param type Specifies the data type of each index in the buffer.  Valid values are  dp::util::DT_UNSIGNED_INT_8, dp::util::DT_UNSIGNED_INT_16, or dp::util::DT_UNSIGNED_INT_32.
           *  \param primitiveRestartIndex The primitiveRestartIndex - see setPrimitiveRestartIndex for info.
           *  \remarks Copies \a count elements from \a indices into this IndexSet, and sets the data type, primitive restart index, and number of
           *  indices.
           *  \remarks Applications are free to use the convenience function setData() to allocate and fill a buffer, or they may fill a buffer directly 
           *  and attach it to this IndexSet.
           *  \sa setData
           */
          DP_SG_CORE_API void setBuffer( const BufferSharedPtr &buffer, unsigned int count,
                                                                  dp::util::DataType type = dp::util::DT_UNSIGNED_INT_32,
                                                                  unsigned int primitiveRestartIndex = ~0 );

          /*! \brief Get the buffer attached to this IndexSet 
           *  \return The buffer
           */
          DP_SG_CORE_API const BufferSharedPtr & getBuffer() const;

          /*! \brief Convenience function to get the indices.
           *  \param destination A pointer to a destination buffer.
           *  \return \c true if there was an attached buffer and any data has been copied into \a destination; \c false otherwise.
           *  \remarks Copies all data from attached buffer into \a destination pointer.  Results are undefined if \a destination is
           *  not large enough to hold all the data.
           *  \note In order to determine the required buffer size, multiply the number of indices in the buffer by the byte size of
           *  the index data type, ie: 
           *
           *  size_t bytes = iset->getNumberOfIndices() * dp::util::getSizeOf( iset->getIndexDataType() );
           */
          DP_SG_CORE_API bool getData( void * destination ) const;

          /*! \brief Get the number of indices that have been attached to this IndexSet 
           *  \return The count of indices.
           */
          DP_SG_CORE_API unsigned int getNumberOfIndices() const;

          /*! \brief Set the "Primitive Restart" index value
           * \param index The index value to use.  At runtime, the value will be masked to use only the number of bits in the index data type 
           * (int, short, byte) before being used by the renderer.  Typically the application will use ~0 as the restart index to avoid interfering 
           * with an actual index.
           * \remarks Primitive Restart is typically used with [LINE|TRIANGLE|QUAD]_STRIP[_ADJACENCY], LINE_LOOP, and TRIANGLE_FAN primitive types, 
           * in order to specify multiple primitives at once.  While Primitive Restart can be used with other primitive types, it will have no benefit.\n
           * When this index is encountered, the primitive type will be restarted as if a second primitive were specified.
           * \note In order to use Primitive Restart, an index must be selected using this method, and the buffer must be packed appropriately 
           * with that Primitive Restart index.
           * \note The default primitive restart index is 0xFFFFFFFF.
           */
          DP_SG_CORE_API void setPrimitiveRestartIndex( unsigned int index );
          DP_SG_CORE_API unsigned int getPrimitiveRestartIndex() const;

          /* Set index data type
           *  \param type Specifies the data type of each index in the input data array.  Valid values are  dp::util::DT_UNSIGNED_INT_8, 
           *  dp::util::DT_UNSIGNED_INT_16, or dp::util::DT_UNSIGNED_INT_32.
           *  \note The default format is dp::util::DT_UNSIGNED_INT_32
           *  \sa setPrimitiveRestartIndex, getPrimitiveRestartIndex
           */
          DP_SG_CORE_API void setIndexDataType( dp::util::DataType type );
          DP_SG_CORE_API dp::util::DataType getIndexDataType() const;

          /*! \brief Assignment operator
           *  \param rhs A reference to the constant IndexSet to copy from.
           *  \return A reference to the assigned IndexSet.
           */
          DP_SG_CORE_API IndexSet & operator=( const IndexSet & rhs );

          /*! \brief Test for equivalence with an other IndexSet.
           *  \param p A pointer to the constant IndexSet to test for equivalence with.
           *  \param deepCompare Optional parameter to perform a deep comparsion; default is \c false.
           *  \return \c true if the IndexSet \a p is equivalent to \c this, otherwise \c false.
           *  \remarks Two IndexSets are equivalent if their data type, index count, and primitive restart index are equal;  and
           *  if \a deepCompare is \c true, their buffer size and contents are also equal.
           *  \sa Primitive, Buffer */
          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const;

          /*! \brief ConstIterator to iterate the indices as a particular index type
           */
          template <typename T>
          class ConstIterator
          {
            public:
              ConstIterator( const IndexSetSharedPtr & iset, unsigned int offset = 0 );
              ConstIterator( const ConstIterator& rhs );
              ConstIterator& operator=( const ConstIterator& rhs );
              ~ConstIterator() {}

              T operator*() const;
              T operator[](size_t index) const;
              ConstIterator& operator++();   // pre-increment
              ConstIterator  operator++(int) const; // post-increment
              ConstIterator  operator+(int offset) const;

            protected:
              Buffer::DataReadLock  m_readLock;
              const void          * m_basePtr;
              dp::util::DataType    m_type;
          };

          REFLECTION_INFO_API( DP_SG_CORE_API, IndexSet );

          using dp::util::Subject::attach;

        protected:
          /*! \brief Default constructor. */
          DP_SG_CORE_API IndexSet();

          /*! \brief Copy constructor.
           *  \param rhs A reference to the constant IndexSet to copy from
           */
          DP_SG_CORE_API IndexSet( const IndexSet& rhs );

          DP_SG_CORE_API void copyDataToBuffer( const void * ptr, unsigned int count );

          /*! \brief Feed the data of this object into the provied HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

          /** \brief Called by BufferObserver if a the buffer for the vertices has been changed **/
          DP_SG_CORE_API void onBufferChanged();

          /** \brief Observe a buffer and notify the IndexSet upon changes **/
          class BufferObserver : public dp::util::Observer
          {
          public:
            void setIndexSet( IndexSet *indexSet )
            {
              m_indexSet = indexSet;
            }

            virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload )
            {
              m_indexSet->onBufferChanged();
            }

            virtual void onDestroyed( const Subject& subject, dp::util::Payload* payload )
            {
              m_indexSet = nullptr;
            }

          private:
            IndexSet* m_indexSet;
          };

        private:
          dp::util::DataType m_dataType;
          unsigned int       m_primitiveRestartIndex;
          unsigned int       m_numberOfIndices;
          BufferSharedPtr    m_buffer;
          BufferObserver     m_bufferObserver;
      };

      inline unsigned int IndexSet::getNumberOfIndices() const
      {
        return m_numberOfIndices;
      }

      inline dp::util::DataType IndexSet::getIndexDataType() const
      {
        return m_dataType;
      }

      inline unsigned int IndexSet::getPrimitiveRestartIndex() const
      {
        return m_primitiveRestartIndex;
      }

      inline const BufferSharedPtr & IndexSet::getBuffer() const
      {
        return m_buffer;
      }

      template< typename T >
      inline IndexSet::ConstIterator<T>::ConstIterator( const IndexSetSharedPtr & iset, unsigned int offset )
        : m_readLock( iset->getBuffer() )
        , m_type( iset->getIndexDataType() )
      {
        // compute offset from base - no reason to store the offset
        m_basePtr = m_readLock.getPtr<unsigned char>() + (offset * dp::util::getSizeOf( m_type ));
      }

      template< typename T >
      inline IndexSet::ConstIterator<T>::ConstIterator( const IndexSet::ConstIterator<T> & rhs )
        : m_readLock( rhs.m_readLock )
        , m_type( rhs.m_type )
        , m_basePtr( rhs.m_basePtr )
      {
      }

      template< typename T >
      inline IndexSet::ConstIterator<T>& IndexSet::ConstIterator<T>::operator=( const IndexSet::ConstIterator<T> & rhs )
      {
        m_readLock = rhs.m_readLock;
        m_type     = rhs.m_type;
        m_basePtr  = rhs.m_basePtr;

        return *this;
      }

      template< typename T >
      inline IndexSet::ConstIterator<T> IndexSet::ConstIterator<T>::operator+( int offset ) const
      {
        ConstIterator<T> tn( *this );

        tn.m_basePtr = reinterpret_cast< const unsigned char * >( m_basePtr ) + (offset * dp::util::getSizeOf( m_type ));

        return tn;
      }

      template< typename T >
      inline IndexSet::ConstIterator<T> IndexSet::ConstIterator<T>::operator++(int) const
      {
        return *this + 1;
      }

      template< typename T >
      inline IndexSet::ConstIterator<T>& IndexSet::ConstIterator<T>::operator++()
      {
        m_basePtr = reinterpret_cast< const unsigned char * >( m_basePtr ) + dp::util::getSizeOf( m_type );

        return *this;
      }

      template< typename T >
      inline T IndexSet::ConstIterator<T>::operator*() const
      {
        switch( m_type )
        {
          case dp::util::DT_UNSIGNED_INT_32:
            return dp::util::checked_cast<T>( *reinterpret_cast<const unsigned int *>(m_basePtr) );
          case dp::util::DT_UNSIGNED_INT_16:
            return dp::util::checked_cast<T>( *reinterpret_cast<const unsigned short *>(m_basePtr) );
          case dp::util::DT_UNSIGNED_INT_8:
            return dp::util::checked_cast<T>( *reinterpret_cast<const unsigned char *>(m_basePtr) );
          default:
            DP_ASSERT(0);
            return (T)(~0);
        }
      }

      template< typename T >
      inline T IndexSet::ConstIterator<T>::operator[]( size_t index ) const
      {
        switch( m_type )
        {
          case dp::util::DT_UNSIGNED_INT_32:
            return dp::util::checked_cast<T>( reinterpret_cast<const unsigned int *>(m_basePtr)[index] );
          case dp::util::DT_UNSIGNED_INT_16:
            return dp::util::checked_cast<T>( reinterpret_cast<const unsigned short *>(m_basePtr)[index] );
          case dp::util::DT_UNSIGNED_INT_8:
            return dp::util::checked_cast<T>( reinterpret_cast<const unsigned char *>(m_basePtr)[index] );
          default:
            DP_ASSERT(0);
            return (T)(~0);
        }
      }

    } // namespace core
  } // namespace sg
} // namespace dp


