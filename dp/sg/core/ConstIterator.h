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
/** @file */

#include <dp/util/SharedPtr.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Iterator base class for various iterators in SceniX
       *  \param FriendType The class being a friend of this ConstIterator.
       *  \param IteratorType The iterator type to wrap.
       *  \remark This class is used to wrap an arbitrary iterator, but provide only const access to it, while
       *  one of the template arguments specifies a class to be a friend of this ConstIterator, thus allowing
       *  unlimited access to just that class.
       *  \note There is no post-increment or post-decrement operator declared.
       *  \sa Camera::HeadLightIterator, GeoNode::StateSetIterator, GeoNode::DrawableIterator */
      template<typename FriendType, typename IteratorType>
      class ConstIterator : public std::iterator<std::input_iterator_tag, typename IteratorType::value_type>
      {
#if defined(__GNUC__)
        friend class dp::util::ObjectTraits<FriendType>::ObjectType;
#else
        friend typename dp::util::ObjectTraits<FriendType>::ObjectType;
#endif

      public:
          /*  \brief Default constructor */
          ConstIterator();

          /*! \brief Construct a ConstIterator out of an iterator
           *  \param it An iterator of the same type being wrapped by this ConstIterator */
          explicit ConstIterator( const IteratorType & it );

          /*! \brief Constant indirection operator, providing a const reference to the variable
           *  \return A constant reference to the variable */
          const typename IteratorType::value_type & operator*() const;

          /*! \brief Constant member operator, providing a const pointer to the variable
           *  \return A constant pointer to the variable */
          const typename IteratorType::value_type * operator->() const;

          /*! \brief Equal operator
           *  \param di The ConstIterator to compare with.
           *  \return \c true, if \a this and \a di are equal, otherwise \c false. */
          bool operator==( const ConstIterator & di ) const;

          /*! \brief Not equal operator
           *  \param di The ConstIterator to compare with.
           *  \return \c true, if \a this and \a di are not equal, otherwise \c false. */
          bool operator!=( const ConstIterator & di ) const;

          /*! \brief Prefix increment operator
           *  \return A reference to the ConstIterator after incrementing. */
          ConstIterator & operator++();

          /*! \brief Prefix decrement operator
           *  \return A reference to the ConstIterator after decrementing. */
          ConstIterator & operator--();

        protected:
          IteratorType m_iter;    //!< The wrapped iterator
      };

      template<typename FriendType, typename IteratorType>
      inline ConstIterator<FriendType,IteratorType>::ConstIterator()
      {
      }

      template<typename FriendType, typename IteratorType>
      inline ConstIterator<FriendType,IteratorType>::ConstIterator( const IteratorType & ti )
        : m_iter( ti )
      {
      }

      template<typename FriendType, typename IteratorType>
      inline const typename IteratorType::value_type & ConstIterator<FriendType,IteratorType>::operator*() const
      {
        return( m_iter.operator*() );
      }

      template<typename FriendType, typename IteratorType>
      inline const typename IteratorType::value_type * ConstIterator<FriendType,IteratorType>::operator->() const
      {
        return( m_iter.operator->() );
      }

      template<typename FriendType, typename IteratorType>
      inline bool ConstIterator<FriendType,IteratorType>::operator==( const ConstIterator & di ) const
      {
        return( m_iter == di.m_iter );
      }

      template<typename FriendType, typename IteratorType>
      inline bool ConstIterator<FriendType,IteratorType>::operator!=( const ConstIterator & di ) const
      {
        return( m_iter != di.m_iter );
      }

      template<typename FriendType, typename IteratorType>
      inline ConstIterator<FriendType,IteratorType> & ConstIterator<FriendType,IteratorType>::operator++()
      {
        ++m_iter;
        return( *this );
      }

      template<typename FriendType, typename IteratorType>
      inline ConstIterator<FriendType,IteratorType> & ConstIterator<FriendType,IteratorType>::operator--()
      {
        --m_iter;
        return( *this );
      }

    } // namespace core
  } // namespace sg
} // namespace dp

