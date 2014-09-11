// Copyright NVIDIA Corporation 2012
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

#include <dp/sg/core/Object.h>
#include <dp/sg/core/SharedPtr.h>
#include <vector>
#include <set>

namespace dp
{
  namespace sg
  {
    namespace core
    {
      
      /*! \brief Helper class to hold a vector of owners of an Object.
       *  \remarks Some classes, like Primitive, LightSource, or StateAttribute, need to know all the
       *  objects that "own" them, i.e. the objects that are referencing them in the tree hierarchy.
       *  This knowledge is needed to carry information from such objects that are not derived from Node
       *  "up-tree" to the Nodes that hold them. That is, the "owner-relationship" is similar to the
       *  "parent-relationship" of the Nodes. Each OwnedObject can only be owned by one specific type of
       *  owners, specified as the template parameter "OwnerType".
       *  \sa Node */
      template<typename OwnerType>
      class OwnedObject : public Object
      {
      public:
        typedef std::vector<typename ObjectTraits<OwnerType>::WeakPtr> OwnerContainer;
        typedef typename OwnerContainer::const_iterator                OwnerIterator;

      public:
        /*! \brief Assignment operator
         *  \param rhs A reference to the constant OwnedObject to copy from
         *  \return A reference to the assigned OwnedObject
         *  \remarks The assignment operator calls the assignment operator of Object.
         *  \note The owners of rhs are not copied.
         *  \sa Object */
        OwnedObject<OwnerType>& operator=(const OwnedObject<OwnerType> & rhs);

        /*! \brief Get the number of owners of this OwnedObject.
         *  \return Number of owners of this OwnedObject.
         *  \remarks An OwnedObject that is part of one or more OwnerType objects holds a pointer
         *  back to each of those OwnerType objects. These are called the owners. So the number
         *  of owners tells you how often this OwnedObject is referenced in a scene tree.
         *  \sa getOwner */
        size_t getNumberOfOwners() const;

        /*! \brief Get the owning OwnerType for a given iterator.
         *  \param it Position of the OwnerType to get in the list of owners.
         *  \return The OwnerType that owns this OwnedObject.
         *  \remarks An OwnedObject that is part of one or more OwnerType objects, holds a
         *  pointer back to each of those OnwerType objects. These are called the owners.
         *  \note The behavior is undefined if \a it is invalid.
         *  \sa getNumberOfOwners */
        typename ObjectTraits<OwnerType>::WeakPtr getOwner( OwnerIterator it ) const;

        /*! \brief Get an iterator to the first entry in the list of the object's owners.
         */
        OwnerIterator ownersBegin() const;

        /*! \brief Get an iterator past the last entry in the list of the object's owners.
         */
        OwnerIterator ownersEnd() const;

        /*! \brief Add an owner to the object.
         *  \param owner The owner that should be added to the object's list of owners.
         *  \remarks In debug mode, the function asserts that the owner is not already in the list.
         */
        void addOwner( typename ObjectTraits<OwnerType>::WeakPtr owner );

        /*! \brief Remove an owner from the object.
         *  \param owner The owner that should be removed from the object's list of owners.
         *  \remarks In debug mode, the function asserts that the owner is in the list.
         */
        void removeOwner( typename ObjectTraits<OwnerType>::WeakPtr owner );

        void markDirty( unsigned int state );

      protected:
        /*! \brief Protected default constructor to prevent instantiation of an OwnedObject.
         *  \remarks An OwnedObject is not intended to be instantiated, but only classes derived from it.
         */
        OwnedObject();

        /*! \brief Protected copy constructor from an Object.
         *  \remarks An OwnedObject is not intended to be instantiated, but only classes derived from it.
         */
        OwnedObject( const Object & rhs );

        /*! \brief Protected copy constructor from an OwnedObject.
         *  \remarks An OwnedObject is not intended to be instantiated, but only classes derived from it.
         */
        OwnedObject( const OwnedObject<OwnerType> &rhs );

        /*! \brief Protected destructor of an OwnedObject.
         *  \remarks An OwnedObject is not intended to be instantiated, but only classes derived from it.
         */
        ~OwnedObject();

      private:
        OwnerContainer  m_owners;
      };

      template<typename OwnerType>
      OwnedObject<OwnerType>::OwnedObject()
      : Object()
      {
      }

      template<typename OwnerType>
      OwnedObject<OwnerType>::OwnedObject( const Object &rhs )
      : Object(rhs)
      {
        // must not copy owners!!!
        // this is what this implementation of the cpy ctor is for!
        DP_ASSERT( m_owners.empty() );
      }

      template<typename OwnerType>
      OwnedObject<OwnerType>::OwnedObject( const OwnedObject<OwnerType> &rhs )
      : Object(rhs)
      {
        // must not copy owners!!!
        // this is what this implementation of the cpy ctor is for!
        DP_ASSERT( m_owners.empty() );
      }

      template<typename OwnerType>
      OwnedObject<OwnerType>::~OwnedObject()
      {
        // if this fires, there was no proper cleanup of owner relationship elsewhere
        DP_ASSERT( m_owners.empty() );
      }

      template<typename OwnerType>
      inline OwnedObject<OwnerType> & OwnedObject<OwnerType>::operator=( const OwnedObject<OwnerType> &rhs )
      {
        Object::operator=( rhs );
        return( *this );
      }

      template<typename OwnerType>
      inline size_t OwnedObject<OwnerType>::getNumberOfOwners( void ) const
      {
        return( m_owners.size() );
      }

      template<typename OwnerType>
      inline typename OwnedObject<OwnerType>::OwnerIterator OwnedObject<OwnerType>::ownersBegin() const
      {
        return m_owners.begin();
      }

      template<typename OwnerType>
      inline typename OwnedObject<OwnerType>::OwnerIterator OwnedObject<OwnerType>::ownersEnd() const
      {
        return m_owners.end();
      }

      template<typename OwnerType>
      inline typename ObjectTraits<OwnerType>::WeakPtr OwnedObject<OwnerType>::getOwner( OwnerIterator it ) const
      {
        return( *it );
      }

      template<typename OwnerType>
      inline void OwnedObject<OwnerType>::addOwner( typename ObjectTraits<OwnerType>::WeakPtr owner )
      {
        m_owners.push_back( owner );
      }
  
      template<typename OwnerType>
      inline void OwnedObject<OwnerType>::removeOwner( typename ObjectTraits<OwnerType>::WeakPtr owner )
      {
        typename OwnerContainer::iterator it = std::find( m_owners.begin(), m_owners.end(), owner );
        DP_ASSERT( it != m_owners.end() );
        *it = m_owners.back();
        m_owners.pop_back();
      }

    } // namespace core
  } // namespace sg
} // namespace dp
