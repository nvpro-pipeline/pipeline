// Copyright NVIDIA Corporation 2002-2005
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

#include <dp/Types.h>
#include <dp/sg/core/Config.h>
#include <dp/sg/core/CoreTypes.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

       /*! \brief Class that represents a path through the tree.
        *  \par Namespace: dp::sg::core
        *  \remarks  A Path represents a scene graph or subgraph. It contains
        *  pointers to a chain of objects, each being a child of the
        *  previous object. \n
        *  A path can hold any kind of Object-derived objects (even Primitives). */
      class Path
      {
        public:
          DP_SG_CORE_API static PathSharedPtr create();
          DP_SG_CORE_API static PathSharedPtr create( PathSharedPtr const& rhs );

          //! Prohibit explicit creation on stack by making the destructor protected.
          DP_SG_CORE_API virtual ~Path();

        public:
          /*! \brief Get the length of the path through the graph.
           *  \return The number of nodes in the path chain. 
           *  \remarks The path length is the exact number of nodes in the path chain. */
          unsigned int getLength() const;

          /*! \brief Test an empty path.
           *  \return When there is no node in the path chain, this function returns true.
           *  \remarks Testing on emptiness with this function is possibly faster than testing 
           *  the length. */
          bool isEmpty() const;  
      
          /*! \brief Get the head object of the path.
           *  \return This function returns a pointer to the constant head object or NULL when the
           *  path is empty.
           *  \remarks The head object is the first object in the path chain. 
           *  the call Path::getFromHead(0) is equivalent to Path::getHead().
           *  \sa getTail, getFromHead, getFromTail */
          ObjectSharedPtr getHead() const;

          /*! \brief Get the tail object of the path.
           *  \return This function returns a pointer to the constant tail object or NULL when the
           *  path is empty.
           *  \remarks The tail object is the last object in the path chain. 
           *  the call Path::getFromTail(0) is equivalent to Path::getTail().
           *  \sa getHead, getFromHead, getFromTail */
          ObjectSharedPtr getTail() const;

          /*! \brief Get an object on the path. 
           *  \param i Null based index of the node to get from the path. The behavior of this 
           *  function is undefined for an invalid index.
           *  \return Pointer to the constant object with the index i.
           *  \remarks The returned object is the object at the requested position. Providing an index
           *  of 0 corresponds to the call to Path::getHead.
           *  \sa getHead, getFromHead, getFromTail */ 
          ObjectSharedPtr getFromHead(unsigned int i) const;

          /*! \brief Get an object on the path. 
           *  \param i Null based index of the node to get from the path. The behavior of this 
           *  function is undefined for an invalid index.
           *  \return Pointer to the constant object with the index i.
           *  \remarks The returned object is the object at the requested position. Providing an index
           *  of 0 corresponds to the call to Path::getTail.
           *  \sa getTail, getFromHead, getFromTail */ 
          ObjectSharedPtr getFromTail(unsigned int i) const;
      
          /*! \brief Remove the last object from the path. 
           *  \remarks This function simply removes the last (tail) object from the current path chain.
           *  It also handles decrementing the reference count of the removed object.\n
           *  Calling this function on an empty path will lead to undefined behavior.
           *  \sa push, truncate */
          void pop();

          /*! \brief Append an object to the path chain. 
           *  \param pObject Pointer to a constant Object. Providing an invalid pointer or a NULL pointer
           *  will lead to undefined behavior.
           *  \remarks This function appends the given Object to the end of the path chain. 
           *  It also correctly increments the reference count of the provide object. 
           *  \sa pop, truncate*/
          DP_SG_CORE_API void push( const ObjectSharedPtr & pObject);
           
          /*! \brief Remove several objects from the path chain. 
           *  \param start The start index where the removal starts. Providing an invalid start 
           *  index will lead to undefined behavior.
           *  \remarks This function removes all following objects beginning from the start index
           *  from the path chain, including the object at the start index. It correctly
           *  decrements the reference counts of all the removed objects.\n
           *  Calling Path::truncate(0) will remove all objects from the path chain.
           *  \sa pop, push*/
          DP_SG_CORE_API void truncate(unsigned int start);

          /*! \brief Get the ModelToWorld and WorldToModel matrices along the path. 
           *  \param modelToWorld Reference for returning the ModelToWorld matrix, excluding the  
           *  tail node transformation.
           *  \param worldToModel Reference for returning the WorldToModel matrix, excluding the 
           *  tail node transformation.
           *  \remarks The accumulated matrices do not contain the tail node transformation. 
           *  These matrices are very handy in the dp::sg::ui::manipulator::TrackballTransformManipulator
           *  derived classes.
           *  \sa dp::sg::ui::manipulator::TrackballTransformManipulator */
          DP_SG_CORE_API void getModelToWorldMatrix( dp::math::Mat44f & modelToWorld
                                                   , dp::math::Mat44f & worldToModel) const;


          /*! \brief Compares two Path instances.
           *  \param rhs The Path instance to compare with.
           *  \returns \c true if the two instances are equal, that is - 
           *  if both represent the same path. \c false otherwise.
           */
          bool operator==(const Path& rhs) const;

          /*! \brief Compares two Path instances.
          *  \param rhs The Path instance to compare with.
          *  \returns \c true if the two instances are not equal, that is - 
          *  if both don't represent the same path. \c false otherwise.
          */
          bool operator!=(const Path& rhs) const;
      
          /*! \brief Compares two Path instances.
          *  \param rhs The Path instance to compare with.
          *  \returns \c true if 'this' is less than rhs, that is -
          *  rhs is deeper than 'this'; or the depths are equivalent, but the paths are different.
          *  \c false otherwise.
          */
          bool operator<(const Path& rhs) const;

        protected:
          /*! \brief Construct a Path object. */
          DP_SG_CORE_API Path();

          /*! \brief Copy constructor.
           *  \param rhs Reference to a constant Path object.
           *  \remarks This copy constructor creates an exact copy of the provided Path object.\n
           *  Internally this class holds a vector of node pointer representing the path through the 
           *  graph. This copy constructor does not perform a deep copy on the pointers in the vector. 
           *  It simply copies the pointers.*/
          DP_SG_CORE_API Path( PathSharedPtr const& rhs);


        private:
          std::vector<ObjectWeakPtr> m_path;   //!< Vector of objects representing a path chain.
      };

      // - - - - - - - - - - - - - - - - - - -
      // inlines
      // - - - - - - - - - - - - - - - - - - -

      inline ObjectSharedPtr Path::getHead() const
      {
        return( m_path.empty() ? ObjectSharedPtr::null : m_path.front()->getSharedPtr<Object>() );
      }

      inline ObjectSharedPtr Path::getTail() const
      {
        return( m_path.empty() ? ObjectSharedPtr::null : m_path.back()->getSharedPtr<Object>() );
      }

      inline ObjectSharedPtr Path::getFromHead( unsigned int i ) const
      {
        DP_ASSERT(0 <= i && i < m_path.size());
        return( m_path[i]->getSharedPtr<Object>() );
      }

      inline ObjectSharedPtr Path::getFromTail( unsigned int i ) const
      {
        DP_ASSERT(0 <= i && i < m_path.size());
        return( m_path[m_path.size()-i-1]->getSharedPtr<Object>() );
      }

      inline unsigned int Path::getLength() const
      {
        return( dp::checked_cast<unsigned int>(m_path.size()) );
      }

      inline void Path::pop()
      {
        DP_ASSERT(getLength() > 0);
        m_path.pop_back();
      }

      inline bool Path::isEmpty() const
      {
        return m_path.empty();
      }

      inline bool Path::operator==(const Path& rhs) const
      {
        return m_path == rhs.m_path;
      }

      inline bool Path::operator!=(const Path& rhs) const
      {
        return m_path != rhs.m_path;
      }

      inline bool Path::operator<(const Path& rhs) const
      {
        return m_path.size() < rhs.m_path.size() ||
              (m_path.size() == rhs.m_path.size() &&
               memcmp( &m_path[0], &rhs.m_path[0], m_path.size() * sizeof( ObjectWeakPtr)) < 0 );
      }

    } // namespace core
  } // namespace sg
} // namespace dp

