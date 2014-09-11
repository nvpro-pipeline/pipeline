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

#include <dp/util/Config.h>
#include <dp/util/Allocator.h> // for IAllocator interface
#include <dp/util/Atom.h>
#include <vector>

#if DPUTIL_RCOBJECT_IDS
#include <map>
#endif

namespace dp
{
  namespace util
  {
    class RCObject;
    template<typename T> class SmartPtr;

    /*! \brief Provides an interface for reference counted objects. 
      * \par Namespace: dp::util
      * \remarks
      * The class provides a protected interface for managing reference counted objects. RCObject inherits
      * from IAllocator to utilize an optimized heap manager for small object allocation.
      * \n\n
      * \note An RCObject can only be constructed on heap. The compiler will reject any
      * attempt to construct an RCObject on stack.
      * \note It is prohibited to explicitly delete an RCObject by calling \c delete on a pointer
      * to an RCObject received from a previous call to \c new. The compiler will complain if any 
      * client code attempts to do so. If client code creates an RCObject, it must pass it to a SmartPtr. That
      * SmartPtr then manages the reference count of this RCObject.
      * \note The reference count of a newly created RCObject is initially zero.
      * \sa dp::util::SmartPtr
      */
    class RCObject : public IAllocator
    {
      template<typename T> friend class SmartPtr;
      template<typename T> friend class SharedHandle;

      public:
        /*! \brief Tests whether this RCobject is shared.
          * \return
          * \c true if the internal reference count is greater than one, \c false otherwise.
          * \remarks
          */
        bool isShared()          const; 

      protected:
        /*! \brief Increments the object's reference count.
          * \remarks
          * For user code to ensure that the data of a reference counted object is valid as long as it
          * uses the object, it should first increment the objects reference count. After usage, 
          * client code should decrement the objects reference count to avoid resource leaks.
          *
          * \note The reference count of a newly created RCObject initially is zero.
          * \sa RCObject::removeRef
          */
        void addRef()            const;

        /*! \brief Decrements the object's reference count. 
          * \remarks
          * It is prohibited to explicitly delete an RCObject. 
          * The object will be automatically deleted, if the reference count changes from one to zero.
          * \sa RCObject::addRef
          */
        void removeRef()         const;

        /*! \brief Constructs an RCObject.
          * \remarks
          * Initializes the reference count to zero. The creator is responsible for setting the
          * reference count to its proper value.
          * \n\n
          * Initially the RCObject is marked as shareable.
          * \n\n
          * An RCObject is intended to serve as base class only. That is, an RCObject can be 
          * instantiated only as an object of a class derived from RCObject.
          */
#if DPUTIL_RCOBJECT_IDS
        DP_UTIL_API
#endif
        RCObject();

        /*! \brief Constructs an RCObject.
          * \remarks
          * A new RCObject is created, and hence, the reference count will be initialized to zero 
          * because no one but the creator is currently referencing this object, and the
          * creator is responsible for setting the reference count to its proper value.
          * \n\n
          * Initially the RCObject is marked as shareable.
          * \n\n
          * An RCObject is intended to serve as base class only. That is, an RCObject can be 
          * instantiated only as an object of a class derived from RCObject.
          */
#if DPUTIL_RCOBJECT_IDS
        DP_UTIL_API
#endif
        RCObject(const RCObject& rhs);

        /*! \brief Assignment operator.
         * \remarks
         * Assigning new content from another object does not change the number of users referencing
         * the object nor does it change its shareable state. Hence, this assignment operator leaves
         * the reference count and the shareable state unchanged.
         */
        RCObject& operator=( RCObject const& /*rhs*/) { return *this; }

        /*! \brief Destructs an RCObject.
          * \remarks
          * An RCObject will be automatically deleted when its reference count changes from one to 
          * zero.
          * \sa RCObject::removeRef
          */
#if DPUTIL_RCOBJECT_IDS
        DP_UTIL_API
#endif
        virtual ~RCObject();

      private:
        /** m_refcnt must be mutable, so we can also call addRef() / removeRef() from const pointers. */
        mutable Atom32 m_refcnt;

#if DPUTIL_RCOBJECT_IDS
        // debug data to uniquely identify an RCObject
        static unsigned int m_debugNextId;                          // unique id for the next instance
        unsigned int m_debugId;                                     // unique id of this instance
#endif
    };

    typedef dp::util::SmartPtr< RCObject > SmartRCObject; //!< Type definition for a SmartPtr of type RCObject

    /*! \brief A reference counted vector.
      * \par Namespace : dp::util
      * \remarks
      * A RCVector is simply a STL vector providing the RCObject interface. A RCVector, once 
      * constructed using the limited set of available constructors, can be used same as a STL vector. 
      */
    template<typename T>
    class RCVector : public std::vector<T>, public RCObject
    {
      public:
        /*! \brief Constructs an initially empty RCVector
         */
        RCVector() {}

        /*! \brief Copy constructor of an RCVector.
         *  \param rhs The RCVector to copy from. */
        RCVector( const RCVector<T> & rhs ) : std::vector<T>(rhs) {}

        /*! \brief Constructs a RCVector of the specified size.
          * \param
          * n Number of elements in the constructed vector.
          * \remarks
          * The constructor specifies a repetition of the specified number \a n of elements of the 
          * default value for type T.
          * \sa std::vector
          */
        explicit RCVector(unsigned int n) : std::vector<T>(n) {}
    };

    //----------------------------------------------------------------------------
    // inline implementations following
    //----------------------------------------------------------------------------


#if !DPUTIL_RCOBJECT_IDS

    // inline implementations for RCObject without unique instance ids
    // see RCObject.cpp for the implementations with ids

    inline RCObject::RCObject()
    {
    }

    inline RCObject::RCObject( RCObject const& /*rhs*/ )
    {
    }

    inline RCObject::~RCObject()
    {
    }

#endif

    inline void RCObject::addRef() const 
    { 
      ++m_refcnt; 
    } 

    inline void RCObject::removeRef() const
    { 
      if ( !--m_refcnt ) 
      {
        delete this; 
      } 
    }

    inline bool RCObject::isShared() const 
    { 
      return m_refcnt > 1;
    }

  } // namespace util
} // namespace dp
