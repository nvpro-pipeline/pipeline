// Copyright NVIDIA Corporation 2002-2010
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

#include <dp/util/Reflection.h>
#include <dp/util/Observer.h>
#include <dp/sg/core/Event.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/SharedPtr.h>
#include <boost/type_traits.hpp>
#include <memory>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      // serves as base class for 'handled' objects
      class HandledObject : public dp::util::Reflection, public std::enable_shared_from_this<HandledObject>
      {
        public:
          virtual ~HandledObject();

          virtual HandledObjectSharedPtr clone() const = 0;

          template<typename T> typename SharedPtr<T> getSharedPtr() const;

        protected:
          HandledObject();
          HandledObject( const HandledObject & );
          HandledObject& operator=(const HandledObject & rhs);

          using std::enable_shared_from_this<HandledObject>::shared_from_this;    // hide this from using with SharedPtrs
      };

      inline HandledObject::HandledObject()
      {
        /* do nothing! */
      }

      inline HandledObject::HandledObject( const HandledObject &rhs )
        : Reflection( rhs )
      {
        /* do nothing! */
      }

      inline HandledObject::~HandledObject()
      {
      }

      inline HandledObject& HandledObject::operator=( const HandledObject &rhs )
      {
        Reflection::operator=( rhs );
        return( *this );
      }

      template <typename T>
      inline typename SharedPtr<T> HandledObject::getSharedPtr() const
      {
        DP_STATIC_ASSERT(( boost::is_base_of<HandledObject,T>::value ));
        return( HandledObjectSharedPtr( const_cast<HandledObject*>(this)->shared_from_this() ).staticCast<T>() );
      }

      //! Detects if the WeakPtr \a p is a WeakPtr to a specified (templated) type.
      /** \returns \c true if \a p is a WeakPtr to the specified type, \c false otherwise. */
      template<typename T>
      inline bool isPtrTo( const HandledObjectWeakPtr & p )
      {
        return( dynamic_cast<const T*>( p ) != nullptr );
      }

    }//namespace core
  }//namespace sg
}//namespace dp

    
      
