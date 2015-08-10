// Copyright NVIDIA Corporation 2015
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

namespace dp
{
  namespace util
  {
    template <typename T>
    class WeakPtr : public std::weak_ptr<T>
    {
      public:
        WeakPtr();
        WeakPtr( std::weak_ptr<T> const& wp );

        // implicit upcast (Derived -> Base)
        template <typename U> WeakPtr( std::weak_ptr<U> const& sp );

        SharedPtr<T> getSharedPtr() const;
    };


    template <typename T>
    inline WeakPtr<T>::WeakPtr()
    {
    }

    template <typename T>
    inline WeakPtr<T>::WeakPtr( std::weak_ptr<T> const& wp )
      : std::weak_ptr<T>( wp )
    {
    }

    template <typename T>
    template <typename U>
    inline WeakPtr<T>::WeakPtr( std::weak_ptr<U> const& wp )
      : std::weak_ptr<T>( wp )
    {
      DP_STATIC_ASSERT(( boost::is_base_of<T,U>::value ));
    }

    template <typename T>
    inline SharedPtr<T> WeakPtr<T>::getSharedPtr() const
    {
      return( this->lock() );
    }

  }//namespace util
}//namespace dp
