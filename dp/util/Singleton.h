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
/** \file */

#include <dp/util/Config.h>

namespace dp
{
  namespace util
  {
    //! A Singleton framework
    /** Ensures a class \c T only has one instance. Defines a global point of access to that object.  
      */
    template <typename T>
    class Singleton
    {
    public:
      //! Single instance access
      /** Receive a pointer to the single instance.
        */
      static T* instance();

    private:
      // not permitted 
      // client must call Singleton<T>::instance() for instantiation
      Singleton() {}
      Singleton(const Singleton<T>&);
      ~Singleton() {}
      // also not permitted
      Singleton<T>& operator=(const Singleton<T>&);

    private:
      static T m_instance; //!< single instance of T
    };

    template <typename T>
    inline T* Singleton<T>::instance()
    {
      return &m_instance;
    };

    template <typename T> T Singleton<T>::m_instance;
  } // namespace util
} // namespace dp
