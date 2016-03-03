// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>


/*! \brief Macro to define the two standard pointer types for a type T.
*  \remark For convenience, for each class T, we define the types TSharedPtr and TWeakPtr */
#define DEFINE_PTR_TYPES(T)                 \
  class T;                                  \
  typedef std::shared_ptr<T>  T##SharedPtr; \
  typedef std::weak_ptr<T>    T##WeakPtr


/*! \brief Macro to define ObjectType and the two standard pointer types of a base type T as part of a templated struct.
*  \remark Using this struct, the standard types SharedPtr and WeakPtr, as well as
*  the ObjectType itself, are easily available within a template context. */
#define SHARED_OBJECT_TRAITS_BASE(T)      \
template <> struct ObjectTraits<T>        \
{                                         \
  typedef T                   ObjectType; \
  typedef std::shared_ptr<T>  SharedPtr;  \
  typedef std::weak_ptr<T>    WeakPtr;    \
}

/*! \brief Macro to define ObjectType, BaseType and the two standard pointer types of a type T, with base type BT, as part of a templated struct.
*  \remark Using this struct, the standard types SharedPtr and WeakPtr, as well as
*  the ObjectType itself and BaseType, are easily available within a template context. */
#define SHARED_OBJECT_TRAITS(T, BT)       \
template <> struct ObjectTraits<T>        \
{                                         \
  typedef T                   ObjectType; \
  typedef BT                  BaseType;   \
  typedef std::shared_ptr<T>  SharedPtr;  \
  typedef std::weak_ptr<T>    WeakPtr;    \
}

namespace dp
{
  namespace util
  {
    template <typename ObjectType> struct ObjectTraits {};
  }//namespace util
}//namespace dp
