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

#include <boost/array.hpp>

namespace dp
{
  namespace util
  {

    // template to determine #elements of an static array, usage sizeof array( yourarray );
    template< typename T, size_t N> char(&array(T(&)[N]))[N];

    template<typename T>
    boost::array<T,3> makeArray( const T & t0, const T & t1, const T & t2 )
    {
      boost::array<T,3> a;
      a[0] = t0;  a[1] = t1;  a[2] = t2;
      return( a );
    }

    template<typename T>
    boost::array<T,4> makeArray( const T & t0, const T & t1, const T & t2, const T & t3 )
    {
      boost::array<T,4> a;
      a[0] = t0;  a[1] = t1;  a[2] = t2;  a[3] = t3;
      return( a );
    }

    template<typename T>
    boost::array<T,9> makeArray( const T & t0, const T & t1, const T & t2, const T & t3, const T & t4, const T & t5, const T & t6, const T & t7, const T & t8 )
    {
      boost::array<T,9> a;
      a[0] = t0;  a[1] = t1;  a[2] = t2;  a[3] = t3;  a[4] = t4;  a[5] = t5;  a[6] = t6;  a[7] = t7;  a[8] = t8;
      return( a );
    }

    template<typename T>
    boost::array<T,16> makeArray( const T & t0, const T & t1, const T & t2, const T & t3, const T & t4, const T & t5, const T & t6, const T & t7, const T & t8, const T & t9, const T & t10, const T & t11, const T & t12, const T & t13, const T & t14, const T & t15 )
    {
      boost::array<T,16> a;
      a[0] = t0;  a[1] = t1;  a[2] = t2;  a[3] = t3;  a[4] = t4;  a[5] = t5;  a[6] = t6;  a[7] = t7;  a[8] = t8;  a[9] = t9;  a[10] = t10;  a[11] = t11;  a[12] = t12;  a[13] = t13;  a[14] = t14;  a[15] = t15;
      return( a );
    }

  } // namespace util
} // namespace dp
