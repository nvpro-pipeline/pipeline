// Copyright NVIDIA Corporation 2012-2015
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

#include <dp/Assert.h>
#include <dp/util/Config.h>
#include <limits>

#if defined(DP_OS_LINUX)
#include <stdint.h>
#include <cstddef>
#endif

#include <stdexcept>
#include <iostream>

#if defined(DP_OS_WINDOWS)
# define DP_ALIGN(size)   __declspec( align( size ) )
#else
# define DP_ALIGN(size)
#endif

namespace dp
{

#if defined(DP_OS_WINDOWS)
  typedef signed   __int8    Int8;    ///<  8-bit   signed integer.
  typedef signed   __int16   Int16;   ///< 16-bit   signed integer.
  typedef signed   __int32   Int32;   ///< 32-bit   signed integer.
  typedef signed   __int64   Int64;   ///< 32-bit   signed integer.
  typedef unsigned __int8    Uint8;   ///<  8-bit unsigned integer.
  typedef unsigned __int16   Uint16;  ///< 16-bit unsigned integer.
  typedef unsigned __int32   Uint32;  ///< 32-bit unsigned integer.
  typedef unsigned __int64   Uint64;  ///< 32-bit unsigned integer.

  typedef float              Float32; ///< 32-bit float.
  typedef double             Float64; ///< 64-bit float.

  typedef bool               Bool;    ///< Boolean value
#endif

#if defined(DP_OS_LINUX)
  typedef int8_t             Int8;    ///<  8-bit   signed integer.
  typedef int16_t            Int16;   ///< 16-bit   signed integer.
  typedef int32_t            Int32;   ///< 32-bit   signed integer.
  typedef int64_t            Int64;   ///< 32-bit   signed integer.
  typedef uint8_t            Uint8;   ///<  8-bit unsigned integer.
  typedef uint16_t           Uint16;  ///< 16-bit unsigned integer.
  typedef uint32_t           Uint32;  ///< 32-bit unsigned integer.
  typedef uint64_t           Uint64;  ///< 32-bit unsigned integer.

  typedef float              Float32; ///< 32-bit float.
  typedef double             Float64; ///< 64-bit float.

#if defined(Bool)
  #undef Bool
#endif
  typedef bool               Bool;    ///< Boolean value
#endif

  // make sure that the bit sizes are right:
  DP_STATIC_ASSERT( sizeof(  Int8) == 1);
  DP_STATIC_ASSERT( sizeof( Int16) == 2);
  DP_STATIC_ASSERT( sizeof( Int32) == 4);
  DP_STATIC_ASSERT( sizeof( Int64) == 8);
  DP_STATIC_ASSERT( sizeof(  Uint8) == 1);
  DP_STATIC_ASSERT( sizeof( Uint16) == 2);
  DP_STATIC_ASSERT( sizeof( Uint32) == 4);
  DP_STATIC_ASSERT( sizeof( Uint64) == 8);
  DP_STATIC_ASSERT( sizeof(Float32) == 4);
  DP_STATIC_ASSERT( sizeof(Float64) == 8);

    
  /** CheckLimit verifies that the value of type TIn does fit into type TOut 
    * 
    * signed bit loss:
    * in signed -> out signed: always okay
    * in signed -> out unsigned: might lose sign bit
    * in unsigned -> out signed: always okay
    * in unsigned -> out unsigned: always okay
    * 
    * digits loss:
    *  InLessEqualThanOut false, out has always enough bits to hold in
    *  InLessEqualThanOut true, verify that all enabled bits from in fit into out.
  **/
  template <bool InLessEqualThanOut, bool outSigned, bool inSigned, typename TOut, typename TIn>
  struct CheckLimit
  {
    static bool checkLimit( TIn in );
  };
    
  template <bool outSigned, bool inSigned, typename TOut, typename TIn>
  struct CheckLimit<true, outSigned, inSigned, TOut, TIn> 
  {
    static bool checkLimit( TIn /*in*/ )
    {
      return true;
    }
  };

  // Tin::digits < Tout::digits, out unsigned, in signed -> might lose sign bit
  template <typename TOut, typename TIn>
  struct CheckLimit<true, false, true, TOut, TIn>
  {
    static bool checkLimit( TIn in )
    {
      return in >= TIn( 0 );
    }
  };

  // TIn::digits > TOut::digits generic
  template <bool outSigned, bool inSigned, typename TOut, typename TIn>
  struct CheckLimit<false, outSigned, inSigned, TOut, TIn> 
  {
    static bool checkLimit( TIn in )
    {
      // since TIn has more digits than TOut the sign bit of TIn will be killed always. Thus it's possible to skip the special case (in >= 0).
      TIn const mask = (TIn(1) << (std::numeric_limits<TOut>::digits)) - 1;

      return ( (in & mask) == in );
    }
  };

  // Tin::digits > TOut::digits, out signed, in signed, simple mask test is not enough because negative numbers have the most significant bits set
  template <typename TOut, typename TIn>
  struct CheckLimit<false, true, true, TOut, TIn>
  {
    static bool checkLimit( TIn in )
    {
      return std::numeric_limits<TOut>::min() <= in && in <= std::numeric_limits<TOut>::max();
    }
  };

  template <typename TOut, typename TIn>
  struct CheckRange
  {
    static bool checkLimit( TIn const & in )
    {
      return CheckLimit<std::numeric_limits<TIn>::digits <= std::numeric_limits<TOut>::digits
                        , std::numeric_limits<TOut>::is_signed, std::numeric_limits<TIn>::is_signed, TOut, TIn>::checkLimit( in );
    }
  };

    
  template<typename TOut, typename TIn>
  inline TOut checked_cast_integer( TIn in )
  {

    // this is exclusively for TOut and Tin being integer
    DP_STATIC_ASSERT(std::numeric_limits<TOut>::is_integer && std::numeric_limits<TIn>::is_integer);
      
    if ( !CheckRange<TOut, TIn>::checkLimit( in ) )
    {
      throw std::runtime_error( "checked_cast detected that the range of the output type is not sufficient for the input value" ); 
    }
      
    return static_cast<TOut>(in);
  }

  // general implementation handle non-pure integer cases
  template <typename TOut, typename TIn, bool integer>
  struct Caster
  {
    TOut operator()(TIn in) { 
      // more checks on non-pure integer to be done
      return static_cast<TOut>(in); 
    }
  };

  // specialization for pure-integer conversion
  template <typename TOut, typename TIn>
  struct Caster<TOut, TIn, true>
  {
    TOut operator()(TIn in) { return checked_cast_integer<TOut>(in); }
  };

  template<typename TOut, typename TIn>
  inline TOut checked_cast( TIn in )
  {
    Caster<TOut, TIn, std::numeric_limits<TOut>::is_integer && std::numeric_limits<TIn>::is_integer> theCaster;
    return theCaster(in);
  }

  enum DataType
  {
      DT_UNSIGNED_INT_8
    , DT_UNSIGNED_INT_16
    , DT_UNSIGNED_INT_32
    , DT_UNSIGNED_INT_64
    , DT_INT_8
    , DT_INT_16
    , DT_INT_32
    , DT_INT_64
    , DT_FLOAT_16
    , DT_FLOAT_32
    , DT_FLOAT_64
    , DT_NUM_DATATYPES
    , DT_UNKNOWN
  };

  enum PixelFormat
  {
      PF_R
    , PF_RG
    , PF_RGB
    , PF_RGBA
    , PF_BGR
    , PF_BGRA
    , PF_LUMINANCE
    , PF_ALPHA
    , PF_LUMINANCE_ALPHA
    , PF_DEPTH_COMPONENT
    , PF_DEPTH_STENCIL
    , PF_STENCIL_INDEX
    , PF_NATIVE
    , PF_NUM_PIXELFORMATS
    , PF_UNKNOWN
  };

  inline size_t getSizeOf( DataType dataType )
  {
    switch( dataType )
    {
    case DT_INT_8:
    case DT_UNSIGNED_INT_8:
      return 1;
    case DT_INT_16:
    case DT_UNSIGNED_INT_16:
    case DT_FLOAT_16:
      return 2;
    case DT_INT_32:
    case DT_UNSIGNED_INT_32:
    case DT_FLOAT_32:
      return 4;
    case DT_INT_64:
    case DT_UNSIGNED_INT_64:
    case DT_FLOAT_64:
      return 8;
    case DT_UNKNOWN:
      return 0;
    default:
      DP_ASSERT( !"unsupported type" );
      return 0;
    }
  }

  inline unsigned char getComponentCount( PixelFormat pixelFormat)
  {
    switch( pixelFormat )
    {
    case PF_R:
    case PF_LUMINANCE:
    case PF_ALPHA:
    case PF_DEPTH_COMPONENT:
    case PF_STENCIL_INDEX:
      return 1;
    case PF_RG:
    case PF_LUMINANCE_ALPHA:
      return 2;
    case PF_RGB:
    case PF_BGR:
      return 3;
    case PF_RGBA:
    case PF_BGRA:
      return 4;
      // TODO
      /*
      case PF_NATIVE:
      return ;
      */
    default:
      DP_ASSERT( !"unsupported pixel format" );
      return 0;
    }
  }

  template <typename T> struct Type2EnumType        { static DataType const type = DT_UNKNOWN; };
  template <> struct Type2EnumType<char>            { static DataType const type = DT_INT_8; };
  template <> struct Type2EnumType<unsigned char>   { static DataType const type = DT_UNSIGNED_INT_8; };
  template <> struct Type2EnumType<short>           { static DataType const type = DT_INT_16; };
  template <> struct Type2EnumType<unsigned short>  { static DataType const type = DT_UNSIGNED_INT_16; };
  template <> struct Type2EnumType<int>             { static DataType const type = DT_INT_32; };
  template <> struct Type2EnumType<unsigned int>    { static DataType const type = DT_UNSIGNED_INT_32; };
  template <> struct Type2EnumType<float>           { static DataType const type = DT_FLOAT_32; };
  template <> struct Type2EnumType<double>          { static DataType const type = DT_FLOAT_64; };

  inline bool isIntegerType(DataType type)
  {
    return !(type == DT_FLOAT_16 || type == DT_FLOAT_32 || type == DT_FLOAT_64); 
  }

  inline bool isFloatingPointType(DataType type)
  {
    return !isIntegerType(type);
  }

  enum GeometryPrimitiveType
  {
      GPT_POINTS
    , GPT_LINE_STRIP
    , GPT_LINE_LOOP
    , GPT_LINES
    , GPT_TRIANGLE_STRIP
    , GPT_TRIANGLE_FAN
    , GPT_TRIANGLES
    , GPT_QUAD_STRIP
    , GPT_QUADS
    , GPT_POLYGON
    , GPT_PATCHES
    , GPT_NATIVE
    , GPT_NUM_PRIMITIVETYPES
  };

} // namespace dp
