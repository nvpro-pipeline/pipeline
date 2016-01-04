// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <string>

namespace dp
{
  namespace util
  {
    /** Supported semantics **/
    enum class Semantic
    {
        VALUE        //!< Element does not have a special semantic
      , POSITION     //!< Element is a position
      , DIRECTION    //!< Element is a direction, i.e. for directional lights
      , ORIENTATION  //!< Element is an orientation
      , DIMENSION    //!< Element is a dimension
      , SCALING      //!< Element is a scaling factor
      , COLOR        //!< Element is a color
      , OBJECT       //!< Element is an object
    };

    inline std::string semanticToString( Semantic semantic )
    {
      switch( semantic )
      {
        case Semantic::VALUE:
          return "VALUE";
        case Semantic::POSITION:
          return "POSITION";
        case Semantic::DIRECTION:
          return "DIRECTION";
        case Semantic::ORIENTATION:
          return "ORIENTATION";
        case Semantic::SCALING:
          return "SCALING";
        case Semantic::COLOR:
          return "COLOR";
        case Semantic::OBJECT:
          return "OBJECT";
        default:
          return "UNKNOWN";
      }
    }

    inline Semantic stringToSemantic( const std::string & semantic )
    {
      if ( semantic == "COLOR" )
      {
        return( Semantic::COLOR );
      }
      else if ( semantic == "DIRECTION" )
      {
        return( Semantic::DIRECTION );
      }
      else if ( semantic == "OBJECT" )
      {
        return( Semantic::OBJECT );
      }
      else if ( semantic == "ORIENTATION" )
      {
        return( Semantic::ORIENTATION );
      }
      else if ( semantic == "POSITION" )
      {
        return( Semantic::POSITION );
      }
      else if ( semantic == "SCALING" )
      {
        return( Semantic::SCALING );
      }
      else if ( ( semantic == "VALUE" ) || semantic.empty() )
      {
        return( Semantic::VALUE );
      }
      else
      {
        DP_ASSERT( !"unknown semantic string" );
        return( Semantic::VALUE );
      }
    }

  } // namespace util
} // namespace dp
