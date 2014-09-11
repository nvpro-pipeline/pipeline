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

#include <string>

namespace dp
{
  namespace util
  {
    /** Supported semantics **/
    enum Semantic
    {
        SEMANTIC_VALUE        //!< Element does not have a special semantic
      , SEMANTIC_POSITION     //!< Element is a position
      , SEMANTIC_DIRECTION    //!< Element is a direction, i.e. for directional lights
      , SEMANTIC_ORIENTATION  //!< Element is an orientation
      , SEMANTIC_DIMENSION    //!< Element is a dimension
      , SEMANTIC_SCALING      //!< Element is a scaling factor
      , SEMANTIC_COLOR        //!< Element is a color
      , SEMANTIC_OBJECT       //!< Element is an object
    };

    inline std::string semanticToString( Semantic semantic )
    {
      switch( semantic )
      {
        case SEMANTIC_VALUE:
          return "VALUE";
        case SEMANTIC_POSITION:
          return "POSITION";
        case SEMANTIC_DIRECTION:
          return "DIRECTION";
        case SEMANTIC_ORIENTATION:
          return "ORIENTATION";
        case SEMANTIC_SCALING:
          return "SCALING";
        case SEMANTIC_COLOR:
          return "COLOR";
        case SEMANTIC_OBJECT:
          return "OBJECT";
        default:
          return "UNKNOWN";
      }
    }

    inline Semantic stringToSemantic( const std::string & semantic )
    {
      if ( semantic == "COLOR" )
      {
        return( SEMANTIC_COLOR );
      }
      else if ( semantic == "DIRECTION" )
      {
        return( SEMANTIC_DIRECTION );
      }
      else if ( semantic == "OBJECT" )
      {
        return( SEMANTIC_OBJECT );
      }
      else if ( semantic == "ORIENTATION" )
      {
        return( SEMANTIC_ORIENTATION );
      }
      else if ( semantic == "POSITION" )
      {
        return( SEMANTIC_POSITION );
      }
      else if ( semantic == "SCALING" )
      {
        return( SEMANTIC_SCALING );
      }
      else if ( ( semantic == "VALUE" ) || semantic.empty() )
      {
        return( SEMANTIC_VALUE );
      }
      else
      {
        DP_ASSERT( !"unknown semantic string" );
        return( SEMANTIC_VALUE );
      }
    }

  } // namespace util
} // namespace dp
