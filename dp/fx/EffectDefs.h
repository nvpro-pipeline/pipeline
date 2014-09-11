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
  namespace fx
  {

    enum Domain
    {
        // GLSL
        DOMAIN_VERTEX
      , DOMAIN_TESSELLATION_CONTROL
      , DOMAIN_TESSELLATION_EVALUATION
      , DOMAIN_GEOMETRY
      , DOMAIN_FRAGMENT
      // Generic
      , DOMAIN_PIPELINE
    };
    
    enum Manager
    {   // GLSL
        MANAGER_UNIFORM
      , MANAGER_UNIFORM_BUFFER_OBJECT_RIX
      , MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX
      , MANAGER_SHADER_STORAGE_BUFFER_OBJECT
      , MANAGER_SHADER_STORAGE_BUFFER_OBJECT_RIX
      , MANAGER_SHADERBUFFER
        // Unknown
      , MANAGER_UNKNOWN
    };


    inline std::string getDomainName( Domain domain )
    {
      switch ( domain )
      {
      case DOMAIN_VERTEX:                  return "vertex";
      case DOMAIN_FRAGMENT:                return "fragment";
      case DOMAIN_GEOMETRY:                return "geometry";
      case DOMAIN_TESSELLATION_CONTROL:    return "tessellation_control";
      case DOMAIN_TESSELLATION_EVALUATION: return "tessellation_evaluation";
      // DAR FIXME Do I need the invented "light" domain here?
      default:                             return "unknown";
      }
    }
  }
}

