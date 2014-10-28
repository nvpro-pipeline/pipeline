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

#include <dp/fx/EnumSpec.h>
#include <dp/fx/ParameterGroupSpec.h>
#include <dp/fx/EffectSpec.h>
#include <boost/scoped_array.hpp>

namespace dp
{
  namespace fx
  {

    class ParameterGroupData;
    typedef std::shared_ptr<ParameterGroupData> SmartParameterGroupData;

    class ParameterGroupData
    {
    public:
      DP_FX_API const dp::fx::SmartParameterGroupSpec & getParameterGroupSpec() const;
      DP_FX_API const std::string& getName() const;
      DP_FX_API const void * getParameter( dp::fx::ParameterGroupSpec::iterator it ) const;
      DP_FX_API virtual bool operator==( const ParameterGroupData& parameterGroupData ) const;

    protected:
      DP_FX_API ParameterGroupData( const SmartParameterGroupSpec& parameterGroupSpec, const std::string& name );
      DP_FX_API ~ParameterGroupData();

    protected:
      dp::fx::SmartParameterGroupSpec m_parameterGroupSpec;
      std::vector<char>               m_data;
      std::string                     m_name;
    };

  } // namespace fx
} // namespace dp
