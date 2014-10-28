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

#include <dp/fx/EffectDefs.h>
#include <dp/fx/EnumSpec.h>
#include <memory>
#include <set>

namespace dp
{
  namespace fx
  {

    class GeneratorConfiguration
    {
    public:
      dp::fx::Manager manager;
    };

    class Snippet
    {
    public:
      DP_FX_API virtual std::string getSnippet( GeneratorConfiguration& config ) = 0;

      DP_FX_API bool addRequiredEnumSpec( const SmartEnumSpec & enumSpec );
      DP_FX_API const std::set<SmartEnumSpec> & getRequiredEnumSpecs() const;

    private:
      std::set<SmartEnumSpec>  m_requiredEnumSpecs;
    };

    typedef std::shared_ptr<Snippet> SmartSnippet;

    DP_FX_API std::string generateSnippets( std::vector<SmartSnippet> snippets, GeneratorConfiguration& config );


  } // namespace fx
} // namespace dp
