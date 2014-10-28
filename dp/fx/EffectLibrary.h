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

#include <dp/fx/Config.h>
#include <dp/fx/EffectData.h>
#include <dp/fx/EffectDefs.h>
#include <dp/fx/EffectSpec.h>
#include <dp/fx/ParameterGroupData.h>
#include <dp/fx/ParameterGroupSpec.h>
#include <dp/fx/ParameterGroupLayout.h>
#include <dp/fx/inc/Snippet.h>

#include <map>
#include <string>
#include <vector>

namespace dp
{
  namespace fx
  {
    SMART_TYPES( ShaderPipeline );

    class ShaderPipeline
    {
    public:
      typedef std::vector<SmartParameterGroupSpec> ParameterGroupSpecContainer;

      struct Stage
      {
        Domain       domain;
        SmartSnippet source;
        std::string  entrypoint;
        std::vector<SmartParameterGroupSpec> parameterGroupSpecs;
        std::vector<std::string>             systemSpecs;
      };

      typedef std::vector<Stage> Stages;
      typedef Stages::const_iterator iterator;

      DP_FX_API iterator beginStages() const;
      DP_FX_API iterator endStages() const;
      DP_FX_API iterator getStage( Domain domain );

    protected:
      DP_FX_API ShaderPipeline();

    protected:
      Stages m_stages;
    };

    // TODO hide implementation details
    class ShaderPipelineConfiguration
    {
    public:
      struct SpecInfo
      {
        SpecInfo() {}

        SpecInfo( SmartEffectSpec const& ds, SmartEffectSpec const& ss)
          : domainSpec( ds )
          , systemSpec( ss )
        {
        }

        SmartEffectSpec domainSpec;
        SmartEffectSpec systemSpec;
      };

      typedef std::map< Domain, SpecInfo > EffectSpecPerDomain;

      DP_FX_API ShaderPipelineConfiguration( std::string const& name );

      DP_FX_API std::string const& getName() const;

      DP_FX_API void addEffectSpec( Domain domain, SmartEffectSpec const& effectSpec, SmartEffectSpec const& systemSpec = SmartEffectSpec() );
      DP_FX_API ShaderPipelineConfiguration::EffectSpecPerDomain getEffectSpecs() const;

      /** add a piece of sourcecode after the parameter declarations and before the rest of the shader**/
      DP_FX_API void addSourceCode( Domain domain, std::string const& code );
      DP_FX_API void addSourceCode( Domain domain, std::vector<std::string> const& code );

      DP_FX_API void setManager( Manager manager );
      DP_FX_API Manager getManager() const;
      DP_FX_API void setTechnique( std::string const& technique  );
      DP_FX_API std::string const& getTechnique() const;
      DP_FX_API std::vector<std::string> const& getSourceCodes( Domain domain ) const;

    protected:
      Manager m_manager;
      std::string m_technique;
      std::string m_name;

      typedef std::vector<std::string> StringVector;
      std::map<Domain, StringVector> m_codeSnippets;

      EffectSpecPerDomain m_effectSpecPerDomain;
    };

    class EffectLibrary
    {
    public:
      DP_FX_API virtual ~EffectLibrary();

      static DP_FX_API EffectLibrary* instance();

      DP_FX_API virtual bool loadEffects(const std::string& filename, const std::vector<std::string> &searchPaths = std::vector<std::string>() ) = 0;
      DP_FX_API virtual bool save( const SmartEffectData& effectData, const std::string& filename ) = 0;

      DP_FX_API virtual void getEffectNames( std::vector<std::string>& names ) = 0;
      DP_FX_API virtual void getEffectNames( const std::string & filename, EffectSpec::Type type, std::vector<std::string> & names ) const = 0;

      DP_FX_API virtual const SmartEffectSpec& getEffectSpec( std::string const& effectName ) const = 0;
      DP_FX_API virtual std::string const& getEffectFile( std::string const& effectName ) const = 0;

      DP_FX_API virtual const SmartParameterGroupSpec & getParameterGroupSpec( const std::string & pgsName ) const = 0;

      DP_FX_API virtual const SmartEnumSpec & getEnumSpec( const std::string & enumName ) const = 0;

      DP_FX_API virtual dp::fx::SmartParameterGroupLayout getParameterGroupLayout( const dp::fx::SmartParameterGroupSpec& spec, dp::fx::Manager manager );

      DP_FX_API virtual const SmartParameterGroupData& getParameterGroupData( const std::string& name ) const = 0;

      DP_FX_API virtual const SmartEffectData& getEffectData( const std::string& name ) const = 0;

      DP_FX_API virtual SmartShaderPipeline generateShaderPipeline( const dp::fx::ShaderPipelineConfiguration& configuration ) = 0;

      DP_FX_API virtual std::vector<std::string> getRegisteredExtensions() const = 0;

      DP_FX_API virtual bool effectHasTechnique( SmartEffectSpec const& effectSpec, std::string const& techniqueName, bool rasterizer ) const = 0;

    }; // EffectLibrary

  } // fx
} // dp

