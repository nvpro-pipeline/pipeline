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


#include <dp/util/File.h>
#include <dp/util/Singleton.h>
#include <dp/util/Tokenizer.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/fx/glsl/UniformGenerator.h>
#include <dp/fx/glsl/UniformGeneratorGLSLStandard.h>
#include <dp/fx/glsl/UniformGeneratorGLSLShaderBufferLoad.h>
#include <dp/fx/glsl/UniformGeneratorUBOStd140.h>
#include <dp/fx/glsl/UniformGeneratorSSBOStd140.h>
#include <dp/fx/inc/EffectLibraryImpl.h>
#include <boost/bind.hpp>
#include <utility>

using std::string;

namespace dp
{
  namespace fx
  {

    /************************************************************************/
    /* ShaderPipeline                                                       */
    /************************************************************************/

    ShaderPipeline::iterator ShaderPipeline::beginStages() const
    {
      return m_stages.begin();
    }

    ShaderPipeline::iterator ShaderPipeline::endStages() const
    {
      return m_stages.end();
    }
    
    ShaderPipeline::iterator ShaderPipeline::getStage( Domain domain )
    { 
      return std::find_if( beginStages(), endStages(), boost::bind( &Stage::domain, _1) == domain);
    }

    /************************************************************************/
    /* ShaderPipelineConfiguration                                          */
    /************************************************************************/

    ShaderPipelineConfiguration::ShaderPipelineConfiguration( std::string const& name )
      : m_manager( MANAGER_UNIFORM )
      , m_name( name )
    {
    }

    std::string const& ShaderPipelineConfiguration::getName() const
    {
      return( m_name );
    }

    void ShaderPipelineConfiguration::addEffectSpec( Domain domain, SmartEffectSpec const& effectSpec, SmartEffectSpec const& systemSpec )
    {
      DP_ASSERT( !effectSpec || ( effectSpec->getType() != EffectSpec::EST_PIPELINE ) );
      if ( effectSpec || systemSpec )
      {
        m_effectSpecPerDomain[domain] = SpecInfo( effectSpec, systemSpec );
      }
    }

    void ShaderPipelineConfiguration::addSourceCode( Domain domain, std::string const& code )
    {
      m_codeSnippets[domain].push_back(code);
    }

    void ShaderPipelineConfiguration::addSourceCode( Domain domain, std::vector<std::string> const& code )
    {
      std::copy( code.begin(), code.end(), std::back_inserter(m_codeSnippets[domain]) );
    }

    std::vector<std::string> const& ShaderPipelineConfiguration::getSourceCodes( Domain domain ) const
    {
      static std::vector<std::string> empty;
      std::map<Domain, StringVector>::const_iterator it = m_codeSnippets.find( domain );
      return it != m_codeSnippets.end() ? it->second : empty;
    }


    void ShaderPipelineConfiguration::setManager( Manager manager )
    {
      m_manager = manager;
    }

    Manager ShaderPipelineConfiguration::getManager() const
    {
      return m_manager;
    }

    void ShaderPipelineConfiguration::setTechnique( std::string const& technique  )
    {
      m_technique = technique;
    }
    
    std::string const& ShaderPipelineConfiguration::getTechnique() const
    {
      return m_technique;
    }


    /************************************************************************/
    /* ShaderPipelineImpl                                                   */
    /************************************************************************/
    class ShaderPipelineImpl : public ShaderPipeline
    {
    public:
      void addStage( const Stage& stage )
      {
        DP_ASSERT( std::find_if( m_stages.begin(), m_stages.end(), boost::bind( &Stage::domain, _1) == stage.domain ) == m_stages.end() );
        m_stages.push_back( stage );
      }
    };

    typedef dp::util::SmartPtr<ShaderPipelineImpl> SmartShaderPipelineImpl;

    /************************************************************************/
    /* EffectLibrary                                                        */
    /************************************************************************/
    EffectLibrary::~EffectLibrary()
    {
    }

    EffectLibrary* EffectLibrary::instance()
    {
      return EffectLibraryImpl::instance();
    }

    dp::fx::SmartParameterGroupLayout EffectLibrary::getParameterGroupLayout( const dp::fx::SmartParameterGroupSpec& spec, dp::fx::Manager manager )
    {
      switch ( manager )
      {
      case MANAGER_UNIFORM:
        return dp::fx::glsl::UniformGeneratorGLSLStandard().getParameterGroupLayout(spec);
      case MANAGER_SHADERBUFFER:
        return dp::fx::glsl::UniformGeneratorGLSLShaderBufferLoad().getParameterGroupLayout(spec);
      case MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX:
        return dp::fx::glsl::UniformGeneratorUBOStd140(true).getParameterGroupLayout(spec);
      case MANAGER_UNIFORM_BUFFER_OBJECT_RIX:
        return dp::fx::glsl::UniformGeneratorUBOStd140(false).getParameterGroupLayout(spec);
      case MANAGER_SHADER_STORAGE_BUFFER_OBJECT:
        return dp::fx::glsl::UniformGeneratorSSBOStd140(true).getParameterGroupLayout(spec);
      case MANAGER_SHADER_STORAGE_BUFFER_OBJECT_RIX:
        return dp::fx::glsl::UniformGeneratorSSBOStd140(false).getParameterGroupLayout(spec);
      default:
        DP_ASSERT(!"Unsupported manager");
        return SmartParameterGroupLayout::null;
      }
    }


  } // fx
} // dp


