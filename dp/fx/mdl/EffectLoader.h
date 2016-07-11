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

#include <dp/fx/mdl/Config.h>
#include <dp/fx/mdl/inc/MaterialBuilder.h>
#include <dp/fx/inc/EffectLoader.h>

namespace dp
{
  namespace fx
  {
    namespace mdl
    {

      /************************************************************************/
      /* Technique                                                            */
      /************************************************************************/
      DEFINE_PTR_TYPES( Technique );

      class Technique
      {
      public:
        typedef std::map<std::string, dp::fx::SnippetSharedPtr> SignatureSnippets;

      public:
        static TechniqueSharedPtr create( std::string const& type );
        std::string const & getType() const;

        void addSnippet( std::string const & signature, dp::fx::SnippetSharedPtr const & snippet );
        SignatureSnippets const & getSnippets() const;

      protected:
        Technique( std::string const & type );

      private:
        std::string       m_type;
        SignatureSnippets m_snippets;
      };

      /************************************************************************/
      /* DomainSpec                                                           */
      /************************************************************************/
      DEFINE_PTR_TYPES( DomainSpec );

      class DomainSpec
      {
        public:
          typedef std::map<std::string, TechniqueSharedPtr> Techniques;
          typedef std::vector<ParameterGroupSpecSharedPtr> ParameterGroupSpecsContainer;

        public:
          static DomainSpecSharedPtr create( std::string const & name, dp::fx::Domain domain, ParameterGroupSpecsContainer const & specs, bool transparent, Techniques const & techniques );

          TechniqueSharedPtr getTechnique( std::string const & name );
          ParameterGroupSpecsContainer const & getParameterGroups() const;
          dp::fx::Domain getDomain() const;
          bool isTransparent() const;

        protected:
          DomainSpec( std::string const & name, dp::fx::Domain domain, ParameterGroupSpecsContainer const & specs, bool transparent, Techniques const & techniques );

        private:
          std::string                  m_name;
          dp::fx::Domain               m_domain;
          ParameterGroupSpecsContainer m_parameterGroups;
          bool                         m_transparent;
          Techniques                   m_techniques;
      };

      /************************************************************************/
      /* EffectSpec                                                           */
      /************************************************************************/
      DEFINE_PTR_TYPES( EffectSpec );

      class EffectSpec : public dp::fx::EffectSpec
      {
        public:
          typedef std::map<dp::fx::Domain, DomainSpecSharedPtr> DomainSpecs;

        public:
          static EffectSpecSharedPtr create( std::string const & name, DomainSpecs const & domainSpecs );

          DomainSpecs const & getDomainSpecs() const;
          DomainSpecSharedPtr const & getDomainSpec( dp::fx::Domain stageSpec ) const;

        protected:
          EffectSpec( std::string const & name, DomainSpecs const & domainSpecs );

        private:
          static ParameterGroupSpecsContainer gatherParameterGroupSpecs( DomainSpecs const & domainSpecs );
          static bool                         gatherTransparency( DomainSpecs const & domainSpecs );

          DomainSpecs m_domainSpecs;
      };


      DEFINE_PTR_TYPES( EffectLoader );

      class EffectLoader : public dp::fx::EffectLoader
      {
      public:
        DP_FX_MDL_API static EffectLoaderSharedPtr create( EffectLibraryImpl * effectLibrary );
        DP_FX_MDL_API virtual ~EffectLoader();

        DP_FX_MDL_API virtual bool loadEffects( std::string const& filename, dp::util::FileFinder const& fileFinder );
        DP_FX_MDL_API virtual bool save( const EffectDataSharedPtr& effectData, const std::string& filename );

        DP_FX_MDL_API virtual bool getShaderSnippets( const dp::fx::ShaderPipelineConfiguration & configuration
          , dp::fx::Domain domain
          , std::string& entrypoint
          , std::vector<dp::fx::SnippetSharedPtr> & snippets );

        DP_FX_MDL_API virtual ShaderPipelineSharedPtr generateShaderPipeline( const dp::fx::ShaderPipelineConfiguration& configuration );
        DP_FX_MDL_API virtual bool effectHasTechnique( dp::fx::EffectSpecSharedPtr const& effectSpec, std::string const& techniqueName, bool rasterizer );

      protected:
        DP_FX_MDL_API EffectLoader( EffectLibraryImpl * effectLibrary );

      private:
        std::set<std::string> m_loadedFiles;
        MaterialBuilder       m_materialBuilder;
      }; // EffectLoader

    } // mdl
  } // fx
} // dp

