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

#include <dp/fx/inc/EffectDataPrivate.h>
#include <dp/fx/inc/EffectLoader.h>
#include <dp/fx/inc/ParameterGroupDataPrivate.h>
#include <dp/fx/inc/Snippet.h>
#include <dp/fx/xml/Config.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/fx/EffectSpec.h>
#include <dp/fx/ParameterGroupSpec.h>
#include <dp/rix/core/RiX.h>
#include <dp/util/FileFinder.h>
#include <dp/util/SharedPtr.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

class TiXmlDocument;
class TiXmlElement;

namespace dp
{
  namespace fx
  {
    namespace xml
    {

      /************************************************************************/
      /* Technique                                                            */
      /************************************************************************/
      DEFINE_PTR_TYPES( Technique );

      class Technique
      {
      public:
        typedef std::map<std::string, dp::fx::SnippetSharedPtr> SignatureSnippets;
        typedef std::map<dp::fx::Domain, SignatureSnippets> DomainSignatures;

      public:
        static TechniqueSharedPtr create( std::string const& type );
        std::string const & getType() const;

        void addDomainSnippet( dp::fx::Domain domain, std::string const & signature, dp::fx::SnippetSharedPtr const & snippet );
        SignatureSnippets const & getSignatures( dp::fx::Domain domain ) const;

      protected:
        Technique( std::string const & type);

      private:
        std::string    m_type;
        DomainSignatures m_domainSignatures;
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
      /* DomainData                                                           */
      /************************************************************************/
      DEFINE_PTR_TYPES( DomainData );

      class DomainData
      {
      public:
        static DomainDataSharedPtr create( DomainSpecSharedPtr const & domainSpec, std::string const & name, std::vector<dp::fx::ParameterGroupDataSharedPtr> const & parameterGroupDatas, bool transparent );

        DomainSpecSharedPtr const & getDomainSpec() const;
        ParameterGroupDataSharedPtr const & getParameterGroupData( DomainSpec::ParameterGroupSpecsContainer::const_iterator it ) const;
        std::string const & getName() const;
        bool getTransparent() const;

      protected:
        DomainData( DomainSpecSharedPtr const & domainSpec, std::string const & name, std::vector<dp::fx::ParameterGroupDataSharedPtr> const & parameterGroupDatas, bool transparent );

      private:
        DomainSpecSharedPtr                               m_domainSpec;
        boost::scoped_array<ParameterGroupDataSharedPtr>  m_parameterGroupDatas;
        std::string                                       m_name;
        bool                                              m_transparent;
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
        DP_FX_XML_API static EffectLoaderSharedPtr create( EffectLibraryImpl * effectLibrary );
        DP_FX_XML_API ~EffectLoader();

        DP_FX_XML_API virtual bool loadEffects( const std::string& filename );
        DP_FX_XML_API virtual bool save( const EffectDataSharedPtr& effectData, const std::string& filename );

        DP_FX_XML_API virtual bool getShaderSnippets( const dp::fx::ShaderPipelineConfiguration & configuration
                                                    , dp::fx::Domain domain
                                                    , std::string& entrypoint
                                                    , std::vector<dp::fx::SnippetSharedPtr> & snippets );

        DP_FX_XML_API virtual ShaderPipelineSharedPtr generateShaderPipeline( const dp::fx::ShaderPipelineConfiguration& configuration );
        DP_FX_XML_API virtual bool effectHasTechnique( dp::fx::EffectSpecSharedPtr const& effectSpec, std::string const& techniqueName, bool rasterizer );

      protected:
        DP_FX_XML_API EffectLoader( EffectLibraryImpl * effectLibrary );

      private:
        enum EffectElementType
        {
          EET_NONE,
          EET_ENUM,
          EET_EFFECT,
          EET_PARAMETER_GROUP,
          EET_PARAMETER,
          EET_PARAMETER_GROUP_DATA,
          EET_TECHNIQUE,
          EET_GLSL,
          EET_SOURCE,
          EET_INCLUDE,
          EET_PIPELINE_SPEC,
          EET_PIPELINE_DATA,
          EET_UNKNOWN
        };

      private:
        void parseLibrary( TiXmlElement * root );
        void parseEnum( TiXmlElement * element );
        void parseLightEffect( TiXmlElement * effect);
        void parseDomainSpec( TiXmlElement * effect);
        void parseEffect( TiXmlElement * effect );
        TechniqueSharedPtr parseTechnique( TiXmlElement *techique, dp::fx::Domain domain );
        ParameterGroupSpecSharedPtr parseParameterGroup( TiXmlElement * pg );
        void parseParameter( TiXmlElement * param, std::vector<ParameterSpec> & psc );
        void parseInclude( TiXmlElement * effect );

        /** \brief Parse sources within GLSL/CUDA tag **/
        SnippetSharedPtr parseSources( TiXmlElement * effect );

        dp::fx::ParameterGroupDataSharedPtr parseParameterGroupData( TiXmlElement * pg );
        void parseParameterData( TiXmlElement * param, const dp::fx::ParameterGroupDataPrivateSharedPtr& pgd );

        // pipeline
        void parsePipelineSpec( TiXmlElement * effect );
        void parsePipelineData( TiXmlElement * effect );

        EffectElementType getTypeFromElement(TiXmlElement *element);
        unsigned int getParameterTypeFromGLSLType(const std::string &glslType);

        SnippetSharedPtr getSourceSnippet( std::string const & filename );
        SnippetSharedPtr getParameterSnippet( std::string const & inout, std::string const & type, TiXmlElement *element );

      private:
        typedef std::map<std::string, DomainSpecSharedPtr> DomainSpecs;
        typedef std::map<std::string, DomainDataSharedPtr> DomainDatas;
        DomainSpecs m_domainSpecs;
        DomainDatas m_domainDatas;
 
        // Explicitly chosen name to reduce confusion with two other typedefs named ParameterGroupDatas.
        typedef std::map<std::string, ParameterGroupDataSharedPtr> ParameterGroupDataLookup;
        ParameterGroupDataLookup m_parameterGroupDataLookup;

        std::set<std::string> m_loadedFiles;

        dp::util::FileFinder  m_fileFinder;

        std::map<std::string, unsigned int> m_mapGLSLtoPT;
        std::map<unsigned int, std::string> m_mapPTtoGLSL;
      }; // EffectLibrary
    }
  } // fx
} // dp

