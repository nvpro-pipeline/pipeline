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


#include <dp/DP.h>
#include <dp/fx/mdl/EffectLoader.h>
#include <dp/fx/inc/EffectLibraryImpl.h>
#include <dp/fx/inc/EffectDataPrivate.h>
#include <dp/fx/inc/FileSnippet.h>
#include <dp/fx/inc/ParameterGroupDataPrivate.h>
#include <dp/fx/inc/SnippetListSnippet.h>
#include <dp/fx/inc/StringSnippet.h>
#include <dp/util/File.h>
#include <boost/algorithm/string/replace.hpp>
#include <boost/bind.hpp>

namespace dp
{
  namespace fx
  {
    namespace mdl
    {

      /************************************************************************/
      /* ShaderPipelineImpl                                                   */
      /************************************************************************/
      DEFINE_PTR_TYPES( ShaderPipelineImpl );

      class ShaderPipelineImpl : public ShaderPipeline
      {
        public:
          static ShaderPipelineImplSharedPtr create()
          {
            return( std::shared_ptr<ShaderPipelineImpl>( new ShaderPipelineImpl() ) );
          }

          void addStage( const Stage& stage )
          {
            DP_ASSERT( std::find_if( m_stages.begin(), m_stages.end(), boost::bind( &Stage::domain, _1) == stage.domain ) == m_stages.end() );
            m_stages.push_back( stage );
          }

        protected:
          ShaderPipelineImpl()
          {}
      };

      /************************************************************************/
      /* Technique                                                            */
      /************************************************************************/
      TechniqueSharedPtr Technique::create( std::string const& type )
      {
        return( std::shared_ptr<Technique>( new Technique( type ) ) );
      }

      Technique::Technique( std::string const & type)
        : m_type( type )
      {
      }

      void Technique::addSnippet( std::string const & signature, dp::fx::SnippetSharedPtr const & snippet )
      {
        if ( m_snippets.find( signature ) != m_snippets.end() )
        {
          throw std::runtime_error( std::string( "signature " + signature + " has already been added to the domain" ) );
        }
        m_snippets[signature] = snippet;
      }

      Technique::SignatureSnippets const & Technique::getSnippets() const
      {
        return( m_snippets );
      }

      std::string const & Technique::getType() const
      {
        return m_type;
      }

      /************************************************************************/
      /* DomainSpec                                                           */
      /************************************************************************/
      DomainSpecSharedPtr DomainSpec::create( std::string const & name, dp::fx::Domain domain, ParameterGroupSpecsContainer const & specs, bool transparent, Techniques const & techniques )
      {
        return( std::shared_ptr<DomainSpec>( new DomainSpec( name, domain, specs, transparent, techniques ) ) );
      }

      DomainSpec::DomainSpec( std::string const & name, dp::fx::Domain domain, ParameterGroupSpecsContainer const & specs, bool transparent, Techniques const & techniques )
        : m_name( name )
        , m_domain( domain )
        , m_parameterGroups( specs )
        , m_transparent( transparent )
        , m_techniques( techniques)
      {

      }

      TechniqueSharedPtr DomainSpec::getTechnique( std::string const & name )
      {
        Techniques::const_iterator it = m_techniques.find( name );
        if ( it != m_techniques.end() )
        {
          return it->second;
        }
        // The DomainSpec's technique doesn't match the queried one.
        // Return nullptr so that it's going to be ignored. 
        return TechniqueSharedPtr();
      }

      DomainSpec::ParameterGroupSpecsContainer const & DomainSpec::getParameterGroups() const
      {
        return m_parameterGroups;
      }

      dp::fx::Domain DomainSpec::getDomain() const
      {
        return m_domain;
      }

      bool DomainSpec::isTransparent() const
      {
        return m_transparent;
      }

      /************************************************************************/
      /* EffectSpec                                                           */
      /************************************************************************/
      EffectSpecSharedPtr EffectSpec::create( std::string const & name, DomainSpecs const & domainSpecs )
      {
        return( std::shared_ptr<EffectSpec>( new EffectSpec( name, domainSpecs ) ) );
      }

      EffectSpec::EffectSpec( std::string const & name, DomainSpecs const & domainSpecs )
        : dp::fx::EffectSpec( name, Type::PIPELINE, gatherParameterGroupSpecs( domainSpecs ), gatherTransparency( domainSpecs ) )
        , m_domainSpecs( domainSpecs )
      {
      }

      EffectSpec::DomainSpecs const & EffectSpec::getDomainSpecs() const
      {
        return m_domainSpecs;
      }

      DomainSpecSharedPtr const & EffectSpec::getDomainSpec( dp::fx::Domain domainSpec ) const
      {
        DomainSpecs::const_iterator it = m_domainSpecs.find( domainSpec );
        if ( m_domainSpecs.end() == it )
        {
          throw std::runtime_error( "missing DomainSpec for given domain" );
        }

        return it->second;
      }

      EffectSpec::ParameterGroupSpecsContainer EffectSpec::gatherParameterGroupSpecs( DomainSpecs const & domainSpecs )
      {
        // gather all ParameterGroupSpecs and ensure that each one exists only once
        std::set<dp::fx::ParameterGroupSpecSharedPtr> gatheredSpecs;
        for ( DomainSpecs::const_iterator it = domainSpecs.begin(); it != domainSpecs.end(); ++it )
        {
          for (DomainSpec::ParameterGroupSpecsContainer::const_iterator it2 = it->second->getParameterGroups().begin(); it2 != it->second->getParameterGroups().end(); ++it2 )
          {
            gatheredSpecs.insert(*it2);
          }
        }

        ParameterGroupSpecsContainer returnValue;
        std::copy( gatheredSpecs.begin(), gatheredSpecs.end(), std::back_inserter( returnValue ) );
        return returnValue;
      }

      bool EffectSpec::gatherTransparency( DomainSpecs const & domainSpecs )
      {
        bool transparency = false;
        for ( DomainSpecs::const_iterator it = domainSpecs.begin(); it != domainSpecs.end(); ++it )
        {
          transparency |= it->second->isTransparent();
        }

        return transparency;
      }

      EffectLoaderSharedPtr EffectLoader::create( EffectLibraryImpl * effectLibrary )
      {
        return( std::shared_ptr<EffectLoader>( new EffectLoader( effectLibrary ) ) );
      }

      EffectLoader::EffectLoader( EffectLibraryImpl * effectLibrary )
        : dp::fx::EffectLoader( effectLibrary )
        , m_materialBuilder( dp::home() + "/media/dpfx/mdl/MDL.cfg" )
      {
      }

      EffectLoader::~EffectLoader()
      {
      }

      dp::fx::SnippetSharedPtr buildParameterSnippet( std::string const& type, std::string const& name, unsigned int location )
      {
        std::ostringstream oss;
        oss << "layout(location = " << location << ") in " << type << " " << name << ";" << std::endl;
        return( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
      }

      void buildParameterSnippets( std::vector<dp::fx::SnippetSharedPtr> & snippets )
      {
        snippets.push_back( buildParameterSnippet( "vec4", "attrPosition", 0 ) );
        snippets.push_back( buildParameterSnippet( "vec3", "attrNormal", 2 ) );
        snippets.push_back( buildParameterSnippet( "vec3", "attrTexCoord0", 8 ) );
        snippets.push_back( buildParameterSnippet( "vec3", "attrTangent", 14 ) );
        snippets.push_back( buildParameterSnippet( "vec3", "attrBinormal", 15 ) );
      }

      void buildEnumsSnippet( std::vector<dp::fx::SnippetSharedPtr> & snippets, std::set<std::string> const& enums, std::map<std::string,EnumData> const& specs )
      {
        if ( !enums.empty() )
        {
          std::ostringstream oss;
          oss << "// Enumerations section" << std::endl;
          for ( std::set<std::string>::const_iterator it = enums.begin(); it != enums.end(); ++it )
          {
            std::map<std::string,EnumData>::const_iterator sit = specs.find( *it );
            DP_ASSERT( sit != specs.end() );
            oss << "// enum " << *it << ":" << std::endl;
            oss << "#define " << *it << " int" << std::endl;
            for ( size_t i = 0; i<sit->second.values.size() ;  i++ )
            {
              oss << "#define " << sit->second.values[i].first << " " << sit->second.values[i].second << std::endl;
            }
            oss << std::endl;
          }
          oss << std::endl;
          snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
        }
      }

      void buildGlobalsSnippet( std::vector<dp::fx::SnippetSharedPtr> & snippets, dp::fx::Domain domain )
      {
        std::ostringstream oss;
        oss << std::endl
            << "// Global variables" << std::endl
            << "vec3 stateNormal;" << std::endl
            << "vec3 texCoord0;" << std::endl
            << "vec3 tangent;" << std::endl
            << "vec3 binormal;" << std::endl;
        if ( domain == dp::fx::Domain::VERTEX )
        {
          oss << "vec4 worldPos;" << std::endl
              << std::endl
              << "out gl_PerVertex" << std::endl
              << "{" << std::endl
              << "  vec4 gl_Position;" << std::endl
              << "};" << std::endl
              << std::endl;
        }
        else
        {
          oss << "vec3 lightDir;" << std::endl
              << "vec3 lightDiffuse;" << std::endl
              << "vec3 lightSpecular;" << std::endl
              << "float materialIOR;" << std::endl
              << "vec3 viewDir;" << std::endl;
        }
        oss << std::endl;
        snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
      }

      std::string getVaryingType( std::string const& varyingName )
      {
        static const std::map<std::string,std::string> varyings =
        {
          { "varBinormal",    "vec3" },
          { "varEyePos",      "vec3" },
          { "varNormal",      "vec3" },
          { "varObjBinormal", "vec3" },
          { "varObjPos",      "vec3" },
          { "varObjTangent",  "vec3" },
          { "varTangent",     "vec3" },
          { "varTexCoord0",   "vec3" },
          { "varWorldPos",    "vec3" }
        };
        std::map<std::string, std::string>::const_iterator it = varyings.find( varyingName );
        DP_ASSERT( it != varyings.end() );
        return( it->second );
      }

      void buildVaryingsSnippet( std::vector<dp::fx::SnippetSharedPtr> & snippets, std::set<std::string> const& varyings, std::string const& prefix )
      {
        if ( !varyings.empty() )
        {
          std::ostringstream oss;
          oss << "// Varyings" << std::endl;
          for ( std::set<std::string>::const_iterator it = varyings.begin(); it != varyings.end(); ++it )
          {
            oss << prefix << " " << getVaryingType( *it ) << " " << *it << ";" << std::endl;
          }
          oss << std::endl;
          snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
        }
      }

      void buildStructuresSnippet( std::vector<dp::fx::SnippetSharedPtr> & snippets, std::set<std::string> const& structures, std::map<std::string, StructureData> const& structureSpecs )
      {
        if ( !structures.empty() )
        {
          std::ostringstream oss;
          oss << "// Structures section" << std::endl;
          for ( std::set<std::string>::const_iterator it = structures.begin(); it != structures.end(); ++it )
          {
            std::map<std::string, StructureData>::const_iterator specIt = structureSpecs.find( *it );
            DP_ASSERT( specIt != structureSpecs.end() );
            oss << "struct " << specIt->second.name << std::endl;
            oss << "{" << std::endl;
            for ( size_t i=0 ; i<specIt->second.members.size() ; i++ )
            {
              oss << "  " << specIt->second.members[i].first << " " << specIt->second.members[i].second << ";" << std::endl;
            }
            oss << "};" << std::endl;
            oss << std::endl;
          }
          oss << std::endl;
          snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
        }
      }

      void buildFunctionSnippets( std::vector<dp::fx::SnippetSharedPtr> & snippets, std::vector<std::string> const& functions )
      {
        for ( std::vector<std::string>::const_iterator it = functions.begin(); it != functions.end(); ++it )
        {
          snippets.push_back( std::make_shared<dp::fx::FileSnippet>( dp::home() + std::string( "/media/dpfx/mdl/" ) + *it + std::string( ".glsl" ) ) );
        }
      }

      void buildEvalGeometrySnippet( std::vector<dp::fx::SnippetSharedPtr> & snippets, dp::fx::Domain domain, GeometryData const& geometryData )
      {
        if ( domain == dp::fx::Domain::VERTEX )
        {
          std::ostringstream oss;
          oss << "vec4 evalWorldPos()" << std::endl
              << "{" << std::endl
              << "  return( sys_WorldMatrix * ( attrPosition + vec4( " << geometryData.displacement << ", 0.0f ) ) );" << std::endl
              << "}" << std::endl << std::endl;
          snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
        }
        else
        {
          {
            std::ostringstream oss;
            oss << "float evalCutoutOpacity( in vec3 normal )" << std::endl
                << "{" << std::endl
                << "  return( clamp( " << geometryData.cutoutOpacity << ", 0.0f, 1.0f ) );" << std::endl
                << "}" << std::endl << std::endl;
            snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
          }
          {
            std::ostringstream oss;
            oss << "vec3 evalNormal( in vec3 normal )" << std::endl
                << "{" << std::endl
                << "  return( " << geometryData.normal << " );" << std::endl
                << "}" << std::endl << std::endl;
            snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
          }
        }
      }

      void buildTemporariesSnippet(std::vector<dp::fx::SnippetSharedPtr> & snippets, std::set<unsigned int> const& temporaries, std::map<unsigned int, TemporaryData> const& temporariesSpecs)
      {
        if (!temporaries.empty())
        {
          std::ostringstream oss;
          oss << "// Temporaries Declarations" << std::endl;
          for (std::set<unsigned int>::const_iterator it = temporaries.begin(); it != temporaries.end(); ++it)
          {
            std::map<unsigned int, TemporaryData>::const_iterator tit = temporariesSpecs.find(*it);
            DP_ASSERT(tit != temporariesSpecs.end());
            oss << tit->second.type << " temporary" << tit->first << ";" << std::endl;
          }
          oss << std::endl;
          snippets.push_back(std::make_shared<dp::fx::StringSnippet>(oss.str()));
        }
      }

      std::string getVaryingEval( std::string const& varyingName )
      {
        static const std::map<std::string, std::string> varyings =
        {
          { "varBinormal",    "normalize( ( sys_WorldMatrix * vec4( attrBinormal, 0.0f ) ).xyz )"           },
          { "varEyePos",      "vec3( sys_ViewMatrixI[3][0], sys_ViewMatrixI[3][1], sys_ViewMatrixI[3][2] )" },
          { "varNormal",      "normalize( ( sys_WorldMatrixIT * vec4( attrNormal, 0.0f ) ).xyz )"           },
          { "varObjBinormal", "normalize( attrBinormal )"                                                   },
          { "varObjPos",      "attrPosition.xyz"                                                            },
          { "varObjTangent",  "normalize( attrTangent )"                                                    },
          { "varTangent",     "normalize( ( sys_WorldMatrix * vec4( attrTangent, 0.0f ) ).xyz )"            },
          { "varTexCoord0",   "attrTexCoord0"                                                               },
          { "varWorldPos",    "worldPos.xyz"                                                                }
        };
        std::map<std::string, std::string>::const_iterator it = varyings.find( varyingName );
        DP_ASSERT( it != varyings.end() );
        return( it->second );
      }

      void buildEvalTemporariesSnippet(std::vector<dp::fx::SnippetSharedPtr> & snippets, dp::fx::Domain domain, std::set<unsigned int> const& temporaries, std::map<unsigned int, TemporaryData> const& temporarySpecs)
      {
        if (domain == dp::fx::Domain::VERTEX)
        {
          std::ostringstream oss;
          oss << "void evalTemporaries( in vec3 normal )" << std::endl
            << "{" << std::endl;
          for (std::set<unsigned int>::const_iterator it = temporaries.begin(); it != temporaries.end(); ++it)
          {
            std::map<unsigned int, TemporaryData>::const_iterator tit = temporarySpecs.find(*it);
            DP_ASSERT(tit != temporarySpecs.end());
            if (!tit->second.eval.empty())
            {
              oss << "  temporary" << tit->first << " = " << tit->second.eval << ";" << std::endl;
            }
          }
          oss << "}" << std::endl << std::endl;
          snippets.push_back(std::make_shared<dp::fx::StringSnippet>(oss.str()));
        }
        else
        {
          DP_ASSERT(domain == dp::fx::Domain::FRAGMENT);

          // filter temporaries and temporaries per light source
          std::vector<unsigned int> t, tpls;
          for (std::set<unsigned int>::const_iterator it = temporaries.begin(); it != temporaries.end(); ++it)
          {
            std::map<unsigned int, TemporaryData>::const_iterator tit = temporarySpecs.find(*it);
            DP_ASSERT(tit != temporarySpecs.end());
            if (tit->second.eval.find("lightDiffuse") != std::string::npos)   // just take this one as criterion; might need more
            {
              tpls.push_back(tit->first);
            }
            else if (!tit->second.eval.empty())
            {
              // check if any temporaryPerLightSource is referenced in it
              bool found = false;
              for (size_t i = 0; !found && i<tpls.size(); ++i)
              {
                std::ostringstream oss;
                oss << "temporary" << tpls[i];
                if (tit->second.eval.find(oss.str()) != std::string::npos)
                {
                  tpls.push_back(tit->first);
                  found = true;
                }
              }
              if (!found)
              {
                t.push_back(tit->first);
              }
            }
          }

          {
            std::ostringstream oss;
            oss << "void evalTemporaries( in vec3 normal )" << std::endl
              << "{" << std::endl;
            for (size_t i = 0; i<t.size(); ++i)
            {
              std::map<unsigned int, TemporaryData>::const_iterator it = temporarySpecs.find(t[i]);
              DP_ASSERT(it != temporarySpecs.end());
              oss << "  temporary" << it->first << " = " << it->second.eval << ";" << std::endl;
            }
            // add Environment temporaries "per light source" here in the temporaries section !
            for (size_t i = 0; i<tpls.size(); ++i)
            {
              std::map<unsigned int, TemporaryData>::const_iterator it = temporarySpecs.find(tpls[i]);
              DP_ASSERT(it != temporarySpecs.end());
            }
            oss << "}" << std::endl << std::endl;
            snippets.push_back(std::make_shared<dp::fx::StringSnippet>(oss.str()));
          }

          {
            std::ostringstream oss;
            oss << "void evalTemporariesPerLightSource()" << std::endl
              << "{" << std::endl;
            for (size_t i = 0; i<tpls.size(); ++i)
            {
              std::map<unsigned int, TemporaryData>::const_iterator it = temporarySpecs.find(tpls[i]);
              DP_ASSERT(it != temporarySpecs.end());
              oss << "  temporary" << it->first << " = " << it->second.eval << ";" << std::endl;
            }
            oss << "}" << std::endl << std::endl;
            snippets.push_back(std::make_shared<dp::fx::StringSnippet>(oss.str()));
          }
        }
      }

      void buildEvalVaryingsSnippet(std::vector<dp::fx::SnippetSharedPtr> & snippets, std::set<std::string> const& varyings)
      {
        std::ostringstream oss;
        oss << "void evalVaryings()" << std::endl
            << "{" << std::endl;
        for ( std::set<std::string>::const_iterator it = varyings.begin(); it != varyings.end(); ++it )
        {
          oss << "  " << *it << " = " << getVaryingEval( *it ) << ";" << std::endl;
        }
        oss << "}" << std::endl << std::endl;
        snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
      }

      void buildEvalIORSnippet( std::vector<dp::fx::SnippetSharedPtr> & snippets, std::string const& ior )
      {
        std::ostringstream oss;
        oss << "float evalIOR( in vec3 normal )" << std::endl
            << "{" << std::endl
            << "  return( mdl_math_luminance( " << ior << " ) );" << std::endl
            << "}" << std::endl << std::endl;
        snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
      }

      void buildEvalSurfaceSnippet( std::vector<dp::fx::SnippetSharedPtr> & snippets, std::string const& postFix, SurfaceData const& surfaceData )
      {
        {
          std::ostringstream oss;
          oss << "vec4 evalColor" << postFix << "( in vec3 normal )" << std::endl
              << "{" << std::endl
              << "  return( " << surfaceData.scattering << " );" << std::endl
              << "}" << std::endl << std::endl;
          snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
        }
        {
          std::ostringstream oss;
          oss << "vec3 evalMaterialEmissive" << postFix << "( in vec3 normal )" << std::endl
              << "{" << std::endl
              << "  return( ( " << surfaceData.emission << " ).intensity );" << std::endl
              << "}" << std::endl << std::endl;
          snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
        }
        {
          std::string environ = surfaceData.scattering;
          static std::set<std::string> environmentFunctions = 
          {
            { "backscatteringGlossyReflectionBSDF"  },
            { "diffuseReflectionBSDF"               },
            { "diffuseTransmissionBSDF"             },
            { "measuredBSDF"                        },
            { "simpleGlossyBSDF"                    },
            { "specularBSDF"                        }
          };
          for ( std::set<std::string>::const_iterator it = environmentFunctions.begin(); it != environmentFunctions.end(); ++it )
          {
            boost::algorithm::replace_all( environ, *it, *it + "Environment" );
          }
          std::ostringstream oss;
          oss << "vec4 evalEnvironment" << postFix << "( in vec3 normal )" << std::endl
              << "{" << std::endl
              << "  return( " << ( ( environ == surfaceData.scattering ) ? "vec4(0,0,0,1)" : environ ) << " );" << std::endl
              << "}" << std::endl << std::endl;
          snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
        }
      }

      void buildEvalThinWalledSnippet( std::vector<dp::fx::SnippetSharedPtr> & snippets, std::string const& thinWalled )
      {
        std::ostringstream oss;
        oss << "bool evalThinWalled()" << std::endl
            << "{" << std::endl
            << "  return( " << thinWalled << " );" << std::endl
            << "}" << std::endl << std::endl;
        snippets.push_back( std::make_shared<dp::fx::StringSnippet>( oss.str() ) );
      }

      unsigned int getType( std::string const& type )
      {
        static const std::map<std::string, unsigned int> typeMapping =
        {
          { "bool",       dp::fx::PT_BOOL                                 },
          { "float",      dp::fx::PT_FLOAT32                              },
          { "int",        dp::fx::PT_INT32                                },
          { "sampler2D",  dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_2D  },
          { "sampler3D",  dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_3D  },
          { "vec2",       dp::fx::PT_FLOAT32 | dp::fx::PT_VECTOR2         },
          { "vec3",       dp::fx::PT_FLOAT32 | dp::fx::PT_VECTOR3         },
          { "vec4",       dp::fx::PT_FLOAT32 | dp::fx::PT_VECTOR4         }
        };
        std::map<std::string, unsigned int>::const_iterator it = typeMapping.find( type );
        DP_ASSERT( it != typeMapping.end() );
        return( it->second );
      }

      dp::fx::ParameterSpec createParameterSpec( ParameterData const& pd, std::map<std::string,dp::fx::EnumSpecSharedPtr> const& enumSpecs )
      {
        std::map<std::string,dp::fx::EnumSpecSharedPtr>::const_iterator it = enumSpecs.find( pd.type );
        if ( it != enumSpecs.end() )
        {
          return( dp::fx::ParameterSpec( pd.name, it->second, 0, pd.value, pd.annotations ) );
        }
        else
        {
          return( dp::fx::ParameterSpec( pd.name, getType( pd.type ), dp::util::stringToSemantic( pd.semantic), 0, pd.value, pd.annotations) );
        }
      }

      bool EffectLoader::loadEffects( std::string const& filename, dp::util::FileFinder const& fileFinder )
      {
        DP_ASSERT( dp::util::fileExists( filename ) );
        if ( m_loadedFiles.find( filename ) == m_loadedFiles.end() )
        {
          m_materialBuilder.parseFile( filename, fileFinder );

          std::map<std::string,MaterialData> const& materials = m_materialBuilder.getMaterials();
          for ( std::map<std::string, MaterialData>::const_iterator mit = materials.begin(); mit != materials.end(); ++mit )
          {
            dp::fx::mdl::EffectSpec::DomainSpecs domainSpecs;
            for ( std::map<dp::fx::Domain, StageData>::const_iterator sit = mit->second.stageData.begin(); sit != mit->second.stageData.end(); ++sit )
            {
              // gather the enums
              std::map<std::string,dp::fx::EnumSpecSharedPtr> enumSpecs;
              for ( std::set<std::string>::const_iterator eit = sit->second.enums.begin(); eit != sit->second.enums.end(); ++eit )
              {
                if ( !getEffectLibrary()->getEnumSpec( *eit ) )
                {
                  std::map<std::string,EnumData>::const_iterator esit = mit->second.enums.find( *eit );
                  DP_ASSERT( esit != mit->second.enums.end() );
                  std::vector<std::string> enums;
                  for ( std::vector<std::pair<std::string,int>>::const_iterator enumIt = esit->second.values.begin() ; enumIt != esit->second.values.end() ; enumIt++ )
                  {
                    enums.push_back( enumIt->first );
                  }
                  getEffectLibrary()->registerSpec( dp::fx::EnumSpec::create( esit->second.name, enums ) );
                }
                enumSpecs[*eit] = getEffectLibrary()->getEnumSpec( *eit );
              }

              // create ParameterGroupSpecs per domain
              std::vector<dp::fx::ParameterGroupSpecSharedPtr> parameterGroupSpecs;
              if ( !sit->second.parameters.empty() )
              {
                std::vector<dp::fx::ParameterSpec> params;
                for ( std::set<unsigned int>::const_iterator pit = sit->second.parameters.begin() ; pit != sit->second.parameters.end() ; ++pit )
                {
                  DP_ASSERT( *pit < mit->second.parameters.size() );
                  params.push_back( createParameterSpec( mit->second.parameters[*pit], enumSpecs ) );
                }

                dp::fx::ParameterGroupSpecSharedPtr pgs = dp::fx::ParameterGroupSpec::create( mit->first + "_" + dp::fx::getDomainName( sit->first ) + "_parameters", params );
                dp::fx::ParameterGroupSpecSharedPtr registeredPGS = getEffectLibrary()->registerSpec( pgs );
                if ( registeredPGS == pgs ) // a new spec had been added
                {
                  getEffectLibrary()->registerParameterGroupData( dp::fx::ParameterGroupDataPrivate::create( registeredPGS, registeredPGS->getName() ) );
                }
                parameterGroupSpecs.push_back( registeredPGS );
              }

              // snippets for "forward" technique
              std::vector<dp::fx::SnippetSharedPtr> snippets;
              if ( sit->first == dp::fx::Domain::VERTEX )
              {
                buildParameterSnippets( snippets );
              }
              buildEnumsSnippet( snippets, sit->second.enums, mit->second.enums );
              buildGlobalsSnippet( snippets, sit->first );
              buildVaryingsSnippet( snippets, mit->second.varyings, sit->first == dp::fx::Domain::VERTEX ? "out" : "in" );
              buildStructuresSnippet(snippets, sit->second.structures, mit->second.structures);
              buildTemporariesSnippet(snippets, sit->second.temporaries, mit->second.temporaries);
              if (sit->first == dp::fx::Domain::FRAGMENT)
              {
                snippets.push_back( std::make_shared<dp::fx::FileSnippet>( dp::home() + std::string( "/media/effects/xml/standard_lights/glsl/ambient_diffuse_specular.glsl" ) ) );
              }
              buildFunctionSnippets( snippets, sit->second.functions );
              buildEvalGeometrySnippet( snippets, sit->first, mit->second.geometryData );
              buildEvalTemporariesSnippet(snippets, sit->first, sit->second.temporaries, mit->second.temporaries);
              if (sit->first == dp::fx::Domain::VERTEX)
              {
                buildEvalVaryingsSnippet( snippets, mit->second.varyings );
                snippets.push_back( std::make_shared<dp::fx::FileSnippet>( dp::home() + "/media/dpfx/mdl/mdl_vs.glsl" ) );
              }
              else
              {
                DP_ASSERT( sit->first == dp::fx::Domain::FRAGMENT );
                buildEvalIORSnippet( snippets, mit->second.ior );
                buildEvalSurfaceSnippet( snippets, "Back", mit->second.backfaceData );
                buildEvalSurfaceSnippet( snippets, "Front", mit->second.surfaceData );
                buildEvalThinWalledSnippet( snippets, mit->second.thinWalled );
                snippets.push_back( std::make_shared<dp::fx::FileSnippet>( dp::home() + "/media/dpfx/mdl/mdl_fs.glsl" ) );
              }
              dp::fx::mdl::TechniqueSharedPtr forwardTechnique = dp::fx::mdl::Technique::create( "forward" );
              forwardTechnique->addSnippet( "v3f_n3f_t03f_ta3f_bi3f", std::make_shared<dp::fx::SnippetListSnippet>( snippets ) );

              // snippets for "depthPass" technique
              snippets.clear();
              if ( sit->first == dp::fx::Domain::VERTEX )
              {
                buildParameterSnippets( snippets );
                buildEnumsSnippet( snippets, sit->second.enums, mit->second.enums );
                buildGlobalsSnippet( snippets, sit->first );
                buildVaryingsSnippet( snippets, mit->second.varyings, "out" );
                buildStructuresSnippet( snippets, sit->second.structures, mit->second.structures );
                buildTemporariesSnippet(snippets, sit->second.temporaries, mit->second.temporaries);
                buildFunctionSnippets(snippets, sit->second.functions);
                buildEvalTemporariesSnippet(snippets, sit->first, sit->second.temporaries, mit->second.temporaries);
                buildEvalGeometrySnippet( snippets, sit->first, mit->second.geometryData );
                snippets.push_back( std::make_shared<dp::fx::FileSnippet>( dp::home() + "/media/dpfx/mdl/depthPass_vs.glsl" ) );
              }
              else
              {
                DP_ASSERT( sit->first == dp::fx::Domain::FRAGMENT );
                snippets.push_back( std::make_shared<dp::fx::FileSnippet>( dp::home() + "/media/dpfx/mdl/depthPass_fs.glsl" ) );
              }
              dp::fx::mdl::TechniqueSharedPtr depthPassTechnique = dp::fx::mdl::Technique::create( "depthPass" );
              depthPassTechnique->addSnippet( "v3f_n3f_t03f_ta3f_bi3f", std::make_shared<dp::fx::SnippetListSnippet>( snippets ) );

              std::map<std::string, dp::fx::mdl::TechniqueSharedPtr> techniques;
              techniques["forward"] = forwardTechnique;
              techniques["depthPass"] = depthPassTechnique;

              // put parameters and techniques together into a domainSpec
              dp::fx::mdl::DomainSpecSharedPtr domainSpec = dp::fx::mdl::DomainSpec::create( mit->first + "_" + dp::fx::getDomainName( sit->first ) + "_domain", sit->first, parameterGroupSpecs
                                                                                           , mit->second.transparent, techniques );
              domainSpecs.insert( std::make_pair( sit->first, domainSpec ) );
            }

            // finally, create the effect
            dp::fx::EffectSpecSharedPtr es = dp::fx::mdl::EffectSpec::create( mit->first, domainSpecs );
            dp::fx::EffectSpecSharedPtr registeredES = getEffectLibrary()->registerSpec( es, this );
            if ( registeredES == es )
            {
              dp::fx::EffectDataPrivateSharedPtr ed = dp::fx::EffectDataPrivate::create( registeredES, registeredES->getName() );
              for ( dp::fx::EffectSpec::iterator it = registeredES->beginParameterGroupSpecs(); it != registeredES->endParameterGroupSpecs(); ++it )
              {
                ed->setParameterGroupData( it, getEffectLibrary()->getParameterGroupData( ( *it )->getName() ) );
              }
              getEffectLibrary()->registerEffectData( ed );
            }

          }

          m_loadedFiles.insert( filename );
        }

        return true;
      }

      bool EffectLoader::getShaderSnippets( const dp::fx::ShaderPipelineConfiguration& configuration
                                          , dp::fx::Domain domain
                                          , std::string& entrypoint // TODO remove entrypoint
                                          , std::vector<dp::fx::SnippetSharedPtr>& snippets )
      {
        snippets.clear();

        dp::fx::mdl::EffectSpecSharedPtr const& effectSpec = std::static_pointer_cast<dp::fx::mdl::EffectSpec>(dp::fx::EffectLibrary::instance()->getEffectSpec(configuration.getName()));

        // All other domains have only one set of code snippets per technique and ignore the signature.
        dp::fx::Domain signatureDomain = Domain::FRAGMENT;
        std::string signature;

        switch ( domain )
        {
          case Domain::VERTEX:
          case Domain::GEOMETRY:
          case Domain::TESSELLATION_CONTROL:
          case Domain::TESSELLATION_EVALUATION:
            {
              // Geometry shaders need to be matched to the underlying domain. (See FIXME above.)
              DomainSpecSharedPtr const & signatureDomainSpec = effectSpec->getDomainSpec( signatureDomain );
              DP_ASSERT( signatureDomainSpec );
              dp::fx::mdl::TechniqueSharedPtr const & signatureTechnique = signatureDomainSpec->getTechnique( configuration.getTechnique() );
              if ( !signatureTechnique )
              {
                return false;  // Ignore domains which don't have a matching technique.
              }
              Technique::SignatureSnippets const & signatureSnippets = signatureTechnique->getSnippets();
              DP_ASSERT( signatureSnippets.size() == 1 );
              signature = signatureSnippets.begin()->first;
            }
            // Intentional fall through!
          case Domain::FRAGMENT:
            {
              DomainSpecSharedPtr const & domainSpec = effectSpec->getDomainSpec( domain );
              dp::fx::mdl::TechniqueSharedPtr const & technique = domainSpec->getTechnique( configuration.getTechnique() );
              if ( !technique )
              {
                return false;  // Ignore domains which don't have a matching technique.
              }
              Technique::SignatureSnippets const & signatureSnippets = technique->getSnippets();
              // signature.empty() means we didn't ask for a geometry shader, but one of the system or fragment level domains. Latter have a single snippets block per technique.
              Technique::SignatureSnippets::const_iterator it = (signature.empty()) ? signatureSnippets.begin() : signatureSnippets.find( signature );
              DP_ASSERT( it != signatureSnippets.end() );
              snippets.push_back( it->second );
            }
            break;

          default:
            DP_ASSERT(!" EffectLoader::getShaderSnippets(): Unexpected domain." );
            return false;
        }

        return true;
      }

      bool EffectLoader::effectHasTechnique( dp::fx::EffectSpecSharedPtr const& effectSpec, std::string const& techniqueName, bool /*rasterizer*/ )
      {
        bool hasTechnique = true;
        EffectSpecSharedPtr xmlEffectSpec = std::static_pointer_cast<dp::fx::mdl::EffectSpec>(effectSpec);
        for ( EffectSpec::DomainSpecs::const_iterator it = xmlEffectSpec->getDomainSpecs().begin() ; it != xmlEffectSpec->getDomainSpecs().end() && hasTechnique ; ++it )
        {
          switch( it->second->getDomain() )
          {
            case Domain::VERTEX :
            case Domain::TESSELLATION_CONTROL :
            case Domain::TESSELLATION_EVALUATION :
            case Domain::GEOMETRY :
            case Domain::FRAGMENT :
              hasTechnique = !!it->second->getTechnique( techniqueName );
              break;
            default :
              break;
          }
        }
        return( hasTechnique );
      }

      dp::fx::ShaderPipelineSharedPtr EffectLoader::generateShaderPipeline( const dp::fx::ShaderPipelineConfiguration& configuration )
      {
        ShaderPipelineImplSharedPtr shaderPipeline = ShaderPipelineImpl::create();

        EffectSpecSharedPtr effectSpec = std::static_pointer_cast<dp::fx::mdl::EffectSpec>(dp::fx::EffectLibrary::instance()->getEffectSpec(configuration.getName()));
        EffectSpec::DomainSpecs const & domainSpecs = effectSpec->getDomainSpecs();

        for ( EffectSpec::DomainSpecs::const_iterator it = domainSpecs.begin(); it != domainSpecs.end(); ++it ) 
        {
          DP_ASSERT( it->first != Domain::PIPELINE );

          ShaderPipeline::Stage stage;
          stage.domain = it->first;
          stage.parameterGroupSpecs = it->second->getParameterGroups();

          // generate snippets
          std::vector<dp::fx::SnippetSharedPtr> snippets;
          if ( getShaderSnippets( configuration, stage.domain, stage.entrypoint, snippets ) )
          {
            stage.source = std::make_shared<SnippetListSnippet>( snippets );

            if ( stage.domain == Domain::VERTEX
              || stage.domain == Domain::GEOMETRY
              || stage.domain == Domain::TESSELLATION_CONTROL
              || stage.domain == Domain::TESSELLATION_EVALUATION)
            {
              stage.systemSpecs.push_back( "sys_matrices" );
              stage.systemSpecs.push_back( "sys_camera" );
            }
            else if ( stage.domain == Domain::FRAGMENT )
            {
              stage.systemSpecs.push_back( "sys_Fragment" );
            }

            shaderPipeline->addStage( stage );
          }
        }

        return shaderPipeline;
      }

      bool EffectLoader::save( const EffectDataSharedPtr& effectData, const std::string& filename )
      {
        DP_ASSERT( !"not yet implemented" );
        return( false );
      }

    } // mdl
  } // fx
} // dp

namespace
{
  bool registerEffectLibrary()
  {
    dp::fx::EffectLibraryImpl *eli = dynamic_cast<dp::fx::EffectLibraryImpl*>( dp::fx::EffectLibrary::instance() );
    DP_ASSERT( eli );
    if ( eli )
    {
      eli->registerEffectLoader( dp::fx::mdl::EffectLoader::create( eli ), ".mdl" );
      return true;
    }
    return false;
  }
} // namespace anonymous

extern "C"
{
  DP_FX_API bool dp_fx_mdl_initialized = registerEffectLibrary();
}
