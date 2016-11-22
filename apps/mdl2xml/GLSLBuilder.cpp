// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/split.hpp>
#include <dp/DP.h>
#include <dp/fx/ParameterConversion.h>
#include <dp/util/File.h>
#include "GLSLBuilder.h"

void GLSLBuilder::buildAttributes( TiXmlElement * glslElement )
{
  buildSourceElement( glslElement, "vec4", "attrPosition", "0" );
  buildSourceElement( glslElement, "vec3", "attrNormal", "2" );
  buildSourceElement( glslElement, "vec3", "attrTexCoord0", "8" );
  buildSourceElement( glslElement, "vec3", "attrTangent", "14" );
  buildSourceElement( glslElement, "vec3", "attrBinormal", "15" );
}

void GLSLBuilder::buildEffect( TiXmlElement * parent, dp::fx::Domain domain, std::map<std::string,dp::fx::mdl::MaterialData>::const_iterator material )
{
  TiXmlElement * vertexEffectElement = new TiXmlElement( "effect" );
  switch( domain )
  {
    case dp::fx::Domain::VERTEX :
      vertexEffectElement->SetAttribute( "id", ( material->first + std::string( "VS" ) ).c_str() );
      vertexEffectElement->SetAttribute( "domain", "vertex" );
      break;
    case dp::fx::Domain::FRAGMENT :
      vertexEffectElement->SetAttribute( "id", ( material->first + std::string( "FS" ) ).c_str() );
      vertexEffectElement->SetAttribute( "domain", "fragment" );
      if ( material->second.transparent )
      {
        vertexEffectElement->SetAttribute( "transparent", "true" );
      }
      break;
    default :
      DP_ASSERT( false );
      return;
  }

  std::map<dp::fx::Domain,dp::fx::mdl::StageData>::const_iterator stage = material->second.stageData.find( domain );
  DP_ASSERT( stage != material->second.stageData.end() );
  buildParameterGroup( vertexEffectElement, stage->second.parameters, material->second.parameters, material->first );
  buildTechniqueForward( vertexEffectElement, stage, material->second );
  buildTechniqueDepthPass( vertexEffectElement, stage, material->second );

  parent->LinkEndChild( vertexEffectElement );
}

void GLSLBuilder::buildEnums( TiXmlElement * parent, std::map<std::string,dp::fx::mdl::MaterialData> const& materials )
{
  std::set<std::string> enums;
  for ( std::map<std::string,dp::fx::mdl::MaterialData>::const_iterator mit = materials.begin() ; mit!=materials.end() ; ++mit )
  {
    for ( std::map<dp::fx::Domain,dp::fx::mdl::StageData>::const_iterator sit = mit->second.stageData.begin() ; sit != mit->second.stageData.end() ; ++sit )
    {
      for ( std::set<std::string>::const_iterator eit = sit->second.enums.begin() ; eit != sit->second.enums.end() ; ++eit )
      {
        if ( enums.find( *eit ) == enums.end() )
        {
          TiXmlElement * enumElement = new TiXmlElement( "enum" );
          enumElement->SetAttribute( "type", eit->c_str() );

          std::map<std::string,dp::fx::mdl::EnumData>::const_iterator dit = mit->second.enums.find( *eit );
          DP_ASSERT( dit != mit->second.enums.end() );
          std::ostringstream oss;
          oss << dit->second.values[0].first;
          for ( unsigned int i=1 ; i<dit->second.values.size() ; i++ )
          {
            oss << " " << dit->second.values[i].first;
          }
          enumElement->SetAttribute( "values", oss.str().c_str() );
          parent->LinkEndChild( enumElement );
        }
        enums.insert( *eit );
      }
    }
  }
}

void GLSLBuilder::buildParameter( TiXmlElement * parent, dp::fx::mdl::ParameterData const& pd )
{
  std::vector<std::string> parameters;
  boost::algorithm::split( parameters, pd.value, boost::algorithm::is_any_of( "( ,)" ), boost::algorithm::token_compress_on );
  DP_ASSERT( ( parameters.size() == 1 ) || ( 3 < parameters.size() ) );

  std::string parameterValue;
  if ( parameters.size() == 1 )
  {
    DP_ASSERT( parameters[0] == pd.value );
    parameterValue = pd.value;
  }
  else
  {
    // for multi-element parameters skip the first (like "vec3") and the last (being "")
    DP_ASSERT( parameters.back().empty() );
    parameterValue = parameters[1];
    for ( size_t i=2 ; i<parameters.size() - 1 ; i++ )
    {
      parameterValue += " " + parameters[i];
    }
  }

  TiXmlElement * parameterElement = new TiXmlElement( "parameter" );
  parameterElement->SetAttribute( "type", pd.type.c_str() );
  parameterElement->SetAttribute( "name", pd.name.c_str() );
  parameterElement->SetAttribute( "semantic", pd.semantic.c_str() );
  parameterElement->SetAttribute( "value", parameterValue.c_str() );
  if ( ! pd.annotations.empty() )
  {
    parameterElement->SetAttribute( "annotation", pd.annotations.c_str() );
  }
  parent->LinkEndChild( parameterElement );
}

void GLSLBuilder::buildParameterGroup( TiXmlElement * parent, std::set<unsigned int> const& stageParameters, std::vector<dp::fx::mdl::ParameterData> const& materialParameters, std::string const& materialName )
{
  if ( !stageParameters.empty() )
  {
    TiXmlElement * parameterGroupElement = new TiXmlElement( "parameterGroup" );
    std::string id = materialName + ( ( strcmp( parent->Attribute( "domain" ), "fragment" ) == 0 ) ? "FragmentParameters" : "VertexParameters" );
    parameterGroupElement->SetAttribute( "id", id.c_str() );
    for ( std::set<unsigned int>::const_iterator it = stageParameters.begin() ; it != stageParameters.end() ; ++it )
    {
      DP_ASSERT( *it < materialParameters.size() );
      buildParameter( parameterGroupElement, materialParameters[*it] );
    }
    parent->LinkEndChild( parameterGroupElement );
  }
}

TiXmlElement * GLSLBuilder::buildPipelines( std::map<std::string,dp::fx::mdl::MaterialData> const& materials )
{
  TiXmlElement * libraryElement = nullptr;
  if ( !materials.empty() )
  {
    libraryElement = new TiXmlElement( "library" );
    buildEnums( libraryElement, materials );
    for ( std::map<std::string,dp::fx::mdl::MaterialData>::const_iterator it = materials.begin() ; it!=materials.end() ; ++it )
    {
      m_structures = it->second.structures;
      buildEffect( libraryElement, dp::fx::Domain::VERTEX, it );
      buildEffect( libraryElement, dp::fx::Domain::FRAGMENT, it );
      buildPipelineSpec( libraryElement, it->first );
    }
  }
  return( libraryElement );
}

void GLSLBuilder::buildPipelineSpec( TiXmlElement * parent, std::string const& baseName )
{
  TiXmlElement * pipelineSpecElement = new TiXmlElement( "PipelineSpec" );
  pipelineSpecElement->SetAttribute( "id", baseName.c_str() );
  pipelineSpecElement->SetAttribute( "vertex", ( baseName + std::string( "VS" ) ).c_str() );
  pipelineSpecElement->SetAttribute( "fragment", ( baseName + std::string( "FS" ) ).c_str() );
  parent->LinkEndChild( pipelineSpecElement );
}

void GLSLBuilder::buildSourceElement( TiXmlElement * parent, std::string const& source )
{
  TiXmlElement * sourceElement = new TiXmlElement( "source" );
  sourceElement->SetAttribute( "string", source.c_str() );
  parent->LinkEndChild( sourceElement );
}

void GLSLBuilder::buildSourceElement( TiXmlElement * parent, std::string const& input, std::string const& name, std::string const& location )
{
  TiXmlElement * sourceElement = new TiXmlElement( "source" );
  sourceElement->SetAttribute( "input", input.c_str() );
  sourceElement->SetAttribute( "name", name.c_str() );
  sourceElement->SetAttribute( "location", location.c_str() );
  parent->LinkEndChild( sourceElement );
}

void GLSLBuilder::buildSourceElementEnums( TiXmlElement * parent, std::set<std::string> const& usedEnums, std::map<std::string,dp::fx::mdl::EnumData> const& enumData )
{
  if ( !usedEnums.empty() )
  {
    std::ostringstream oss;
    oss << "// Enumerations section" << std::endl;
    for ( std::set<std::string>::const_iterator it = usedEnums.begin() ; it != usedEnums.end() ; ++it )
    {
      std::map<std::string,dp::fx::mdl::EnumData>::const_iterator eit = enumData.find( *it );
      DP_ASSERT( eit != enumData.end() );
      oss << "// enum " << *it << ":" << std::endl;
      oss << "#define " << *it << "\tint" << std::endl;
      for ( size_t i=0 ; i<eit->second.values.size() ; i++ )
      {
        oss << "#define " << eit->second.values[i].first << "\t" << eit->second.values[i].second << std::endl;
      }
      oss << std::endl;
    }
    buildSourceElement( parent, oss.str() );
  }
}

void GLSLBuilder::buildSourceElementEvalIOR( TiXmlElement * parent, std::string const& ior )
{
  std::ostringstream oss;
  oss << "float evalIOR( in vec3 normal )" << std::endl
      << "{" << std::endl
      << "  return( mdl_math_luminance( " << ior << " ) );" << std::endl
      << "}" << std::endl << std::endl;
  buildSourceElement( parent, oss.str() );
}

void GLSLBuilder::buildSourceElementEvalSurface( TiXmlElement * parent, dp::fx::mdl::SurfaceData const& surfaceData, std::string const& postFix )
{
  {
    std::ostringstream oss;
    oss << "vec4 evalColor" << postFix << "( in vec3 normal )" << std::endl
        << "{" << std::endl
        << "  return( " << surfaceData.scattering << " );" << std::endl
        << "}" << std::endl << std::endl;
    buildSourceElement( parent, oss.str() );
  }
  {
    std::ostringstream oss;
    oss << "vec3 evalMaterialEmissive" << postFix << "( in vec3 normal )" << std::endl
        << "{" << std::endl
        << "  return( ( " << surfaceData.emission << " ).intensity );" << std::endl
        << "}" << std::endl << std::endl;
    buildSourceElement( parent, oss.str() );
  }
  {
    std::string environString = surfaceData.scattering;
    static std::set<std::string> environmentFunctions = 
    {
      { "mdl_df_backscatteringGlossyReflectionBSDF" },
      { "mdl_df_diffuseReflectionBSDF"              },
      { "mdl_df_diffuseTransmissionBSDF"            },
      { "mdl_df_measuredBSDF"                       },
      { "mdl_df_simpleGlossyBSDF"                   },
      { "mdl_df_specularBSDF"                       }
    };
    for ( std::set<std::string>::const_iterator it = environmentFunctions.begin() ; it != environmentFunctions.end() ; ++it )
    {
      boost::algorithm::replace_all( environString, *it, *it + "Environment" );
    }
    std::ostringstream oss;
    oss << "vec4 evalEnvironment" << postFix << "( in vec3 normal )" << std::endl
        << "{" << std::endl
        << "  return( " << ( ( environString == surfaceData.scattering ) ? "vec4(0,0,0,1)" : environString ) << " );" << std::endl
        << "}" << std::endl << std::endl;
    buildSourceElement( parent, oss.str() );
  }
}

void GLSLBuilder::buildSourceElementEvalTemporaries(TiXmlElement * parent, std::set<unsigned int> const& stageTemporaries, std::map<unsigned int, dp::fx::mdl::TemporaryData> const& temporaries, dp::fx::Domain domain)
{
  if (domain == dp::fx::Domain::VERTEX)
  {
    std::ostringstream oss;
    oss << "void evalTemporaries( in vec3 normal )" << std::endl
      << "{" << std::endl;
    for (std::set<unsigned int>::const_iterator it = stageTemporaries.begin(); it != stageTemporaries.end(); ++it)
    {
      std::map<unsigned int, dp::fx::mdl::TemporaryData>::const_iterator tit = temporaries.find(*it);
      DP_ASSERT(tit != temporaries.end());
      if (!tit->second.eval.empty())
      {
        oss << "  temporary" << tit->first << " = " << tit->second.eval << ";" << std::endl;
      }
    }
    oss << "}" << std::endl << std::endl;
    buildSourceElement(parent, oss.str());
  }
  else
  {
    DP_ASSERT(domain == dp::fx::Domain::FRAGMENT);

    // filter temporaries and temporaries per light source
    std::vector<unsigned int> t, tpls;
    for (std::set<unsigned int>::const_iterator it = stageTemporaries.begin(); it != stageTemporaries.end(); ++it)
    {
      std::map<unsigned int, dp::fx::mdl::TemporaryData>::const_iterator tit = temporaries.find(*it);
      DP_ASSERT(tit != temporaries.end());
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
        std::map<unsigned int, dp::fx::mdl::TemporaryData>::const_iterator it = temporaries.find(t[i]);
        DP_ASSERT(it != temporaries.end());
        oss << "  temporary" << it->first << " = " << it->second.eval << ";" << std::endl;
      }
      // add Environment temporaries "per light source" here in the temporaries section !
      for (size_t i = 0; i<tpls.size(); ++i)
      {
        std::map<unsigned int, dp::fx::mdl::TemporaryData>::const_iterator it = temporaries.find(tpls[i]);
        DP_ASSERT(it != temporaries.end());
      }
      oss << "}" << std::endl << std::endl;
      buildSourceElement(parent, oss.str());
    }

    {
      std::ostringstream oss;
      oss << "void evalTemporariesPerLightSource()" << std::endl
        << "{" << std::endl;
      for (size_t i = 0; i<tpls.size(); ++i)
      {
        std::map<unsigned int, dp::fx::mdl::TemporaryData>::const_iterator it = temporaries.find(tpls[i]);
        DP_ASSERT(it != temporaries.end());
        oss << "  temporary" << it->first << " = " << it->second.eval << ";" << std::endl;
      }
      oss << "}" << std::endl << std::endl;
      buildSourceElement(parent, oss.str());
    }
  }
}

void GLSLBuilder::buildSourceElementEvalThinWalled(TiXmlElement * parent, std::string const& thinWalled)
{
  std::ostringstream oss;
  oss << "bool evalThinWalled()" << std::endl
      << "{" << std::endl
      << "  return( " << thinWalled << " );" << std::endl
      << "}" << std::endl << std::endl;
  buildSourceElement( parent, oss.str() );
}

void GLSLBuilder::buildSourceElementEvalVaryings( TiXmlElement * parent, std::set<std::string> const& varyings )
{
  std::ostringstream oss;
  oss << "void evalVaryings()" << std::endl
      << "{" << std::endl;
  for ( std::set<std::string>::const_iterator it = varyings.begin() ; it != varyings.end() ; ++it )
  {
    oss << "  " << *it << " = " << getVaryingData(*it).eval << ";" << std::endl;
  }
  oss << "}" << std::endl << std::endl;
  buildSourceElement( parent, oss.str() );
}

void GLSLBuilder::buildSourceElementFunctions( TiXmlElement * parent, std::vector<std::string> const& functions )
{
  for ( std::vector<std::string>::const_iterator it = functions.begin() ; it != functions.end() ; ++it )
  {
    std::ostringstream oss;
    oss << dp::util::loadStringFromFile( dp::home() + std::string( "/media/dpfx/mdl/" ) + *it + std::string( ".glsl" ) ) << std::endl;
    buildSourceElement( parent, oss.str() );
  }
}

void GLSLBuilder::buildSourceElementEvalGeometry( TiXmlElement * parent, dp::fx::mdl::GeometryData const& geometryData, dp::fx::Domain domain )
{
  if ( domain == dp::fx::Domain::VERTEX )
  {
    std::ostringstream oss;
    oss << "vec4 evalWorldPos()" << std::endl
        << "{" << std::endl
        << "  return( sys_WorldMatrix * ( attrPosition + vec4( " << geometryData.displacement << ", 0.0f ) ) );" << std::endl
        << "}" << std::endl << std::endl;
    buildSourceElement( parent, oss.str() );
  }
  else
  {
    {
      std::ostringstream oss;
      oss << "float evalCutoutOpacity( in vec3 normal )" << std::endl
          << "{" << std::endl
          << "  return( clamp( " << geometryData.cutoutOpacity << ", 0.0f, 1.0f ) );" << std::endl
          << "}" << std::endl << std::endl;
      buildSourceElement( parent, oss.str() );
    }
    {
      std::ostringstream oss;
      oss << "vec3 evalNormal( in vec3 normal )" << std::endl
          << "{" << std::endl
          << "  return( " << geometryData.normal << " );" << std::endl
          << "}" << std::endl << std::endl;
      buildSourceElement( parent, oss.str() );
    }
  }
}

void GLSLBuilder::buildSourceElementGlobals( TiXmlElement * parent, dp::fx::Domain domain )
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
    oss << "vec3 lightDir;" << std::endl;
    oss << "vec3 lightDiffuse;" << std::endl;
    oss << "vec3 lightSpecular;" << std::endl;
    oss << "float materialIOR;" << std::endl;
    oss << "vec3 viewDir;" << std::endl;
  }
  oss << std::endl;
  buildSourceElement( parent, oss.str() );
}

void GLSLBuilder::buildSourceElementStructures( TiXmlElement * parent, std::set<std::string> const& structures )
{
  if ( !structures.empty() )
  {
    std::ostringstream oss;
    oss << "// Structures section" << std::endl;
    for ( std::set<std::string>::const_iterator it = structures.begin() ; it != structures.end() ; ++it )
    {
      std::map<std::string,dp::fx::mdl::StructureData>::const_iterator specIt = m_structures.find( *it );
      DP_ASSERT( specIt != m_structures.end() );
      oss << "struct " << specIt->second.name << std::endl;
      oss << "{" << std::endl;
      for ( size_t i=0 ; i<specIt->second.members.size() ; i++ )
      {
        oss << "  " << specIt->second.members[i].first << "\t" << specIt->second.members[i].second << ";" << std::endl;
      }
      oss << "};" << std::endl;
      oss << std::endl;
    }
    oss << std::endl;
    buildSourceElement( parent, oss.str() );
  }
}

void GLSLBuilder::buildSourceElementTemporaries(TiXmlElement * parent, std::set<unsigned int> const& temporariesSet, std::map<unsigned int, dp::fx::mdl::TemporaryData> const& temporariesMap)
{
  if (!temporariesSet.empty())
  {
    std::ostringstream oss;
    oss << "// Temporaries Declarations" << std::endl;
    for (std::set<unsigned int>::const_iterator it = temporariesSet.begin(); it != temporariesSet.end(); ++it)
    {
      std::map<unsigned int, dp::fx::mdl::TemporaryData>::const_iterator tit = temporariesMap.find(*it);
      DP_ASSERT(tit != temporariesMap.end());

      oss << tit->second.type << " temporary" << tit->first << ";" << std::endl;
    }
    oss << std::endl;
    buildSourceElement(parent, oss.str());
  }
}

void GLSLBuilder::buildSourceElementVaryings( TiXmlElement * parent, std::set<std::string> const& varyings, std::string const& prefix )
{
  if ( ! varyings.empty() )
  {
    std::ostringstream oss;
    oss << "// Varyings" << std::endl;
    for ( std::set<std::string>::const_iterator it = varyings.begin() ; it != varyings.end() ; ++it )
    {
      oss << prefix << " " << getVaryingData(*it).type << " " << *it << ";" << std::endl;
    }
    oss << std::endl;
    buildSourceElement( parent, oss.str() );
  }
}

void GLSLBuilder::buildTechniqueDepthPass( TiXmlElement * parent, std::map<dp::fx::Domain,dp::fx::mdl::StageData>::const_iterator stage, dp::fx::mdl::MaterialData const& material )
{
  TiXmlElement * techniqueElement = new TiXmlElement( "technique" );
  techniqueElement->SetAttribute( "type", "depthPass" );
  {
    TiXmlElement * glslElement = new TiXmlElement( "glsl" );
    glslElement->SetAttribute( "signature", "v3f_n3f_t03f_ta3f_bi3f" );
    {
      if ( stage->first == dp::fx::Domain::VERTEX )
      {
        buildAttributes( glslElement );
        buildSourceElementEnums( glslElement, stage->second.enums, material.enums );
        buildSourceElementGlobals( glslElement, stage->first );
        buildSourceElementVaryings( glslElement, material.varyings, "out" );
        buildSourceElementStructures( glslElement, stage->second.structures );
        buildSourceElementTemporaries(glslElement, stage->second.temporaries, material.temporaries);
        buildSourceElementFunctions(glslElement, stage->second.functions);
        buildSourceElementEvalTemporaries(glslElement, stage->second.temporaries, material.temporaries, stage->first);
        buildSourceElementEvalGeometry(glslElement, material.geometryData, stage->first);
      }
      buildSourceElement( glslElement, dp::util::loadStringFromFile( dp::home() + ( stage->first == dp::fx::Domain::VERTEX ? "/media/dpfx/mdl/depthPass_vs.glsl" : "/media/dpfx/mdl/depthPass_fs.glsl" ) ) );
    }
    techniqueElement->LinkEndChild( glslElement );
  }
  parent->LinkEndChild( techniqueElement );
}

void GLSLBuilder::buildTechniqueForward( TiXmlElement * parent, std::map<dp::fx::Domain,dp::fx::mdl::StageData>::const_iterator stage, dp::fx::mdl::MaterialData const& material )
{
  TiXmlElement * techniqueElement = new TiXmlElement( "technique" );
  techniqueElement->SetAttribute( "type", "forward" );
  {
    TiXmlElement * glslElement = new TiXmlElement( "glsl" );
    glslElement->SetAttribute( "signature", "v3f_n3f_t03f_ta3f_bi3f" );
    {
      if ( stage->first == dp::fx::Domain::VERTEX )
      {
        buildAttributes( glslElement );
      }
      buildSourceElementEnums( glslElement, stage->second.enums, material.enums );
      buildSourceElementGlobals( glslElement, stage->first );
      buildSourceElementVaryings( glslElement, material.varyings, stage->first == dp::fx::Domain::VERTEX ? "out" : "in" );
      buildSourceElementStructures( glslElement, stage->second.structures );
      buildSourceElementTemporaries(glslElement, stage->second.temporaries, material.temporaries);
      if (stage->first == dp::fx::Domain::FRAGMENT)
      {
        buildSourceElement( glslElement, dp::util::loadStringFromFile( dp::home() + std::string( "/media/effects/xml/standard_lights/glsl/ambient_diffuse_specular.glsl" ) ) + std::string( "\n" ) );
      }
      buildSourceElementFunctions( glslElement, stage->second.functions );
      buildSourceElementEvalGeometry( glslElement, material.geometryData, stage->first );
      buildSourceElementEvalTemporaries(glslElement, stage->second.temporaries, material.temporaries, stage->first);
      if (stage->first == dp::fx::Domain::VERTEX)
      {
        buildSourceElementEvalVaryings( glslElement, material.varyings );
        buildSourceElement( glslElement, dp::util::loadStringFromFile( dp::home() + "/media/dpfx/mdl/mdl_vs.glsl" ) );
      }
      else
      {
        buildSourceElementEvalIOR( glslElement, material.ior );
        buildSourceElementEvalSurface( glslElement, material.backfaceData, "Back" );
        buildSourceElementEvalSurface( glslElement, material.surfaceData, "Front" );
        buildSourceElementEvalThinWalled( glslElement, material.thinWalled );
        buildSourceElement( glslElement, dp::util::loadStringFromFile( dp::home() + "/media/dpfx/mdl/mdl_fs.glsl" ) );
      }
    }
    techniqueElement->LinkEndChild( glslElement );
  }
  parent->LinkEndChild( techniqueElement );
}

GLSLBuilder::VaryingData const& GLSLBuilder::getVaryingData( std::string const& varyingName )
{
  static const std::map<std::string,VaryingData> varyings =
  {
    { "varBinormal",    VaryingData( "vec3", "normalize( ( sys_WorldMatrix * vec4( attrBinormal, 0.0f ) ).xyz )" )            },
    { "varEyePos",      VaryingData( "vec3", "vec3( sys_ViewMatrixI[3][0], sys_ViewMatrixI[3][1], sys_ViewMatrixI[3][2] )" )  },
    { "varNormal",      VaryingData( "vec3", "normalize( ( sys_WorldMatrixIT * vec4( attrNormal, 0.0f ) ).xyz )" )            },
    { "varObjBinormal", VaryingData( "vec3", "normalize( attrBinormal )" )                                                    },
    { "varObjPos",      VaryingData( "vec3", "attrPosition.xyz" )                                                             },
    { "varObjTangent",  VaryingData( "vec3", "normalize( attrTangent )" )                                                     },
    { "varTangent",     VaryingData( "vec3", "normalize( ( sys_WorldMatrix * vec4( attrTangent, 0.0f ) ).xyz )" )             },
    { "varTexCoord0",   VaryingData( "vec3", "attrTexCoord0" )                                                                },
    { "varWorldPos",    VaryingData( "vec3", "worldPos.xyz" )                                                                 }
  };
  std::map<std::string,VaryingData>::const_iterator it = varyings.find( varyingName );
  DP_ASSERT( it != varyings.end() );
  return( it->second );
}
