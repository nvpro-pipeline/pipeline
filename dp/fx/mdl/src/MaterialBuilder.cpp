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

#include <dp/Types.h>
#include <dp/fx/mdl/inc/MaterialBuilder.h>
#include <dp/util/File.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <tinyxml.h>

namespace dp
{
  namespace fx
  {
    namespace mdl
    {
      static std::string convertColons( std::string const& name )
      {
        DP_ASSERT( !boost::contains( name, "_" ) );
        std::string convertedName( name );
        boost::replace_all( convertedName, "::", "_" );
        return( convertedName );
      }

      static std::string convertName( std::string const& name )
      {
        // convert camel_case to camelCase
        std::string convertedName( name );
        for ( size_t pos=convertedName.find( '_', 0 ) ; pos != std::string::npos ; pos=convertedName.find( '_', pos ) )
        {
          convertedName.erase( pos, 1 );
          DP_ASSERT( pos < convertedName.length() );
          convertedName[pos] = toupper( convertedName[pos] );
        }

        // convert trailing "Bsdf", "Edf", "Ior", and "Vdf" into "BSDF", "EDF", "IOR", and "VDF", respectively
        if ( ( 3 < convertedName.length() ) && ( convertedName.substr( convertedName.length() - 4 ) == "Bsdf" ) )
        {
          convertedName.replace( convertedName.length() - 4, 4, "BSDF" );
        }
        else if ( ( 2 < convertedName.length() ) && ( convertedName.substr( convertedName.length() - 3 ) == "Edf" ) )
        {
          convertedName.replace( convertedName.length() - 3, 3, "EDF" );
        }
        else if ( ( 2 < convertedName.length() ) && ( convertedName.substr( convertedName.length() - 3 ) == "Ior" ) )
        {
          convertedName.replace( convertedName.length() - 3, 3, "IOR" );
        }
        else if ( ( 2 < convertedName.length() ) && ( convertedName.substr( convertedName.length() - 3 ) == "Vdf" ) )
        {
          convertedName.replace( convertedName.length() - 3, 3, "VDF" );
        }
        return( convertedName );
      }

      static std::string convertUnderscores( std::string const& name )
      {
        // convert mdl_base_name to mdl::base::name
        DP_ASSERT( !boost::contains( name, "::" ) );
        std::string convertedName( name );
        boost::replace_all( convertedName, "_", "::" );
        return( convertedName );
      }

      void parseConfigFunction( TiXmlElement * functionElement, std::map<std::string, FunctionData> & functions )
      {
        DP_ASSERT( functionElement->Attribute( "name" ) );
        std::string functionName = functionElement->Attribute( "name" );
        DP_ASSERT( functions.find( functionName ) == functions.end() );
        FunctionData & functionData = functions[functionName];

        // parse function dependencies
        for ( TiXmlElement * element = functionElement->FirstChildElement() ; element ; element = element->NextSiblingElement() )
        {
          DP_ASSERT( element->Value() );
          std::string value = element->Value();
          DP_ASSERT( value == "dependency" );

          DP_ASSERT( element->Attribute( "type" ) && element->Attribute( "name" ) );
          std::string type = element->Attribute( "type" );
          std::string name = element->Attribute( "name" );
          if ( type == "function" )
          {
            DP_ASSERT( functionData.functionDependencies.find( name ) == functionData.functionDependencies.end() );
            functionData.functionDependencies.insert( name );
          }
          else
          {
            DP_ASSERT( type == "varying" );
            DP_ASSERT( functionData.varyingDependencies.find( name ) == functionData.varyingDependencies.end() );
            functionData.varyingDependencies.insert( name );
          }
        }
      }

      static std::string stripArguments( std::string const& name )
      {
        return( name.substr( 0, name.find( '(' ) ) );
      }

      static std::string stripArray( std::string const& name )
      {
        return( name.substr( 0, name.find( '[' ) ) );
      }

      std::string stripParameterValue( std::string const& value )
      {
        size_t pos = value.find( '(' );
        if ( pos != std::string::npos )
        {
          std::string result( value.substr( pos + 1 ) );
          boost::replace_all( result, ",", " " );
          boost::erase_all( result, ")" );
          return( result );
        }
        return( value );
      }

      static std::string translateFunctionName( std::string const& name )
      {
        static std::map<std::string,std::string> nameMap =
        {
          { "mdl_color",    "vec3"  },
          { "mdl_math_abs", "abs"   },
          { "mdl_math_cos", "cos"   },
          { "mdl_math_max", "max"   },
          { "mdl_math_sin", "sin"   },
          { "mdl_float3",   "vec3"  },
          { "mdl_float4x4", "mat4"  }
        };

        std::map<std::string,std::string>::const_iterator it = nameMap.find( name );
        return( it != nameMap.end() ? it->second : name );
      }

      std::string unconvertName( std::string const& name )
      {
        // convert camelCase to camel_case
        std::string convertedName( name );
        for ( size_t pos=0 ; pos < convertedName.size() ; pos++ )
        {
          if ( isupper( convertedName[pos] ) )
          {
            convertedName.insert( pos, "_" );
            convertedName[pos+1] = tolower( convertedName[pos+1] );
          }
        }
        DP_ASSERT( !boost::ends_with( name, "BSDF" ) && !boost::ends_with( name, "EDF" ) && !boost::ends_with( name, "IOR" ) && !boost::ends_with( name, "VDF" ) );
        return( convertedName );
      }

      MaterialBuilder::MaterialBuilder( std::string const& configFile )
        : m_currentStage( nullptr )
        , m_currentTemporaryIdx( ~0 )
        , m_insideAnnotation( false )
        , m_insideParameter( false )
      {
        boost::shared_ptr<TiXmlDocument> cfg( new TiXmlDocument( configFile.c_str() ) );
        DP_ASSERT( cfg );
        if ( ! cfg->LoadFile() )
        {
          DP_ASSERT( cfg->Error() );
          std::cerr << "MaterialBuilder: failed to load file <" << configFile << ">:" << std::endl
                    << "\t" << cfg->ErrorDesc() << std::endl
                    << "\t in line " << cfg->ErrorRow() << ", column " << cfg->ErrorCol() << std::endl << std::endl;
          throw std::runtime_error( "MaterialBuilder: failed to load file" );
        }

        TiXmlHandle libraryHandle = cfg->FirstChildElement( "library" );   // The required XML root node.
        TiXmlElement * root = libraryHandle.Element();
        if ( root )
        {
          for ( TiXmlElement * element = root->FirstChildElement() ; element ; element = element->NextSiblingElement() )
          {
            DP_ASSERT( element->Value() );
            if ( strcmp( element->Value(), "function" ) == 0 )
            {
              parseConfigFunction( element, m_functions );
            }
            else
            {
              std::cerr << "Unknown element <" << element->Value() << "> in config file." << std::endl;
              DP_ASSERT( false );
            }
          }
        }
      }

      std::map<std::string,MaterialData> const& MaterialBuilder::getMaterials() const
      {
        return( m_materials );
      }

      bool MaterialBuilder::annotationBegin( std::string const& name )
      {
        DP_ASSERT( !m_currentCall.empty() );
        DP_ASSERT( !m_insideAnnotation );
        m_insideAnnotation = true;
        m_annotations.push_back( Argument( convertColons( convertName( stripArguments( name ) ) ) ) );
        m_currentCall.push( &m_annotations.back() );
        return true;
      }

      void MaterialBuilder::annotationEnd()
      {
        DP_ASSERT( m_insideAnnotation );
        DP_ASSERT( 1 < m_currentCall.size() );
        m_insideAnnotation = false;
        m_currentCall.pop();
      }

      bool MaterialBuilder::argumentBegin( unsigned int idx, std::string const& type, std::string const& name )
      {
        if ( m_currentCall.top()->name == "mdl_materialGeometry" )
        {
          if ( name == "displacement" )
          {
            m_currentStage = &m_currentMaterial->second.stageData[dp::fx::Domain::VERTEX];
          }
          else
          {
            DP_ASSERT( ( name == "cutout_opacity" ) || ( name == "normal" ) );
            m_currentStage = &m_currentMaterial->second.stageData[dp::fx::Domain::FRAGMENT];
          }
        }
        return true;
      }

      void MaterialBuilder::argumentEnd()
      {
      }

      bool MaterialBuilder::arrayBegin( std::string const& type, size_t size )
      {
        DP_ASSERT( !m_currentCall.empty() );
        bool traverse = (0 < size);
        if (traverse)
        {
          std::string t = translateType(convertColons(convertName(type)));
          m_currentCall.top()->arguments.push_back(std::make_pair(t, Argument(t)));
          m_currentCall.push(&m_currentCall.top()->arguments.back().second);
        }
        return traverse;
      }

      void MaterialBuilder::arrayEnd()
      {
        DP_ASSERT( 1 < m_currentCall.size() );
        DP_ASSERT( !m_currentCall.top()->arguments.empty() );
#if !defined(NDEBUG)
        for ( size_t i=1 ; i<m_currentCall.top()->arguments.size() ; i++ )
        {
          DP_ASSERT( m_currentCall.top()->arguments[0].first == m_currentCall.top()->arguments[i].first );
        }
#endif

        std::ostringstream oss;
        oss << m_currentCall.top()->arguments[0].first << "[" << m_currentCall.top()->arguments.size() << "]";
        m_currentCall.top()->name = oss.str();

        m_currentCall.pop();
      }

      bool MaterialBuilder::callBegin( std::string const& type, std::string const& name )
      {
        bool goOn = true;
        std::string callName = translateFunctionName( convertColons( convertName( stripArguments( name ) ) ) );

        static const std::map<std::string,std::pair<std::string,std::string>> constantFunctionsMap =
        {
          { "mdl_df_diffuseEDF",              std::make_pair( "vec4", "vec4( 0, 0, 0, 1 )" )  },
          { "mdl_df_lightProfileMaximum",     std::make_pair( "float", "0" )                  },
          { "mdl_df_lightProfilePower",       std::make_pair( "float", "0" )                  },
          { "mdl_df_measuredEDF",             std::make_pair( "vec4", "vec4( 0, 0, 0, 1 )" )  },
          { "mdl_df_spotEDF",                 std::make_pair( "vec4", "vec4( 0, 0, 0, 1 )" )  },
          { "mdl_state_normal",               std::make_pair( "vec3", "stateNormal" )         },
          { "mdl_state_roundedCornerNormal",  std::make_pair( "vec3", "stateNormal" )         }
        };
        std::map<std::string,std::pair<std::string,std::string>>::const_iterator it = constantFunctionsMap.find( callName );

        if ( it != constantFunctionsMap.end() )
        {
          goOn = false;
          m_currentCall.top()->arguments.push_back( it->second );
        }
        else
        {
          storeFunctionCall( callName );
          m_currentCall.top()->arguments.push_back( std::make_pair( translateType( convertColons( convertName( type ) ) ), Argument( callName ) ) );
          m_currentCall.push( &m_currentCall.top()->arguments.back().second );
        }
        return( goOn );
      }

      void MaterialBuilder::callEnd()
      {
        DP_ASSERT( 1 < m_currentCall.size() );

        // replace the last arguments of the BSDF functions (a string) by an vec3 argument named normal
        static const std::map<std::string,size_t> replaceLastArgumentFunctions =
        {
          { "mdl_df_backscatteringGlossyReflectionBSDF",  5 },
          { "mdl_df_diffuseReflectionBSDF",               3 },
          { "mdl_df_diffuseTransmissionBSDF",             2 },
          { "mdl_df_measuredBSDF",                        4 },
          { "mdl_df_simpleGlossyBSDF",                    6 },
          { "mdl_df_specularBSDF",                        3 }
        };
        std::map<std::string,size_t>::const_iterator it = replaceLastArgumentFunctions.find( m_currentCall.top()->name );
        if ( it != replaceLastArgumentFunctions.end() )
        {
          DP_ASSERT( m_currentCall.top()->arguments.size() == it->second );
          DP_ASSERT( m_currentCall.top()->arguments.back().first == "string" );
          DP_ASSERT( m_currentCall.top()->arguments.back().second.empty() );
          m_currentCall.top()->arguments.back() = std::make_pair( "vec3", Argument( "normal" ) );
        }

        // replace some functions with temporaries, which are evaluated just once
        static const std::set<std::string> localFunctions =
        {
          "mdl_base_blendColorLayers",
          "mdl_base_colorLayer",
          "mdl_base_fileBumpTexture",
          "mdl_base_fileTexture",
          "mdl_base_flowNoiseTexture",
          "mdl_base_tangentSpaceNormalTexture",
          "mdl_base_transformCoordinate"
        };
        bool adjust = (m_currentTemporaryIdx == ~0) && (localFunctions.find(m_currentCall.top()->name) != localFunctions.end());
        if (adjust)
        {
          TemporaryData td;
          //td.type = m_currentCall.top()->arguments.back().first;
          td.eval = m_currentCall.top()->name + "(";
          if (!m_currentCall.top()->arguments.empty())
          {
            td.eval += " " + resolveArgument(m_currentCall.top()->arguments[0].second);
            for (size_t i = 1; i<m_currentCall.top()->arguments.size(); i++)
            {
              td.eval += ", " + resolveArgument(m_currentCall.top()->arguments[i].second);
            }
            td.eval += " ";
          }
          td.eval += ")";

          m_currentMaterial->second.temporaries.insert(std::make_pair(++m_currentMaterial->second.maxTemporaryIndex, td));
        }

        m_currentCall.pop();

        if (adjust)
        {
          std::ostringstream oss;
          oss << "temporary" << m_currentMaterial->second.maxTemporaryIndex;
          m_currentCall.top()->arguments.back().second.name = oss.str();
          m_currentCall.top()->arguments.back().second.arguments.clear();
          m_currentMaterial->second.temporaries[m_currentMaterial->second.maxTemporaryIndex].type = m_currentCall.top()->arguments.back().first;
          m_currentStage->temporaries.insert(m_currentMaterial->second.maxTemporaryIndex);
        }

        // if the function returns a structure, store its usage
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        if ( !m_currentCall.top()->arguments.empty() && ( m_currentMaterial->second.structures.find( m_currentCall.top()->arguments.back().first ) != m_currentMaterial->second.structures.end() ) )
        {
          m_currentStage->structures.insert( m_currentCall.top()->arguments.back().first );
        }
      }

      void MaterialBuilder::defaultRef( std::string const& type )
      {
        DP_ASSERT( !m_currentCall.empty() );

        if ( ( type == "Bsdf" ) || ( type == "Edf" ) || ( type == "LightProfile" ) || ( type == "Vdf" ) )
        {
          m_currentCall.top()->arguments.push_back( std::make_pair( "vec4", Argument( "vec4(0,0,0,1)" ) ) );   // some default for a default BSDF or EDF or LightProfile
        }
        else
        {
          DP_ASSERT( type == "Texture" );
          m_currentCall.top()->arguments.push_back( std::make_pair( "sampler2D", Argument( "unknownTexture.png" ) ) );
          m_currentCall.top()->arguments.push_back( std::make_pair( "float", Argument( "1.0" ) ) );
        }
      }

      bool MaterialBuilder::enumTypeBegin( std::string const& name, size_t size )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        bool traverse = (m_currentMaterial->second.enums.find(name) == m_currentMaterial->second.enums.end());
        if (traverse)
        {
          DP_ASSERT(m_currentEnum == m_currentMaterial->second.enums.end());
          m_currentEnum = m_currentMaterial->second.enums.insert(std::make_pair(convertColons(convertName(name)), EnumData())).first;
          m_currentEnum->second.name = m_currentEnum->first;
          m_currentEnum->second.values.reserve(size);
        }
        return traverse;
      }

      void MaterialBuilder::enumTypeEnd()
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( m_currentEnum != m_currentMaterial->second.enums.end() );
        if ( m_currentStage )
        {
          m_currentStage->enums.insert( m_currentEnum->first );
        }
        m_currentEnum = m_currentMaterial->second.enums.end();
      }

      void MaterialBuilder::enumTypeValue( std::string const& name, int value )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( m_currentEnum != m_currentMaterial->second.enums.end() );
        m_currentEnum->second.values.push_back( std::make_pair( name, value ) );
      }

      bool MaterialBuilder::fieldBegin( std::string const& name )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( m_argument.empty() );
        DP_ASSERT( m_currentCall.empty() );

        bool handleField = ( name != "volume" );
        if ( handleField )
        {
          m_argument.name = name;
          m_currentCall.push( &m_argument );
          if ( name != "geometry" )   // currentStage for "geometry" is handled in argumentBegin !
          {
            DP_ASSERT( ( name == "backface" ) || ( name == "ior" ) || ( name == "surface" ) || ( name == "thin_walled" ) );
            m_currentStage = &m_currentMaterial->second.stageData[dp::fx::Domain::FRAGMENT];
          }
        }
        return( handleField );
      }

      void MaterialBuilder::fieldEnd()
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( m_currentCall.size() == 1 );
        DP_ASSERT( m_currentStage != nullptr );

        std::string name = m_currentCall.top()->name;
        if ( name == "backface" )
        {
          getSurfaceData( m_currentMaterial->second.backfaceData );
        }
        else if ( name == "geometry" )
        {
          DP_ASSERT( ( m_currentCall.top()->arguments.size() == 1 ) && ( m_currentCall.top()->arguments[0].first == "_materialGeometry" ) );
          DP_ASSERT( m_currentCall.top()->arguments[0].second.arguments.size() == 3 );
          m_currentMaterial->second.geometryData.displacement  = resolveArgument( m_currentCall.top()->arguments[0].second.arguments[0].second );
          m_currentMaterial->second.geometryData.cutoutOpacity = resolveArgument( m_currentCall.top()->arguments[0].second.arguments[1].second );
          m_currentMaterial->second.geometryData.normal        = resolveArgument( m_currentCall.top()->arguments[0].second.arguments[2].second );
          m_currentMaterial->second.transparent |= ( m_currentMaterial->second.geometryData.cutoutOpacity != "1" );
        }
        else if ( name == "ior" )
        {
          DP_ASSERT( m_currentCall.top()->arguments.size() == 1 );
          m_currentMaterial->second.ior = resolveArgument( m_currentCall.top()->arguments[0].second );
        }
        else if ( name == "surface" )
        {
          getSurfaceData( m_currentMaterial->second.surfaceData );
        }
        else
        {
          DP_ASSERT( name == "thin_walled" );
          DP_ASSERT( m_currentCall.top()->arguments.size() == 1 );
          m_currentMaterial->second.thinWalled = resolveArgument( m_currentCall.top()->arguments[0].second );
        }
        m_currentCall.pop();
        m_argument.clear();
        m_currentStage = nullptr;
      }

      bool MaterialBuilder::fileBegin( std::string const& name )
      {
        DP_ASSERT( dp::util::fileExists( name ) );
        m_materials.clear();
        return true;
      }

      void MaterialBuilder::fileEnd()
      {
        m_temporarySamplerMap.clear();
      }

      bool MaterialBuilder::materialBegin( std::string const& name, dp::math::Vec4ui const& hash )
      {
        DP_ASSERT( m_materials.find( name ) == m_materials.end() );
        m_currentMaterial = m_materials.insert(std::make_pair(name, MaterialData())).first;
        m_currentEnum = m_currentMaterial->second.enums.end();

        // store function "luminance", used in evalIOR into the fragment stage
        m_currentStage = &m_currentMaterial->second.stageData[dp::fx::Domain::FRAGMENT];
        storeFunctionCall( "mdl_math_luminance" );
        m_currentStage = nullptr;

        m_currentMaterial->second.varyings.insert( "varNormal" );
        m_currentMaterial->second.varyings.insert( "varTexCoord0" );
        m_currentMaterial->second.varyings.insert( "varWorldPos" );
        m_currentMaterial->second.varyings.insert( "varEyePos" );
        m_currentMaterial->second.varyings.insert( "varTangent" );
        m_currentMaterial->second.varyings.insert( "varBinormal" );
        return true;
      }

      void MaterialBuilder::materialEnd()
      {
        m_currentMaterial = m_materials.end();
        m_temporaryBuddies.clear();
        m_temporarySamplerMap.clear();
      }

      bool MaterialBuilder::matrixBegin( std::string const& type )
      {
        DP_ASSERT( !m_currentCall.empty() );
        std::string t = translateType( convertColons( convertName( type ) ) );
        m_currentCall.top()->arguments.push_back( std::make_pair( t, Argument( t ) ) );
        m_currentCall.push( &m_currentCall.top()->arguments.back().second );
        return true;
      }

      void MaterialBuilder::matrixEnd()
      {
        DP_ASSERT( 1 < m_currentCall.size() );
        DP_ASSERT( !m_currentCall.top()->arguments.empty() );
#if !defined(NDEBUG)
        for ( size_t i=1 ; i<m_currentCall.top()->arguments.size() ; i++ )
        {
          DP_ASSERT( m_currentCall.top()->arguments[0].first == m_currentCall.top()->arguments[i].first );
        }
#endif
        m_currentCall.pop();
      }

      bool MaterialBuilder::parameterBegin( unsigned int index, std::string const& name )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( m_argument.empty() );
        DP_ASSERT( m_currentCall.empty() );
        DP_ASSERT( !m_insideParameter );
        DP_ASSERT( !boost::starts_with( name, "unnamedParameter" ) );

        if ( name.empty() )
        {
          std::ostringstream oss;
          oss << "unnamedParameter" << index;
          m_argument.name = oss.str();
        }
        else
        {
          m_argument.name = name;
        }
        m_currentCall.push( &m_argument );

        m_insideParameter = true;
        return true;
      }

      void MaterialBuilder::parameterEnd()
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( !m_argument.empty() && ( m_currentCall.size() == 1 ) && !m_currentCall.top()->arguments.empty() );
        DP_ASSERT( ( m_currentCall.top()->arguments.size() == 1 ) || ( ( m_currentCall.top()->arguments.size() == 2 ) && ( m_currentCall.top()->arguments[0].first == "sampler2D" ) ) );
        DP_ASSERT( m_insideParameter );

        std::string name = m_currentCall.top()->name;
        std::string type = m_currentCall.top()->arguments.front().first;
        std::string value = resolveArgument( m_currentCall.top()->arguments.front().second );
        std::string annotations = resolveAnnotations();

        if ( boost::starts_with( name, "unnamedParameter" ) )
        {
          DP_ASSERT( annotations.empty() );
          annotations = "_anno_hidden()";
        }

        std::string semantic = "VALUE";
        if ( type == "color" )
        {
          type = "vec3";
          semantic = "COLOR";
        }

        if ( ( type == "sampler2D" ) || ( type == "sampler3D" ) )
        {
          DP_ASSERT( !value.empty() );
          std::string fileName = m_fileFinder.findRecursive( value );
          m_currentMaterial->second.parameters.push_back( ParameterData( type, name, fileName.empty() ? value : fileName, "VALUE", annotations ) );
        }
        else
        {
          m_currentMaterial->second.parameters.push_back( ParameterData( type, name, stripParameterValue( value ), semantic, annotations ) );
        }
        m_currentMaterial->second.parameterIndirection.push_back( dp::checked_cast<unsigned int>(m_currentMaterial->second.parameters.size()) - 1 );

        if ( m_currentCall.top()->arguments.size() == 2 )
        {
          DP_ASSERT( m_currentCall.top()->arguments.back().first == "float" );
          DP_ASSERT( m_currentCall.top()->arguments.back().second.arguments.empty() );
          std::string gammaName = name + "Gamma";
          m_currentMaterial->second.parameters.push_back( ParameterData( "float", gammaName, m_currentCall.top()->arguments.back().second.name, "VALUE", "_anno_hidden()" ) );
        }
        m_currentCall.pop();
        m_argument.clear();
        m_insideParameter = false;
      }

      void MaterialBuilder::referenceParameter( unsigned int idx )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( !m_currentCall.empty() );
        DP_ASSERT( idx < m_currentMaterial->second.parameterIndirection.size() );

        unsigned int redirectedIndex = m_currentMaterial->second.parameterIndirection[idx];
        DP_ASSERT( redirectedIndex < m_currentMaterial->second.parameters.size() );

        ParameterData const& pd = m_currentMaterial->second.parameters[redirectedIndex];
        m_currentCall.top()->arguments.push_back( std::make_pair( pd.type, Argument( pd.name ) ) );
        m_currentStage->parameters.insert( redirectedIndex );

        if ( pd.type == "sampler2D" )
        {
          ++redirectedIndex;
          DP_ASSERT( std::find( m_currentMaterial->second.parameterIndirection.begin(), m_currentMaterial->second.parameterIndirection.end(), redirectedIndex ) == m_currentMaterial->second.parameterIndirection.end() );
          DP_ASSERT( redirectedIndex < m_currentMaterial->second.parameters.size() );
          ParameterData const& secondPD = m_currentMaterial->second.parameters[redirectedIndex];
          DP_ASSERT( secondPD.type == "float" );
          m_currentCall.top()->arguments.push_back( std::make_pair( secondPD.type, Argument( secondPD.name ) ) );
          m_currentStage->parameters.insert( redirectedIndex );
        }
        else if ( m_currentMaterial->second.enums.find( pd.type ) != m_currentMaterial->second.enums.end() )
        {
          DP_ASSERT( m_currentStage );
          m_currentStage->enums.insert( pd.type );
          m_currentCall.top()->arguments.back().first = "int";    // replace all enum types by "int"!
        }
      }

      void MaterialBuilder::referenceTemporary( unsigned int idx )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( !m_currentCall.empty() );

        std::map<unsigned int,std::string>::const_iterator it = m_temporarySamplerMap.find( idx );
        if ( it != m_temporarySamplerMap.end() )
        {
          m_currentCall.top()->arguments.push_back( std::make_pair( "sampler2D", Argument( it->second ) ) );
        }

        std::map<unsigned int, TemporaryData>::const_iterator tit = m_currentMaterial->second.temporaries.find(idx);
        DP_ASSERT(tit != m_currentMaterial->second.temporaries.end());

        std::ostringstream oss;
        oss << "temporary" << idx;

        m_currentCall.top()->arguments.push_back(std::make_pair(tit->second.type, oss.str()));
        m_currentStage->append( m_currentMaterial->second.temporaries[idx].stage );
        m_currentStage->temporaries.insert( idx );

        std::map<unsigned int, unsigned int>::const_iterator bit = m_temporaryBuddies.find( idx );
        if ( bit != m_temporaryBuddies.end() )
        {
          m_currentStage->temporaries.insert( bit->second );
        }
      }

      bool MaterialBuilder::structureBegin( std::string const& type )
      {
        DP_ASSERT( !m_currentCall.empty() );

        std::string callName = convertColons( convertName( type ) );
        if (m_currentStage)
        {
          m_currentStage->structures.insert(callName);
        }
        if ( !m_insideAnnotation )
        {
          m_currentCall.top()->arguments.push_back( std::make_pair( callName, Argument( callName ) ) );
          m_currentCall.push( &m_currentCall.top()->arguments.back().second );
        }
        return true;
      }

      void MaterialBuilder::structureEnd()
      {
        DP_ASSERT( !m_currentCall.empty() );
        if ( !m_insideAnnotation )
        {
          m_currentCall.pop();
        }
      }

      bool MaterialBuilder::structureTypeBegin( std::string const& name )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        bool traverse = (m_currentMaterial->second.structures.find(convertColons(convertName(name))) == m_currentMaterial->second.structures.end());
        if (traverse)
        {
          m_structureStack.push(StructureData());
          m_structureStack.top().name = convertColons(convertName(name));
        }
        return traverse;
      }

      void MaterialBuilder::structureTypeElement( std::string const& type, std::string const& name )
      {
        DP_ASSERT( !m_structureStack.empty() );
        m_structureStack.top().members.push_back( std::make_pair( translateType( convertColons( convertName( type ) ) ), convertName( name ) ) );
      }

      void MaterialBuilder::structureTypeEnd()
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( !m_structureStack.empty() );
        DP_ASSERT( m_currentMaterial->second.structures.find( m_structureStack.top().name ) == m_currentMaterial->second.structures.end() );
        m_currentMaterial->second.structures.insert( std::make_pair( m_structureStack.top().name, m_structureStack.top( ) ) );

        if ( m_currentStage )
        {
          m_currentStage->structures.insert( m_structureStack.top().name );
        }

        m_structureStack.pop();
      }

      bool MaterialBuilder::temporaryBegin( unsigned int idx )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( m_argument.empty() );
        DP_ASSERT( m_currentCall.empty() );
        DP_ASSERT( m_currentTemporaryIdx == ~0 );
        DP_ASSERT( m_currentStage == nullptr );
        DP_ASSERT((m_currentMaterial->second.maxTemporaryIndex == ~0) || (m_currentMaterial->second.maxTemporaryIndex < idx));

        m_currentTemporaryIdx = idx;
        m_currentCall.push( &m_argument );
        m_currentStage = &m_temporaryStage;
        m_currentMaterial->second.maxTemporaryIndex = idx;
        return true;
      }

      void MaterialBuilder::temporaryEnd()
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( !m_argument.empty() );
        DP_ASSERT( m_currentTemporaryIdx != ~0 );
        DP_ASSERT( m_currentStage == &m_temporaryStage );
        DP_ASSERT( m_currentCall.size() == 1 );

        if ( m_currentCall.top()->arguments.size() == 1 )
        {
          // create an entry in the temporaries map
          TemporaryData td;
          td.stage = m_temporaryStage;
          m_temporaryStage.clear();
          td.type = m_currentCall.top()->arguments.back().first;
          td.eval = resolveArgument( m_currentCall.top()->arguments.back().second );

          m_currentMaterial->second.temporaries.insert( std::make_pair( m_currentTemporaryIdx, td ) );
        }
        else
        {
          DP_ASSERT( m_currentCall.top()->arguments.size() == 2 );
          if ( m_currentCall.top()->arguments[0].first == "sampler2D" )
          {
            // sampler2D can't be used as temporaries
            DP_ASSERT( m_temporarySamplerMap.find( m_currentTemporaryIdx ) == m_temporarySamplerMap.end() );
            m_temporarySamplerMap[m_currentTemporaryIdx] = resolveArgument( m_currentCall.top()->arguments[0].second );

            // create an entry in the temporaries map with index m_currentTemporaryIdx for the gamma
            TemporaryData td;
            td.stage = m_temporaryStage;
            m_temporaryStage.clear();
            td.type = m_currentCall.top()->arguments[1].first;
            td.eval = resolveArgument( m_currentCall.top()->arguments[1].second );
            m_currentMaterial->second.temporaries.insert( std::make_pair( m_currentTemporaryIdx, td ) );
          }
          else
          {
            DP_ASSERT( ( m_currentCall.top()->arguments[0].first == "vec4" ) && ( m_currentCall.top()->arguments[1].first == "_materialEmission" ) );
            // create two entries in the temporaries map, one with index m_currentTemporaryIdx, one with index ( UINT_MAX - m_currentTemporaryIdx )
            TemporaryData td;
            td.stage = m_temporaryStage;
            m_temporaryStage.clear();
            td.type = m_currentCall.top()->arguments[0].first;
            td.eval = resolveArgument( m_currentCall.top()->arguments[0].second );
            m_currentMaterial->second.temporaries.insert( std::make_pair( m_currentTemporaryIdx, td ) );

            td.type = m_currentCall.top()->arguments[1].first;
            td.eval = resolveArgument( m_currentCall.top()->arguments[1].second );
            m_currentMaterial->second.temporaries.insert( std::make_pair( std::numeric_limits<unsigned int>::max() - m_currentTemporaryIdx, td ) );

            DP_ASSERT( m_temporaryBuddies.find( m_currentTemporaryIdx ) == m_temporaryBuddies.end() );
            m_temporaryBuddies[m_currentTemporaryIdx] = std::numeric_limits<unsigned int>::max() - m_currentTemporaryIdx;
          }
        }

        m_currentCall.pop();
        m_argument.clear();
        m_currentTemporaryIdx = ~0;
        m_currentStage = nullptr;
      }

      void MaterialBuilder::valueBool( bool value )
      {
        DP_ASSERT( !m_currentCall.empty() );
        m_currentCall.top()->arguments.push_back( std::make_pair( "bool", Argument( value ? "true" : "false" ) ) );
      }

      void MaterialBuilder::valueBsdfMeasurement( std::string const& value )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( !m_currentCall.empty() );

        std::string usedName = value.empty() ? "unknownTexture.png" : value;
        if ( !m_insideParameter )
        {
          std::ostringstream oss;
          oss << "hiddenArgument" << m_currentMaterial->second.parameters.size();
          usedName = oss.str();

          m_currentStage->parameters.insert( dp::checked_cast<unsigned int>( m_currentMaterial->second.parameters.size() ) );
          m_currentMaterial->second.parameters.push_back( ParameterData( "sampler3D", usedName, value, "VALUE", "_anno_hidden()" ) );
        }
        m_currentCall.top()->arguments.push_back( std::make_pair( "sampler3D", Argument( usedName ) ) );
      }

      void MaterialBuilder::valueColor( dp::math::Vec3f const& value )
      {
        DP_ASSERT( !m_currentCall.empty() );

        std::ostringstream oss;
        oss << "vec3( " << value[0] << ", " << value[1] << ", " << value[2] << " )";
        m_currentCall.top()->arguments.push_back( std::make_pair( m_insideParameter ? "color" : "vec3", Argument( oss.str() ) ) );
      }

      void MaterialBuilder::valueEnum( std::string const& type, int value, std::string const& name )
      {
        std::string convertedType = convertColons( convertName( type ) );
        if ( m_currentStage )
        {
          m_currentStage->enums.insert( convertedType );
        }

        DP_ASSERT( !m_currentCall.empty() );
        m_currentCall.top()->arguments.push_back( std::make_pair( convertedType, Argument( name ) ) );
      }

      void MaterialBuilder::valueFloat( float value )
      {
        DP_ASSERT( !m_currentCall.empty() );

        std::ostringstream oss;
        oss << value;
        m_currentCall.top()->arguments.push_back( std::make_pair( "float", Argument( oss.str() ) ) );
      }

      void MaterialBuilder::valueInt( int value )
      {
        DP_ASSERT( !m_currentCall.empty() );

        std::ostringstream oss;
        oss << value;
        m_currentCall.top()->arguments.push_back( std::make_pair( "int", Argument( oss.str() ) ) );
      }

      void MaterialBuilder::valueLightProfile( std::string const& value )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        DP_ASSERT( !m_currentCall.empty() );

        std::string usedName = value.empty() ? "unknownLightProfile.ies" : value;
        if ( !m_insideParameter )
        {
          std::ostringstream oss;
          oss << "hiddenArgument" << m_currentMaterial->second.parameters.size();
          usedName = oss.str();

          m_currentStage->parameters.insert( dp::checked_cast<unsigned int>( m_currentMaterial->second.parameters.size() ) );
          m_currentMaterial->second.parameters.push_back( ParameterData( "lightProfile", usedName, value, "VALUE", "_anno_hidden()" ) );
        }
        m_currentCall.top()->arguments.push_back( std::make_pair( "lightProfile", Argument( usedName ) ) );
      }

      void MaterialBuilder::valueString( std::string const& value )
      {
        DP_ASSERT( !m_currentCall.empty() );
        m_currentCall.top()->arguments.push_back( std::make_pair( "string", Argument( value ) ) );
      }

      void MaterialBuilder::valueTexture( std::string const& name, GammaMode gamma )
      {
        DP_ASSERT( !m_currentCall.empty() );

        // for texture data, we add one additional argument gamma onto the stack
        std::string gammaString;
        switch( gamma )
        {
          case MDLTokenizer::GammaMode::DEFAULT:
            gammaString = "0.0";
            break;
          case MDLTokenizer::GammaMode::LINEAR:
            gammaString = "1.0";
            break;
          case MDLTokenizer::GammaMode::SRGB:
            gammaString = "2.2";
            break;
          default :
            DP_ASSERT( false );
        }

        std::string usedName = name.empty() ? "unknownTexture.png" : name;
        if ( !m_insideParameter )
        {
          std::ostringstream oss;
          oss << "hiddenArgument" << m_currentMaterial->second.parameters.size();
          usedName = oss.str();

          m_currentStage->parameters.insert( dp::checked_cast<unsigned int>( m_currentMaterial->second.parameters.size() ) );
          m_currentMaterial->second.parameters.push_back( ParameterData( "sampler2D", usedName, name, "VALUE", "_anno_hidden()" ) );

          std::string gammaName = usedName + "Gamma";
          m_currentStage->parameters.insert( dp::checked_cast<unsigned int>( m_currentMaterial->second.parameters.size() ) );
          m_currentMaterial->second.parameters.push_back( ParameterData( "float", gammaName, gammaString, "VALUE", "_anno_hidden()" ) );
        }
        m_currentCall.top()->arguments.push_back( std::make_pair( "sampler2D", Argument( usedName ) ) );
        m_currentCall.top()->arguments.push_back( std::make_pair( "float", Argument( gammaString ) ) );
      }

      bool MaterialBuilder::vectorBegin( std::string const& type )
      {
        DP_ASSERT( !m_currentCall.empty() );
        std::string t = translateType( convertColons( convertName( type ) ) );
        m_currentCall.top()->arguments.push_back( std::make_pair( t, Argument( t ) ) );
        m_currentCall.push( &m_currentCall.top()->arguments.back().second );
        return true;
      }

      void MaterialBuilder::vectorEnd()
      {
        DP_ASSERT( 1 < m_currentCall.size() );
        DP_ASSERT( !m_currentCall.top()->arguments.empty() );
#if !defined(NDEBUG)
        for ( size_t i=1 ; i<m_currentCall.top()->arguments.size() ; i++ )
        {
          DP_ASSERT( m_currentCall.top()->arguments[0].first == m_currentCall.top()->arguments[i].first );
        }
#endif
        m_currentCall.pop();
      }

      void MaterialBuilder::getSurfaceData( SurfaceData & surfaceData )
      {
        DP_ASSERT( !m_currentCall.empty() );
        DP_ASSERT( ( m_currentCall.top()->name == "backface" ) || ( m_currentCall.top()->name == "surface" ) );
        DP_ASSERT( m_currentCall.top()->arguments.size() == 1 );
        DP_ASSERT( m_currentCall.top()->arguments[0].first == "_materialSurface" );
        if ( m_currentCall.top()->arguments[0].second.arguments.size() == 2 )
        {
          DP_ASSERT( boost::ends_with( m_currentCall.top()->arguments[0].second.name, "_materialSurface" ) );
          surfaceData.scattering = resolveArgument( m_currentCall.top()->arguments[0].second.arguments[0].second );
          surfaceData.emission   = resolveArgument( m_currentCall.top()->arguments[0].second.arguments[1].second );
        }
        else
        {
          std::string result = resolveArgument( m_currentCall.top()->arguments[0].second );
          surfaceData.scattering = "( " + result + " ).scattering";
          surfaceData.emission   = "( " + result + " ).emission";
        }
        DP_ASSERT( m_temporaryBuddies.empty() );
      }

      MaterialBuilder::Argument & MaterialBuilder::getTargetArg( Argument & arg, size_t idx ) const
      {
        if ( arg.arguments[idx].second.name == "mdl_df_measuredCurveFactor" )
        {
          return( arg.arguments[idx].second.arguments[1].second );
        }
        else
        {
          return( arg.arguments[idx].second );
        }
      }

      std::string MaterialBuilder::resolveAnnotations()
      {
        std::string anno;
        if ( !m_annotations.empty() )
        {
          anno = resolveArgument( m_annotations[0] );
          for ( size_t i=1 ; i<m_annotations.size() ; i++ )
          {
            anno += "; " + resolveArgument( m_annotations[i] );
          }
          m_annotations.clear();
        }
        return( anno );
      }

      std::string MaterialBuilder::resolveArgument( Argument & arg, bool embrace )
      {
        // determine transparency
        m_currentMaterial->second.transparent |= ( arg.name == "mdl_df_specularBSDF" ) && ( arg.arguments[1].second.name != "scatter_reflect" );

        // for some functions, add one argument at the end of one other
        static const std::map<std::string,std::pair<size_t,size_t>> remapArgumentFunctions =
        {
          { "mdl_df_customCurveLayer",    { 6, 4 } },
          { "mdl_df_fresnelLayer",        { 4, 2 } },
          { "mdl_df_measuredCurveLayer",  { 4, 2 } },
          { "mdl_df_weightedLayer",       { 3, 1 } }
        };
        std::map<std::string,std::pair<size_t,size_t>>::const_iterator remapIt = remapArgumentFunctions.find( arg.name );
        if ( remapIt != remapArgumentFunctions.end() )
        {
          DP_ASSERT( ( remapIt->second.second < remapIt->second.first ) && ( remapIt->second.first <= arg.arguments.size() ) );
          if ( arg.arguments[remapIt->second.second].second.name == "mdl_df_normalizedMix" )
          {
            // this is very special !!
            DP_ASSERT( arg.arguments[remapIt->second.second].second.arguments.size() == 1 );
            DP_ASSERT( boost::starts_with( arg.arguments[remapIt->second.second].second.arguments[0].first, "_df_bsdfComponent[" )
                    && ( arg.arguments[remapIt->second.second].second.arguments[0].first.back() == ']' ) );
            DP_ASSERT( arg.arguments[remapIt->second.second].second.arguments[0].second.name == "mdl_T[]" );
            for ( size_t i=0 ; i<arg.arguments[remapIt->second.second].second.arguments[0].second.arguments.size() ; i++ )
            {
              DP_ASSERT( arg.arguments[remapIt->second.second].second.arguments[0].second.arguments[i].first == "_df_bsdfComponent" );
              DP_ASSERT( arg.arguments[remapIt->second.second].second.arguments[0].second.arguments[i].second.name == "mdl_df_bsdfComponent" );
              DP_ASSERT( arg.arguments[remapIt->second.second].second.arguments[0].second.arguments[i].second.arguments.size() == 2 );
              arg.arguments[remapIt->second.second].second.arguments[0].second.arguments[i].second.arguments[1].second.arguments.back() = arg.arguments[remapIt->second.first];
            }
          }
          else
          {
            Argument & targetArg = getTargetArg( arg, remapIt->second.second );
            if (!targetArg.arguments.empty())
            {
              DP_ASSERT(targetArg.arguments.back().first == arg.arguments[remapIt->second.first].first);
              targetArg.arguments.back() = arg.arguments[remapIt->second.first];
            }
          }
        }

        // remove last argument on some functions
        static const std::map<std::string,size_t> skipLastArgumentFunctions =
        {
          { "mdl_df_customCurveLayer",  7 },
          { "mdl_df_weightedLayer",     4 }
        };
        std::map<std::string,size_t>::const_iterator it = skipLastArgumentFunctions.find( arg.name );
        if ( it != skipLastArgumentFunctions.end() )
        {
          DP_ASSERT( arg.arguments.size() == it->second );
          arg.arguments.pop_back();
        }

        // special handling for array constructor
        if ( arg.name == "mdl_T[]" )
        {
          DP_ASSERT( !arg.arguments.empty() );
          std::ostringstream oss;
          oss << arg.arguments[0].first << "[" << arg.arguments.size() << "]";
          arg.name = oss.str();
        }

        std::string result;
        bool doEmbrace = embrace && !arg.arguments.empty() && (arg.name.find("operator") != std::string::npos);
        if (doEmbrace)
        {
          result = "( ";
        }
        // some functions need some special handling
        if ( ( arg.name == "mdl_float2@" ) || ( arg.name == "mdl_float3@" ) )
        {
          DP_ASSERT( arg.arguments.size() == 2 );
          DP_ASSERT( ( arg.arguments[1].second.name == "0" ) || ( arg.arguments[1].second.name == "1" ) || ( ( arg.name == "mdl_float3@" ) && ( arg.arguments[1].second.name == "2" ) ) );
          result += resolveArgument( arg.arguments[0].second ) + "." + ( ( arg.arguments[1].second.name == "0" ) ? "x" : ( arg.arguments[1].second.name == "1" ) ? "y" : "z" );
        }
        else if ( boost::algorithm::starts_with( arg.name, "mdl_operator" ) )
        {
          switch( arg.arguments.size() )
          {
            case 1:
              DP_ASSERT((arg.name == "mdl_operator-") || (arg.name == "mdl_operator!"));
              result += arg.name.substr(12) + resolveArgument(arg.arguments[0].second, true);
              break;
            case 2:
#if !defined(NDEBUG)
              {
                static std::set<std::string> checkedValues = {{"+"}, {"-"}, {"*"}, {"/"}, {"<"}, {"=="}, {"<="}, {"!="}, {"&&"}};
                DP_ASSERT(checkedValues.find(arg.name.substr(12)) != checkedValues.end());
              }
#endif
              result += resolveArgument(arg.arguments[0].second, true) + " " + arg.name.substr(12) + " " + resolveArgument(arg.arguments[1].second, true);
              break;
            case 3 :
              DP_ASSERT( arg.name == "mdl_operator?" );
              result += resolveArgument( arg.arguments[0].second, true ) + " ? " + resolveArgument( arg.arguments[1].second, true ) + " : " + resolveArgument( arg.arguments[2].second, true );
              break;
            default :
              DP_ASSERT( false );
              break;
          }
        }
        else if ( arg.name == "mdl_tex_operator?" )
        {
          DP_ASSERT( arg.arguments.size() == 3 );
          result += resolveArgument( arg.arguments[0].second, true ) + " ? " + resolveArgument( arg.arguments[1].second, true ) + " : " + resolveArgument( arg.arguments[2].second, true );
        }
        else if ( !arg.arguments.empty() && ( arg.name.find( '.' ) != std::string::npos ) )
        {
          // args with arguments (i.e. function calls) with '.'
          DP_ASSERT(arg.arguments.size() == 1);
          result += resolveArgument( arg.arguments[0].second ) + arg.name.substr( arg.name.find( '.' ) );
        }
        else
        {
          result += arg.name;
          if ( !arg.arguments.empty() )
          {
            result += "( " + resolveArgument( arg.arguments[0].second );
            for ( size_t i=1 ; i<arg.arguments.size() ; i++ )
            {
              result += ", " + resolveArgument( arg.arguments[i].second );
            }
            result += " )";
          }
        }
        if (doEmbrace)
        {
          result += " )";
        }
        return( result );
      }

      void MaterialBuilder::storeFunctionCall( std::string const& name )
      {
        static const std::set<std::string> ignoreCalls =
        {
          "abs", "cos", "mat4", "max", "mdl_materialGeometry", "sin", "vec3"
        };

        if ( !m_insideAnnotation && ( name.find_first_of( "+-*/.[?&!=<@" ) == std::string::npos ) && ( ignoreCalls.find( name ) == ignoreCalls.end() ) )
        {
          std::vector<std::string>::const_iterator it = std::find( m_currentStage->functions.begin(), m_currentStage->functions.end(), name );
          if ( it == m_currentStage->functions.end() )
          {
            std::map<std::string,FunctionData>::const_iterator fit = m_functions.find( name );
            if ( fit != m_functions.end() )
            {
              for ( std::set<std::string>::const_iterator fdit = fit->second.functionDependencies.begin() ; fdit != fit->second.functionDependencies.end() ; ++fdit )
              {
                triggerTokenizeFunctionReturnType(unconvertName(convertUnderscores(*fdit)));
                storeFunctionCall( *fdit );
              }
              m_currentMaterial->second.varyings.insert( fit->second.varyingDependencies.begin(), fit->second.varyingDependencies.end() );
            }
            m_currentStage->functions.push_back( name );
          }
        }
      }

      std::string MaterialBuilder::translateType( std::string const& type )
      {
        DP_ASSERT( m_currentMaterial != m_materials.end() );
        std::string baseType = type.substr( 0, type.find( '[' ) );
        std::string translatedType;
        if ( m_currentMaterial->second.enums.find( baseType ) != m_currentMaterial->second.enums.end() )
        {
          translatedType = "int";   // map all enums to int
        }
        else if ( m_currentMaterial->second.structures.find( baseType ) != m_currentMaterial->second.structures.end() )
        {
          translatedType = baseType;
        }
        else
        {
          const std::map<std::string,std::string> typeMap =
          {
            { "Bool",               "bool"      },
            { "BSDF",               "vec4"      },
            { "BsdfMeasurement",    "sampler3D" },
            { "Color",              "vec3"      },
            { "EDF",                "vec4"      },
            { "Float",              "float"     },
            { "Float<2>",           "vec2"      },
            { "Float<3>",           "vec3"      },
            { "Float<4>",           "vec4"      },
            { "Float<3,3>",         "mat3"      },
            { "Float<4,4>",         "mat4"      },
            { "LightProfile",       "vec4"      },
            { "Sint32",             "int"       },
            { "Texture",            "sampler2D" },
            { "VDF",                "vec4"      }
          };
          std::map<std::string,std::string>::const_iterator it = typeMap.find( baseType );
          if ( it != typeMap.end() )
          {
            translatedType = it->second;
          }
          else
          {
            const std::set<std::string> annotationTypes =
            {
              "_anno_description", "_anno_displayName", "_anno_hardRange", "_anno_inGroup", "_anno_softRange"
            };
            std::set<std::string>::const_iterator it = annotationTypes.find( baseType );
            if ( it != annotationTypes.end() )
            {
              translatedType = *it;
            }
            else
            {
#if !defined(NDEBUG)
              const std::set<std::string> ignoreSet =
              {
                "bool", "float", "int", "Parameter", "Ref", "String", "Temporary"
              };
              DP_ASSERT( ignoreSet.find( baseType ) != ignoreSet.end() );
#endif
              translatedType = baseType;
            }
          }
        }
        if ( baseType != type )
        {
          translatedType += type.substr( type.find( '[' ) );
        }
        return( translatedType );
      }

    } // mdl
  } // fx
} // dp