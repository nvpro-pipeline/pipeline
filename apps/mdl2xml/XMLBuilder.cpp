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

#include "XMLBuilder.h"
#include <dp/util/File.h>

XMLBuilder::XMLBuilder()
  : m_enumElement( nullptr )
{
}

TiXmlElement * XMLBuilder::getXMLTree()
{
  return( m_libraryElement );
}

bool XMLBuilder::annotationBegin( std::string const& name )
{
  m_materialElements.back().push( new TiXmlElement( "annotation" ) );
  m_materialElements.back().top()->SetAttribute( "text", std::string( name + "(" ).c_str() );
  return true;
}

void XMLBuilder::annotationEnd()
{
  DP_ASSERT( ! m_materialElements.back().empty() );
  DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "annotation" ) == 0 );
  std::string text = m_materialElements.back().top()->Attribute( "text" );
  if ( text.back() == ',' )
  {
    text.pop_back();
  }
  text += ")";
  m_materialElements.back().top()->SetAttribute( "text", text.c_str() );
  popElement();
}

bool XMLBuilder::argumentBegin( unsigned int idx, std::string const& type, std::string const& name )
{
  m_currentType = type;
  m_materialElements.back().push( new TiXmlElement( "argument" ) );
  m_materialElements.back().top()->SetAttribute( "name", name.c_str() );
  return true;
}

void XMLBuilder::argumentEnd()
{
  m_materialElements.back().top()->FirstChildElement();
  popElement();
}

bool XMLBuilder::arrayBegin( std::string const& type, size_t size )
{
  DP_ASSERT( !m_materialElements.back().empty() );
  DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 );
  bool traverse = (0 < size);
  if (traverse)
  {
    m_materialElements.back().push(new TiXmlElement("array"));
    m_materialElements.back().top()->SetAttribute("type", type.c_str());
  }
  return traverse;
}

void XMLBuilder::arrayEnd()
{
  DP_ASSERT( !m_materialElements.back().empty() );
  DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "array" ) == 0 );
  popElement();
  DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 );
}

bool XMLBuilder::callBegin( std::string const& type, std::string const& name )
{
  m_materialElements.back().push( new TiXmlElement( "call" ) );
  m_materialElements.back().top()->SetAttribute( "name", name.c_str() );
  m_materialElements.back().top()->SetAttribute( "type", type.c_str() );
  return( true );
}

void XMLBuilder::callEnd()
{
  if ( strcmp( m_materialElements.back().top()->Value(), "call" ) == 0 )
  {
    popElement();
  }
}

void XMLBuilder::defaultRef( std::string const& type )
{
  const std::map<std::string,std::string> defaultRefMap =
  {
    { "Bsdf",         "0 0 0 1"             },
    { "Edf",          "0 0 0 1"             },
    { "LightProfile", "0 0 0 1"             },
    { "Texture",      "unknownTexture.png"  },
    { "Vdf",          "0 0 0 1"             }
  };
  std::map<std::string,std::string>::const_iterator it = defaultRefMap.find( type );
  DP_ASSERT( it != defaultRefMap.end() );

  DP_ASSERT( !m_materialElements.back().empty() );
  if ( ( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 )
    || ( strcmp( m_materialElements.back().top()->Value(), "parameter" ) == 0 ) )
  {
    m_materialElements.back().top()->SetAttribute( "value", it->second.c_str() );
  }
  else if ( strcmp( m_materialElements.back().top()->Value(), "parameter_struct" ) == 0 )
  {
    std::string value;
    if ( m_materialElements.back().top()->Attribute( "value" ) )
    {
      value = std::string( m_materialElements.back().top()->Attribute( "value" ) ) + " ";
    }
    m_materialElements.back().top()->SetAttribute( "value", ( value + it->second ).c_str() );
  }
  else
  {
    m_materialElements.back().push( new TiXmlElement( "data" ) );
    m_materialElements.back().top()->SetAttribute( "type", type.c_str() );
    m_materialElements.back().top()->SetAttribute( "value", it->second.c_str() );
    popElement();
  }
}

bool XMLBuilder::enumTypeBegin( std::string const& name, size_t size )
{
  DP_ASSERT( m_enumElement == nullptr );
  bool traverse = (m_enums.find(name) == m_enums.end());
  if (traverse)
  {
    m_enums.insert(name);
    m_enumElement = new TiXmlElement("enum");
    m_enumElement->SetAttribute("name", name.c_str());
  }
  return traverse;
}

void XMLBuilder::enumTypeEnd()
{
  DP_ASSERT( m_enumElement != nullptr );
  m_libraryElement->LinkEndChild( m_enumElement );
  m_enumElement = nullptr;
}

void XMLBuilder::enumTypeValue( std::string const& name, int value )
{
  DP_ASSERT( m_enumElement != nullptr );
  m_enumElement->LinkEndChild( new TiXmlElement( name.c_str() ) );
}

bool XMLBuilder::fieldBegin( std::string const& name )
{
  m_materialElements.back().push( new TiXmlElement( name.c_str() ) );
  if (name == "thin_walled")
  {
    m_currentType = "Bool";
  }
  return( true );
}

void XMLBuilder::fieldEnd()
{
  DP_ASSERT( m_materialElements.back().top()->FirstChildElement() );
  popElement();
}

bool XMLBuilder::fileBegin( std::string const& name )
{
  DP_ASSERT( dp::util::fileExists( name ) );
  m_fileName = name;

  DP_ASSERT( m_materialElements.empty() );

  m_libraryElement = new TiXmlElement( "library" );
  return true;
}

void XMLBuilder::fileEnd()
{
  if ( m_materialElements.empty() )
  {
    delete m_libraryElement;
    m_libraryElement = nullptr;
  }
  else
  {
    for ( size_t i=0 ; i<m_materialElements.size() ; i++ )
    {
      DP_ASSERT( m_materialElements[i].size() == 1 );
      m_libraryElement->LinkEndChild( m_materialElements[i].top() );
    }
    m_materialElements.clear();
  }
}

bool XMLBuilder::materialBegin( std::string const& name, dp::math::Vec4ui const& hash )
{
  TiXmlElement * materialElement = new TiXmlElement( "material" );
  materialElement->SetAttribute( "name", name.c_str() );
  m_materialElements.push_back( std::stack<TiXmlElement*>() );
  m_materialElements.back().push( materialElement );
  return true;
}

void XMLBuilder::materialEnd()
{
  m_enums.clear();
  m_structures.clear();
  m_temporaryToParameterMap.clear();
  DP_ASSERT( m_materialElements.back().size() == 1 );
}

bool XMLBuilder::matrixBegin( std::string const& type )
{
  DP_ASSERT( !m_materialElements.back().empty() );
  DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 );
  m_materialElements.back().push( new TiXmlElement( "matrix" ) );
  m_materialElements.back().top()->SetAttribute( "type", type.c_str() );
  return true;
}

void XMLBuilder::matrixEnd()
{
  DP_ASSERT( !m_materialElements.back().empty() );
  DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "matrix" ) == 0 );
  popElement();
  DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 );
}

bool XMLBuilder::parameterBegin( unsigned int index, std::string const& name )
{
  TiXmlElement * parameterElement = new TiXmlElement( "parameter" );
  parameterElement->SetAttribute( "name", name.c_str() );
  m_materialElements.back().push( parameterElement );
  return true;
}

void XMLBuilder::parameterEnd()
{
  popElement();
}

void XMLBuilder::referenceParameter( unsigned int idx )
{
  if ( strcmp( m_materialElements.back().top()->Value(), "temporary" ) == 0 )
  {
    // if a temporary just references a parameter, we'll remap all usages of that temporary to the referenced parameter
    DP_ASSERT( m_materialElements.back().top()->Attribute( "ID" ) );
    int id = atoi( m_materialElements.back().top()->Attribute( "ID" ) );
    m_temporaryToParameterMap[id] = idx;
  }
  else
  {
    std::ostringstream oss;
    oss << idx;

    TiXmlElement * parameterReferenceElement = new TiXmlElement( "parameter_ref" );
    parameterReferenceElement->SetAttribute( "ID", oss.str().c_str() );
    m_materialElements.back().top()->LinkEndChild( parameterReferenceElement );
  }
}

void XMLBuilder::referenceTemporary( unsigned int idx )
{
  std::map<unsigned int,unsigned int>::const_iterator it = m_temporaryToParameterMap.find( idx );
  if ( it == m_temporaryToParameterMap.end() )
  {
    std::ostringstream oss;
    oss << idx;

    TiXmlElement * temporaryReferenceElement = new TiXmlElement( "temporary_ref" );
    temporaryReferenceElement->SetAttribute( "ID", oss.str().c_str() );
    m_materialElements.back().top()->LinkEndChild( temporaryReferenceElement );
  }
  else
  {
    // remap temporaries trivially mapped to parameters to, well, that parameter
    referenceParameter( it->second );
  }
}

bool XMLBuilder::structureBegin( std::string const& type )
{
  DP_ASSERT( !m_materialElements.back().empty() );
#if !defined(NDEBUG)
  std::string v = m_materialElements.back().top()->Value();
#endif
  if ( strcmp( m_materialElements.back().top()->Value(), "annotation" ) == 0 )
  {
    // ignore structure begin/end on annotations
  }
  else if ( strcmp( m_materialElements.back().top()->Value(), "parameter" ) == 0 )
  {
    m_materialElements.back().push( new TiXmlElement( "parameter_struct" ) );
  }
  else
  {
    DP_ASSERT( ( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 )
            || ( strcmp( m_materialElements.back().top()->Value(), "array" ) == 0 )
            || ( strcmp( m_materialElements.back().top()->Value(), "backface" ) == 0 )
            || ( strcmp( m_materialElements.back().top()->Value(), "struct" ) == 0 )
            || ( strcmp( m_materialElements.back().top()->Value(), "volume" ) == 0 ) );
    m_materialElements.back().push( new TiXmlElement( "struct" ) );
    m_materialElements.back().top()->SetAttribute( "type", type.c_str() );
  }
  return true;
}

void XMLBuilder::structureEnd()
{
  DP_ASSERT( !m_materialElements.back().empty() );
  if ( strcmp( m_materialElements.back().top()->Value(), "annotation" ) == 0 )
  {
    // ignore structure begin/end on annotations
  }
  else if ( strcmp( m_materialElements.back().top()->Value(), "parameter_struct" ) == 0 )
  {
    std::string value = m_materialElements.back().top()->Attribute( "value" );
    m_materialElements.back().pop();
    DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "parameter" ) == 0 );
    m_materialElements.back().top()->SetAttribute( "value", value.c_str() );
  }
  else
  {
    DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "struct" ) == 0 );
    popElement();
    DP_ASSERT( ( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 )
            || ( strcmp( m_materialElements.back().top()->Value(), "array" ) == 0 )
            || ( strcmp( m_materialElements.back().top()->Value(), "backface") == 0 )
            || ( strcmp( m_materialElements.back().top()->Value(), "struct" ) == 0 )
            || ( strcmp( m_materialElements.back().top()->Value(), "volume" ) == 0 ) );
  }
}

bool XMLBuilder::structureTypeBegin( std::string const& name )
{
  bool traverse = (m_structures.find(name) == m_structures.end());
  if (traverse)
  {
    m_structures.insert(name);
    m_structureStack.push(new TiXmlElement("struct"));
    m_structureStack.top()->SetAttribute("name", name.c_str());
  }
  return traverse;
}

void XMLBuilder::structureTypeElement( std::string const& type, std::string const& name )
{
  DP_ASSERT( !m_structureStack.empty() );
  TiXmlElement * element = new TiXmlElement( "field" );
  element->SetAttribute( "type", type.c_str() );
  element->SetAttribute( "name", name.c_str() );
  m_structureStack.top()->LinkEndChild( element );
}

void XMLBuilder::structureTypeEnd()
{
  DP_ASSERT( !m_structureStack.empty() );
  if ( strncmp( m_structureStack.top()->Attribute( "name" ), "::anno::", 8 ) != 0 )   // filter out annotation structure declarations!
  {
    m_libraryElement->LinkEndChild( m_structureStack.top() );
  }
  m_structureStack.pop();
}

bool XMLBuilder::temporaryBegin( unsigned int idx )
{
  std::ostringstream oss;
  oss << idx;

  TiXmlElement * temporaryElement = new TiXmlElement( "temporary" );
  temporaryElement->SetAttribute( "ID", oss.str().c_str() );
  m_materialElements.back().push( temporaryElement );
  return true;
}

void XMLBuilder::temporaryEnd()
{
  if (! m_materialElements.back().top()->FirstChildElement() )
  {
    m_materialElements.back().pop();
  }
  else
  {
    popElement();
  }
}

void addGamma( TiXmlElement * element, dp::fx::mdl::MDLTokenizer::GammaMode gamma )
{
  std::string gammaString;
  switch( gamma )
  {
    case dp::fx::mdl::MDLTokenizer::GammaMode::DEFAULT :
      gammaString = "gamma_default";
      break;
    case dp::fx::mdl::MDLTokenizer::GammaMode::LINEAR :
      gammaString = "gamma_linear";
      break;
    case dp::fx::mdl::MDLTokenizer::GammaMode::SRGB :
      gammaString = "gamma_srgb";
      break;
    default :
      DP_ASSERT( false );
      break;
  }
  element->SetAttribute( "gamma", gammaString.c_str() );
}

void XMLBuilder::valueBool( bool value )
{
  valueString( value ? "true" : "false" );
}

void XMLBuilder::valueBsdfMeasurement( std::string const& value )
{
  valueString( value );
}

void XMLBuilder::valueColor( dp::math::Vec3f const& value )
{
  DP_ASSERT( !m_materialElements.back().empty() );

  std::ostringstream oss;
  oss << value[0] << " " << value[1] << " " << value[2];

  if ( ( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 )
    || ( strcmp( m_materialElements.back().top()->Value(), "array" ) == 0 )
    || ( strcmp( m_materialElements.back().top()->Value(), "ior" ) == 0 )      // field "ior" might be directly specified by a Color
    || ( strcmp( m_materialElements.back().top()->Value(), "struct" ) == 0 ) )
  {
    m_materialElements.back().push( new TiXmlElement( "data" ) );
    m_materialElements.back().top()->SetAttribute( "type", m_currentType.c_str() );
    m_materialElements.back().top()->SetAttribute( "value", oss.str().c_str() );
    popElement();
  }
  else if ( strcmp( m_materialElements.back().top()->Value(), "parameter_struct" ) == 0 )
  {
    std::string value;
    if ( m_materialElements.back().top()->Attribute( "value" ) )
    {
      value = std::string( m_materialElements.back().top()->Attribute( "value" ) ) + " ";
    }
    m_materialElements.back().top()->SetAttribute( "value", ( value + oss.str() ).c_str() );
  }
  else
  {
    DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "parameter" ) == 0 );
    m_materialElements.back().top()->SetAttribute( "value", oss.str().c_str() );
  }
}

void XMLBuilder::valueEnum( std::string const& type, int value, std::string const& name )
{
  DP_ASSERT( !m_materialElements.back().empty() );
  if ( ( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 )
    || ( strcmp( m_materialElements.back().top()->Value(), "array" ) == 0 )
    || ( strcmp( m_materialElements.back().top()->Value(), "struct" ) == 0 ) )
  {
    m_materialElements.back().push( new TiXmlElement( "data" ) );
    m_materialElements.back().top()->SetAttribute( "type", m_currentType.c_str() );
    m_materialElements.back().top()->SetAttribute( "value", name.c_str() );
    popElement();
  }
  else if ( strcmp( m_materialElements.back().top()->Value(), "parameter_struct" ) == 0 )
  {
    std::string value;
    if ( m_materialElements.back().top()->Attribute( "value" ) )
    {
      value = std::string( m_materialElements.back().top()->Attribute( "value" ) ) + " ";
    }
    m_materialElements.back().top()->SetAttribute( "value", ( value + name ).c_str() );
  }
  else
  {
    DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "parameter" ) == 0 );
    m_materialElements.back().top()->SetAttribute( "value", name.c_str() );
  }
}

void XMLBuilder::valueFloat( float value )
{
  std::ostringstream oss;
  oss << value;
  valueString( oss.str() );
}

void XMLBuilder::valueInt( int data )
{
  std::ostringstream oss;
  oss << data;
  valueString( oss.str() );
}

void XMLBuilder::valueLightProfile( std::string const& value )
{
  valueString( value );
}

void XMLBuilder::valueString( std::string const& value )
{
  DP_ASSERT( !m_materialElements.back().empty() );
  std::string dataUsage( m_materialElements.back().top()->Value() );
  if ( dataUsage == "annotation" )
  {
    DP_ASSERT( m_materialElements.back().top()->Attribute( "text" ) );
    std::string extendedText = std::string( m_materialElements.back().top()->Attribute( "text" ) ) + value + ",";
    m_materialElements.back().top()->SetAttribute( "text", extendedText.c_str() );
  }
  else if ( ( dataUsage == "argument" )
         || ( dataUsage == "array" )
         || ( dataUsage == "struct" )
         || ( dataUsage == "vector" )
         || ( ( dataUsage == "thin_walled" ) && ( m_currentType == "Bool" ) ) )  // field "thin_walled" might be directly specified by a boolean
  {
    m_materialElements.back().push( new TiXmlElement( "data" ) );
    m_materialElements.back().top()->SetAttribute( "type", m_currentType.c_str() );
    m_materialElements.back().top()->SetAttribute( "value", value.c_str() );
    popElement();
  }
  else
  {
    DP_ASSERT( dataUsage == "parameter" );
    m_materialElements.back().top()->SetAttribute( "value", value.c_str() );
  }
}

void XMLBuilder::valueTexture( std::string const& name, GammaMode gamma )
{
  DP_ASSERT( !m_materialElements.back().empty() );

  if ( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 )
  {
    m_materialElements.back().push( new TiXmlElement( "data" ) );
    m_materialElements.back().top()->SetAttribute( "type", m_currentType.c_str() );
    m_materialElements.back().top()->SetAttribute( "value", name.c_str() );
    addGamma( m_materialElements.back().top(), gamma );
    popElement();
  }
  else
  {
    DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "parameter" ) == 0 );
    m_materialElements.back().top()->SetAttribute( "value", name.c_str() );
    addGamma( m_materialElements.back().top(), gamma );
  }
}

bool XMLBuilder::vectorBegin( std::string const& type )
{
  DP_ASSERT( !m_materialElements.back().empty() );
  std::string v = m_materialElements.back().top()->Value();
  DP_ASSERT( ( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 )
          || ( strcmp( m_materialElements.back().top()->Value(), "matrix" ) == 0 )
          || ( strcmp( m_materialElements.back().top()->Value(), "parameter" ) == 0 ) );
  m_materialElements.back().push( new TiXmlElement( "vector" ) );
  m_materialElements.back().top()->SetAttribute( "type", type.c_str() );
  return true;
}

void XMLBuilder::vectorEnd()
{
  DP_ASSERT( !m_materialElements.back().empty() );
  DP_ASSERT( strcmp( m_materialElements.back().top()->Value(), "vector" ) == 0 );
  popElement();
  DP_ASSERT( ( strcmp( m_materialElements.back().top()->Value(), "argument" ) == 0 )
          || ( strcmp( m_materialElements.back().top()->Value(), "matrix" ) == 0 )
          || ( strcmp( m_materialElements.back().top()->Value(), "parameter" ) == 0 ) );
}

void XMLBuilder::popElement()
{
  TiXmlElement * element = m_materialElements.back().top();
  m_materialElements.back().pop();
  m_materialElements.back().top()->LinkEndChild( element );
}
