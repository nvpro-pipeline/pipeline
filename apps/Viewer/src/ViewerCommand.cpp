// Copyright NVIDIA Corporation 2009-2010
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


#include "Viewer.h"
#include "ViewerCommand.h"

#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/ParameterGroupData.h>

using namespace dp::sg::core;

ViewerCommand::ViewerCommand( unsigned int updateFlags, bool parameterCommand, int id )
  : m_updateFlags( updateFlags )
  , m_parameterCommand( parameterCommand )
  , m_id( id )
{
}

ViewerCommand::~ViewerCommand() 
{
}

int ViewerCommand::id() const 
{ 
  return (m_id >= -1) ? m_id : -m_id; 
}

bool ViewerCommand::isParameterCommand() const 
{ 
  return m_parameterCommand; 
}

void ViewerCommand::undo()
{
  if( doUndo() )
  {
    update();
  }
}

void ViewerCommand::redo()
{
  if( doRedo() )
  {
    update();
  }
}

void ViewerCommand::update()
{
  if( m_updateFlags & UPDATE_MATERIAL )
  { 
    GetApp()->emitMaterialChanged();
  }

  if ( m_updateFlags & UPDATE_SCENE_TREE )
  {
    GetApp()->emitSceneTreeChanged();
  }
}

CommandReplaceEffect::CommandReplaceEffect( const dp::sg::core::GeoNodeSharedPtr & geoNode, const dp::sg::core::EffectDataSharedPtr & newEffect )
  : ViewerCommand( ViewerCommand::UPDATE_ITEMMODELS | ViewerCommand::UPDATE_MATERIAL )
  , m_geoNode( geoNode )
  , m_newEffect( newEffect )
{
  m_oldEffect = m_geoNode->getMaterialEffect();

  // special handling for enum "scatter_mode": set transparent flag in EffectData
  // when the EffectSpec says, it's transparent, it might in fact be opaque, if there is a parameter "scatter_mode" with value "scatter_reflect".
  {
    const dp::fx::EffectSpecSharedPtr & es = m_newEffect->getEffectSpec();

    if ( es->getTransparent() )
    {
      bool found = false;
      for ( dp::fx::EffectSpec::iterator esit = es->beginParameterGroupSpecs() ; !found && ( esit != es->endParameterGroupSpecs() ) ; ++esit )
      {
        for ( dp::fx::ParameterGroupSpec::iterator pgsit = (*esit)->beginParameterSpecs() ; pgsit != (*esit)->endParameterSpecs() ; ++pgsit )
        {
          if ( pgsit->first.getEnumSpec() && ( pgsit->first.getEnumSpec()->getType() == "scatter_mode" ) )
          {
            DP_ASSERT( ( pgsit->first.getEnumSpec()->getValueCount() == 3 )
                    && ( pgsit->first.getEnumSpec()->getValueName( 0 ) == "scatter_reflect" )
                    && ( pgsit->first.getEnumSpec()->getValueName( 1 ) == "scatter_transmit" )
                    && ( pgsit->first.getEnumSpec()->getValueName( 2 ) == "scatter_reflect_transmit" ) );
            found = true;
            if ( 0 == m_newEffect->getParameterGroupData( esit )->getParameter<int>( pgsit ) )   // is it opaque ?
            {
              m_newEffect->setTransparent( false );
              break;
            }
          }
        }
      }
    }
  }

  setText( "Replace " + QString( m_oldEffect ? m_oldEffect->getName().c_str() : "NULL" )
         + " with "   + QString( m_newEffect->getName().c_str() ) );
}

CommandReplaceEffect::~CommandReplaceEffect()
{
  // nothing for now
}

bool CommandReplaceEffect::doUndo()
{
  m_geoNode->setMaterialEffect( m_oldEffect );

  GetApp()->emitSceneTreeChanged();
  return true;
}

bool CommandReplaceEffect::doRedo()
{
  m_geoNode->setMaterialEffect( m_newEffect );

  GetApp()->emitSceneTreeChanged();
  return true;
}

CommandGenerateTangentSpace::CommandGenerateTangentSpace( const dp::sg::core::PrimitiveSharedPtr & primitive )
  : ViewerCommand( ViewerCommand::UPDATE_MATERIAL )
  , m_primitive(primitive)
{
  setText( QString( "Generate Tangent Space" ) );
}

CommandGenerateTangentSpace::~CommandGenerateTangentSpace()
{
}

bool CommandGenerateTangentSpace::doUndo()
{
  // reset the tangent space texture coordinates again
  m_primitive->getVertexAttributeSet()->setVertexAttribute( VertexAttributeSet::DP_SG_TEXCOORD6, dp::sg::core::VertexAttribute(), false );
  m_primitive->getVertexAttributeSet()->setVertexAttribute( VertexAttributeSet::DP_SG_TEXCOORD7, dp::sg::core::VertexAttribute(), false );
  return( true );
}

bool CommandGenerateTangentSpace::doRedo()
{
  // create the texture coordinates
  m_primitive->generateTangentSpace();
  return( true );
}

CommandGenerateTextureCoordinates::CommandGenerateTextureCoordinates( const dp::sg::core::PrimitiveSharedPtr & primitive, dp::sg::core::TextureCoordType tct )
  : ViewerCommand( ViewerCommand::UPDATE_MATERIAL )
  , m_primitive(primitive)
  , m_tct(tct)
{
  QString type;
  switch( tct )
  {
    case TCT_CYLINDRICAL :
      type = QString( "Cylindrical" );
      break;
    case TCT_PLANAR :
      type = QString( "Planar" );
      break;
    case TCT_SPHERICAL :
      type = QString( "Spherical" );
      break;
    default :
      DP_ASSERT( false );
      type = QString( "Unknown" );
      break;
  }
  setText( QString( "Generate " ) + type + QString( " Texture Coordinates" ) );
}

CommandGenerateTextureCoordinates::~CommandGenerateTextureCoordinates()
{
}

bool CommandGenerateTextureCoordinates::doUndo()
{
  // reset the texture coordinates again
  m_primitive->getVertexAttributeSet()->setVertexAttribute( VertexAttributeSet::DP_SG_TEXCOORD0, dp::sg::core::VertexAttribute(), false );
  return( true );
}

bool CommandGenerateTextureCoordinates::doRedo()
{
  // create the texture coordinates
  m_primitive->generateTexCoords( m_tct );
  return( true );
}
