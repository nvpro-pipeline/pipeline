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


#include <test/testfw/manager/Manager.h>
#include "feature_transform_attribute.h"

#include <dp/gl/RenderTarget.h>
#include <dp/math/Trafo.h>
#include <dp/util/Array.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>

#include <limits>

using namespace dp;
using namespace rix::core;

//Automatically add the test to the module's global test list
REGISTER_TEST("feature_transform_attribute", "tests RiX texture sampler support", create_feature_transform_attribute);


Feature_transform_attribute::Feature_transform_attribute()
  : m_textureHandle(nullptr)
  , m_vertexContainerDescriptor(nullptr)
  , m_fragmentContainerDescriptor(nullptr)
  , m_renderData(nullptr)
  , m_rix(nullptr)
{
}

Feature_transform_attribute::~Feature_transform_attribute()
{
}

bool Feature_transform_attribute::onInit()
{
  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = static_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  m_displayTarget.inplaceCast<dp::gl::RenderTarget>()->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );

  m_renderData = new test::framework::RenderDataRiX;

  glEnable(GL_DEPTH_TEST);

  createScene();

  return true;
}

void Feature_transform_attribute::createScene()
{

  const size_t nc  = 4;             // number of components (rgba)
  const size_t bpc = sizeof(float); // bytes per component

  const size_t tex2DWidth     = 8;
  const size_t tex2DHeight    = 8;
  const size_t tex2DArraySize = tex2DWidth * tex2DHeight * nc;
  const size_t tex2DSize      = tex2DArraySize * bpc;
  float        tex2D[tex2DArraySize];
  for( unsigned int i = 0; i < tex2DWidth; ++i )
  {
    for( unsigned int j = 0; j < tex2DWidth; ++j )
    {
      unsigned int pos = (i * tex2DWidth + j) * bpc;
      tex2D[  pos] = 1.0f *  (( i ^ j ) & 1);
      tex2D[1+pos] = 1.0f * ((( i ^ j ) & 2) / 2);
      tex2D[2+pos] = 1.0f * ((( i ^ j ) & 4) / 4);
      tex2D[3+pos] = 1.0f;
    }
  }

  const char * vertexShader = "//vertex shader\n"
    "#version 330\n"
    "layout(location=0) in vec3 Position;\n"
    "layout(location=8) in vec2 texCoord0;\n"
    "uniform mat4 world2view;\n"
    "uniform mat4 model2world;\n"
    "out vec2 vTexCoords;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = world2view * model2world * vec4( Position, 1.0 );\n"
    "  vTexCoords = texCoord0;\n"
    "}\n";

  const char * fragmentShader = "//fragment shader\n"
    "#version 330\n"
    "uniform sampler2D tex;\n"
    "in vec2 vTexCoords;\n"
    "uniform vec4 color;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = color + texture( tex, vTexCoords.xy );\n"
    "}\n";

  dp::rix::util::GeometryDataSharedPtr mesh[4];
  mesh[0] = dp::rix::util::createQuad( dp::rix::util::AttributeID::POSITION | dp::rix::util::AttributeID::TEXCOORD0
                                     , math::Vec3f(-0.125f*m_height, -0.125f*m_height, 0.0f)
                                     , math::Vec3f(0.125f*m_height, -0.125f*m_height, 0.0f)
                                     , math::Vec3f(-0.125f*m_height, 0.125f*m_height, 0.0f)
                                     , math::Vec2f(0.25f, 0.25f), math::Vec2f(0.75f, 0.25f)
                                     , math::Vec2f(0.25f, 0.75f) );

  float angle = math::PI_QUARTER;

  math::Mat33f matTex;
  matTex[0] = math::Vec3f(cosf(angle),  sinf(angle), -0.5f*cosf(angle)-0.5f*sinf(angle)+0.5f);
  matTex[1] = math::Vec3f(-sinf(angle), cosf(angle), -0.5f*cosf(angle)+0.5f*sinf(angle)+0.5f);
  matTex[2] = math::Vec3f(0.0f,         0.0f,         1.0f);

  math::Mat33f matPos;
  matPos[0] = math::Vec3f(cosf(angle),  sinf(angle), 0.0f);
  matPos[1] = math::Vec3f(-sinf(angle), cosf(angle), 0.0f);
  matPos[2] = math::Vec3f(0.0f,         0.0f,        1.0f);

  mesh[1] = transformAttribute(matPos, dp::rix::util::AttributeID::POSITION, mesh[0]);
  mesh[2] = transformAttribute(matTex, dp::rix::util::AttributeID::TEXCOORD0, mesh[0]);
  mesh[3] = transformAttribute(matTex, dp::rix::util::AttributeID::TEXCOORD0, mesh[0]);
  mesh[3] = transformAttribute(matPos, dp::rix::util::AttributeID::POSITION, mesh[0], true, mesh[3]);

  GeometrySharedHandle geometry[4];
  geometry[0] = dp::rix::util::generateGeometry(mesh[0], m_rix);
  geometry[1] = dp::rix::util::generateGeometry(mesh[1], m_rix);
  geometry[2] = dp::rix::util::generateGeometry(mesh[2], m_rix);
  geometry[3] = dp::rix::util::generateGeometry(mesh[3], m_rix);

  std::vector<ProgramParameter> vertexProgramParameters;
  vertexProgramParameters.push_back( ProgramParameter("model2world", ContainerParameterType::MAT4X4) );
  vertexProgramParameters.push_back( ProgramParameter("world2view", ContainerParameterType::MAT4X4) );

  std::vector<ProgramParameter> fragmentProgramParameters;
  fragmentProgramParameters.push_back( ProgramParameter("tex", ContainerParameterType::SAMPLER) );
  fragmentProgramParameters.push_back( ProgramParameter("color", ContainerParameterType::FLOAT4) );

  m_vertexContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &vertexProgramParameters[0],
    dp::checked_cast<unsigned int>(vertexProgramParameters.size()) ) );

  m_fragmentContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &fragmentProgramParameters[0],
    dp::checked_cast<unsigned int>(fragmentProgramParameters.size()) ) );

  m_containerEntryModel2world = m_rix->containerDescriptorGetEntry( m_vertexContainerDescriptor, "model2world" );
  m_containerEntryWorld2view  = m_rix->containerDescriptorGetEntry( m_vertexContainerDescriptor, "world2view" );

  m_containerEntryBuffer = m_rix->containerDescriptorGetEntry( m_fragmentContainerDescriptor, "tex" );
  m_containerEntryColor  = m_rix->containerDescriptorGetEntry( m_fragmentContainerDescriptor, "color" );

  const char* shaders[] = {vertexShader, fragmentShader};
  ShaderType  shaderTypes[] = { ShaderType::VERTEX_SHADER, ShaderType::FRAGMENT_SHADER };
  ProgramShaderCode programShaderCode( sizeof dp::util::array( shaders ) , shaders, shaderTypes );

  std::vector<ContainerDescriptorSharedHandle> containerDescriptors;
  containerDescriptors.push_back( m_vertexContainerDescriptor );
  containerDescriptors.push_back( m_fragmentContainerDescriptor );

  ProgramDescription programDescription( programShaderCode, &containerDescriptors[0], dp::checked_cast<unsigned int>(containerDescriptors.size()));
  m_programSampler = m_rix->programCreate( programDescription );

  // prepare & set texture
  TextureDescription textureDescription( TextureType::_2D, InternalTextureFormat::RGBA32F, dp::PixelFormat::RGBA, dp::DataType::FLOAT_32, tex2DWidth, tex2DHeight, 0, 0, true );
  m_textureHandle = m_rix->textureCreate( textureDescription );

  TextureDataPtr textureDataPtr( tex2D, dp::PixelFormat::RGBA, dp::DataType::FLOAT_32 );
  m_rix->textureSetData( m_textureHandle, textureDataPtr );

  RenderGroupSharedHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);

  generateGI( geometry[0], 0.25f*m_width, 0.75f*m_height, 0.0f );
  generateGI( geometry[1], 0.75f*m_width, 0.75f*m_height, 0.0f );
  generateGI( geometry[2], 0.25f*m_width, 0.25f*m_height, 0.0f );
  generateGI( geometry[3], 0.75f*m_width, 0.25f*m_height, 0.0f );

}

bool Feature_transform_attribute::onRun( unsigned int idx )
{
  render(m_renderData, m_displayTarget);

  return true;
}

bool Feature_transform_attribute::onClear()
{
  delete m_renderData;

  return true;
}

void Feature_transform_attribute::generateGI( GeometrySharedHandle geometry, const float transX, const float transY, const float transZ )
{
  // TODO: here there should be more vertex containers: one for the w2v and several for the m2w matrices
  ContainerSharedHandle fragmentContainer = m_rix->containerCreate( m_fragmentContainerDescriptor );
  ContainerSharedHandle vertexContainer   = m_rix->containerCreate( m_vertexContainerDescriptor );

  ProgramSharedHandle programs[] = { m_programSampler };
  ProgramPipelineSharedHandle programPipeline = m_rix->programPipelineCreate( programs, sizeof util::array( programs ) );

  GeometryInstanceSharedHandle geometryInstance = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance, geometry );
  m_rix->geometryInstanceSetProgramPipeline(  geometryInstance, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance, vertexContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance, fragmentContainer );

  m_rix->renderGroupAddGeometryInstance( m_renderData->getRenderGroup(), geometryInstance );

  // prepare & set parameters
  math::Mat44f ortho = math::makeOrtho<float>( 0.0f, 1.0f*m_width, 0.0f, 1.0f*m_height, -1.0f, 1.0f );
  m_rix->containerSetData( vertexContainer, m_containerEntryWorld2view, ContainerDataRaw( 0, ortho.getPtr(), 16*sizeof(float) ) );


  math::Trafo trafo;
  trafo.setTranslation( math::Vec3f(transX, transY, transZ) );

  m_rix->containerSetData( vertexContainer, m_containerEntryModel2world, ContainerDataRaw( 0, trafo.getMatrix().getPtr(), 16*sizeof(float) ) );

  SamplerStateDataCommon samplerStateDataCommon( SamplerStateFilterMode::NEAREST, SamplerStateFilterMode::NEAREST );
  SamplerStateSharedHandle samplerStateHandle = m_rix->samplerStateCreate(samplerStateDataCommon);

  rix::core::SamplerSharedHandle sampler = m_rix->samplerCreate();
  m_rix->samplerSetTexture( sampler, m_textureHandle );
  m_rix->samplerSetSamplerState( sampler, samplerStateHandle );
  m_rix->containerSetData( fragmentContainer, m_containerEntryBuffer, ContainerDataSampler( sampler ) );

  float color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
  m_rix->containerSetData( fragmentContainer, m_containerEntryColor, ContainerDataRaw( 0, color, 4*sizeof(float) ) );
}
