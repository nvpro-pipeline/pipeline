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


#include <test/testfw/manager/Manager.h>
#include "feature_texture_sampler.h"

#include <dp/gl/RenderTarget.h>
#include <dp/math/Trafo.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>

#include <limits>

using namespace dp;
using namespace rix::core;

//Automatically add the test to the module's global test list
REGISTER_TEST("feature_texture_sampler", "tests RiX texture sampler support", create_feature_texture_sampler);


Feature_texture_sampler::Feature_texture_sampler()
  : m_textureHandle(nullptr)
  , m_renderData(nullptr)
  , m_rix(nullptr)
{
}

Feature_texture_sampler::~Feature_texture_sampler()
{
}

bool Feature_texture_sampler::onInit()
{
  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = static_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  util::smart_cast<dp::gl::RenderTarget>( m_displayTarget )->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );

  m_renderData = new test::framework::RenderDataRiX;

  createScene();

  return true;
}

void Feature_texture_sampler::createScene()
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

  dp::util::generator::SmartGeometryData mesh = dp::util::generator::createRectangle( dp::util::generator::ATTRIB_POSITION | dp::util::generator::ATTRIB_TEXCOORD0, 0.0f, 0.25f*m_height, 0.25f*m_width, 0.0f );
  m_geometry = dp::rix::util::generateGeometry(mesh, m_rix);


  std::vector<ProgramParameter> vertexProgramParameters;
  vertexProgramParameters.push_back( ProgramParameter("model2world", CPT_MAT4X4) );
  vertexProgramParameters.push_back( ProgramParameter("world2view", CPT_MAT4X4) );

  std::vector<ProgramParameter> fragmentProgramParameters;
  fragmentProgramParameters.push_back( ProgramParameter("tex", CPT_SAMPLER) );
  fragmentProgramParameters.push_back( ProgramParameter("color", CPT_FLOAT4) );

  m_vertexContainerDescriptor = 
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &vertexProgramParameters[0],
    util::checked_cast<unsigned int>(vertexProgramParameters.size()) ) );

  m_fragmentContainerDescriptor = 
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &fragmentProgramParameters[0],
    util::checked_cast<unsigned int>(fragmentProgramParameters.size()) ) );

  m_containerEntryModel2world = m_rix->containerDescriptorGetEntry( m_vertexContainerDescriptor, "model2world" );
  m_containerEntryWorld2view  = m_rix->containerDescriptorGetEntry( m_vertexContainerDescriptor, "world2view" );

  m_containerEntryBuffer = m_rix->containerDescriptorGetEntry( m_fragmentContainerDescriptor, "tex" );
  m_containerEntryColor  = m_rix->containerDescriptorGetEntry( m_fragmentContainerDescriptor, "color" );

  const char* shaders[] = {vertexShader, fragmentShader};
  ShaderType  shaderTypes[] = { ST_VERTEX_SHADER, ST_FRAGMENT_SHADER };
  ProgramShaderCode programShaderCode( sizeof util::array( shaders ) , shaders, shaderTypes );

  std::vector<ContainerDescriptorSharedHandle> containerDescriptors;
  containerDescriptors.push_back( m_vertexContainerDescriptor );
  containerDescriptors.push_back( m_fragmentContainerDescriptor );


  ProgramDescription programDescription( programShaderCode, &containerDescriptors[0], util::checked_cast<unsigned int>(containerDescriptors.size()));
  m_programSampler = m_rix->programCreate( programDescription );


  // prepare & set texture
  TextureDescription textureDescription( TT_2D, ITF_RGBA32F, dp::util::PF_RGBA, dp::util::DT_FLOAT_32, tex2DWidth, tex2DHeight, 0, 0, true );
  m_textureHandle = m_rix->textureCreate( textureDescription );

  TextureDataPtr textureDataPtr( tex2D, dp::util::PF_RGBA, dp::util::DT_FLOAT_32 );
  m_rix->textureSetData( m_textureHandle, textureDataPtr );

  RenderGroupSharedHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);

  generateGI( SSFM_NEAREST, SSFM_NEAREST, 0.9f, -0.0f, 0.75f*m_height, 0.0f );
  generateGI( SSFM_LINEAR, SSFM_LINEAR,   0.9f, 0.25f*m_width, 0.75f*m_height, 0.0f );
}

bool Feature_texture_sampler::onRun( unsigned int idx )
{
  render(m_renderData, m_displayTarget);

  return true;  
}

bool Feature_texture_sampler::onClear()
{
  delete m_renderData;

  return true;
}

void Feature_texture_sampler::generateGI( SamplerStateFilterMode minFilterMode, SamplerStateFilterMode magFilterMode, const float scale, const float transX, const float transY, const float transZ )
{
  // TODO: here there should be more vertex containers: one for the w2v and several for the m2w matrices
  ContainerSharedHandle fragmentContainer = m_rix->containerCreate( m_fragmentContainerDescriptor );
  ContainerSharedHandle vertexContainer   = m_rix->containerCreate( m_vertexContainerDescriptor );

  ProgramSharedHandle programs[] = { m_programSampler };
  ProgramPipelineSharedHandle programPipeline = m_rix->programPipelineCreate( programs, sizeof util::array( programs ) );

  GeometryInstanceSharedHandle geometryInstance = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance, m_geometry );
  m_rix->geometryInstanceSetProgramPipeline(  geometryInstance, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance, vertexContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance, fragmentContainer );

  m_rix->renderGroupAddGeometryInstance( m_renderData->getRenderGroup(), geometryInstance );

  // prepare & set parameters
  math::Mat44f ortho = math::makeOrtho<float>( 0.0f, 1.0f*m_width, 0.0f, 1.0f*m_height, -1.0f, 1.0f );
  m_rix->containerSetData( vertexContainer, m_containerEntryWorld2view, ContainerDataRaw( 0, ortho.getPtr(), 16*sizeof(float) ) );


  math::Trafo trafo;
  trafo.setScaling( math::Vec3f(scale, scale, scale) );
  trafo.setTranslation( math::Vec3f(transX, transY, transZ) );

  m_rix->containerSetData( vertexContainer, m_containerEntryModel2world, ContainerDataRaw( 0, trafo.getMatrix().getPtr(), 16*sizeof(float) ) );

  SamplerStateDataCommon samplerStateDataCommon( minFilterMode, magFilterMode );
  SamplerStateSharedHandle samplerStateHandle = m_rix->samplerStateCreate(samplerStateDataCommon);

  rix::core::SamplerSharedHandle sampler = m_rix->samplerCreate();
  m_rix->samplerSetTexture( sampler, m_textureHandle );
  m_rix->samplerSetSamplerState( sampler, samplerStateHandle );
  m_rix->containerSetData( fragmentContainer, m_containerEntryBuffer, ContainerDataSampler( sampler ) );

  float color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
  m_rix->containerSetData( fragmentContainer, m_containerEntryColor, ContainerDataRaw( 0, color, 4*sizeof(float) ) );
}
