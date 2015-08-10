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
#include "feature_texture_buffer.h"

#include <dp/gl/RenderTarget.h>
#include <dp/math/Trafo.h>
#include <dp/util/SharedPtr.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>

#include <limits>

using namespace dp;
using namespace rix::core;

//Automatically add the test to the module's global test list
REGISTER_TEST("feature_texture_buffer", "tests RiX texture buffer support", create_feature_texture_buffer);

Feature_texture_buffer::Feature_texture_buffer()
  : m_renderData(nullptr)
  , m_rix(nullptr)
{
}

Feature_texture_buffer::~Feature_texture_buffer()
{
}

bool Feature_texture_buffer::onInit()
{
  DP_ASSERT( dynamic_cast<rix::core::test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = static_cast<rix::core::test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  dp::util::shared_cast<dp::gl::RenderTarget>( m_displayTarget )->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );
  
  m_renderData = new rix::core::test::framework::RenderDataRiX;
  
  createScene();

  return true;  
}

void Feature_texture_buffer::createScene()
{
  const size_t nc = 4;              // number of components (rgba)
  const size_t bpc = sizeof(float); // bytes per component

  const size_t tex1DWidth = 8;                   // texture width
  const size_t tex1DArraySize = tex1DWidth * nc; // texture array size
  const size_t tex1DSize = tex1DArraySize * bpc; // texture size in bytes

  float tex1D[tex1DWidth * nc];
  for( unsigned int i=0; i < tex1DWidth; ++i )
  {
    unsigned int pos = i * bpc;
    tex1D[  pos] = 1.0f * (i & 1) / 1;
    tex1D[1+pos] = 1.0f * (i & 2) / 2;
    tex1D[2+pos] = 1.0f * (i & 4) / 4;
    tex1D[3+pos] = 1.0f;
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

  const char * fragmentShader = "//samplerBuffer\n"
    "#version 330\n"
    "uniform samplerBuffer buf;\n"
    "in vec2 vTexCoords;\n"
    "uniform vec4 color;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = color + texelFetch( buf, int(textureSize(buf)*vTexCoords.x));\n"
    "}\n";

  //
  // prepare render API objects
  //

  dp::rix::util::GeometryDataSharedPtr mesh = dp::rix::util::createRectangle( dp::rix::util::ATTRIB_POSITION | dp::rix::util::ATTRIB_TEXCOORD0, 0.0f, 0.5f*m_height, 0.5f*m_width, 0.0f );
  GeometrySharedHandle geometry = dp::rix::util::generateGeometry(mesh, m_rix);

  std::vector<ProgramParameter> vertexProgramParameters;
  vertexProgramParameters.push_back( ProgramParameter("model2world", CPT_MAT4X4) );
  vertexProgramParameters.push_back( ProgramParameter("world2view", CPT_MAT4X4) );

  std::vector<ProgramParameter> fragmentProgramParameters;
  fragmentProgramParameters.push_back( ProgramParameter("buf", CPT_SAMPLER) );
  fragmentProgramParameters.push_back( ProgramParameter("color", CPT_FLOAT4) );

  ContainerDescriptorSharedHandle vertexContainerDescriptor = 
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &vertexProgramParameters[0],
    dp::checked_cast<unsigned int>(vertexProgramParameters.size()) ) );

  ContainerDescriptorSharedHandle fragmentContainerDescriptor = 
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &fragmentProgramParameters[0],
    dp::checked_cast<unsigned int>(fragmentProgramParameters.size()) ) );

  ContainerEntry containerEntryModel2world = m_rix->containerDescriptorGetEntry( vertexContainerDescriptor, "model2world" );
  ContainerEntry containerEntryWorld2view  = m_rix->containerDescriptorGetEntry( vertexContainerDescriptor, "world2view" );

  ContainerEntry containerEntryBuffer = m_rix->containerDescriptorGetEntry( fragmentContainerDescriptor, "buf" );
  ContainerEntry containerEntryColor = m_rix->containerDescriptorGetEntry( fragmentContainerDescriptor, "color" );

  std::vector<ContainerDescriptorSharedHandle> containerDescriptors;
  containerDescriptors.push_back( vertexContainerDescriptor );
  containerDescriptors.push_back( fragmentContainerDescriptor );

  const char* shaders[] = {vertexShader, fragmentShader};
  ShaderType  shaderTypes[] = { ST_VERTEX_SHADER, ST_FRAGMENT_SHADER };
  ProgramShaderCode programShaderCode( sizeof util::array( shaders ), shaders, shaderTypes );

  ProgramDescription programDescription( programShaderCode, &containerDescriptors[0], dp::checked_cast<unsigned int>(containerDescriptors.size() ));


  // here we need a current context
  {
    ProgramSharedHandle programSampler = m_rix->programCreate( programDescription );

    // TODO: here there should be more vertex containers: one for the w2v and several for the m2w matrices
    ContainerSharedHandle vertexContainer   = m_rix->containerCreate( vertexContainerDescriptor );
    ContainerSharedHandle fragmentContainer = m_rix->containerCreate( fragmentContainerDescriptor );

    ProgramSharedHandle programs[] = { programSampler };
    ProgramPipelineSharedHandle programPipeline = m_rix->programPipelineCreate( programs, sizeof util::array( programs ) );

    GeometryInstanceSharedHandle geometryInstance = m_rix->geometryInstanceCreate();
    m_rix->geometryInstanceSetGeometry(  geometryInstance, geometry );
    m_rix->geometryInstanceSetProgramPipeline( geometryInstance, programPipeline );
    m_rix->geometryInstanceUseContainer( geometryInstance, vertexContainer );
    m_rix->geometryInstanceUseContainer( geometryInstance, fragmentContainer );

    RenderGroupSharedHandle renderGroup = m_rix->renderGroupCreate();
    m_renderData->setRenderGroup(renderGroup);
    m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance );

    // prepare & set parameters

    math::Mat44f ortho = math::makeOrtho<float>( 0.0f, 1.0f*m_width, 0.0f, 1.0f*m_height, -1.0f, 1.0f );
    m_rix->containerSetData( vertexContainer, containerEntryWorld2view, ContainerDataRaw( 0, ortho.getPtr(), 16*sizeof(float) ) );

    math::Trafo trafo;
    trafo.setScaling( math::Vec3f(1.8f, 1.8f, 1.0f) );
    trafo.setTranslation( math::Vec3f(0.05f*m_width, 0.05f*m_height, 0.0f) );
    m_rix->containerSetData( vertexContainer, containerEntryModel2world, ContainerDataRaw( 0, trafo.getMatrix().getPtr(), 16*sizeof(float) ) );

    // prepare & set texture
    TextureDescription textureDescription( TT_BUFFER, ITF_RGBA32F, dp::PF_RGBA, dp::DT_FLOAT_32 );

    BufferSharedHandle textureBuffer = m_rix->bufferCreate();
    m_rix->bufferSetSize( textureBuffer, tex1DSize );
    m_rix->bufferUpdateData( textureBuffer, 0, tex1D, tex1DSize );

    TextureSharedHandle texture = m_rix->textureCreate( textureDescription );

    TextureDataBuffer textureData( textureBuffer );

    m_rix->textureSetData( texture, textureData );

    rix::core::SamplerSharedHandle sampler = m_rix->samplerCreate();
    m_rix->samplerSetTexture( sampler, texture );
    m_rix->containerSetData( fragmentContainer, containerEntryBuffer, ContainerDataSampler( sampler ) );

    float color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    m_rix->containerSetData( fragmentContainer, containerEntryColor, ContainerDataRaw( 0, color, 4*sizeof(float) ) );
  }
}

bool Feature_texture_buffer::onRun( unsigned int idx )
{
  render(m_renderData, m_displayTarget);

  return true;  
}

bool Feature_texture_buffer::onClear()
{
  delete m_renderData;

  return true;
}
