// Copyright (c) 2011-2015, NVIDIA CORPORATION. All rights reserved.
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
#include "benchmark_vertexThroughput.h"

#include <dp/util/File.h>
#include <dp/math/math.h>
#include <dp/util/Array.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>

#include <limits>

using namespace dp;
using namespace testfw;
using namespace core;
using namespace util;
using namespace rix::util;
using namespace rix::core;

//Automatically add the test to the module's global test list
REGISTER_TEST("benchmark_vertexThroughput", "tests performance with varying vertex count", create_benchmark_vertexThroughput);

const float identity[16] =
{
  1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 1.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 1.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 1.0f
};

static void getOrthoProjection(float mat[16],
                        const float left,   const float right,
                        const float bottom, const float top,
                        const float front,  const float back)
{
  for( size_t i=0; i<16; ++i )
  {
    mat[i] = identity[i];
  }
  mat[0] = 2.0f/(right-left);
  mat[5] = 2.0f/(top-bottom);
  mat[10]= -2.0f/(back-front);
  mat[12]= -(right + left)/(right-left);
  mat[13]= -(top + bottom)/(top-bottom);
  mat[14]= -(back + front)/(back - front);
}

static void getTransformMatrix(float mat[16],
  const float scaleX, const float scaleY, const float scaleZ,
  const float transX, const float transY, const float transZ )
{
  for( size_t i=0; i<16; ++i )
  {
    mat[i] = identity[i];
  }

  mat[ 0] = scaleX;
  mat[ 5] = scaleY;
  mat[10] = scaleZ;
  mat[12] = transX;
  mat[13] = transY;
  mat[14] = transZ;
}

Benchmark_vertexThroughput::Benchmark_vertexThroughput()
  : m_renderData(nullptr)
  , m_rix(nullptr)
  , m_vertexShader(nullptr)
  , m_fragmentShaderSampler2D(nullptr)
  , m_containerDescriptorWorld2View(nullptr)
  , m_vertexContainerW2V(nullptr)
  , m_geometryInstance(nullptr)
{
}

Benchmark_vertexThroughput::~Benchmark_vertexThroughput()
{
}

bool Benchmark_vertexThroughput::onInit()
{
  m_renderData = new test::framework::RenderDataRiX;
  m_rix = dynamic_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();

  glEnable(GL_DEPTH_TEST);
  createScene();

  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) )
  m_displayTarget.inplaceCast<dp::gl::RenderTarget>()->setClearColor( 0.46f, 0.72f, 0.0f, 0.0f );

  return true;
}

bool Benchmark_vertexThroughput::onRunInit( unsigned int i )
{

  GeometryDataSharedPtr geometryData = rix::util::createSphere( ATTRIB_POSITION | ATTRIB_NORMAL | ATTRIB_TEXCOORD0 | ATTRIB_TANGENT | ATTRIB_BINORMAL, getSubdivs(i), getSubdivs(i) );
  GeometrySharedHandle geometry = rix::util::generateGeometry(geometryData, m_rix);
  m_rix->geometryInstanceSetGeometry(  m_geometryInstance, geometry );

  return true;
}


bool Benchmark_vertexThroughput::onRun(unsigned int i)
{
  render(m_renderData, m_displayTarget);

  return true;
}

bool Benchmark_vertexThroughput::onRunCheck( unsigned int i )
{
  return i < 50;
}

bool Benchmark_vertexThroughput::onClear()
{
  delete m_renderData;

  return true;
}

void Benchmark_vertexThroughput::createScene()
{

  m_vertexShader = "//vertex shader\n"
    "#version 330\n"
    "layout(location=0) in vec3 Position;\n"
    "uniform mat4 world2view;\n"
    "uniform mat4 model2world;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = world2view * model2world * vec4( Position, 1.0 );\n"
    "}\n";

  m_fragmentShaderSampler2D = "//sampler2D\n"
    "#version 330\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    " Color = vec4(0.0, 0.0, 0.0, 0.0);\n"
    "}\n";

  std::vector<ProgramParameter> programParameterWorld2View;
  programParameterWorld2View.push_back( ProgramParameter("world2view", CPT_MAT4X4) );

  m_containerDescriptorWorld2View = m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &programParameterWorld2View[0], 1 ) );

  dp::rix::core::ContainerEntry containerEntryWorld2View  = m_rix->containerDescriptorGetEntry( m_containerDescriptorWorld2View, "world2view" );

  m_vertexContainerW2V = m_rix->containerCreate( m_containerDescriptorWorld2View );

  RenderGroupSharedHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);

  // prepare & set world2view matrix
  float ortho[16];
  getOrthoProjection( ortho, -2.0f, 2.0f, -2.0f, 2.0f, -2.0f, 2.0f );
  m_rix->containerSetData( m_vertexContainerW2V, containerEntryWorld2View, ContainerDataRaw( 0, ortho, 16*sizeof(float) ) );

  generateGeometry( m_vertexShader, m_fragmentShaderSampler2D, 1.0f, 0.0f, 0.0f, 0.0f );

}

void Benchmark_vertexThroughput::generateGeometry(const char * vertexShader,
                                                  const char * fragmentShader,
                                                  const float scale,
                                                  const float transX,
                                                  const float transY,
                                                  const float transZ)
{
  std::vector<ProgramParameter> vertexProgramParameters;
  vertexProgramParameters.push_back( ProgramParameter("model2world", CPT_MAT4X4) );

  ContainerDescriptorSharedHandle vertexContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &vertexProgramParameters[0],
    dp::checked_cast<unsigned int>(vertexProgramParameters.size()) ) );

  ContainerEntry containerEntryModel2World = m_rix->containerDescriptorGetEntry( vertexContainerDescriptor, "model2world" );


  std::vector<ContainerDescriptorSharedHandle> containerDescriptors;
  containerDescriptors.push_back( m_containerDescriptorWorld2View );
  containerDescriptors.push_back( vertexContainerDescriptor );

  const char* shaders[] = {vertexShader, fragmentShader};
  ShaderType  shaderTypes[] = { ST_VERTEX_SHADER, ST_FRAGMENT_SHADER };
  ProgramShaderCode programShaderCode( sizeof dp::util::array( shaders ), shaders, shaderTypes );

  ProgramDescription programDescription( programShaderCode, &containerDescriptors[0], dp::checked_cast<unsigned int>(containerDescriptors.size() ));

  ProgramSharedHandle program = m_rix->programCreate( programDescription );

  ContainerSharedHandle vertexContainer  = m_rix->containerCreate( vertexContainerDescriptor );

  ProgramSharedHandle programs[] = { program };
  ProgramPipelineSharedHandle programPipeline = m_rix->programPipelineCreate( programs, sizeof util::array( programs ) );

  if( !m_geometryInstance )
  {
    m_geometryInstance = m_rix->geometryInstanceCreate();
    m_rix->geometryInstanceSetProgramPipeline(   m_geometryInstance, programPipeline );
    m_rix->geometryInstanceUseContainer( m_geometryInstance, m_vertexContainerW2V );
    m_rix->geometryInstanceUseContainer( m_geometryInstance, vertexContainer );
  }

  m_rix->renderGroupAddGeometryInstance( m_renderData->getRenderGroup(), m_geometryInstance );

  // prepare & set model2world matrix
  float mat[16];
  getTransformMatrix( mat, scale, scale, scale, transX, transY, transZ );
  m_rix->containerSetData( vertexContainer, containerEntryModel2World, ContainerDataRaw( 0, mat, 16*sizeof(float) ) );

}

unsigned int Benchmark_vertexThroughput::getSubdivs(unsigned int i)
{
  return 32+16*i;
}

unsigned int Benchmark_vertexThroughput::getNumIndices(unsigned int i)
{
  unsigned int subdivs = getSubdivs(i);
  return 2*subdivs + subdivs*(subdivs-2);
}

unsigned int Benchmark_vertexThroughput::getNumVerts(unsigned int i)
{
  unsigned int subdivs = getSubdivs(i);
  return 2 + (subdivs+1)*(subdivs-1);
}

std::string& Benchmark_vertexThroughput::getDescriptionOnRunInit( unsigned int i )
{
  m_curDesc = util::to_string( getNumIndices(i) );
  return m_curDesc;
}

std::string& Benchmark_vertexThroughput::getDescriptionOnRun( unsigned int i )
{
  m_curDesc = util::to_string( getNumIndices(i) );
  return m_curDesc;
}

std::string& Benchmark_vertexThroughput::getDescriptionOnRunClear( unsigned int i )
{
  m_curDesc = util::to_string( getNumIndices(i) );
  return m_curDesc;
}

