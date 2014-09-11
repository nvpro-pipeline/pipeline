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
#include "##test##.h"

#include <dp/math/Trafo.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>

#include <limits>

//Automatically add the test to the module's global test list
REGISTER_TEST("##test##", "##description##", create_##test##);

using namespace dp;
using namespace rix::core;

##test##::##test##()
{
}

##test##::~##test##()
{
}

bool ##test##::onInit()
{
  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = static_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  m_renderData = new test::framework::RenderDataRiX;
  util::smart_cast<util::gl::RenderTargetGL>( m_displayTarget )->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );

  createScene();

  return true;  
}

bool ##test##::onRun( unsigned int idx )
{
  render(m_renderData, m_displayTarget);

  return true;
}

bool ##test##::onClear()
{
  delete m_renderData;

  return true;
}

void ##test##::createScene()
{
  const char * vertexShader = ""
    "#version 330\n"
    "layout(location=0) in vec3 Position;\n"
    "uniform mat4 world2view;\n"
    "uniform mat4 model2world;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = world2view * model2world * vec4( Position, 1.0 );\n"
    "}\n";

  const char * fragmentShader = "" 
    "#version 330\n"
    "uniform vec4 color;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = color;\n"
    "}\n";

  math::Mat44f ortho = math::makeOrtho<float>( 0.0f, 1.0f*m_width, 0.0f, 1.0f*m_height, -1.0f, 1.0f );

  math::Trafo model2world0;
  model2world0.setTranslation( math::Vec3f(0.5f*m_width, 0.5f*m_height, 0.0f) );
  math::Trafo model2world1;
  model2world1.setTranslation( math::Vec3f(0.0f, 0.5f*m_height, 0.0f) );
  float bluish[4] = { 0.3f, 0.0f, 1.0f, 1.0f };
  float redish[4] = { 1.0f, 0.0f, 0.3f, 1.0f };


  dp::util::generator::SmartGeometryData mesh = createTriangle( dp::util::generator::ATTRIB_POSITION 
    , math::Vec3f( 0.0f, 0.0f, 0.0f )
    , math::Vec3f( 0.5f*m_width, 0.0f, 0.0f )
    , math::Vec3f( 0.0f, 0.5f*m_height, 0.0f) );
  SmartGeometryHandle geometry = dp::rix::util::generateGeometry(mesh, m_rix);


  // Container Descriptors

  ProgramParameter vertexConstProgramParameters[] = {
    ProgramParameter("world2view",  CPT_MAT4X4)
  };

  ProgramParameter vertexVarProgramParameters[] = {
    ProgramParameter("model2world", CPT_MAT4X4)
  };

  ProgramParameter fragmentProgramParameters[] = {
    ProgramParameter("color", CPT_FLOAT4) 
  };

  SmartContainerDescriptorHandle vertConstContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( vertexConstProgramParameters, 
    sizeof util::array(vertexConstProgramParameters) ) );

  SmartContainerDescriptorHandle vertVarContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( vertexVarProgramParameters, 
    sizeof util::array(vertexVarProgramParameters) ) );

  SmartContainerDescriptorHandle fragContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( fragmentProgramParameters,
    sizeof util::array(fragmentProgramParameters) ) );


  // Program

  ProgramShaderCode vertShader( vertexShader, ST_VERTEX_SHADER );
  ProgramShaderCode fragShader( fragmentShader, ST_FRAGMENT_SHADER );

  ContainerDescriptorHandle vertContainerDescriptors[] = { vertConstContainerDescriptor, vertVarContainerDescriptor };
  ContainerDescriptorHandle fragContainerDescriptors[] = { fragContainerDescriptor };

  ProgramDescription vertProgramDescription( vertShader, vertContainerDescriptors, sizeof util::array(vertContainerDescriptors) );
  ProgramDescription fragProgramDescription( fragShader, fragContainerDescriptors, sizeof util::array(fragContainerDescriptors) );

  SmartProgramHandle vertProgram = m_rix->programCreate( vertProgramDescription );
  SmartProgramHandle fragProgram = m_rix->programCreate( fragProgramDescription );


  // Program Pipeline

  ProgramHandle programs[] = {vertProgram, fragProgram};
  SmartProgramPipelineHandle programPipeline = m_rix->programPipelineCreate( programs, sizeof util::array(programs) );


  // Containers

  SmartContainerHandle vertConstContainer = m_rix->containerCreate( vertConstContainerDescriptor );
  SmartContainerHandle vertVarContainer0  = m_rix->containerCreate( vertVarContainerDescriptor );
  SmartContainerHandle vertVarContainer1  = m_rix->containerCreate( vertVarContainerDescriptor );
  SmartContainerHandle fragContainer0     = m_rix->containerCreate( fragContainerDescriptor );
  SmartContainerHandle fragContainer1     = m_rix->containerCreate( fragContainerDescriptor );


  // Geometry Instances

  SmartGeometryInstanceHandle geometryInstance0 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance0, geometry );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance0, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance0, vertConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance0, vertVarContainer0 );
  m_rix->geometryInstanceUseContainer( geometryInstance0, fragContainer0 );

  SmartGeometryInstanceHandle geometryInstance1 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance1, geometry );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance1, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance1, vertConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance1, vertVarContainer1 );
  m_rix->geometryInstanceUseContainer( geometryInstance1, fragContainer1 );



  // Render Group

  SmartRenderGroupHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance0 );
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance1 );


  // Get container entries

  ContainerEntry containerEntryWorld2view  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "world2view" );
  ContainerEntry containerEntryModel2world = m_rix->containerDescriptorGetEntry( vertVarContainerDescriptor, "model2world" );
  ContainerEntry containerEntryColor       = m_rix->containerDescriptorGetEntry( fragContainerDescriptor, "color" );


  // Set Container Data

  m_rix->containerSetData( vertConstContainer,  containerEntryWorld2view,  ContainerDataRaw( 0, ortho.getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( vertVarContainer0,   containerEntryModel2world, ContainerDataRaw( 0, model2world0.getMatrix().getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( fragContainer0,      containerEntryColor,       ContainerDataRaw( 0, redish, 4*sizeof(float) ) );
  m_rix->containerSetData( vertVarContainer1,   containerEntryModel2world, ContainerDataRaw( 0, model2world1.getMatrix().getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( fragContainer1,      containerEntryColor,       ContainerDataRaw( 0, bluish, 4*sizeof(float) ) );

}
