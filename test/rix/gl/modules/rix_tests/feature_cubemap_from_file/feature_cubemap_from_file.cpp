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
#include "feature_cubemap_from_file.h"

#include <dp/gl/RenderTarget.h>
#include <dp/util/Image.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>
#include <test/rix/core/helpers/TextureHelper.h>

#include <limits>

//Automatically add the test to the module's global test list
REGISTER_TEST("feature_cubemap_from_file", "Loads a cubemap and uses it on a sphere", create_feature_cubemap_from_file);

using namespace dp;
using namespace math;
using namespace rix::core;
using namespace rix::util;

Feature_cubemap_from_file::Feature_cubemap_from_file()
  : m_renderData(nullptr)
  , m_rix(nullptr)
  , m_numFrames(0)
  , m_vertViewProjContainer(nullptr)
{
}

Feature_cubemap_from_file::~Feature_cubemap_from_file()
{
}

bool Feature_cubemap_from_file::onInit()
{
  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = static_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  m_displayTarget.inplaceCast<dp::gl::RenderTarget>()->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );

  m_renderData = new test::framework::RenderDataRiX;


  m_numFrames = 32;
  m_orbitRad = 3.0f;
  m_zPanSegment = 2.0f;
  createScene();

  return true;
}

bool Feature_cubemap_from_file::onRun( unsigned int idx )
{
  float phase = (float)idx/m_numFrames;
  float anglePhase = 2.0f*dp::math::PI*phase;

  Mat44f world2View = makeLookAt<float>( Vec3f( m_orbitRad*sin(anglePhase)
                                              , m_orbitRad*cos(anglePhase)
                                              , m_zPanSegment - m_zPanSegment*phase)
                                       , Vec3f(0.0f, 0.0f, 0.0f)
                                       , Vec3f(0.0f, 1.0f, 0.0f) );

  math::Mat44f world2ViewI = world2View;
  world2ViewI.invert();
  Mat44f world2Clip = world2View * m_view2Clip;

  m_rix->containerSetData( m_vertViewProjContainer,  m_containerEntryWorld2clip, ContainerDataRaw( 0, world2Clip.getPtr(),  16*sizeof(float) ) );
  m_rix->containerSetData( m_vertViewProjContainer,  m_containerEntryView2world, ContainerDataRaw( 0, world2ViewI.getPtr(),  16*sizeof(float) ) );

  render(m_renderData, m_displayTarget);

  return true;
}

bool Feature_cubemap_from_file::onClear()
{
  delete m_renderData;

  return true;
}

void Feature_cubemap_from_file::createScene()
{
  const char * vertexShader = ""
    "#version 400\n"
    "layout(location=0) in vec3 Position;\n"
    "layout(location=2) in vec3 Normal;\n\n"
    "layout(location=8) in vec2 TexCoord;\n\n"

    "uniform mat4 view2world;\n"
    "uniform mat4 world2clip;\n\n"

    "uniform mat4 model2world;\n"
    "uniform mat4 model2worldIT;\n\n"

    "out vec4 vPosition;\n"
    "out vec2 vTexCoord;\n"
    "out vec3 vNormal;\n\n"

    "out vec3 vEyePos;\n"

    "void main(void)\n"
    "{\n"
    "  vPosition  = model2world * vec4( Position, 1.0 );\n\n"

    "  vTexCoord      = TexCoord;\n"
    "  vNormal       = ( model2worldIT * vec4( Normal, 0.0f ) ).xyz;\n"
    "  vEyePos       = vec3( view2world[3][0], view2world[3][1], view2world[3][2] );\n"
    "  gl_Position    = world2clip * vPosition;\n"
    "}\n";

  const char * fragmentCubemapShader = ""
    "#version 400\n"
    "uniform samplerCube cubeTex;\n\n"

    "in vec4 vPosition;\n"
    "in vec2 vTexCoord;\n"
    "in vec3 vNormal;\n\n"

    "in vec3 vEyePos;\n\n"

    "layout(location = 0, index = 0) out vec4 Color;\n\n"

    "void main(void)\n"
    "{\n"
    "  vec3 wo = normalize(vPosition.xyz - vEyePos);\n"
    "  vec3 ns = -normalize(vNormal);\n"
    "  vec3 rgb = texture( cubeTex, reflect( wo, ns ) ).rgb;\n"
    "  Color = vec4( rgb, 1.0 );\n"
    "}\n";

  //Geometry
  GeometrySharedHandle simpleSphere = rix::util::generateGeometry( createSphere( AttributeID::POSITION | AttributeID::TEXCOORD0 | AttributeID::NORMAL, 64, 64), m_rix );

  // Container Descriptors

  ProgramParameter vertexConstProgramParameters[] = {
    ProgramParameter("view2world", ContainerParameterType::MAT4X4),
    ProgramParameter("world2clip", ContainerParameterType::MAT4X4)
  };

  ProgramParameter vertexVarProgramParameters[] = {
    ProgramParameter("model2world", ContainerParameterType::MAT4X4),
    ProgramParameter("model2worldIT", ContainerParameterType::MAT4X4)
  };

  ProgramParameter fragmentTexturedProgramParameters[] = {
    ProgramParameter("cubeTex", ContainerParameterType::SAMPLER)
  };

  ContainerDescriptorSharedHandle vertConstContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( vertexConstProgramParameters,
    sizeof testfw::core::array(vertexConstProgramParameters) ) );

  ContainerDescriptorSharedHandle vertVarContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( vertexVarProgramParameters,
    sizeof testfw::core::array(vertexVarProgramParameters) ) );

  ContainerDescriptorSharedHandle fragCubemapReflectionDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( fragmentTexturedProgramParameters,
    sizeof testfw::core::array(fragmentTexturedProgramParameters) ) );


  // Program

  ProgramShaderCode vertShader( vertexShader, ShaderType::VERTEX_SHADER );
  ProgramShaderCode fragTexturedShader( fragmentCubemapShader, ShaderType::FRAGMENT_SHADER );

  ContainerDescriptorSharedHandle vertContainerDescriptors[] = { vertConstContainerDescriptor, vertVarContainerDescriptor };
  ContainerDescriptorSharedHandle fragTexturedContainerDescriptors[] = { fragCubemapReflectionDescriptor };

  ProgramDescription vertProgramDescription( vertShader, vertContainerDescriptors, sizeof testfw::core::array(vertContainerDescriptors) );
  ProgramDescription fragTexturedProgramDescription( fragTexturedShader, fragTexturedContainerDescriptors, sizeof testfw::core::array(fragTexturedContainerDescriptors) );

  ProgramSharedHandle vertProgram = m_rix->programCreate( vertProgramDescription );
  ProgramSharedHandle fragTexturedProgram = m_rix->programCreate( fragTexturedProgramDescription );


  // Program Pipeline

  ProgramSharedHandle programsTextured[] = {vertProgram, fragTexturedProgram};
  ProgramPipelineSharedHandle programTexturedPipeline = m_rix->programPipelineCreate( programsTextured, sizeof testfw::core::array(programsTextured) );


  // Containers

  m_vertViewProjContainer   = m_rix->containerCreate( vertConstContainerDescriptor );

  ContainerSharedHandle vertCubemapReflectionData = m_rix->containerCreate( vertVarContainerDescriptor );
  ContainerSharedHandle fragCubemapReflectionData = m_rix->containerCreate( fragCubemapReflectionDescriptor );

  // Container Data
  m_aspectRatio = (float)m_height/m_width;
  m_view2Clip = makeFrustum<float>(-0.1f, 0.1f, -0.1f*m_aspectRatio, 0.1f*m_aspectRatio, 0.1f, 50.0f);

  Trafo model2world;
  math::Mat44f model2worldIT = ~model2world.getInverse();

  // Geometry Instances

  GeometryInstanceSharedHandle geometryInstanceCubemappedSphere = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstanceCubemappedSphere, simpleSphere );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstanceCubemappedSphere, programTexturedPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstanceCubemappedSphere, m_vertViewProjContainer );
  m_rix->geometryInstanceUseContainer( geometryInstanceCubemappedSphere, vertCubemapReflectionData );
  m_rix->geometryInstanceUseContainer( geometryInstanceCubemappedSphere, fragCubemapReflectionData );



  // Render Group
  RenderGroupSharedHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstanceCubemappedSphere );


  // Get container entries
  m_containerEntryWorld2clip  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "world2clip" );
  m_containerEntryView2world  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "view2world" );

  ContainerEntry containerEntryModel2world = m_rix->containerDescriptorGetEntry( vertVarContainerDescriptor, "model2world" );
  ContainerEntry containerEntryModel2worldIT = m_rix->containerDescriptorGetEntry( vertVarContainerDescriptor, "model2worldIT" );
  ContainerEntry containerEntryCubemap     = m_rix->containerDescriptorGetEntry( fragCubemapReflectionDescriptor, "cubeTex" );


  SamplerStateDataCommon samplerStateDataCommon( SamplerStateFilterMode::NEAREST, SamplerStateFilterMode::NEAREST );
  SamplerStateSharedHandle samplerStateHandle = m_rix->samplerStateCreate(samplerStateDataCommon);

  std::string dpBasePath = getenv("DPHOME");
  std::string texturePath = dpBasePath + "/test/rix/gl/modules/" + CURRENT_MODULE_NAME + "/textures/";
  TextureSharedHandle textureCubemap = dp::rix::util::createCubemapFromFile(m_rix, texturePath + "nvlobby_cube_mipmap.dds");

  // Set Container Data
  m_rix->containerSetData( vertCubemapReflectionData,   containerEntryModel2world,   ContainerDataRaw( 0, model2world.getMatrix().getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( vertCubemapReflectionData,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2worldIT.getPtr(), 16 * sizeof(float) ) );
  rix::core::SamplerSharedHandle sampler = m_rix->samplerCreate();
  m_rix->samplerSetTexture( sampler, textureCubemap );
  m_rix->samplerSetSamplerState( sampler, samplerStateHandle );
  m_rix->containerSetData( fragCubemapReflectionData,    containerEntryCubemap,       ContainerDataSampler( sampler ) );

}

bool Feature_cubemap_from_file::onRunCheck( unsigned int i )
{
  return i < m_numFrames;
}
