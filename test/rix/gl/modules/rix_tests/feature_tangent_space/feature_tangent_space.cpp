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
#include "feature_tangent_space.h"

#include <dp/gl/RenderTarget.h>
#include <dp/util/Image.h>
#include <dp/util/Types.h>
#include <dp/math/Trafo.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>
#include <test/rix/core/helpers/TextureHelper.h>

#include <limits>

//Automatically add the test to the module's global test list
REGISTER_TEST("feature_tangent_space", "tests simple usage of the RiX API", create_feature_tangent_space);

using namespace dp;
using namespace rix::core;
using namespace util::generator;

Feature_tangent_space::Feature_tangent_space()
  : m_numFrames(0)
  , m_containerEntryLightDir(-1)
  , m_fragConstContainer( nullptr )
  , m_renderData( nullptr )
  , m_rix( nullptr )
{
}

Feature_tangent_space::~Feature_tangent_space()
{
}

bool Feature_tangent_space::onInit()
{
  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = static_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  util::smart_cast<dp::gl::RenderTarget>( m_displayTarget )->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );

  m_renderData = new test::framework::RenderDataRiX;

  glEnable(GL_DEPTH_TEST);

  m_numFrames = 32;

  createScene();

  return true;
}

bool Feature_tangent_space::onRun( unsigned int idx )
{
  float angle = 2.0f*math::PI*idx/m_numFrames;

  math::Vec3f lightDirection(sinf(angle), cosf(angle), -1.0f);
  normalize(lightDirection);
  m_rix->containerSetData( m_fragConstContainer, m_containerEntryLightDir, ContainerDataRaw( 0, &lightDirection[0], 3 * sizeof(float) ) );

  render(m_renderData, m_displayTarget);

  return true;
}

bool Feature_tangent_space::onClear()
{
  delete m_renderData;

  return true;
}

void Feature_tangent_space::createScene()
{
  const char * vertexShader = ""
    "#version 400\n"
    "layout(location=0) in vec3 Position;\n"
    "layout(location=8) in vec2 TexCoord;\n\n"

    "layout(location=2) in vec3 Normal;\n"
    "layout(location=14) in vec3 Tangent;\n"
    "layout(location=15) in vec3 Binormal;\n"
    "uniform mat4 model2world;\n"
    "uniform mat4 model2worldIT;\n\n"

    "uniform mat4 world2view;\n"
    "uniform mat4 view2world;\n"
    "uniform mat4 view2clip;\n"
    "uniform mat4 world2clip;\n\n"

    "out vec3 vPosition;\n"
    "out vec2 vTexCoord;\n"
    "out vec3 vEyePos;\n"
    "out vec3 vNormal;\n"
    "out vec3 vTangent;\n"
    "out vec3 vBinormal;\n\n"

    "void main(void)\n"
    "{\n"
    "  vec4 worldPos = model2world * vec4( Position, 1.0f );\n\n"

    "  vPosition     = worldPos.xyz;\n"
    "  vTexCoord     = TexCoord;\n\n"

    "  vNormal       = ( model2worldIT * vec4( Normal, 0.0f ) ).xyz;\n"
    "  vTangent      = ( model2world * vec4( Tangent, 0.0f ) ).xyz;\n"
    "  vBinormal     = ( model2world * vec4( Binormal, 0.0f ) ).xyz;\n\n"

    "  //Get the translation part of the inverse view matrix\n"
    "  vEyePos       = vec3( view2world[3][0], view2world[3][1], view2world[3][2] );\n"
    "  gl_Position   = world2clip * worldPos;\n"
    "}\n";

  const char * fragmentShader = "" 
    "#version 400\n"
    "uniform vec4 color;\n"
    "uniform sampler2D bumpTex;\n"
    "uniform vec3 lightDir;\n\n"

    "in vec3 vPosition;\n"
    "in vec2 vTexCoord;\n"
    "in vec3 vEyePos;\n"
    "in vec3 vNormal;\n"
    "in vec3 vTangent;\n"
    "in vec3 vBinormal;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n\n"

    "// Phong lighting\n"
    "vec3 eval(in vec3 wo, in vec3 ns, in vec3 wi, \n"
    "          in vec3 ambientColor, in vec3 diffuseColor, in vec3 specularColor,\n"
    "          in float ambient, in float diffuse, in float specular, in float exponent)\n"
    "{\n"
    "  float shine = 0.0f;\n"
    "  float ns_dot_wi = max(0.0f, dot(ns, wi));\n"
    "  if(0.0f < ns_dot_wi)\n"
    "  {\n"
    "  // Phong\n"
    "    vec3 R = reflect(-wi, ns);\n"
    "    float r_dot_wo = max(0.0f, dot(R, wo));\n"
    "    shine = 0.0f < exponent ? pow(r_dot_wo, exponent) : 1.0f;\n"
    "  }\n"
    "  return ambient * ambientColor +\n"
    "         ns_dot_wi * diffuse * diffuseColor +\n"
    "         shine * specular * specularColor;\n"
    "}\n\n"

    "void main(void)\n"
    "{\n"
    "  vec3 emissiveColor = vec3(0.0f, 0.0f, 0.0f);\n"
    "  vec3 ambientColor  = vec3(0.0f, 0.0f, 0.0f);\n"
    "  vec3 diffuseColor  = color.xyz;\n"
    "  vec3 specularColor = vec3(1.0f, 1.0f, 1.0f);\n"
    "  float ambient       = 0.0f;\n"
    "  float diffuse       = 1.0f;\n"
    "  float specular      = 1.0f;\n"
    "  float exponent      = 16.0f;\n\n"

    "  //Sample the normal map texel and clamp it to [-1, 1]\n"
    "  vec3 normalTexel = 2.0f * texture(bumpTex, vTexCoord).xyz - 1.0f;\n"
    "  //Transform the sampled normal into world space\n"
    "  vec3 ns = normalTexel.x*normalize(vTangent) + normalTexel.y*normalize(vBinormal) + normalTexel.z*normalize(vNormal);\n"

    "  //Direction from our Eye to the fragment at hand in world space\n"
    "  vec3 wo = normalize(vEyePos - vPosition);\n\n"

    "  //Normalized light direction in tangent space\n"
    "  vec3 wi = -normalize(lightDir);\n\n"

    "  //Evaluate the final color using phong shading\n"
    "  vec3 rgb = eval( wo, ns, wi, \n"
    "              ambientColor, diffuseColor, specularColor, \n"
    "              ambient, diffuse, specular, exponent);\n"
    "  Color = vec4( rgb, 1.0f );\n\n"

    "}\n";
  //Geometry

  SmartGeometryData sphereData = createSphere( ATTRIB_POSITION | ATTRIB_NORMAL | ATTRIB_TEXCOORD0 | ATTRIB_TANGENT | ATTRIB_BINORMAL, 64, 32);
  GeometrySharedHandle sphere = rix::util::generateGeometry(sphereData, m_rix);

  SmartGeometryData cylinderData = createCylinder( ATTRIB_POSITION | ATTRIB_NORMAL | ATTRIB_TEXCOORD0 | ATTRIB_TANGENT |ATTRIB_BINORMAL, 64, 2 );
  GeometrySharedHandle cylinder = rix::util::generateGeometry(cylinderData, m_rix);
  
  SmartGeometryData quadData = createQuad( ATTRIB_POSITION | ATTRIB_NORMAL | ATTRIB_TEXCOORD0 | ATTRIB_TANGENT |ATTRIB_BINORMAL );
  GeometrySharedHandle quad = rix::util::generateGeometry(quadData, m_rix);

  // Container Descriptors

  ProgramParameter vertexConstProgramParameters[] = {
    ProgramParameter("world2view", CPT_MAT4X4),
    ProgramParameter("view2world", CPT_MAT4X4),
    ProgramParameter("view2clip",  CPT_MAT4X4),
    ProgramParameter("world2clip", CPT_MAT4X4)
  };

  ProgramParameter vertexVarProgramParameters[] = {
    ProgramParameter("model2world", CPT_MAT4X4),
    ProgramParameter("model2worldIT", CPT_MAT4X4)
  };

  ProgramParameter fragmentProgramParameters[] = {
    ProgramParameter("color", CPT_FLOAT4),
    ProgramParameter("bumpTex", CPT_SAMPLER)
  };


  ProgramParameter fragmentConstProgramParameters[] = {
    ProgramParameter("lightDir", CPT_FLOAT3)
  };

  ContainerDescriptorSharedHandle vertConstContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( vertexConstProgramParameters, 
    sizeof testfw::core::array(vertexConstProgramParameters) ) );

  ContainerDescriptorSharedHandle vertVarContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( vertexVarProgramParameters, 
    sizeof testfw::core::array(vertexVarProgramParameters) ) );

  ContainerDescriptorSharedHandle fragContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( fragmentProgramParameters,
    sizeof testfw::core::array(fragmentProgramParameters) ) );

  ContainerDescriptorSharedHandle fragConstContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( fragmentConstProgramParameters,
    sizeof testfw::core::array(fragmentConstProgramParameters) ) );


  // Program

  ProgramShaderCode vertShader( vertexShader, ST_VERTEX_SHADER );
  ProgramShaderCode fragShader( fragmentShader, ST_FRAGMENT_SHADER );

  ContainerDescriptorSharedHandle vertContainerDescriptors[] = { vertConstContainerDescriptor, vertVarContainerDescriptor };
  ContainerDescriptorSharedHandle fragContainerDescriptors[] = { fragContainerDescriptor, fragConstContainerDescriptor };

  ProgramDescription vertProgramDescription( vertShader, vertContainerDescriptors, sizeof testfw::core::array(vertContainerDescriptors) );
  ProgramDescription fragProgramDescription( fragShader, fragContainerDescriptors, sizeof testfw::core::array(fragContainerDescriptors) );

  ProgramSharedHandle vertProgram = m_rix->programCreate( vertProgramDescription );
  ProgramSharedHandle fragProgram = m_rix->programCreate( fragProgramDescription );


  // Program Pipeline

  ProgramSharedHandle programs[] = {vertProgram, fragProgram};
  ProgramPipelineSharedHandle programPipeline = m_rix->programPipelineCreate( programs, sizeof testfw::core::array(programs) );


  // Containers

  ContainerSharedHandle vertConstContainer = m_rix->containerCreate( vertConstContainerDescriptor );
  ContainerSharedHandle vertVarContainer0  = m_rix->containerCreate( vertVarContainerDescriptor );
  ContainerSharedHandle vertVarContainer1  = m_rix->containerCreate( vertVarContainerDescriptor );
  ContainerSharedHandle vertVarContainer2  = m_rix->containerCreate( vertVarContainerDescriptor );
  ContainerSharedHandle vertVarContainer3  = m_rix->containerCreate( vertVarContainerDescriptor );
  m_fragConstContainer = m_rix->containerCreate( fragConstContainerDescriptor );
  ContainerSharedHandle fragContainer0     = m_rix->containerCreate( fragContainerDescriptor );
  ContainerSharedHandle fragContainer1     = m_rix->containerCreate( fragContainerDescriptor );
  ContainerSharedHandle fragContainer2     = m_rix->containerCreate( fragContainerDescriptor );
  ContainerSharedHandle fragContainer3     = m_rix->containerCreate( fragContainerDescriptor );

  // Container Data

  math::Mat44f view2Clip = math::makeFrustum<float>(-0.1f, 0.1f, -0.1f * m_height / m_width, 0.1f * m_height / m_width, 0.1f, 50.0f);
  math::Mat44f world2View = math::makeLookAt<float>( math::Vec3f(0.0f, 0.0f, 5.0f), math::Vec3f(0.0f, 0.0f, 0.0f), math::Vec3f(0.0f, 1.0f, 0.0f) );
  math::Mat44f world2ViewI = world2View;
  world2ViewI.invert();
  math::Mat44f world2Clip = world2View * view2Clip;

  math::Trafo model2world0;
  model2world0.setTranslation( math::Vec3f(-3.6f, -2.0f, 0.0f) );
  math::Mat44f model2world0IT = ~model2world0.getInverse();

  math::Trafo model2world1;
  model2world1.setTranslation( math::Vec3f(0.0f, -2.0f, 0.0f) );
  math::Mat44f model2world1IT = ~model2world1.getInverse();

  math::Trafo model2world2;
  model2world2.setTranslation( math::Vec3f(3.6f, -2.0f, 0.0f) );
  math::Mat44f model2world2IT = ~model2world2.getInverse();

  math::Trafo model2world3;
  model2world3.setTranslation( math::Vec3f(-0.5f, -0.1f, 4.0f) );
  math::Mat44f model2world3IT = ~model2world3.getInverse();

  float bluish[4] = { 0.3f, 0.0f, 1.0f, 1.0f };
  float redish[4] = { 1.0f, 0.0f, 0.3f, 1.0f };
  float greenish[4] = { 0.3f, 1.0f, 0.3f, 1.0f };
  float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };


  // Geometry Instances

  GeometryInstanceSharedHandle geometryInstance0 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance0, sphere );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance0, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance0, vertConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance0, vertVarContainer0 );
  m_rix->geometryInstanceUseContainer( geometryInstance0, m_fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance0, fragContainer0 );

  GeometryInstanceSharedHandle geometryInstance1 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance1, cylinder );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance1, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance1, vertConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance1, vertVarContainer1 );
  m_rix->geometryInstanceUseContainer( geometryInstance1, m_fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance1, fragContainer1 );

  GeometryInstanceSharedHandle geometryInstance2 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance2, sphere );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance2, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance2, vertConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance2, vertVarContainer2 );
  m_rix->geometryInstanceUseContainer( geometryInstance2, m_fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance2, fragContainer2 );


  GeometryInstanceSharedHandle geometryInstance3 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance3, quad );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance3, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance3, vertConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance3, vertVarContainer3 );
  m_rix->geometryInstanceUseContainer( geometryInstance3, m_fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance3, fragContainer3 );


  // Render Group

  RenderGroupSharedHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance0 );
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance1 );
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance2 );
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance3 );


  // Get container entries

  ContainerEntry containerEntryView2clip   = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "view2clip" );
  ContainerEntry containerEntryWorld2view  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "world2view" );
  ContainerEntry containerEntryView2world  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "view2world" );
  ContainerEntry containerEntryWorld2clip  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "world2clip" );

  ContainerEntry containerEntryModel2world = m_rix->containerDescriptorGetEntry( vertVarContainerDescriptor, "model2world" );
  ContainerEntry containerEntryModel2worldIT = m_rix->containerDescriptorGetEntry( vertVarContainerDescriptor, "model2worldIT" );

  ContainerEntry containerEntryColor       = m_rix->containerDescriptorGetEntry( fragContainerDescriptor, "color" );
  ContainerEntry containerEntryBumpTex     = m_rix->containerDescriptorGetEntry( fragContainerDescriptor, "bumpTex" );
  m_containerEntryLightDir    = m_rix->containerDescriptorGetEntry( fragConstContainerDescriptor, "lightDir" );


  // Set Container Data

  SamplerStateDataCommon samplerStateDataCommon( SSFM_NEAREST, SSFM_NEAREST );
  SamplerStateSharedHandle samplerStateHandle = m_rix->samplerStateCreate(samplerStateDataCommon);

  m_rix->containerSetData( vertConstContainer,  containerEntryView2clip,  ContainerDataRaw( 0, view2Clip.getPtr(),   16 * sizeof(float) ) );
  m_rix->containerSetData( vertConstContainer,  containerEntryWorld2view, ContainerDataRaw( 0, world2View.getPtr(),  16 * sizeof(float) ) );
  m_rix->containerSetData( vertConstContainer,  containerEntryView2world, ContainerDataRaw( 0, world2ViewI.getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( vertConstContainer,  containerEntryWorld2clip, ContainerDataRaw( 0, world2Clip.getPtr(),  16 * sizeof(float) ) );


  m_rix->containerSetData( vertVarContainer0,   containerEntryModel2world, ContainerDataRaw( 0, model2world0.getMatrix().getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( vertVarContainer0,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world0IT.getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( fragContainer0,      containerEntryColor,       ContainerDataRaw( 0, redish, 4 * sizeof(float) ) );
  
  m_rix->containerSetData( vertVarContainer1,   containerEntryModel2world, ContainerDataRaw( 0, model2world1.getMatrix().getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( vertVarContainer1,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world1IT.getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( fragContainer1,      containerEntryColor,       ContainerDataRaw( 0, greenish, 4 * sizeof(float) ) );
  
  m_rix->containerSetData( vertVarContainer2,   containerEntryModel2world, ContainerDataRaw( 0, model2world2.getMatrix().getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( vertVarContainer2,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world2IT.getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( fragContainer2,      containerEntryColor,       ContainerDataRaw( 0, bluish, 4 * sizeof(float) ) );

  m_rix->containerSetData( vertVarContainer3,   containerEntryModel2world, ContainerDataRaw( 0, model2world3.getMatrix().getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( vertVarContainer3,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world3IT.getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( fragContainer3,      containerEntryColor,       ContainerDataRaw( 0, white, 4 * sizeof(float) ) );

  dp::util::generator::SmartTextureObjectData noiseTexture = dp::util::generator::createNoiseTexture( math::Vec2ui(256, 256), 10.0f, 20.0f );
  dp::util::generator::SmartTextureObjectData normalTexture = dp::util::generator::convertHeightMapToNormalMap( noiseTexture, 0.014f );
  TextureSharedHandle noiseNormalMap = dp::rix::util::generateTexture( m_rix, normalTexture, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_32, ITF_RGBA8 );

  dp::util::generator::SmartTextureObjectData pyramidNormalTexture = dp::util::generator::createPyramidNormalMap( math::Vec2ui(256, 256), math::Vec2ui(16, 16), 0.03125f );
  TextureSharedHandle pyramidNormalMap = dp::rix::util::generateTexture( m_rix, pyramidNormalTexture, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_32, ITF_RGBA8 );

  rix::core::SamplerSharedHandle samplerNoiseNormalMap = m_rix->samplerCreate();
  m_rix->samplerSetTexture( samplerNoiseNormalMap, noiseNormalMap );
  m_rix->samplerSetSamplerState( samplerNoiseNormalMap, samplerStateHandle );

  rix::core::SamplerSharedHandle samplerPyramidNormalMap = m_rix->samplerCreate();
  m_rix->samplerSetTexture( samplerPyramidNormalMap, pyramidNormalMap );
  m_rix->samplerSetSamplerState( samplerPyramidNormalMap, samplerStateHandle );

  m_rix->containerSetData( fragContainer0, containerEntryBumpTex, ContainerDataSampler( samplerNoiseNormalMap ) );
  m_rix->containerSetData( fragContainer1, containerEntryBumpTex, ContainerDataSampler( samplerNoiseNormalMap ) );
  m_rix->containerSetData( fragContainer2, containerEntryBumpTex, ContainerDataSampler( samplerNoiseNormalMap ) );
  m_rix->containerSetData( fragContainer3, containerEntryBumpTex, ContainerDataSampler( samplerPyramidNormalMap) );

}

bool Feature_tangent_space::onRunCheck( unsigned int i )
{
  return i < m_numFrames;
}
