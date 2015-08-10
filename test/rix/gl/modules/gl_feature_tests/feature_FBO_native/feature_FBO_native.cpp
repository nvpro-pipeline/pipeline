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
#include "feature_FBO_native.h"

#include <dp/util/Image.h>
#include <dp/util/SharedPtr.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>
#include <test/rix/core/helpers/TextureHelper.h>

#include <limits>

//Automatically add the test to the module's global test list
REGISTER_TEST("feature_FBO_native", "Render to texture with native gl calls", create_feature_FBO_native);

using namespace dp;
using namespace math;
using namespace rix::core;
using namespace rix::util;

Feature_FBO_native::Feature_FBO_native()
  : m_renderData(nullptr)
  , m_rix(nullptr)
  , m_containerEntryFBOTexture(0)
  , m_containerEntryWorld2view(0)
  , m_containerEntryView2world(0)
  , m_containerEntryView2clip(0)
  , m_containerEntryWorld2clip(0)
{
}

Feature_FBO_native::~Feature_FBO_native()
{
}

bool Feature_FBO_native::onInit()
{
  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = dynamic_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  dp::util::shared_cast<dp::gl::RenderTarget>( m_displayTarget )->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );
  
  m_renderData = new test::framework::RenderDataRiX;

  glEnable(GL_DEPTH_TEST);

  glewInit();

  glGenFramebuffers( 1, &m_framebufferName );
  createScene();

  return true;  
}

bool Feature_FBO_native::onRun( unsigned int idx )
{
  glBindFramebuffer( GL_FRAMEBUFFER, m_framebufferName );
  if(!idx)
  {
    glFramebufferTexture( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_colorTexture->getGLId(), 0 );
    glFramebufferTexture( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_depthTexture->getGLId(), 0 );
  }
  glDrawBuffer(GL_COLOR_ATTACHMENT0);

  glClearColor( 0.72f, 0.46f, 0.0f, 1.0f );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  m_rix->render(m_renderGroupScene);


  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  m_renderData->setRenderGroup(m_renderGroupRTT);
  render(m_renderData, m_displayTarget);

  return true;
}

bool Feature_FBO_native::onClear()
{
  delete m_renderData;

  glDeleteFramebuffers(1, &m_framebufferName);

  return true;
}

void Feature_FBO_native::createScene()
{
  const char * vertexShader = ""
    "#version 400\n"
    "layout(location=0) in vec3 Position;\n"
    "layout(location=2) in vec3 Normal;\n"
    "layout(location=8) in vec2 TexCoord;\n\n"

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
    "  vec4 worldPos = model2world * vec4( Position, 1.0 );\n\n"

    "  vPosition     = worldPos.xyz;\n"
    "  vTexCoord     = TexCoord;\n"
    "  vNormal       = ( model2worldIT * vec4( Normal, 0.0 ) ).xyz;\n\n"

    "  //Get the translation part of the inverse view matrix\n"
    "  vEyePos       = vec3( view2world[3][0], view2world[3][1], view2world[3][2] );\n"
    "  gl_Position   = world2clip * worldPos;\n"
    "}\n";

  const char * fragmentTexturePhongShader = "" 
    "#version 400\n"
    "uniform sampler2D diffuseTex;\n"
    "uniform vec3 lightDir;\n\n"

    "in vec3 vPosition;\n"
    "in vec2 vTexCoord;\n"
    "in vec3 vNormal;\n"
    "in vec3 vEyePos;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n\n"

    "// Phong lighting\n"
    "vec3 eval(in vec3 wo, in vec3 ns, in vec3 wi, \n"
    "          in vec3 ambientColor, in vec3 diffuseColor, in vec3 specularColor,\n"
    "          in float ambient, in float diffuse, in float specular, in float exponent)\n"
    "{\n"
    "  float shine = 0.0;\n"
    "  float ns_dot_wi = max(0.0, dot(ns, wi));\n"
    "  if(0.0 < ns_dot_wi)\n"
    "  {\n"
    "  // Phong\n"
    "    vec3 R = reflect(-wi, ns);\n"
    "    float r_dot_wo = max(0.0, dot(R, wo));\n"
    "    shine = 0.0 < exponent ? pow(r_dot_wo, exponent) : 1.0;\n"
    "  }\n"
    "  return ambient * ambientColor +\n"
    "         ns_dot_wi * diffuse * diffuseColor +\n"
    "         shine * specular * specularColor;\n"
    "}\n\n"

    "void main(void)\n"
    "{\n"
    "  vec3 emissiveColor;\n"
    "  vec3 ambientColor;\n"
    "  vec3 diffuseColor;\n"
    "  vec3 specularColor;\n"
    "  float ambient;\n"
    "  float diffuse;\n"
    "  float specular;\n"
    "  float exponent;\n"

    "  emissiveColor = vec3(0.0, 0.0, 0.0);\n"
    "  ambientColor  = vec3(0.0, 0.0, 0.0);\n"
    "  diffuseColor  = texture(diffuseTex, vTexCoord).xyz;\n"
    "  specularColor = vec3(1.0, 1.0, 1.0);\n"
    "  exponent      = 16.0;\n"
    "  ambient       = 0.0;\n"
    "  diffuse       = 1.0;\n"
    "  specular      = 1.0;\n\n"

    "  //Direction from our Eye to the fragment at hand in tangent space\n"
    "  vec3 wo = normalize(vEyePos - vPosition);\n\n"

    "  //Normalized light direction in tangent space\n"
    "  vec3 wi = normalize(lightDir);\n\n"

    "  //Normal at fragment at hand read from the normal map\n"
    "  vec3 ns = normalize(vNormal);\n"
    "  vec3 rgb = eval( wo, ns, wi, \n"
    "              ambientColor, diffuseColor, specularColor, \n"
    "              ambient, diffuse, specular, exponent);\n"
    "  Color = vec4( rgb, 1.0 );\n\n"

    "}\n";

  const char * fragmentTextureRTTShader = "" 
    "#version 400\n"
    "uniform sampler2D FBOTex;\n"

    "//in vec3 vPosition;\n"
    "in vec2 vTexCoord;\n"
    "//in vec3 vNormal;\n"
    "//in vec3 vEyePos;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n\n"

    "/*// Phong lighting\n"
    "vec3 eval(in vec3 wo, in vec3 ns, in vec3 wi, \n"
    "          in vec3 ambientColor, in vec3 diffuseColor, in vec3 specularColor,\n"
    "          in float ambient, in float diffuse, in float specular, in float exponent)\n"
    "{\n"
    "  float shine = 0.0;\n"
    "  float ns_dot_wi = max(0.0, dot(ns, wi));\n"
    "  if(0.0 < ns_dot_wi)\n"
    "  {\n"
    "  // Phong\n"
    "    vec3 R = reflect(-wi, ns);\n"
    "    float r_dot_wo = max(0.0, dot(R, wo));\n"
    "    shine = 0.0 < exponent ? pow(r_dot_wo, exponent) : 1.0;\n"
    "  }\n"
    "  return ambient * ambientColor +\n"
    "         ns_dot_wi * diffuse * diffuseColor +\n"
    "         shine * specular * specularColor;\n"
    "}*/\n\n"

    "void main(void)\n"
    "{\n"
    "  Color = vec4( texture(FBOTex, vTexCoord).xyz, 1.0 );\n"
    "}\n";

  //Geometry
  GeometryDataSharedPtr cylinderDataNormal  = createCylinder( ATTRIB_POSITION | ATTRIB_NORMAL | ATTRIB_TEXCOORD0, 64 );
  GeometryDataSharedPtr cylinderDataCut     = createCylinder( ATTRIB_POSITION | ATTRIB_NORMAL | ATTRIB_TEXCOORD0, 64, 8, 3.0f*PI_HALF );
  GeometryDataSharedPtr cylinderDataCutTube = createCylinder( ATTRIB_POSITION | ATTRIB_NORMAL | ATTRIB_TEXCOORD0, 64, 8, 3.0f*PI_HALF, 0.5f );

  GeometryDataSharedPtr quadDataRTTScreen = createQuad( ATTRIB_POSITION | ATTRIB_TEXCOORD0 );
  
  GeometrySharedHandle cylinderNormal = rix::util::generateGeometry(cylinderDataNormal, m_rix);
  GeometrySharedHandle cylinderCut = rix::util::generateGeometry(cylinderDataCut, m_rix);
  GeometrySharedHandle cylinderCutTube = rix::util::generateGeometry(cylinderDataCutTube, m_rix);
  GeometrySharedHandle quadRTTScreen = rix::util::generateGeometry(quadDataRTTScreen, m_rix);

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

  ProgramParameter fragmentTexturedPhongProgramParameters[] = {
    ProgramParameter("diffuseTex", CPT_SAMPLER) 
  };
  ProgramParameter fragmentTextureRTTProgramParameters[] = {
    ProgramParameter("FBOTex", CPT_SAMPLER) 
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

  ContainerDescriptorSharedHandle fragTexturedPhongContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( fragmentTexturedPhongProgramParameters,
    sizeof testfw::core::array(fragmentTexturedPhongProgramParameters) ) );
  ContainerDescriptorSharedHandle fragTextureRTTContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( fragmentTextureRTTProgramParameters,
    sizeof testfw::core::array(fragmentTextureRTTProgramParameters) ) );

  ContainerDescriptorSharedHandle fragConstContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( fragmentConstProgramParameters,
    sizeof testfw::core::array(fragmentConstProgramParameters) ) );


  // Program

  ProgramShaderCode vertShader( vertexShader, ST_VERTEX_SHADER );
  ProgramShaderCode fragTexturedPhongShader( fragmentTexturePhongShader, ST_FRAGMENT_SHADER );
  ProgramShaderCode fragTextureRTTShader( fragmentTextureRTTShader, ST_FRAGMENT_SHADER );

  ContainerDescriptorSharedHandle vertContainerDescriptors[] = { vertConstContainerDescriptor, vertVarContainerDescriptor };
  ContainerDescriptorSharedHandle fragTexturedPhongContainerDescriptors[] = { fragTexturedPhongContainerDescriptor, fragConstContainerDescriptor };
  ContainerDescriptorSharedHandle fragTextureRTTContainerDescriptors[] = { fragTextureRTTContainerDescriptor  };

  ProgramDescription vertProgramDescription( vertShader, vertContainerDescriptors, sizeof testfw::core::array(vertContainerDescriptors) );
  ProgramDescription fragTexturedPhongProgramDescription( fragTexturedPhongShader, fragTexturedPhongContainerDescriptors, sizeof testfw::core::array(fragTexturedPhongContainerDescriptors) );
  ProgramDescription fragmentTextureRTTProgramDescription( fragTextureRTTShader, fragTextureRTTContainerDescriptors, sizeof testfw::core::array(fragTextureRTTContainerDescriptors) );

  ProgramSharedHandle vertProgram = m_rix->programCreate( vertProgramDescription );

  ProgramSharedHandle fragTexturedPhongProgram = m_rix->programCreate( fragTexturedPhongProgramDescription );
  ProgramSharedHandle fragTextureRTTProgram = m_rix->programCreate( fragmentTextureRTTProgramDescription );


  // Phong material Program Pipeline

  ProgramSharedHandle programsTexturedPhong[] = {vertProgram, fragTexturedPhongProgram};
  ProgramPipelineSharedHandle programTexturedPipeline = m_rix->programPipelineCreate( programsTexturedPhong, sizeof testfw::core::array(programsTexturedPhong) );

  // FBO texture material Program Pipeline
  ProgramSharedHandle programsTextureRTT[] = {vertProgram, fragTextureRTTProgram};
  ProgramPipelineSharedHandle programTextureRTTPipeline = m_rix->programPipelineCreate( programsTextureRTT, sizeof testfw::core::array(programsTextureRTT) );


  // Get container entries

  m_containerEntryView2clip   = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "view2clip" );
  m_containerEntryWorld2view  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "world2view" );
  m_containerEntryView2world  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "view2world" );
  m_containerEntryWorld2clip  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "world2clip" );

  ContainerEntry containerEntryModel2world = m_rix->containerDescriptorGetEntry( vertVarContainerDescriptor, "model2world" );
  ContainerEntry containerEntryModel2worldIT = m_rix->containerDescriptorGetEntry( vertVarContainerDescriptor, "model2worldIT" );

  ContainerEntry containerEntryTexture     = m_rix->containerDescriptorGetEntry( fragTexturedPhongContainerDescriptor, "diffuseTex" );
  ContainerEntry containerEntryLightDir    = m_rix->containerDescriptorGetEntry( fragConstContainerDescriptor, "lightDir" );


  m_containerEntryFBOTexture = m_rix->containerDescriptorGetEntry( fragTextureRTTContainerDescriptor, "FBOTex" );




  // Containers

  m_vertViewProjContainer          = m_rix->containerCreate( vertConstContainerDescriptor );
  ContainerSharedHandle fragConstContainer             = m_rix->containerCreate( fragConstContainerDescriptor );

  ContainerSharedHandle vertCylinderNormalTransformContainer  = m_rix->containerCreate( vertVarContainerDescriptor );
  ContainerSharedHandle fragCylinderNormalMaterialContainer   = m_rix->containerCreate( fragTexturedPhongContainerDescriptor );

  ContainerSharedHandle vertCylinderCutTransformContainer = m_rix->containerCreate( vertVarContainerDescriptor );
  ContainerSharedHandle fragCylinderCutMaterialContainer  = m_rix->containerCreate( fragTexturedPhongContainerDescriptor );

  ContainerSharedHandle vertCylinderCutTubeTransformContainer  = m_rix->containerCreate( vertVarContainerDescriptor );
  ContainerSharedHandle fragCylinderCutTubeMaterialContainer   = m_rix->containerCreate( fragTexturedPhongContainerDescriptor );

  ContainerSharedHandle vertRTTContainer  = m_rix->containerCreate( vertVarContainerDescriptor );
  m_fragRTTContainer   = m_rix->containerCreate( fragTextureRTTContainerDescriptor );

  // Geometry Instances

  GeometryInstanceSharedHandle geometryInstance0 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance0, cylinderNormal );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance0, programTexturedPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance0, m_vertViewProjContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance0, vertCylinderNormalTransformContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance0, fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance0, fragCylinderNormalMaterialContainer );

  GeometryInstanceSharedHandle geometryInstance1 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance1, cylinderCut );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance1, programTexturedPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance1, m_vertViewProjContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance1, vertCylinderCutTransformContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance1, fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance1, fragCylinderCutMaterialContainer );

  GeometryInstanceSharedHandle geometryInstance2 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance2, cylinderCutTube );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance2, programTexturedPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance2, m_vertViewProjContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance2, vertCylinderCutTubeTransformContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance2, fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance2, fragCylinderCutTubeMaterialContainer );


  GeometryInstanceSharedHandle geometryInstanceRTT = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstanceRTT, quadRTTScreen );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstanceRTT, programTextureRTTPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstanceRTT, m_vertViewProjContainer );
  m_rix->geometryInstanceUseContainer( geometryInstanceRTT, vertRTTContainer );
  m_rix->geometryInstanceUseContainer( geometryInstanceRTT, m_fragRTTContainer );



  // Render Group

  m_renderGroupScene = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(m_renderGroupScene);
  m_rix->renderGroupAddGeometryInstance( m_renderGroupScene, geometryInstance0 );
  m_rix->renderGroupAddGeometryInstance( m_renderGroupScene, geometryInstance1 );
  m_rix->renderGroupAddGeometryInstance( m_renderGroupScene, geometryInstance2 );

  m_renderGroupRTT = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(m_renderGroupRTT);
  m_rix->renderGroupAddGeometryInstance( m_renderGroupRTT, geometryInstanceRTT );

  // Data

  m_view2Clip = makeFrustum<float>(-0.1f, 0.1f, -0.1f*m_height/m_width, 0.1f*m_height/m_width, 0.1f, 50.0f);
  m_world2View = makeLookAt<float>( Vec3f(0.0f, 3.0f, 4.0f), Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f) );
  m_world2ViewLookBack = makeLookAt<float>( Vec3f(0.0f, 0.0f, 3.0f), Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f) );
  
  Mat44f world2ViewI = m_world2View;
  world2ViewI.invert();
  Mat44f world2ViewILookBack = m_world2View;
  world2ViewILookBack.invert();
  
  Mat44f world2Clip = m_world2View * m_view2Clip;
  Mat44f world2ClipLookBack = m_world2ViewLookBack * m_view2Clip;

  Trafo model2world0;
  model2world0.setTranslation( Vec3f(-3.0f, 0.0f, 0.0f) );
  Mat44f model2world0IT = ~model2world0.getInverse();

  Trafo model2world1;
  model2world1.setTranslation( Vec3f(0.0f, 0.0f, 0.0f) );
  model2world1.setOrientation( Quatf( Vec3f(0.0f, 1.0f, 0.0), 5.0f*PI_QUARTER) );
  Mat44f model2world1IT = ~model2world1.getInverse();

  Trafo model2world2;
  model2world2.setTranslation( Vec3f(3.0f, 0.0f, 0.0f) );
  model2world2.setOrientation( Quatf( Vec3f(0.0f, 1.0f, 0.0), 4.5f*PI_QUARTER) );
  Mat44f model2world2IT = ~model2world2.getInverse();

  Trafo model2worldScreen;
  model2worldScreen.setTranslation( Vec3f(0.0f, 2.0f, 2.0f) );
  model2worldScreen.setOrientation( Quatf( Vec3f(0.0f, 1.0f, 0.0f), PI_QUARTER) );
  Mat44f model2worldScreenIT = ~model2worldScreen.getInverse();


  // Set Container Data

  m_rix->containerSetData( m_vertViewProjContainer,  m_containerEntryView2clip,  ContainerDataRaw( 0, m_view2Clip.getPtr(),   16*sizeof(float) ) );
  m_rix->containerSetData( m_vertViewProjContainer,  m_containerEntryWorld2view, ContainerDataRaw( 0, m_world2View.getPtr(),  16*sizeof(float) ) );
  m_rix->containerSetData( m_vertViewProjContainer,  m_containerEntryView2world, ContainerDataRaw( 0, world2ViewI.getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( m_vertViewProjContainer,  m_containerEntryWorld2clip, ContainerDataRaw( 0, world2Clip.getPtr(),  16*sizeof(float) ) );

  SamplerStateDataCommon samplerStateDataCommon( SSFM_NEAREST, SSFM_NEAREST );
  SamplerStateSharedHandle samplerStateHandle = m_rix->samplerStateCreate(samplerStateDataCommon);
  TextureSharedHandle diffuseMap = dp::rix::util::generateTexture( m_rix, dp::rix::util::createTextureGradient( Vec2ui(128, 128), Vec4f(1.0, 0.0f, 0.0f, 1.0), Vec4f(0.0, 1.0f, 0.0f, 1.0), Vec4f(0.0, 0.0f, 1.0f, 1.0) ), dp::PF_RGBA, dp::DT_UNSIGNED_INT_32, ITF_RGBA8 );

  //Allocate a native gl texture

  TextureSharedHandle textureFBO;
  {
    TextureDescription textureDescription( TT_2D, ITF_RGBA32F, dp::PF_RGBA, dp::DT_FLOAT_32, m_width, m_height );
    textureFBO = m_rix->textureCreate( textureDescription );

    m_colorTexture = dp::gl::Texture2D::create( GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT, m_width, m_height );
    m_depthTexture = dp::gl::Texture2D::create( GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT, GL_FLOAT, m_width, m_height );

    rix::gl::TextureDataGLTexture textureDataGLTexture( m_colorTexture );
    m_rix->textureSetData( textureFBO, textureDataGLTexture );
  }

  rix::core::SamplerSharedHandle samplerDiffuseMap = m_rix->samplerCreate();
  m_rix->samplerSetTexture( samplerDiffuseMap, diffuseMap );
  m_rix->samplerSetSamplerState( samplerDiffuseMap, samplerStateHandle );

  rix::core::SamplerSharedHandle samplerFBO = m_rix->samplerCreate();
  m_rix->samplerSetTexture( samplerFBO, textureFBO );

  m_rix->containerSetData( vertCylinderNormalTransformContainer,   containerEntryModel2world,   ContainerDataRaw( 0, model2world0.getMatrix().getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( vertCylinderNormalTransformContainer,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world0IT.getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( fragCylinderNormalMaterialContainer,    containerEntryTexture,       ContainerDataSampler( samplerDiffuseMap ) );

  m_rix->containerSetData( vertCylinderCutTransformContainer,   containerEntryModel2world,   ContainerDataRaw( 0, model2world1.getMatrix().getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( vertCylinderCutTransformContainer,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world1IT.getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( fragCylinderCutMaterialContainer,    containerEntryTexture,       ContainerDataSampler( samplerDiffuseMap ) );

  m_rix->containerSetData( vertCylinderCutTubeTransformContainer,   containerEntryModel2world,   ContainerDataRaw( 0, model2world2.getMatrix().getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( vertCylinderCutTubeTransformContainer,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world2IT.getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( fragCylinderCutTubeMaterialContainer,    containerEntryTexture,       ContainerDataSampler( samplerDiffuseMap ) );

  m_rix->containerSetData( vertRTTContainer,   containerEntryModel2world,   ContainerDataRaw( 0, model2worldScreen.getMatrix().getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( vertRTTContainer,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2worldScreenIT.getPtr(), 16*sizeof(float) ) );
  m_rix->containerSetData( m_fragRTTContainer, m_containerEntryFBOTexture,  ContainerDataSampler( samplerFBO ) );


  Vec3f lightDirection(-0.5f, 1.0f, 1.0f);

  m_rix->containerSetData( fragConstContainer, containerEntryLightDir, ContainerDataRaw( 0, &lightDirection[0], 3*sizeof(float) ) );

}

bool Feature_FBO_native::onRunInit( unsigned int i )
{
  return true;
}

bool Feature_FBO_native::onRunCheck( unsigned int i )
{
  return i < 2;
}
