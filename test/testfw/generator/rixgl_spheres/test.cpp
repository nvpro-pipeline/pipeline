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

#include <dp/util/Image.h>
#include <dp/util/Types.h>
#include <dp/math/Trafo.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>

#include <limits>

//Automatically add the test to the module's global test list
REGISTER_TEST("##test##", "##description##", create_##test##);

using namespace dp;
using namespace rix::core;
using namespace util::generator;

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
  const size_t numVertices = 3;
  const size_t coordsPerVertex = 3;
  const size_t vertexCoordsSize = numVertices * coordsPerVertex;
  const float vertexCoords[vertexCoordsSize] =
  {
    0.0f, 0.0f, -1.0f,
    1.0f, 0.0f, -1.0f,
    0.0f, 1.0f, -1.0f
  };

  const size_t indexSetSize = 1 * 3; //numTriangles * indicesPerTriangle
  const unsigned char indexSet[indexSetSize] =
  {
    0, 1, 2
  };

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

  const char * fragmentShader = "" 
    "#version 400\n"
    "uniform vec4 color;\n"
    "uniform sampler2D bumpTex;\n"
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
    "  vec3 emissiveColor = vec3(0.0, 0.0, 0.0);\n"
    "  vec3 ambientColor  = vec3(0.0, 0.0, 0.0);\n"
    "  vec3 diffuseColor  = color.xyz;\n"
    "  vec3 specularColor = vec3(1.0, 1.0, 1.0);\n"
    "  float ambient       = 0.0;\n"
    "  float diffuse       = 1.0;\n"
    "  float specular      = 1.0;\n"
    "  float exponent      = 16.0;\n\n"

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

  //Geometry

  SmartGeometryData sphereData = createSphere( ATTRIB_POSITION | ATTRIB_NORMAL
    , 64, 64 );
  SmartGeometryHandle sphere = rix::util::generateGeometry(sphereData, m_rix);

  SmartGeometryData cylinderData = createCylinder( ATTRIB_POSITION | ATTRIB_NORMAL
    , 64 );
  SmartGeometryHandle cylinder = rix::util::generateGeometry(cylinderData, m_rix);

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
    ProgramParameter("color", CPT_FLOAT4) 
  };


  ProgramParameter fragmentConstProgramParameters[] = {
    ProgramParameter("lightDir", CPT_FLOAT3)
  };

  SmartContainerDescriptorHandle vertConstContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( vertexConstProgramParameters, 
    sizeof testfw::core::array(vertexConstProgramParameters) ) );

  SmartContainerDescriptorHandle vertVarContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( vertexVarProgramParameters, 
    sizeof testfw::core::array(vertexVarProgramParameters) ) );

  SmartContainerDescriptorHandle fragContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( fragmentProgramParameters,
    sizeof testfw::core::array(fragmentProgramParameters) ) );

  SmartContainerDescriptorHandle fragConstContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( fragmentConstProgramParameters,
    sizeof testfw::core::array(fragmentConstProgramParameters) ) );


  // Program

  ProgramShaderCode vertShader( vertexShader, ST_VERTEX_SHADER );
  ProgramShaderCode fragShader( fragmentShader, ST_FRAGMENT_SHADER );

  ContainerDescriptorHandle vertContainerDescriptors[] = { vertConstContainerDescriptor, vertVarContainerDescriptor };
  ContainerDescriptorHandle fragContainerDescriptors[] = { fragContainerDescriptor, fragConstContainerDescriptor  };

  ProgramDescription vertProgramDescription( vertShader, vertContainerDescriptors, sizeof testfw::core::array(vertContainerDescriptors) );
  ProgramDescription fragProgramDescription( fragShader, fragContainerDescriptors, sizeof testfw::core::array(fragContainerDescriptors) );

  SmartProgramHandle vertProgram = m_rix->programCreate( vertProgramDescription );
  SmartProgramHandle fragProgram = m_rix->programCreate( fragProgramDescription );


  // Program Pipeline

  ProgramHandle programs[] = {vertProgram, fragProgram};
  SmartProgramPipelineHandle programPipeline = m_rix->programPipelineCreate( programs, sizeof testfw::core::array(programs) );


  // Containers

  SmartContainerHandle vertViewProjContainer          = m_rix->containerCreate( vertConstContainerDescriptor );
  SmartContainerHandle vertSphere1TransformContainer  = m_rix->containerCreate( vertVarContainerDescriptor );
  SmartContainerHandle vertCylinderTransformContainer = m_rix->containerCreate( vertVarContainerDescriptor );
  SmartContainerHandle vertSphere2TransformContainer  = m_rix->containerCreate( vertVarContainerDescriptor );
  SmartContainerHandle fragConstContainer             = m_rix->containerCreate( fragConstContainerDescriptor );
  SmartContainerHandle fragSphere1MaterialContainer   = m_rix->containerCreate( fragContainerDescriptor );
  SmartContainerHandle fragCylinderMaterialContainer  = m_rix->containerCreate( fragContainerDescriptor );
  SmartContainerHandle fragSphere2MaterialContainer   = m_rix->containerCreate( fragContainerDescriptor );

  // Container Data

  math::Mat44f view2Clip = math::makeFrustum<float>(-0.1f, 0.1f, -0.1f * m_height / m_width, 0.1f * m_height / m_width, 0.1f, 50.0f);
  math::Mat44f world2View = math::makeLookAt<float>( math::Vec3f(0.0f, 3.0f, 4.0f), math::Vec3f(0.0f, 0.0f, 0.0f), math::Vec3f(0.0f, 1.0f, 0.0f) );
  math::Mat44f world2ViewI = world2View;
  world2ViewI.invert();
  math::Mat44f world2Clip = world2View * view2Clip;

  math::Trafo model2world0;
  model2world0.setTranslation( math::Vec3f(-3.0f, 0.0f, 0.0f) );
  math::Mat44f model2world0IT = ~model2world0.getInverse();

  math::Trafo model2world1;
  model2world1.setTranslation( math::Vec3f(0.0f, 0.0f, 0.0f) );
  math::Mat44f model2world1IT = ~model2world1.getInverse();

  math::Trafo model2world2;
  model2world2.setTranslation( math::Vec3f(3.0f, 0.0f, 0.0f) );
  math::Mat44f model2world2IT = ~model2world2.getInverse();

  float bluish[4] = { 0.3f, 0.0f, 1.0f, 1.0f };
  float redish[4] = { 1.0f, 0.0f, 0.3f, 1.0f };
  float greenish[4] = { 0.3f, 1.0f, 0.3f, 1.0f };


  // Geometry Instances

  SmartGeometryInstanceHandle geometryInstance0 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance0, sphere );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance0, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance0, vertViewProjContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance0, vertSphere1TransformContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance0, fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance0, fragSphere1MaterialContainer );

  SmartGeometryInstanceHandle geometryInstance1 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance1, cylinder );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance1, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance1, vertViewProjContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance1, vertCylinderTransformContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance1, fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance1, fragCylinderMaterialContainer );

  SmartGeometryInstanceHandle geometryInstance2 = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance2, sphere );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance2, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance2, vertViewProjContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance2, vertSphere2TransformContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance2, fragConstContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance2, fragSphere2MaterialContainer );



  // Render Group

  SmartRenderGroupHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance0 );
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance1 );
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance2 );


  // Get container entries

  ContainerEntry containerEntryView2clip   = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "view2clip" );
  ContainerEntry containerEntryWorld2view  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "world2view" );
  ContainerEntry containerEntryView2world  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "view2world" );
  ContainerEntry containerEntryWorld2clip  = m_rix->containerDescriptorGetEntry( vertConstContainerDescriptor, "world2clip" );

  ContainerEntry containerEntryModel2world = m_rix->containerDescriptorGetEntry( vertVarContainerDescriptor, "model2world" );
  ContainerEntry containerEntryModel2worldIT = m_rix->containerDescriptorGetEntry( vertVarContainerDescriptor, "model2worldIT" );

  ContainerEntry containerEntryColor       = m_rix->containerDescriptorGetEntry( fragContainerDescriptor, "color" );
  ContainerEntry containerEntryLightDir    = m_rix->containerDescriptorGetEntry( fragConstContainerDescriptor, "lightDir" );


  // Set Container Data

  m_rix->containerSetData( vertViewProjContainer,  containerEntryView2clip,  ContainerDataRaw( 0, view2Clip.getPtr(),   16 * sizeof(float) ) );
  m_rix->containerSetData( vertViewProjContainer,  containerEntryWorld2view, ContainerDataRaw( 0, world2View.getPtr(),  16 * sizeof(float) ) );
  m_rix->containerSetData( vertViewProjContainer,  containerEntryView2world, ContainerDataRaw( 0, world2ViewI.getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( vertViewProjContainer,  containerEntryWorld2clip, ContainerDataRaw( 0, world2Clip.getPtr(),  16 * sizeof(float) ) );


  m_rix->containerSetData( vertSphere1TransformContainer,   containerEntryModel2world,   ContainerDataRaw( 0, model2world0.getMatrix().getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( vertSphere1TransformContainer,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world0IT.getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( fragSphere1MaterialContainer,    containerEntryColor,         ContainerDataRaw( 0, redish, 4 * sizeof(float) ) );

  m_rix->containerSetData( vertCylinderTransformContainer,   containerEntryModel2world,   ContainerDataRaw( 0, model2world1.getMatrix().getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( vertCylinderTransformContainer,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world1IT.getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( fragCylinderMaterialContainer,    containerEntryColor,         ContainerDataRaw( 0, greenish, 4 * sizeof(float) ) );

  m_rix->containerSetData( vertSphere2TransformContainer,   containerEntryModel2world,   ContainerDataRaw( 0, model2world2.getMatrix().getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( vertSphere2TransformContainer,   containerEntryModel2worldIT, ContainerDataRaw( 0, model2world2IT.getPtr(), 16 * sizeof(float) ) );
  m_rix->containerSetData( fragSphere2MaterialContainer,    containerEntryColor,         ContainerDataRaw( 0, bluish, 4 * sizeof(float) ) );


  math::Vec3f lightDirection(-0.5f, 1.0f, 1.0f);

  m_rix->containerSetData( fragConstContainer, containerEntryLightDir, ContainerDataRaw( 0, &lightDirection[0], 3 * sizeof(float) ) );

}
