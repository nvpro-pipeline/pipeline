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
#include "feature_texture_cube_maps.h"

#include <dp/gl/RenderTarget.h>
#include <dp/util/Array.h>
#include <dp/util/SharedPtr.h>
#include <dp/util/simplexnoise1234.h>
#include <dp/math/Vecnt.h>

#include <test/rix/core/framework/RiXBackend.h>

#include <limits>

using namespace dp;
using namespace rix::core;

//Automatically add the test to the module's global test list
REGISTER_TEST("feature_texture_cube_maps", "tests cube maps for RiX", create_feature_texture_cube_maps);

using namespace math;

const float identity[16] = 
{
  1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 1.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 1.0f, 0.0f, 
  0.0f, 0.0f, 0.0f, 1.0f
};


Feature_texture_cube_maps::Feature_texture_cube_maps()
  : m_renderData(nullptr)
  , m_rix(nullptr)
{

}

Feature_texture_cube_maps::~Feature_texture_cube_maps()
{

}

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

bool Feature_texture_cube_maps::onInit()
{
  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = static_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  dp::util::shared_cast<dp::gl::RenderTarget>( m_displayTarget )->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );

  m_renderData = new test::framework::RenderDataRiX;

  createScene();

  return true;
}

bool Feature_texture_cube_maps::onRun( unsigned int idx )
{
  render(m_renderData, m_displayTarget);

  return true;
}

bool Feature_texture_cube_maps::onClear()
{
  delete m_renderData;

  return true;
}

Vec3f getCubeVector( size_t side, size_t cubesize, size_t x, size_t y)
{
  float s,t,sc,tc;
  s = (x + 0.5f) / (float)cubesize;
  t = (y + 0.5f) / (float)cubesize;
  sc = s*2 - 1;
  tc = t*2 - 1;

  Vec3f vector;

  switch(side){
  case 0:
    vector[0] = 1;
    vector[1] = -tc;
    vector[2] = -sc;
    break;
  case 1:
    vector[0] = -1;
    vector[1] = -tc;
    vector[2] = sc;
    break;
  case 2:
    vector[0] = sc;
    vector[1] = 1;
    vector[2] = tc;
    break;
  case 3:
    vector[0] = sc;
    vector[1] = -1;
    vector[2] = -tc;
    break;
  case 4:
    vector[0] = sc;
    vector[1] = -tc;
    vector[2] = 1;
    break;
  case 5:
    vector[0] = -sc;
    vector[1] = -tc;
    vector[2] = -1;
    break;
  }

  vector.normalize();
  return vector;
}


void Feature_texture_cube_maps::createScene()
{
  const float identity[16] =
  {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  };

  float model2world[16] =
  {
    1.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  1.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  1.0f,  0.0f,
    0.0f,  0.0f,  0.0f,  1.0f
  };

  const size_t numVertices = 14;
  const size_t coordsPerVertex = 3;
  const size_t vertexCoordsSize = numVertices * coordsPerVertex;
  const float vertexCoords[vertexCoordsSize] =
  {
    1.0f, 0.0f, 0.0f,  // 0
    2.0f, 0.0f, 0.0f,  // 1 vertical cross:
    1.0f, 1.0f, 0.0f,  // 2           
    2.0f, 1.0f, 0.0f,  // 3     C D   
    0.0f, 2.0f, 0.0f,  // 4   8 9 A B 
    1.0f, 2.0f, 0.0f,  // 5   4 5 6 7 
    2.0f, 2.0f, 0.0f,  // 6     2 3   
    3.0f, 2.0f, 0.0f,  // 7     0 1   
    0.0f, 3.0f, 0.0f,  // 8           
    1.0f, 3.0f, 0.0f,  // 9           
    2.0f, 3.0f, 0.0f,  // A           
    3.0f, 3.0f, 0.0f,  // B           
    1.0f, 4.0f, 0.0f,  // C           
    2.0f, 4.0f, 0.0f,  // D           
  };

  const size_t texCoordsPerVertex = 4;
  const size_t textureCoordsSize = numVertices * texCoordsPerVertex;
  const float textureCoords[textureCoordsSize] = 
  {
    -1.0f, +1.0f, -1.0f, 0.0f,  // 0
    +1.0f, +1.0f, -1.0f, 0.0f,  // 1 texture coordinates as if the
    -1.0f, -1.0f, -1.0f, 0.0f,  // 2 vertical cross was folded to a cube
    +1.0f, -1.0f, -1.0f, 0.0f,  // 3
    -1.0f, -1.0f, -1.0f, 0.0f,  // 4        08C     1BD 
    -1.0f, -1.0f, +1.0f, 0.0f,  // 5                    
    +1.0f, -1.0f, +1.0f, 0.0f,  // 6    9       A       
    +1.0f, -1.0f, -1.0f, 0.0f,  // 7                    
    -1.0f, +1.0f, -1.0f, 0.0f,  // 8                    
    -1.0f, +1.0f, +1.0f, 0.0f,  // 9        24     37   
    +1.0f, +1.0f, +1.0f, 0.0f,  // A    5       6       
    +1.0f, +1.0f, -1.0f, 0.0f,  // B
    -1.0f, +1.0f, -1.0f, 0.0f,  // C
    +1.0f, +1.0f, -1.0f, 0.0f,  // D
  };

  const size_t indexSetSize = 12*3;
  const unsigned char indexSet[indexSetSize] =
  {
    2,  0,  3,
    3,  0,  1,
    5,  2,  6,
    6,  2,  3,
    8,  4,  9,
    9,  4,  5,
    9,  5, 10,
   10,  5,  6,
   10,  6, 11,
   11,  6,  7,
   12,  9, 13,
   13,  9, 10,
  };

  const char * vertexShader = "//vertex shader\n"
    "#version 330\n"
    "layout(location=0) in vec3 Position;\n"
    "layout(location=8) in vec4 texCoord0;\n"
    "uniform mat4 world2view;\n"
    "uniform mat4 model2world;\n"
    "out vec4 vTexCoords;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = world2view * ( model2world * vec4( Position, 1.0 ) );\n"
    "  vTexCoords = texCoord0;\n"
    "}\n";

  const char * fragmentShaderColor = "" 
    "#version 330\n"
    "uniform samplerCube tex;\n"
    "uniform vec4 color;\n"
    "in vec4 vTexCoords;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = texture(tex, vTexCoords.xyz);\n"
    "}\n";

  BufferSharedHandle vertexCoordBuffer = m_rix->bufferCreate();
  size_t vertexCoordBufferSize = vertexCoordsSize*sizeof(float);
  m_rix->bufferSetSize( vertexCoordBuffer, vertexCoordBufferSize );
  m_rix->bufferUpdateData( vertexCoordBuffer, 0, vertexCoords, vertexCoordBufferSize );

  BufferSharedHandle textureCoordBuffer = m_rix->bufferCreate();
  m_rix->bufferSetSize( textureCoordBuffer, textureCoordsSize*sizeof(float) );
  m_rix->bufferUpdateData( textureCoordBuffer, 0, textureCoords, textureCoordsSize*sizeof(float) );

  BufferSharedHandle indexBuffer = m_rix->bufferCreate();
  m_rix->bufferSetSize( indexBuffer, indexSetSize*sizeof(unsigned char) );
  m_rix->bufferUpdateData( indexBuffer, 0, indexSet, indexSetSize*sizeof(unsigned char) );

  // vertex attributes: 
  // 0: position, stream 0
  // 8: texture coordinate 0, stream 1
  VertexFormatInfo   vertexInfos[] = {
    VertexFormatInfo( 0, dp::util::DT_FLOAT_32, coordsPerVertex, false, 0, 0, coordsPerVertex*sizeof(float)),
    VertexFormatInfo( 8, dp::util::DT_FLOAT_32, texCoordsPerVertex, false, 1, 0, texCoordsPerVertex*sizeof(float)),
  };
  VertexFormatDescription vertexFormatDescription( vertexInfos, sizeof util::array(vertexInfos) );
  VertexFormatSharedHandle vertexFormat = m_rix->vertexFormatCreate( vertexFormatDescription );

  VertexDataSharedHandle vertexData = m_rix->vertexDataCreate();
  m_rix->vertexDataSet( vertexData, 0, vertexCoordBuffer,  0, numVertices ); // vertex stream 0, offset 0
  m_rix->vertexDataSet( vertexData, 1, textureCoordBuffer, 0, numVertices ); // vertex stream 1, offset 0

  VertexAttributesSharedHandle vertexAttributes = m_rix->vertexAttributesCreate();
  m_rix->vertexAttributesSet( vertexAttributes, vertexData, vertexFormat );

  IndicesSharedHandle indices = m_rix->indicesCreate();
  m_rix->indicesSetData( indices, dp::util::DT_UNSIGNED_INT_8, indexBuffer, 0, indexSetSize );

  GeometryDescriptionSharedHandle geometryDescription = m_rix->geometryDescriptionCreate();
  m_rix->geometryDescriptionSet( geometryDescription, GPT_TRIANGLES );

  GeometrySharedHandle geometry = m_rix->geometryCreate();
  m_rix->geometrySetData( geometry, geometryDescription, vertexAttributes, indices );

  std::vector<ProgramParameter> vertexProgramParameters;
  vertexProgramParameters.push_back( ProgramParameter("model2world", CPT_MAT4X4) );
  vertexProgramParameters.push_back( ProgramParameter("world2view", CPT_MAT4X4) );

  std::vector<ProgramParameter> fragmentProgramParameters;
  fragmentProgramParameters.push_back( ProgramParameter("color", CPT_FLOAT4) );
  fragmentProgramParameters.push_back( ProgramParameter("tex",   CPT_SAMPLER) );

  ContainerDescriptorSharedHandle vertexContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &vertexProgramParameters[0],
    (unsigned int)(vertexProgramParameters.size()) ) );
  ContainerDescriptorSharedHandle fragmentContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &fragmentProgramParameters[0],
    (unsigned int)(fragmentProgramParameters.size()) ) );

  ContainerEntry containerEntryModel2World = m_rix->containerDescriptorGetEntry( vertexContainerDescriptor, "model2world" );
  ContainerEntry containerEntryWorld2View  = m_rix->containerDescriptorGetEntry( vertexContainerDescriptor, "world2view" );
  ContainerEntry containerEntryColor       = m_rix->containerDescriptorGetEntry( fragmentContainerDescriptor, "color" );
  ContainerEntry containerEntryTex         = m_rix->containerDescriptorGetEntry( fragmentContainerDescriptor, "tex" );

  std::vector<ContainerDescriptorSharedHandle> containerDescriptorsColor;
  containerDescriptorsColor.push_back( vertexContainerDescriptor );
  containerDescriptorsColor.push_back( fragmentContainerDescriptor );

  const char* shaders[] = {vertexShader, fragmentShaderColor};
  ShaderType  shaderTypes[] = { ST_VERTEX_SHADER, ST_FRAGMENT_SHADER };
  ProgramShaderCode programShaderCode( sizeof util::array( shaders ), shaders, shaderTypes );

  ProgramDescription programDescription( programShaderCode, &containerDescriptorsColor[0], util::checked_cast<unsigned int>(containerDescriptorsColor.size() ));

  ProgramSharedHandle programColor = m_rix->programCreate( programDescription );
  ContainerSharedHandle vertexContainerColor = m_rix->containerCreate( vertexContainerDescriptor );
  ContainerSharedHandle fragmentContainerColor = m_rix->containerCreate( fragmentContainerDescriptor );

  ProgramSharedHandle programs[] = { programColor };
  ProgramPipelineSharedHandle programPipeline = m_rix->programPipelineCreate( programs, sizeof util::array( programs ) );

  GeometryInstanceSharedHandle geometryInstance = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry( geometryInstance, geometry );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance, vertexContainerColor );
  m_rix->geometryInstanceUseContainer( geometryInstance, fragmentContainerColor );

  RenderGroupSharedHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);
  m_rix->renderGroupAddGeometryInstance( renderGroup, geometryInstance );

  float ortho[16];
  getOrthoProjection( ortho, -1.0f, 4.0f, -1.0f, 5.0f, -1.0f, 1.0f );
  float color[4] = { 0.3f, 0.0f, 1.0f, 1.0f };

  m_rix->containerSetData( vertexContainerColor, containerEntryWorld2View, ContainerDataRaw( 0, ortho, 16*sizeof(float) ) );
  m_rix->containerSetData( vertexContainerColor, containerEntryModel2World, ContainerDataRaw( 0, model2world, 16*sizeof(float) ) );
  m_rix->containerSetData( fragmentContainerColor, containerEntryColor, ContainerDataRaw( 0, color, 4*sizeof(float) ) );

  
  size_t texWidth = 128;

  TextureSharedHandle texture = createDebugCubeMap( texWidth );
  //TextureSharedHandle texture = createColorCubeMap( texWidth );
  
  SamplerStateDataCommon samplerStateData( SSFM_LINEAR, SSFM_LINEAR );

  SamplerStateSharedHandle samplerState = m_rix->samplerStateCreate( samplerStateData );
  rix::core::SamplerSharedHandle sampler = m_rix->samplerCreate();
  m_rix->samplerSetTexture( sampler, texture );
  
  m_rix->containerSetData( fragmentContainerColor, containerEntryTex, ContainerDataSampler( sampler ) );

}

TextureSharedHandle Feature_texture_cube_maps::createDebugCubeMap( size_t texWidth , float salt /*=0.0f */ )
{
  // tex dimensions
  const size_t texHeight = texWidth;
  const size_t numLayers = 6;         // number of faces a cube has....please dont change this.

  // components per color
  const size_t texCpc    = 4;

  const size_t texSize = texWidth * texHeight * texCpc;

  std::vector<void const *> tex;
  std::vector<std::vector<unsigned char> > texData;

  tex.resize( numLayers );
  texData.resize( numLayers );

  // bit patterns for  r, c, g, m, b, y
  const char col[6] = { 4, 3, 2, 5, 1, 6 };

  for( size_t layer = 0; layer < numLayers; ++layer )
  {
    texData[layer].resize(texSize);
    tex[layer] = &texData[layer][0];

    for( size_t y = 0; y < texHeight; ++y )
    {
      for( size_t x = 0; x < texWidth; ++x )
      {
        size_t pos = ( y * texWidth + x ) * texCpc;

        Vec3f vec = getCubeVector( layer, texWidth, x, y );

        vec *= 3.0f;

        float intensity = 0.5f * ( dp::util::SimplexNoise1234::noise( vec[0], vec[1], vec[2], salt ) + 1.0f );
        texData[layer][  pos] = (unsigned char)(intensity * 255 * ((col[layer] & 4)/4));
        texData[layer][1+pos] = (unsigned char)(intensity * 255 * ((col[layer] & 2)/2));
        texData[layer][2+pos] = (unsigned char)(intensity * 255 * ((col[layer] & 1)/1));
        texData[layer][3+pos] = 255;
      }
    }
  }

  TextureDescription textureDescription( TT_CUBEMAP, ITF_RGBA8, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8, texWidth, texWidth );

  TextureSharedHandle texture = m_rix->textureCreate( textureDescription );

  TextureDataPtr textureDataPtr( &tex[0], 0, 6, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8 );

  m_rix->textureSetData( texture, textureDataPtr );

  return texture;
}

TextureSharedHandle Feature_texture_cube_maps::createColorCubeMap( size_t texWidth , float salt /*=0.0f*/, float r/*=1.0f*/, float g/*=1.0f*/, float b/*=1.0f */ )
{
  // tex dimensions
  const size_t texHeight = texWidth;
  const size_t numLayers = 6;         // number of faces a cube has....please dont change this.

  // components per color
  const size_t texCpc    = 4;

  const size_t texSize = texWidth * texHeight * texCpc;

  std::vector<void const *> tex;
  std::vector<std::vector<unsigned char> > texData;

  tex.resize( numLayers );
  texData.resize( numLayers );
  tex.resize( numLayers );
  texData.resize( numLayers );
  for( size_t layer = 0; layer < numLayers; ++layer )
  {
    texData[layer].resize(texSize);
    tex[layer] = &texData[layer][0];

    for( size_t y = 0; y < texHeight; ++y )
    {
      for( size_t x = 0; x < texWidth; ++x )
      {
        size_t pos = ( y * texWidth + x ) * texCpc;

        Vec3f vec = getCubeVector( layer, texWidth, x, y );

        vec *= 3.0f;

        texData[layer][  pos] = (unsigned char)(r * 255 * 0.5f * ( dp::util::SimplexNoise1234::noise( vec[0], vec[1], vec[2], 1.0f + salt ) + 1.0f ));
        texData[layer][1+pos] = (unsigned char)(g * 255 * 0.5f * ( dp::util::SimplexNoise1234::noise( vec[0], vec[1], vec[2], 2.0f + salt ) + 1.0f ));
        texData[layer][2+pos] = (unsigned char)(b * 255 * 0.5f * ( dp::util::SimplexNoise1234::noise( vec[0], vec[1], vec[2], 3.0f + salt ) + 1.0f ));
        texData[layer][3+pos] = 255;
      }
    }
  }

  TextureDescription textureDescription( TT_CUBEMAP, ITF_RGBA8, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8, texWidth, texWidth );

  TextureSharedHandle texture = m_rix->textureCreate( textureDescription );

  TextureDataPtr textureDataPtr( &tex[0], 0, 6, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8 );

  m_rix->textureSetData( texture, textureDataPtr );

  return texture;
}
