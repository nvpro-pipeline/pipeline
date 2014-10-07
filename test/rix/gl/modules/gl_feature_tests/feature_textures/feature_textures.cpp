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
#include "feature_textures.h"

#include <dp/util/Image.h>
#include <dp/util/SharedPtr.h>
#include <dp/util/Types.h>

#include <test/rix/core/framework/RiXBackend.h>

#include <limits>

using namespace dp::util;
using namespace dp::rix::core;

//Automatically add the test to the module's global test list
REGISTER_TEST("feature_textures", "tests RiX texture types", create_feature_textures);

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

Feature_textures::Feature_textures()
  : m_vertexAttributes(nullptr)
  , m_renderData(nullptr)
  , m_rix(nullptr)
{
}

Feature_textures::~Feature_textures()
{
}

bool Feature_textures::onInit()
{
  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = static_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  dp::util::shared_cast<dp::gl::RenderTarget>( m_displayTarget )->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );

  m_renderData = new test::framework::RenderDataRiX;

  createScene();

  return true;
}

bool Feature_textures::onRun( unsigned int idx )
{
  render(m_renderData, m_displayTarget);

  return true;  
}

bool Feature_textures::onClear()
{
  delete m_renderData;

  return true;
}

void Feature_textures::createScene()
{
  //
  // prepare data
  //
  const size_t numVertices = 8;
  const size_t coordsPerVertex = 3;
  const size_t vertexCoordsSize = numVertices * coordsPerVertex;
  const float vertexCoords[vertexCoordsSize] = 
  {
    0.0f,  0.0f,  0.0f,    //                    
    1.0f,  0.0f,  0.0f,    //       7       6    
    1.0f,  1.0f,  0.0f,    //    3       2       
    0.0f,  1.0f,  0.0f,    //                    
    0.0f,  0.0f, -1.0f,    //                    
    1.0f,  0.0f, -1.0f,    //       4       5    
    1.0f,  1.0f, -1.0f,    //    0       1       
    0.0f,  1.0f, -1.0f,    //                    
  };

  const size_t texCoordsPerVertex = 4;
  const size_t textureCoordsSize = numVertices * texCoordsPerVertex;
  const float textureCoords[textureCoordsSize] = 
  {
    0.0f,  0.0f,  0.0f,  1.0f,
    1.0f,  0.0f,  0.0f,  1.0f,
    1.0f,  1.0f,  0.0f,  1.0f,
    0.0f,  1.0f,  0.0f,  1.0f,
    0.0f,  0.0f,  1.0f,  1.0f,
    1.0f,  0.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,  1.0f,
    0.0f,  1.0f,  1.0f,  1.0f
  };

  // bytes per color/component
  const size_t bpc = 4;

  // general 3D texture data
  const size_t texWidth  = 16;
  const size_t texHeight = 16;
  const size_t texDepth  = 16;
  const size_t texSize   = texWidth * texHeight * texDepth * bpc;
  unsigned char tex[texSize];
  for( size_t z = 0; z < texDepth; ++z )
  {
    for( size_t y = 0; y < texHeight; ++y )
    {
      for( size_t x = 0; x < texWidth; ++x )
      {
        size_t pos = (( z * texHeight + y ) * texWidth + x) * bpc;
        /*
        tex[  pos] = (unsigned char)( 255 * (float)x / (float)(texWidth-1) );
        tex[1+pos] = (unsigned char)( 255 * (float)y / (float)(texHeight-1) );
        tex[2+pos] = (unsigned char)( 255 * (float)z / (float)(texDepth-1) );
        tex[3+pos] = (unsigned char)( 255 );
        */
        tex[  pos] = 255 * (x & 1);
        tex[1+pos] = 255 * (y & 1);
        tex[2+pos] = 255 * (z & 1);
        tex[3+pos] = 255;
        
      }
    }
  }

  // special 1D array for 1D texture arrays
  const size_t tex1DArraySize = 2;
  void * tex1DArray[tex1DArraySize];
  for( size_t l = 0; l < tex1DArraySize; ++l )
  {
    tex1DArray[l] = &tex[ texWidth * bpc * l ];
  }

  // special 2D array for 2D texture arrays
  const size_t tex2DArraySize = 2;
  void * tex2DArray[tex2DArraySize];
  for( size_t l = 0; l < tex2DArraySize; ++l )
  {
    tex2DArray[l] = &tex[texWidth * texHeight * bpc * l];
  }

  const char * vertexShader = "//vertex shader\n"
    "#version 330\n"
    "layout(location=0) in vec3 Position;\n"
    "layout(location=8) in vec4 texCoord0;\n"
    "uniform mat4 world2view;\n"
    "uniform mat4 model2world;\n"
    "out vec4 vTexCoords;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = world2view * model2world * vec4( Position, 1.0 );\n"
    "  vTexCoords = texCoord0;\n"
    "}\n";

  const char * fragmentShaderSampler1D = "//sampler1D\n"
    "#version 330\n"
    "uniform sampler1D tex;\n"
    "in vec4 vTexCoords;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = texture( tex, vTexCoords.x );\n"
    "}\n";

  const char * fragmentShaderSampler1DArray = "//sampler1DArray\n"
    "#version 330\n"
    "uniform sampler1DArray tex;\n"
    "in vec4 vTexCoords;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = texture( tex, vTexCoords.xz );\n"
    "}\n";

  const char * fragmentShaderSampler2D = "//sampler2D\n"
    "#version 330\n"
    "uniform sampler2D tex;\n"
    "in vec4 vTexCoords;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = texture( tex, vTexCoords.xy );\n"
    "}\n";

  const char * fragmentShaderSampler2DRect = "//sampler2DRect\n"
    "#version 330\n"
    "uniform sampler2DRect tex;\n"
    "in vec4 vTexCoords;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = texture( tex, vTexCoords.xy * textureSize(tex) );\n"
    "}\n";
  
  const char * fragmentShaderSampler2DArray = "//sampler2DArray\n"
    "#version 330\n"
    "uniform sampler2DArray tex;\n"
    "in vec4 vTexCoords;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = texture( tex, vTexCoords.xyz );\n"
    "}\n";

  const char * fragmentShaderSampler3D = "//sampler3D\n"
    "#version 330\n"
    "uniform sampler3D tex;\n"
    "in vec4 vTexCoords;\n"
    "layout(location = 0, index = 0) out vec4 Color;\n"
    "void main()\n"
    "{\n"
    "  Color = texture( tex, vTexCoords.xyz );\n"
    "}\n";

  //
  // prepare render API objects
  //
  // 
  // vertex data (common for all geometry)
  //
  BufferSharedHandle vertexCoordBuffer = m_rix->bufferCreate();
  size_t vertexCoordBufferSize = vertexCoordsSize*sizeof(float);
  m_rix->bufferSetSize(vertexCoordBuffer, vertexCoordBufferSize);
  m_rix->bufferUpdateData( vertexCoordBuffer, 0, vertexCoords, vertexCoordBufferSize );

  BufferSharedHandle textureCoordBuffer = m_rix->bufferCreate();
  size_t textureCoordBufferSize = textureCoordsSize*sizeof(float);
  m_rix->bufferSetSize(textureCoordBuffer, textureCoordBufferSize);
  m_rix->bufferUpdateData( textureCoordBuffer, 0, textureCoords, textureCoordBufferSize );
  
  // vertex attributes: 
  // 0: position, stream 0
  // 8: texture coordinate 0, stream 1
  VertexFormatInfo   vertexInfos[] = {
    VertexFormatInfo( 0, dp::util::DT_FLOAT_32, coordsPerVertex, false, 0, 0, coordsPerVertex*sizeof(float)),
    VertexFormatInfo( 8, dp::util::DT_FLOAT_32, texCoordsPerVertex, false, 1, 0, texCoordsPerVertex*sizeof(float)),
  };
  VertexFormatDescription vertexFormatDescription( vertexInfos, sizeof dp::util::array(vertexInfos) );
  VertexFormatSharedHandle vertexFormat = m_rix->vertexFormatCreate( vertexFormatDescription );

  VertexDataSharedHandle vertexData = m_rix->vertexDataCreate();
  m_rix->vertexDataSet( vertexData, 0, vertexCoordBuffer,  0, numVertices ); // vertex stream 0, offset 0
  m_rix->vertexDataSet( vertexData, 1, textureCoordBuffer, 0, numVertices ); // vertex stream 1, offset 0

  m_vertexAttributes = m_rix->vertexAttributesCreate();
  m_rix->vertexAttributesSet( m_vertexAttributes, vertexData, vertexFormat );

  std::vector<ProgramParameter> programParameterWorld2View;
  programParameterWorld2View.push_back( ProgramParameter("world2view", CPT_MAT4X4) );

  m_containerDescriptorWorld2View = m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &programParameterWorld2View[0], 1 ) );

  ContainerEntry containerEntryWorld2View  = m_rix->containerDescriptorGetEntry( m_containerDescriptorWorld2View, "world2view" );

  m_vertexContainerW2V = m_rix->containerCreate( m_containerDescriptorWorld2View );
  
  InternalTextureFormat itf = ITF_RGBA8;
  TextureDescription textureDescription1D     ( TT_1D,           itf, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8, texWidth );
  TextureDescription textureDescription2D     ( TT_2D,           itf, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8, texWidth, texHeight );
  TextureDescription textureDescription3D     ( TT_3D,           itf, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8, texWidth, texHeight, texDepth );
  TextureDescription textureDescription1DArray( TT_1D_ARRAY,     itf, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8, texWidth, 0        , 0,        tex1DArraySize );
  TextureDescription textureDescription2DArray( TT_2D_ARRAY,     itf, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8, texWidth, texHeight, 0,        tex2DArraySize );
  TextureDescription textureDescription2DRect ( TT_2D_RECTANGLE, itf, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8, std::min<size_t>(texWidth, 5), std::min<size_t>(texHeight, 7) );

  // just pass in texture data
  TextureDataPtr textureData( tex, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8 );

  // use a vector of pointers into texture data
  TextureDataPtr textureData1DArray( tex1DArray, 0, tex1DArraySize, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8 );
  TextureDataPtr textureData2DArray( tex2DArray, 0, tex2DArraySize, dp::util::PF_RGBA, dp::util::DT_UNSIGNED_INT_8 );

  TextureSharedHandle texture1D      = m_rix->textureCreate( textureDescription1D );
  TextureSharedHandle texture2D      = m_rix->textureCreate( textureDescription2D );
  TextureSharedHandle texture3D      = m_rix->textureCreate( textureDescription3D );
  TextureSharedHandle texture1DArray = m_rix->textureCreate( textureDescription1DArray );
  TextureSharedHandle texture2DArray = m_rix->textureCreate( textureDescription2DArray );
  TextureSharedHandle texture2DRect  = m_rix->textureCreate( textureDescription2DRect );

  m_rix->textureSetData( texture1D, textureData );
  m_rix->textureSetData( texture2D, textureData );
  m_rix->textureSetData( texture3D, textureData );
  m_rix->textureSetData( texture1DArray, textureData1DArray );
  m_rix->textureSetData( texture2DArray, textureData2DArray );
  m_rix->textureSetData( texture2DRect, textureData );

  RenderGroupSharedHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);

  // prepare & set world2view matrix
  float ortho[16];
  getOrthoProjection( ortho, -2.0f, 2.0f, -2.0f, 2.0f, -2.0f, 2.0f );
  m_rix->containerSetData( m_vertexContainerW2V, containerEntryWorld2View, ContainerDataRaw( 0, ortho, 16*sizeof(float) ) );

  const size_t indexSetQuadSize = 2*3;
  const unsigned int indexSetQuad[indexSetQuadSize] = { 0, 1, 3,  7, 5, 6 };

  const size_t indexSetTri3DSize = 2*3;
  const unsigned int indexSetTri3D[indexSetTri3DSize] = { 0, 5, 7,  7, 5, 2 };

  generateGI( vertexShader, fragmentShaderSampler1D, CPT_SAMPLER, texture1D,
    0.9f, -2.0f, 1.0f, 0.0f, indexSetQuad, indexSetQuadSize );

  generateGI( vertexShader, fragmentShaderSampler2D, CPT_SAMPLER, texture2D,
    0.9f, -1.0f, 1.0f, 0.0f,  indexSetQuad, indexSetQuadSize );

  generateGI( vertexShader, fragmentShaderSampler2DRect, CPT_SAMPLER, texture2DRect,
    0.9f, 0.0f, 1.0f, 0.0f, indexSetQuad, indexSetQuadSize );

  generateGI( vertexShader, fragmentShaderSampler2DArray, CPT_SAMPLER, texture2DArray,
    0.9f, 1.0f, 1.0f, 0.0f, indexSetQuad, indexSetQuadSize );

  generateGI( vertexShader, fragmentShaderSampler1DArray, CPT_SAMPLER, texture1DArray,
    0.9f, -2.0f, 0.0f, 0.0f, indexSetQuad, indexSetQuadSize );

  generateGI( vertexShader, fragmentShaderSampler3D, CPT_SAMPLER, texture3D,
    0.9f, -1.0f, 0.0f, 0.0f, indexSetTri3D, indexSetTri3DSize );

#if 0
  //
  //
  // BUFFER ADDRESS TEST BEGIN
  //
  // 
  {

    const char * fragmentShaderBufferAddress = "//buffer address\n"
      "#version 330\n"
      "#extension GL_NV_shader_buffer_load : enable\n"
      "uniform vec4 *tex;\n"
      "in vec4 vTexCoords;\n"
      "layout(location = 0, index = 0) out vec4 Color;\n"
      "void main()\n"
      "{\n"
      "  Color = tex[(int)(vTexCoords.x*16*1024*1024)];\n"
      "}\n";

    const size_t numEntries = 16*1024*1024;
    float* bufData = new float[ numEntries * 4 ];
    GLuint64 bufAddress;
    m_buf = dp::gl::Buffer::create();

    for( unsigned int i=0; i<numEntries; ++i )
    {
      const size_t pos = i*4;
      bufData[  pos] = (float)i/(float)(numEntries-1);
      bufData[1+pos] = 0.0f;
      bufData[2+pos] = (float)(numEntries-1 - i)/(float)(numEntries-1);
      bufData[3+pos] = 1.0f;
    }

    glNamedBufferDataEXT( m_buf->getGLId(), numEntries*sizeof(float)*4, bufData, GL_STATIC_DRAW );

    glMakeNamedBufferResidentNV( m_buf->getGLId(), GL_READ_ONLY );

    glGetNamedBufferParameterui64vNV( m_buf->getGLId(), GL_BUFFER_GPU_ADDRESS_NV, &bufAddress );


    const size_t indexSetQuadSize = 2*3;
    const unsigned int indexSetQuad[indexSetQuadSize] = { 0, 1, 3,  7, 5, 6 };
    // index data
    BufferSharedHandle indexBuffer = m_rix->bufferCreate();
    m_rix->bufferUpdateData( indexBuffer, 0, indexSetQuad, indexSetQuadSize*sizeof(unsigned int) );

    IndicesSharedHandle indices = m_rix->indicesCreate();
    m_rix->indicesSetData( indices, dp::util::DT_UNSIGNED_INT_32, indexBuffer, 0, indexSetQuadSize );

    // geometry
    GeometryDescriptionSharedHandle geometryDescription = m_rix->geometryDescriptionCreate();
    m_rix->geometryDescriptionSet( geometryDescription, GPT_TRIANGLES );

    GeometrySharedHandle geometry = m_rix->geometryCreate();
    m_rix->geometrySetData( geometry, geometryDescription, m_vertexAttributes, indices );

    // vertex shader parameter model2world
    std::vector<ProgramParameter> vertexProgramParameters;
    vertexProgramParameters.push_back( ProgramParameter("model2world", CPT_MAT4X4) );

    ContainerDescriptorSharedHandle vertexContainerDescriptor =
      m_rix->containerDescriptorCreate( &vertexProgramParameters[0],
      checked_cast<unsigned int>(vertexProgramParameters.size()) );

    ContainerEntry containerEntryModel2World = m_rix->containerDescriptorGetEntry( vertexContainerDescriptor, "model2world" );

    // fragment shader parameter tex
    std::vector<ProgramParameter> fragmentProgramParameters;
    fragmentProgramParameters.push_back( ProgramParameter("tex", CPT_BUFFER_ADDRESS) );

    ContainerDescriptorSharedHandle fragmentContainerDescriptor = 
      m_rix->containerDescriptorCreate( &fragmentProgramParameters[0],
      checked_cast<unsigned int>(fragmentProgramParameters.size()) );

    ContainerEntry containerEntryTex = m_rix->containerDescriptorGetEntry( fragmentContainerDescriptor, "tex" );

    std::vector<ContainerDescriptorHandle> containerDescriptors;
    containerDescriptors.push_back( m_containerDescriptorWorld2View );
    containerDescriptors.push_back( vertexContainerDescriptor );
    containerDescriptors.push_back( fragmentContainerDescriptor );

    std::vector<ProgramShaderCode> shaders;
    shaders.push_back( ProgramShader( vertexShader, GL_VERTEX_SHADER ) );
    shaders.push_back( ProgramShader( fragmentShaderBufferAddress, GL_FRAGMENT_SHADER ) );

    GL::ProgramDescriptionGL programDescription;
    programDescription.m_shaders        = &shaders[0];
    programDescription.m_numShaders     = checked_cast<unsigned int>(shaders.size());
    programDescription.m_descriptors    = &containerDescriptors[0];
    programDescription.m_numDescriptors = checked_cast<unsigned int>(containerDescriptors.size());
    ProgramSharedHandle program = m_rix->programCreate( programDescription );

    ContainerSharedHandle vertexContainer  = m_rix->containerCreate( vertexContainerDescriptor );
    ContainerSharedHandle fragmentContainer = m_rix->containerCreate( fragmentContainerDescriptor );

    GeometryInstanceSharedHandle geometryInstance = m_rix->geometryInstanceCreate();
    m_rix->geometryInstanceSetGeometry(  geometryInstance, geometry );
    m_rix->geometryInstanceSetProgram(   geometryInstance, program );
    m_rix->geometryInstanceUseContainer( geometryInstance, m_vertexContainerW2V );
    m_rix->geometryInstanceUseContainer( geometryInstance, vertexContainer );
    m_rix->geometryInstanceUseContainer( geometryInstance, fragmentContainer );

    m_rix->renderGroupAddGeometryInstance( m_renderGroup, geometryInstance );

    // prepare & set model2world matrix
    float mat[16];
    getTransformMatrix( mat, 3.9f, 0.9f, 1.0f, -1.95f, -1.0f, 0.0f );
    m_rix->containerSetData( vertexContainer, containerEntryModel2World, 0, mat, 16*sizeof(float) );

    // prepare & set texture
    m_rix->containerSetData( fragmentContainer, containerEntryTex, 0, &bufAddress, sizeof(TextureHandle) );

  }
  //
  //
  // BUFFER ADDRESS TEST END
  //
  // 
  // 
#endif
}

void Feature_textures::generateGI
  (
  const char * vertexShader,
  const char * fragmentShader,
  ContainerParameterType samplerType,
  TextureSharedHandle texture,
  const float scale, const float transX, const float transY, const float transZ,
  const unsigned int * indexSet,
  const size_t indexSetSize
  )
{
  // index data
  BufferSharedHandle indexBuffer = m_rix->bufferCreate();
  size_t indexBufferSize = indexSetSize*sizeof(unsigned int);
  m_rix->bufferSetSize(indexBuffer, indexBufferSize);
  m_rix->bufferUpdateData( indexBuffer, 0, indexSet, indexBufferSize );

  IndicesSharedHandle indices = m_rix->indicesCreate();
  m_rix->indicesSetData( indices, dp::util::DT_UNSIGNED_INT_32, indexBuffer, 0, indexSetSize );

  // geometry
  GeometryDescriptionSharedHandle geometryDescription = m_rix->geometryDescriptionCreate();
  m_rix->geometryDescriptionSet( geometryDescription, GPT_TRIANGLES );

  GeometrySharedHandle geometry = m_rix->geometryCreate();
  m_rix->geometrySetData( geometry, geometryDescription, m_vertexAttributes, indices );

  // vertex shader parameter model2world
  std::vector<ProgramParameter> vertexProgramParameters;
  vertexProgramParameters.push_back( ProgramParameter("model2world", CPT_MAT4X4) );

  ContainerDescriptorSharedHandle vertexContainerDescriptor =
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &vertexProgramParameters[0],
    checked_cast<unsigned int>(vertexProgramParameters.size()) ) );

  ContainerEntry containerEntryModel2World = m_rix->containerDescriptorGetEntry( vertexContainerDescriptor, "model2world" );

  // fragment shader parameter tex
  std::vector<ProgramParameter> fragmentProgramParameters;
  fragmentProgramParameters.push_back( ProgramParameter("tex", samplerType) );

  ContainerDescriptorSharedHandle fragmentContainerDescriptor = 
    m_rix->containerDescriptorCreate( ProgramParameterDescriptorCommon( &fragmentProgramParameters[0],
    checked_cast<unsigned int>(fragmentProgramParameters.size()) ) );

  ContainerEntry containerEntryTex = m_rix->containerDescriptorGetEntry( fragmentContainerDescriptor, "tex" );

  std::vector<ContainerDescriptorSharedHandle> containerDescriptors;
  containerDescriptors.push_back( m_containerDescriptorWorld2View );
  containerDescriptors.push_back( vertexContainerDescriptor );
  containerDescriptors.push_back( fragmentContainerDescriptor );


  const char* shaders[] = {vertexShader, fragmentShader};
  ShaderType  shaderTypes[] = { ST_VERTEX_SHADER, ST_FRAGMENT_SHADER };
  ProgramShaderCode programShaderCode( sizeof dp::util::array( shaders ), shaders, shaderTypes );

  ProgramDescription programDescription( programShaderCode, &containerDescriptors[0], checked_cast<unsigned int>(containerDescriptors.size() ));
  
  ProgramSharedHandle program = m_rix->programCreate( programDescription );

  ContainerSharedHandle vertexContainer  = m_rix->containerCreate( vertexContainerDescriptor );
  ContainerSharedHandle fragmentContainer = m_rix->containerCreate( fragmentContainerDescriptor );

  ProgramSharedHandle programs[] = { program };
  ProgramPipelineSharedHandle programPipeline = m_rix->programPipelineCreate( programs, sizeof dp::util::array( programs ) );

  GeometryInstanceSharedHandle geometryInstance = m_rix->geometryInstanceCreate();
  m_rix->geometryInstanceSetGeometry(  geometryInstance, geometry );
  m_rix->geometryInstanceSetProgramPipeline( geometryInstance, programPipeline );
  m_rix->geometryInstanceUseContainer( geometryInstance, m_vertexContainerW2V );
  m_rix->geometryInstanceUseContainer( geometryInstance, vertexContainer );
  m_rix->geometryInstanceUseContainer( geometryInstance, fragmentContainer );

  m_rix->renderGroupAddGeometryInstance( m_renderData->getRenderGroup(), geometryInstance );

  // prepare & set model2world matrix
  float mat[16];
  getTransformMatrix( mat, scale, scale, scale, transX, transY, transZ );
  m_rix->containerSetData( vertexContainer, containerEntryModel2World, ContainerDataRaw( 0, mat, 16*sizeof(float) ) );

  // prepare & set sampler & texture
  SamplerStateDataCommon samplerStateDataCommon( SSFM_LINEAR, SSFM_LINEAR );
  SamplerStateSharedHandle samplerStateHandle = m_rix->samplerStateCreate( samplerStateDataCommon );

  rix::core::SamplerSharedHandle sampler = m_rix->samplerCreate();
  m_rix->samplerSetTexture( sampler, texture );

  if( samplerType != CPT_SAMPLER )
  { 
    m_rix->samplerSetSamplerState( sampler, samplerStateHandle );
  }
  m_rix->containerSetData( fragmentContainer, containerEntryTex, ContainerDataSampler( sampler ) );
}
