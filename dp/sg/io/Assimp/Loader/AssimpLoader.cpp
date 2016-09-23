// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/io/PlugInterface.h> // definition of UPITID_VERSION,
#include <dp/sg/io/PlugInterfaceID.h> // definition of UPITID_VERSION, UPITID_SCENE_LOADER, and UPITID_SCENE_SAVER

#include <dp/Exception.h>
#include <dp/sg/core/Config.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/io/IO.h>
#include <dp/sg/io/PlugInterface.h>
#include <dp/sg/io/PlugInterfaceID.h>

#include <dp/math/Vecnt.h>
#include <dp/math/Quatt.h>

#include <dp/sg/core/BufferHost.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/PipelineData.h>

#include <dp/util/PlugIn.h>
#include <dp/util/File.h>
#include <dp/util/Locale.h>

#include <dp/fx/EffectLibrary.h>

#include <boost/algorithm/string.hpp>

#include <dp/sg/io/Assimp/Loader/AssimpLoader.h>
#include <assimp/cimport.h>
#include <assimp/postprocess.h>

// stl headers
#include <algorithm>

using namespace dp::sg::core;
using namespace dp::math;

using namespace std;

// define a unique plug-interface ID for SceneLoader
const dp::util::UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION);

const std::vector<dp::util::UPIID> UPIIDs {
  { ".3D", PITID_SCENE_LOADER },
  { ".3DS", PITID_SCENE_LOADER },
  //{ ".3MF", PITID_SCENE_LOADER }, Test file from Assimp crashes during load
  //{ ".AC", PITID_SCENE_LOADER }, Test file from Assimp crashes during load
  { ".ASE", PITID_SCENE_LOADER }, 
  { ".B3D", PITID_SCENE_LOADER }, // Test scene shows no lighting, analyze normals
  { ".BLEND", PITID_SCENE_LOADER },
  { ".DAE", PITID_SCENE_LOADER },
  { ".OBJ", PITID_SCENE_LOADER },
  //{ ".PLY", PITID_SCENE_LOADER }, // Keep our own loader for now since it handles duplicated vertices better than Assimp
  { ".STL", PITID_SCENE_LOADER }
  // TODO test and add all file formats from Assimp
};

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids = UPIIDs;
}

bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  if ( std::find(UPIIDs.begin(), UPIIDs.end(), piid) != UPIIDs.end())
  {
    pi = AssimpLoader::create();
    return( !!pi );
  }

  return false;
}

AssimpLoaderSharedPtr AssimpLoader::create()
{
  return( std::shared_ptr<AssimpLoader>( new AssimpLoader() ) );
}

AssimpLoader::AssimpLoader()
{
}

AssimpLoader::~AssimpLoader()
{
}


SceneSharedPtr
AssimpLoader::load( std::string const& filename, dp::util::FileFinder const& fileFinder, dp::sg::ui::ViewStateSharedPtr & viewState )
{
  if ( !dp::util::fileExists(filename) )
  {
    throw dp::FileNotFoundException( filename );
  }

  // set locale temporarily to standard "C" locale
  dp::util::Locale tl("C");

  m_viewState = viewState;
  m_fileFinder = fileFinder;

  aiScene const * importedScene = aiImportFile(filename.c_str(),
    aiProcess_CalcTangentSpace |
    aiProcess_Triangulate |
    aiProcess_JoinIdenticalVertices |
    aiProcess_SortByPType
  );

  if (!importedScene) {
    throw std::runtime_error("failed to load scene");
  }

  NodeSharedPtr hGroup(Group::create());
  hGroup->setName(filename);

  State state;

  // create toplevel group
  state.baseDir = boost::filesystem::path(filename).parent_path();

  // generate materials
  for (unsigned int materialIndex = 0; materialIndex < importedScene->mNumMaterials; ++materialIndex)
  {
    state.materials.push_back(processMaterial(importedScene->mMaterials[materialIndex], state));
  }

  // generate meshes
  for (unsigned int meshIndex = 0; meshIndex < importedScene->mNumMeshes; ++meshIndex)
  {
    state.meshes.push_back(processMesh(importedScene->mMeshes[meshIndex], state));
  }

  hGroup = processNode(importedScene->mRootNode, state);

  aiReleaseImport(importedScene);

  dp::sg::core::SceneSharedPtr scene = dp::sg::core::Scene::create();
  scene->setRootNode(hGroup);


  return scene;
}

dp::sg::core::NodeSharedPtr AssimpLoader::processNode(aiNode const* node, State &state)
{
  dp::sg::core::TransformSharedPtr transform = dp::sg::core::Transform::create();
  dp::math::Mat44f matrix;
  matrix[0][0] = float(node->mTransformation.a1);
  matrix[0][1] = float(node->mTransformation.b1);
  matrix[0][2] = float(node->mTransformation.c1);
  matrix[0][3] = float(node->mTransformation.d1);

  matrix[1][0] = float(node->mTransformation.a2);
  matrix[1][1] = float(node->mTransformation.b2);
  matrix[1][2] = float(node->mTransformation.c2);
  matrix[1][3] = float(node->mTransformation.d2);

  matrix[2][0] = float(node->mTransformation.a3);
  matrix[2][1] = float(node->mTransformation.b3);
  matrix[2][2] = float(node->mTransformation.c3);
  matrix[2][3] = float(node->mTransformation.d3);

  matrix[3][0] = float(node->mTransformation.a4);
  matrix[3][1] = float(node->mTransformation.b4);
  matrix[3][2] = float(node->mTransformation.c4);
  matrix[3][3] = float(node->mTransformation.d4);

  transform->setMatrix(matrix);

  for (unsigned int meshIndex = 0; meshIndex < node->mNumMeshes; ++meshIndex)
  {
    transform->addChild(state.meshes[node->mMeshes[meshIndex]]);
  }

  for (unsigned int childIndex = 0; childIndex < node->mNumChildren; ++childIndex)
  {
    transform->addChild(processNode(node->mChildren[childIndex], state));
  }

  return transform;
}

dp::sg::core::GeoNodeSharedPtr AssimpLoader::processMesh(aiMesh const* mesh, State const& state)
{
  dp::sg::core::VertexAttributeSetSharedPtr vas = dp::sg::core::VertexAttributeSet::create();

  // positions
  vas->setVertexData(VertexAttributeSet::AttributeID::POSITION, 0, 3, dp::DataType::FLOAT_32, mesh->mVertices, 3 * sizeof(float), mesh->mNumVertices, true);

  // normals
  if (mesh->HasNormals())
  {
    vas->setVertexData(VertexAttributeSet::AttributeID::NORMAL, 0, 3, dp::DataType::FLOAT_32, mesh->mNormals, 3 * sizeof(float), mesh->mNumVertices, true);
  }

  // vertex colors
  static_assert(sizeof(*mesh->mColors[0]) == 16, "float colors supported only");
  if (mesh->HasVertexColors(0))
  {
    vas->setVertexData(VertexAttributeSet::AttributeID::COLOR, 0, 4, dp::DataType::FLOAT_32, mesh->mColors[0], 4 * sizeof(float), mesh->mNumVertices, true);
  }
  if (AI_MAX_NUMBER_OF_COLOR_SETS >= 1 && mesh->HasVertexColors(1))
  {
    vas->setVertexData(VertexAttributeSet::AttributeID::COLOR, 0, 4, dp::DataType::FLOAT_32, mesh->mColors[1], 4 * sizeof(float), mesh->mNumVertices, true);
  }

  // texture coordinates
  unsigned int maxNumTextureCoords = std::min(AI_MAX_NUMBER_OF_TEXTURECOORDS, 8);
  for (unsigned int texCoordIndex = 0; texCoordIndex < maxNumTextureCoords; ++texCoordIndex)
  {
    if (mesh->HasTextureCoords(texCoordIndex))
    {
      unsigned int numUVComponents = mesh->mNumUVComponents[texCoordIndex];
      vas->setVertexData(dp::sg::core::VertexAttributeSet::AttributeID(int(VertexAttributeSet::AttributeID::TEXCOORD0) + texCoordIndex),
                         0, numUVComponents, dp::DataType::FLOAT_32, mesh->mTextureCoords[texCoordIndex], 3 * sizeof(float), mesh->mNumVertices, true);
    }
  }

  // tangents and bitangents
  if (mesh->HasTangentsAndBitangents())
  {
    vas->setVertexData(VertexAttributeSet::AttributeID::TANGENT, 0, 3, dp::DataType::FLOAT_32, mesh->mTangents, 3 * sizeof(float), mesh->mNumVertices, true);
    vas->setVertexData(VertexAttributeSet::AttributeID::BINORMAL, 0, 3, dp::DataType::FLOAT_32, mesh->mBitangents, 3 * sizeof(float), mesh->mNumVertices, true);
  }

  dp::sg::core::PrimitiveSharedPtr primitive = dp::sg::core::Primitive::create(PrimitiveType::TRIANGLES);
  primitive->setVertexAttributeSet(vas);

  // Now generate IndexSet
  dp::sg::core::IndexSetSharedPtr indexSet = dp::sg::core::IndexSet::create();
  std::vector<uint32_t> indices;
  indices.reserve(mesh->mNumFaces * 3);
  for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
  {
    aiFace const & face = mesh->mFaces[faceIndex];
    // support only triangles by now
    assert(face.mNumIndices == 3);
    indices.push_back(face.mIndices[0]);
    indices.push_back(face.mIndices[1]);
    indices.push_back(face.mIndices[2]);
  }
  indexSet->setData(indices.data(), (uint32_t)indices.size(), dp::DataType::UNSIGNED_INT_32);
  primitive->setIndexSet(indexSet);

  // generate normals here?
  if (!mesh->HasNormals() && mesh->HasPositions())
  {
    primitive->generateNormals();
  }

  dp::sg::core::GeoNodeSharedPtr geoNode = dp::sg::core::GeoNode::create();
  geoNode->setPrimitive(primitive);
  geoNode->setMaterialPipeline(dp::sg::core::createStandardMaterialData());

  geoNode->setMaterialPipeline(state.materials[mesh->mMaterialIndex]);

  return geoNode;
}

dp::math::Vec3f getVec3f(aiColor3D & color)
{
  return dp::math::Vec3f(float(color.r), float(color.g), float(color.b));
}

dp::sg::core::PipelineDataSharedPtr AssimpLoader::processMaterial(aiMaterial const* material, State const& state)
{
  aiString name;
  aiColor3D ambientColor(0.0f, 0.0f, 0.0f);
  aiColor3D diffuseColor(0.0f, 0.0f, 0.0f);
  aiColor3D specularColor(0.0f, 0.0f, 0.0f);
  aiColor3D emissiveColor(0.0f, 0.0f, 0.0f);
  float specularExponent = 0.0f;
  float opacity = 1.0f;
  float refraction = 1.0f;

  material->Get(AI_MATKEY_NAME, name);
  material->Get(AI_MATKEY_COLOR_AMBIENT, ambientColor);
  material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor);
  material->Get(AI_MATKEY_COLOR_SPECULAR, specularColor);
  material->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor);
  material->Get(AI_MATKEY_SHININESS, specularExponent);
  material->Get(AI_MATKEY_OPACITY, opacity);
  material->Get(AI_MATKEY_REFRACTI, refraction);

  if (specularExponent == 0.0f)
  {
    specularColor = {0.0f, 0.0f, 0.0f};
  }

  dp::sg::core::PipelineDataSharedPtr pipeline =  dp::sg::core::createStandardMaterialData(getVec3f(ambientColor),
                                                                                           getVec3f(diffuseColor),
                                                                                           getVec3f(specularColor), specularExponent,
                                                                                           getVec3f(emissiveColor),
                                                                                           opacity, 0.0f, refraction);

  if (material->GetTextureCount(aiTextureType_DIFFUSE))
  {
    aiString aiTexturePath;
    material->GetTexture(aiTextureType_DIFFUSE, 0, &aiTexturePath);

    boost::filesystem::path texturePath = state.baseDir / std::string(aiTexturePath.C_Str());
    dp::sg::core::TextureHostSharedPtr texture = dp::sg::io::loadTextureHost(texturePath.string());
    if (!texture)
    {
      throw std::runtime_error("failed to load texture");
    }
    dp::sg::core::SamplerSharedPtr sampler = dp::sg::core::Sampler::create(texture);

    dp::fx::EffectSpecSharedPtr const & standardSpec = dp::sg::core::getStandardMaterialSpec();
    DP_ASSERT(standardSpec);

    dp::fx::EffectSpec::iterator textureSpecIt = standardSpec->findParameterGroupSpec(std::string("standardTextureParameters"));
    DP_ASSERT(textureSpecIt != standardSpec->endParameterGroupSpecs());

    dp::sg::core::ParameterGroupDataSharedPtr textureData = dp::sg::core::ParameterGroupData::create(*textureSpecIt);

    DP_VERIFY(textureData->setParameter("textureEnable", true));
    DP_VERIFY(textureData->setParameter("sampler", sampler));

    pipeline->setParameterGroupData(textureData);
  }
  return pipeline;
}