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

#include "glTFLoader.h"

#include <dp/math/Vecnt.h>
#include <dp/math/Quatt.h>

#include <dp/sg/core/BufferHost.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Primitive.h>

#include <dp/util/PlugIn.h>
#include <dp/util/File.h>
#include <dp/util/Locale.h>

#include <dp/fx/EffectLibrary.h>

#include <gl/GL.h>

#include <boost/algorithm/string.hpp>

// stl headers
#include <algorithm>

using namespace dp::sg::core;
using namespace dp::math;

using namespace std;

// define a unique plug-interface ID for SceneLoader
const dp::util::UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION);

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(dp::util::UPIID(".gltf", PITID_SCENE_LOADER));
}

bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  const dp::util::UPIID PIID_GLTF_SCENE_LOADER = dp::util::UPIID(".gltf", PITID_SCENE_LOADER);

  if ( piid == PIID_GLTF_SCENE_LOADER )
  {
    pi = glTFLoader::create();
    return( !!pi );
  }

  return false;
}

glTFLoaderSharedPtr glTFLoader::create()
{
  return( std::shared_ptr<glTFLoader>( new glTFLoader() ) );
}

glTFLoader::glTFLoader()
{
}

glTFLoader::~glTFLoader()
{
}


SceneSharedPtr
glTFLoader::load( std::string const& filename, dp::util::FileFinder const& fileFinder, dp::sg::ui::ViewStateSharedPtr & viewState )
{
  if ( !dp::util::fileExists(filename) )
  {
    throw dp::FileNotFoundException( filename );
  }

  // set locale temporarily to standard "C" locale
  dp::util::Locale tl("C");

  State state;
  std::ifstream jsonFile(filename, std::ifstream::binary);
  jsonFile >> state.root;

#if 0
  // check for failure
  if (!(doc.LoadFile() && doc.FirstChild()))
  {
    throw std::runtime_error(std::string("Failed to load file <" + filename + ">"));
  }
#endif

  m_viewState = viewState;
  m_fileFinder = fileFinder;

  // create toplevel group
  GroupSharedPtr hGroup(Group::create());
  hGroup->setName( filename );
  boost::filesystem::path baseDir = boost::filesystem::path(filename).parent_path();

  // Load buffers
  Json::Value const &buffers = state.root["buffers"];
  for (auto const& name : buffers.getMemberNames())
  {
    state.buffers[name] = parseBuffer(baseDir, buffers[name]);
  }

  // Load BufferViews
  Json::Value const &bufferViews = state.root["bufferViews"];
  for (auto const& name : bufferViews.getMemberNames())
  {
    state.bufferViews[name] = parseBufferView(bufferViews[name], state);
  }

  // Load Meshes
  Json::Value meshes = state.root["meshes"];
  for (auto &meshName : meshes.getMemberNames())
  {
    state.meshes[meshName] = parseMesh(meshes[meshName], state);
  }

  // Load Nodes
  Json::Value nodes = state.root["nodes"];
  for (auto &nodeName : nodes.getMemberNames())
  {
    state.nodes[nodeName] = parseNode(nodes[nodeName], state);
  }

  // Build up node hierarchy
  for (auto &nodeName : nodes.getMemberNames())
  {
    Json::Value const& children = nodes[nodeName]["children"];
    if (children.isArray())
    {
      dp::sg::core::TransformSharedPtr& transform = std::static_pointer_cast<dp::sg::core::Transform>(state.nodes[nodeName]);
      for (auto const& child : children)
      {
        transform->addChild(state.nodes[child.asString()]);
      }
    }
  }

  // Create Scene
  std::string defaultScene = state.root["scene"].asString();
  Json::Value jsonScene = state.root["scenes"][defaultScene];

  // Add nodes to Scene
  for (auto const& jsonNode : jsonScene["nodes"])
  {
    hGroup->addChild(state.nodes[jsonNode.asString()]);
  }

  // create scene
  SceneSharedPtr hScene = Scene::create();

  // and add group as scene's toplevel
  hScene->setRootNode( hGroup );

  m_fileFinder.clear();

  return hScene;
}

glTFLoader::BufferView glTFLoader::parseBufferView(Json::Value const &jsonBufferView, State& state)
{
  BufferView bufferView;
  bufferView.offset = jsonBufferView["byteOffset"].asUInt64();
  bufferView.byteLength = jsonBufferView["byteLength"].asUInt64();
  bufferView.buffer = state.buffers[jsonBufferView["buffer"].asString()];

  return bufferView;
}

std::shared_ptr<dp::sg::core::Buffer> glTFLoader::parseBuffer(boost::filesystem::path const& basePath, Json::Value const &jsonBuffer)
{
  std::shared_ptr<dp::sg::core::BufferHost> buffer = dp::sg::core::BufferHost::create();

  // TODO support >32-bit buffers?
  buffer->setSize(jsonBuffer["byteLength"].asUInt());
  
  std::ifstream binary((basePath / jsonBuffer["uri"].asString()).string(), std::ifstream::binary);
  binary.read(reinterpret_cast<char*>(dp::sg::core::Buffer::DataWriteLock(buffer, dp::sg::core::Buffer::MapMode::WRITE).getPtr()), buffer->getSize());
  binary.close();

  return buffer;
}

std::shared_ptr<dp::sg::core::Node> glTFLoader::parseNode(Json::Value const &jsonNode, State& state)
{
  std::shared_ptr<dp::sg::core::Transform> transform = dp::sg::core::Transform::create();

  Json::Value const& jsonMatrix = jsonNode["matrix"];
  dp::math::Mat44f matrix;

  for (int index = 0;index < 16;++index)
  {
    matrix[index / 4][index % 4] = jsonMatrix[index].asFloat();
  }

  transform->setMatrix(matrix);

  Json::Value const& meshes = jsonNode["meshes"];
  if (meshes.isArray())
  {
    for (auto const& mesh : meshes)
    {
      transform->addChild(state.meshes[mesh.asString()]);
    }
  }

  return transform;
}

dp::DataType getDataTypeFromGLComponentType(GLenum componentType)
{
  switch (componentType) {
    case GL_BYTE:
      return dp::DataType::INT_8;
    case GL_UNSIGNED_BYTE:
      return dp::DataType::UNSIGNED_INT_8;
    case GL_SHORT:
      return dp::DataType::INT_16;
    case GL_UNSIGNED_SHORT:
      return dp::DataType::UNSIGNED_INT_16;
    case GL_INT:
      return dp::DataType::INT_32;
    case GL_UNSIGNED_INT:
      return dp::DataType::UNSIGNED_INT_32;
    case GL_FLOAT:
      return dp::DataType::FLOAT_32;
    default:
      throw std::runtime_error("unsupported format");
  }
}

uint32_t getNumberOfComponents(std::string const &type)
{
  if (type == "SCALAR")
    return 1;
  if (type == "VEC2")
    return 2;
  if (type == "VEC3")
    return 3;
  if (type == "VEC4")
    return 4;
  if (type == "MAT2")
    return 4;
  if (type == "MAT3")
    return 9;
  if (type == "MAT4")
    return 16;
  throw std::runtime_error("unknown datatype");
}

dp::sg::core::VertexAttribute glTFLoader::getVertexAttributeFromAccessor(std::string const& name, State& state)
{
  auto it = state.vertexAttributes.find(name);
  if (it != state.vertexAttributes.end())
  {
    return it->second;
  }

  Json::Value accessor = state.root["accessors"][name];
  VertexAttribute va;

  BufferView &bufferView = state.bufferViews[accessor["bufferView"].asString()];

  // TODO The vertex attribute of the SceneGraph should be separate from the data and have it's own offset within the bytestream
  va.setData(getNumberOfComponents(accessor["type"].asString()), getDataTypeFromGLComponentType(accessor["componentType"].asInt())
                                  , bufferView.buffer, static_cast<uint32_t>(bufferView.offset + accessor["byteOffset"].asInt()), accessor["byteStride"].asInt(), accessor["count"].asInt() );
  state.vertexAttributes[name] = va;

  return va;
}

std::shared_ptr<dp::sg::core::IndexSet> glTFLoader::getIndexSetFromAccessor(std::string const& name, State& state)
{
  auto it = state.indexSets.find(name);
  if (it != state.indexSets.end())
  {
    return it->second;
  }

  std::shared_ptr<dp::sg::core::IndexSet> indexSet = IndexSet::create();

  Json::Value accessor = state.root["accessors"][name];
  BufferView &bufferView = state.bufferViews[accessor["bufferView"].asString()];
  indexSet->setBuffer(bufferView.buffer, accessor["count"].asUInt(), getDataTypeFromGLComponentType(accessor["componentType"].asUInt()));
  DP_ASSERT(accessor["type"].asString() == "SCALAR");
  DP_ASSERT(accessor["stride"].asUInt() == 0);

  state.indexSets[name] = indexSet;

  return indexSet;
}


std::shared_ptr<dp::sg::core::GeoNode> glTFLoader::parseMesh(Json::Value const& mesh, State& state)
{
  std::shared_ptr<dp::sg::core::Primitive> primitive = dp::sg::core::Primitive::create(PrimitiveType::TRIANGLES);
  std::shared_ptr<dp::sg::core::VertexAttributeSet> vertexAttributeSet = dp::sg::core::VertexAttributeSet::create();

  primitive->setName(mesh["name"].asString());
  for (auto & element : mesh["primitives"])
  {
    Json::Value const& attributes = element["attributes"];
    for (std::string const& attributeName : attributes.getMemberNames())
    {
      Json::Value const& attribute = attributes[attributeName];
      if (attributeName == "POSITION")
      {
        vertexAttributeSet->setVertexAttribute(dp::sg::core::VertexAttributeSet::AttributeID::POSITION, getVertexAttributeFromAccessor(attribute.asString(), state));
      }

      if (attributeName == "NORMAL")
      {
        vertexAttributeSet->setVertexAttribute(dp::sg::core::VertexAttributeSet::AttributeID::NORMAL, getVertexAttributeFromAccessor(attribute.asString(), state));
      }

      if (attributeName == "JOINT")
      {
        std::cerr << "ignoring joint" << std::endl;
        vertexAttributeSet->setVertexAttribute(dp::sg::core::VertexAttributeSet::AttributeID::UNUSED_1, getVertexAttributeFromAccessor(attribute.asString(), state));
      }

      if (attributeName == "WEIGHT")
      {
        vertexAttributeSet->setVertexAttribute(dp::sg::core::VertexAttributeSet::AttributeID::VERTEX_WEIGHT, getVertexAttributeFromAccessor(attribute.asString(), state));
      }

      if (attributeName == "COLOR" || attributeName == "COLOR_0")
      {
        vertexAttributeSet->setVertexAttribute(dp::sg::core::VertexAttributeSet::AttributeID::COLOR, getVertexAttributeFromAccessor(attribute.asString(), state));
      }

      if (attributeName == "COLOR_1")
      {
        vertexAttributeSet->setVertexAttribute(dp::sg::core::VertexAttributeSet::AttributeID::SECONDARY_COLOR, getVertexAttributeFromAccessor(attribute.asString(), state));
      }

      if (boost::algorithm::starts_with(attributeName, "TEXTURE_"))
      {
        int32_t textureIndex = atoi(attributeName.c_str() + strlen("TEXTURE_"));
        vertexAttributeSet->setVertexAttribute(dp::sg::core::VertexAttributeSet::AttributeID(int32_t(dp::sg::core::VertexAttributeSet::AttributeID::TEXCOORD0) + textureIndex)
                                             , getVertexAttributeFromAccessor(attribute.asString(), state));
      }
    }

    Json::Value const& indices = element["indices"];
    if (!indices.isNull())
    {
      primitive->setIndexSet(getIndexSetFromAccessor(indices.asString(), state));
    }

    // support one element for now
    break;
  }
  primitive->setVertexAttributeSet(vertexAttributeSet);

  std::shared_ptr<dp::sg::core::GeoNode> geoNode = dp::sg::core::GeoNode::create();

  geoNode->setPrimitive(primitive);
  dp::fx::EffectDataSharedPtr phongEffectData = dp::fx::EffectLibrary::instance()->getEffectData("standardMaterial");
  DP_ASSERT(phongEffectData);

  geoNode->setMaterialPipeline(dp::sg::core::PipelineData::create(phongEffectData));

  return geoNode;
}
