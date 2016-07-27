// Copyright NVIDIA Corporation 2016
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

#pragma once

#if ! defined( DOXYGEN_IGNORE )

#include <string>
#include <vector>
#include <map>

#include <dp/sg/core/Config.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/VertexAttribute.h> // is this missing in CoreTypes.h
#include <dp/sg/io/PlugInterface.h>
#include <dp/util/FileFinder.h>
#include <json/json.h>

#include <boost/filesystem.hpp>

#ifdef _WIN32
// microsoft specific storage-class defines
# ifdef glTFLoader_EXPORTS
#  define glTFLoader_API __declspec(dllexport)
# else
#  define glTFLoader_API __declspec(dllimport)
# endif
#else
# define glTFLoader_API
#endif

// exports required for a scene loader plug-in
extern "C"
{
  glTFLoader_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi);
  glTFLoader_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

#define INVOKE_CALLBACK(cb) if( callback() ) callback()->cb

DEFINE_PTR_TYPES( glTFLoader );

class glTFLoader : public dp::sg::io::SceneLoader
{
public:
  static glTFLoaderSharedPtr create();
  virtual ~glTFLoader(void);

  dp::sg::core::SceneSharedPtr load( std::string const& filename, dp::util::FileFinder const& fileFinder, dp::sg::ui::ViewStateSharedPtr & viewState );

protected:
  glTFLoader();

private:
  struct BufferView 
  {
    size_t offset;
    size_t byteLength;
    std::shared_ptr<dp::sg::core::Buffer> buffer;
  };

  struct State
  {
    Json::Value                                                     root;
    std::map<std::string, std::shared_ptr<dp::sg::core::GeoNode>>   meshes;
    std::map<std::string, std::shared_ptr<dp::sg::core::Buffer>>    buffers;
    std::map<std::string, BufferView>                               bufferViews;
    std::map<std::string, dp::sg::core::VertexAttribute>            vertexAttributes;
    std::map<std::string, std::shared_ptr<dp::sg::core::IndexSet>>  indexSets;
    std::map<std::string, dp::sg::core::NodeSharedPtr>              nodes;
  };


  std::shared_ptr<dp::sg::core::GeoNode> parseMesh(Json::Value const &mesh, State& state);
  std::shared_ptr<dp::sg::core::Buffer> parseBuffer(boost::filesystem::path const& basePath, Json::Value const &jsonBuffer);
  std::shared_ptr<dp::sg::core::Node> parseNode(Json::Value const &node, State& state);
  glTFLoader::BufferView parseBufferView(Json::Value const &jsonBufferView, State& state);
  dp::sg::core::VertexAttribute getVertexAttributeFromAccessor(std::string const& name, State& state);
  std::shared_ptr<dp::sg::core::IndexSet> getIndexSetFromAccessor(std::string const& name, State& state);


  dp::util::FileFinder  m_fileFinder;
  dp::sg::ui::ViewStateWeakPtr m_viewState;
};

#endif // ! defined( DOXYGEN_IGNORE )
