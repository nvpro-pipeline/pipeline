// Copyright NVIDIA Corporation 2002-2012
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


#include <stdio.h>
#include <sstream>
#include <fstream>
#include <functional>
#include <dp/sg/core/Scene.h>
#include <dp/util/File.h>
#include <dp/util/Tools.h>
#include <dp/sg/algorithm/Optimize.h>
#include <dp/sg/io/CSF/Saver/inc/CSFSaver.h>
#include <dp/sg/io/CSF/Saver/inc/ExtractGeometryTraverser.h>
#include <dp/sg/io/CSF/Saver/inc/OffsetManager.h>
#include <dp/sg/io/CSF/Saver/inc/CSFSGWrapper.h>

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;
using namespace std;

// supported Plug Interface ID
const UPITID PITID_SCENE_SAVER(UPITID_SCENE_SAVER, UPITID_VERSION); // plug-in type
const UPIID  PIID_CSF_SCENE_SAVER(".CSF", PITID_SCENE_SAVER); // plug-in ID

#if defined( _WIN32 )
BOOL APIENTRY DllMain(HANDLE hModule, DWORD reason, LPVOID lpReserved)
{
  return TRUE;
}
#elif defined( LINUX )
#endif

bool getPlugInterface(const UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  if ( piid == PIID_CSF_SCENE_SAVER )
  {
    pi = CSFSaver::create();
    return( !!pi );
  }
  return false;
}

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(PIID_CSF_SCENE_SAVER);
}

CSFSaverSharedPtr CSFSaver::create()
{
  return( std::shared_ptr<CSFSaver>( new CSFSaver() ) );
}

CSFSaver::CSFSaver()
{
}

CSFSaver::~CSFSaver()
{
}

bool CSFSaver::save( dp::sg::core::SceneSharedPtr const& scene, dp::sg::ui::ViewStateSharedPtr const& viewState, string const& filename )
{
  ExtractGeometryTraverser extractor;
  size_t materialIndex = 0;

  // extract all geometry from the scene and add it to the triangle mesh
  extractor.setViewState( viewState );
  extractor.apply( scene );

  // these structures are referenced by pointers
  // and must stay alive post CSF_save
  std::vector<CSFSGMaterial> & sgMaterials  = extractor.getMaterials();
  std::vector<CSFSGGeometry> & sgGeometries = extractor.getGeometries();
  std::vector<CSFSGNode> & sgNodes = extractor.getNodes();

  CSFile file;


  // FIXME should filter and replace geometries
  // that are marked with alternativePrimitiveGeometryExists
  // to reduce amount of geometries

  file.hasUniqueNodes = 1;
  file.numGeometries = checked_cast<int>(sgGeometries.size());
  file.numMaterials = checked_cast<int>(sgMaterials.size());
  file.numNodes = checked_cast<int>(sgNodes.size());
  file.rootIDX = 0;

  if (!file.numGeometries || !file.numMaterials || !file.numNodes)
  {
    return false;
  }

  // these structures are referenced by pointers
  // and must stay alive post CSF_save
  std::vector<CSFGeometry>  geometries(file.numGeometries);
  std::vector<CSFMaterial>  materials(file.numMaterials);
  std::vector<CSFNode>      nodes(file.numNodes);

  file.geometries = &geometries[0];
  file.materials  = &materials[0];
  file.nodes      = &nodes[0];

  // Geometries
  unsigned int i = 0;
  for(std::vector<CSFSGGeometry>::iterator it = sgGeometries.begin(); it != sgGeometries.end(); it++, i++)
  {
    memset(geometries[i].matrix, 0, 16 * sizeof(float));
    for(int j = 0; j < 4; j++)
    {
      geometries[i].matrix[j * 4 + j] = 1.0f;
    }

    geometries[i].numIndexSolid = checked_cast<int>(it->indices.size());
    geometries[i].numIndexWire = 0;
    geometries[i].numVertices = checked_cast<int>(it->vertices.size());
    geometries[i].numParts = checked_cast<int>(it->parts.size());

    geometries[i].parts = it->parts.size() ? &it->parts[0] : nullptr;

    geometries[i].indexSolid = it->indices.size() ? &it->indices[0] : nullptr;
    geometries[i].indexWire = nullptr;

    geometries[i].vertex = it->vertices.size() ? (float*)&it->vertices[0] : nullptr;
    geometries[i].normal = it->vertices.size() ? (float*)&it->normals[0] : nullptr;
    geometries[i].tex =    it->vertices.size() ? (float*)&it->texcoords[0] : nullptr;
  }

  // Materials
  i = 0;
  for(std::vector<CSFSGMaterial>::iterator it = sgMaterials.begin(); it != sgMaterials.end(); i++, it++)
  {
    for(int j = 0; j < 4; j++){
      materials[i].color[j] = it->diffuse[j];
    }
    size_t nameSize = sizeof(materials[i].name);

    strncpy(materials[i].name, it->name.c_str(), nameSize-1);
    materials[i].name[nameSize-1] = 0;
    materials[i].numBytes = 0;
    materials[i].type = 0;
  }

  // Nodes
  i = 0;
  for(vector<CSFSGNode>::iterator it = sgNodes.begin(); it != sgNodes.end(); i++, it++)
  {
    nodes[i].numChildren = checked_cast<int>(it->children.size());
    nodes[i].children = it->children.size() ? &it->children[0] : nullptr;
    nodes[i].geometryIDX = it->geometryIDX;
    nodes[i].parts = nullptr;

    if(nodes[i].geometryIDX != -1)
    {
      CSFSGGeometry& geometry = sgGeometries[it->geometryIDX];
      nodes[i].numParts = checked_cast<int>( geometry.parts.size() );

      // because geometries get primitives added in a deferred fashion
      // this object might reference geometry which in the end has more parts
      // then during creation time of this object

      CSFNodePart unused;
      unused.active = 0;
      unused.linewidth = 1.0f;
      unused.materialIDX = 0;

      for(size_t j = it->parts.size(); j < geometry.parts.size(); j++)
      {
        it->parts.push_back(unused);
      }

      nodes[i].parts = it->parts.size() ? &it->parts[0] : nullptr;
    }
    else
    {
      nodes[i].numParts = 0;
    }

    for(int j = 0; j < 16; j++)
    {
      nodes[i].objectTM[j] = it->objectTM[j / 4][j & 3];
      nodes[i].worldTM [j] = it->worldTM [j / 4][j & 3];
    }
  }

  bool success = CSFile_save(&file, filename.c_str()) == CADSCENEFILE_NOERROR;

  
  return success;
}
