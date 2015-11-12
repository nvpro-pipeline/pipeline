// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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

//
// 3DSLoader.h
// This file loader can be used to load .3ds files into devtech platform.

#pragma once

#if ! defined( DOXYGEN_IGNORE )

#include <string>
#include <vector>
#include <map>

#include <dp/math/Trafo.h>
#include <dp/math/Vecnt.h>

#include <dp/sg/core/Config.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/io/PlugInterface.h>

#include <dp/util/PlugInCallback.h>

#include <lib3ds/lib3ds.h>
#include <lib3ds/lib3ds_impl.h>

#ifdef _WIN32
// microsoft specific storage-class defines
# ifdef THREEDSLOADER_EXPORTS
#  define THREEDSLOADER_API __declspec(dllexport)
# else
#  define THREEDSLOADER_API __declspec(dllimport)
# endif
#else
# define THREEDSLOADER_API
#endif

// exports required for a scene loader plug-in
extern "C"
{
THREEDSLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi);
THREEDSLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

#define INVOKE_CALLBACK(cb) if( callback() ) callback()->cb

using std::vector;

// flags that are used to indicate which tracks should be read in the constructAnimation() function
typedef enum ThreeDSAnimFlags
{
  P_TRACK = BIT(0),
  R_TRACK = BIT(1),
  S_TRACK = BIT(2),
  ROLL_TRACK = BIT(3)
} ThreeDSAnimFlags;

// struct containing all the data for one smoothing group in a scene
typedef struct SmoothingData
{
  vector <dp::math::Vec3f> vertices;
  vector <dp::math::Vec3f> normals;
  vector <dp::math::Vec2f> texCoords;
  vector <int> order; // // a list of the vertex indices in the order in which they were inserted into the group
  vector <int> vertMap; // maps the original vertex index to the vertex index within the smoothing group
} SmoothingData;

// struct containing all the data for all faces belonging to the same material in a mesh
typedef struct MatGroupData
{
  vector <dp::math::Vec3f> vertices;
  vector <dp::math::Vec3f> normals;
  vector <dp::math::Vec2f> texCoords;
  vector <int> visibilities; // a list of the lists of visibility flags for each face for each material
  vector <unsigned int> indices;
  vector <int> vertMap; // vertMap[j] is the new index of the jth vertex in the material group
} MatGroupData;

DEFINE_PTR_TYPES( ThreeDSLoader );

// Note 'ThreeDSLoader' instead of '3DSLoader' because legal
// C++ class names cannot start with numbers
class ThreeDSLoader : public dp::sg::io::SceneLoader
{
public:
  static ThreeDSLoaderSharedPtr create();
  virtual ~ThreeDSLoader(void);

  dp::sg::core::SceneSharedPtr load( const std::string& filename, dp::util::FileFinder const& fileFinder, dp::sg::ui::ViewStateSharedPtr & viewState );

protected:
  ThreeDSLoader();

private:
  // build the scene from the loaded 3ds data structure
  void buildScene( dp::sg::core::GroupSharedPtr const& root, Lib3dsFile * data );

  // recursively build the node tree from the 3ds data
  void buildTree( dp::sg::core::GroupSharedPtr const& parent, Lib3dsNode * n, dp::math::Vec3f &piv, Lib3dsFile * data );

  // recursively search the node tree for a node of the given name and type
  Lib3dsNode *searchNodeTree( Lib3dsNode *n, int nodeType, char *nodeName);

  // add all meshes at the top level of the scene
  void constructFlatHierarchy( Lib3dsFile *data );

  // set the trafo to represent the orientation of this node
  void orientNode( dp::math::Trafo &, Lib3dsMeshInstanceNode * mnode, dp::math::Vec3f &piv);

  // given all the necessary data, set up and add a PerspectiveCamera to the scene
  void configureCamera( dp::sg::core::GroupSharedPtr const& parent, Lib3dsNode *n, dp::math::Vec3f &piv, bool hasTarget, Lib3dsFile *data );

  // given all the necessary data, add a spotlight target or camera target to the scene
  void configureTarget( dp::sg::core::GroupSharedPtr const& parent, Lib3dsNode *n, dp::math::Vec3f &piv, bool isCamera, Lib3dsFile *data );

  // given all the necessary data, add a PointLight to the scene
  void configurePointlight( dp::sg::core::GroupSharedPtr const& parent, Lib3dsNode *n, dp::math::Vec3f &piv, Lib3dsFile *data );

  // given all the necessary data, add a SpotLight to the scene
  void configureSpotlight( dp::sg::core::GroupSharedPtr const& parent, Lib3dsNode *n, dp::math::Vec3f &piv, bool hasTarget, Lib3dsFile *data );

  // recursively add all children of the given node to the scene
  int addAllChildren( dp::sg::core::GroupSharedPtr const& parent, Lib3dsNode *n, dp::math::Vec3f &piv, Lib3dsFile *data );

  // read the 3ds mesh data and add all of its geometry to the scene
  bool constructGeometry( dp::sg::core::GroupSharedPtr const& group, char *name, Lib3dsFile * data );

  // parse and load all materials in the scene
  void constructMaterials( std::vector< dp::sg::core::PipelineDataSharedPtr > & materials, Lib3dsFile *data );

  // parse and load a texture from the 3ds data structure
  dp::sg::core::ParameterGroupDataSharedPtr createTexture( Lib3dsTextureMap &texture, dp::util::FileFinder const& fileFinder, const std::string & filename, bool isEnvMap );

  // lineraly interpolate between two vectors as a function of the current frame between a left and right frame
  void vecInterp(dp::math::Vec3f &target, dp::math::Vec3f &lData, dp::math::Vec3f &rData, int leftFrame, int rightFrame, int currFrame);

#if defined(KEEP_ANIMATION)
  // add the animation data to the AnimatedTransform based on the various 3ds tracks
  void constructAnimation( dp::sg::core::AnimatedTransform *anim, dp::math::Vec3f &parentPivot, dp::math::Vec3f &pivot,
                           Lib3dsTrack *pTrack, Lib3dsTrack *rTrack, Lib3dsTrack *sTrack, Lib3dsTrack *rollTrack, int flags);
#endif

  // resolve the camera and spotlight target, position and roll callbacks from the global lists
  void postProcessCamerasAndLights(void);

  // fill any of the given tracks with zero keys with default values
  void checkTracks(Lib3dsTrack *pTrack, Lib3dsTrack *rTrack, Lib3dsTrack *sTrack, Lib3dsTrack *rollTrack);

  // remove all temporary data
  void cleanup();

private:
  dp::util::FileFinder  m_fileFinder;
  dp::sg::core::SceneSharedPtr m_scene;
  dp::sg::core::GroupSharedPtr m_topLevel;

  // a list of materialEffects and geometryEffects
  std::vector< dp::sg::core::PipelineDataSharedPtr > m_materials;

  vector < bool > m_hasTexture;
  vector < bool > m_isWire;
  bool m_wirePresent;

  // global lists of cameras, spotlights, and targets used for resolving callbacks
  std::vector<dp::sg::core::PerspectiveCameraSharedPtr>   m_camList;
  std::vector<dp::sg::core::LightSourceSharedPtr>         m_spotList;
  std::map<std::string, dp::sg::core::TransformSharedPtr> m_camLocationList;
  std::map<std::string, dp::sg::core::TransformSharedPtr> m_spotLocationList;
  std::map<std::string, dp::sg::core::TransformSharedPtr> m_camTargetList;
  std::map<std::string, dp::sg::core::TransformSharedPtr> m_spotTargetList;

  int m_numMaterials;
  int m_numFrames;
};

#endif // ! defined( DOXYGEN_IGNORE )
