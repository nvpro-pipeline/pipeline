// Copyright NVIDIA Corporation 2002-2015
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


// 3DSLoader.cpp
//

#include <dp/sg/io/PlugInterface.h> // definition of UPITID_VERSION,
#include <dp/sg/io/PlugInterfaceID.h> // definition of UPITID_VERSION, UPITID_SCENE_LOADER, and UPITID_SCENE_SAVER

#include <dp/Exception.h>
#include <dp/math/Quatt.h>
#include <dp/math/Vecnt.h>
#include <dp/math/Spherent.h>

#include <dp/fx/EffectSpec.h>

#include <dp/sg/core/Config.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/io/PlugInterface.h>
#include <dp/util/File.h>
#include <dp/util/Locale.h>
#include "3DSLoader.h"

#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/IndexSet.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/io/IO.h>

// for trig function atanf
#include <math.h>
#include <sstream>
#include <map>

#define MAX_SHINE 100 // the maximum specular exponent ("shininess")
#define LIGHT_INTENSITY 0.7f // the default light intensity if none is supplied
#define CREASE_ANGLE 45.0f // the default crease angle for smoothing local vertex normals
#define CORRUPTED_BUFFER 1e+06f // the cutoff for a "very large" float indicating an invalid texcoord

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using std::pair;
using std::string;
using std::vector;
using std::map;


// unique plug-in types
const dp::util::UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION);

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(dp::util::UPIID(".3DS", PITID_SCENE_LOADER));
}

bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  const dp::util::UPIID PIID_3DS_SCENE_LOADER = dp::util::UPIID(".3DS", PITID_SCENE_LOADER);

  if ( piid == PIID_3DS_SCENE_LOADER )
  {
    pi = ThreeDSLoader::create();
    return( !!pi );
  }

  return false;
}


ThreeDSLoaderSharedPtr ThreeDSLoader::create()
{
  return( std::shared_ptr<ThreeDSLoader>( new ThreeDSLoader() ) );
}

ThreeDSLoader::ThreeDSLoader()
  : m_wirePresent(false)
  , m_numMaterials(0)
  , m_numFrames(0)
{
}

ThreeDSLoader::~ThreeDSLoader()
{
}

SceneSharedPtr ThreeDSLoader::load( std::string const& filename, dp::util::FileFinder const& fileFinder, dp::sg::ui::ViewStateSharedPtr & viewState )
{
  if ( !dp::util::fileExists(filename) )
  {
    throw dp::FileNotFoundException( filename );
  }

  // set locale temporarily to standard "C" locale
  dp::util::Locale tl("C");

  Lib3dsFile *f = lib3ds_file_open( filename.c_str() );
  if ( !f )
  {
    throw std::runtime_error( std::string("Failed to load 3ds file: " + filename ) );
  }

  // the file was successfully parsed and the 3ds data structure has been loaded into memory  

  // set search paths for textures
  m_fileFinder = fileFinder;
  m_fileFinder.addSearchPath( dp::util::getFilePath( filename ) );

  // create toplevel scene
  m_scene = Scene::create();

  // create toplevel group
  m_topLevel = Group::create();
  m_topLevel->setName( "3DS scene" );

  // build scene
  buildScene( m_topLevel, f );

  // use the lib3ds library to free the memory reserved for the Lib3dsFile
  lib3ds_file_free( f );

  m_scene->setRootNode( m_topLevel );

  // now that we've collected all of the camera and spotlight data, add the callbacks
  postProcessCamerasAndLights();

  // keep a reference to the scene before cleaning up the resources
  SceneSharedPtr scene = m_scene;

  cleanup();

  return scene;
}

void ThreeDSLoader::cleanup()
{
  m_fileFinder.clear();
  m_scene.reset();
  m_topLevel.reset();

  m_materials.clear();

  m_hasTexture.clear();
  m_isWire.clear();
  m_wirePresent = false;

  // global lists of cameras, spotlights, and targets used for resolving callbacks
  m_camList.clear();
  m_spotList.clear();
  m_camLocationList.clear();
  m_spotLocationList.clear();
  m_camTargetList.clear();
  m_spotTargetList.clear();

  m_numMaterials = 0;
  m_numFrames = 0;
}

bool noHierarchy( Lib3dsNode * node )
{
  return( !node || ( ( strcmp( node->name, "$$$DUMMY" ) == 0 ) && noHierarchy( node->next ) && noHierarchy( node->childs ) ) );
}

void 
ThreeDSLoader::buildScene( GroupSharedPtr const& parent, Lib3dsFile * data )
{
  if (data == NULL)
  {
     return;
  }

  Lib3dsNode *node = data->nodes;

  if ( noHierarchy( node ) && ( 0 < data->nmeshes ) )
  {
    // this 3ds file has no node hierarchy. construct a basic "flat" hierarchy with all meshes
    // at the top level and continue
    constructFlatHierarchy( data );
    node = data->nodes;

    // print a notice to the console and continue
    INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","3ds file " + std::string(data->name) +
                                               " did not contain scene node hierarchy. Inserted all geometry at the top level.\n"));
  }

  // the number of frames in the animation is one plus the recorded number in the file
  m_numFrames = data->frames+1;
  m_numMaterials = data->nmaterials;

  m_materials.resize(m_numMaterials);
  
  // flag vector indicating which materials come with textures
  m_hasTexture.resize(m_numMaterials+1,false);

  // instantiate a vector of flags for whether each material is a wireframe (last element is the no-material group)
  m_isWire.resize(m_numMaterials+1,false);

  // add all of the materials to the global material vector
  constructMaterials( m_materials, data );

  Vec3f nullPivot (0,0,0);

  // build the hierarchy for all nodes beneath the top-level group
  while(node) 
  {
    buildTree( parent, node, nullPivot, data );
    node = node->next;
  }

  m_materials.clear();
}

void
ThreeDSLoader::buildTree(GroupSharedPtr const& parent, Lib3dsNode *n, Vec3f &piv, Lib3dsFile *data) 
{
  // act differently depending on what type of node this is
  switch(n->type) 
  {
    case LIB3DS_NODE_AMBIENT_COLOR:
      {
        // set the scene's ambient color
        Lib3dsAmbientColorNode *acnode = (Lib3dsAmbientColorNode *)n;
        Lib3dsKey *cKey = acnode->color_track.keys;
        m_scene->setAmbientColor(Vec3f(cKey->value[0],cKey->value[1],cKey->value[2]));
      } 
      break;

    case LIB3DS_NODE_CAMERA: 
      {
        // search for this camera's target in the scene
        Lib3dsNode *targetNode = searchNodeTree(data->nodes, LIB3DS_NODE_CAMERA_TARGET, n->name);

        // check if we've found the camera's target node
        if(!targetNode)
        {
          // print a warning message to the console and continue
          INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","target node " + std::string(n->name) + 
                                               " could not be located for the associated camera in the 3ds file. Adding non-animated camera to the scene.\n"));

          // set up camera without target data
          configureCamera(parent, n, piv, false, data);
        } 
        else
        {
          // set up camera with target data
          configureCamera(parent, n, piv, true, data);
        }
      }
      break;

    case LIB3DS_NODE_CAMERA_TARGET:
      {
        // search for this target's camera in the scene
        Lib3dsNode *camNode = searchNodeTree(data->nodes, LIB3DS_NODE_CAMERA, n->name);

        if(!camNode)
        {
          // print a warning message to the console and continue
          INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","target node " + std::string(n->name) + 
                                               " found with no associated camera in the 3ds file. Skipping this degenerate target.\n"));
        }
        else
        {
          // set up camera target
          configureTarget(parent, n, piv, true, data);
        }
      } 
      break;

    case LIB3DS_NODE_OMNILIGHT:
      {
        // set up pointlight
        configurePointlight(parent, n, piv, data);
      }
      break;

    case LIB3DS_NODE_SPOTLIGHT:
      {
        // search for this spotlight's target in the scene
        Lib3dsNode *targetNode = searchNodeTree(data->nodes, LIB3DS_NODE_SPOTLIGHT_TARGET, n->name);

        // check if we've found the spotlight's target node
        if(!targetNode)
        {
          // print a warning message to the console and continue
          INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","target node " + std::string(n->name) + 
                                               " could not be located for the associated spotlight in the 3ds file. Adding non-animated spotlight to the scene.\n"));

          // set up spotlight without target data
          configureSpotlight(parent, n, piv, false, data);
        } 
        else
        {
          // set up spotlight with target data
          configureSpotlight(parent, n, piv, true, data);
        }
      }
      break;

    case LIB3DS_NODE_SPOTLIGHT_TARGET: 
      {
        // search for this target's spotlight in the scene
        Lib3dsNode *spotNode = searchNodeTree(data->nodes, LIB3DS_NODE_SPOTLIGHT, n->name);

        if(!spotNode)
        {
          // print a warning message to the console and continue
          INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","target node " + std::string(n->name) + 
                                               " found with no associated spotlight in the 3ds file. Skipping this degenerate target.\n"));
        }
        else
        {
          // set up spotlight target
          configureTarget(parent, n, piv, false, data);
        }
      } 
      break;

    case LIB3DS_NODE_MESH_INSTANCE:
      Lib3dsMeshInstanceNode *mnode = (Lib3dsMeshInstanceNode *)n;
      
      Lib3dsTrack *pTrack = &mnode->pos_track;
      Lib3dsTrack *rTrack = &mnode->rot_track;
      Lib3dsTrack *sTrack = &mnode->scl_track;

      // fill any tracks with zero or null keys with default values
      checkTracks(pTrack, rTrack, sTrack, NULL);

      // this group is animated if any of this node's tracks have more than one key
      bool isAnimated = pTrack->nkeys > 1 ||
                rTrack->nkeys > 1 ||
                sTrack->nkeys > 1;

      Vec3f thisPivot = Vec3f(mnode->pivot[0],mnode->pivot[1],mnode->pivot[2]);
      
      if(strcmp(n->name,"$$$DUMMY") == 0) 
      {
        // this node is a dummy group (not a visible object) used to group other meshes together
#if defined(KEEP_ANIMATION)
        if(isAnimated) 
        {
          AnimatedTransformSharedPtr hAnim( AnimatedTransform::create() );
          AnimatedTransformLock anim ( hAnim );

          anim->setName(mnode->instance_name);

          // construct the animation for this group
          constructAnimation(anim,piv,thisPivot,pTrack,rTrack,sTrack,NULL,P_TRACK|R_TRACK|S_TRACK);

          // add this transform as a child to the group above
          parent->addChild( hAnim );


          // add all of this transform's children to the tree
          addAllChildren( anim, n, thisPivot, data);
        }
        else // non-animated group
#endif
        {
          TransformSharedPtr hTrans( Transform::create() );
          hTrans->setName(mnode->instance_name);

          Trafo t;

          // configure the trafo to reflect the orientation of this node
          orientNode(t, mnode, piv);

          hTrans->setTrafo(t);

          // add this group as a child to the group above
          parent->addChild( hTrans );

          // add all of this group's children to the tree
          addAllChildren( hTrans, n, thisPivot, data);
        }
      }
      else 
      {
        // this node is an actual mesh
        GroupSharedPtr group = Group::create();

        // construct the GeoNode with the proper mesh geometry
        bool hadGeometry = constructGeometry( group, n->name, data );

#if defined(KEEP_ANIMATION)
        if(isAnimated) 
        {
          AnimatedTransformSharedPtr hAnim( AnimatedTransform::create() );
          AnimatedTransformLock anim ( hAnim );
          
          anim->setName(std::string("Transform of mesh ") +
                        std::string( n->name ));
          
          // do the actual trafo construction for this animation
          constructAnimation(anim, piv, thisPivot,
                             &mnode->pos_track,&mnode->rot_track,&mnode->scl_track,NULL,P_TRACK|R_TRACK|S_TRACK);
          
          // only add this GeoNode if it has geometric data (vertices, faces, etc)
          if(hadGeometry)
          {
            // add the GeoNode beneath the animated transform
            anim->addChild( group );
          }

          // add all of this node's children to the tree
          int numChildren = addAllChildren( anim, n, thisPivot, data );

          // only add this animation node if there is geometry beneath it
          if(hadGeometry || numChildren > 0)
          {
            // add the transform to the parent above
            parent->addChild(hAnim);
          }
        }
        else // mesh is not animated
#endif
        {
          TransformSharedPtr hTrans( Transform::create() );

          std::string transName = std::string("Transform of mesh ") + std::string(n->name) + std::string(mnode->instance_name);

          hTrans->setName(transName);
          
          Trafo t;
          
          // configure the trafo to reflect the orientation of this node
          orientNode(t, mnode, piv);
          
          hTrans->setTrafo(t);
          
          // only add the GeoNode if it had geometric data
          if(hadGeometry)
          {
            // add this GeoNode directly to its parent
            hTrans->addChild( group );
          }

          // add all of this node's children to the tree
          int numChildren = addAllChildren( hTrans, n, thisPivot, data );

          // only add this animation node if there is geometry beneath it
          if(hadGeometry || numChildren > 0)
          {
            // add this group as a child to the group above
            parent->addChild( hTrans );
          }
        }
      }
      break;
  };
}

void
ThreeDSLoader::orientNode(Trafo &t, Lib3dsMeshInstanceNode * mnode, Vec3f &piv)
{
  Lib3dsKey *pKey = mnode->pos_track.keys;
  Lib3dsKey *rKey = mnode->rot_track.keys;
  Lib3dsKey *sKey = mnode->scl_track.keys;
  
  Vec3f thisPivot (mnode->pivot[0],mnode->pivot[1],mnode->pivot[2]);
  Vec3f meshPivot = thisPivot - piv;

  t.setCenter(thisPivot);
  t.setTranslation(Vec3f(pKey->value[0]-meshPivot[0],pKey->value[1]-meshPivot[1],pKey->value[2]-meshPivot[2]));

  Vec3f axis (rKey->value[0],rKey->value[1],rKey->value[2]);

  if(length(axis) != 0) 
  { 
    // rotate quaternion around its negative axis
    Quatf q (axis,-rKey->value[3]);

    t.setOrientation(q);
  }

  t.setScaling(Vec3f(sKey->value[0],sKey->value[1],sKey->value[2]));
}

void
ThreeDSLoader::configureCamera( GroupSharedPtr const& parent, Lib3dsNode *n, Vec3f &piv, bool hasTarget, Lib3dsFile *data )
{
  Lib3dsCameraNode *cnode = (Lib3dsCameraNode *)n;
  Lib3dsTrack pTrack = cnode->pos_track; // position track for this camera's animation
  Lib3dsTrack rollTrack = cnode->roll_track; // roll track for this camera's animation

  // search for the correct camera
  Lib3dsCamera *c;
  bool cameraFound = false;
  for(int i=0; i<data->ncameras; i++) 
  {
    c = data->cameras[i];
    if(strcmp(c->name,n->name) == 0)
    {
      cameraFound = true;
      break;
    }
  }

  if(!cameraFound)
  {
      // print a warning message to the console and do not process camera
      INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","camera data for '" + std::string(n->name) + 
                                           "' could not be located in the 3ds file. Did not add this camera to the scene.\n"));
      return;
  }

  PerspectiveCameraSharedPtr hCamera( PerspectiveCamera::create() );
  hCamera->setName(c->name);

  // set the field of view of the camera: 3ds is in degrees, dp::sg::core in radians
  if(c->fov!=0.f)
  {
    hCamera->setFieldOfView(degToRad(c->fov));
  }
  else
  {
    hCamera->setFieldOfView(PI_HALF);
  }

  // add this camera to the scene
  m_scene->addCamera(hCamera);

  // check if we've been given target position data
  if(!hasTarget)
  {
    // if not, add a non-animated camera and return
    DP_ASSERT(!"callbacks not supported");
    return;
  }

  // add this camera to the global list to be postprocessed with callbacks later
  m_camList.push_back(hCamera);
 
  // fill any tracks with zero or null keys to default values
  checkTracks(&pTrack, NULL, NULL, &rollTrack);

  // if this camera has no animation, add the callback and we are done
#if defined(KEEP_ANIMATION)
  if(pTrack.nkeys == 1 && rollTrack.nkeys == 1)
#endif
  {
    TransformSharedPtr hTrans( Transform::create() );
    {
      Trafo t;
      Vec3f pos (pTrack.keys->value[0],pTrack.keys->value[1],pTrack.keys->value[2]);
      t.setTranslation(pos - piv);

      float angle = 0;
      if(rollTrack.nkeys == 1)
      {
        angle = degToRad(rollTrack.keys->value[0]);
      }
      Quatf roll (Vec3f(0,0,1), angle);
      t.setOrientation(roll);

      hTrans->setTrafo(t);

      // add this Transform to the global camera location list to be postprocessed later
      m_camLocationList[n->name] = hTrans;

      // add all of this camera's children (linked meshes) to the scene
      addAllChildren( hTrans, n, piv, data );
    }

    parent->addChild( hTrans );
  }
#if defined(KEEP_ANIMATION)
  else
  {
    // this camera is animated

    AnimatedTransformSharedPtr hAnim( AnimatedTransform::create() );
    AnimatedTransformLock anim ( hAnim );

    // calculate all the animation trafos for this camera
    Vec3f emptyPivot (0,0,0);
    constructAnimation(anim,piv,emptyPivot,&pTrack,NULL,NULL,&rollTrack, P_TRACK | ROLL_TRACK);

    // add this AnimatedTransform to the global camera location list to be postprocessed later
    m_camLocationList[n->name] = hAnim.getWeakPtr();

    // add all of this chamera's children (linked meshes) to the scene
    addAllChildren( anim, n, piv, data );

    // add this transform as a child to the group above
    parent->addChild( hAnim );
  }
#endif
}

void 
ThreeDSLoader::configurePointlight( GroupSharedPtr const& parent, Lib3dsNode *n, Vec3f &piv, Lib3dsFile *data )
{
  Lib3dsOmnilightNode *onode = (Lib3dsOmnilightNode *)n;

  Lib3dsLight *li;
  bool omnilightFound = false;
  for(int k=0; k<data->nlights; k++)
  {
    li = data->lights[k];
    if(strcmp(n->name,li->name) == 0)
    {
      omnilightFound = true;
      break;
    }
  }

  if(!omnilightFound)
  {
    // print a warning message to the console and continue
    INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","point light data for '" + std::string(n->name) + 
                                         "' could not be located in the 3ds file. Did not add this point light to the scene.\n"));
    return;
  }

  // point lights with a multiplier of zero tend to be from older files. the normal intensity of
  // the point light is between 0.5 and 1.0
  float intensity = li->multiplier ? li->multiplier : LIGHT_INTENSITY;

  // set the color of this omnilight
  Lib3dsKey *cKey = onode->color_track.keys;
  Vec3f color = intensity * Vec3f( cKey->value[0], cKey->value[1], cKey->value[2] );

  // the 3ds file format only specifies constant attenuation
  boost::array<float,3> attenuations = makeArray( FLT_EPSILON < li->attenuation ? li->attenuation : 1.0f, 0.0f, 0.0f );

  LightSourceSharedPtr pointLight = createStandardPointLight( Vec3f( 0.0f, 0.0f, 0.0f ), color, color, color, attenuations );
  pointLight->setName( li->name );
  pointLight->setEnabled( !li->off );
  pointLight->setShadowCasting( !!li->shadowed );
  
  Lib3dsTrack pTrack = onode->pos_track;

  // fill any tracks with zero or null keys to default values
  checkTracks(&pTrack, NULL, NULL, NULL);

  // if this omnilight is not animated, add the callback to the transform and we are done and we are done
#if defined(KEEP_ANIMATION)
  if(pTrack.nkeys == 1)
#endif
  {
    TransformSharedPtr hTrans( Transform::create() );
    {
      Trafo t;

      // set the position of this omnilight
      Lib3dsKey *pKey = pTrack.keys;
      
      Vec3f pos (pKey->value[0],pKey->value[1],pKey->value[2]);
      Vec3f origPos = pos - piv;
      t.setTranslation(origPos);

      hTrans->setTrafo(t);

      // add the light to the Transform
      hTrans->addChild( pointLight );

       // add all the children of this point light to the scene
       addAllChildren( hTrans, n, piv, data );
    }

    // add this transform to the parent above
    parent->addChild( hTrans );
  }
#if defined(KEEP_ANIMATION)
  else
  {
    // this omnilight is animated

    AnimatedTransformSharedPtr hAnim( AnimatedTransform::create() );
    AnimatedTransformLock anim ( hAnim );

    // construct the animation trafos for this point light
    Vec3f emptyPivot (0,0,0);
    constructAnimation(anim,piv,emptyPivot,&pTrack,NULL,NULL,NULL,P_TRACK);

    // add the light to the Transform
    anim->addChild( pointLight );

    // add all the children of this point light to the scene
    addAllChildren( anim, n, piv, data );

    // add this transform as a child to the group above
    parent->addChild( hAnim );
  }
#endif
}

void 
ThreeDSLoader::configureSpotlight( GroupSharedPtr const& parent, Lib3dsNode *n, Vec3f &piv, bool hasTarget, Lib3dsFile *data )
{
  Lib3dsSpotlightNode *snode = (Lib3dsSpotlightNode *)n;
  Lib3dsTrack pTrack = snode->pos_track;

  // search for the correct spotlight
  bool spotlightFound = false;
  Lib3dsLight *li;
  for(int i=0; i<data->nlights; i++)
  {
    li = data->lights[i];
    if(strcmp(li->name,n->name) == 0)
    {
      spotlightFound = true;
      break;
    }
  }
  
  if(!spotlightFound)
  {
    // print a warning message to the console and do not process this spotlight
    INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","spotlight data for '" + std::string(n->name) + 
                                               "' could not be located in the 3ds file. Did not add this spotlight to the scene.\n"));
    return;
  }

  // spot lights with a multiplier of zero tend to be from older files. the normal intensity of
  // the point light is between 0.5 and 1.0
  float intensity = li->multiplier ? li->multiplier : LIGHT_INTENSITY;

  // set the color of this spotlight
  Lib3dsKey *cKey = snode->color_track.keys;
  Vec3f color = intensity * Vec3f( cKey->value[0], cKey->value[1], cKey->value[2] );

  // the 3ds file format only specifies constant attenuation
  boost::array<float,3> attenuations = makeArray( FLT_EPSILON < li->attenuation ? li->attenuation : 1.0f, 0.0f, 0.0f );

  LightSourceSharedPtr spotLight = createStandardSpotLight( Vec3f( 0.0f, 0.0f, 0.0f ), Vec3f( 0.0f, 0.0f, -1.0f )
                                                          , color, color, color, attenuations
                                                          , li->falloff ? snode->falloff_track.keys->value[0] : 0.0f
                                                          , li->spot_aspect ? radToDeg( 2 * atanf((float)(0.5 * li->spot_aspect)) ) : 45.0f );
  spotLight->setName( n->name );
  spotLight->setEnabled( !li->off );
  spotLight->setShadowCasting( !!li->shadowed );

   // check if we've been given target position data
  if(!hasTarget)
  {
    // if not, spotlight is already added to parent as non-animated -> return
    return;
  }

  // add this spotlight to the global list to be postprocessed later
  m_spotList.push_back( spotLight );

  // fill any tracks with zero or null keys to default values
  checkTracks(&pTrack, NULL, NULL, NULL);

  // if this spotlight has no animation, add it to the global list and we are done
#if defined(KEEP_ANIMATION)
  if(pTrack.nkeys == 1)
#endif
  {
    TransformSharedPtr hTrans( Transform::create() );
    {
      Trafo t;
      Vec3f pos (pTrack.keys->value[0],pTrack.keys->value[1],pTrack.keys->value[2]);
      t.setTranslation(pos - piv);

      hTrans->setTrafo(t);

      // add this Transform to the global spotlight location list to be postprocessed later
      m_spotLocationList[n->name] = hTrans;

      // add the spotlight below the transform
      hTrans->addChild( spotLight );

      // add all of this spotlight's children (linked meshes) to the scene
      addAllChildren( hTrans, n, piv, data );
    }

    parent->addChild( hTrans );
  }
#if defined(KEEP_ANIMATION)
  else
  {
    // this spotlight is animated

    AnimatedTransformSharedPtr hAnim( AnimatedTransform::create() );
    AnimatedTransformLock anim ( hAnim );

    // calculate all the animation trafos for this spotlight
    Vec3f emptyPivot (0,0,0);
    constructAnimation(anim,piv,emptyPivot,&pTrack,NULL,NULL,NULL, P_TRACK );

    // add this AnimatedTransform to the global spotlight location list to be postprocessed later
    m_spotLocationList[n->name] = hAnim.getWeakPtr();

    // add all of this spotlight's children (linked meshes) to the scene
    addAllChildren( anim, n, piv, data );

    // add this transform as a child to the group above
    parent->addChild( hAnim );
  }
#endif
}


void
ThreeDSLoader::configureTarget( GroupSharedPtr const& parent, Lib3dsNode *n, Vec3f &piv, bool isCamera, Lib3dsFile *data )
{
  Lib3dsTargetNode *tnode = (Lib3dsTargetNode *)n;
  Lib3dsTrack pTrack = tnode->pos_track;

  // fill any tracks with zero or null keys to default values
  checkTracks(&pTrack, NULL, NULL, NULL);

#if 0
  if(pTrack.nkeys == 1)
#endif
  {
    // this target is not animated
    TransformSharedPtr hTrans( Transform::create() );
    {
      // we insert a transform into the scene to represent a target
      hTrans->setName(std::string(n->name) + " target");

      Trafo t;
      Vec3f pos (pTrack.keys->value[0],pTrack.keys->value[1],pTrack.keys->value[2]);

      // set the tranformation to the target's position
      t.setTranslation(pos - piv);

      hTrans->setTrafo(t);

      if(isCamera) // we're dealing with a camera
      {
        // add this Transform to the global camera target list to be postprocessed later
        m_camTargetList[n->name] = hTrans;
      }
      else // we're dealling with a spotlight
      {
        // add this Tranform to the global spotlight target list to be postprocessed later
        m_spotTargetList[n->name] = hTrans;
      }

      // add all of this target's children to the transform
      addAllChildren( hTrans, n, piv, data );
    }

    parent->addChild( hTrans );
  }
#if defined(KEEP_ANIMATION)
  else
  {
    // this target is animated
    AnimatedTransformSharedPtr hAnim( AnimatedTransform::create() );
    {
      AnimatedTransformLock anim ( hAnim );

      anim->setName(std::string(n->name) + " target");

      Vec3f emptyPivot (0,0,0);
      constructAnimation(anim, piv, emptyPivot, &pTrack, NULL, NULL, NULL, P_TRACK);

      if(isCamera) // we're dealing with a camera
      {
        // add this AnimatedTranform to the global camera target list to be postprocessed later
        m_camTargetList[n->name] = hAnim.getWeakPtr();
      }
      else // we're dealing with a spotlight
      {
        // add this AnimatedTranform to the global spotlight target list to be postprocessed later
        m_spotTargetList[n->name] = hAnim.getWeakPtr();
      }

      // add all of this target's children to the animated transform
      addAllChildren( anim, n, piv, data );
    }

    parent->addChild( hAnim );
  }
#endif
}

void 
ThreeDSLoader::checkTracks(Lib3dsTrack *pTrack, Lib3dsTrack *rTrack, Lib3dsTrack *sTrack, Lib3dsTrack *rollTrack)
{
  // default position is (0,0,0)
  if(pTrack && (pTrack->keys==NULL || pTrack->nkeys <= 0))
  {
    lib3ds_track_resize(pTrack, 1);
    DP_ASSERT( pTrack->keys );    
    if ( pTrack->keys )   
    {
      pTrack->keys->value[0] = 0;
      pTrack->keys->value[1] = 0;
      pTrack->keys->value[2] = 0;
    }
  }

  // default orientation is (0,0,0,1) (unit quaternion)
  if(rTrack && (rTrack->keys==NULL || rTrack->nkeys <= 0))
  {
    lib3ds_track_resize(rTrack, 1);
    DP_ASSERT( rTrack->keys );
    if ( rTrack->keys )
    {
      rTrack->keys->value[0] = 0;
      rTrack->keys->value[1] = 0;
      rTrack->keys->value[2] = 0;
      rTrack->keys->value[3] = 1;
    }
  }

  // default scale is (1,1,1)
  if(sTrack && (sTrack->keys==NULL || sTrack->nkeys <= 0))
  {
    lib3ds_track_resize(sTrack, 1);
    DP_ASSERT( sTrack->keys );
    if ( sTrack->keys )
    {
      sTrack->keys->value[0] = 1;
      sTrack->keys->value[1] = 1;
      sTrack->keys->value[2] = 1;
    }
  }

  // default roll is 0
  if(rollTrack && (rollTrack->keys==NULL || rollTrack->nkeys <= 0))
  {
    lib3ds_track_resize(rollTrack, 1);
    DP_ASSERT( rollTrack->keys );    
    if ( rollTrack->keys )
    {
      rollTrack->keys->value[0] = 0;
    }
  }
}

int
ThreeDSLoader::addAllChildren( GroupSharedPtr const& parent, Lib3dsNode *n, Vec3f &piv, Lib3dsFile *data )
{
  int childCount = 0;

  Lib3dsNode *child = n->childs;
  while(child) 
  {
    buildTree( parent, child, piv, data );
    childCount++;
    child = child->next;
  }

  return childCount;
}

Lib3dsNode *
ThreeDSLoader::searchNodeTree( Lib3dsNode *n, int nodeType, char *nodeName)
{
  if(!n)
  {
    // base case; we have reached end of list
    return NULL;
  }

  if(n->type == nodeType && strcmp(nodeName,n->name) == 0)
  {
    // this node is the one we're looking for; return it
    return n;
  }
  else
  {
    Lib3dsNode *found = searchNodeTree(n->next, nodeType, nodeName);

    return found ? found : searchNodeTree(n->childs, nodeType, nodeName);
  }
}

void
ThreeDSLoader::constructFlatHierarchy( Lib3dsFile * data )
{
  Lib3dsMesh *m = data->meshes[0];

  // add the first mesh to the top level group of the file
  Lib3dsNode * startNode = (Lib3dsNode *)lib3ds_node_new_mesh_instance(m,m->name,NULL,NULL,NULL);
  Lib3dsNode * node = startNode;

  // iterate over the rest of the meshes and add them to the top level group as siblings of each other
  for(int i=1; i<data->nmeshes; i++)
  {
    m = data->meshes[i];
    node->next = (Lib3dsNode *)lib3ds_node_new_mesh_instance(m,m->name,NULL,NULL,NULL);
    node = node->next;
  }

  node->next = data->nodes;
  data->nodes = startNode;
}

void
ThreeDSLoader::vecInterp(Vec3f &target, Vec3f &lData, Vec3f &rData, int leftFrame, int rightFrame, int currFrame)
{
  // if we're given coincident frames, return the left one
  if(rightFrame==leftFrame) 
  {
    target = lData; 
    return;
  }

  // linearly interpolate between the two key values
  float alpha = ((float)(currFrame-leftFrame))/(rightFrame-leftFrame);
  target = lerp( alpha, lData, rData);
}

#if defined(KEEP_ANIMATION)
void
ThreeDSLoader::constructAnimation( AnimatedTransform *anim, Vec3f &parentPivot, Vec3f &pivot, Lib3dsTrack *pTrack, Lib3dsTrack *rTrack, Lib3dsTrack *sTrack, 
                                                                   Lib3dsTrack *rollTrack, int flags)
{
  FramedTrafoAnimationDescriptionSharedPtr hDesc = FramedTrafoAnimationDescription::create();
  FramedTrafoAnimationDescriptionLock desc( hDesc );
  
  Vec3f accumPivot = pivot - parentPivot;

  // determine which types of animation we are including
  bool position = (flags & P_TRACK) != 0;
  bool rotation = (flags & R_TRACK) != 0;
  bool scale = (flags & S_TRACK) != 0;
  bool roll = (flags & ROLL_TRACK) != 0;

  // pointers to the current and next frames of each animation type
  Lib3dsKey *pKey,*pNext;
  Lib3dsKey *rKey,*rNext;
  Lib3dsKey *sKey,*sNext;
  Lib3dsKey *rollKey,*rollNext;

  // the values of the current and next frames of each animation type
  Vec3f currPos,nextPos;
  Quatf currRot,nextRot;
  Vec3f currScl,nextScl;
  float currRoll,nextRoll;

  bool pComplete,rComplete,sComplete,rollComplete; // flags determining whether we've reached the last frame of each type
  int pKeysPassed,rKeysPassed,sKeysPassed,rollKeysPassed; // keep track of how many keys of each type we've passed

  if(position)
  {
    // we are animating position
    pKey = pTrack->keys;

    // check if there is more than one frame
    if(pTrack->nkeys>1)
    {
      pNext = pKey + 1;
      pComplete = false;
    }
    else
    {
      pNext = pKey;
      pComplete = true;
    }
    pKeysPassed = 1;

    // calculate the positions of the current and next frame (relative to the pivot)
    currPos = Vec3f(pKey->value[0],pKey->value[1],pKey->value[2]) - accumPivot;
    nextPos = Vec3f(pNext->value[0],pNext->value[1],pNext->value[2]) - accumPivot;
  }

  if(rotation)
  {
    // we are animating rotation
    rKey = rTrack->keys;

    // check if there is more than one frame
    if(rTrack->nkeys>1)
    {
      rNext = rKey + 1;
      rComplete = false;
    }
    else
    {
      rNext = rKey;
      rComplete = true;
    }
    rKeysPassed = 1;

    // avoid the edge case where 3ds max sets the first rotation frame to a null quaternion
    Vec3f axis (rKey->value[0],rKey->value[1],rKey->value[2]);
    if(length(axis)==0)
    {
      axis[0] = 1;
    }

    // at all times, we will be rotated on the interval [currRot,nextRot)
    // note that these are negated; 3ds max saves rotations in the negative direction
    currRot = Quatf(axis,-rKey->value[3]);
    nextRot = currRot * Quatf(Vec3f(rNext->value[0],rNext->value[1],rNext->value[2]),-rNext->value[3]);
  }

  if(scale)
  {
    // we are animating scale
    sKey = sTrack->keys;

    // check if there is more than one frame
    if(sTrack->nkeys>1)
    {
      sNext = sKey + 1;
      sComplete = false;
    }
    else
    {
      sNext = sKey;
      sComplete = true;
    }
    sKeysPassed = 1;

    // calculate the scales of the current and next frame
    currScl = Vec3f(sKey->value[0],sKey->value[1],sKey->value[2]);
    nextScl = Vec3f(sNext->value[0],sNext->value[1],sNext->value[2]);
  }

  if(roll)
  {
    // we are animating roll
    rollKey = rollTrack->keys;

    // check if there is more than one frame
    if(rollTrack->nkeys>1)
    {
      rollNext = rollKey + 1;
      rollComplete = false;
    }
    else
    {
      rollNext = rollKey;
      rollComplete = true;
    }
    rollKeysPassed = 1;

    // calculate the roll of the current and next frame
    currRoll = rollKey->value[0];
    nextRoll = rollNext->value[0];
  }

  // reserve the correct number of frames
  int numFrames = m_numFrames;
  desc->reserveSteps(numFrames);

  // create the animation trafos
  for(int k=0; k<numFrames; k++)
  {
    Trafo t;
    Trafo tUnlink;

    t.setCenter(pivot);
    
    // add position animation if required
    if(position)
    {
      // if this is the last position key, use it; otherwise interpolate
      Vec3f trans;
      if(pComplete)
      {
        trans = currPos;
      }
      else
      {
        vecInterp(trans, currPos, nextPos, pKey->frame, pNext->frame, k);
      }

      t.setTranslation(trans);

      // if we've reached our right position frame, move to the next interval
      if((!pComplete)&&(pNext->frame == k))
      {
        pKey = pNext;
        pNext = pNext+1;
        pKeysPassed++;
        
        // check if we've reached the last position frame
        if(pKeysPassed>pTrack->nkeys-1) 
        {
          pNext = pKey;
          pComplete = true;
        }
        
        // update the frame positions
        currPos = nextPos;
        nextPos = Vec3f(pNext->value[0],pNext->value[1],pNext->value[2]) - accumPivot;
      }
    } // position

    // add roll animation if required
    if(roll)
    {
      Vec3f dir (0,0,1); // default z-axis to compute roll orientation

      Quatf qa (dir, degToRad(currRoll));
      // if we're at the last roll frame, set the orientation with this roll; otherwise, interpolate
      if(rollComplete)
      {
        t.setOrientation(qa);
      }
      else
      {
        Quatf qb (dir,degToRad(nextRoll));

        float alpha = ((float)(k-rollKey->frame))/(rollNext->frame-rollKey->frame);
        Quatf rot = lerp(alpha, qa, qb);
        rot.normalize();

        t.setOrientation(rot);
      }

       // if we've reached our right roll frame, move to the next interval
      if((!rollComplete)&&(rollNext->frame == k))
      {
        rollKey = rollNext;
        rollNext = rollNext+1;
        rollKeysPassed++;

        // check if we've reached the last roll frame
        if(rollKeysPassed>rollTrack->nkeys-1) 
        {
          rollNext = rollKey;
          rollComplete = true;
        }

        currRoll = nextRoll;
        nextRoll = rollNext->value[0];
      }
    } // roll
    
    // add rotation animation if required
    // note: 3ds saves rotation animation as incremental rotations - how much the rotation has
    // changed since the last frame
    // rotation and roll will never both be present, so we don't need to worry about double-setting the
    // orientation
    if(rotation)
    {
      Quatf rot; // the calculated rotation
      if(rComplete)
      {
        // we are at our last rotation frame, use it
        rot = currRot;
      }
      else
      {
        // linear interpolation
        float alpha = ((float)(k-rKey->frame))/(rNext->frame-rKey->frame);
        rot = lerp(alpha, currRot, nextRot);
      }

      t.setOrientation(rot);

      // if we've reached our right rotation frame, move to the next interval
      if((!rComplete)&&(rNext->frame == k))
      {
        rKey = rNext;
        rNext = rNext+1;
        
        // check if we've reached the last rotation frame
        if(rKeysPassed>rTrack->nkeys-1) 
        {
          rNext = rKey;
          rComplete = true;
        }

        // update frame rotations. we rotate the next frame by our current orientation because 3ds files 
        // use incremental (delta) angles, rather than absolute ones
        currRot = t.getOrientation();
        nextRot = currRot * Quatf(Vec3f(rNext->value[0],rNext->value[1],rNext->value[2]),-rNext->value[3]);
      }
    }

    // add scaling animation if required
    if(scale)
    {
      // if this is the last scaling key, use it; otherwise, interpolate
      Vec3f scale;
      if(sComplete)
      {
        scale = currScl;
      }
      else
      {
        vecInterp(scale, currScl, nextScl, sKey->frame, sNext->frame, k);
      }

      t.setScaling(scale);
   
      // if we've reached our right scaling frame, move to the next interval
      if((!sComplete)&&(sNext->frame == k))
      {
        sKey = sNext;
        sNext = sNext+1;
        sKeysPassed++;
        
        // check if we've reached the last scaling frame
        if(sKeysPassed>sTrack->nkeys-1) 
        {
          sNext = sKey;
          sComplete = true;
        }

        currScl = nextScl;
        nextScl = Vec3f(sNext->value[0],sNext->value[1],sNext->value[2]);
      }
    }


    // add the trafo to the animation
    desc->addStep(t);
  }

  // put the description into an animation
  TrafoAnimationSharedPtr hTrafo = TrafoAnimation::create();
  TrafoAnimationLock (hTrafo)->setDescription( hDesc );

  // add the animation
  anim->setAnimation( hTrafo );
}
#endif

bool
ThreeDSLoader::constructGeometry(GroupSharedPtr const& group, char *name, Lib3dsFile *data) 
{  
  DP_ASSERT(data && data->meshes); // should already be true!

  Lib3dsMesh *currMesh = NULL;
  bool meshFound = false;

  for(int k=0; k<data->nmeshes; k++) 
  {
    currMesh = data->meshes[k];
    if(currMesh && strcmp(currMesh->name,name) == 0) 
    {
      meshFound = true;
      break;
    }

  }

  if(!meshFound) 
  {
    // no matching mesh has been found, return false
    return false;
  }

  group->setName( name );

  unsigned int j;
  unsigned int nVertices = currMesh->nvertices;
  unsigned int nFaces = currMesh->nfaces;

  if(nVertices==0)
  {
    // this degenerate mesh has no vertices - don't add it. print a warning to the console and return
    INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","mesh " + std::string(currMesh->name) +
                                               " has 0 vertices. It was not added to the scene.\n"));
    return false;
  }
  else if(nFaces==0)
  {
    // this degenerate mesh has no faces - don't add it. print a warning to the console and return
    INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","mesh " + std::string(currMesh->name) +
                                               " has 0 faces. It was not added to the scene.\n"));
    return false;
  }

  bool hasMaterial = false;
  bool hasTexture = (currMesh->texcos != NULL);

  // construct the tranformation matrix for this mesh in order to un-transform all of its vertices
  Mat44f meshMatrixInv ( makeArray( currMesh->matrix[0][0], currMesh->matrix[0][1], currMesh->matrix[0][2], currMesh->matrix[0][3],
                                    currMesh->matrix[1][0], currMesh->matrix[1][1], currMesh->matrix[1][2], currMesh->matrix[1][3],
                                    currMesh->matrix[2][0], currMesh->matrix[2][1], currMesh->matrix[2][2], currMesh->matrix[2][3],
                                    currMesh->matrix[3][0], currMesh->matrix[3][1], currMesh->matrix[3][2], currMesh->matrix[3][3] ) );


  meshMatrixInv.invert();               

  // allocate a vector with enough space for all verts in the mesh
  std::vector<Vec3f> vertex(nVertices);

  // allocate a vector with enough space for all texture coordinates in the mesh
  std::vector<Vec2f> texCoords(nVertices);

  float (*v)[3] = currMesh->vertices;
  float (*tex)[2] = currMesh->texcos;

  // transform all matrices to local as opposed to global coordinates
  for(j=0; j<nVertices; j++) 
  {
    Vec4f augmented (v[j][0],v[j][1],v[j][2],1); 
    Vec4f transformed = augmented * meshMatrixInv;
    setVec(vertex[j], transformed[0],
          transformed[1],
          transformed[2]);

    // if there are textures present, add the texture coordinates for the jth vertex
    if(hasTexture)
    {
      float f0 = tex[j][0];
      float f1 = tex[j][1];
      
      // check if the texture coordinates contain invalid numbers; set them to zero
      if(abs(f0) > CORRUPTED_BUFFER)
      {
        // warn the user that we've replaced the invalid texco with zero
        INVOKE_CALLBACK(onInvalidValue( j, "texture coordinate value", "mesh->texcos[j][0]", f0 ));

        f0 = 0.f;
      }
      if(abs(f1) > CORRUPTED_BUFFER)
      {
        // warn the user that we've replaced the invalid texco with zero
        INVOKE_CALLBACK(onInvalidValue( j, "texture coordinate value", "mesh->texco[j][1]", f1 ));

        f1 = 0.f;
      }

      setVec(texCoords[j], f0, f1);
    }
  }

  // allocate a vector with enough space for all faces in the mesh
  vector<unsigned int> indices(3*nFaces);

  map<int,int> smoothDict; // a map of smoothing group number to index in the smoothGroups vector
  int nextSlot = 0; // next available index number in the smoothGroups vector
  int nextIndex = nVertices; // the next vertex index if we have to add more
  
  vector < SmoothingData > smoothGroups; 
 
  // keep track of whether each vertex has already been used in a smoothing group
  vector <bool> vertexUsed (nVertices, false);

  Lib3dsFace *currFace = currMesh->faces;

  for(j=0; j<nFaces; j++)
  {
    int group = currFace->smoothing_group;

    // update the material flag with whether this face is associated with a material
    hasMaterial = hasMaterial || (currFace->material >= 0);

    indices[3*j+0] = currFace->index[0];
    indices[3*j+1] = currFace->index[1];
    indices[3*j+2] = currFace->index[2];

    // calculate the face normal for this face
    Vec3f one = vertex[indices[3*j+1]] - vertex[indices[3*j+0]];
    Vec3f two = vertex[indices[3*j+2]] - vertex[indices[3*j+0]];
    Vec3f faceNorm = one ^ two;
    faceNorm.normalize();

    // if there are no entries for this smoothing group, create a new vertex list for it
    if(smoothDict[group]==0)
    {
      SmoothingData smooth = {  vector<Vec3f>(), // vertices
                                vector<Vec3f>(), // normals
                                vector<Vec2f>(), // texCoords
                                vector<int>(), // order
                                vector<int>(nVertices,0)  }; // vertMap
      smoothGroups.push_back( smooth );
      
      nextSlot++;
      smoothDict[group] = nextSlot;
    }

    int listNum = smoothDict[group] - 1;
    SmoothingData *smooth = &smoothGroups[listNum];

    for(int k=0; k<3; k++)
    {
      int idx = currFace->index[k];
      int vertNum = (smooth->vertMap)[idx];

      if(vertNum == 0)
      {
        // we haven't recorded this vertex; add the face normal to the normal list for this smoothing group
        smooth->normals.push_back(faceNorm);

        // check if we've already used this vertex in another group
        if(!vertexUsed[idx])
        {
          // flag that the vertex has been used in its same index in the group
          (smooth->vertMap)[idx] = 1;

          // save this index to the smoothing group's ordered vertex list
          smooth->order.push_back(idx);
          
          // indicate that this vertex has been used in a group
          vertexUsed[idx] = true;

          // add a copy of this vertex to the smoothing group vertex list
          smooth->vertices.push_back(Vec3f(vertex[idx]));
          
          if(hasTexture)
          {
            // add the associated texture coordinate for this vertex to the smoothing group texcoords list
            smooth->texCoords.push_back(texCoords[idx]);
          }
        }
        else // we've already used the vertex, we need to add another copy of it to the end of the list
        {
          int newIndex = nextIndex;

          Vec3f copied (vertex[idx]);

          // add a copy of this index to the end of the global vertex list
          vertex.push_back(copied);

          // store where we saved the new vertex
          (smooth->vertMap)[idx] = newIndex;

          // save this index to the smoothing group's ordered vertex list
          smooth->order.push_back(newIndex);

          smooth->vertices.push_back(copied);

          if(hasTexture)
          {
            // if we have textures, do the same procedure for the texture coordinates
            Vec2f copiedTex (texCoords[idx]);
            texCoords.push_back(copiedTex);
            smooth->texCoords.push_back(copiedTex);
          }

          // change the index in the face definition
          indices[3*j+k] = newIndex;
          nextIndex++;
        }
      }
      else if(vertNum == 1)
      {
        // we've stored this vertex in this group under the same index
        // we don't have to do anything here
      }
      else
      {
        // vertNum holds the new index of the vertex in this group
        // adjust the face index to point to it
        indices[3*j+k] = vertNum;
      }
    }
    ++currFace;
  }

  IndexSetSharedPtr iset( IndexSet::create() );
  iset->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );

  // calculate the vertex normals for each smoothing group in this mesh
  for(j=0; j<smoothGroups.size(); j++)
  {
    SmoothingData *smooth = &smoothGroups[j];

    DP_ASSERT(smooth->vertices.size() == smooth->normals.size());

    Vec3f *verts = (Vec3f *)&(smooth->vertices[0]);
    Sphere3f bs = boundingSphere(verts,(int)smooth->vertices.size());

    // handle the edge case where all vertices of the smoothing group were coincident
    if(bs.getRadius() == 0.f)
    {
      bs.setRadius(FLT_EPSILON);
    }

    smoothNormals(smooth->vertices,bs,CREASE_ANGLE,smooth->normals);
  }

  // create the master list of vertex normals
  vector<Vec3f> normals (vertex.size());

  currFace = currMesh->faces;

  for(j=0; j<smoothGroups.size(); j++)
  {
    vector<int> order = smoothGroups[j].order;
    for(unsigned int k=0; k<order.size(); k++)
    {
      normals[order[k]] = (smoothGroups[j].normals)[k];
    }
  }

  // update the number of vertices in the mesh
  nVertices = (int)vertex.size();

  if(m_numMaterials == 0 || !hasMaterial)
  {
    VertexAttributeSetSharedPtr hVas(VertexAttributeSet::create());
    hVas->setVertices( &vertex[0], dp::checked_cast<unsigned int>( vertex.size() ) );
    hVas->setNormals( &normals[0], dp::checked_cast<unsigned int>( normals.size() ) );

    // there are no materials present; export this whole mesh as one Primitive
    // create the triangle group to hold the vertex data, faces, and possibly texture coordinates
    PrimitiveSharedPtr hTris( Primitive::create( PRIMITIVE_TRIANGLES ) );
    hTris->setIndexSet( iset );
    hTris->setVertexAttributeSet( hVas );

    // add this Triangles Primitive to the Group along with a NULL stateset indicating no textures/materials
    GeoNodeSharedPtr geoNode = GeoNode::create();
    geoNode->setPrimitive( hTris );

    group->addChild( geoNode );

    return true;
  }
  else
  {
    // material/textures are present in this mesh

    int numMaterialGroups = m_numMaterials + 1; // we add an extra group for faces with no material

    vector < MatGroupData > matGroups (numMaterialGroups);     // a list of the data for each material group

    vector <int> matIndices; // a list of the material indices we process
    
    // an array of flags indicating whether or not a list of vertices for each material has been created
    vector <bool> listCreated (numMaterialGroups,false);

    Lib3dsFace *currFace = currMesh->faces;

    for(j=0; j<nFaces; j++) 
    {
      // get the material index of this face
      int mat = currFace->material;

      if(mat<0)
      {
        // we have found a face with no material; assign it to the last material group
        mat = m_numMaterials;
      }

      // check if we have initialized lists for this type of material
      if(!listCreated[mat])
      {
        // if not, instantiate lists
        matIndices.push_back(mat);

        MatGroupData group = {  vector<Vec3f>(), // vertices
                                vector<Vec3f>(), // normals
                                vector<Vec2f>(), // texCoords
                                vector<int>(),   // visibilities
                                vector<unsigned int>(), // indices
                                vector<int>(nVertices,-1)   }; // vertMap
        // add the data to the groups list
        matGroups[mat] = group;

        // indicate that we've instantiated the lists now
        listCreated[mat] = true;
      }

      MatGroupData *group = &matGroups[mat];

      for(int i=0; i<3; i++)
      {
        int index = (group->vertMap)[indices[3*j+i]];
        if(index<0)
        {
           // we haven't stored this vertex for this material before; add it and update the index
           group->vertices.push_back(vertex[indices[3*j+i]]);

           // also add the vertex's normal to the normal list
           group->normals.push_back(normals[indices[3*j+i]]);

           // if there are textures present, add the vertex's texture coordinates to the list
           if(hasTexture)
           {
              group->texCoords.push_back(texCoords[indices[3*j+i]]);
           }

           int newIndex = (int)group->vertices.size()-1;
           (group->vertMap)[indices[3*j+i]] = newIndex;
           group->indices.push_back( newIndex );
        }
        else
        {
           group->indices.push_back( index );
        }
      }

      // add this flag's visibility flags if there are wireframes in the scene
      if(m_isWire[mat])
      {
        group->visibilities.push_back(currFace->flags);
      }

      ++currFace;  
    }

    for(unsigned int k=0; k<matIndices.size(); k++)
    {
      // for each material, add a Primitive/StateSet pair to the GeoNode

      int mat = matIndices.at(k);

      MatGroupData *matGroup = &matGroups[mat];

      PrimitiveSharedPtr primitive;
      { 
        VertexAttributeSetSharedPtr hVas(VertexAttributeSet::create());

        hVas->setVertices(&(matGroup->vertices.at(0)), dp::checked_cast<unsigned int>( matGroup->vertices.size() ) );
        hVas->setNormals(&(matGroup->normals.at(0)), dp::checked_cast<unsigned int>( matGroup->normals.size() ) );
        if(hasTexture)
        {
          // if there was a texture, it was bound to texunit 0
          hVas->setTexCoords(0, &(matGroup->texCoords.at(0)), dp::checked_cast<unsigned int>( matGroup->texCoords.size() ) );
        }
        else if(m_hasTexture[mat])
        {
           // we're missing texture coordinates for a mesh with a texture
           INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","Mesh '" + std::string(currMesh->name) +
                                               "' did not contain texture coordinates but was assigned a texture. Texture may not show correctly.\n"));
        }

        // if this material is supposed to be a wireframe, the primitive set is a LineStrips, otherwise it's a Triangles
        if(m_isWire[mat])
        {
          vector<unsigned int> indices;
          vector<int>* vVec = &(matGroup->visibilities);
          DP_ASSERT( matGroup->indices.size() % 3 == 0 );
          int numFaces = dp::checked_cast<int>(matGroup->indices.size() / 3);

          for(int k=0; k<numFaces; k++)
          {
            int visFlags = vVec->at(k);

            // read the face flags to see which edges should be visible in the wireframe
            // A is the first vertex in the face, B the second, C the third
            bool showAB = (visFlags & LIB3DS_FACE_VIS_AB) > 0;
            bool showAC = (visFlags & LIB3DS_FACE_VIS_AC) > 0;
            bool showBC = (visFlags & LIB3DS_FACE_VIS_BC) > 0;

            // add lines to the segment list depending on which edges are visible
            if(showAB)
            {
              indices.push_back( matGroup->indices[3*k+0] );
              indices.push_back( matGroup->indices[3*k+1] );
            }

            if(showBC)
            {
              indices.push_back( matGroup->indices[3*k+1] );
              indices.push_back( matGroup->indices[3*k+2] );
            }

            if(showAC)
            {
              indices.push_back( matGroup->indices[3*k+2] );
              indices.push_back( matGroup->indices[3*k+0] );
            }
          }

          PrimitiveSharedPtr hLines(Primitive::create( PRIMITIVE_LINES ));
          if ( !indices.empty() )
          {
            // add the list of segments to the Lines object
            IndexSetSharedPtr iset( IndexSet::create() );
            iset->setData( &indices[0], dp::checked_cast<unsigned int>(indices.size()) );
            hLines->setIndexSet( iset );
          }
          // add the vertex data to the LineStrips object
          hLines->setVertexAttributeSet( hVas );

          primitive = hLines;
        }
        else
        {
          PrimitiveSharedPtr hTris = Primitive::create( PRIMITIVE_TRIANGLES );
          // add the list of faces to the Triangles object
          IndexSetSharedPtr iset( IndexSet::create() );
          iset->setData( &matGroup->indices[0], dp::checked_cast<unsigned int>(matGroup->indices.size()) );
          hTris->setIndexSet( iset );
          // add the vertex data to the Triangles object
          hTris->setVertexAttributeSet( hVas );

          primitive = hTris;
        }
      }
      DP_ASSERT( primitive );

      // check if we are processing the group of faces with no materials
      GeoNodeSharedPtr geoNode = GeoNode::create();
      if ( mat != m_numMaterials )
      {
        geoNode->setMaterialEffect( m_materials[mat] );
      }
      geoNode->setPrimitive( primitive );

      group->addChild( geoNode );
    }

    return true;
  } //end if materials present
}

void
ThreeDSLoader::postProcessCamerasAndLights()
{
  vector <PerspectiveCameraSharedPtr>::iterator camIter = m_camList.begin();
  
  // process each camera in the global list
  while(camIter != m_camList.end())
  {
    std::string camName = (*camIter)->getName();

    // check if we've stored both a location and target for this camera in the global maps
    TransformSharedPtr const& camLoc = m_camLocationList[camName];
    TransformSharedPtr const& camTarget = m_camTargetList[camName];

    // if both the location and target were found, add the callbacks
    if(camLoc && camTarget)
    {
      DP_ASSERT(!"callbacks not supported");
    }
    else
    {
      // print a warning to the console and continue
      INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","Camera " + camName +
                                               " did not have both location and target at postprocess. It may be configured incorrectly.\n"));
    }

    ++camIter;
  } 

  vector <LightSourceSharedPtr>::iterator spotIter = m_spotList.begin();
  
  // process each spotlight in the global list
  while(spotIter != m_spotList.end())
  {
    std::string spotName = (*spotIter)->getName();

    // check if we've stored both a location and target for this spotlight in the global maps
    TransformSharedPtr const& spotLoc = m_spotLocationList[spotName];
    TransformSharedPtr const& spotTarget = m_spotTargetList[spotName];

    // if both the location and target were found, add the callbacks
    if(spotLoc && spotTarget)
    {
      DP_ASSERT(!"callbacks not supported");
    }
    else
    {
      // print a warning to the console and continue
      INVOKE_CALLBACK(onUnLocalizedMessage("3DSLoader warning","Spotlight " + spotName +
                                               " did not have both location and target at postprocess. It may be configured incorrectly.\n"));
    }

    ++spotIter;
  } 
}

void
ThreeDSLoader::constructMaterials( std::vector<dp::sg::core::EffectDataSharedPtr> & materials, Lib3dsFile *data )
{
  m_wirePresent = false;

  // iterate over all materials saved in the 3ds file
  for(int i=0; i<m_numMaterials; i++)
  {
    Lib3dsMaterial *m = data->materials[i];
    
    // check if this material is supposed to be a wireframe
    bool usesWire = (m->use_wire == 1);
    m_wirePresent = m_wirePresent || usesWire;
    m_isWire[i] = usesWire;

    Lib3dsTextureMap texMap = m->texture1_map;
    std::string texName(texMap.name);
    std::string reflectName(m->reflection_map.name);
    bool diffuseTexture = false;
    bool reflMap = false;
    
    // if there is a diffuse texture map then flag that textures are present
    if(texName.length()>0)
    {
      diffuseTexture = true;
    }

    // if there is a diffuse texture map then flag that textures are present
    if(reflectName.length()>0)
    {
      reflMap = true;
    }

    // create the material effect
    materials[i] = EffectData::create( getStandardMaterialSpec() );
    {
      EffectDataSharedPtr const& me = materials[i];
      me->setName( m->name );

      const dp::fx::EffectSpecSharedPtr & es = me->getEffectSpec();
      dp::fx::EffectSpec::iterator pgsit = es->findParameterGroupSpec( string( "standardMaterialParameters" ) );
      DP_ASSERT( pgsit != es->endParameterGroupSpecs() );
      ParameterGroupDataSharedPtr materialParameters = ParameterGroupData::create( *pgsit );
      {
        const dp::fx::ParameterGroupSpecSharedPtr & pgs = materialParameters->getParameterGroupSpec();

        Vec3f ambientColor( m->ambient[0], m->ambient[1], m->ambient[2] );
        Vec3f diffuseColor( m->diffuse[0], m->diffuse[1], m->diffuse[2] );
        if ( diffuseTexture && ( length( diffuseColor ) == 0 ) )
        {
          // if a diffuse texture is present, we want to set the material's diffuse color to white
          diffuseColor = Vec3f( 1.0f, 1.0f, 1.0f );
        }
        else if ( diffuseTexture && ( length( ambientColor ) == 0 ) )
        {
          // if we have a black ambient color, set it to the diffuse color
          ambientColor = diffuseColor;
        }
        Vec3f specularColor( m->specular[0], m->specular[1], m->specular[2] );
        // scale of [0,1] intensity of emissive color
        Vec3f emissiveColor( m->self_illum, m->self_illum, m->self_illum );

        // set the shininess of the material (glossiness in Max)
        // note: there is also a shin_strength variable correlating to the Max "specular level" variable which
        // increases the width of specular highlights. we do not know how this maps to OpenGL functions
        // rescale shininess from scale [0.0,1.0] to [0.0,MAX_SHINE]
        float specularExponent = MAX_SHINE * ( 1.0f - m->shininess );
        float opacity = 1.0f - m->transparency;

        DP_VERIFY( materialParameters->setParameter( "frontAmbientColor", ambientColor ) );
        DP_VERIFY( materialParameters->setParameter( "frontDiffuseColor", diffuseColor ) );
        DP_VERIFY( materialParameters->setParameter( "frontSpecularColor", specularColor ) );
        DP_VERIFY( materialParameters->setParameter( "frontEmissiveColor", emissiveColor ) );
        DP_VERIFY( materialParameters->setParameter( "frontSpecularExponent", specularExponent ) );
        DP_VERIFY( materialParameters->setParameter( "frontOpacity", opacity ) );
        if ( m->two_sided )
        {
          DP_VERIFY( materialParameters->setParameter( "backAmbientColor", ambientColor ) );
          DP_VERIFY( materialParameters->setParameter( "backDiffuseColor", diffuseColor ) );
          DP_VERIFY( materialParameters->setParameter( "backSpecularColor", specularColor ) );
          DP_VERIFY( materialParameters->setParameter( "backEmissiveColor", emissiveColor ) );
          DP_VERIFY( materialParameters->setParameter( "backSpecularExponent", specularExponent ) );
          DP_VERIFY( materialParameters->setParameter( "backOpacity", opacity ) );
        }
        DP_VERIFY( materialParameters->setParameter<bool>( "twoSidedLighting", m->two_sided != 0 ) );
      }
      me->setParameterGroupData( pgsit, materialParameters );

      me->setTransparent( 0.0f < m->transparency );

      if ( diffuseTexture || reflMap )
      {
        const dp::fx::EffectSpecSharedPtr & es = me->getEffectSpec();
        dp::fx::EffectSpec::iterator pgsit = es->findParameterGroupSpec( string( "standardTextureParameters" ) );
        DP_ASSERT( pgsit != es->endParameterGroupSpecs() );

        // this material includes a texture
        m_hasTexture[i] = true;

        if ( diffuseTexture )
        {
          int idx = (int)texName.length()-4;
          if(texName.substr(idx) == ".CEL")
          {
             texName.replace(idx,4,".GIF");
          }

          ParameterGroupDataSharedPtr textureParameters = createTexture( m->texture1_map, m_fileFinder, texName, false );
          if ( textureParameters )
          {
            me->setParameterGroupData( pgsit, textureParameters );
          }
          else
          {
            INVOKE_CALLBACK( onFileNotFound( texName ) );
          }
        }

        if ( reflMap )
        {
          // attempt to create reflection map texture
          ParameterGroupDataSharedPtr textureParameters = createTexture( m->reflection_map, m_fileFinder, reflectName, true );
          if ( textureParameters )
          {
            me->setParameterGroupData( pgsit, textureParameters );
          }
          else
          {
            INVOKE_CALLBACK( onFileNotFound( reflectName ) );
          }
        }
      }
    }

    // if the wire width of the wireframe is not the default 1.f, add a geometry effect
    if ( usesWire && ( 0.0f < m->wire_size ) && ( 1.0f != m->wire_size ) )
    {
      dp::fx::EffectSpecSharedPtr const & effectSpec = materials[i]->getEffectSpec();
      dp::fx::EffectSpec::iterator pgsit = effectSpec->findParameterGroupSpec( string( "standardGeometryParameters" ) );
      DP_ASSERT( pgsit != effectSpec->endParameterGroupSpecs() );

      dp::sg::core::ParameterGroupDataSharedPtr geometryParameters = dp::sg::core::ParameterGroupData::create( *pgsit );
      DP_VERIFY( geometryParameters->setParameter( "lineWidth", m->wire_size ) );
    }
  }
}

ParameterGroupDataSharedPtr ThreeDSLoader::createTexture( Lib3dsTextureMap &texture
                                                        , dp::util::FileFinder const& fileFinder
                                                        , const std::string & filename
                                                        , bool isEnvMap )
{
  ParameterGroupDataSharedPtr parameterGroupData;

  // first, create texture from file name
  TextureHostSharedPtr textureHost = dp::sg::io::loadTextureHost( filename, fileFinder );
  if ( textureHost )
  {
    TextureWrapMode wrapS, wrapT;

    // set the correct texture wrap modes depending on whether this map should be tiled or mirrored
    if ( texture.flags & LIB3DS_TEXTURE_NO_TILE )
    {
      if ( texture.flags & LIB3DS_TEXTURE_MIRROR )
      {
        wrapS = wrapT = TWM_MIRROR_CLAMP;
      }
      else
      {
        wrapS = wrapT = TWM_CLAMP;
      }
    }
    else
    {
      if ( texture.flags & LIB3DS_TEXTURE_MIRROR )
      {
        wrapS = wrapT = TWM_MIRROR_REPEAT;
      }
      else
      {
        wrapS = wrapT = TWM_REPEAT;
      }
    }

    SamplerSharedPtr sampler = Sampler::create( textureHost );
    sampler->setWrapMode( TWCA_S, wrapS );
    sampler->setWrapMode( TWCA_T, wrapT );

    // create a standardTexturesParameter ParameterGroupData
    parameterGroupData = createStandardTextureParameterData( sampler );
    parameterGroupData->setName( filename );

    Mat44f trafo(  makeArray(  texture.scale[0],              0.0f, 0.0f, 0.0f
                            ,              0.0f,  texture.scale[1], 0.0f, 0.0f
                            ,              0.0f,              0.0f, 1.0f, 0.0f
                            , texture.offset[0], texture.offset[1], 0.0f, 1.0f ) );
    DP_VERIFY( parameterGroupData->setParameter( "textureMatrix", trafo ) );

    if ( isEnvMap )
    {
      static boost::array<dp::fx::EnumSpec::StorageType,4> texGenMode( makeArray<dp::fx::EnumSpec::StorageType>( TGM_REFLECTION_MAP, TGM_REFLECTION_MAP, TGM_REFLECTION_MAP, TGM_OFF ) );

      DP_VERIFY( (parameterGroupData->setParameterArray<dp::fx::EnumSpec::StorageType,4>( "genMode", texGenMode )) );
      DP_VERIFY( parameterGroupData->setParameter<Vec4f>( "envColor", Vec4f( texture.percent, texture.percent, texture.percent, texture.percent ) ) );
      DP_VERIFY( parameterGroupData->setParameter<dp::fx::EnumSpec::StorageType>( "envMode", TEM_INTERPOLATE ) );
    }
  }
  return( parameterGroupData );
}
