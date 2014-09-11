// Copyright NVIDIA Corporation 2002-2005
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
// FLTLoader.h
//

#pragma once

#if ! defined( DOXYGEN_IGNORE )

#include <string>
#include <vector>
#include <map>
#include <list>
#include <flt.h>

#include <dp/math/Vecnt.h>
#include <dp/sg/core/nvsgapi.h>
#include <dp/sg/core/nvsg.h>
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/io/PlugInterface.h>
#include <flt.h>
#include "FLTTextureAttribute.h"
#include "TaiManager.h"

#ifdef _WIN32
// microsoft specific storage-class defines
# ifdef FLTLOADER_EXPORTS
#  define FLTLOADER_API __declspec(dllexport)
# else
#  define FLTLOADER_API __declspec(dllimport)
# endif
#else
# define FLTLOADER_API
#endif

// exports required for a scene loader plug-in
extern "C"
{
FLTLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugIn *& pi);
FLTLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

class StackedAtlasManager;

#define INVOKE_CALLBACK(cb) if( callback() ) callback()->cb

class FLTLoader : public dp::sg::io::SceneLoader
{
public:
  FLTLoader();

  void deleteThis( void );

  dp::sg::core::SceneSharedPtr load( std::string const& filename
                                   , std::vector<std::string> const& searchPaths
                                   , dp::sg::ui::ViewStateSharedPtr & viewState );

protected:
  virtual ~FLTLoader(void);

private:
  void buildScene( dp::sg::core::GroupSharedPtr const& group, FltNode *, FltFile *, int, bool = true );

  enum LookupEnum
  {
    LOOKUP_DEFAULT,
    LOOKUP_DETAIL, 
    LOOKUP_NORMALMAP
  };

  // textures
  struct TextureEntry
  {
    // the texture return
    dp::sg::core::ParameterGroupDataSharedPtr textureParameters;

    // 3rd texture coord - if using stacked atlas
    int layer;

    // these are set if the entry came from a texture atlas
    float  uscale;
    float  utrans;
    float  vscale;
    float  vtrans;
  };

  bool lookupTexture( TextureEntry & result, FltFile * flt, 
                      uint16 indx, int16 tmIndx = -1,
                      LookupEnum detail = LOOKUP_DEFAULT );

  dp::sg::core::ParameterGroupDataSharedPtr createNormalizationCubeMap( unsigned int dimension );

  // materials
  dp::sg::core::EffectDataSharedPtr const & lookupMaterial( FltFile * flt, uint16 indx, dp::math::Vec3f &color, float trans);

  // instances
  bool createInstance( FltFile * flt, FltNode * root );
  const dp::sg::core::GroupSharedPtr & lookupInstance( FltFile * flt, uint16 inst );

  // extrefs (stored by file name)
  bool createExtRef( const std::string & file, const dp::sg::core::GroupSharedPtr & grp );
  dp::sg::core::GroupSharedPtr lookupExtRef( const std::string & file );

  typedef unsigned long long int mapEntryType;
  inline mapEntryType cvtToMapEntry( FltFile * flt, uint16 indx,
                                                    uint16 add = 0 )
  {
    mapEntryType ull;

    ull = ((((mapEntryType)fltGetFileID( flt ))<<32)&0xffffffff00000000ull) | 
              ((indx<<16)&0xffff0000 | (add&0xffff));

    return ull;
  }

  enum ShaderProgramType
  {
    FLT_VERTEX_SHADER,
    FLT_FRAGMENT_SHADER
  };

  // colors
  enum ColorLookupEnum
  {
    COLOR_NOCOLOR,
    COLOR_NOTAVAILABLE,
    COLOR_OK
  };

  ColorLookupEnum 
        getFaceColor( FltFile * flt, FltFace * face, dp::math::Vec4f & result );
  ColorLookupEnum 
        getVertexColor( FltFile * flt, FltVertex * v, dp::math::Vec4f & result );

  uint32 nextFileID( void ) { return m_fileIDs++; }

  void buildObject( dp::sg::core::GroupSharedPtr const& parent, FltNode * node, FltFile * flt, int );
  void buildFace( dp::sg::core::GroupSharedPtr const& parent, FltNode * node, FltFile * flt, bool isSubFace = false );
  inline bool collect( FltNode * node );
  void collectDescend( dp::sg::core::GroupSharedPtr const& parent, FltNode * , FltFile * );
  void collectGeometry( dp::sg::core::GroupSharedPtr const& parent, FltNode * , FltFile * );
  void optimizeGeometry( const dp::sg::core::GroupSharedPtr & group );

  //
  // current workarounds:
  //
  // CHILDLESS_HEADER - parse all siblings of header node if no children
  // LIGHTING_MODE=[FLAT|LIT|GOURAUD|LIT_GOURAUD] - force a lighting mode for
  //                                                all polygons.
  //
  void setWorkaroundMode( const std::string & mode );
  bool getWorkaround( const std::string & e, 
                      std::string * result = 0 );

  bool faceNormal( dp::math::Vec3f &, FltVertex ** vlist, unsigned int numVerts );

  std::map<mapEntryType,std::pair<FLTTextureAttribute,dp::sg::core::ParameterGroupDataSharedPtr> > m_textureCache;
  std::map<std::string, std::string> m_workarounds;
  std::vector< std::string > m_texturesNotFound;
  std::vector< std::string > m_shadersNotFound;
  std::vector< std::string > m_searchPaths;
  std::map<mapEntryType,dp::sg::core::GroupSharedPtr> m_instanceNodes;
  std::map<std::string, dp::sg::core::GroupSharedPtr> m_extRefs;
  dp::sg::core::ParameterGroupDataSharedPtr m_normalizationCubeMap;
  bool removeDegenerates( std::vector<unsigned int> & indexList, 
                          std::vector<dp::math::Vec3f> & verts );

  void addAnnotation( dp::sg::core::NodeSharedPtr const& nNode, FltNode * fNode );
  bool hasComment( FltNode *fNode );

  TaiManager * getTaiManager() { return m_taiManager; }
  StackedAtlasManager * getStackedAtlasManager() 
    { return m_stackedAtlasManager; }

  dp::sg::core::ParameterGroupDataSharedPtr loadAtlasTexture( FltFile *, const std::string & );
  std::map<std::string, dp::sg::core::ParameterGroupDataSharedPtr> m_atlasCache;
  std::map<std::string, dp::sg::core::TextureHostSharedPtr> m_textureNameCache;

  unsigned int m_numTextures;
  unsigned int m_numMaterials;

  bool m_rgbColorMode;
  bool m_buildMipmapsAtLoadTime;
  bool m_useStackedAtlasManager;

  uint32 m_fileIDs;

  TaiManager * m_taiManager;
  StackedAtlasManager * m_stackedAtlasManager;

  dp::sg::core::EffectDataSharedPtr m_pointGeometryEffect;
  std::map<FltMaterial*,dp::sg::core::EffectDataSharedPtr>            m_materialMap;
  std::multimap<dp::util::HashKey,dp::sg::core::EffectDataSharedPtr>  m_materialCache;
};

#endif // ! defined( DOXYGEN_IGNORE )
