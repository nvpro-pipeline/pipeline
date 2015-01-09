// Copyright NVIDIA Corporation 2010
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
// HOOPSLoader.h
//

#pragma once

#if ! defined( DOXYGEN_IGNORE )

#include <string>
#include <vector>

#include <dp/math/Vecnt.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/io/PlugInterface.h>

#include <A3DSDKIncludes.h>

#ifdef _WIN32
// microsoft specific storage-class defines
# ifdef HOOPSLOADER_EXPORTS
#  define HOOPSLOADER_API __declspec(dllexport)
# else
#  define HOOPSLOADER_API __declspec(dllimport)
# endif
#else
# define HOOPSLOADER_API
#endif

// exports required for a scene loader plug-in
extern "C"
{
HOOPSLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugIn *& pi);
HOOPSLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}


#define INVOKE_CALLBACK(cb) if( callback() ) callback()->cb

class HOOPSLoader : public dp::sg::io::SceneLoader
{
  struct BaseData
  {
    BaseData()
      : name("unnamed")
      , layerIndex( A3D_DEFAULT_LAYER )
      , styleIndex( A3D_DEFAULT_STYLE_INDEX )
      , behavior(0)
    {
    }

    std::string name;
    A3DUns32 layerIndex;
    A3DUns32 styleIndex;
    A3DUns16 behavior;
  };

public:
  HOOPSLoader();

  void deleteThis( void );

  dp::sg::core::SceneSharedPtr
  HOOPSLoader::load( std::string const& filename
                   , std::vector<std::string> const& searchPaths
                   , dp::sg::ui::ViewStateSharedPtr & viewState );

protected:
  virtual ~HOOPSLoader(void);
  bool initHOOPS( const A3DUTF8Char * customerKey, const A3DUTF8Char * variableKey );
  bool loadFile( const std::string & file );
  void clear();

private:
  class CascadedAttributes
  {
    public:
      CascadedAttributes();
      CascadedAttributes( const A3DRootBaseWithGraphics * pBase, const CascadedAttributes & parent );
      CascadedAttributes( const A3DRootBaseWithGraphics * pBase, const A3DTessBase * pTessBase, const A3DTess3DData & t3d, A3DUns32 idx, const CascadedAttributes & parent );
      ~CascadedAttributes();

      const A3DMiscCascadedAttributes * getAttributes() const { return( m_pAttr ); }

    private:
      CascadedAttributes( const CascadedAttributes & rhs );
      CascadedAttributes & operator=( const CascadedAttributes & rhs );

    private:
      A3DMiscCascadedAttributes * m_pAttr;
  };

private:
  A3DStatus adaptBrepModels( const A3DEntity * pEntity, A3DEntity ** pConvertedEntity );
  A3DStatus traverseRepItem( const A3DRiRepresentationItem* pRepItem, const CascadedAttributes & parentCA, bool assembly = false );
  A3DStatus traverseSet( const A3DRiSet* pSet, const CascadedAttributes & parentCA );
  A3DStatus traversePartDef(const A3DAsmPartDefinition* pPart, const CascadedAttributes & parentCA, bool assembly);
  A3DStatus traversePOccurrence( const A3DAsmProductOccurrence* pOccurrence, const CascadedAttributes & parentCA, bool assembly = false );
  A3DStatus traverseModel();
  A3DStatus traverseGlobal( A3DGlobal * );
  A3DStatus traverseTess3D( const A3DRiRepresentationItem * pRepItem, const CascadedAttributes & parentCA, bool assembly );
  A3DStatus traverseTess3DWire( const A3DRiRepresentationItem * pRepItem, const CascadedAttributes & parentCA, bool assembly );
  A3DStatus traverseBrepModel( const A3DRiBrepModel * pBrepModel );
  A3DStatus traverseRiCurve( const A3DRiCurve * pCurve );
  dp::sg::core::GroupSharedPtr traverseTransform( const A3DMiscCartesianTransformation * trans3D );

  A3DStatus parseTess3D( const A3DRiRepresentationItem * pRepItem, const A3DTess3D * tess3D, const BaseData & bd, const CascadedAttributes & parentCA, bool assembly );
  A3DStatus parseTess3DWire( const A3DRiRepresentationItem * pRepItem, const A3DTess3DWire * tessw, const BaseData & bd, const CascadedAttributes & parentCA, bool assembly );

  // group stack management
  dp::sg::core::GroupSharedPtr getCurrentGroup();
  void pushGroup( const dp::sg::core::GroupSharedPtr & group );
  void popGroup();

  std::string safeNodeName( const A3DEntity * entity );
  bool traverseBase( const A3DEntity * entity, BaseData & data );

  // palette management
  bool getStyle( A3DUns32 style, A3DGraphStyleData & result );
  bool getRGBColor( A3DUns32 index, dp::math::Vec3f & result );
  dp::sg::core::ParameterGroupDataSharedPtr createGeometryParameterGroupData( const A3DMiscCascadedAttributes * pAttr );
  dp::sg::core::EffectDataSharedPtr createMaterialEffect( const A3DMiscCascadedAttributes * pAttr );
  dp::sg::core::EffectDataSharedPtr createLineEffect( const A3DMiscCascadedAttributes * pAttr, dp::math::Vec3f & color );

  void reportError( const std::string & context, A3DStatus err );
  void reportUnsupported( const std::string & context );

private:
  class GraphStyleDataCompare
  {
    public:
      bool operator()( const A3DGraphStyleData & a, const A3DGraphStyleData & b ) const
      {
        return( memcmp( &a, &b, sizeof(A3DGraphStyleData) ) < 0 );
      }
  };
  typedef std::map<A3DGraphStyleData,dp::sg::core::EffectDataSharedPtr,GraphStyleDataCompare>  GraphStyleMap;
  typedef std::map<A3DGraphStyleData,dp::sg::core::ParameterGroupDataSharedPtr,GraphStyleDataCompare>  GraphStyleGeometryMap;

  typedef std::pair<const A3DAsmProductOccurrence*,A3DGraphStyleData>   POGraphStylePair;
  class POGSCompare
  {
    public:
      bool operator()( const POGraphStylePair & a, const POGraphStylePair & b ) const
      {
        return( ( a.first < b.first ) || ( ( a.first == b.first ) && ( memcmp( &a.second, &b.second, sizeof(A3DGraphStyleData) ) < 0 ) ) );
      }
  };
  typedef std::map<POGraphStylePair,dp::sg::core::GroupSharedPtr,POGSCompare> POMap;

private:
  A3DAsmModelFile * m_modelFile;
  dp::sg::core::SceneSharedPtr m_scene;
  std::vector< dp::sg::core::GroupSharedPtr > m_groupStack;
  unsigned int m_geometryCounter;
  unsigned int m_materialCounter;

  // parsed from globals
  std::vector< A3DGraphStyleData >  m_styles;
  std::vector< unsigned short >     m_linePatterns;
  std::vector< dp::math::Vec3f >    m_rgbColors;
  GraphStyleGeometryMap             m_geometryParameterGroupDatas;
  GraphStyleMap                     m_materialEffects;
  POMap                             m_POs;
};

inline void HOOPSLoader::pushGroup( const dp::sg::core::GroupSharedPtr & group )
{
  // if there is already a group on the stack, we add this one as a child
  if( m_groupStack.size() )
  {
    m_groupStack.back()->addChild( group );
  }

  m_groupStack.push_back( group );
}

inline void HOOPSLoader::popGroup()
{
  DP_ASSERT( m_groupStack.size() );
  m_groupStack.pop_back();
}

inline dp::sg::core::GroupSharedPtr HOOPSLoader::getCurrentGroup()
{
  return m_groupStack.back();
}

inline bool HOOPSLoader::getStyle( A3DUns32 style, A3DGraphStyleData & result )
{
  if( style < m_styles.size() )
  {
    result = m_styles[style];
    return true;
  }

  return false;
}

inline bool HOOPSLoader::getRGBColor( A3DUns32 index, dp::math::Vec3f & result )
{
  // stupid!!
  index /= 3;

  if( index < m_rgbColors.size() )
  {
    result = m_rgbColors[index];
    return true;
  }

  return false;
}

#endif // ! defined( DOXYGEN_IGNORE )

