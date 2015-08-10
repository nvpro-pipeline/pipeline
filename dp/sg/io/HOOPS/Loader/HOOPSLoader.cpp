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
// HOOPSLoader.cpp
//

#include <dp/sg/io/PlugInterface.h> // definition of UPITID_VERSION,
#include <dp/sg/io/PlugInterfaceID.h> // definition of UPITID_VERSION, UPITID_SCENE_LOADER, and UPITID_SCENE_SAVER

#include <dp/Exception.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/VertexAttributeSet.h>
#include <dp/sg/core/Primitive.h>
#include <dp/util/File.h>
#include <dp/util/Locale.h>
#include <dp/util/Timer.h>
#include <dp/math/Vecnt.h>
#include <dp/fx/EffectLibrary.h>

#define INITIALIZE_A3D_API
#include <A3DSDKIncludes.h>
#include <A3DSDKLicenseKey.h>
#undef  INITIALIZE_A3D_API

#include "HOOPSLoader.h"
#include "effect.h"

using namespace dp::sg::core;
using namespace dp::math;
using namespace dp::util;
using std::make_pair;
using std::map;
using std::pair;
using std::string;
using std::vector;


// unique plug-in types
const dp::util::UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION);

static char * cadFormats[] =
{
  ".3dxml",       // Dassault Systems: CATIA V6
  ".arc",         // Siemens: I-deas
  ".asm",         // Siemens: Solid Edge, PTC
  ".catpart",     // Dassault Systems: CATIA V5
  ".catproduct",  // Dassault Systems: CATIA V5
  ".catshape",    // Dassault Systems: CATIA V5
  ".cgr",         // Dassault Systems: CATIA Graphics Representation
  ".dlv",         // Dassault Systems: CATIA V4
  ".exp",         // Dassault Systems: CATIA V4
  ".iam",         // Autodesk: Inventor
  ".ifc",         // Industry Foundation Classes
  ".iges",        // IGES
  ".igs",         // IGES
  ".ipt",         // Autodesk: Inventor
  ".jt",          // Siemens: JT
  ".mf1",         // Siemens: I-deas
  ".model",       // Dassault Systems: CATIA V4
  ".neu",         // PTC
  ".par",         // Siemens: Solid Edge
  ".pkg",         // Siemens: I-deas
  ".prc",         // PRC
  ".prt",         // PTC, Siemens: NX
  ".psm",         // Siemens: Solid Edge
  ".pwd",         // Siemens: Solid Edge
  ".session",     // Dassault Systems: CATIA V4
  ".sldasm",      // Dassault Systems: SolidWorks
  ".sldprt",      // Dassault Systems: SolidWorks
  ".step",        // STEP
  ".stl",         // Stereo Lithography
  ".stp",         // STEP
  ".vda"          // VDA-FS
  ".vrml",        // VRML V1.0 and V2.0
//  ".wrl",         // VRML V1.0 and V2.0
  ".x_b",         // Siemens: Parasolid
  ".x_t",         // Siemens: Parasolid
  ".xas",         // PTC
  ".xmt",         // Siemens: Parasolid
  ".xmt_txt",     // Siemens: Parasolid
  ".xpr",         // PTC
};
#define NUM_CAD_FORMATS (sizeof( cadFormats ) / sizeof( char * ))

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  for ( unsigned int i=0 ; i<NUM_CAD_FORMATS ; i++ )
  {
    piids.push_back(dp::util::UPIID(cadFormats[i], PITID_SCENE_LOADER));
  }
}

bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugIn *& pi)
{
  for( unsigned int i = 0; i < NUM_CAD_FORMATS; i ++ )
  {
    if ( piid == dp::util::UPIID(cadFormats[i], PITID_SCENE_LOADER) )
    {
      pi = new HOOPSLoader();
      return( true );
    }
  }

  return false;
}


// need something different here, use this for now
#define CHECK_RET(function_call) {\
  A3DStatus iRet = function_call;\
  if (iRet != A3D_SUCCESS)\
    {\
      reportError( "HOOPS 3DX ERROR", iRet );\
      DP_ASSERT( false );\
      return iRet;\
    }\
  }

class CascadedAttributesData
{
  public:
    CascadedAttributesData( const A3DMiscCascadedAttributes * pAttr )
    {
      A3D_INITIALIZE_DATA( A3DMiscCascadedAttributesData, m_data );
      DP_VERIFY( A3DMiscCascadedAttributesGet( pAttr, &m_data ) == A3D_SUCCESS );
      DP_ASSERT( /*m_data.m_bShow &&*/ !m_data.m_bRemoved );    // don't know what m_bShow really means! We have files where this is never set!
      DP_ASSERT( !m_data.m_sStyle.m_bVPicture && !m_data.m_sStyle.m_bSpecialCulling && !m_data.m_sStyle.m_bNoLight );
    }

    ~CascadedAttributesData()
    {
      DP_VERIFY( A3DMiscCascadedAttributesGet( NULL, &m_data ) == A3D_SUCCESS );
    }

    const A3DGraphStyleData & getStyle() const
    {
      return( m_data.m_sStyle );
    }

  private:
    A3DMiscCascadedAttributesData m_data;
};

HOOPSLoader::CascadedAttributes::CascadedAttributes()
: m_pAttr(nullptr)
{
  DP_VERIFY( A3DMiscCascadedAttributesCreate( &m_pAttr ) == A3D_SUCCESS );
}

HOOPSLoader::CascadedAttributes::CascadedAttributes( const A3DRootBaseWithGraphics * pBase, const CascadedAttributes & parent )
: m_pAttr(nullptr)
{
  DP_VERIFY( A3DMiscCascadedAttributesCreate( &m_pAttr ) == A3D_SUCCESS );
  DP_VERIFY( A3DMiscCascadedAttributesPush( m_pAttr, pBase, parent.getAttributes() ) == A3D_SUCCESS );
}

HOOPSLoader::CascadedAttributes::CascadedAttributes( const A3DRiRepresentationItem * pRepItem, const A3DTessBase * pTessBase, const A3DTess3DData & t3d, A3DUns32 idx, const CascadedAttributes & parent )
: m_pAttr(nullptr)
{
  DP_VERIFY( A3DMiscCascadedAttributesCreate( &m_pAttr ) == A3D_SUCCESS );
  DP_VERIFY( A3DMiscCascadedAttributesPushTessFace( m_pAttr, pRepItem, pTessBase, &t3d.m_psFaceTessData[idx], idx, parent.getAttributes() ) == A3D_SUCCESS );
}

HOOPSLoader::CascadedAttributes::~CascadedAttributes()
{
  DP_VERIFY( A3DMiscCascadedAttributesDelete( m_pAttr ) == A3D_SUCCESS );
}

HOOPSLoader::HOOPSLoader()
  : m_modelFile(0)
  , m_geometryCounter(0)
  , m_materialCounter(0)
{
  if( !initHOOPS( pcCustomerKey, pcVariableKey ) )
  {
    INVOKE_CALLBACK( onUnLocalizedMessage( "ERROR", "Unable to initialize HOOPS 3DX Library!\n" ) );
  }
}

static unsigned int sInitializationCount = 0;

HOOPSLoader::~HOOPSLoader()
{
  clear();

  sInitializationCount--;
  if ( sInitializationCount == 0 )
  {
    A3DDllTerminate();
    A3DSDKUnloadLibrary();
  }
}

void
HOOPSLoader::reportError( const string & context, A3DStatus code )
{
  // decode error and spit it out
  string err( A3DMiscGetErrorMsg( code ) );
  err += "\n";

  INVOKE_CALLBACK( onUnLocalizedMessage( context, err ) );
}

void
HOOPSLoader::reportUnsupported( const string & context )
{
  string unsup = context + "\n";
  INVOKE_CALLBACK( onUnLocalizedMessage( "HOOPSLoader: Unsupported", unsup ) );
}

void
HOOPSLoader::deleteThis()
{
  // was instantiated using 'new'. hence kill it with 'delete'
  delete this;
}

bool
HOOPSLoader::initHOOPS( const A3DUTF8Char * customerKey, const A3DUTF8Char * variableKey )
{
  if ( sInitializationCount == 0 )
  {
    if( !A3DSDKLoadLibrary() )
    {
      return false;
    }

    A3DInt32 ret = A3DLicPutLicense( A3DLicPutLicenseFile, customerKey, variableKey );
    if ( ret != A3D_SUCCESS )
    {
      reportError( "HOOPS 3DX ERROR: A3DLicPutLicense", A3D_ERROR );
      return false;
    }
    
    A3DInt32 iMajorVersion=0, iMinorVersion=0;
    DP_VERIFY( A3DDllGetVersion(&iMajorVersion, &iMinorVersion) == A3D_SUCCESS );
    if (    ( iMajorVersion != A3D_DLL_MAJORVERSION )
        ||  ( ( iMajorVersion == A3D_DLL_MAJORVERSION ) && ( iMinorVersion < A3D_DLL_MINORVERSION ) ) )
    {
      reportError( "HOOPS 3DX ERROR: A3DDllGetVersion returned incompatible version", A3D_ERROR ); 
      return false;
    }

#if 1
    A3DStatus iRet = A3DDllInitialize( A3D_DLL_MAJORVERSION, A3D_DLL_MINORVERSION );
    if ( iRet != A3D_SUCCESS )
#else
    if( A3DDllInitialize( A3D_DLL_MAJORVERSION, A3D_DLL_MINORVERSION ) != A3D_SUCCESS )
#endif
    {
      reportError( "HOOPS 3DX ERROR: A3DDllInitialize", iRet );
      return false;
    }
  }
  sInitializationCount++;
  return true;
}

string
HOOPSLoader::safeNodeName( const A3DEntity * entity )
{
  string result("unnamed");

  if( entity )
  {
    A3DRootBaseData sData;
    A3D_INITIALIZE_DATA( A3DRootBaseData, sData );

    if( A3DRootBaseGet( entity, &sData ) == A3D_SUCCESS )
    {
      if( sData.m_pcName && strlen( sData.m_pcName ) )
      {
        result = sData.m_pcName;
      }

      A3DRootBaseGet( NULL, &sData );
    }
  }

  return result;
}

#define HANDLE_BREP   0

bool
HOOPSLoader::loadFile( const string & file )
{
  A3DStatus iRet = A3D_SUCCESS;

  // using these defaults from their Viewer example
  A3DRWParamsLoadData sReadParam;
  A3D_INITIALIZE_DATA( A3DRWParamsLoadData, sReadParam);
  sReadParam.m_sGeneral.m_bReadSolids=true;
  sReadParam.m_sGeneral.m_bReadSurfaces=true;
  sReadParam.m_sGeneral.m_bReadWireframes=true;
  sReadParam.m_sGeneral.m_bReadPmis=false;
  sReadParam.m_sGeneral.m_bReadAttributes=false;
  sReadParam.m_sGeneral.m_bReadHiddenObjects=false;
  sReadParam.m_sGeneral.m_bReadConstructionAndReferences=false;
  sReadParam.m_sGeneral.m_bReadActiveFilter=true;
  sReadParam.m_sGeneral.m_bReadDrawings=false;
#if HANDLE_BREP
  sReadParam.m_sGeneral.m_eReadGeomTessMode=kA3DReadGeomOnly;
#else
  sReadParam.m_sGeneral.m_eReadGeomTessMode=kA3DReadGeomAndTess;
#endif
  sReadParam.m_sGeneral.m_eDefaultUnit=kA3DUnitUnknown;
  sReadParam.m_sTessellation.m_eTessellationLevelOfDetail = kA3DTessLODMedium;
  sReadParam.m_sAssembly.m_bUseRootDirectory=true;

  iRet = A3DAsmModelFileLoadFromFile(file.c_str(), &sReadParam, &m_modelFile );
  if (iRet != A3D_SUCCESS && iRet != A3D_LOAD_MULTI_MODELS_CADFILE && iRet != A3D_LOAD_MISSING_COMPONENTS )
  {
    // decode error and spit it out
    reportError( "HOOPS 3DX: Unable to Import File", iRet );
    return false;
  }

  return true;
}

void
HOOPSLoader::clear()
{
  if( m_modelFile )
  {
    DP_ASSERT( sInitializationCount );
    A3DAsmModelFileDelete( m_modelFile );
    m_modelFile = 0;
  }

  m_scene.reset();
  m_geometryCounter = 0;
  m_materialCounter = 0;
  m_groupStack.clear();
  m_linePatterns.clear();
  m_styles.clear();
  m_rgbColors.clear();
  m_geometryParameterGroupDatas.clear();
  m_materialEffects.clear();
  m_POs.clear();
}

//
// MMM - this is probably incomplete, and is untested..
//
A3DStatus HOOPSLoader::adaptBrepModels( const A3DEntity * pEntity, A3DEntity ** pConvertedEntity )
{
  A3DEEntityType eType = kA3DTypeUnknown;
  CHECK_RET( A3DEntityGetType( pEntity, &eType ) );
  DP_ASSERT( eType == kA3DTypeRiBrepModel );

#if 0
  static A3DUns32 acceptableSurfaces[] = { kA3DTypeSurfBlend01, kA3DTypeSurfBlend02, kA3DTypeSurfBlend03, kA3DTypeSurfNurbs, kA3DTypeSurfCone
                                         , kA3DTypeSurfCylinder, kA3DTypeSurfCylindrical, kA3DTypeSurfOffset, kA3DTypeSurfPipe, kA3DTypeSurfPlane
                                         , kA3DTypeSurfRuled, kA3DTypeSurfSphere, kA3DTypeSurfRevolution, kA3DTypeSurfExtrusion, kA3DTypeSurfFromCurves
                                         , kA3DTypeSurfTorus, kA3DTypeSurfTransform, kA3DTypeSurfBlend04 };
  static A3DUns32 acceptableCurves[] = { kA3DTypeCrvBlend02Boundary, kA3DTypeCrvNurbs, kA3DTypeCrvCircle, kA3DTypeCrvComposite, kA3DTypeCrvOnSurf
                                       , kA3DTypeCrvEllipse, kA3DTypeCrvEquation, kA3DTypeCrvHelix, kA3DTypeCrvHyperbola, kA3DTypeCrvIntersection
                                       , kA3DTypeCrvLine, kA3DTypeCrvOffset, kA3DTypeCrvParabola, kA3DTypeCrvPolyLine, kA3DTypeCrvTransform };
#else
  static A3DUns32 acceptableSurfaces[] = { kA3DTypeSurfNurbs };
  static A3DUns32 acceptableCurves[] = { kA3DTypeCrvNurbs };
#endif

  A3DCopyAndAdaptBrepModelData sA3DCopyAndAdaptBrepModelData;
  A3D_INITIALIZE_DATA( A3DCopyAndAdaptBrepModelData, sA3DCopyAndAdaptBrepModelData );

  sA3DCopyAndAdaptBrepModelData.m_bUseSameParam = true;
  sA3DCopyAndAdaptBrepModelData.m_dTol = 1e-6;
  sA3DCopyAndAdaptBrepModelData.m_bDeleteCrossingUV = true;
  sA3DCopyAndAdaptBrepModelData.m_bSplitFaces = true;
  sA3DCopyAndAdaptBrepModelData.m_bSplitClosedSurfaces = true;
  sA3DCopyAndAdaptBrepModelData.m_bForceComputeUV = true;
  sA3DCopyAndAdaptBrepModelData.m_bForceCompute3D = false;
  sA3DCopyAndAdaptBrepModelData.m_uiAcceptableSurfacesSize = sizeof(acceptableSurfaces) / sizeof(acceptableSurfaces[0]);
  sA3DCopyAndAdaptBrepModelData.m_puiAcceptableSurfaces = &acceptableSurfaces[0];
  sA3DCopyAndAdaptBrepModelData.m_uiAcceptableCurvesSize = sizeof(acceptableCurves) / sizeof(acceptableCurves[0]);
  sA3DCopyAndAdaptBrepModelData.m_puiAcceptableCurves = &acceptableCurves[0];

  return( A3DCopyAndAdaptBrepModel( pEntity, &sA3DCopyAndAdaptBrepModelData, pConvertedEntity ) );
}

GeoNodeSharedPtr createGeoNode( PrimitiveType pt, const vector<Vec3f> & vertices, const vector<Vec3f> & normals, const vector<Vec2f> & textures
                              , const vector<unsigned int> & indices, const string & name, unsigned int hints, const EffectDataSharedPtr & materialEffect )
{
  // create VAS
  VertexAttributeSetSharedPtr vertexAttributeSet = VertexAttributeSet::create();
  vertexAttributeSet->setVertices( &vertices[0], dp::checked_cast<unsigned int>( vertices.size() ) );
  if ( !normals.empty() )
  {
    vertexAttributeSet->setNormals( &normals[0], dp::checked_cast<unsigned int>( normals.size() ) );
  }
  if ( !textures.empty() )
  {
    vertexAttributeSet->setTexCoords( 0, &textures[0], dp::checked_cast<unsigned int>( textures.size() ) );
  }

  // create Primitive
  PrimitiveSharedPtr primitive = Primitive::create( pt );
  primitive->setVertexAttributeSet( vertexAttributeSet );

  if ( ! indices.empty() )
  {
    IndexSetSharedPtr indexSet = IndexSet::create();
    indexSet->setData( &indices[0], dp::checked_cast<unsigned int>( indices.size() ) );
    primitive->setIndexSet( indexSet );
  }

  // create GeoNode
  GeoNodeSharedPtr geoNode = GeoNode::create();
  geoNode->setName( name );
  geoNode->setHints( hints );
  geoNode->setPrimitive( primitive );
  geoNode->setMaterialEffect( materialEffect );

  return( geoNode );
}

template<unsigned int n>
class IndexMap : public map<Vecnt<n,unsigned int>, unsigned int>
{
};

template<unsigned int n>
struct IndexData
{
  vector<unsigned int>  newIndices;
  IndexMap<n>           indexMap;
};

template<unsigned int N>
Vecnt<N,unsigned int> getIndexSet( A3DUns32 * indices, unsigned int & idx, const A3DTessBaseData & tbd, const A3DTess3DData & t3d
                                 , bool oneNormal = false, unsigned int normalIndex = 0 )
{
  Vecnt<N,unsigned int> resIdx;
  resIdx[0] = oneNormal ? normalIndex : indices[idx++];
  DP_ASSERT( (resIdx[0]+2) < t3d.m_uiNormalSize );
  for ( unsigned int k=2 ; k<N ; k++ )    // get the texture coordinates
  {
    resIdx[k-1] = indices[idx++];
    DP_ASSERT( (resIdx[k-1]+1) < t3d.m_uiTextureCoordSize );
  }
  resIdx[N-1] = indices[idx++];
  DP_ASSERT( (resIdx[N-1]+2) < tbd.m_uiCoordSize );
  return( resIdx );
}

template<unsigned int N>
void gatherFromTriangles( const A3DTessBaseData & tbd, const A3DTess3DData & t3d, const A3DTessFaceData & tfd
                        , unsigned int & sizeIndex, unsigned int & startOffset, IndexData<N> & indexData )
{
  A3DUns32 * indices = t3d.m_puiTriangulatedIndexes;
  DP_ASSERT( sizeIndex < tfd.m_uiSizesTriangulatedSize );
  unsigned int count = tfd.m_puiSizesTriangulated[sizeIndex++];

  indexData.newIndices.reserve( indexData.newIndices.size() + 3 * count );

  unsigned int triIndex = tfd.m_uiStartTriangulated + startOffset;
  DP_ASSERT( (3*N*count + triIndex) <= t3d.m_uiTriangulatedIndexSize );
  Vecnt<N,unsigned int> is;
  for ( unsigned int i=0 ; i<count ; i++ )
  {
    // note that these indices represent the actual starting index in the array, not the primitive index (ie: multiplied by 3)
    // interleaved n, {t,...t,} v, n, {t,...t,} v, n, {t,...t,} v
    for ( unsigned int j=0 ; j<3 ; j++ )
    {
      is = getIndexSet<N>( indices, triIndex, tbd, t3d );
      pair<IndexMap<N>::const_iterator,bool> pitb = indexData.indexMap.insert( make_pair( is, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
      indexData.newIndices.push_back( pitb.first->second );
    }
  }

  startOffset += N * 3 * count;
  DP_ASSERT( startOffset == triIndex - tfd.m_uiStartTriangulated );
}

template<unsigned int N>
void gatherFromTrianglesOneNormal( const A3DTessBaseData & tbd, const A3DTess3DData & t3d, const A3DTessFaceData & tfd
                                 , unsigned int & sizeIndex, unsigned int & startOffset, IndexData<N> & indexData )
{
  A3DUns32 * indices = t3d.m_puiTriangulatedIndexes;
  DP_ASSERT( sizeIndex < tfd.m_uiSizesTriangulatedSize );
  unsigned int count = tfd.m_puiSizesTriangulated[sizeIndex++];

  indexData.newIndices.reserve( indexData.newIndices.size() + 3 * count );

  unsigned int triIndex = tfd.m_uiStartTriangulated + startOffset;
  DP_ASSERT( ( 1 + 3*(N-1)*count + triIndex ) <= t3d.m_uiTriangulatedIndexSize );
  Vecnt<N,unsigned int> is;
  for ( unsigned int i=0 ; i<count ; i++ )
  {
    // note that these indices represent the actual starting index in the array, not the primitive index (ie: multiplied by 3)
    // interleaved n, v, v, v
    is[0] = indices[triIndex++];
    DP_ASSERT( (is[0]+2) < t3d.m_uiNormalSize );
    for ( unsigned int j=0 ; j<3 ; j++ )
    {
      is = getIndexSet<N>( indices, triIndex, tbd, t3d, true, is[0] );
      pair<IndexMap<N>::const_iterator,bool> pitb = indexData.indexMap.insert( make_pair( is, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
      indexData.newIndices.push_back( pitb.first->second );
    }
  }

  startOffset += ( 1 + 3 * ( N - 1 ) ) * count;
  DP_ASSERT( startOffset == triIndex - tfd.m_uiStartTriangulated );
}

template<unsigned int N>
void gatherFromFans( const A3DTessBaseData & tbd, const A3DTess3DData & t3d, const A3DTessFaceData & tfd
                   , unsigned int & sizeIndex, unsigned int & startOffset, IndexData<N> & indexData, bool oneNormal )
{
  A3DUns32 * indices = t3d.m_puiTriangulatedIndexes;
  DP_ASSERT( sizeIndex < tfd.m_uiSizesTriangulatedSize );
  unsigned int fanCount = tfd.m_puiSizesTriangulated[sizeIndex++];

  // first pass to get the complete number of indices from this fans
  DP_ASSERT( sizeIndex + fanCount <= tfd.m_uiSizesTriangulatedSize );
  unsigned int completeCount = 0;
  for ( unsigned int i=0, j=sizeIndex ; i<fanCount ; i++, j++ )
  {
    unsigned int count = tfd.m_puiSizesTriangulated[j] & kA3DTessFaceDataNormalMask;   // hide the flag on the high bit!!
    completeCount += 3 * ( count - 2 );
  }
  indexData.newIndices.reserve( indexData.newIndices.size() + completeCount );

  // next pass to get really get the indices
  unsigned int idx = tfd.m_uiStartTriangulated + startOffset;
  Vecnt<N,unsigned int> is0, is2;
  for ( unsigned int i=0 ; i<fanCount ; i++ )
  {
    DP_ASSERT( sizeIndex < tfd.m_uiSizesTriangulatedSize );
    bool oneNormalLocal = oneNormal || ( tfd.m_puiSizesTriangulated[sizeIndex] & kA3DTessFaceDataNormalSingle );

    unsigned int count = tfd.m_puiSizesTriangulated[sizeIndex++] & kA3DTessFaceDataNormalMask;    // hide the flag on the high bit!!
    DP_ASSERT( 2 < count );
    DP_ASSERT( oneNormalLocal ? ( (count + 1 + idx) <= t3d.m_uiTriangulatedIndexSize ) : ( (2 * count + idx) <= t3d.m_uiTriangulatedIndexSize ) );

    is0 = getIndexSet<N>( indices, idx, tbd, t3d );
    pair<IndexMap<N>::const_iterator,bool> pitb = indexData.indexMap.insert( make_pair( is0, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
    unsigned int idx0 = pitb.first->second;

    is2[0] = oneNormalLocal ? is0[0] : indices[idx++];
    is2[1] = indices[idx++];
    DP_ASSERT( (is2[1]+2) < tbd.m_uiCoordSize && (is2[0]+2) < t3d.m_uiNormalSize );
    pitb = indexData.indexMap.insert( make_pair( is2, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
    unsigned int idx2 = pitb.first->second;

    for ( unsigned int j=2 ; j<count ; j++ )
    {
      unsigned int idx1 = idx2;

      is2 = getIndexSet<N>( indices, idx, tbd, t3d, oneNormalLocal, is0[0] );
      pitb = indexData.indexMap.insert( make_pair( is2, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
      idx2 = pitb.first->second;

      indexData.newIndices.push_back( idx0 );
      indexData.newIndices.push_back( idx1 );
      indexData.newIndices.push_back( idx2 );
    }
  }

  startOffset = idx - tfd.m_uiStartTriangulated;
}

template<unsigned int N>
void gatherFromStrips( const A3DTessBaseData & tbd, const A3DTess3DData & t3d, const A3DTessFaceData & tfd
                     , unsigned int & sizeIndex, unsigned int & startOffset, IndexData<N> & indexData, bool oneNormal )
{
  A3DUns32  * indices = t3d.m_puiTriangulatedIndexes;
  A3DDouble * norms  = t3d.m_pdNormals;
  A3DDouble * verts  = tbd.m_pdCoords;
  DP_ASSERT( sizeIndex < tfd.m_uiSizesTriangulatedSize );
  unsigned int stripCount = tfd.m_puiSizesTriangulated[sizeIndex++];

  // first pass to get the complete number of indices from this strips
  DP_ASSERT( sizeIndex + stripCount <= tfd.m_uiSizesTriangulatedSize );
  unsigned int completeCount = 0;
  for ( unsigned int i=0, j=sizeIndex ; i<stripCount ; i++, j++ )
  {
    unsigned int count = tfd.m_puiSizesTriangulated[j] & kA3DTessFaceDataNormalMask;   // hide the flag on the high bit!!
    completeCount += 3 * ( count - 2 );
  }
  indexData.newIndices.reserve( indexData.newIndices.size() + completeCount );

  // next pass to get really get the indices
  unsigned int idx = tfd.m_uiStartTriangulated + startOffset;
  Vecnt<N,unsigned int> is0, is1, is2;
  for ( unsigned int i=0 ; i<stripCount ; i++ )
  {
    DP_ASSERT( sizeIndex < tfd.m_uiSizesTriangulatedSize );
    bool oneNormalLocal  = oneNormal || ( tfd.m_puiSizesTriangulated[sizeIndex] & kA3DTessFaceDataNormalSingle );

    unsigned int count = tfd.m_puiSizesTriangulated[sizeIndex++] & kA3DTessFaceDataNormalMask;    // hide the flag on the high bit!!
    DP_ASSERT( 2 < count );
    DP_ASSERT( ( oneNormalLocal ? (1 + count + idx) : (2 * count + idx) ) <= t3d.m_uiTriangulatedIndexSize );

    is1 = getIndexSet<N>( indices, idx, tbd, t3d );
    pair<IndexMap<N>::const_iterator,bool> pitb = indexData.indexMap.insert( make_pair( is1, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
    unsigned int idx1 = pitb.first->second;

    is2 = getIndexSet<N>( indices, idx, tbd, t3d, oneNormalLocal, is1[0] );
    pitb = indexData.indexMap.insert( make_pair( is2, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
    unsigned int idx2 = pitb.first->second;

    for ( unsigned int j=2 ; j<count ; j++ )
    {
      is0 = is1;
      unsigned int idx0 = idx1;

      is1 = is2;
      idx1 = idx2;

      is2 =getIndexSet<N>( indices, idx, tbd, t3d, oneNormalLocal, is0[0] );
      pitb = indexData.indexMap.insert( make_pair( is2, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
      idx2 = pitb.first->second;

      if ( j % 2 )
      {
        indexData.newIndices.push_back( idx1 );
        indexData.newIndices.push_back( idx0 );
      }
      else
      {
        indexData.newIndices.push_back( idx0 );
        indexData.newIndices.push_back( idx1 );
      }
      indexData.newIndices.push_back( idx2 );
    }
  }

  startOffset = idx - tfd.m_uiStartTriangulated;
}

void gatherFromLineStrips( const A3DTessBaseData & tbd, const A3DTess3DData & t3d, const A3DTessFaceData & tfd, IndexData<1> & indexData )
{
  A3DUns32  * indices = t3d.m_puiWireIndexes;
  unsigned int stripCount = tfd.m_uiSizesWiresSize;

  // first pass to get the complete number of indices from this line strips
  unsigned int completeCount = 0;
  for ( unsigned int i=0 ; i<stripCount ; i++ )
  {
    completeCount += tfd.m_puiSizesWires[i] & ~( kA3DTessFaceDataWireIsClosing | kA3DTessFaceDataWireIsNotDrawn );
  }
  indexData.newIndices.reserve( indexData.newIndices.size() + stripCount + completeCount );   // plus 1 per strip for the restart index ~0

  // next pass to get really get the indices
  unsigned int idx = tfd.m_uiStartWire;
  Vecnt<1,unsigned int> v;
  for ( unsigned int i=0 ; i<stripCount ; i++ )
  {
    unsigned int count = tfd.m_puiSizesWires[i] & ~( kA3DTessFaceDataWireIsClosing | kA3DTessFaceDataWireIsNotDrawn );
    DP_ASSERT( idx + count <= t3d.m_uiWireIndexSize );
    for ( unsigned int j=0 ; j<count ; j++ )
    {
      v[0] = indices[idx++];
      DP_ASSERT( (v[0]+2) < tbd.m_uiCoordSize );

      pair<IndexMap<1>::const_iterator,bool> pitb = indexData.indexMap.insert( make_pair( v, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
      indexData.newIndices.push_back( pitb.first->second );
    }

    if ( tfd.m_puiSizesWires[i] & kA3DTessFaceDataWireIsClosing )
    {
      v[0] = indices[idx-count];
      DP_ASSERT( (v[0]+2) < tbd.m_uiCoordSize );

      pair<IndexMap<1>::const_iterator,bool> pitb = indexData.indexMap.insert( make_pair( v, dp::checked_cast<unsigned int>(indexData.indexMap.size()) ) );
      indexData.newIndices.push_back( pitb.first->second );
    }

    indexData.newIndices.push_back( ~0 );
  }
}

void skipFanOrStripe( const A3DTessFaceData & tfd, unsigned int fixedIndices, unsigned int indicesPerElement, unsigned int & sizeIndex, unsigned int & startOffset )
{
  unsigned int count = tfd.m_puiSizesTriangulated[sizeIndex++];
  unsigned int offset = 0;
  for ( unsigned int i=0 ; i<count ; i++ )
  {
    offset += tfd.m_puiSizesTriangulated[sizeIndex++] & kA3DTessFaceDataNormalMask;    // hide any flag on the high bit!!
  }
  startOffset += fixedIndices + indicesPerElement * offset;
}

void skipTriangle( const A3DTessFaceData & tfd, unsigned int indicesPerElement, unsigned int & sizeIndex, unsigned int & startOffset )
{
  startOffset += indicesPerElement * tfd.m_puiSizesTriangulated[sizeIndex];
  sizeIndex++;
}

A3DStatus HOOPSLoader::parseTess3D( const A3DRiRepresentationItem * pRepItem, const A3DTess3D * tess, const BaseData & bd, const CascadedAttributes & parentCA, bool assembly )
{
  // The Tess3D uses BaseData, Tess3D Data, and face data
  A3DTessBaseData tbd;
  A3D_INITIALIZE_DATA( A3DTessBaseData, tbd );
  CHECK_RET( A3DTessBaseGet( tess, &tbd ) );

  A3DTess3DData t3d;
  A3D_INITIALIZE_DATA( A3DTess3DData, t3d );
  CHECK_RET( A3DTess3DGet( tess, &t3d ) );

  //if ( t3d.m_bHasFaces )    removed this check, as there are files with faces, but without thad flag set !
  {
    typedef struct
    {
      IndexData<1>  one;
      IndexData<2>  two;
      IndexData<3>  three;
    }                   IndexData;
    typedef map<EffectDataSharedPtr,IndexData>  EffectIndicesMap;

    EffectIndicesMap effectIndicesMap;

    // extract the faces
    for ( A3DUns32 ui=0 ; ui<t3d.m_uiFaceTessSize ; ui++ )
    {
      const A3DTessFaceData & tfd = t3d.m_psFaceTessData[ui];
      A3DUns32 flags = tfd.m_usUsedEntitiesFlags;

      CascadedAttributes ca( pRepItem, tess, t3d, ui, parentCA );
      EffectDataSharedPtr effectData = createMaterialEffect( ca.getAttributes() );

      //
      // NOTE: Data can be one or more of the following, represented in this order according to the docs.
      //        LAME!!
      //

      unsigned int sizeIndex = 0;
      unsigned int startOffset = 0;

      DP_ASSERT( ! ( flags & kA3DTessFaceDataPolyface ) );

      //Simple triangle.
      if ( flags & kA3DTessFaceDataTriangle )
      {
        gatherFromTriangles( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].two );
      }

      //Triangle fan.
      if ( flags & kA3DTessFaceDataTriangleFan )
      {
        gatherFromFans( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].two, false );
      }

      //Triangle strip. 
      if ( flags & kA3DTessFaceDataTriangleStripe )
      {
        gatherFromStrips( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].two, false );
      }

      //Simple triangle with one normal. 
      if ( flags & kA3DTessFaceDataTriangleOneNormal )
      {
        // don't care about kA3DTessFaceDataNormalSingle here !!
        gatherFromTrianglesOneNormal( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].two );
      }

      //Triangle fan with one normal and other characteristics depending on kA3DTessFaceDataNormalSingle. 
      if ( flags & kA3DTessFaceDataTriangleFanOneNormal )
      {
        if( flags & kA3DTessFaceDataNormalSingle )
        {
          DP_ASSERT( !"never passed this path" );
          gatherFromFans( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].two, true );
        }
        else
        {
          gatherFromFans( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].two, false );
        }
      }

      //Triangle strip with one normal and with indexes as specified by the kA3DTessFaceDataNormalSingle bit. 
      if ( flags & kA3DTessFaceDataTriangleStripeOneNormal )
      {
        if( flags & kA3DTessFaceDataNormalSingle )
        {
          DP_ASSERT( !"never passed this path" );
          gatherFromStrips( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].two, true );
        }
        else
        {
          gatherFromStrips( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].two, false );
        }
      }

      DP_ASSERT( ! ( flags & kA3DTessFaceDataPolyfaceTextured ) );

      //Simple triangle with texture. 
      if ( flags & kA3DTessFaceDataTriangleTextured )
      {
        if ( tfd.m_uiTextureCoordIndexesSize == 1 )
        {
          gatherFromTriangles( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].three );
        }
        else
        {
          DP_ASSERT( !"never passed this path" );
          reportUnsupported("multi-textured kA3DTessFaceDataTriangleTextured");
          skipTriangle( tfd, 3 * ( 2 + tfd.m_uiTextureCoordIndexesSize ), sizeIndex, startOffset );
        }
      }

      //Triangle fan with texture. 
      if ( flags & kA3DTessFaceDataTriangleFanTextured )
      {
        DP_ASSERT( !"never passed this path" );
        if ( tfd.m_uiTextureCoordIndexesSize == 1 )
        {
          gatherFromFans( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].three, false );
        }
        else
        {
          reportUnsupported("multi-textured kA3DTessFaceDataTriangleFanTextured");
          skipFanOrStripe( tfd, 0, 2 + tfd.m_uiTextureCoordIndexesSize, sizeIndex, startOffset );
        }
      }

      //Triangle strip with texture. 
      if ( flags & kA3DTessFaceDataTriangleStripeTextured )
      {
        DP_ASSERT( !"never passed this path" );
        if ( tfd.m_uiTextureCoordIndexesSize == 1 )
        {
          gatherFromStrips( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].two, false );
        }
        else
        {
          reportUnsupported("kA3DTessFaceDataTriangleStripeTextured");
          skipFanOrStripe( tfd, 0, 2 + tfd.m_uiTextureCoordIndexesSize, sizeIndex, startOffset );
        }
      }

      DP_ASSERT( ! ( flags & kA3DTessFaceDataPolyfaceOneNormalTextured ) );

      //Simple triangle with one normal and texture. 
      if ( flags & kA3DTessFaceDataTriangleOneNormalTextured )
      {
        DP_ASSERT( !"never passed this path" );
        if ( tfd.m_uiTextureCoordIndexesSize == 1 )
        {
          if( flags & kA3DTessFaceDataNormalSingle )
          {
            gatherFromTrianglesOneNormal( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].three );
          }
          else
          {
            gatherFromTriangles( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].three );
          }
        }
        else
        {
          reportUnsupported("kA3DTessFaceDataTriangleOneNormalTextured");
          skipTriangle( tfd, 1 + 3 * ( 1 + tfd.m_uiTextureCoordIndexesSize ), sizeIndex, startOffset );
        }
      }

      //Triangle fan with one normal and texture. 
      if ( flags & kA3DTessFaceDataTriangleFanOneNormalTextured )
      {
        DP_ASSERT( !"never passed this path" );
        if ( tfd.m_uiTextureCoordIndexesSize == 1 )
        {
          if( flags & kA3DTessFaceDataNormalSingle )
          {
            gatherFromFans( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].three, true );
          }
          else
          {
            gatherFromFans( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].three, false );
          }
        }
        else
        {
          reportUnsupported("kA3DTessFaceDataTriangleFanOneNormalTextured");
          if( flags & kA3DTessFaceDataNormalSingle )
          {
            skipFanOrStripe( tfd, 1, ( 1 + tfd.m_uiTextureCoordIndexesSize ), sizeIndex, startOffset );
          }
          else
          {
            skipFanOrStripe( tfd, 0, ( 2 + tfd.m_uiTextureCoordIndexesSize ), sizeIndex, startOffset );
          }
        }
      }

      //Triangle strip with one normal and texture. 
      if ( flags & kA3DTessFaceDataTriangleStripeOneNormalTextured )
      {
        DP_ASSERT( !"never passed this path" );
        if ( tfd.m_uiTextureCoordIndexesSize == 1 )
        {
          if( flags & kA3DTessFaceDataNormalSingle )
          {
            DP_ASSERT( !"never passed this path" );
            gatherFromStrips( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].three, true );
          }
          else
          {
            gatherFromStrips( tbd, t3d, tfd, sizeIndex, startOffset, effectIndicesMap[effectData].three, false );
          }
        }
        else
        {
          reportUnsupported("kA3DTessFaceDataTriangleStripeOneNormalTextured");
          if( flags & kA3DTessFaceDataNormalSingle )
          {
            skipFanOrStripe( tfd, 1, ( 1 + tfd.m_uiTextureCoordIndexesSize ), sizeIndex, startOffset );
          }
          else
          {
            skipFanOrStripe( tfd, 0, ( 2 + tfd.m_uiTextureCoordIndexesSize ), sizeIndex, startOffset );
          }
        }
      }

      if ( tfd.m_uiSizesWiresSize )
      {
        gatherFromLineStrips( tbd, t3d, tfd, effectIndicesMap[effectData].one );
      }
    }

    for ( EffectIndicesMap::iterator eimit = effectIndicesMap.begin(); eimit != effectIndicesMap.end() ; ++eimit )
    {
      unsigned int hints = 0;
      if( ((bd.behavior & kA3DGraphicsShow) != kA3DGraphicsShow ) ||
          (bd.behavior & kA3DGraphicsRemoved ) ) // not sure about this one..
      {
        // set the hidden flag for this geometry..
        hints = Object::DP_SG_HINT_ALWAYS_INVISIBLE;
      }
      if ( assembly )
      {
        hints |= Object::DP_SG_HINT_ASSEMBLY ;
      }

      GroupSharedPtr const& g = getCurrentGroup();
      if ( ! eimit->second.one.newIndices.empty() )
      {
        eimit->second.one.newIndices.pop_back();   // pop back last ~0
        DP_ASSERT( !eimit->second.one.newIndices.empty() );

        A3DDouble * verts  = tbd.m_pdCoords;
        vector<Vec3f> vertices( eimit->second.one.indexMap.size() );
        for ( IndexMap<1>::const_iterator it = eimit->second.one.indexMap.begin() ; it != eimit->second.one.indexMap.end() ; ++it )
        {
          const Vecnt<1,unsigned int> & idx = it->first;
          vertices[it->second] = Vec3f( (float)verts[idx[0]+0], (float)verts[idx[0]+1], (float)verts[idx[0]+2] );
        }

        vector<Vec3f> normals;
        g->addChild( createGeoNode( PRIMITIVE_LINE_STRIP, vertices, vector<Vec3f>(), vector<Vec2f>(), eimit->second.one.newIndices, bd.name, hints, eimit->first ) );
      }

      if ( ! eimit->second.two.newIndices.empty() )
      {
        A3DDouble * verts = tbd.m_pdCoords;
        A3DDouble * norms = t3d.m_pdNormals;
        size_t n = eimit->second.two.indexMap.size();
        vector<Vec3f> normals( n );
        vector<Vec3f> vertices( n );
        for ( IndexMap<2>::const_iterator it = eimit->second.two.indexMap.begin() ; it != eimit->second.two.indexMap.end() ; ++it )
        {
          const Vecnt<2,unsigned int> & idx = it->first;
          normals[it->second] = Vec3f( (float)norms[idx[0]+0], (float)norms[idx[0]+1], (float)norms[idx[0]+2] );
          vertices[it->second] = Vec3f( (float)verts[idx[1]+0], (float)verts[idx[1]+1], (float)verts[idx[1]+2] );
        }

        g->addChild( createGeoNode( PRIMITIVE_TRIANGLES, vertices, normals, vector<Vec2f>(), eimit->second.two.newIndices, bd.name, hints, eimit->first ) );
      }

      if ( ! eimit->second.three.newIndices.empty() )
      {
        A3DDouble * verts = tbd.m_pdCoords;
        A3DDouble * norms = t3d.m_pdNormals;
        A3DDouble * texts = t3d.m_pdTextureCoords;
        size_t n = eimit->second.three.indexMap.size();
        vector<Vec3f> normals( n );
        vector<Vec3f> vertices( n );
        vector<Vec2f> textures( n );
        for ( IndexMap<3>::const_iterator it = eimit->second.three.indexMap.begin() ; it != eimit->second.three.indexMap.end() ; ++it )
        {
          const Vecnt<3,unsigned int> & idx = it->first;
          normals[it->second] = Vec3f( (float)norms[idx[0]+0], (float)norms[idx[0]+1], (float)norms[idx[0]+2] );
          textures[it->second] = Vec2f( (float)texts[idx[1]+0], (float)texts[idx[1]+1] );
          vertices[it->second] = Vec3f( (float)verts[idx[2]+0], (float)verts[idx[2]+1], (float)verts[idx[2]+2] );
        }

        g->addChild( createGeoNode( PRIMITIVE_TRIANGLES, vertices, normals, textures, eimit->second.three.newIndices, bd.name, hints, eimit->first ) );
      }
    }
  }

  A3DTess3DGet( NULL, &t3d );
  A3DTessBaseGet( NULL, &tbd );
  return( A3D_SUCCESS );
}

A3DStatus HOOPSLoader::parseTess3DWire( const A3DRiRepresentationItem * pRepItem, const A3DTess3DWire * wire, const BaseData & baseData, const CascadedAttributes & parentCA, bool assembly )
{
  A3DTessBaseData tbd;
  A3D_INITIALIZE_DATA( A3DTessBaseData, tbd );
  CHECK_RET( A3DTessBaseGet( wire, &tbd ) );

  A3DTess3DWireData t3wd;
  A3D_INITIALIZE_DATA( A3DTess3DWireData, t3wd );
  CHECK_RET( A3DTess3DWireGet( wire, &t3wd ) );

  unsigned int hints = 0;
  if( ((baseData.behavior & kA3DGraphicsShow) != kA3DGraphicsShow ) ||
      (baseData.behavior & kA3DGraphicsRemoved ) ) // not sure about this one..
  {
    DP_ASSERT( !"never passed this path" );
    // set the hidden flag for this geometry..
    hints = Object::DP_SG_HINT_ALWAYS_INVISIBLE;
  }

  CascadedAttributes ca( pRepItem, parentCA );

  Vec3f color;
  EffectDataSharedPtr pmg = createLineEffect( ca.getAttributes(), color );

  A3DUns32  * indices = t3wd.m_puiSizesWires;
  unsigned int wIndex = 0;
  A3DDouble * verts  = tbd.m_pdCoords;

  // extract the wires
  if( t3wd.m_uiSizesWiresSize ) 
  {
    while( wIndex < t3wd.m_uiSizesWiresSize )
    {
      //
      // Data is packed as:
      //
      // N, index0, index1, .., countN-1, N, index0, ...
      //
      DP_ASSERT( wIndex < t3wd.m_uiSizesWiresSize );
      bool closed = !!( indices[wIndex] & kA3DTess3DWireDataIsClosing );
      bool continuous = !!( indices[wIndex] & kA3DTess3DWireDataIsContinuous );
      unsigned int count = indices[ wIndex++ ] & ~( kA3DTess3DWireDataIsClosing | kA3DTess3DWireDataIsContinuous );
      DP_ASSERT( wIndex + count <= t3wd.m_uiSizesWiresSize );

      vector< Vec3f > vertices;
      vector< Vec3f > colors;
      vertices.reserve( count );
      colors.reserve( count );

      unsigned int vnidx = closed ? indices[wIndex] : 0;

      if ( continuous )
      {
        // get the last vertex of the previous strip as the start vertex
        DP_ASSERT( 1 < wIndex );
        unsigned int v0idx = indices[wIndex-2];
        DP_ASSERT( (v0idx+2) < tbd.m_uiCoordSize );
        vertices.push_back( Vec3f((float)(verts[ v0idx + 0 ]), (float)(verts[ v0idx + 1 ]), (float)(verts[ v0idx + 2 ])) );
        colors.push_back( color );
      }

      for( unsigned int i = 0; i < count; i ++ )
      {
        // note that these indices represent the actual starting index in the array, not the primitive index (ie: multiplied by 3)
        DP_ASSERT( wIndex < t3wd.m_uiSizesWiresSize );
        unsigned int v0idx = indices[ wIndex++ ];

        DP_ASSERT( (v0idx+2) < tbd.m_uiCoordSize );
        vertices.push_back( Vec3f((float)(verts[ v0idx + 0 ]), (float)(verts[ v0idx + 1 ]), (float)(verts[ v0idx + 2 ])) );

        // duplicate color to all verts
        colors.push_back( color );
      }

      if ( closed )
      {
        DP_ASSERT( (vnidx+2) < tbd.m_uiCoordSize );
        vertices.push_back( Vec3f((float)(verts[ vnidx + 0 ]), (float)(verts[ vnidx + 1 ]), (float)(verts[ vnidx + 2 ])) );
        colors.push_back( color );
      }

      // create VAS
      VertexAttributeSetSharedPtr vasSP( VertexAttributeSet::create() );
      vasSP->setVertices( &vertices[0], count );
      vasSP->setColors  ( &colors[0], count );

      // create Primitive
      PrimitiveSharedPtr primSP( Primitive::create( PRIMITIVE_LINE_STRIP ) );
      primSP->setVertexAttributeSet( vasSP );

      // create GeoNode
      GeoNodeSharedPtr geoNodeSP( GeoNode::create() );
      geoNodeSP->setName( baseData.name );
      geoNodeSP->setHints( hints );
      geoNodeSP->setPrimitive( primSP );
      geoNodeSP->setMaterialEffect( pmg );

      // add it to the current Group
      getCurrentGroup()->addChild( geoNodeSP );
    }
  }
  else
  {
    //
    // if no WireSizes were specified, then data is all verts in the coords list
    // 
    unsigned int count = tbd.m_uiCoordSize / 3;

    vector< Vec3f > vertices;
    vector< Vec3f > colors;
    vertices.reserve( count );
    colors.reserve( count );

    for( unsigned int i = 0; i < tbd.m_uiCoordSize; i += 3 )
    {
      vertices.push_back( Vec3f((float)(verts[ i + 0 ]), (float)(verts[ i + 1 ]), (float)(verts[ i + 2 ])) );
      colors.push_back( color );
    }

    // create VAS
    VertexAttributeSetSharedPtr vasSP( VertexAttributeSet::create() );
    vasSP->setVertices( &vertices[0], dp::checked_cast< unsigned int >( vertices.size() ) );
    vasSP->setColors( &colors[0], dp::checked_cast< unsigned int >( colors.size() ) );

    // create Primitive
    PrimitiveSharedPtr primSP( Primitive::create( PRIMITIVE_LINE_STRIP ) );
    primSP->setVertexAttributeSet( vasSP );

    // create GeoNode
    GeoNodeSharedPtr geoNodeSP( GeoNode::create() );
    geoNodeSP->setName( baseData.name );
    geoNodeSP->setHints( hints );
    geoNodeSP->setPrimitive( primSP );
    geoNodeSP->setMaterialEffect( pmg );
    if ( assembly )
    {
      geoNodeSP->addHints( Object::DP_SG_HINT_ASSEMBLY );
    }

    // add it to the current Group
    getCurrentGroup()->addChild( geoNodeSP );
  }

  A3DTess3DWireGet( NULL, &t3wd );
  A3DTessBaseGet( NULL, &tbd );
  return( A3D_SUCCESS );
}

bool
HOOPSLoader::traverseBase( const A3DEntity * entity, BaseData & data )
{
  bool result = false;
  bool isGraphics = ( A3DEntityIsBaseWithGraphicsType( entity ) == TRUE );
  bool isBase     = ( A3DEntityIsBaseType( entity ) == TRUE );

  //
  // Objects may be a graphics and root, or just a root base.
  //

  if( isBase || isGraphics )
  {
    A3DRootBaseData sData;
    A3D_INITIALIZE_DATA( A3DRootBaseData, sData );

    if( A3DRootBaseGet( entity, &sData ) == A3D_SUCCESS )
    {
      if( sData.m_pcName && strlen( sData.m_pcName ) )
      {
        data.name = sData.m_pcName;
      }
      else
      {
        // this is the default anyway
        data.name = "unnamed";
      }

      A3DRootBaseGet( NULL, &sData );
    }

    if( isGraphics )
    {
      A3DRootBaseWithGraphicsData bgData;
      A3D_INITIALIZE_DATA( A3DRootBaseWithGraphicsData, bgData );

      if ( A3DRootBaseWithGraphicsGet( entity, &bgData ) == A3D_SUCCESS )
      {
        if ( bgData.m_pGraphics )
        {
          A3DGraphicsData gData;
          A3D_INITIALIZE_DATA( A3DGraphicsData, gData );

          if ( A3DGraphicsGet( bgData.m_pGraphics, &gData ) == A3D_SUCCESS )
          {
            data.layerIndex = gData.m_uiLayerIndex;
            data.styleIndex = gData.m_uiStyleIndex;
            data.behavior   = gData.m_usBehaviour;

            A3DGraphicsGet( NULL,  &gData );
          }
        }

        A3DRootBaseWithGraphicsGet( NULL, &bgData );
      }
    }

    return true;
  }
  else
  {
    return false;
  }
}

A3DStatus HOOPSLoader::traverseRepItem( const A3DRiRepresentationItem * pRepItem, const CascadedAttributes & parentCA, bool assembly )
{
#if HANDLE_BREP
  A3DRiRepresentationItemData sRiData;
  A3D_INITIALIZE_DATA( A3DRiRepresentationItemData, sRiData );
  CHECK_RET( A3DRiRepresentationItemGet( pRepItem, &sRiData ) );
  if ( sRiData.m_pTessBase )
  {
    A3DEEntityType eType;
    CHECK_RET( A3DEntityGetType( sRiData.m_pTessBase, &eType ) );
    DP_ASSERT( eType != kA3DTypeUnknown );
  }

  A3DEEntityType eType;
  CHECK_RET( A3DEntityGetType( pRepItem, &eType ) );
  DP_ASSERT( eType != kA3DTypeUnknown );

  switch( eType )
  {
    case kA3DTypeTess3D :
      CHECK_RET( traverseTess3D( pRepItem, parentCA, assembly ) );
      break;

    case kA3DTypeRiBrepModel :
      CHECK_RET( traverseBrepModel( (const A3DRiBrepModel *)pRepItem ) );
      break;

    case kA3DTypeRiCurve :
      CHECK_RET( traverseRiCurve( (const A3DRiCurve *)pRepItem ) );
      break;

    case kA3DTypeTess3DWire :               // A3DTess3DWire* pTess3DWire = (A3DTess3DWire*)sRiData.m_pTessBase;
    case kA3DTypeTessMarkup :               // A3DTessMarkup* pTessMarkup = (A3DTessMarkup*)sRiData.m_pTessBase;
    case kA3DTypeRiDirection :              // if(! sRiData.m_pTessBase ) { reportUnsupported( "Ri Directions" ); }
    case kA3DTypeRiPlane :                  // if(! sRiData.m_pTessBase ) { reportUnsupported( "Ri Planes" ); }
    case kA3DTypeRiPointSet :               // if(! sRiData.m_pTessBase ) { reportUnsupported( "Ri Point Sets" ); }
    case kA3DTypeRiPolyBrepModel :          // if(! sRiData.m_pTessBase ) { reportUnsupported( "Ri Poly Brep Model" ); }
    case kA3DTypeRiSet :                    // CHECK_RET( traverseSet( pRepItem, parentCA ) );
    case kA3DTypeRiCoordinateSystem :       // reportUnsupported( "Ri Coordinate Systems" );
    default :
      DP_ASSERT( !"never passed this path" );
      break;
  }
  return( A3D_SUCCESS );
#else
  A3DRiRepresentationItemData sRiData;
  A3D_INITIALIZE_DATA( A3DRiRepresentationItemData, sRiData );
  CHECK_RET( A3DRiRepresentationItemGet( pRepItem, &sRiData ) );

  if ( sRiData.m_pTessBase )
  {
    A3DEEntityType eType;
    CHECK_RET( A3DEntityGetType( sRiData.m_pTessBase, &eType ) );
    DP_ASSERT( eType != kA3DTypeUnknown );

    switch( eType )
    {
      case kA3DTypeTess3D :
        CHECK_RET( traverseTess3D( pRepItem, parentCA, assembly ) );
        break;
      case kA3DTypeTess3DWire :
        CHECK_RET( traverseTess3DWire( pRepItem, parentCA, assembly ) );
        break;
      default :
        DP_ASSERT( !"never passed this path" );
        break;
    }
  }

  return( A3DRiRepresentationItemGet( NULL, &sRiData ) );
#endif
}

A3DStatus HOOPSLoader::traverseTess3D( const A3DRiRepresentationItem * pRepItem, const CascadedAttributes & parentCA, bool assembly )
{
  A3DRiRepresentationItemData sRiData;
  A3D_INITIALIZE_DATA( A3DRiRepresentationItemData, sRiData );
  CHECK_RET( A3DRiRepresentationItemGet( pRepItem, &sRiData ) );

  if ( sRiData.m_pCoordinateSystem )
  {
    A3DRiCoordinateSystemData sCSysData;
    A3D_INITIALIZE_DATA( A3DRiCoordinateSystemData, sCSysData );
    CHECK_RET( A3DRiCoordinateSystemGet( sRiData.m_pCoordinateSystem, &sCSysData ) );

    pushGroup( traverseTransform( sCSysData.m_pTransformation ) );

    CHECK_RET( A3DRiCoordinateSystemGet( NULL, &sCSysData ) );
  }
  if ( sRiData.m_pTessBase )
  {
    BaseData bd;
    bool hasData = traverseBase( pRepItem, bd );
    CHECK_RET( parseTess3D( pRepItem, (A3DTess3D *)sRiData.m_pTessBase, bd, parentCA, assembly ) );
  }
  if ( sRiData.m_pCoordinateSystem )
  {
    popGroup();
  }

  return( A3DRiRepresentationItemGet( NULL, &sRiData ) );
}

A3DStatus HOOPSLoader::traverseTess3DWire( const A3DRiRepresentationItem * pRepItem, const CascadedAttributes & parentCA, bool assembly )
{
  A3DRiRepresentationItemData sRiData;
  A3D_INITIALIZE_DATA( A3DRiRepresentationItemData, sRiData );
  CHECK_RET( A3DRiRepresentationItemGet( pRepItem, &sRiData ) );

  if ( sRiData.m_pCoordinateSystem )
  {
    A3DRiCoordinateSystemData sCSysData;
    A3D_INITIALIZE_DATA( A3DRiCoordinateSystemData, sCSysData );
    CHECK_RET( A3DRiCoordinateSystemGet( sRiData.m_pCoordinateSystem, &sCSysData ) );

    pushGroup( traverseTransform( sCSysData.m_pTransformation ) );

    CHECK_RET( A3DRiCoordinateSystemGet( NULL, &sCSysData ) );
  }
  if ( sRiData.m_pTessBase )
  {
    BaseData bd;
    bool hasData = traverseBase( pRepItem, bd );
    CHECK_RET( parseTess3DWire( pRepItem, (A3DTess3D *)sRiData.m_pTessBase, bd, parentCA, assembly ) );
  }
  if ( sRiData.m_pCoordinateSystem )
  {
    popGroup();
  }

  return( A3DRiRepresentationItemGet( NULL, &sRiData ) );
}

A3DStatus HOOPSLoader::traverseBrepModel( const A3DRiBrepModel * pBrepModel )
{
  // What is A3DCopyAndAdaptBrepModel supposed to give?
  //      Seems, it always returns something of type kA3DTypeRiBrepModel -> infinite recursion
  //reportUnsupported("RiBrepModel");   // floods the message area with messages, which makes loading really slow!!
#if HANDLE_BREP
  A3DRiRepresentationItem * adaptedRepItem = 0;
  CHECK_RET( adaptBrepModels( pBrepModel, &adaptedRepItem ) );
  CHECK_RET( traverseRepItem( adaptedRepItem ) );
  return( A3DEntityDelete( adaptedRepItem ) );
#else
  return( A3D_SUCCESS );
#endif
}

A3DStatus HOOPSLoader::traverseRiCurve( const A3DRiCurve * pCurve )
{
  reportUnsupported("RiCurve");
#if 0
  A3DRiCurveData sData;
  A3D_INITIALIZE_DATA( A3DRiCurveData, sData );
  CHECK_RET( A3DRiCurveGet( pRICrv, &sData ) );
  return( A3DRiCurveGet( NULL, &sData ) );
#else
  return( A3D_SUCCESS );
#endif
}

A3DStatus HOOPSLoader::traverseSet( const A3DRiSet* pSet, const CascadedAttributes & parentCA )
{
  CascadedAttributes ca( pSet, parentCA );

  A3DRiSetData sData;
  A3D_INITIALIZE_DATA( A3DRiSetData, sData);

  CHECK_RET( A3DRiSetGet( pSet, &sData) );

  for(A3DUns32 ui=0; ui<sData.m_uiRepItemsSize;ui++)
  {
    CHECK_RET( traverseRepItem( sData.m_ppRepItems[ui], ca ) );
  }
  A3DRiSetGet( NULL, &sData);

  return( A3D_SUCCESS );
}

A3DStatus HOOPSLoader::traversePartDef( const A3DAsmPartDefinition * pPart, const CascadedAttributes & parentCA, bool assembly )
{
  CascadedAttributes ca( pPart, parentCA );

  A3DAsmPartDefinitionData sData;
  A3D_INITIALIZE_DATA( A3DAsmPartDefinitionData, sData );
  CHECK_RET( A3DAsmPartDefinitionGet(  pPart, &sData ) );

  for ( A3DUns32 ui=0 ; ui<sData.m_uiRepItemsSize ; ui++ )
  {
    CHECK_RET( traverseRepItem( sData.m_ppRepItems[ui], ca, assembly ) );
  }

  for ( A3DUns32 ui=0 ; ui<sData.m_uiAnnotationsSize ; ui++ )
  {
    // what's an annotation ?
    DP_ASSERT( !"never passed this path" );
  }

  for ( A3DUns32 ui=0 ; ui<sData.m_uiViewsSize ; ui++ )
  {
    // what's a view ?
    DP_ASSERT( !"never passed this path" );
  }

  for ( A3DUns32 ui=0 ; ui<sData.m_uiDrawingModelsSize ; ui++ )
  {
    // what's a drawing model ?
    DP_ASSERT( !"never passed this path" );
  }

  A3DAsmPartDefinitionGet( NULL,  &sData );
  return A3D_SUCCESS;
}

GroupSharedPtr HOOPSLoader::traverseTransform( const A3DMiscCartesianTransformation * trans3D )
{
  DP_ASSERT( trans3D );

  A3DMiscCartesianTransformationData sData;
  A3D_INITIALIZE_DATA( A3DMiscCartesianTransformationData, sData );

  A3DStatus iRet = A3DMiscCartesianTransformationGet( trans3D, &sData );

  if ( ( iRet != A3D_SUCCESS ) || ( sData.m_ucBehaviour == kA3DTransformationIdentity ) )
  {
    return( Group::create() );
  }

  Trafo trafo;

  Vec3f origin( float(sData.m_sOrigin.m_dX),  float(sData.m_sOrigin.m_dY),  float(sData.m_sOrigin.m_dZ)  );
  Vec3f xvec  ( float(sData.m_sXVector.m_dX), float(sData.m_sXVector.m_dY), float(sData.m_sXVector.m_dZ) );
  Vec3f yvec  ( float(sData.m_sYVector.m_dX), float(sData.m_sYVector.m_dY), float(sData.m_sYVector.m_dZ) );
  Vec3f scale ( float(sData.m_sScale.m_dX),   float(sData.m_sScale.m_dY),   float(sData.m_sScale.m_dZ)   );

  if( sData.m_ucBehaviour & kA3DTransformationTranslate )
  {
    trafo.setTranslation( origin );
  }

  if( sData.m_ucBehaviour & kA3DTransformationRotate )
  {
    // cross to get zvec
    Vec3f zvec = xvec ^ yvec;

    normalize( xvec );
    normalize( yvec );
    normalize( zvec );

    Mat33f mat( makeArray( xvec, yvec, zvec ) );
    Quatf quat( mat );

    trafo.setOrientation( quat );
  }

  if ( sData.m_ucBehaviour & kA3DTransformationMirror ||
        sData.m_ucBehaviour & kA3DTransformationScale ||
        sData.m_ucBehaviour & kA3DTransformationNonUniformScale )
  {
    if ( sData.m_ucBehaviour & kA3DTransformationMirror )
    {
      // mirror means: flip over z-axis
      scale[2] *= -1.0f;
    }
    trafo.setScaling( scale );
  }

  TransformSharedPtr transSP( Transform::create() );
  transSP->setTrafo( trafo );

  A3DMiscCartesianTransformationGet( NULL, &sData);

  return( transSP );
}

A3DStatus HOOPSLoader::traversePOccurrence( const A3DAsmProductOccurrence * pOccurrence, const CascadedAttributes & parentCA, bool assembly )
{
  // We need to get an answer from HOOPS SDK help, how to filter out multiple occuring elements!
  // It seems, that we get the same pOccurrence multiple times, but with different attributes
  // We filter on both (pOccurrence and A3DGraphStyle), but that does not solve the doubling of the geometry!

  CascadedAttributes ca( pOccurrence, parentCA );
  CascadedAttributesData cad( ca.getAttributes() );

  POMap::const_iterator it = m_POs.find( make_pair( pOccurrence, cad.getStyle() ) );
  if ( it != m_POs.end() )
  {
    DP_ASSERT( getCurrentGroup() );
    getCurrentGroup()->addChild( it->second );
    return( A3D_SUCCESS );
  }

  A3DAsmProductOccurrenceData sData;
  A3D_INITIALIZE_DATA( A3DAsmProductOccurrenceData, sData );
  CHECK_RET( A3DAsmProductOccurrenceGet( pOccurrence , &sData ) );

  // if this is an assembly, and holds more than just a part, mark the Group/Transform with DP_SG_HINT_ASSEMBLY
  // if this is an assembly, but not an assemblyGroup, mark the GeoNode with DP_SG_HINT_ASSEMBLY
  bool assemblyGroup = assembly && ( sData.m_pPrototype || sData.m_pExternalData || sData.m_uiPOccurrencesSize );
  bool assemblyPart = assembly && !assemblyGroup;

  //
  // If there is a "Location" then it is a transform, make sure all 
  // children are added to the transform.
  //
  GroupSharedPtr group = sData.m_pLocation ? traverseTransform( sData.m_pLocation ) : Group::create();
  group->setName( safeNodeName( pOccurrence ) );
  if ( assemblyGroup )
  {
    group->addHints( Object::DP_SG_HINT_ASSEMBLY );
  }
  pushGroup( group );
  m_POs[make_pair(pOccurrence,cad.getStyle())] = group;

  if ( sData.m_pPrototype )
  {
    CHECK_RET( traversePOccurrence( sData.m_pPrototype, ca, true ) );
  }

  if ( sData.m_pExternalData )
  {
    // What's external data?
    CHECK_RET( traversePOccurrence( sData.m_pExternalData, ca, true ) );
  }

  for ( A3DUns32 ui=0 ; ui<sData.m_uiPOccurrencesSize ; ui++ )
  {
    CHECK_RET( traversePOccurrence( sData.m_ppPOccurrences[ui], ca ) );
  }

  if ( sData.m_pPart )
  {
    CHECK_RET( traversePartDef( sData.m_pPart, ca, assemblyPart ) );
  }

  for ( A3DUns32 ui=0 ; ui<sData.m_uiEntityReferenceSize ; ui++ )
  {
    // what's an entity reference ?
    DP_ASSERT( !"never passed this path" );
  }

  for ( A3DUns32 ui=0 ; ui<sData.m_uiAnnotationsSize ; ui++ )
  {
    // what's an annotation ?
    //DP_ASSERT( !"never passed this path" );
  }

  for ( A3DUns32 ui=0 ; ui<sData.m_uiViewsSize ; ui++ )
  {
    // what's a view ?
    //DP_ASSERT( !"never passed this path" );
  }

  if ( sData.m_pEntityFilter )
  {
    // what's an entity filter ?
    DP_ASSERT( !"never passed this path" );
  }

  for ( A3DUns32 ui=0 ; ui<sData.m_uiDisplayFilterSize ; ui++ )
  {
    // what's a display filter ?
    //DP_ASSERT( !"never passed this path" );
  }

  for ( A3DUns32 ui=0 ; ui<sData.m_uiSceneDisplayParameterSize ; ui++ )
  {
    // what's a scene display parameter ?
    //DP_ASSERT( !"never passed this path" );
  }

  popGroup();

  // do we need to handle m_eProductLoadStatus ?
  // do we need to handle m_uiProductFlags ?
  // do we need to handle m_bUnitFromCAD and m_dUnit ?
  // do we need to handle m_eModellerType ?

  return( A3DAsmProductOccurrenceGet( NULL, &sData ) );
}

A3DStatus HOOPSLoader::traverseGlobal( A3DGlobal * global )
{
  A3DStatus iRet=A3D_SUCCESS;
  A3DGlobalData sData;
  A3D_INITIALIZE_DATA( A3DGlobalData, sData);

  iRet=A3DGlobalGet( global, &sData);
  if ( iRet==A3D_SUCCESS )
  {
    A3DUns32 ui,uiSize=sData.m_uiColorsSize;
    if (uiSize)
    {
      A3DGraphRgbColorData sColorData;
      A3D_INITIALIZE_DATA( A3DGraphRgbColorData, sColorData);

      for (ui=0;ui<uiSize;ui++)
      {
        //
        // WTF!! WHY DO YOU HAVE TO MULTIPLY BY 3 HERE??  STUPID!!
        //
        if( A3DGlobalGetGraphRgbColorData( ui*3, &sColorData ) == A3D_SUCCESS )
        {
          m_rgbColors.push_back( Vec3f( float( sColorData.m_dRed ), float( sColorData.m_dGreen ), float( sColorData.m_dBlue ) ) );
        }
        else
        {
          // some default?
          m_rgbColors.push_back( Vec3f( 0.8f, 0.8f, 0.8f ) );
        }
      }
    }

    if (sData.m_uiStylesSize)
    {
      uiSize=sData.m_uiStylesSize;
      A3DGraphStyleData sStyleData;
      A3D_INITIALIZE_DATA( A3DGraphStyleData, sStyleData);

      for (ui=0;ui<uiSize;ui++)
      {
        iRet=A3DGlobalGetGraphStyleData(ui,&sStyleData);
        m_styles.push_back( sStyleData );

        A3DGlobalGetGraphStyleData( A3D_DEFAULT_STYLE_INDEX, &sStyleData );
      }
    }

    if (sData.m_uiLinePatternsSize)
    {
      uiSize=sData.m_uiLinePatternsSize;
      A3DGraphLinePatternData sLinePatternData;
      A3D_INITIALIZE_DATA( A3DGraphLinePatternData, sLinePatternData);

      //
      // We compute a stipple pattern based on the values in the lengths array.  First, compute a
      // percentage for each value in the lengths array.  Since the stipple pattern is 16 bits, we 
      // compute the number of bits based on the length in the array.  
      //
      // We start with "pen down", but m_dPhase is the distance from the start to begin, so we 
      // really start with pen-up for m_dPhase, then alternate down, up, for each set of values in
      // the m_pdLengths array.
      //

      for (ui=0;ui<uiSize;ui++)
      {
        if( A3DGlobalGetGraphLinePatternData(ui,&sLinePatternData) == A3D_SUCCESS )
        {
          // first, get total
          // start with dPhase
          A3DDouble total = sLinePatternData.m_dPhase;
          for( A3DUns32 l = 0; l < sLinePatternData.m_uiNumberOfLengths; l ++ )
          {
            total += sLinePatternData.m_pdLengths[ l ];
          }

          // now loop through and compute values (+1 for the dPhase)
          vector< unsigned int > values( sLinePatternData.m_uiNumberOfLengths + 1 );

          // values are number of bits in each position in the stipple pattern
          values[0] = (unsigned int)( floor( sLinePatternData.m_dPhase / total * 16.0 + 0.5 ) ); 
          for( A3DUns32 l = 0; l < sLinePatternData.m_uiNumberOfLengths; l ++ )
          {
            values[l+1] = (unsigned int)( floor( sLinePatternData.m_pdLengths[l] / total * 16.0 + 0.5 ) ); 
          }

          unsigned short pattern = 0;

          // start by shifting in zeroes
          unsigned char shiftIn = 0x00;
          for( size_t v = 0; v < values.size(); v ++ )
          {
            for( unsigned int s = 0; s < values[v]; s ++ )
            {
              pattern <<= 1;
              pattern |= shiftIn;
            }

            // swap our shiftin bit for pen-up / pen-down
            shiftIn ^= 0x01;
          }

          m_linePatterns.push_back( pattern );
        }

        A3DGlobalGetGraphLinePatternData(A3D_DEFAULT_LINEPATTERN_INDEX,&sLinePatternData);
      }
    }

    if (sData.m_uiMaterialsSize)
    {
      uiSize=sData.m_uiMaterialsSize;

      A3DBool bIsTexture = FALSE;
      for (ui=0;ui<uiSize;ui++)
      {
        A3DGlobalIsMaterialTexture(ui,&bIsTexture);

        if (bIsTexture)
        {  
          A3DGraphTextureApplicationData sTextureApplicationData;
          A3D_INITIALIZE_DATA( A3DGraphTextureApplicationData, sTextureApplicationData);
          iRet=A3DGlobalGetGraphTextureApplicationData(ui,&sTextureApplicationData);

          reportUnsupported( "Textured Materials" );

          A3DGlobalGetGraphTextureApplicationData(A3D_DEFAULT_MATERIAL_INDEX,&sTextureApplicationData);
        }
        else  
        {
          A3DGraphMaterialData sMaterialData;
          A3D_INITIALIZE_DATA( A3DGraphMaterialData, sMaterialData);
          if( A3DGlobalGetGraphMaterialData(ui,&sMaterialData) == A3D_SUCCESS )
          {
            // materials are supported, we just don't do anything with them here...
          }

          A3DGlobalGetGraphMaterialData(A3D_DEFAULT_MATERIAL_INDEX,&sMaterialData);
        }
      }
    }

    if (sData.m_uiTextureDefinitionsSize)
    {
      uiSize=sData.m_uiTextureDefinitionsSize;
      A3DGraphTextureDefinitionData sTextureDefinitionData;
      A3D_INITIALIZE_DATA( A3DGraphTextureDefinitionData, sTextureDefinitionData);
      for (ui=0;ui<uiSize;ui++)
      {
        iRet=A3DGlobalGetGraphTextureDefinitionData(ui,&sTextureDefinitionData);

        // no report here
        
        A3DGlobalGetGraphTextureDefinitionData(A3D_DEFAULT_TEXTURE_DEFINITION_INDEX,&sTextureDefinitionData);
      }
    }

    if (sData.m_uiPicturesSize)
    {
      uiSize=sData.m_uiPicturesSize;
      A3DGraphPictureData sPictureData;
      A3D_INITIALIZE_DATA( A3DGraphPictureData, sPictureData);

      for (ui=0;ui<uiSize;ui++)
      {
        iRet=A3DGlobalGetGraphPictureData(ui,&sPictureData);

        // no report here
        
        A3DGlobalGetGraphPictureData(A3D_DEFAULT_MATERIAL_INDEX,&sPictureData);
      }
    }

    if (sData.m_uiFillPatternsSize)
    {
      uiSize=sData.m_uiFillPatternsSize;
      A3DEEntityType ePatternType;
      for (ui=0;ui<uiSize;ui++)
      {
        iRet=A3DGlobalGetFillPatternType(ui,&ePatternType);

        switch (ePatternType)
        {
        case kA3DTypeGraphHatchingPattern:
          {
            A3DGraphHatchingPatternData sHatchingPatternData;
            A3D_INITIALIZE_DATA( A3DGraphHatchingPatternData, sHatchingPatternData);
            iRet=A3DGlobalGetGraphHatchingPatternData(A3D_DEFAULT_PATTERN_INDEX,&sHatchingPatternData);

            // do something..

            A3DGlobalGetGraphHatchingPatternData(A3D_DEFAULT_PATTERN_INDEX,&sHatchingPatternData);
            break;
          }
        case kA3DTypeGraphSolidPattern:
          {
            A3DGraphSolidPatternData sSolidPatternData;
            A3D_INITIALIZE_DATA( A3DGraphSolidPatternData, sSolidPatternData);
            iRet=A3DGlobalGetGraphSolidPatternData(A3D_DEFAULT_PATTERN_INDEX,&sSolidPatternData);

            // do something..
            
            A3DGlobalGetGraphSolidPatternData(A3D_DEFAULT_PATTERN_INDEX,&sSolidPatternData);
            break;
          }
        case kA3DTypeGraphDottingPattern:
          {
            A3DGraphDottingPatternData sDottingPatternData;
            A3D_INITIALIZE_DATA( A3DGraphDottingPatternData, sDottingPatternData);
            iRet=A3DGlobalGetGraphDottingPatternData(A3D_DEFAULT_PATTERN_INDEX,&sDottingPatternData);

            // do something..
            
            A3DGlobalGetGraphDottingPatternData(A3D_DEFAULT_PATTERN_INDEX,&sDottingPatternData);
            break;
          }
        case kA3DTypeGraphVPicturePattern:
          {
            A3DGraphVPicturePatternData sVPicturePatternData;
            A3D_INITIALIZE_DATA( A3DGraphVPicturePatternData, sVPicturePatternData);
            iRet=A3DGlobalGetGraphVPicturePatternData(A3D_DEFAULT_PATTERN_INDEX,&sVPicturePatternData);

            // do something..
            
            A3DGlobalGetGraphVPicturePatternData(A3D_DEFAULT_PATTERN_INDEX,&sVPicturePatternData);
            break;
          }
        }
      }
    }
  }

  return A3D_SUCCESS;
}

A3DStatus 
HOOPSLoader::traverseModel()
{
  A3DStatus iRet;

  // first, traverse global - contains all the file-wide palettes
  A3DGlobal * global = 0;
  iRet = A3DGlobalGetPointer( &global );
  if( iRet == A3D_SUCCESS && global )
  {
    traverseGlobal( global );
  }

  // now, traverse model data
  A3DAsmModelFileData sData;
  A3D_INITIALIZE_DATA( A3DAsmModelFileData, sData);
  CHECK_RET( A3DAsmModelFileGet( m_modelFile , &sData ) );

  // create the root entry of attributes
  CascadedAttributes ca;

  // look through all occurrences
  A3DUns32 ui;
  for (ui=0;ui<sData.m_uiPOccurrencesSize;ui++)
  {
    traversePOccurrence( sData.m_ppPOccurrences[ui], ca );
  }

  return( A3DAsmModelFileGet( NULL, &sData ) );
}

SceneSharedPtr
HOOPSLoader::load( string const& filename, dp::util::FileFinder const& fileFinder, dp::sg::ui::ViewStateSharedPtr & viewState )
{
  if ( sInitializationCount == 0 )
  {
    throw ( std::runtime_error( "HOOPSLoader not yet initialized" ) );
  }

  if ( !dp::util::fileExists(filename) )
  {
    throw dp::FileNotFoundException( filename );
  }


  // set the locale temporarily to the default "C" to make atof behave predictably
  dp::util::Locale tl("C");

  if( !loadFile( filename ) )
  {
    throw std::runtime_error( "Failed to load file: " + filename );
  }

  m_scene = Scene::create();
  // create the root node
  GroupSharedPtr root( Group::create() );
  root->setName( filename );

  m_scene->setRootNode( root );
  pushGroup( root );

  // traverse the whole file building tree
  traverseModel();

  // save scene if we built one
  SceneSharedPtr scene = m_scene;

  // clear everything in case we are called multiple times
  clear();

  return scene;
}

dp::sg::core::ParameterGroupDataSharedPtr HOOPSLoader::createGeometryParameterGroupData( const A3DMiscCascadedAttributes * pAttr )
{
  CascadedAttributesData cad( pAttr );
  const A3DGraphStyleData & style = cad.getStyle();

  // TODO: style.m_dWidth is in mm! What should we do to translate into pixels here?
  //       passing it as it is into the effect for now
  DP_ASSERT( style.m_dWidth );

  dp::sg::core::ParameterGroupDataSharedPtr parameterGroupData;
  GraphStyleGeometryMap::const_iterator it = m_geometryParameterGroupDatas.find( style );
  if ( it != m_geometryParameterGroupDatas.end() )
  {
    parameterGroupData = it->second;
  }
  else
  {
    dp::fx::ParameterGroupDataSharedPtr fxParameterGroupData = dp::fx::EffectLibrary::instance()->getParameterGroupData("standardGeometryParameters");
    parameterGroupData = dp::sg::core::ParameterGroupData::create( fxParameterGroupData );
    DP_ASSERT( parameterGroupData );
    DP_VERIFY( parameterGroupData->setParameter<float>( "lineWidth", (float)style.m_dWidth ) );

    m_geometryParameterGroupDatas[style] = parameterGroupData;
  }
  return( parameterGroupData );
}

EffectDataSharedPtr HOOPSLoader::createMaterialEffect( const A3DMiscCascadedAttributes * pAttr )
{
  CascadedAttributesData cad( pAttr );
  const A3DGraphStyleData & style = cad.getStyle();

  DP_ASSERT( !style.m_bVPicture && ( style.m_uiLinePatternIndex < m_linePatterns.size() ) );

  EffectDataSharedPtr materialEffect;
  GraphStyleMap::const_iterator it = m_materialEffects.find( style );
  if ( it != m_materialEffects.end() )
  {
    materialEffect = it->second;
  }
  else
  {
    std::ostringstream oss;
    oss << "Material" << m_materialCounter++;

    materialEffect = createStandardMaterialData();
    materialEffect->setName( oss.str() );

    const ParameterGroupDataSharedPtr & parameterGroupData = materialEffect->findParameterGroupData( string( "standardMaterialParameters" ) );
    DP_ASSERT( parameterGroupData );
    DP_VERIFY( parameterGroupData->setParameter( "lineStipplePattern", m_linePatterns[style.m_uiLinePatternIndex] ) );

    if ( style.m_bMaterial )
    {
      A3DUns32 materialIndex = style.m_uiRgbColorIndex;
      A3DBool isTexture;
      DP_VERIFY( A3DGlobalIsMaterialTexture( style.m_uiRgbColorIndex, &isTexture ) == A3D_SUCCESS );
      if ( isTexture )
      {
        A3DGraphTextureApplicationData sTextureApplicationData;
        A3D_INITIALIZE_DATA( A3DGraphTextureApplicationData, sTextureApplicationData );
        if ( A3DGlobalGetGraphTextureApplicationData( style.m_uiRgbColorIndex, &sTextureApplicationData ) == A3D_SUCCESS )
        {
          materialIndex = sTextureApplicationData.m_uiMaterialIndex;

          A3DGraphTextureDefinitionData sTextureDefinitionData;
          A3D_INITIALIZE_DATA( A3DGraphTextureDefinitionData, sTextureDefinitionData );
          if ( A3DGlobalGetGraphTextureDefinitionData( sTextureApplicationData.m_uiTextureDefinitionIndex, &sTextureDefinitionData ) == A3D_SUCCESS )
          {
            A3DGraphPictureData sPictureData;
            A3D_INITIALIZE_DATA( A3DGraphPictureData, sPictureData );
            if ( A3DGlobalGetGraphPictureData( sTextureDefinitionData.m_uiPictureIndex, &sPictureData ) == A3D_SUCCESS )
            {
              DP_ASSERT( sPictureData.m_eFormat == kA3DPictureJpg );
              // now we can load the data from sPictureData.m_pucBinaryData, of size sPictureData.m_uiSize
              // sPictureData.m_uiPixelWidth and sPictureData.m_uiPixelWidth might add some information?
              A3DGlobalGetGraphPictureData( A3D_DEFAULT_MATERIAL_INDEX, &sPictureData );
            }
            DP_ASSERT( sTextureDefinitionData.m_ucTextureDimension == 2 );
            DP_ASSERT( sTextureDefinitionData.m_eMappingType == kA3DTextureMappingTypeStored );
            DP_ASSERT( sTextureDefinitionData.m_eMappingOperator == kA3DTextureMappingOperatorUnknown );
            DP_ASSERT( sTextureDefinitionData.m_pOperatorTransfo == nullptr );
            DP_ASSERT( sTextureDefinitionData.m_uiMappingAttributes == kA3DTextureMappingDiffuse );
            DP_ASSERT( sTextureDefinitionData.m_uiMappingAttributesIntensitySize == 0 );
            DP_ASSERT( sTextureDefinitionData.m_pdMappingAttributesIntensity == nullptr );
            DP_ASSERT( sTextureDefinitionData.m_uiMappingAttributesComponentsSize == 0 );
            DP_ASSERT( sTextureDefinitionData.m_pucMappingAttributesComponents == nullptr );
            // ignore sTextureDefinitionData.m_eTextureFunction for now (holds Modulate/Replace/Blend/Decal)
            DP_ASSERT(    ( sTextureDefinitionData.m_dRed == 0.0 ) && ( sTextureDefinitionData.m_dGreen == 0.0 )
                        &&  ( sTextureDefinitionData.m_dBlue == 0.0 ) && ( sTextureDefinitionData.m_dAlpha == 0.0 ) );
            DP_ASSERT( sTextureDefinitionData.m_eBlend_src_RGB == kA3DTextureBlendParameterUnknown );
            DP_ASSERT( sTextureDefinitionData.m_eBlend_dst_RGB == kA3DTextureBlendParameterUnknown );
            DP_ASSERT( sTextureDefinitionData.m_eBlend_src_Alpha == kA3DTextureBlendParameterUnknown );
            DP_ASSERT( sTextureDefinitionData.m_eBlend_dst_Alpha == kA3DTextureBlendParameterUnknown );
            DP_ASSERT( sTextureDefinitionData.m_ucTextureApplyingMode == kA3DTextureApplyingModeNone );
            DP_ASSERT( sTextureDefinitionData.m_eTextureAlphaTest == kA3DTextureAlphaTestUnknown );
            DP_ASSERT( sTextureDefinitionData.m_dAlphaTestReference == 0.0 );
            // handle sTextureDefinitionData.m_eTextureWrappingModeS and .m_eTextureWrappingModeT
            DP_ASSERT( sTextureDefinitionData.m_pTextureTransfo == nullptr );

            A3DGlobalGetGraphTextureDefinitionData( A3D_DEFAULT_TEXTURE_DEFINITION_INDEX, &sTextureDefinitionData );
          }

          DP_ASSERT( sTextureApplicationData.m_iUVCoordinatesIndex == 0 );
          DP_ASSERT( sTextureApplicationData.m_uiNextTextureApplicationIndex == A3D_DEFAULT_MATERIAL_INDEX );
          A3DGlobalGetGraphTextureApplicationData( A3D_DEFAULT_MATERIAL_INDEX, &sTextureApplicationData );
        }
      }
      A3DGraphMaterialData sMaterialData;
      A3D_INITIALIZE_DATA( A3DGraphMaterialData, sMaterialData);
      if ( A3DGlobalGetGraphMaterialData( materialIndex, &sMaterialData ) == A3D_SUCCESS )
      {
        Vec3f color;
        if ( getRGBColor( sMaterialData.m_uiAmbient, color ) )
        {
          DP_VERIFY( parameterGroupData->setParameter( "frontAmbientColor", color ) );
          DP_VERIFY( parameterGroupData->setParameter( "backAmbientColor", color ) );
        }

        if ( getRGBColor( sMaterialData.m_uiDiffuse, color ) )
        {
          DP_VERIFY( parameterGroupData->setParameter( "frontDiffuseColor", color ) );
          DP_VERIFY( parameterGroupData->setParameter( "backDiffuseColor", color ) );
        }

        if ( getRGBColor( sMaterialData.m_uiEmissive, color ) )
        {
          DP_VERIFY( parameterGroupData->setParameter( "frontEmissiveColor", color ) );
          DP_VERIFY( parameterGroupData->setParameter( "backEmissiveColor", color ) );
        }

        if ( getRGBColor( sMaterialData.m_uiSpecular, color ) )
        {
          DP_VERIFY( parameterGroupData->setParameter( "frontSpecularColor", color ) );
          DP_VERIFY( parameterGroupData->setParameter( "backSpecularColor", color ) );
        }

        // scale shininess [0,1] to specular exponent [0,128]
        float se = (float)sMaterialData.m_dShininess * 128;
        DP_VERIFY( parameterGroupData->setParameter( "frontSpecularExponent", se ) );
        DP_VERIFY( parameterGroupData->setParameter( "backSpecularExponent", se ) );

        A3DGlobalGetGraphMaterialData(A3D_DEFAULT_MATERIAL_INDEX, &sMaterialData);
      }
    }
    else
    {
      Vec3f faceColor;
      // otherwise, see if we can find the face color
      if ( getRGBColor( style.m_uiRgbColorIndex, faceColor ) )
      {
        DP_VERIFY( parameterGroupData->setParameter( "frontDiffuseColor", faceColor ) );
        DP_VERIFY( parameterGroupData->setParameter( "backDiffuseColor", faceColor ) );
      }
    }

    // set transparency but only if it has value
    if ( style.m_bIsTransparencyDefined && style.m_ucTransparency != 255 )
    {
      float v = 1.0f - float( style.m_ucTransparency ) / 255.0f;
      DP_VERIFY( parameterGroupData->setParameter( "frontOpacity", v ) );
      DP_VERIFY( parameterGroupData->setParameter( "backOpacity", v ) );
      materialEffect->setTransparent( true );
    }

    materialEffect->setParameterGroupData( createGeometryParameterGroupData( pAttr ) );

    m_materialEffects[style] = materialEffect;
  }

  return( materialEffect );
}

EffectDataSharedPtr HOOPSLoader::createLineEffect( const A3DMiscCascadedAttributes * pAttr, Vec3f & color )
{
  EffectDataSharedPtr materialEffect = createMaterialEffect( pAttr );

  // get the line color
  const ParameterGroupDataSharedPtr & parameterGroupData = materialEffect->findParameterGroupData( string( "standardMaterialParameters" ) );
  DP_ASSERT( parameterGroupData );
  {
    const dp::fx::ParameterGroupSpecSharedPtr & pgs = parameterGroupData->getParameterGroupSpec();
    color = parameterGroupData->getParameter<Vec3f>( pgs->findParameterSpec( "frontDiffuseColor" ) );
  }
#if !defined(NDEBUG)
  CascadedAttributesData cad( pAttr );
  const A3DGraphStyleData & style = cad.getStyle();
  DP_ASSERT( !style.m_bMaterial );
  Vec3f dbgColor;
  if ( getRGBColor( style.m_uiRgbColorIndex, dbgColor ) )
  {
    DP_ASSERT( color == dbgColor );
  }
#endif

  return( materialEffect );
}
