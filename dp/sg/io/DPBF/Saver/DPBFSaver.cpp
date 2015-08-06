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


// DPBFSaver.cpp : Defines the entry point for the DLL application.
//

#include <dp/DP.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/util/PlugInCallback.h>
#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/ClipPlane.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/MatrixCamera.h>
#include <dp/sg/core/ParallelCamera.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Sampler.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/TextureHost.h>
#include <dp/sg/core/Transform.h>
#include <dp/sg/core/HandledObject.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/io/PlugInterfaceID.h>
#include <dp/sg/io/DPBF/Saver/inc/DPBFSaver.h>
#include <dp/util/Locale.h>
#include <algorithm>
#include <functional>
#include <string>

using namespace dp::sg::core;
using namespace dp::math;
using namespace dp::util;
using namespace dp::util;
using dp::util::UPITID;
using dp::util::UPIID;
using dp::util::PlugIn;
using std::pair;
using std::vector;
using std::string;
using std::map;
using std::for_each;
using std::find;
using std::transform;
using std::unary_function;
using std::make_pair;

#if defined (_WIN64)
template<unsigned int HI, unsigned int LO>
struct mask64
{
  static const unsigned long long mask = (((unsigned long long)HI<<32)|LO);
};
#endif

// convenient macro
#ifndef  INVOKE_CALLBACK
  #define INVOKE_CALLBACK(cb) if ( callback() ) callback()->cb
#endif


inline float2_t& assign(float2_t& lhs, const Vec2f& rhs)
{
  lhs[0] = rhs[0];
  lhs[1] = rhs[1];
  return lhs;
}

inline float3_t& assign(float3_t& lhs, const Vec3f& rhs)
{
  lhs[0] = rhs[0];
  lhs[1] = rhs[1];
  lhs[2] = rhs[2];
  return lhs;
}

template <typename Vec4Type>
inline float4_t& assign(float4_t& lhs, const Vec4Type& rhs)
{
  lhs[0] = rhs[0];
  lhs[1] = rhs[1];
  lhs[2] = rhs[2];
  lhs[3] = rhs[3];
  return lhs;
}

inline float44_t & assign( float44_t & lhs, const Mat44f & rhs )
{
  assign( lhs[0], rhs[0] );
  assign( lhs[1], rhs[1] );
  assign( lhs[2], rhs[2] );
  assign( lhs[3], rhs[3] );
  return( lhs );
}

inline trafo_t & assign( trafo_t & lhs, const Trafo & rhs )
{
  assign(lhs.orientation, rhs.getOrientation());
  assign(lhs.scaling, rhs.getScaling());
  assign(lhs.scaleOrientation, rhs.getScaleOrientation());
  assign(lhs.translation, rhs.getTranslation());
  assign(lhs.center, rhs.getCenter());
  return(lhs);
}

// supported Plug Interface ID
const UPITID PITID_SCENE_SAVER(UPITID_SCENE_SAVER, UPITID_VERSION); // plug-in type
const UPIID  PIID_DPBF_SCENE_SAVER(".DPBF", PITID_SCENE_SAVER); // plug-in ID

#if defined(_WIN32)
BOOL APIENTRY DllMain(HANDLE hModule, DWORD  reason, LPVOID lpReserved)
{
  if (reason == DLL_PROCESS_ATTACH)
  {
    int i=0;
  }

  return TRUE;
}
#endif

bool getPlugInterface(const UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  if ( piid==PIID_DPBF_SCENE_SAVER )
  {
    pi = DPBFSaver::create();
    return( !!pi );
  }
  return false;
}

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(PIID_DPBF_SCENE_SAVER);
}

DPBFSaverSharedPtr DPBFSaver::create()
{
  return( std::shared_ptr<DPBFSaver>( new DPBFSaver() ) );
}

DPBFSaver::DPBFSaver()
{
}

DPBFSaver::~DPBFSaver()
{
}

bool DPBFSaver::save( SceneSharedPtr const& scene, dp::sg::ui::ViewStateSharedPtr const& viewState, string const& filename)
{
  // set locale temporarily to standard "C" locale
  dp::util::Locale tl("C");

  DPBFSaveTraverser saver;
  saver.setViewState(viewState);
  saver.setFileName( filename );
  bool success = saver.preCalculateFileSize(scene);
  if ( success )
  {
    saver.apply(scene);
    success = saver.getSuccess();
  }
  if ( ! success )
  {
    INVOKE_CALLBACK( onInvalidFile( filename, saver.getErrorMessage() ) );
  }
  return( success );
}

DPBFSaveTraverser::DPBFSaveTraverser()
: m_calculateStorageRequirements(false)
, m_preCalculatedFileSize(0)
, m_fileOffset(0)
, m_pic(NULL)
, m_success(false)
{
  m_basePaths.resize( 2 );
  m_basePaths[1] = dp::home() + std::string( "/" );
  convertPath( m_basePaths[1] );
}

bool DPBFSaveTraverser::preCalculateFileSize( const SceneSharedPtr & scene )
{
  m_calculateStorageRequirements = true;
  m_preCalculatedFileSize = 0;
  apply(scene); // apply non-culled!
  DP_ASSERT(m_preCalculatedFileSize);
  m_calculateStorageRequirements = false;

  bool ok = ( m_preCalculatedFileSize < UINT_MAX );
  if ( !ok )
  {
    m_errorMessage = "File size would exceed addressable storage of 4GB!";
  }

  return( ok );
}

uint_t DPBFSaveTraverser::alloc(void *& objPtr, unsigned int numBytes)
{
  // undefined behaviour if this gets called while calculating storage requirements
  DP_ASSERT(!calculatingStorageRequirements());

  // this function requires a valid file mapping
  DP_ASSERT( m_fm && m_fm->isValid() ); 
  
  // wrong calculated storage requirements if this fires!
  // this happens if pseudoAlloc calls do not match their real alloc counterparts.
  // this might be critical if we are near a page boundary. note that the file mapping 
  // always enlarges the mapping size to a multiple of the systems page size. 
  DP_ASSERT(m_fileOffset+numBytes<=m_preCalculatedFileSize);
  
  // object size cannot be zero, see Offset_AutoPtr
  DP_ASSERT(numBytes);  

  // always allocate a multiple of 4 bytes to keep offsets 4-byte aligned
  numBytes = (numBytes+3)&~3;

  // we map the object at the current file offset
  objPtr = m_fm->mapIn( m_fileOffset, numBytes );
  if ( ! objPtr )
  {
    if ( m_pic )
    {
      m_pic->onFileMappingFailed(m_fm->getLastError());
    }
    return 0;
  }

  // advance the file offset by numBytes for next alloc
  unsigned int offset = m_fileOffset;
  m_fileOffset += numBytes;
  
  // initialize object to zero
  memset(objPtr, 0, numBytes);
#if !defined (NDEBUG) && defined(_WIN64)
  // must not exceed 32 bits for offsets!
  DP_ASSERT(!(offset & mask64<-1,0>::mask)); 
#endif
  return offset;
}

void DPBFSaveTraverser::dealloc(void * offsetPtr)
{
  m_fm->mapOut( offsetPtr );
}

uint_t DPBFSaveTraverser::pseudoAlloc(unsigned int numBytes)
{
  // always allocate a multiple of 4 bytes to keep offsets 4-byte aligned
  m_preCalculatedFileSize += ((numBytes+3)&~3);
  return 0;
}


void DPBFSaveTraverser::doApply( const NodeSharedPtr & root )
{
  NBFHeader * nbfHdr;

  m_objectOffsetMap.clear();
  m_fileOffset = 0;
  m_success = true;
  m_errorMessage = "";

  if ( !calculatingStorageRequirements() )
  {
    DP_ASSERT( m_preCalculatedFileSize < UINT_MAX );
    m_fm = new WriteMapping( m_fileName, dp::checked_cast<size_t>(m_preCalculatedFileSize) );
    if ( ! m_fm || ! m_fm->isValid() )
    {
      m_errorMessage = "Could not establish a file mapping!";
      m_success = false;
      return;
    }
    DP_ASSERT( m_fm && m_fm->isValid() );

    // NBF requires the NBFHeader at offset 0!
    // Therefore, allocate the header before traversing the tree!
    alloc( (void*&)nbfHdr, sizeof(NBFHeader) );
    DP_ASSERT(nbfHdr);
  } 
  else
  {
    pseudoAlloc(sizeof(NBFHeader));
  }

  // now walk the scene graph using the base implementation
  SharedTraverser::doApply(root);
  uint_t sceneOffs = handleScene( m_scene ); // extra scene processing
  uint_t vsOffs = m_viewState ? handleViewState( m_viewState ) : 0; // extra viewstate processing

  if ( !calculatingStorageRequirements() )
  {
    DP_ASSERT(nbfHdr);
    // write header
    // ... signature
    nbfHdr->signature[0] = '#';
    nbfHdr->signature[1] = 'N';
    nbfHdr->signature[2] = 'B';
    nbfHdr->signature[3] = 'F';

    // ... NBF version
    nbfHdr->nbfMajorVersion = DPBF_VER_MAJOR;
    nbfHdr->nbfMinorVersion = DPBF_VER_MINOR;
    nbfHdr->nbfBugfixLevel  = DPBF_VER_BUGFIX;

    // ... DP version
    nbfHdr->dpMajorVersion = (ubyte_t)DP_VER_MAJOR;
    nbfHdr->dpMinorVersion = (ubyte_t)DP_VER_MINOR;
    // DP does not provide a bugfix level

    // ... time stamp
#if defined(_WIN32)
    SYSTEMTIME sysTime;
    GetSystemTime(&sysTime);

    nbfHdr->dayLastModified     = (ubyte_t)(sysTime.wDay & 0x00FF);    
    nbfHdr->monthLastModified   = (ubyte_t)(sysTime.wMonth & 0x00FF);  
    nbfHdr->yearLastModified[0] = (ubyte_t)((sysTime.wYear>>8) & 0x00FF);
    nbfHdr->yearLastModified[1] = (ubyte_t)(sysTime.wYear & 0x00FF);
    nbfHdr->secondLastModified  = (ubyte_t)(sysTime.wSecond & 0x00FF); 
    nbfHdr->minuteLastModified  = (ubyte_t)(sysTime.wMinute & 0x00FF);
    nbfHdr->hourLastModified    = (ubyte_t)(sysTime.wHour & 0x00FF);
#elif defined(LINUX)
    time_t calenderTime = time(NULL);
    struct tm brokenDownTime;
    memcpy(&brokenDownTime, localtime(&calenderTime), sizeof(brokenDownTime));
    // need to convert year: // tm.tm_year means 'years since 1900'
    short tm_year = (short)((brokenDownTime.tm_year+1900) & 0x0000FFFF); 

    nbfHdr->dayLastModified     = (ubyte_t)(brokenDownTime.tm_mday & 0x000000FF);    
    nbfHdr->monthLastModified   = (ubyte_t)(brokenDownTime.tm_mon & 0x000000FF);
    nbfHdr->yearLastModified[0] = (ubyte_t)((tm_year>>8) & 0x00FF);
    nbfHdr->yearLastModified[1] = (ubyte_t)(tm_year & 0x00FF);
    nbfHdr->secondLastModified  = (ubyte_t)(brokenDownTime.tm_sec & 0x000000FF); 
    nbfHdr->minuteLastModified  = (ubyte_t)(brokenDownTime.tm_min & 0x000000FF);
    nbfHdr->hourLastModified    = (ubyte_t)(brokenDownTime.tm_hour & 0x000000FF);
#endif

    // ... scene offset
    DP_ASSERT(sceneOffs);
    nbfHdr->scene = sceneOffs;

    // optional viewstate
    nbfHdr->viewState = vsOffs;

    dealloc( nbfHdr );  // unmap header now
    delete m_fm;                  // delete file mapping at the end
    m_fm = NULL;
  }
}

// Scene
uint_t DPBFSaveTraverser::handleScene( SceneSharedPtr const& scene )
{
  uint_t sceneOffs = 0; // start with an invalid offset

  // traverse scene cameras
  for ( Scene::CameraIterator sci = scene->beginCameras() ; sci != scene->endCameras() ; ++sci )
  {
    traverseObject( *sci );
  }

  if ( calculatingStorageRequirements() )
  {
    sceneOffs = pseudoAlloc(sizeof(NBFScene));
    // determine storage requirements for additional attachments
    pseudoAlloc((scene->getNumberOfCameras())*sizeof(uint_t)); // camera offsets

    if ( scene->getBackImage() )
    {
      pseudoAlloc(sizeof(texImage_t)); // a Scene maintains a BackImage
      TextureHostSharedPtr const& ti = scene->getBackImage();
      if ( ! ti->getFileName().empty() )
      {
        // determine storage requirements for the image file name
        pseudoAlloc( dp::checked_cast<unsigned int>(ti->getFileName().length()+1)*sizeof(char)); // ... texture image file
      }
      else if ( 0 < ti->getNumberOfImages() )
      { // no filename specified, but images
        // determine storage requirements for raw pixel data of image 0
        pseudoAlloc( ti->getNumberOfBytes() );
      }
    }
  }
  else 
  {
    // allocate scene object and write its offset
    Offset_AutoPtr<NBFScene> scenePtr(this, sceneOffs);        

    // write the scene data

    assign(scenePtr->ambientColor, scene->getAmbientColor());
    assign(scenePtr->backColor, scene->getBackColor());

    // ... texture image        
    if ( scene->getBackImage() )
    {
      Offset_AutoPtr<texImage_t> img(this, scenePtr->backImg);         
      TextureHostSharedPtr const& texImg = scene->getBackImage();
      string file(texImg->getFileName());
      writeTexImage( file, texImg, img );
    }

    // scene cameras
    if ( scene->getNumberOfCameras() )
    {       
      // allocate slot where to write camera offsets below
      scenePtr->numCameras = scene->getNumberOfCameras();
      Offset_AutoPtr<uint_t> camOffs(this, scenePtr->cameras, scenePtr->numCameras);
       
      // now walk the cameras in scene
      unsigned int i = 0;
      for ( Scene::CameraIterator sci = scene->beginCameras() ; sci != scene->endCameras() ; ++sci, ++i )
      { // write offset
        DP_ASSERT( m_objectOffsetMap.find( sci->getWeakPtr() ) != m_objectOffsetMap.end() );
        camOffs[i] = m_objectOffsetMap[sci->getWeakPtr()];
      }
    }
    
    scenePtr->numObjectLinks = dp::checked_cast<uint_t>( m_links.size() );
    if ( m_links.size() )
    {
      // allocate slot where to write the links below
      Offset_AutoPtr<NBFLink> links(this, scenePtr->objectLinks, scenePtr->numObjectLinks);
      // walk the links now
      for ( unsigned int i=0; i<scenePtr->numObjectLinks ; ++i )
      {
        // write offset
        DP_ASSERT( m_objectOffsetMap.find( m_links[i].subject ) != m_objectOffsetMap.end() );
        DP_ASSERT( m_objectOffsetMap.find( m_links[i].observer ) != m_objectOffsetMap.end() );
        links[i].linkID = m_links[i].id;
        links[i].subject = m_objectOffsetMap[m_links[i].subject];
        links[i].observer = m_objectOffsetMap[m_links[i].observer];
      }
    }

    // root node
    if ( scene->getRootNode() )
    { // write offset to scene's root node
      DP_ASSERT(m_objectOffsetMap.find(scene->getRootNode().getWeakPtr())!=m_objectOffsetMap.end());
      scenePtr->root = m_objectOffsetMap[scene->getRootNode().getWeakPtr()];
    }
  }
  return sceneOffs;
}

uint_t DPBFSaveTraverser::handleViewState( dp::sg::ui::ViewStateSharedPtr const& viewState )
{
  uint_t vsOffs = 0;
  // append view state if provided
  if ( viewState )
  {
    if ( calculatingStorageRequirements() )
    {
      pseudoAlloc(sizeof(NBFViewState));
    }
    else
    {      
      Offset_AutoPtr<NBFViewState> viewStatePtr(this, vsOffs);       
      // write view state specific data
      // ... camera
      DP_ASSERT( !viewState->getCamera() || m_objectOffsetMap.find(viewState->getCamera().getWeakPtr())!=m_objectOffsetMap.end() );
      viewStatePtr->camera = viewState->getCamera() ? m_objectOffsetMap[viewState->getCamera().getWeakPtr()] : 0;
      // ... jitter settings
      // ... stereo settings
      viewStatePtr->isStereo = false; //viewState->getRenderTarget() ? viewState->getRenderTarget()->isStereoEnabled() : false;
      viewStatePtr->isStereoAutomatic = viewState->isStereoAutomaticEyeDistanceAdjustment();
      viewStatePtr->stereoAutomaticFactor = viewState->getStereoAutomaticEyeDistanceFactor();
      viewStatePtr->stereoEyeDistance = viewState->getStereoEyeDistance();
      viewStatePtr->targetDistance = viewState->getTargetDistance();
    }
  }
  return vsOffs;
}

// Cameras
void DPBFSaveTraverser::handleParallelCamera(const ParallelCamera *p)
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    SharedTraverser::handleParallelCamera(p);

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFParallelCamera));
      pseudoAllocFrustumCamera(p);
    }
    else
    {
      // allocate camera object and write offset of this camera object
      Offset_AutoPtr<NBFFrustumCamera> camPtr(this, m_objectOffsetMap[ph]);
      writeFrustumCamera( p, camPtr, NBF_PARALLEL_CAMERA );
    }
  }
}

void DPBFSaveTraverser::handlePerspectiveCamera(const PerspectiveCamera *p)
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    SharedTraverser::handlePerspectiveCamera(p);

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFPerspectiveCamera));
      pseudoAllocFrustumCamera(p);
    }
    else
    {
      // allocate camera object and write offset of this camera object      
      Offset_AutoPtr<NBFFrustumCamera> camPtr(this, m_objectOffsetMap[ph]);       
      writeFrustumCamera( p, camPtr, NBF_PERSPECTIVE_CAMERA );
    }
  }
}

void DPBFSaveTraverser::handleMatrixCamera(const MatrixCamera *p)
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    SharedTraverser::handleMatrixCamera(p);

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFMatrixCamera));
      pseudoAllocCamera(p); // a MatrixCamera is a Camera 
    }
    else
    {
      // allocate camera object and write offset of this camera object      
      Offset_AutoPtr<NBFMatrixCamera> camPtr(this, m_objectOffsetMap[ph]);       
      writeCamera(p, camPtr, NBF_MATRIX_CAMERA); // a MatrixCamera is a Camera 
      assign( camPtr->projection, p->getProjection() );
      assign( camPtr->inverseProjection, p->getInverseProjection() );
    }
  }
}

// Nodes
void DPBFSaveTraverser::handleBillboard(const Billboard *p)
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    // call base implementation for further traversing the tree
    SharedTraverser::handleBillboard(p);

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFBillboard));
      pseudoAllocGroup(p); // a Billboard is a Group
    }
    else
    {
      // allocate billboard object and write its offset
      Offset_AutoPtr<NBFBillboard> billboardPtr(this, m_objectOffsetMap[ph]);      
      writeGroup(p, billboardPtr, NBF_BILLBOARD); // a Billboard is a Group

      // write transform data
      assign(billboardPtr->rotationAxis, p->getRotationAxis());
      billboardPtr->alignment = p->getAlignment();
    }
  }
}

void DPBFSaveTraverser::handleEffectData( const EffectData * p )
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    SharedTraverser::handleEffectData( p );         // walk the EffectData's parameter groups invoking the base implementation

    const dp::fx::EffectSpecSharedPtr & es = p->getEffectSpec();
    std::string effectFile = dp::util::makePathRelative( dp::fx::EffectLibrary::instance()->getEffectFile( es->getName() ), m_basePaths );

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFEffectData));
      pseudoAllocObject(p); // an EffectData is an Object
      DP_ASSERT( ! es->getName().empty() );
      pseudoAlloc(2*sizeof(str_t));
      pseudoAlloc(dp::checked_cast<unsigned int>(effectFile.length()+1)*sizeof(char));
      pseudoAlloc(dp::checked_cast<unsigned int>(es->getName().length()+1)*sizeof(char));
      pseudoAlloc((es->getNumberOfParameterGroupSpecs())*sizeof(uint_t)); // ParameterGroupData offsets
    }
    else
    {
      Offset_AutoPtr<NBFEffectData> edPtr(this, m_objectOffsetMap[ph]);
      writeObject(p, edPtr, NBF_EFFECT_DATA); // an EffectData is an Object

      // allocate str_t object to hold the EffectSpec filename
      DP_ASSERT( !effectFile.empty() );
      edPtr->effectFileName.numChars = dp::checked_cast<uint_t>( effectFile.length() );
      Offset_AutoPtr<char> fileChars( this, edPtr->effectFileName.chars, edPtr->effectFileName.numChars + 1 );
      strncpy( fileChars, effectFile.c_str(), edPtr->effectFileName.numChars + 1 );

      // allocate str_t object to hold the EffectSpec name 
      DP_ASSERT( ! es->getName().empty() );
      const string & name = es->getName();
      edPtr->effectSpecName.numChars = dp::checked_cast<uint_t>( name.length() );
      Offset_AutoPtr<char> chars( this, edPtr->effectSpecName.chars, edPtr->effectSpecName.numChars + 1 );
      strncpy( chars, name.c_str(), edPtr->effectSpecName.numChars + 1 );

      // allocate space to hold the offsets of all ParameterGroupData
      Offset_AutoPtr<uint_t> pgds( this, edPtr->parameterGroupData, es->getNumberOfParameterGroupSpecs() );
      unsigned int i = 0;
      for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
      {
        const ParameterGroupDataSharedPtr & pgd = p->getParameterGroupData( it );
        if ( pgd )
        {
          DP_ASSERT( m_objectOffsetMap.find( pgd.getWeakPtr() ) != m_objectOffsetMap.end() );
          pgds[i++] = m_objectOffsetMap[pgd.getWeakPtr()];
        }
        else
        {
          pgds[i++] = 0;
        }
      }
      edPtr->transparent = p->getTransparent();
    }
  }
}

void DPBFSaveTraverser::handleParameterGroupData( const ParameterGroupData * p )
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    SharedTraverser::handleParameterGroupData( p );

    const dp::fx::ParameterGroupSpecSharedPtr & pgs = p->getParameterGroupSpec();
    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFParameterGroupData));
      pseudoAllocObject(p); // a ParameterGroupData is an Object
      DP_ASSERT( !pgs->getName().empty() );
      pseudoAlloc(sizeof(str_t));
      pseudoAlloc(dp::checked_cast<unsigned int>(pgs->getName().length()+1)*sizeof(char));
      pseudoAlloc( pgs->getDataSize() * sizeof(byte_t) );
    }
    else
    {
      Offset_AutoPtr<NBFParameterGroupData> pgdPtr(this, m_objectOffsetMap[ph]);
      writeObject(p, pgdPtr, NBF_PARAMETER_GROUP_DATA); // a ParameterGroupData is an Object

      // allocate str_t object to hold the EffectSpec name 
      DP_ASSERT( ! pgs->getName().empty() );
      const string & name = pgs->getName();
      pgdPtr->parameterGroupSpecName.numChars = dp::checked_cast<uint_t>( name.length() );
      Offset_AutoPtr<char> chars( this, pgdPtr->parameterGroupSpecName.chars, pgdPtr->parameterGroupSpecName.numChars + 1 );
      strncpy( chars, name.c_str(), pgdPtr->parameterGroupSpecName.numChars + 1 );

      pgdPtr->numData = pgs->getDataSize();

      Offset_AutoPtr<byte_t> data( this, pgdPtr->data, pgdPtr->numData );
      for ( dp::fx::ParameterGroupSpec::iterator it = pgs->beginParameterSpecs() ; it != pgs->endParameterSpecs() ; ++it )
      {
        unsigned int type = it->first.getType();
        if ( type & dp::fx::PT_SCALAR_TYPE_MASK )
        {
          memcpy( &data[it->second], p->getParameter( it ), it->first.getSizeInByte() );
        }
        else
        {
          DP_ASSERT( type & dp::fx::PT_POINTER_TYPE_MASK );
          ObjectSharedPtr obj = p->getParameter<ObjectSharedPtr>( it );
          if ( obj )
          {
            DP_ASSERT( m_objectOffsetMap.find( obj.getWeakPtr() ) != m_objectOffsetMap.end() );
            memcpy( &data[it->second], &m_objectOffsetMap[obj.getWeakPtr()], sizeof(uint_t) );
          }
          else
          {
            DP_ASSERT( *(uint_t*)&data[it->second] == 0 );
          }
        }
      }
    }
  }
}

void DPBFSaveTraverser::handleSampler( const Sampler * p )
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    SharedTraverser::handleSampler( p );

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFSampler));
      pseudoAllocObject(p); // a Sampler is an Object
      pseudoAlloc(sizeof(texImage_t));    // a Sampler maintains a Texture

      const TextureSharedPtr & texture = p->getTexture();
      if ( texture && texture.isPtrTo<TextureHost>() )
      {
        TextureHostSharedPtr const& th = p->getTexture().staticCast<TextureHost>();
        if ( ! th->getFileName().empty() )
        {
          // determine storage requirements for the image file name
          DP_ASSERT( th->getFileName().length() <= UINT_MAX );
          pseudoAlloc( dp::checked_cast<unsigned int>(th->getFileName().length()+1)*sizeof(char)); // ... texture image file
        }
        else if ( 0 < th->getNumberOfImages() )
        { // no filename specified, but images
          // determine storage requirements for raw pixel data of image 0
          pseudoAlloc( th->getNumberOfBytes() );
        }
      }
    }
    else
    {
      // allocate sampler object and write its offset
      Offset_AutoPtr<NBFSampler> samplerPtr(this, m_objectOffsetMap[ph]);
      writeObject(p, samplerPtr, NBF_SAMPLER); // a Sampler is an Object

      // ... texture image        
      const TextureSharedPtr & texture = p->getTexture();
      if ( texture && texture.isPtrTo<TextureHost>() )
      {
        Offset_AutoPtr<texImage_t> img(this, samplerPtr->texture);

        TextureHostSharedPtr const& th = p->getTexture().staticCast<TextureHost>();
        string file(th->getFileName());
        writeTexImage( file, th, img );
      }

      assign( samplerPtr->borderColor, p->getBorderColor() );
      samplerPtr->magFilter = p->getMagFilterMode();
      samplerPtr->minFilter = p->getMinFilterMode();
      samplerPtr->texWrapS = p->getWrapModeS();
      samplerPtr->texWrapT = p->getWrapModeT();
      samplerPtr->texWrapR = p->getWrapModeR();
      samplerPtr->compareMode = p->getCompareMode();
    }
  }
}

void DPBFSaveTraverser::handleGeoNode(const GeoNode *p)
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    // walk the GeoNode's geometry by invoking the base implementation
    SharedTraverser::handleGeoNode(p);

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFGeoNode));
      pseudoAllocNode(p); // a GeoNode is a Node
    }
    else
    {
      // allocate node object and write its offset
      Offset_AutoPtr<NBFGeoNode> nodePtr(this, m_objectOffsetMap[ph]);      
      writeNode(p, nodePtr, NBF_GEO_NODE); // a GeoNode is a Node

      // GeoNode specific data    
      nodePtr->materialEffect = p->getMaterialEffect() ? m_objectOffsetMap[p->getMaterialEffect().getWeakPtr()] : 0;
      nodePtr->primitive = p->getPrimitive() ? m_objectOffsetMap[p->getPrimitive().getWeakPtr()] : 0;
      nodePtr->stateSet = 0;
    }
  }
}

void DPBFSaveTraverser::handleGroup(const Group *p)
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    // call base implementation for further traversing the tree
    SharedTraverser::handleGroup(p);

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFGroup));
      pseudoAllocGroup(p); // a Group is a Group
    }
    else
    {
      // allocate group object and write its offset      
      Offset_AutoPtr<NBFGroup> groupPtr(this, m_objectOffsetMap[ph]);       
      writeGroup(p, groupPtr, NBF_GROUP); // a Group is a Group
    }
  }
}

void DPBFSaveTraverser::handleTransform(const Transform *p)
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    // call base implementation for further traversing the tree
    SharedTraverser::handleTransform(p);

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFTransform));
      pseudoAllocTransform(p);
    }
    else
    {
      // allocate transform object and write its offset      
      Offset_AutoPtr<NBFTransform> trafoPtr(this, m_objectOffsetMap[ph]);       
      writeGroup(p, trafoPtr, NBF_TRANSFORM); // a Transform is a Group

      // write transform data
      assign( trafoPtr->trafo, p->getTrafo() );
    }
  }
}

void DPBFSaveTraverser::handleLOD(const LOD *p)
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    // NOTE: the base implementation traverses only active childs of a LOD node, and Hence, 
    // we need to manually traverse all childs here.
    for ( Group::ChildrenConstIterator gcci = p->beginChildren() ; gcci != p->endChildren() ; ++gcci )
    {
      traverseObject( *gcci );
    }

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFLOD));
      pseudoAllocGroup(p); // a LOD is a Group
      // determine storage requirements for ranges
      pseudoAlloc((p->getNumberOfRanges())*sizeof(float));
    }
    else
    {
      // allocate LOD object and write its offset      
      Offset_AutoPtr<NBFLOD> lodPtr(this, m_objectOffsetMap[ph]);      
      writeGroup(p, lodPtr, NBF_LOD); // a LOD is a Group

      // LOD specific data
      assign(lodPtr->center, p->getCenter());
      // ranges      
      lodPtr->numRanges = p->getNumberOfRanges();
      if ( lodPtr->numRanges )
      {
        Offset_AutoPtr<float> ranges(this, lodPtr->ranges, lodPtr->numRanges);   
        memcpy(ranges, p->getRanges(), lodPtr->numRanges*sizeof(float));
      }
    }
  }
}

void DPBFSaveTraverser::handleSwitch(const Switch *p)
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    // NOTE: the base implementation traverses only active childs of a Switch, and Hence, 
    // we need to manually traverse all childs here.
    for ( Group::ChildrenConstIterator gcci = p->beginChildren() ; gcci != p->endChildren() ; ++gcci )
    {
      traverseObject( *gcci );
    }

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFSwitch));
      pseudoAllocGroup(p); // a Switch is a Group
      // calculate requirements for all attached switch masks
      pseudoAlloc(p->getNumberOfMasks()*sizeof(switchMask_t));
      for ( Switch::MaskIterator it = p->getFirstMaskIterator()
          ; it != p->getLastMaskIterator()
          ; it = p->getNextMaskIterator(it) )
      {
        DP_ASSERT( p->getSwitchMask(it).size() <= UINT_MAX );
        pseudoAlloc( dp::checked_cast<unsigned int>(p->getSwitchMask(it).size())*sizeof(uint_t));
      }
    }
    else
    {      
      Offset_AutoPtr<NBFSwitch> switchPtr(this, m_objectOffsetMap[ph]);
      writeGroup(p, switchPtr, NBF_SWITCH); // a Switch is a Group

      // write Switch specific ...
      switchPtr->activeMaskKey = p->getActiveMaskKey();
      
      // allocate memory for all attached switch masks
      DP_ASSERT( p->getNumberOfMasks() <= UINT_MAX );
      switchPtr->numMasks = p->getNumberOfMasks();
      DP_ASSERT(switchPtr->numMasks); // there should be at least a default mask
      Offset_AutoPtr<switchMask_t> masks(this, switchPtr->masks, switchPtr->numMasks);       

      // write all the masks
      unsigned int i = 0; // zero-based index into masks array
      for ( Switch::MaskIterator it = p->getFirstMaskIterator()
          ; it != p->getLastMaskIterator()
          ; it = p->getNextMaskIterator(it), ++i )
      {
        DP_ASSERT(i < switchPtr->numMasks); // severe error if this fires!
        const Switch::SwitchMask& mask = p->getSwitchMask(it);
        masks[i].maskKey = p->getMaskKey(it);
        masks[i].numChildren = dp::checked_cast<uint_t>(mask.size());
        masks[i].children = 0; // just give it a defined offset

        // allocate only if children are available in the mask
        if ( masks[i].numChildren ) 
        {
          // allocate and write indices referring to active children.
          Offset_AutoPtr<uint_t> children(this, masks[i].children, masks[i].numChildren);
          copy(mask.begin(), mask.end(), &children[0]);
        }
      }
    }
  }
}

void DPBFSaveTraverser::handleLightSource( const LightSource * p )
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    // call base implementation for further traversing the tree
    SharedTraverser::handleLightSource( p );

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFLightSource));
      pseudoAllocLightSource(p);
    }
    else
    {
      // allocate light object and write its offset      
      Offset_AutoPtr<NBFLightSource> lightPtr( this, m_objectOffsetMap[ph] );
      writeLightSource( p, lightPtr, NBF_LIGHT_SOURCE);
    }
  }
}

void DPBFSaveTraverser::pseudoAllocPrimitive( const Primitive * p )
{
  DP_ASSERT(calculatingStorageRequirements());

  pseudoAllocObject(p); // a Primitive is an Object
}

void DPBFSaveTraverser::writePrimitive(const Primitive * prim, NBFPrimitive * nbfPrim, uint_t objCode )
{
  DP_ASSERT(!calculatingStorageRequirements());

  writeObject(prim, nbfPrim, objCode ); // a Primitive is an Object

  nbfPrim->primitiveType    = prim->getPrimitiveType();
  if ( prim->getPrimitiveType() == PRIMITIVE_PATCHES )
  {
    nbfPrim->patchesType      = prim->getPatchesType();
    nbfPrim->patchesMode      = prim->getPatchesMode();
    nbfPrim->patchesOrdering  = prim->getPatchesOrdering();
    nbfPrim->patchesSpacing   = prim->getPatchesSpacing();
  }
  nbfPrim->elementOffset    = prim->getElementOffset();
  nbfPrim->elementCount     = prim->getElementCount();
  nbfPrim->instanceCount    = prim->getInstanceCount();
  nbfPrim->renderFlags      = prim->getRenderFlags();

  DP_ASSERT( m_objectOffsetMap.find( prim->getVertexAttributeSet().getWeakPtr() ) != m_objectOffsetMap.end() );
  nbfPrim->vertexAttributeSet = m_objectOffsetMap[prim->getVertexAttributeSet().getWeakPtr()];
  nbfPrim->indexSet = prim->getIndexSet() ? m_objectOffsetMap[prim->getIndexSet().getWeakPtr()] : 0;
}

void DPBFSaveTraverser::handlePrimitive( const Primitive *p )
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    SharedTraverser::traversePrimitive( p );

    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFPrimitive));
      pseudoAllocPrimitive(p);
    }
    else
    {
      if ( !processSharedObject(p, NBF_PRIMITIVE) )
      {
        // allocate object and write its offset
        Offset_AutoPtr<NBFPrimitive> pPtr(this, m_objectOffsetMap[ph]);        
        writePrimitive( p, pPtr, NBF_PRIMITIVE );
      }
    }
  }
}

void DPBFSaveTraverser::handleIndexSet( const IndexSet * p )
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFIndexSet));
      pseudoAllocIndexSet(p);
    }
    else
    {
      // IndexSets can share data
      if ( !processSharedObject(p, NBF_INDEX_SET) )
      {
        // allocate IndexSet object and write its offset
        Offset_AutoPtr<NBFIndexSet> isPtr(this, m_objectOffsetMap[ph]);
        writeObject( p, isPtr, NBF_INDEX_SET );

        isPtr->dataType              = p->getIndexDataType();
        isPtr->primitiveRestartIndex = p->getPrimitiveRestartIndex();
        isPtr->numberOfIndices       = p->getNumberOfIndices();

        unsigned int numBytes = static_cast<unsigned int>(dp::getSizeOf( static_cast<dp::DataType>(isPtr->dataType) ) * isPtr->numberOfIndices);
        Offset_AutoPtr<byte_t> idata( this, isPtr->idata, numBytes );
        Buffer::DataReadLock reader( p->getBuffer() );
        memcpy( idata, reader.getPtr(), numBytes );
      }
    }
  }
}

void DPBFSaveTraverser::handleVertexAttributeSet( const VertexAttributeSet *p )
{
  ObjectWeakPtr ph = getWeakPtr<Object>(p);
  if ( m_objectOffsetMap.find(ph) == m_objectOffsetMap.end() )
  {
    if ( calculatingStorageRequirements() )
    {
      m_objectOffsetMap[ph] = pseudoAlloc(sizeof(NBFVertexAttributeSet));
      pseudoAllocVertexAttributeSet(p);
    }
    else
    {
      // VertexAttributeSets can share data
      if ( !processSharedObject(p, NBF_VERTEX_ATTRIBUTE_SET) )
      {
        // allocate VertexAttributeSet object and write its offset      
        Offset_AutoPtr<NBFVertexAttributeSet> vasPtr(this, m_objectOffsetMap[ph]);       
        writeVertexAttributeSet(p, vasPtr, NBF_VERTEX_ATTRIBUTE_SET); // a VertexAttributeSet is a VertexAttributeSet
      }
    }
  }
}

bool DPBFSaveTraverser::processSharedObject(const Object * obj, uint_t objCode)
{
  DP_ASSERT(!calculatingStorageRequirements());

  if (  obj->isDataShared() 
     && (m_objectDataIDOffsetMap.find(obj->getDataID()) != m_objectDataIDOffsetMap.end())
     )
  { // current object is shared _AND_ the corresponding source object is already registered

    // independent of the object type, it is sufficient to write general object data only,
    // and just link to the corresponding source object
    Offset_AutoPtr<NBFObject> objPtr(this, m_objectOffsetMap[getWeakPtr<Object>(obj)]);

    // write object data
    writeObject(obj, objPtr, objCode);
    objPtr->sourceObject = m_objectDataIDOffsetMap[objPtr->objectDataID];
    
    return true; // finally processed
  }
  
  // assign not finally processed
  return false;
}

void DPBFSaveTraverser::writeObject(const Object* objPtr, NBFObject * nbfObjPtr, uint_t objCode)
{
  nbfObjPtr->objectCode = objCode;
  nbfObjPtr->isShared = objPtr->isDataShared();

  nbfObjPtr->hints = objPtr->getHints();

  nbfObjPtr->objectDataID = objPtr->getDataID();

  if ( !objPtr->getName().empty() )
  {
    // allocate str_t object to hold the name 
    //
    const string& name = objPtr->getName();
    Offset_AutoPtr<str_t> nameStr(this, nbfObjPtr->objectName);
    // allocate memory for the actual string
    nameStr->numChars = dp::checked_cast<uint_t>(name.length()); // without terminating 0!
    Offset_AutoPtr<char> chars(this, nameStr->chars, nameStr->numChars+1);    
    strncpy(chars, name.c_str(), nameStr->numChars+1); // copy string, including terminating 0!
  }

  if ( !objPtr->getAnnotation().empty() )
  {
    // allocate str_t object to hold the annotation
    //
    const string& anno = objPtr->getAnnotation();
    Offset_AutoPtr<str_t> annoStr(this, nbfObjPtr->objectAnno);
    // allocate memory for the actual string
    annoStr->numChars = dp::checked_cast<uint_t>(anno.length()); // without terminating 0!
    Offset_AutoPtr<char> chars(this, annoStr->chars, annoStr->numChars+1);    
    strncpy(chars, anno.c_str(), annoStr->numChars+1); // copy string, including terminating 0!
  }

}

void DPBFSaveTraverser::writeNode(const Node * nodePtr, NBFNode * nbfNodePtr, uint_t objCode)
{
  DP_ASSERT(!calculatingStorageRequirements());
  writeObject(nodePtr, nbfNodePtr, objCode); // a Node is an Object
}

void DPBFSaveTraverser::writeGroup(const Group * grpPtr, NBFGroup * nbfGrpPtr, uint_t objCode)
{
  DP_ASSERT(!calculatingStorageRequirements());
  writeNode(grpPtr, nbfGrpPtr, objCode); // a Group is a Node

  // allocate slot where to write offsets to children below
  nbfGrpPtr->numChildren = grpPtr->getNumberOfChildren();
  Offset_AutoPtr<uint_t> childOffs(this, nbfGrpPtr->children, nbfGrpPtr->numChildren);
   
  // write the offsets now
  unsigned int i=0;
  for ( Group::ChildrenConstIterator gcci = grpPtr->beginChildren() ; gcci != grpPtr->endChildren() ; ++gcci, ++i )
  {
    DP_ASSERT( m_objectOffsetMap.find( gcci->getWeakPtr() ) != m_objectOffsetMap.end() );
    childOffs[i] = m_objectOffsetMap[gcci->getWeakPtr()];
  }

  // allocate slot where to write the clip planes, and write them
  nbfGrpPtr->numClipPlanes = grpPtr->getNumberOfClipPlanes();
  Offset_AutoPtr<plane_t> planes(this, nbfGrpPtr->clipPlanes, nbfGrpPtr->numClipPlanes);
  i = 0;
  for ( Group::ClipPlaneConstIterator gcpci = grpPtr->beginClipPlanes() ; gcpci != grpPtr->endClipPlanes() ; ++gcpci, ++i )
  {
    ClipPlaneSharedPtr const& plane = *gcpci;
    planes[i].active = plane->isEnabled();
    assign(planes[i].normal, plane->getNormal());
    planes[i].offset = plane->getOffset();
  }

}

void DPBFSaveTraverser::writeVertexAttributeSet(const VertexAttributeSet * vasPtr, NBFVertexAttributeSet * nbfVASPtr, uint_t objCode)
{
  DP_ASSERT(!calculatingStorageRequirements());

  writeObject(vasPtr, nbfVASPtr, objCode); // a VertexAttributeSet is an Object

  for ( unsigned int i=0; i<VertexAttributeSet::DP_SG_VERTEX_ATTRIB_COUNT; ++i )
  {
    if ( vasPtr->getSizeOfVertexData(i) )
    {
      nbfVASPtr->enableFlags |= ((!!vasPtr->isEnabled(i+16))<<(i+16)) | ((!!vasPtr->isEnabled(i))<<i);
      nbfVASPtr->normalizeEnableFlags |= ((!!vasPtr->isNormalizeEnabled(i+16))<<(i+16));
      nbfVASPtr->vattribs[i].size = vasPtr->getSizeOfVertexData(i);
      nbfVASPtr->vattribs[i].type = vasPtr->getTypeOfVertexData(i);
      nbfVASPtr->vattribs[i].numVData = vasPtr->getNumberOfVertexData(i);
      uint_t sizeOfVertex = static_cast<unsigned int>(nbfVASPtr->vattribs[i].size * dp::getSizeOf(static_cast<dp::DataType>(nbfVASPtr->vattribs[i].type) ));
      unsigned int numBytes = nbfVASPtr->vattribs[i].numVData * sizeOfVertex;
      Offset_AutoPtr<byte_t> vdata(this, nbfVASPtr->vattribs[i].vdata, numBytes);

      // copy strided vertex data
      byte_t *itDst = vdata;
      Buffer::ConstIterator<char>::Type itSrc = vasPtr->getVertexData<char>(i);
      size_t vdc = vasPtr->getNumberOfVertexData(i);
      for ( size_t index = 0;index < vdc; ++index )
      {
        memcpy( itDst, &itSrc[0], sizeOfVertex );
        ++itSrc;
        itDst += sizeOfVertex;
      }

    }
  }
}

void DPBFSaveTraverser::writeLightSource(const LightSource* lightSrcPtr, NBFLightSource * nbfLightSrcPtr, uint_t objCode)
{
  DP_ASSERT(!calculatingStorageRequirements());
  writeNode(lightSrcPtr, nbfLightSrcPtr, objCode);    // a LightSource is a Node
  nbfLightSrcPtr->castShadow = lightSrcPtr->isShadowCasting();
  nbfLightSrcPtr->enabled = lightSrcPtr->isEnabled();
  // write animation offset only if animation is available
  nbfLightSrcPtr->animation = 0;
  if ( lightSrcPtr->getLightEffect() )
  {
    DP_ASSERT(m_objectOffsetMap.find(lightSrcPtr->getLightEffect().getWeakPtr())!=m_objectOffsetMap.end());
    nbfLightSrcPtr->lightEffect = m_objectOffsetMap[lightSrcPtr->getLightEffect().getWeakPtr()];
  }
}

void DPBFSaveTraverser::writeFrustumCamera( const FrustumCamera * camPtr, NBFFrustumCamera * nbfCamPtr, uint_t objCode )
{
  DP_ASSERT(!calculatingStorageRequirements());
  writeCamera( camPtr, nbfCamPtr, objCode );

  nbfCamPtr->farDist = camPtr->getFarDistance();
  nbfCamPtr->nearDist = camPtr->getNearDistance();
  assign(nbfCamPtr->windowOffset, camPtr->getWindowOffset());
  assign(nbfCamPtr->windowSize, camPtr->getWindowSize());
}

void DPBFSaveTraverser::writeCamera(const Camera * camPtr, NBFCamera * nbfCamPtr, uint_t objCode)
{
  DP_ASSERT(!calculatingStorageRequirements());
  writeObject(camPtr, nbfCamPtr, objCode); // a Camera is an Object

  // camera data
  assign(nbfCamPtr->upVector, camPtr->getUpVector());
  assign(nbfCamPtr->position, camPtr->getPosition());
  assign(nbfCamPtr->direction, camPtr->getDirection());
  nbfCamPtr->focusDist = camPtr->getFocusDistance();
  
  if ( camPtr->getNumberOfHeadLights() )
  {    
    // allocate slot where to write headlight offsets below
    nbfCamPtr->numHeadLights = camPtr->getNumberOfHeadLights();
    Offset_AutoPtr<uint_t> lightOffs(this, nbfCamPtr->headLights, nbfCamPtr->numHeadLights);     
    // walk the headlights and write offset for each
    unsigned int i=0;
    for ( Camera::HeadLightConstIterator hlci = camPtr->beginHeadLights() ; hlci != camPtr->endHeadLights() ; ++hlci, ++i )
    {
      DP_ASSERT(m_objectOffsetMap.find(hlci->getWeakPtr())!=m_objectOffsetMap.end());
      lightOffs[i] = m_objectOffsetMap[hlci->getWeakPtr()];
    }
  }
}

template<typename DPFaceType, typename NBFFaceType>
void DPBFSaveTraverser::writeIndices(const DPFaceType * faces, NBFFaceType* nbfFaces)
{
  DP_ASSERT(!calculatingStorageRequirements());
  DP_ASSERT(faces->hasIndices());  
  // allocate space to write face data to
  nbfFaces->numIndices = faces->getNumberOfIndices();
  Offset_AutoPtr<uint_t> indices(this, nbfFaces->indices, nbfFaces->numIndices);   
  // copy face data
  memcpy( indices, &faces->getIndices()[0], nbfFaces->numIndices * sizeof(uint_t) );
}

void DPBFSaveTraverser::writeTexImage( const string& file, TextureHostSharedPtr const& img, texImage_t * nbfImg)
{
  DP_ASSERT(nbfImg);
  DP_ASSERT(!calculatingStorageRequirements());

  if ( !file.empty() )
  {    
    nbfImg->file.numChars = dp::checked_cast<uint_t>(file.length());
    Offset_AutoPtr<char> chars(this, nbfImg->file.chars, nbfImg->file.numChars+1);
    strcpy(chars, file.c_str());
  }

  if ( img )
  {
    nbfImg->flags = img->getCreationFlags();
    nbfImg->target = (uint_t)(img->getTextureTarget());
    
    // NOTE: write dimension/format specs and raw pixels only if no filename was provided
    // BUT: never store image streams!
    if ( file.empty() && !img->isImageStream() )
    {
      // no filename specified, write raw pixel data instead 
      nbfImg->width = img->getWidth();
      nbfImg->height = img->getHeight();
      nbfImg->depth  = img->getDepth();
      nbfImg->pixelFormat = img->getFormat();
      nbfImg->dataType = img->getType();
      // write raw pixels to file
      uint_t nbytes = img->getNumberOfBytes();

      if ( img->getPixels() )
      {
        Offset_AutoPtr<ubyte_t> pixels(this, nbfImg->pixels, nbytes);
         
        Buffer::DataReadLock buffer(img->getPixels());
        memcpy(pixels, buffer.getPtr(), nbytes);
      }
    }
  }
}

void DPBFSaveTraverser::pseudoAllocIndexSet( const IndexSet * p )
{
  DP_ASSERT(calculatingStorageRequirements());
  pseudoAllocObject( p );
  pseudoAlloc( static_cast<unsigned int>(dp::getSizeOf( p->getIndexDataType() ) * p->getNumberOfIndices()) );
}

void DPBFSaveTraverser::pseudoAllocObject(const Object* p)
{
  DP_ASSERT(calculatingStorageRequirements());

  // strings will be truncated if exceeding 32-bit limit
  //
  DP_ASSERT( p->getName().length() <= UINT_MAX );
  DP_ASSERT( p->getAnnotation().length() <= UINT_MAX );
  
  if ( !p->getName().empty() )
  {
    pseudoAlloc(sizeof(str_t));
    pseudoAlloc(dp::checked_cast<unsigned int>(p->getName().length()+1)*sizeof(char));
  }
  if ( !p->getAnnotation().empty() )
  {
    pseudoAlloc(sizeof(str_t));
    pseudoAlloc(dp::checked_cast<unsigned int>(p->getAnnotation().length()+1)*sizeof(char));
  }
}

void DPBFSaveTraverser::pseudoAllocNode(const Node* p)
{
  DP_ASSERT(calculatingStorageRequirements());
  pseudoAllocObject(p); // a Node is an Object
}

void DPBFSaveTraverser::pseudoAllocGroup(const Group * p)
{
  DP_ASSERT(calculatingStorageRequirements());
  pseudoAllocNode(p); // a Group is a Node 
  pseudoAlloc(  p->getNumberOfChildren()   * sizeof(uint_t)
              + p->getNumberOfClipPlanes() * sizeof(plane_t) );
}

void DPBFSaveTraverser::pseudoAllocLightSource(const LightSource * p)
{
  DP_ASSERT(calculatingStorageRequirements());
  pseudoAllocObject(p); // a LightSource is an Object
}

void DPBFSaveTraverser::pseudoAllocFrustumCamera( const FrustumCamera * p )
{
  DP_ASSERT(calculatingStorageRequirements());
  pseudoAllocCamera(p); // a FrustumCamera is a Camera
}

void DPBFSaveTraverser::pseudoAllocCamera(const Camera * p)
{
  DP_ASSERT(calculatingStorageRequirements());
  pseudoAllocObject(p); // a Camera is an Object
  pseudoAlloc((p->getNumberOfHeadLights())*sizeof(uint_t));
}

void DPBFSaveTraverser::pseudoAllocTransform( const Transform * p )
{
  DP_ASSERT( calculatingStorageRequirements() );
  pseudoAllocGroup(p);  // a Transform is a Group
}

void DPBFSaveTraverser::pseudoAllocTexImage(const string& file, const TextureHost * img)
{
  DP_ASSERT(calculatingStorageRequirements());
  pseudoAlloc(sizeof(texImage_t));
  DP_ASSERT( file.length() <= UINT_MAX );
  pseudoAlloc( dp::checked_cast<unsigned int>(file.length()) + !file.empty());
  if ( file.empty() && img!=NULL )
  {
    pseudoAlloc( img->getNumberOfBytes( ) );
  }
}

void DPBFSaveTraverser::pseudoAllocVertexAttributeSet( const VertexAttributeSet * vas )
{
  pseudoAllocObject(vas);   // a VertexAttributeSet is an Object
  for ( unsigned int i=0; i<VertexAttributeSet::DP_SG_VERTEX_ATTRIB_COUNT; ++i )
  {
    uint_t sizeofVertex = static_cast<uint_t>(vas->getSizeOfVertexData(i) * dp::getSizeOf(vas->getTypeOfVertexData(i)));
    pseudoAlloc(vas->getNumberOfVertexData(i)*sizeofVertex);
  }
}
