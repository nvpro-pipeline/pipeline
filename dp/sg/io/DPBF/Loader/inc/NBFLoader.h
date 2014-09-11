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


#pragma once

#include <dp/util/FileMapping.h>
#include <dp/util/RCObject.h>
#include <dp/sg/io/PlugInterface.h>
#include <NBF.h> // nbf structs 
#include <map>
#include <vector>
#include <string>

// storage-class defines 
#if defined(_WIN32)
# ifdef NBFLOADER_EXPORTS
#  define NBFLOADER_API __declspec(dllexport)
# else
#  define NBFLOADER_API __declspec(dllimport)
# endif
#else
#  define NBFLOADER_API
#endif

#if defined(LINUX)
void lib_init() __attribute__ ((constructor));   // will be called before dlopen() returns
#endif

// exports required for a scene loader plug-in
extern "C"
{
NBFLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugIn *& pi);
NBFLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

struct NBFCamera_nbf_44;
struct NBFGroup_nbf_12;
struct NBFGroup_nbf_11;
struct NBFLightSource_nbf_53;
struct NBFLightSource_nbf_52;
struct NBFLightSource_nbf_50;
struct NBFLightSource_nbf_12;
struct NBFPrimitive_nbf_4d;

//! A Scene Loader for nbf files.
class NBFLoader : public dp::sg::io::SceneLoader
{
public:
  NBFLoader();

  //! Realization of the pure virtual interface function of a PlugIn.
  /** \note Never call \c delete on a PlugIn, always use the member function. */
  void deleteThis( void );

  //! Realization of the pure virtual interface function of a SceneLoader.
  /** Loads a nvb file given by \a filename. It looks for this file and possibly referenced other files like
    * textures or effects at the given path first, then at the current location and finally it searches
    * through the \a searchPaths.
    * \returns  A pointer to the loaded scene. */
  dp::sg::core::SceneSharedPtr load( 
    const std::string& filename            //!<  file to load
  , const std::vector<std::string> &searchPaths //!<  paths to search through
  , dp::sg::ui::ViewStateSharedPtr & viewState     /*!< If the function succeeded, this points to the optional
                                                ViewState stored with the scene. */
  );

protected:
  //! Protected destructor to prevent explicit creation on stack.
  virtual  ~NBFLoader(void);

  //! Load a custom object identified by \a objectCode from the file offset specified by \a offset.
  /** This function is called from the loader's framework if a custom object was detected
    * for the object stored at the particular file offset. 
    *
    * A custom implementation should first evaluate the passed object code. 
    * To map identified objects into memory, a custom implementation should call the member function
    * \link NBFLoader::mapOffset mapOffset \endlink with the file offset and the correct byte size
    * for the identified object as parameters. After that, the corresponding SceniX object can be initialized 
    * from the mapped data. 
    * \returns An ObjectSharedPtr specifying the loaded SceniX object. 
    * \note A custom implementation must not fall back on to the base implementation, as this simply returns
    * a null pointer. */
  virtual dp::sg::core::ObjectSharedPtr loadCustomObject(
    uint_t objectCode //!< NBF object code identifying the custom object. 
  , uint_t offset     //!< Specifies the file offset for the custom object.
  );

  //! Maps \a numBytes bytes at file offset \a offset into process memory. 
  /** This function turns a given offset into a pointer and ensures that a minimum of \a numBytes bytes are mapped.
  * \returns A pointer to the mapped memory. */
  ubyte_t * mapOffset(
    uint_t offset     //!< File offset of the memory block to map.
    , unsigned int numBytes   //!< Amount of bytes to map into process memory.
    );

  //! Unmaps the memory that previously was mapped through mapOffset from the process' address space.
  /** The function accepts a pointer to the mapped file offset that previously was received by a call to mapOffset. */
  void unmapOffset(
    ubyte_t * offsetPtr //!< Address where the file offset was mapped.
    );

private:

  //! An auxiliary helper template class which provides exception safe mapping and unmapping of file offsets.
  /** The purpose of this template class is to turn a mapped offset into an exception safe auto object, 
  * that is - the mapped offset automatically gets unmapped if the object runs out of scope. */
  template<typename T>
  class Offset_AutoPtr
  {
    public:
      //! Maps the specified file offset into process memory.
      /** This constructor is called on instantiation. 
      * It maps \a count objects of type T at file offset \a offset into process memory. */
      Offset_AutoPtr( dp::util::ReadMapping * fm, const dp::util::PlugInCallback * pic, uint_t offset
                    , unsigned int count=1 );

      //! Unmaps the bytes, that have been mapped at instantiation, from process memory. 
      ~Offset_AutoPtr();

      //! Provides pointer-like access to the dumb pointer. 
      T* operator->() const;

      //! De-references the dumb pointer. 
      T& operator*() const;

      //! Implicit conversion to const T*. 
      operator const T*() const;

      //! Resets the object to map another file offset
      /** The function first unmaps previously mapped bytes and after that
      * maps \a count T objects at file offset \a offset into process memory. */
      void reset( uint_t offset, unsigned int count=1 );

    private:
      T * m_ptr;
      const dp::util::PlugInCallback  * m_pic;
      dp::util::ReadMapping           * m_fm;
  };


  dp::util::ReadMapping * m_fm;

  // assign an object to an offset 
  void mapObject(uint_t offset, const dp::sg::core::ObjectSharedPtr & object );
  void remapObject(uint_t offset, const dp::sg::core::ObjectSharedPtr & object );

  // cameras
  dp::sg::core::CameraSharedPtr loadCamera(uint_t offset);
  dp::sg::core::CameraSharedPtr loadCamera_nbf_4c(uint_t offset);
  dp::sg::core::CameraSharedPtr loadCamera_nbf_44(uint_t offset);
  dp::sg::core::MatrixCameraSharedPtr loadMatrixCamera(uint_t offset);
  dp::sg::core::ParallelCameraSharedPtr loadParallelCamera(uint_t offset);
  dp::sg::core::ParallelCameraSharedPtr loadParallelCamera_nbf_4c(uint_t offset);
  dp::sg::core::ParallelCameraSharedPtr loadParallelCamera_nbf_44(uint_t offset);
  dp::sg::core::PerspectiveCameraSharedPtr loadPerspectiveCamera(uint_t offset);
  dp::sg::core::PerspectiveCameraSharedPtr loadPerspectiveCamera_nbf_4c(uint_t offset);
  dp::sg::core::PerspectiveCameraSharedPtr loadPerspectiveCamera_nbf_44(uint_t offset);
  // drawables
  dp::sg::core::PrimitiveSharedPtr loadAnyPrimitive(uint_t offset);
  dp::sg::core::PrimitiveSharedPtr loadSkinnedTriangles(uint_t offset);
  dp::sg::core::PrimitiveSharedPtr loadAnimatedIndependents_nbf_3a( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadIndependents( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadMeshes(uint_t offset);
  dp::sg::core::PrimitiveSharedPtr loadStrips(uint_t offset);
  dp::sg::core::PrimitiveSharedPtr loadPatches_nbf_47( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadQuadPatches_nbf_4d( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadQuadPatches4x4_nbf_4d( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadRectPatches_nbf_4d( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadTriPatches_nbf_4d( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadTriPatches4_nbf_4d( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadQuadPatches_nbf_47( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadQuadPatches4x4_nbf_47( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadRectPatches_nbf_47( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadTriPatches_nbf_47( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadTriPatches4_nbf_47( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadQuadPatches( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadQuadPatches4x4( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadRectPatches( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadTriPatches( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadTriPatches4( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadPrimitive( uint_t offset );
  dp::sg::core::PrimitiveSharedPtr loadPrimitive_nbf_4d( uint_t offset );
  // scene
  dp::sg::core::SceneSharedPtr loadScene(uint_t offset);
  dp::sg::core::SceneSharedPtr loadScene_nbf_b(uint_t offset);
  dp::sg::core::SceneSharedPtr loadScene_nbf_31(uint_t offset);
  dp::sg::core::SceneSharedPtr loadScene_nbf_37(uint_t offset);
  dp::sg::core::SceneSharedPtr loadScene_nbf_3e(uint_t offset);
  dp::sg::core::SceneSharedPtr loadScene_nbf_41(uint_t offset);
  // nodes
  dp::sg::core::NodeSharedPtr loadNode(uint_t offset);
  dp::sg::core::ObjectSharedPtr loadNode_nbf_12(uint_t offset);
  dp::sg::core::NodeSharedPtr loadGeoNode(uint_t offset);
  dp::sg::core::NodeSharedPtr loadGeoNode_nbf_51(uint_t offset);
  dp::sg::core::NodeSharedPtr loadGeoNode_nbf_4e(uint_t offset);
  dp::sg::core::GroupSharedPtr loadGroup(uint_t offset);
  dp::sg::core::GroupSharedPtr loadGroup_nbf_12(uint_t offset);
  dp::sg::core::GroupSharedPtr loadGroup_nbf_11(uint_t offset);
  dp::sg::core::BillboardSharedPtr loadBillboard(uint_t offset);
  dp::sg::core::BillboardSharedPtr loadBillboard_nbf_12(uint_t offset);
  dp::sg::core::BillboardSharedPtr loadBillboard_nbf_11(uint_t offset);
  dp::sg::core::SwitchSharedPtr loadFlipbookAnimation(uint_t offset);
  dp::sg::core::TransformSharedPtr loadTransform(uint_t offset);
  dp::sg::core::TransformSharedPtr loadTransform_nbf_12(uint_t offset);
  dp::sg::core::TransformSharedPtr loadTransform_nbf_11(uint_t offset);
  dp::sg::core::TransformSharedPtr loadTransform_nbf_f(uint_t offset);
  dp::sg::core::TransformSharedPtr loadAnimatedTransform_nbf_54(uint_t offset);
  dp::sg::core::TransformSharedPtr loadAnimatedTransform_nbf_12(uint_t offset);
  dp::sg::core::TransformSharedPtr loadAnimatedTransform_nbf_11(uint_t offset);
  dp::sg::core::TransformSharedPtr loadAnimatedTransform_nbf_f(uint_t offset);
  dp::sg::core::LODSharedPtr loadLOD(uint_t offset);
  dp::sg::core::LODSharedPtr loadLOD_nbf_12(uint_t offset);
  dp::sg::core::LODSharedPtr loadLOD_nbf_11(uint_t offset);
  dp::sg::core::SwitchSharedPtr loadSwitch(uint_t offset);
  dp::sg::core::SwitchSharedPtr loadSwitch_nbf_30(uint_t offset);
  dp::sg::core::SwitchSharedPtr loadSwitch_nbf_12(uint_t offset);
  dp::sg::core::SwitchSharedPtr loadSwitch_nbf_11(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadLightSource(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadLightSource_nbf_53(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadLightSource_nbf_52(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadLightSource_nbf_50(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadLightSource_nbf_12(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadDirectedLight_nbf_53(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadDirectedLight_nbf_52(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadDirectedLight_nbf_50(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadDirectedLight_nbf_12(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadPointLight_nbf_53(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadPointLight_nbf_52(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadPointLight_nbf_50(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadPointLight_nbf_12(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadSpotLight_nbf_53(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadSpotLight_nbf_50(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadSpotLight_nbf_52(uint_t offset);
  dp::sg::core::LightSourceSharedPtr loadSpotLight_nbf_12(uint_t offset);
  // state attribs
  void loadStateAttribute_nbf_54(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadAlphaTestAttribute_nbf_54(uint_t offset);
  dp::sg::core::EffectDataSharedPtr loadBlendAttribute_nbf_54(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadFaceAttribute_nbf_54(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadFaceAttribute_nbf_b(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadLightingAttribute_nbf_54(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadLineAttribute_nbf_54(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadMaterial_nbf_54(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadMaterial_nbf_40(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadMaterial_nbf_3f(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadMaterial_nbf_a(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadPointAttribute_nbf_54(uint_t offset);
  dp::sg::core::EffectDataSharedPtr loadTextureAttribute_nbf_54(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadTextureAttributeItem_nbf_54(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadTextureAttributeItem_nbf_4b(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadTextureAttributeItem_nbf_36(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadTextureAttributeItem_nbf_20(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadTextureAttributeItem_nbf_12(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadTextureAttributeItem_nbf_f(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadTextureAttributeItem_nbf_e(uint_t offset);
  dp::sg::core::ParameterGroupDataSharedPtr loadUnlitColorAttribute_nbf_54(uint_t offset);

  dp::sg::core::SamplerSharedPtr loadSampler( uint_t offset );
  dp::sg::core::SamplerSharedPtr loadSampler_nbf_54( uint_t offset );

  dp::sg::core::TextureHostSharedPtr loadTextureHost(uint_t offset, std::string& file);
  dp::sg::core::TextureHostSharedPtr loadTextureHost_nbf_4b(uint_t offset, std::string& file);

  // state set
  void loadStateSet_nbf_54(uint_t offset);
  void loadStateSet_nbf_4f(uint_t offset);
  void loadStateSet_nbf_10(uint_t offset);
  void loadStateVariant_nbf_4f(uint_t offset);
  // view state
  dp::sg::ui::ViewStateSharedPtr loadViewState(uint_t offset);
  dp::sg::ui::ViewStateSharedPtr loadViewState_nbf_4c(uint_t offset);
  dp::sg::ui::ViewStateSharedPtr loadViewState_nbf_39(uint_t offset);

  dp::sg::core::IndexSetSharedPtr loadIndexSet( uint_t offset );

  // vertex attribute set
  dp::sg::core::VertexAttributeSetSharedPtr loadVertexAttributeSet( uint_t vasOffset);
  dp::sg::core::VertexAttributeSetSharedPtr loadVertexAttributeSet_nbf_54( uint_t vasOffset);
  dp::sg::core::VertexAttributeSetSharedPtr loadVertexAttributeSet_nbf_3a( uint_t vasOffset);
  dp::sg::core::VertexAttributeSetSharedPtr loadVertexAttributeSet_nbf_38( uint_t vasOffset);

  dp::sg::core::EffectDataSharedPtr loadEffectData( uint_t offset );
  dp::sg::core::EffectDataSharedPtr loadEffectData_nbf_55( uint_t offset );
  dp::sg::core::ParameterGroupDataSharedPtr loadParameterGroupData( uint_t offset );

  // shared object handling
  template <typename ObjectType, typename NBFObjectType>
  bool loadSharedObject( typename dp::sg::core::ObjectTraits<ObjectType>::SharedPtr & obj
                       , Offset_AutoPtr<NBFObjectType>& objPtr
                       , dp::sg::core::PrimitiveType pt = dp::sg::core::PRIMITIVE_UNINITIALIZED );

  // read in non-concrete objects
  void readObject(dp::sg::core::ObjectSharedPtr const& dst, const NBFObject * src);
  void readGroup(dp::sg::core::GroupSharedPtr const& dst, const NBFGroup * src);
  void readGroup_nbf_12(dp::sg::core::GroupSharedPtr const& dst, const NBFGroup_nbf_12 * src);
  void readGroup_nbf_11(dp::sg::core::GroupSharedPtr const& dst, const NBFGroup_nbf_11 * src);
  void readPrimitiveSet( dp::sg::core::PrimitiveSharedPtr const& dst, const NBFPrimitiveSet * src );
  void readPrimitive(dp::sg::core::PrimitiveSharedPtr const& dst, const NBFPrimitive * src);
  void readPrimitive_nbf_4d(dp::sg::core::PrimitiveSharedPtr const& dst, const NBFPrimitive_nbf_4d * src);
  void readIndependentPrimitiveSet( dp::sg::core::PrimitiveSharedPtr const& dst, const NBFIndependentPrimitiveSet *src );
  void readLightSource_nbf_53(dp::sg::core::LightSourceSharedPtr const& dst, const NBFLightSource_nbf_53 * src);
  void readLightSource_nbf_52(dp::sg::core::LightSourceSharedPtr const& dst, const NBFLightSource_nbf_52 * src);
  void readLightSource_nbf_50(dp::sg::core::LightSourceSharedPtr const& dst, const NBFLightSource_nbf_50 * src);
  void readLightSource_nbf_12(dp::sg::core::LightSourceSharedPtr const& dst, const NBFLightSource_nbf_12 * src);
  void readNode(dp::sg::core::NodeSharedPtr const& dst, const NBFNode * src);
  void readCamera(dp::sg::core::CameraSharedPtr const& dst, const NBFCamera * src);
  void readCamera_nbf_44(dp::sg::core::FrustumCameraSharedPtr const& dst, const NBFCamera_nbf_44 * src);
  void readFrustumCamera(dp::sg::core::FrustumCameraSharedPtr const& dst, uint_t offset);
  void readFrustumCamera_nbf_4c(dp::sg::core::FrustumCameraSharedPtr const& dst, uint_t offset);
  void readVertexAttributeSet( dp::sg::core::VertexAttributeSetSharedPtr const& dst, const NBFVertexAttributeSet * src );
  // used in loadScene*
  // these needs to be declared as template functions, because they need to handle different NBFScene types
  template <typename NBFSceneType> void readSceneCameras(dp::sg::core::SceneSharedPtr const& nvsgScene, const Offset_AutoPtr<NBFSceneType>& nbfScene);
  template <typename NBFSceneType> void readObjectLinks( dp::sg::core::SceneSharedPtr const& nvsgScene, const Offset_AutoPtr<NBFSceneType> & nbfScene );
  template <typename NBFSceneType> void readSceneRootNode(dp::sg::core::SceneSharedPtr const& nvsgScene, const Offset_AutoPtr<NBFSceneType>& nbfScene);
  // helpers while loading TextureAttributeItem
  template <typename NBFTAIType> dp::sg::core::ParameterGroupDataSharedPtr readTexAttribItem_nbf_54(Offset_AutoPtr<NBFTAIType>& nbfTAI);
  dp::sg::core::TextureSharedPtr readTexture( uint_t offset );
  template <typename NBFTAIType> dp::sg::core::TextureSharedPtr readTexture( const Offset_AutoPtr<NBFTAIType> & src );
  template <typename NBFTAIType> void readTexEnv( dp::sg::core::ParameterGroupDataSharedPtr const& dst, const Offset_AutoPtr<NBFTAIType> & src );
  template <typename NBFTAIType> void readTexGenMode( dp::sg::core::ParameterGroupDataSharedPtr const& dst, const Offset_AutoPtr<NBFTAIType> & src );
  template <typename NBFTAIType> void readTexMatrix( dp::sg::core::ParameterGroupDataSharedPtr const& dst, const Offset_AutoPtr<NBFTAIType> & src );

  // map a string object (helper)
  std::string mapString(const str_t& str);
  std::string mapString(const sstr_t& str); // overload for small strings

  dp::sg::core::ParameterGroupDataSharedPtr getMaterialParameterGroup( const std::string & name );
  dp::sg::core::EffectDataSharedPtr getMaterialEffect();

private:
  dp::util::DataType convertDataType( unsigned int dataType );

  std::vector<std::string>        m_searchPaths;  // a private copy of the search pathes given to us via the load API
  std::map<uint_t, dp::sg::core::ObjectSharedPtr> m_offsetObjectMap; // mapping offsets to SceniX objects
  std::map<dp::sg::core::DataID, dp::sg::core::ObjectSharedPtr> m_sharedObjectsMap; // lookup shared objects given the corresponding objectID

  // private copy of the nbf version used to save the file
  ubyte_t m_nbfMajor;   // major version
  ubyte_t m_nbfMinor;   // minor version
  ubyte_t m_nbfBugfix;  // bugfix level

  bool m_autoClipPlanes_nbf_4c;   // used for carrying auto clip plane state from older cameras to current ViewState

  // call some load routines via function pointers to preserve downward compatibility
  dp::sg::core::TransformSharedPtr (NBFLoader::*m_pfnLoadAnimatedTransform)(uint_t); // ... load AnimatedTransform
  dp::sg::core::BillboardSharedPtr  (NBFLoader::*m_pfnLoadBillboard)(uint_t);     // ... load Billboard
  dp::sg::core::CameraSharedPtr     (NBFLoader::*m_pfnLoadCamera)(uint_t);        // ... load Camera
  dp::sg::core::EffectDataSharedPtr (NBFLoader::*m_pfnLoadEffectData)(uint_t); // ... load EffectData
  dp::sg::core::ParameterGroupDataSharedPtr (NBFLoader::*m_pfnLoadFaceAttribute)(uint_t); // ... load FaceAttribute
  dp::sg::core::NodeSharedPtr       (NBFLoader::*m_pfnLoadGeoNode)(uint_t); // ... load GeoNode
  dp::sg::core::GroupSharedPtr      (NBFLoader::*m_pfnLoadGroup)(uint_t); // ... load Group
  dp::sg::core::LightSourceSharedPtr  (NBFLoader::*m_pfnLoadLightSource)(uint_t);   // ... load LightSource
  dp::sg::core::LODSharedPtr        (NBFLoader::*m_pfnLoadLOD)(uint_t);           // ... load LOD
  dp::sg::core::ParameterGroupDataSharedPtr (NBFLoader::*m_pfnLoadMaterial)(uint_t);      // ... load Material
  dp::sg::core::PrimitiveSharedPtr  (NBFLoader::*m_pfnLoadPrimitive)(uint_t);     // ... load Primitive
  dp::sg::core::SamplerSharedPtr    (NBFLoader::*m_pfnLoadSampler)(uint_t offset);  // ... load Sampler
  dp::sg::core::SceneSharedPtr      (NBFLoader::*m_pfnLoadScene)(uint_t);         // ... load Scene
  dp::sg::core::SwitchSharedPtr     (NBFLoader::*m_pfnLoadSwitch)(uint_t);        // ... load Switch
  void                      (NBFLoader::*m_pfnLoadStateSet)(uint_t);      // ... load StateSet
  dp::sg::core::ParameterGroupDataSharedPtr (NBFLoader::*m_pfnLoadTextureAttributeItem)(uint_t); // ... load TextureAttributeItem
  dp::sg::core::TransformSharedPtr  (NBFLoader::*m_pfnLoadTransform)(uint_t);     // ... load Transform
  dp::sg::core::VertexAttributeSetSharedPtr (NBFLoader::*m_pfnLoadVertexAttributeSet)(uint_t); // ... load VertexAttributeSet
  dp::sg::ui::ViewStateSharedPtr    (NBFLoader::*m_pfnLoadViewState)(uint_t);   //..load ViewState

  dp::sg::core::PrimitiveSharedPtr  (NBFLoader::*m_pfnLoadQuadPatches)   (uint_t offset);
  dp::sg::core::PrimitiveSharedPtr  (NBFLoader::*m_pfnLoadQuadPatches4x4)(uint_t offset);
  dp::sg::core::PrimitiveSharedPtr  (NBFLoader::*m_pfnLoadRectPatches)   (uint_t offset);
  dp::sg::core::PrimitiveSharedPtr  (NBFLoader::*m_pfnLoadTriPatches)    (uint_t offset);
  dp::sg::core::PrimitiveSharedPtr  (NBFLoader::*m_pfnLoadTriPatches4)   (uint_t offset);

  dp::sg::core::TextureHostSharedPtr (NBFLoader::*m_pfnLoadTextureHost)    (uint_t offset, std::string& file);

  std::map<std::string,dp::sg::core::TextureHostWeakPtr> m_textureImages;  // collection of currently loaded TextureHosts

  std::map<dp::sg::core::LightSourceSharedPtr,dp::sg::core::GroupSharedPtr> m_lightSourceToGroup;   // for m_nbfMajor < 0x51: stores light source to referencing group

  std::map<uint_t,dp::sg::core::EffectDataSharedPtr> m_stateSetToEffect;
  std::map<uint_t,dp::sg::core::EffectDataSharedPtr> m_materialToMaterialEffect;
  dp::sg::core::EffectDataSharedPtr m_materialEffect;
  dp::fx::SmartEffectSpec   m_currentEffectSpec;
};

inline void NBFLoader::deleteThis()
{
  delete this;
}

inline dp::sg::core::EffectDataSharedPtr NBFLoader::getMaterialEffect()
{
  if ( ! m_materialEffect )
  {
    m_materialEffect = dp::sg::core::EffectData::create( dp::sg::core::getStandardMaterialSpec() );
  }
  return( m_materialEffect );
}

inline ubyte_t * NBFLoader::mapOffset( uint_t offset, unsigned int numBytes )
{
  DP_ASSERT( m_fm );
  return( (ubyte_t*) m_fm->mapIn( offset, numBytes ) );
}

inline void NBFLoader::unmapOffset( ubyte_t * offsetPtr )
{
  DP_ASSERT( m_fm );
  m_fm->mapOut( offsetPtr );
}

template<typename T>
inline NBFLoader::Offset_AutoPtr<T>::Offset_AutoPtr( dp::util::ReadMapping * fm
                                                   , const dp::util::PlugInCallback * pic
                                                   , uint_t offset, unsigned int count )
: m_ptr(NULL)
, m_fm(fm)
, m_pic(pic)
{
  if ( count )
  {
    m_ptr = (T*)m_fm->mapIn(offset, count*sizeof(T));
    if ( ! m_ptr && m_pic )
    {
      m_pic->onFileMappingFailed(m_fm->getLastError());
    }
  }
 }

template<typename T>
inline NBFLoader::Offset_AutoPtr<T>::~Offset_AutoPtr()
{
  if ( m_ptr )
  {
    m_fm->mapOut((ubyte_t*)m_ptr);
  }
}

template<typename T>
inline T* NBFLoader::Offset_AutoPtr<T>::operator->() const
{
  return m_ptr;
}

template<typename T>
inline T& NBFLoader::Offset_AutoPtr<T>::operator*() const
{
  return *m_ptr;
}

template<typename T>
inline NBFLoader::Offset_AutoPtr<T>::operator const T*() const
{
  return m_ptr;
}

template<typename T>
inline void NBFLoader::Offset_AutoPtr<T>::reset( uint_t offset, unsigned int count )
{
  if ( m_ptr )
  {
    m_fm->mapOut((ubyte_t*)m_ptr);
    m_ptr=NULL;
  }
  if ( count )
  {
    m_ptr = (T*)m_fm->mapIn(offset, count*sizeof(T));
    if ( ! m_ptr && m_pic )
    {
      m_pic->onFileMappingFailed(m_fm->getLastError());
    }
  }
}
