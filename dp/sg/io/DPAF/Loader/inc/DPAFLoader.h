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
/** \file */

#include <dp/sg/core/Config.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/util/File.h>
#include <dp/util/Tools.h>
#include <dp/util/Tokenizer.h>
#include <dp/sg/io/PlugInterface.h>

#include <fstream>

// storage-class defines 
#if ! defined( DOXYGEN_IGNORE )
# if defined(DP_OS_WINDOWS)
#  ifdef DPAFLOADER_EXPORTS
#   define DPAFLOADER_API __declspec(dllexport)
#  else
#   define DPAFLOADER_API __declspec(dllimport)
#  endif
# else
#   define DPAFLOADER_API
# endif
#endif

#if defined(LINUX)
typedef unsigned int DWORD;
void lib_init() __attribute__ ((constructor));   // will be called before dlopen() returns
#endif

// exports required for a scene loader plug-in
extern "C"
{
//! Get the PlugIn interface for this scene loader.
/** Every PlugIn has to resolve this function. It is used to get a pointer to a PlugIn class, in this case a DPAFLoader.
  * If the PlugIn ID \a piid equals \c PIID_DP_SCENE_LOADER, a DPAFLoader is created and returned in \a pi.
  * \returns  true, if the requested PlugIn could be created, otherwise false
  */
DPAFLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi);

//! Query the supported types of PlugIn Interfaces.
DPAFLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

DEFINE_PTR_TYPES( DPAFLoader );

//! A Scene Loader for DPAF files.
/** DPAF files can be produced with the sample Viewer. 
  * They are text files that represent a Scene and a ViewState.  */
class DPAFLoader : public dp::sg::io::SceneLoader
{
  public :
    static DPAFLoaderSharedPtr create();
    virtual ~DPAFLoader();

  public :

    //! Realization of the pure virtual interface function of a SceneLoader.
    /** Loads a DPAF file given by \a filename. It looks for this file and 
      * possibly referenced other files like textures or effects at the given 
      * path first, then at the current location and finally it searches
      * through the \a searchPaths.
      * \returns  A pointer to the loaded scene. */
    dp::sg::core::SceneSharedPtr load( std::string const& filename                  //!<  file to load
                                     , std::vector<std::string> const& searchPaths  //!<  paths to search through
                                     , dp::sg::ui::ViewStateSharedPtr & viewState   /*!< If the function succeeded, this points to the optional
                                                                                         ViewState stored with the scene. */
                                     );

  protected:
    DPAFLoader();

  private:
    struct PrimitiveData
    {
      dp::sg::core::PrimitiveType               primitiveType;
      dp::sg::core::PatchesType                 patchesType;
      dp::sg::core::PatchesMode                 patchesMode;
      dp::sg::core::PatchesSpacing              patchesSpacing;
      dp::sg::core::PatchesOrdering             patchesOrdering;
      unsigned int                              elementCount;
      unsigned int                              elementOffset;
      dp::sg::core::IndexSetSharedPtr           indexSet;
      unsigned int                              instanceCount;
      dp::sg::core::VertexAttributeSetSharedPtr vertexAttributeSet;
    };

  private :
    void                                              cleanup( void );
    bool                                              getNextLine( void );
    const std::string                               & getNextToken( void );
    dp::sg::core::SceneSharedPtr                      import( std::string const& filename, dp::sg::ui::ViewStateSharedPtr & viewState );
    bool                                              onIncompatibleValues( int value0, int value1, const std::string &node, const std::string &field0, const std::string &field1 ) const;
    template<typename T> bool                         onInvalidValue( T value, const std::string &node, const std::string &field ) const;
    bool                                              onEmptyToken( const std::string &tokenType, const std::string &token ) const;
    bool                                              onFileNotFound( const std::string &file ) const;
    bool                                              onFilesNotFound( bool found, const std::vector<std::string> &files ) const;
    void                                              onUnexpectedEndOfFile( bool error ) const;
    void                                              onUnexpectedToken( const std::string &expected, const std::string &token ) const;
    void                                              onUnknownToken( const std::string &context, const std::string &token ) const;
    bool                                              onUndefinedToken( const std::string &context, const std::string &token ) const;
    bool                                              onUnsupportedToken( const std::string &context, const std::string &token ) const;
    dp::sg::core::Billboard::Alignment                readAlignment();
    dp::sg::core::BillboardSharedPtr                  readBillboard( const char *name, const std::string & extName );
    bool                                              readBool( const std::string & token = std::string() );
    dp::sg::core::BufferSharedPtr                     readBuffer( void );
    bool                                              readCameraToken( dp::sg::core::CameraSharedPtr const& camera, std::string & token );
    dp::sg::core::NodeSharedPtr                       readChild( const std::string & token );
    void                                              readChildren( dp::sg::core::GroupSharedPtr const& group );
    bool                                              readFrustumCameraToken( dp::sg::core::FrustumCameraSharedPtr const& camera, std::string & token );
    dp::DataType                                      readType( const char *token = NULL );
    dp::sg::core::EffectDataSharedPtr                 readEffectData( const char * name );
    void                                              readEnumArray( const std::string & token, std::vector<int> & values, const dp::fx::ParameterSpec & ps );
    template<unsigned int m, unsigned int n> dp::math::Matmnt<m,n,char> readEnumMatrix( const std::string & t, const dp::fx::ParameterSpec & ps );
    template<unsigned int n> dp::math::Vecnt<n,char>  readEnumVector( const std::string & token, const dp::fx::ParameterSpec & ps );
    dp::sg::core::GeoNodeSharedPtr                    readGeoNode( const char *name );
    dp::sg::core::GroupSharedPtr                      readGroup( const char *name, const std::string & extName );
    bool                                              readGroupToken( dp::sg::core::GroupSharedPtr const& group, const std::string & token, const std::string & extName );
    void                                              readImages( dp::sg::core::TextureHost * ti );
    dp::sg::core::IndexSetSharedPtr                   readIndexSet( const char * name );
    dp::sg::core::LightSourceSharedPtr                readLightSource( const std::string & token );
    dp::sg::core::LightSourceSharedPtr                readLightSourceReferences( const std::string & token );
    bool                                              readLightSourceToken( dp::sg::core::LightSourceSharedPtr const& light, const std::string &token );
    dp::sg::core::LODSharedPtr                        readLOD( const char *name, const std::string & extName );
    template<unsigned int m, unsigned int n, typename T> dp::math::Matmnt<m,n,T> readMatrix( const std::string & token );
    dp::sg::core::MatrixCameraSharedPtr               readMatrixCamera( const char *name );
    void                                              readMipmaps( unsigned int width, unsigned int height, unsigned int depth, unsigned int noc, dp::sg::core::Image::PixelDataType pt, std::vector<const void *> & mipmaps );
    std::string                                       readName( const std::string & token );
    bool                                              readNodeToken( dp::sg::core::NodeSharedPtr const& node, const std::string & token );
    bool                                              readObjectToken( dp::sg::core::ObjectSharedPtr const& object, const std::string & token );
    bool                                              readObjectToken( const std::string & token, std::string & annotation, unsigned int & hints );
    dp::sg::core::ParallelCameraSharedPtr             readParallelCamera( const char *name );
    void                                              readParameter( dp::sg::core::ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it );
    void                                              readParameterEnum( dp::sg::core::ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it );
    template<unsigned int m, unsigned int n> void     readParameterEnumMN( dp::sg::core::ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it );
    template<unsigned int n> void                     readParameterEnumVN( dp::sg::core::ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it );
    dp::sg::core::ParameterGroupDataSharedPtr         readParameterGroupData( const char * name );
    template<unsigned int m, unsigned int n, typename T> void readParameterMNT( dp::sg::core::ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it );
    template<typename T> void                         readParameterT( dp::sg::core::ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it );
    unsigned int                                      readParameterType();
    template<unsigned int n, typename T> void         readParameterVNT( dp::sg::core::ParameterGroupDataSharedPtr const& pgd, dp::fx::ParameterGroupSpec::iterator it );
    dp::sg::core::PatchesMode                         readPatchesMode();
    dp::sg::core::PatchesOrdering                     readPatchesOrdering();
    dp::sg::core::PatchesSpacing                      readPatchesSpacing();
    dp::sg::core::PerspectiveCameraSharedPtr          readPerspectiveCamera( const char *name );
    dp::sg::core::Image::PixelFormat                  readPixelFormat();
    dp::sg::core::Image::PixelDataType                readPixelType();
    unsigned char                                   * readPixels( unsigned int nov, dp::sg::core::Image::PixelDataType pt );
    dp::sg::core::PrimitiveSharedPtr                  readPrimitive( const char *name );
    void                                              readPrimitiveToken( PrimitiveData & pd, const std::string & token, const std::string & primitiveTypeName );
    dp::math::Quatf                                   readQuatf( const std::string &token );
    dp::sg::core::SamplerSharedPtr                    readSampler( const std::string & name );
    template<typename T> T                            readScalar( const std::string & token );
    template<typename T> void                         readScalarArray( const std::string & token, std::vector<T> & values );
    dp::sg::core::SceneSharedPtr                      readScene( void );
    dp::sg::core::SwitchSharedPtr                     readSwitch( const char *name, const std::string & extName );
    dp::sg::core::TextureCompareMode                  readTextureCompareMode();
    dp::sg::core::TextureMagFilterMode                readTextureMagFilterMode();
    dp::sg::core::TextureMinFilterMode                readTextureMinFilterMode();
    dp::sg::core::TextureHostSharedPtr                readTextureHost( const char *name );
    dp::sg::core::TextureTarget                       readTextureTarget();
    dp::sg::core::TextureWrapMode                     readTextureWrapMode();
    dp::sg::core::TransformSharedPtr                  readTransform( const char *name, const std::string & extName );
    bool                                              readTransformToken( dp::sg::core::TransformSharedPtr const& transform, const std::string & token, const std::string & extName );
    template<unsigned int n, typename T> dp::math::Vecnt<n,T> readVector( const std::string & token );
    void                                              readVertexData( unsigned int type, unsigned char * vdata, std::string token );
    dp::sg::core::VertexAttributeSetSharedPtr         readVertexAttributeSet( const char *name );
    bool                                              readVertexAttributeSetToken( dp::sg::core::VertexAttributeSetSharedPtr const& vas, const std::string & token );
    dp::sg::ui::ViewStateSharedPtr                    readViewState( void );
    void                                              setPrimitiveData( const dp::sg::core::PrimitiveSharedPtr & primitive, const PrimitiveData & data, const std::string & name );
    void                                              storeTextureHost( const std::string & name, const dp::sg::core::TextureHostSharedPtr & tih );
    bool                                              testDPVersion( void );
#if !defined(NDEBUG)
    void                                              assertClean();
#endif
    
    template<typename T>          void            readPixelComponent( const std::string & token, T & pc );
    template<typename T>          unsigned char * readPixels( unsigned int nov );
    template<typename ObjectType> void            storeNamedObject( const std::string & name, std::map<std::string, dp::util::SharedPtr<ObjectType> > & container, const dp::util::SharedPtr<ObjectType> & obj );

  private :
    std::map<std::string,dp::sg::core::BillboardSharedPtr>          m_billboards;
    std::map<std::string,dp::sg::core::BufferSharedPtr>             m_buffers;
    dp::math::Vec3f                                                 m_cameraDirection;
    dp::math::Vec3f                                                 m_cameraUpVector;
    std::string                                                     m_currentLine;
    std::string                                                     m_currentString;
    std::map<std::string,dp::sg::core::EffectDataSharedPtr>         m_effectData;
    std::map<std::string,dp::sg::core::GeoNodeSharedPtr>            m_geoNodes;
    std::map<std::string,dp::sg::core::GroupSharedPtr>              m_groups;
    std::ifstream                                                   m_ifs;
    std::map<std::string,dp::sg::core::IndexSetSharedPtr>           m_indexSets;
    std::map<std::string,dp::sg::core::LightSourceSharedPtr>        m_lightSources;
    unsigned int                                                    m_line;
    std::map<std::string,dp::sg::core::LODSharedPtr>                m_LODs;
    std::map<std::string,dp::sg::core::MatrixCameraSharedPtr>       m_matrixCameras;
    std::map<std::string,dp::sg::core::ObjectSharedPtr>             m_objects;
    std::map<std::string,dp::sg::core::ParallelCameraSharedPtr>     m_parallelCameras;
    std::map<std::string,dp::sg::core::ParameterGroupDataSharedPtr> m_parameterGroupData;
    std::map<std::string,dp::sg::core::PerspectiveCameraSharedPtr>  m_perspectiveCameras;
    std::map<std::string,dp::sg::core::PrimitiveSharedPtr>          m_primitives;
    std::map<std::string,dp::sg::core::PrimitiveSharedPtr>          m_quadMeshes;
    std::map<std::string,dp::sg::core::SamplerSharedPtr>            m_samplers;
    dp::sg::core::SceneSharedPtr                                    m_scene;
    std::vector<std::string>                                        m_searchPaths;
    std::map<std::string,dp::sg::core::SwitchSharedPtr>             m_switches;
    std::map<std::string,dp::sg::core::TextureHostSharedPtr>        m_textureImages;
    std::map<std::string,dp::sg::core::TransformSharedPtr>          m_transforms;
    std::map<std::string,dp::sg::core::VertexAttributeSetSharedPtr> m_vertexAttributeSets;
    dp::util::StrTokenizer                                          m_strTok;
};

template <typename ObjectType>
inline void DPAFLoader::storeNamedObject( const std::string & name
                                        , std::map<std::string, dp::util::SharedPtr<ObjectType> > & container
                                        , dp::util::SharedPtr<ObjectType> const & obj )
{
  if ( obj )
  {
    DP_ASSERT( container.find( name ) == container.end() );
    container[name] = obj;

    // map of all objects for subject/observer links at end of file
    DP_ASSERT( m_objects.find( name ) == m_objects.end() );
    m_objects[name] = obj;
  }
}
