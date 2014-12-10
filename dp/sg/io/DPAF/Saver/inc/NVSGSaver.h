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

#include  <set>
#include  <dp/sg/core/nvsgapi.h>
#include  <dp/sg/io/PlugInterface.h>
#include  <dp/sg/ui/ViewState.h>
#include  <dp/sg/algorithm/Traverser.h>


//  Don't need to document the API specifier
#if ! defined( DOXYGEN_IGNORE )
#if defined(_WIN32)
# ifdef NVSGSAVER_EXPORTS
#  define NVSGSAVER_API __declspec(dllexport)
# else
#  define NVSGSAVER_API __declspec(dllimport)
# endif
#else
# define NVSGSAVER_API
#endif
#endif  //  DOXYGEN_IGNORE

// exports required for a scene loader plug-in
extern "C"
{
//! Get the PlugIn interface for this scene saver.
/** Every PlugIn has to resolve this function. It is used to get a pointer to a PlugIn class, in this case a 
NVSGSaver.
  * If the PlugIn ID \a piid equals \c PIID_NVSG_SCENE_SAVER, a NVSGSaver is created and returned in \a pi.
  * \returns  true, if the requested PlugIn could be created, otherwise false
  */
NVSGSAVER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi);

//! Query the supported types of PlugIn Interfaces.
NVSGSAVER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}

//! A Traverser to traverse a scene on saving to nvsg file format.
/** \note Needs a valid ViewState. Call setViewState prior to apply().*/
class NVSGSaveTraverser : public dp::sg::algorithm::SharedTraverser
{
  public:
    //! Default constructor
    NVSGSaveTraverser();

    //! Sets the FILE where the scene is to be saved to.
    void  setFILE( FILE *fh                       //!<  FILE to save to
                 , std::string const& filename
                 );

  protected:
    //! Controls saving of the scene together with a ViewState.
    void doApply( const dp::sg::core::NodeSharedPtr & root );

    // overloads to process concrete types for saving
    // ...Cameras
    //! Save a \c ParallelCamera.
    /** If the \c ParallelCarmera \a p is encountered on saving the first time, it is traversed with \a root and then it's
      * saved. */
    virtual void handleParallelCamera(const dp::sg::core::ParallelCamera *p);

    //! Save a \c PerspectiveCamera.
    /** If the \c PerspectiveCamera \a p is encountered on saving the first time, it is traversed with \a root and then
      * it's saved. */
    virtual void handlePerspectiveCamera(const dp::sg::core::PerspectiveCamera *p);

    //! Save a \c MatrixCamera.
    /** If the \c MatrixCamera \a p is encountered on saving the first time, it is traversed with \a root and then
      * it's saved. */
    virtual void handleMatrixCamera( const dp::sg::core::MatrixCamera * p );

    // ...Nodes
    //! Save a \c Billboard.
    /** If the \c Billboard \a p is encountered on saving the first time, it is traversed and then saved. */
    virtual void handleBillboard(const dp::sg::core::Billboard *p);

    //! Save a \c GeoNode.
    /** If the \c GeoNode \a p is encountered on saving the first time, it is traversed and then saved. */
    virtual void handleGeoNode(const dp::sg::core::GeoNode *p);

    //! Save a \c Group.
    /** If the \c Group \a p is encountered on saving the first time, it is traversed and then saved. */
    virtual void handleGroup( const dp::sg::core::Group *p );

    //! Save a \c Transform.
    /** If the \c Transform \a p is encountered on saving the first time, it is traversed and then saved. */
    virtual void handleTransform(const dp::sg::core::Transform *p);

    //! Save a \c LOD.
    /** If the \c LOD \a p is encountered on saving the first time, all it's children are traversed, no matter which might
      * be currently active, then it is saved. */
    virtual void handleLOD(const dp::sg::core::LOD *p);

    //! Save a \c Switch.
    /** If the \c Switch \a p is encountered on saving the first time, all it's children are traversed, no matter which
      * might be currently active, then it is saved. */
    virtual void handleSwitch(const dp::sg::core::Switch *p);

    virtual void handleLightSource( const dp::sg::core::LightSource * p );

    //! Save a \c Primitive.
    /** If the \c Primitive \a p is encountered on saving the first time, it is saved. */
    virtual void handlePrimitive(const dp::sg::core::Primitive *p);

    //! Save an \c IndexSet.
    /** If the \c IndexSet \a p is encountered on saving the first time, it is saved. */
    virtual void handleIndexSet( const dp::sg::core::IndexSet * p );

    //! Save a \c VertexAttributeSet.
    /** If the \c VertexAttributeSet \a p is encountered on saving the first time, it is saved. */
    virtual void handleVertexAttributeSet( const dp::sg::core::VertexAttributeSet *p );

    virtual void handleEffectData( const dp::sg::core::EffectData * p );
    virtual void handleParameterGroupData( const dp::sg::core::ParameterGroupData * p );
    virtual void handleSampler( const dp::sg::core::Sampler * p );

private:
    void    cameraData( const dp::sg::core::Camera *p );
    void    frustumCameraData( const dp::sg::core::FrustumCamera *p );
    const   std::string  getName( const std::string &name );
    std::string getObjectName( const dp::sg::core::Object *p );
    void    objectData( const dp::sg::core::Object *p ); // writes object data
    void    groupData( const dp::sg::core::Group *p );
    void    initUnnamedCounters( void );
    bool    isFirstTime( const dp::sg::core::HandledObject * p );
    void    lightSourceData( const dp::sg::core::LightSource *p );
    void    nodeData( const dp::sg::core::Node *p );
    std::string parameterString( const dp::sg::core::ParameterGroupData * p, dp::fx::ParameterGroupSpec::iterator it );
    void    primitiveData( const dp::sg::core::Primitive *p );
    void    textureImage( const dp::sg::core::TextureHostSharedPtr & tih );
    void    transformData( const dp::sg::core::Transform * p );
    void    vertexAttributeSetData( const dp::sg::core::VertexAttributeSet * p );
    void    writeVertexData( const dp::sg::core::VertexAttribute & va );
    void    buffer( const dp::sg::core::Buffer *p );
  private:
    struct CallbackLink
    {
      std::string           name;
      dp::sg::core::ObjectWeakPtr  subject;
      dp::sg::core::ObjectWeakPtr  observer;
    };

  private:
    FILE * m_fh;

    std::vector<std::string>                                m_basePaths;
    std::string                                             m_effectSpecName;
    std::map<dp::sg::core::DataID, std::string>             m_sharedData;
    std::set<const dp::sg::core::HandledObject *>           m_sharedObjects;
    std::map<const dp::sg::core::Buffer *, std::string>     m_storedBuffers;
    std::map<const dp::sg::core::Sampler *, std::string>    m_storedSamplers;
    std::map<dp::sg::core::ObjectWeakPtr, std::string>      m_objectNames;
    unsigned int                                            m_nameCount;
    std::set<std::string>                                   m_nameSet;
    std::map<dp::sg::core::TextureHostWeakPtr, std::string> m_textureImageNames;
    std::map<dp::sg::core::TransformWeakPtr, std::string>   m_unnamedTransforms;
    std::vector<CallbackLink>                               m_links;
};

DEFINE_PTR_TYPES( NVSGSaver );

//! A Scene Saver for nvsg files.
/** NVSG files can be produced with the sample ViewerVR. 
  * They are text files that represent a Scene and a ViewState.  */
class NVSGSaver : public dp::sg::io::SceneSaver
{
  public :
    static NVSGSaverSharedPtr create();
    virtual ~NVSGSaver();

    //! Realization of the pure virtual interface function of a SceneSaver.
    /** Saves the \a scene and the \a viewState to \a filename. 
      * The \a viewState may be NULL. */
    bool  save( dp::sg::core::SceneSharedPtr    const& scene      //!<  scene to save
              , dp::sg::ui::ViewStateSharedPtr  const& viewState  //!<  view state to save
              , std::string                     const& filename   //!<  file name to save to
              );
  protected:
    NVSGSaver();
};
