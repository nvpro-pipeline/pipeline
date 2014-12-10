// Copyright NVIDIA Corporation 2002-2010
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

#include <map>
#include <vector>
#include <dp/fx/EffectSpec.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/io/PlugInterface.h>
#include <dp/util/File.h>
#include <dp/util/FileMapping.h>
#include <dp/sg/algorithm/Traverser.h>
#include <NBF.h>

#if defined(_WIN32)
# ifdef NBFSAVER_EXPORTS
#  define NBFSAVER_API __declspec(dllexport)
# else
#  define NBFSAVER_API __declspec(dllimport)
# endif
#else
# define NBFSAVER_API
#endif

// exports required for a scene loader plug-in
extern "C"
{
NBFSAVER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::PlugInSharedPtr & pi);
NBFSAVER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}


/*! \brief A Traverser to traverse a scene on saving to "NBF" binary file format.
 *  \remarks For each Object in the tree, the handle<Object> function is called twice from the
 *  NBFSaver framework. The first time it is called to calculate storage requirements, and the
 *  second time it is called to store the specified object to a file.\n
 *  The implementation calls calculatingStorageRequirements to determine whether the storage
 *  requirement calculation is active or not. During storage requirement calculation no file
 *  mapping is available, and therefore no file offsets may be allocated. Within this
 *  calculation pass, the implementation uses pseudoAlloc instead of alloc. pseudoAlloc simply
 *  adds up the number of bytes required for the corresponding NBF structure and attached data
 *  The implementation uses alloc in the second pass to map the required amount of memory from
 *  the file mapping maintained by the NBFSaver. alloc returns a pointer to the file offset
 *  which is then used to write the corresponding NBF data to the file.
 */
class NBFSaveTraverser : public dp::sg::algorithm::SharedTraverser
{
  public:
    /*! \brief Default constructor */
    NBFSaveTraverser();

    /*! \brief Calculate the expected file size.
     *  \param scene The Scene to use in file size calculation.
     *  \return \c true if the file size is less than 4G, and thus can be saved. Otherwise \c false.
     *  \remarks In this function the complete scene is traversed once to determine the required
     *  file size. As NBF is a 32bit file format, only files up to 4G can be saved. */
    bool preCalculateFileSize( const dp::sg::core::SceneSharedPtr & scene );

    /*! \brief Set the file name to save the scene in.
     *  \param fileName The name of the file to save in. */
    void setFileName( const std::string & fileName );

    /*! \brief Set the PlugInCallback used for warnings and error messages.
     *  \param pic A pointer to the PlugInCallback object to use.
     *  \remarks The PlugInCallback has a number of functions that are called in case an error or
     *  warning is to be reported while saving. */
    void setPlugInCallback( const dp::util::PlugInCallback * pic );

    /*! \brief Get the success state of the latest saving operation.
     *  \return \c true if the latest saving was successful, otherwise \c false. */
    bool getSuccess() const;

    /*! \brief Get the error message if the latest saving operation was not successful.
     *  \return The string containing the last error. */
    std::string getErrorMessage() const;

  protected:
    /*! \brief Determine whether the current traverser pass is calculating storage requirements.
     *  \returns \c true if the current pass is to calculate storage requirements, \c false if the
     *  current pass is to write the scene to a file that has been mapped into the process' address
     *  space by the framework.
     *  \remarks While saving a scene, the scene graph will be traversed twice. The first pass is to
     *  calculate storage requirements. The second pass writes the scene data to a file that has
     *  been mapped into the process' address space by the framework.\n
     *  Handler routines should call this function to determine what the current pass is intended
     *  for.
     *  \sa pseudoAlloc */
    bool calculatingStorageRequirements() const;

    /*! \brief Allocate a portion of the currently mapped file into the process' address space.
     *  \param offsetPtr A reference to the pointer to get the address of the mapped memory block.
     *  \param numBytes The number of bytes to be mapped into the process' address space.
     *  \return The file offset of the mapped memory block
     *  \remarks The function maps the amount of \a numBytes bytes of the currently mapped file into
     *  the process' address space. The start address of the mapped memory block will be assigned to
     *  \a ptr.
     *  \note The behavior is undefined if called while the current traverser pass is calculating
     *  storage requirements.
     *  \sa calculatingStorageRequirements, dealloc */
    uint_t alloc( void*& offsetPtr, unsigned int numBytes );

    /*! \brief Deallocate a portion of memory that is currently mapped to a file.
     *  \param offsetPtr A mapped memory address, previously allocated with alloc.
     *  \note: The memory, which is part of a potentially larger file mapping, actually won't be
     *  freed until that file mapping is unused (unreferenced).
     *  \sa alloc */
    void dealloc( void * offsetPtr );

    /*! \brief Add \a numBytes to the currently calculated file size.
     *  \param numBytes The number of bytes to add to the currently calculated file size.
     *  \return This function always returns zero.
     *  \remarks The function does not allocate anything, it just adds \a numBytes to the currently
     *  calculated file size.
     *  \sa calculatingStorageRequirements */
    uint_t pseudoAlloc( unsigned int numBytes );

    /*! \brief Override of the traversal initiating interface.
     *  \param root The Node to use as the root of the save operation.
     *  \remarks The framework calls this method to perform the file save operation.  If a Scene and
     *  ViewState have been provided, they are locked before this function is called. */
    virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

    /*! \brief Save a ParallelCamera.
     *  \param camera A pointer to the read-locked ParallelCamera to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc, handleMatrixCamera, handleParallelCamera */
    virtual void handleParallelCamera( const dp::sg::core::ParallelCamera *camera );

    /*! \brief Save a PerspectiveCamera.
     *  \param camera A pointer to the read-locked PerspectiveCamera to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc, handleMatrixCamera, handlePerspectiveCamera */
    virtual void handlePerspectiveCamera( const dp::sg::core::PerspectiveCamera *camera );

    /*! \brief Save a MatrixCamera.
     *  \param camera A pointer to the read-locked MatrixCamera to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc, handleParallelCamera, handlePerspectiveCamera */
    virtual void handleMatrixCamera( const dp::sg::core::MatrixCamera * camera );

    /*! \brief Save a Billboard.
     *  \param billboard A pointer to the read-locked Billboard to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc */
    virtual void handleBillboard( const dp::sg::core::Billboard *billboard );

    /*! \brief Save a GeoNode.
     *  \param gnode A pointer to the read-locked GeoNode to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc */
    virtual void handleGeoNode( const dp::sg::core::GeoNode *gnode );

    /*! \brief Save a Group.
     *  \param group A pointer to the read-locked Group to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc */
    virtual void handleGroup( const dp::sg::core::Group *group );

    /*! \brief Save a Transform.
     *  \param trafo A pointer to the read-locked Transform to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc */
    virtual void handleTransform( const dp::sg::core::Transform *trafo );

    /*! \brief Save a LOD.
     *  \param lod A pointer to the read-locked LOD to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc */
    virtual void handleLOD( const dp::sg::core::LOD *lod );

    /*! \brief Save a Switch.
     *  \param swtch A pointer to the read-locked Switch to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc */
    virtual void handleSwitch( const dp::sg::core::Switch *swtch );

    virtual void handleLightSource( const dp::sg::core::LightSource * p );

    /*! \brief Save a Primitive.
     *  \param prim A pointer to the read-locked Primitive to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc */
    virtual void handlePrimitive( const dp::sg::core::Primitive *prim );

    /*! \brief Save an IndexSet.
     *  \param p A pointer to the read-locked IndexSet to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc */
    virtual void handleIndexSet( const dp::sg::core::IndexSet *p );

    /*! \brief Save a VertexAttributeSet.
     *  \param vas A pointer to the read-locked VertexAttributeSet to save.
     *  \sa calculatingStorageRequirements, pseudoAlloc */
    virtual void handleVertexAttributeSet( const dp::sg::core::VertexAttributeSet *vas );

    virtual void handleEffectData( const dp::sg::core::EffectData * p );
    virtual void handleParameterGroupData( const dp::sg::core::ParameterGroupData * p );
    virtual void handleSampler( const dp::sg::core::Sampler * p );


  private:
    struct Mapping     
    {
      Mapping();
      ubyte_t       * basePtr;
      unsigned int    size;    // needed with linux's unmap call only  
      int             refCnt;  // reflects if the view is in use or not
    };

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
        Offset_AutoPtr( NBFSaveTraverser * svr, uint_t& fileOffset, unsigned int count=1 );

        //! Constructor
        /** This constructor does not allocate any mapping space yet.
        * Use the alloc() member function, to initialize the pointer later. */
        Offset_AutoPtr( NBFSaveTraverser * svr );

        //! Unmaps the bytes, that have been mapped at instantiation, from process memory. 
        ~Offset_AutoPtr();

        //! Provides pointer-like access to the dumb pointer. 
        T* operator->();

        //! De-references the dumb pointer. 
        T& operator*();

        //! Implicit conversion to const T*. 
        operator T*();

        //! Allocate the pointer object
        /** This function can be used to create the mapping later,
        * when using the constructor that does not allocate any space yet. */    
        void alloc( uint_t& fileOffset, unsigned int count=1 );

      private:
        T * m_ptr;    
        NBFSaveTraverser * m_svr;
    };

  private:
    // operator() to process a Scene (not part of the public interface!)
    // returns the file offset to the allocated and initialized NBFScene object
    // behavior is undefined if p is an invalid pointer
    uint_t handleScene( dp::sg::core::SceneSharedPtr const& scene );
    // operator() to process a ViewState (not part of the public interface!)
    // returns the file offset to the allocated and initialized NBFViewState object
    // behavior is undefined if p is an invalid pointer
    uint_t handleViewState( dp::sg::ui::ViewStateSharedPtr const& viewState );

    // write routines for non-concrete types
    // ... write tasks similar for all Objects
    void writeObject(const dp::sg::core::Object* nvsgObjPtr, NBFObject * nbfObjPtr, uint_t objCode);
    // ... write task similar for Nodes
    void writeNode(const dp::sg::core::Node* nvsgNodePtr, NBFNode * nbfNodePtr, uint_t objCode);
    // ... write task similar for Groups
    void writeGroup(const dp::sg::core::Group * nvsgGrpPtr, NBFGroup * nbfGrpPtr, uint_t objCode);
    // ... write task similar for LightSources
    void writeLightSource(const dp::sg::core::LightSource* nvsgLightSrcPtr, NBFLightSource * nbfLightSrcPtr, uint_t objCode);
    // ... write task similar for Cameras
    void writeCamera(const dp::sg::core::Camera * nvsgCamPtr, NBFCamera * nbfCamPtr, uint_t objCode);
    void writeFrustumCamera( const dp::sg::core::FrustumCamera * nvsgCamPtr, NBFFrustumCamera * nbfCampPtr, uint_t objCode );
    // .. write faces
    template<typename NVSGFaceType, typename NBFFaceType>
    void writeIndices(const NVSGFaceType * nvsgFaces, NBFFaceType * nbfFaces);
    void writePrimitive( const dp::sg::core::Primitive * nvsgPrimitive, NBFPrimitive * nbfPrimitive, uint_t objCode );
    // .. write a single texture image
    void writeTexImage(const std::string& file, dp::sg::core::TextureHostSharedPtr const& nvsgImg, texImage_t * nbfImg);
    void writeVertexAttributeSet(const dp::sg::core::VertexAttributeSet * nvsgVASPtr, NBFVertexAttributeSet * nbfVASPtr, uint_t objCode);

    // shared objects processing 
    bool processSharedObject(const dp::sg::core::Object * obj, uint_t objCode);

    // map cnt * sizeof(Type) bytes from the mapped file 
    template <typename Type>
    uint_t alloc(Type *& objPtr, unsigned int cnt=1); 

    // pseudo allocation routines for calculation of storage requirements
    void pseudoAllocIndexSet(const dp::sg::core::IndexSet* p);
    void pseudoAllocObject(const dp::sg::core::Object* p);
    void pseudoAllocNode(const dp::sg::core::Node* p);
    void pseudoAllocGroup(const dp::sg::core::Group * p);
    void pseudoAllocLightSource(const dp::sg::core::LightSource * p);
    void pseudoAllocPrimitive(const dp::sg::core::Primitive * p); 
    void pseudoAllocFrustumCamera( const dp::sg::core::FrustumCamera * p );
    void pseudoAllocCamera(const dp::sg::core::Camera * p);
    void pseudoAllocTransform( const dp::sg::core::Transform * p );
    void pseudoAllocTexImage(const std::string& file, const dp::sg::core::TextureHost * img);
    void pseudoAllocVertexAttributeSet( const dp::sg::core::VertexAttributeSet * vas );
  
    ubyte_t * mapOffset( uint_t offset, unsigned int numBytes );
    void unmapOffset( ubyte_t * offsetPtr );

  private:
    struct CallbackLink
    {
      unsigned int         id;
      dp::sg::core::ObjectWeakPtr subject;
      dp::sg::core::ObjectWeakPtr observer;
    };

  private:
    std::vector<std::string>  m_basePaths;
    bool                      m_calculateStorageRequirements; // pre-calculate what the file size would be
    unsigned long long        m_preCalculatedFileSize; // required file size in bytes
    std::string               m_fileName;
    dp::util::WriteMapping  * m_fm;   //!< writable file mapping
    bool                      m_success;  //!< flags if saving was successful
    std::string               m_errorMessage; //!< contains the error if saving was not successful
    unsigned int              m_fileOffset; // actual file offset

    std::map<dp::sg::core::ObjectWeakPtr, uint_t>       m_objectOffsetMap; // mapping NVSG objects to the corresponding offsets in file mapping
    std::map<dp::sg::core::DataID, uint_t>              m_objectDataIDOffsetMap; // mapping object IDs of shared objects to corresponding offsets

    const dp::util::PlugInCallback  * m_pic;

    std::vector<CallbackLink>         m_links;      // links from subject to observer
};

inline void NBFSaveTraverser::setFileName( const std::string & fileName )
{
  m_fileName = fileName;
  dp::util::convertPath( m_fileName );
  m_basePaths[0] = dp::util::getFilePath( m_fileName );
}

inline void NBFSaveTraverser::setPlugInCallback( const dp::util::PlugInCallback * pic )
{
  m_pic = pic;
}

inline bool NBFSaveTraverser::getSuccess() const
{
  return( m_success );
}

inline std::string NBFSaveTraverser::getErrorMessage() const
{
  return( m_errorMessage );
}

DEFINE_PTR_TYPES( NBFSaver );

//! A scene saver capable to save a NVSG scene to a "NBF" (Nvsg Binary File format) file.
class NBFSaver : public dp::sg::io::SceneSaver
{
public :
  static NBFSaverSharedPtr create();
  virtual ~NBFSaver();

  //! Realization of the pure virtual interface function of a SceneSaver.
  /** Saves the \a scene and the \a viewState to \a filename. 
    * The \a viewState may be NULL. */
  bool save( dp::sg::core::SceneSharedPtr   const& scene     //!<  scene to save
           , dp::sg::ui::ViewStateSharedPtr const& viewState //!<  view state to save
           , std::string                    const& filename  //!<  file name to save to
           );

protected:
  NBFSaver();
};


inline bool NBFSaveTraverser::calculatingStorageRequirements() const
{
  return m_calculateStorageRequirements;
}

inline NBFSaveTraverser::Mapping::Mapping()
: refCnt(0),
  basePtr(NULL),
  size(0)
{}

template<typename T>
inline NBFSaveTraverser::Offset_AutoPtr<T>::Offset_AutoPtr( NBFSaveTraverser * svr
                                                          , uint_t& fileOffset, unsigned int count )
: m_ptr(NULL)
, m_svr(svr)
{
  DP_ASSERT(count != 0xCCCCCCCC);  // check for passing uninitialized variables

  if ( count )
  {
    // call the void * version
    fileOffset = m_svr->alloc((void *&)m_ptr, count*sizeof(T));
  }
}

template<typename T>
inline NBFSaveTraverser::Offset_AutoPtr<T>::Offset_AutoPtr( NBFSaveTraverser * svr )
: m_ptr(NULL)
, m_svr(svr)
{
}

template<typename T>
inline NBFSaveTraverser::Offset_AutoPtr<T>::~Offset_AutoPtr()
{
  if ( m_ptr )
  {
    m_svr->dealloc(m_ptr);
  }
}

template<typename T>
inline T* NBFSaveTraverser::Offset_AutoPtr<T>::operator->()
{
  DP_ASSERT(m_ptr);
  return m_ptr;
}

template<typename T>
inline T& NBFSaveTraverser::Offset_AutoPtr<T>::operator*()
{
  DP_ASSERT(m_ptr);
  return *m_ptr;
}

template<typename T>
inline NBFSaveTraverser::Offset_AutoPtr<T>::operator T*()
{
  DP_ASSERT(m_ptr);
  return m_ptr;
}

template<typename T>
inline void NBFSaveTraverser::Offset_AutoPtr<T>::alloc( uint_t& fileOffset, unsigned int count )
{
  DP_ASSERT(m_ptr == NULL);
  DP_ASSERT(count != 0xCCCCCCCC);  // check for passing uninitialized variables

  if ( count )
  {
    fileOffset = m_svr->alloc((void *&)m_ptr, count*sizeof(T));      
  }    
}

inline ubyte_t * NBFSaveTraverser::mapOffset( uint_t offset, unsigned int numBytes )
{
  DP_ASSERT( m_fm );
  return( (ubyte_t*) m_fm->mapIn( offset, numBytes ) );
}

inline void NBFSaveTraverser::unmapOffset( ubyte_t * offsetPtr )
{
  DP_ASSERT( m_fm );
  m_fm->mapOut( offsetPtr );
}

