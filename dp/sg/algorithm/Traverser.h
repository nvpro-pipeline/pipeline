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
/** \file */

#include <dp/sg/algorithm/Config.h>
#include <dp/sg/core/CoreTypes.h> // forward declarations of core type
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Object.h> // ObjectCode defines
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Path.h>  // for traversal id
#include <dp/sg/ui/ViewState.h>
#include <dp/util/BitMask.h>
#include <dp/util/RCObject.h> // base class definition
#include <dp/util/Reflection.h>
#include <boost/scoped_ptr.hpp>
#include <stack>
#include <memory>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {
      /*! \brief Virtual base class providing the common interface part of ExclusiveTraverser and SharedTraverser.
       *  \remarks A traverser serves as a link between a scene graph and a defined operation to be
       *  performed on the scene graph. That is, according to a certain scene graph's hierarchy, a
       *  traverser iterates over all scene graph components and performs a defined operation for
       *  each by means of an overloadable handler routine.
       *  The Traverser class offers base functionality for all traversers. There are two traversers
       *  derived from Traverser: SharedTraverser and ExclusiveTraverser, that provide read-only
       *  access and read-write access, respectively, to the elements of the scene graph.
       *  \sa ExclusiveTraverser, SharedTraverser */
      class Traverser : public dp::util::Reflection
      {
        public:
          /*! \brief Set a ViewState to be used with this Traverser.
           *  \param viewState The ViewState to be used.
           *  \remarks If \a viewState is a different pointer than the current ViewState pointer, its
           *  reference count is incremented and the reference count of a previously set ViewState is
           *  decremented. Any caches for the ViewState are marked as invalid.
           *  \note The ViewState's TraversalMask will be used in conjunction with any TraversalMaskOverride to direct
           *  scene traversal.  See setTraversalMask for more info.
           *  \sa setTraversalMaskOverride, setTraversalMask, Object::setTraversalMask */
          DP_SG_ALGORITHM_API virtual void setViewState( dp::sg::ui::ViewStateSharedPtr const& viewState );

          /*! \brief Set the TraversalMask to be used with this Traverser.
           *  \param mask The mask to be used.
           *  \remarks This method provides a way to set the traversal mask for this Traverser if no ViewState is 
           *  specified.  If a ViewState is specified, the TraversalMask in the ViewState will be used instead.
           *  \note The TraversalMask is used in conjuction with the OverrideTraversalMask and every scene graph node's 
           *  TraversalMask to determine whether the node (and therefore possibly the entire subgraph) is traversed.  
           *  The traverser's override traversal mask is OR'd with the node's traversal mask and that result is ANDed with 
           *  the traversal mask.  If the result is nonzero then the node is traversed, otherwise it is ignored.  IE:
           *  ( ( (Traverser::TraversalMaskOverride | Object::TraversalMask) & ViewState::TraversalMask ) != 0 ) the node is
           *  traversed.
           *  \note Setting the TraversalMask to 0 will cause no nodes to be traversed.  Setting the TraversalMask to ~0 and
           *  the TraversalMaskOverride to ~0 will cause all nodes to be traversed regardless of the Object::TraversalMask.
           *  \note The default traversal mask is always ~0 so that all nodes are traversed.
           *  \sa getTraversalMask, getTraversalMaskOverride, ViewState::getTraversalMask, Object::getTraversalMask */
          DP_SG_ALGORITHM_API virtual void setTraversalMask( unsigned int mask );

          /*! \brief Get the current TraversalMask value
           *  \return mask The mask to be used.
           *  \sa setTraversalMask */
          DP_SG_ALGORITHM_API virtual unsigned int getTraversalMask() const;

          /*! \brief Set an override TraversalMask to be used with this Traverser.
           *  \param mask The mask to be used.
           *  \remarks This method provides a way to override the Object::TraversalMask that is used when determining
           *  whether to traverse scene graph objects.  This mask is OR'd with each dp::sg::core::Object's TraversalMask before
           *  being ANDed with the Traverser or ViewState's TraversalMask.  See Traverser::setTraversalMask for more info.
           *  \note The default TraversalMaskOverride is 0 so that it does not affect traversal.
           *  \sa getTraversalMaskOverride, ViewState::setTraversalMask, Object::setTraversalMask */
          DP_SG_ALGORITHM_API virtual void setTraversalMaskOverride( unsigned int mask );

          /*! \brief Get the current TraversalMask override
           *  \return mask The mask in use.
           *  \sa setTraversalMask */
          DP_SG_ALGORITHM_API virtual unsigned int getTraversalMaskOverride() const;

          /*! \brief Start traversal of the scene attached to the current ViewState
           *  \remarks If preApply is successful, the Scene and the ViewState are locked, the
           *  traversal is executed by calling doApply, the Scene and the ViewState are unlocked
           *  again, and postApply is called with \a root.
           *  \sa apply */
          DP_SG_ALGORITHM_API void apply( );

          /*! \brief Start traversal from a specified node.
           *  \param root The Node to start traversal with.
           *  \remarks If preApply is successful, the Scene and the ViewState are locked, the
           *  traversal is executed by calling doApply, the Scene and the ViewState are unlocked
           *  again, and postApply is called with \a root.
           *  \note Some traverser need a valid ViewState object which has to be set via setViewState.
           *  \sa apply */
          DP_SG_ALGORITHM_API void apply( const dp::sg::core::NodeSharedPtr & root );

          /*! \brief Start the traversal of a Scene.
           *  \param scene The Scene to traverse.
           *  \remarks This convenience function creates a temporary ViewState which is either a clone
                       of an ViewState set before or a newly created ViewState. Afterwards the \a scene
                       is attached to the ViewState and traversal starts. After traversal the orignal
                       ViewState will be restored.
           *  \sa apply */
          DP_SG_ALGORITHM_API void apply( const dp::sg::core::SceneSharedPtr & scene );

          /*! \brief Start the traversal of the scene attached to the ViewState
           *  \param scene A smart pointer to the ViewState containing a scene to traverse
           *  \remarks This convenience function temporary changed the ViewState of this traverser.
           *  \sa apply */
          DP_SG_ALGORITHM_API void apply( dp::sg::ui::ViewStateSharedPtr const& viewstate );

        protected:
          /*! \brief Default constructor.
           *  \note The constructor is protected, and hence, a Traverser object cannot be
           *  instantiated directly, but only by deriving from Traverser. */
          DP_SG_ALGORITHM_API Traverser();

          /*! \brief Protected destructor to prevent instantiation. */
          DP_SG_ALGORITHM_API virtual ~Traverser();

          /*! \brief Interface for initiating the traversal.
           *  \param root The Node to start traversal at.
           *  \remarks This function is called from the framework after the Scene and the ViewState,
           *  if set, are locked. The actual traversal is assumed to be done here.
           *  \sa preApply, postApply */
          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root ) = 0;

          /*! \brief Interface for performing the object traversal.
           *  \param p The Object to traverse.
           *  \remarks This function is called by the framework to really traverse an object.
           *  \sa traverseObject */
          DP_SG_ALGORITHM_API virtual void doTraverseObject( const dp::sg::core::ObjectSharedPtr & p ) = 0;

          /*! \brief Start traversing an Object.
           *  \param obj The Object to traverse.
           *  \remarks If the traversal was interrupted, the function immediately returns. Otherwise
           *  the virtual function doTraverseObject is called.
           *  \note The behavior is undefined if \a obj points to an invalid location.
           *  \sa doTraverseObject */
          DP_SG_ALGORITHM_API void traverseObject( const dp::sg::core::ObjectSharedPtr & obj );

          /*! \brief Notifier called from the framework directly before traversal starts.
           *  \param root The Node to be traversed.
           *  \return The default implementation always return \c true in order to continue traversal
           *  of \a root to proceed.
           *  \remarks Custom traversers might override this notifier to process some pre-traversal
           *  work. A customized preApply notifier should return \c true to force the traversal of
           *  the specified root node to proceed. Returning \c false prevents the specified root node
           *  from being traversed.\n
           *  At the time of call, a custom override can rely on the optional Scene and the optional 
           *  ViewState both _not_ being locked for read or write access by the calling thread. 
           *  \note If the function returns \c false, traversal of \a root is skipped.
           *  \note Not calling the base implementation might result in undefined runtime behavior.
           *  \par Example:
           *  \code
           *    bool MyTraverser::preApply( const dp::sg::core::NodeSharedPtr & root )
           *    {
           *      //  Do some pre-traversal bookholding or whatever
           *      //  ...
           *
           *      //  call the base implementation and return
           *      return( Traverser::preApply( root ) );
           *    }
           *  \endcode
           *  \sa postApply */
          DP_SG_ALGORITHM_API virtual bool preApply( const dp::sg::core::NodeSharedPtr & root );

          /*! \brief Notifier called from the framework directly after traversal.
           *  \param root The Node that was just traversed.
           *  \remarks Custom traversers might override this notifier to process some post-traversal
           *  work. It is strongly recommended for custom overrides to call the base implementation
           *  in their code.\n
           *  At the time of call, a custom override can rely on the optional Scene and the optional 
           *  ViewState both _not_ being locked for read or write access by the calling thread. 
           *  \note Not calling the base implementation might result in undefined runtime behavior.
           *  \sa preApply */
          DP_SG_ALGORITHM_API virtual void postApply( const dp::sg::core::NodeSharedPtr & root );

          /*! \brief Notifier called from the framework directly before a Group is traversed.
           *  \param grp A pointer to the constant Group to traverse.
           *  \return The default implementation always returns \c true, in order to continue
           *  traversal.
           *  \remarks Custom traversers can override this function for any tasks that need to be
           *  done directly before the Group \a grp is traversed.
           *  \note If the function returns \c false, traversal of \a grp is skipped.
           *  \note Not calling the base implementation might result in undefined runtime behavior.
           *  \sa postTraverseGroup */
          DP_SG_ALGORITHM_API virtual bool preTraverseGroup( const dp::sg::core::Group * grp );

          /*! \brief Notifier called from the framework directly after a Group has been traversed.
           *  \param grp A pointer to the constant Group that was just traversed.
           *  \remarks Custom traversers can override this function for any tasks that need to be
           *  done directly after the Group \a grp has been traversed.
           *  \note Not calling the base implementation might result in undefined runtime behavior.
           *  \sa preTraverseGroup */
          DP_SG_ALGORITHM_API virtual void postTraverseGroup( const dp::sg::core::Group * grp );

          /*! \brief Dispatcher to the particular overrides to traverse Group-derived Nodes.
           *  \param grp A pointer to the constant Group-derived Node to traverse.
           *  \remarks If preTraverseGroup() returned true, the lights contained in \a grp are
           *  traversed, and then, depending on the actual type a special traversal is performed. 
           *  After that, postTraverseGroup() is called.
           *  \sa preTraverseGroup, postTraverseGroup */
          template <typename T> void traverseGroup( const T * grp );

          /*! \brief Initiates GeoNode traversal.
           *  \param gnode The GeoNode to traverse.
           *  \remarks The function traversers through the MaterialData and the Primitive of the GeoNode.
           *  \sa handleGeoNode */
          DP_SG_ALGORITHM_API void traverseGeoNode( const dp::sg::core::GeoNode * gnode );

          DP_SG_ALGORITHM_API void traverseEffectData( const dp::sg::core::EffectData * ed );

          DP_SG_ALGORITHM_API void traverseParameterGroupData( const dp::sg::core::ParameterGroupData * pgd );

          /*! \brief Traversal of a Primitive.
           *  \param prim A pointer to the constant Primitive to traverse.
           */
          DP_SG_ALGORITHM_API void traversePrimitive( const dp::sg::core::Primitive * prim );

          DP_SG_ALGORITHM_API void traverseLockedObject( const dp::sg::core::Object * obj );

          /*! \brief Called right after traversal of an object.
           *  \param object A pointer to the read-locked object just traversed.
           *  \remarks The default implementation does nothing.
           */
          DP_SG_ALGORITHM_API virtual void postTraverseObject( const dp::sg::core::Object * object );

          /*! \brief Get the current texture unit associated with current TextureAttributeItem, while traversing a TextureAttribute.
           *  \return The current texture unit.
           *  \sa handleTextureAttribute, handleTextureAttributeItem */
          DP_SG_ALGORITHM_API unsigned int getCurrentTextureUnit() const;

          //! Member Function Table template
          template<typename T>
          class MemFunTbl
          {
            typedef void (T::*PMFN) ();         //!< pointer to member function type
          public:
            explicit MemFunTbl(size_t size);    //!< reserve 'size' table entries, to avoid frequend allocations
            PMFN operator[](size_t i) const;    //!< read-only access the function pointer stored at index i; behavior is undefined for invalid indices
            template <typename U>
            void addEntry(size_t i, U pmfn); //!< register function pointer pmfn at index i; former entry at i will by overridden
            bool testEntry(size_t i) const;     //!< test if entry at i is valid
          private:
            std::vector<PMFN> m_ftbl;           //!< simply use a plain vector as function table
          };

          // members
          unsigned int                    m_currentAttrib;
          MemFunTbl<Traverser>            m_mftbl;          //!< The table of handler functions.
          dp::sg::core::NodeSharedPtr     m_root;           //!< A pointer to the current root node.
          dp::sg::core::SceneSharedPtr    m_scene;          //!< A pointer to the current scene.
          dp::sg::ui::ViewStateSharedPtr  m_viewState;      //!< A pointer to the current view state.
          dp::sg::core::CameraSharedPtr   m_camera;

        private:
          DP_SG_ALGORITHM_API unsigned int     getObjectTraversalCode( const dp::sg::core::Object * object );  // NOTE: used inline, therefore needs to be exported via DP_SG_ALGORITHM_API

          // traverse overrides for Group derived (have only read-only access to objects)
          DP_SG_ALGORITHM_API void traverse( const dp::sg::core::Group * );
          DP_SG_ALGORITHM_API void traverse( const dp::sg::core::Switch * );

          unsigned int  m_currentTextureUnit;   //!< used in traverseTextureAttribute

          unsigned int  m_traversalMask;           //!< traversal mask to use
          unsigned int  m_traversalMaskOverride;   //!< traversal mask override to use
      };

      /************************************************************************/
      /* Traverser MemFunTbl                                                  */
      /************************************************************************/
      template<typename T>
      Traverser::MemFunTbl<T>::MemFunTbl(size_t size)
      { // table entries to avoid frequent allocations at addEntry
        m_ftbl.resize(size, NULL); 
      }

      template<typename T>
      bool Traverser::MemFunTbl<T>::testEntry(size_t i) const
      { // test if entry at i is a valid entry
        return (i < m_ftbl.size()) && !!m_ftbl[i]; 
      }

      template<typename T>
      typename Traverser::MemFunTbl<T>::PMFN Traverser::MemFunTbl<T>::operator[](size_t i) const
      { // read-only access to function pointer stored at index i
        DP_ASSERT(testEntry(i)); // undefined behavior for invalid entries
        return m_ftbl[i];
      }

      template<typename T>
      template<typename U>
      void Traverser::MemFunTbl<T>::addEntry(size_t i, U pmfn)
      { // register function pointer
        if ( m_ftbl.size() <= i )
        { // add 32 table entries
          m_ftbl.resize(i+0x20, NULL); 
        }
        m_ftbl[i]=*(PMFN*)&pmfn;
      }

      /************************************************************************/
      /* Traverser                                                            */
      /************************************************************************/

      template <typename T> 
      void Traverser::traverseGroup(const T * grp)
      {
        if ( preTraverseGroup(grp) )
        {
          // initiate group traversal
          traverse(grp);
          postTraverseGroup(grp);
        }
      }

      /*! \brief Base class providing an interface for read-only traversing of a given scene graph.
      *  \remarks The SharedTraverser class offers base functionality for all traversers that intend
      *  to provide read-only operations on a given scene graph. For all known and concrete \link
      *  dp::sg::core::Object Objects \endlink this class provides a set of overloadable handler routines,
      *  each ensuring that all components following the actual object in the graph hierarchy will be
      *  traversed correctly. Hence, for derived traversers it is recommended to always call the base
      *  implementation of an overridden handler routine for traversing purposes.\n
      *  To provide new read-only operations to be applied on known and concrete components or
      *  objects arranged in a scene graph, it is sufficient to derive a new traverser from either
      *  SharedTraverser or one of its derived traversers, and override the corresponding handler
      *  routines as needed.
      *  \sa ExclusiveTraverser */
      class SharedTraverser : public Traverser
      {
        public:
          /*! \brief Default constructor. */
          DP_SG_ALGORITHM_API SharedTraverser();

          /*! \brief Get the current TraversalMask value
           *  \return mask The mask to be used.
           *  \remarks If a ViewState has been set on this SharedTraverser, this method will return the
           *  ViewState's TraversalMask.  Otherwise, it will return the traversal mask set with Traverser::setTraversalMask.
           *  \sa setTraversalMask */
          DP_SG_ALGORITHM_API virtual unsigned int getTraversalMask() const;

        protected:
          /*! \brief Protected Destructor to prevent instantiation. */
          DP_SG_ALGORITHM_API virtual ~SharedTraverser();
      
          /*! \brief Template function to add a handler routine for a new class derived from Object.
           *  \param objectCode Object code to identify an object type at run-time.
           *  \param handler Specifies the address of the handler routine.
           *  \remarks The function registers the handler routine specified by \a handler to handle a
           *  concrete class, derived from Object, that is explicitly identified by \a objectCode.\n
           *  A handler routine must be a member function of a Traverser-derived class. It must return
           *  void and expect a pointer to the concrete object as a first parameter, and can have one
           *  additional (optional) argument of arbitrary type. A handler routine must not have a
           *  default argument! If a handler routine is intended to remain overloadable, it should be
           *  declared virtual.\n
           *  For derived classes intended to provide new operations for known objects arranged in a
           *  scene graph, it is strongly recommended not to add new handler routines for those objects
           *  but to override the corresponding handler routine. In this context, we refer to 'known
           *  objects' as objects that are known by the Traverser base class and for which a
           *  corresponding handler routine is provided.\n
           *  However, \c addObjectHandler is indispensable if newly invented objects, and hence, 
           *  objects that are unknown by the Traverser base class, need to be considered for
           *  traversing.
           *  \note Any previously registered handler routine for the specified object code will be
           *  overridden by the new handler.
           *  \par Example:
           *  \code
           *    addObjectHandler( 
           *      OC_NEWGROUP   // object code for the new object NewGroup
           *                    // this must be provided by the author of NewGroup
           *      , &MySharedTraverser::handleNewGroup // handler function  
           *    );    
           *  \endcode */
          template <typename T, typename U> 
          void addObjectHandler( unsigned int objectCode, void (T::*handler)(const U*) );

          /*! \brief Template function to add a handler routine for a new class derived from Object.
           *  \param objectCode Object code to identify an object type at run-time.
           *  \param handler Specifies the address of the handler routine.
           *  \remarks The function registers the handler routine specified by \a handler to handle a
           *  concrete class, derived from Object, that is explicitly identified by \a objectCode.\n
           *  A handler routine must be a member function of a Traverser-derived class. It must return
           *  void and expect a pointer to the concrete object as a first parameter, and can have one
           *  additional (optional) argument of arbitrary type. A handler routine must not have a
           *  default argument! If a handler routine is intended to remain overloadable, it should be
           *  declared virtual.\n
           *  For derived classes intended to provide new operations for known objects arranged in a
           *  scene graph, it is strongly recommended not to add new handler routines for those objects
           *  but to override the corresponding handler routine. In this context, we refer to 'known
           *  objects' as objects that are known by the Traverser base class and for which a
           *  corresponding handler routine is provided.\n
           *  However, \c addObjectHandler is indispensable if newly invented objects, and hence, 
           *  objects that are unknown by the Traverser base class, need to be considered for
           *  traversing.
           *  \note Any previously registered handler routine for the specified object code will be
           *  overridden by the new handler.
           *  \par Example:
           *  \code
           *    addObjectHandler( 
           *      OC_NEWGROUP   // object code for the new object NewGroup
           *                    // this must be provided by the author of NewGroup
           *      , &MySharedTraverser::handleNewGroup // handler function  
           *    );    
           *  \endcode */
          template <typename T, typename U, typename V> 
          void addObjectHandler( unsigned int objectCode, void (T::*handler)(const U*, V) );

          /*! \brief Override of the traversal initiating interface.
           *  \param root The Node to start traversal at.
           *  \remarks This function is called from the framework after the Scene and the ViewState,
           *  if set, are locked. The actual traversal is done here.
           *  \sa preApply, postApply, unlockViewState */
          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

          /*! \brief Function for performing the object traversal.
           *  \param p The Object to traverse.
           *  \remarks This function is called by the framework to really traverse an object.
           *  \sa traverseObject */
          DP_SG_ALGORITHM_API virtual void doTraverseObject( const dp::sg::core::ObjectSharedPtr & p );

          /*! \brief Handler function for a ParallelCamera.
           *  \param camera A pointer to the read-locked ParallelCamera being traversed.
           *  \remarks This function is called from the framework whenever a ParallelCamera is
           *  encountered on traversal. The ParallelCamera \a camera is already read-locked.\n
           *  The base implementation traverses through all the head lights of this camera.
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa handleMatrixCamera, handlePerspectiveCamera */
          DP_SG_ALGORITHM_API virtual void handleParallelCamera( const dp::sg::core::ParallelCamera * camera );

          /*! \brief Handler function for a PerspectiveCamera.
           *  \param camera A pointer to the read-locked PerspectiveCamera being traversed.
           *  \remarks This function is called from the framework whenever a PerspectiveCamera is
           *  encountered on traversal. The PerspectiveCamera \a camera is already read-locked.\n
           *  The base implementation traverses through all the head lights of this camera.
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa handleMatrixCamera, handleParallelCamera */
          DP_SG_ALGORITHM_API virtual void handlePerspectiveCamera( const dp::sg::core::PerspectiveCamera * camera );

          /*! \brief Handler function for a MatrixCamera.
           *  \param camera A pointer to the read-locked MatrixCamera being traversed.
           *  \remarks This function is called from the framework whenever a MatrixCamera is
           *  encountered on traversal. The MatrixCamera \a camera is already read-locked.\n
           *  The base implementation traverses through all the head lights of this camera.
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa handleParallelCamera, handlePerspectiveCamera */
          DP_SG_ALGORITHM_API virtual void handleMatrixCamera( const dp::sg::core::MatrixCamera * camera );

          /*! \brief Handler function for a Group.
           *  \param group A pointer to the read-locked Group being traversed.
           *  \remarks This function is called from the framework whenever a Group is encountered on
           *  traversal. The Group \a group is already read-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup */
          DP_SG_ALGORITHM_API virtual void handleGroup( const dp::sg::core::Group * group );

          /*! \brief Handler function for a Transform.
           *  \param trafo A pointer to the read-locked Transform being traversed.
           *  \remarks This function is called from the framework whenever a Transform is encountered
           *  on traversal. The Transform \a trafo is already read-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup, handleGroup */
          DP_SG_ALGORITHM_API virtual void handleTransform( const dp::sg::core::Transform * trafo );

          /*! \brief Handler function for an LOD.
           *  \param lod A pointer to the read-locked LOD being traversed.
           *  \remarks This function is called from the framework whenever an LOD is encountered on
           *  traversal. The LOD \a lod is already read-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup, handleGroup */
          DP_SG_ALGORITHM_API virtual void handleLOD( const dp::sg::core::LOD * lod );

          /*! \brief Handler function for a Switch.
           *  \param swtch A pointer to the read-locked Switch being traversed.
           *  \remarks This function is called from the framework whenever a Switch is encountered on
           *  traversal. The Switch \a switch is already read-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup, handleGroup */
          DP_SG_ALGORITHM_API virtual void handleSwitch( const dp::sg::core::Switch * swtch );

          /*! \brief Handler function for a Billboard.
           *  \param billboard A pointer to the read-locked Billboard being traversed.
           *  \remarks This function is called from the framework whenever a Billboard is encountered on
           *  traversal. The Billboard \a billboard is already read-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup, handleGroup */
          DP_SG_ALGORITHM_API virtual void handleBillboard( const dp::sg::core::Billboard * billboard );

          DP_SG_ALGORITHM_API virtual void handleLightSource( const dp::sg::core::LightSource * light );

          /*! \brief Handler function for a GeoNode.
           *  \param gnode A pointer to the read-locked GeoNode being traversed.
           *  \remarks This function is called from the framework whenever a GeoNode is encountered on
           *  traversal. The GeoNode \a gnode is already read-locked.\n
           *  The base implementation just calls Traverser::traverseGeoNode().
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGeoNode */
          DP_SG_ALGORITHM_API virtual void handleGeoNode( const dp::sg::core::GeoNode * gnode );

          /*! \brief Handler function for a Primitive
           *  \param primitive A pointer to the read-locked Primitive being traversed.
           *  \remarks This function is called from the framework whenever a Primitive is encountered on
           *  traversal. The Primitive \a primitive is already read-locked.\n
           *  The base implementation just calls Traverser::traversePrimitive().
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traversePrimitive */
          DP_SG_ALGORITHM_API virtual void handlePrimitive( const dp::sg::core::Primitive * primitive );

          DP_SG_ALGORITHM_API virtual void handleEffectData( const dp::sg::core::EffectData * ed );

          DP_SG_ALGORITHM_API virtual void handleParameterGroupData( const dp::sg::core::ParameterGroupData * pgd );
          DP_SG_ALGORITHM_API virtual void handleSampler( const dp::sg::core::Sampler * p );

          /*! \brief Handler function for an IndexSet.
           *  \param iset A pointer to the read-locked IndexSet being traversed.
           *  \remarks This function is called from the framework whenever an IndexSet is encountered
           *  on traversal. The IndexSet \a p is already read-locked.\n
           *  The base implementation just does nothing.
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue traversal. */
          DP_SG_ALGORITHM_API virtual void handleIndexSet( const dp::sg::core::IndexSet * iset );

          /*! \brief Handler function for a VertexAttributeSet.
           *  \param vas A pointer to the read-locked VertexAttributeSet being traversed.
           *  \remarks This function is called from the framework whenever a VertexAttributeSet is
           *  encountered on traversal. The VertexAttributeSet \a vas is already read-locked.\n
           *  The base implementation just does nothing.
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal. */
          DP_SG_ALGORITHM_API virtual void handleVertexAttributeSet( const dp::sg::core::VertexAttributeSet *vas );

          /*! \brief Function for common handling of Camera classes.
           *  \param camera A pointer to the read-locked Camera being traversed.
           *  \remarks This function is called from the framework whenever a Camera is encountered
           *  on traversal. The Camera \a camera is already read-locked.\n
           *  The base implementation traverses the headlights of \a camera.
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa LightSource */
          DP_SG_ALGORITHM_API virtual void traverseCamera( const dp::sg::core::Camera * camera );

          /*! \brief Function for common handling of FrustumCamera classes.
           *  \param camera A pointer to the read-locked FrustumCamera being traversed.
           *  \remarks This function is called from the framework whenever a FrustumCamera is encountered
           *  on traversal. The FrustumCamera \a camera is already read-locked.\n
           *  The base implementation traverses the headlights of \a camera.
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa LightSource */
          DP_SG_ALGORITHM_API virtual void traverseFrustumCamera( const dp::sg::core::FrustumCamera * camera );

          /*! \brief Function for common handling of LightSource classes.
           *  \param light A pointer to the read-locked LightSource being traversed.
           *  \remarks This function is called from the framework whenever a LightSource is encountered
           *  on traversal. The LightSource \a light is already read-locked.\n
           *  The base implementation traverses the Animation associated with \a light, if there is one.
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa TrafoAnimation */
          DP_SG_ALGORITHM_API virtual void traverseLightSource( const dp::sg::core::LightSource * light );

          /*! \brief Function for common handling of Primitive classes.
           *  \param p A pointer to the read-locked Primitive being traversed.
           *  \remarks This function is called from the framework whenever a Primitive is encountered
           *  on traversal. The Primitive \a p is already read-locked.\n
           *  The base implementation first traverses the Skin, if there is one, and then the
           *  VertexAttributeSet of \a p, if there is one. 
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa VertexAttributeSet, Skin */
          DP_SG_ALGORITHM_API virtual void traversePrimitive( const dp::sg::core::Primitive * p );
      };

    /*! \brief Base class providing an interface for read-write traversing of a given scene graph.
     *  \remarks The ExclusiveTraverser class offers base functionality for all traversers that intend
     *  to provide read-write operations on a given scene graph. For all known and concrete \link
     *  dp::sg::core::Object Objects \endlink this class provides a set of overloadable handler routines,
     *  each ensuring that all components following the actual object in the graph hierarchy will be
     *  traversed correctly. Hence, for derived traversers it is recommended to always call the base
     *  implementation of an overridden handler routine for traversing purposes.\n
     *  To provide new read-write operations to be applied on known and concrete components or
     *  objects arranged in a scene graph, it is sufficient to derive a new traverser from either
     *  ExclusiveTraverser or one of its derived traversers, and override the corresponding handler
     *  routines as needed.
     *  \sa SharedTraverser */
      class ExclusiveTraverser : public Traverser
      {
        public:
          /*! \brief Default constructor. */
          DP_SG_ALGORITHM_API ExclusiveTraverser();

          //! Query if the latest traversal did modify the tree.
          /** \returns \c true if the tree was modified, otherwise false */
          DP_SG_ALGORITHM_API bool getTreeModified( void ) const;

          /*! \brief Get the current TraversalMask value
           *  \return mask The mask to be used.
           *  \remarks If a ViewState has been set on this ExclusiveTraverser, this method will return the
           *  ViewState's TraversalMask.  Otherwise, it will return the traversal mask set with Traverser::setTraversalMask.
           *  \sa setTraversalMask */
          DP_SG_ALGORITHM_API virtual unsigned int getTraversalMask() const;

        protected:
          /*! \brief Protected Destructor to prevent instantiation. */
          DP_SG_ALGORITHM_API virtual ~ExclusiveTraverser();

          /*! \brief Template function to add a handler routine for a new class derived from Object.
           *  \param objectCode Object code to identify an object type at run-time.
           *  \param handler Specifies the address of the handler routine.
           *  \remarks The function registers the handler routine specified by \a handler to handle a
           *  concrete class, derived from Object, that is explicitly identified by \a objectCode.\n
           *  A handler routine must be a member function of a Traverser-derived class. It must return
           *  void and expect a pointer to the concrete object as a first parameter, and can have one
           *  additional (optional) argument of arbitrary type. A handler routine must not have a
           *  default argument! If a handler routine is intended to remain overloadable, it should be
           *  declared virtual.\n
           *  For derived classes intended to provide new operations for known objects arranged in a
           *  scene graph, it is strongly recommended not to add new handler routines for those objects
           *  but to override the corresponding handler routine. In this context, we refer to 'known
           *  objects' as objects that are known by the Traverser base class and for which a
           *  corresponding handler routine is provided.\n
           *  However, \c addObjectHandler is indispensable if newly invented objects, and hence, 
           *  objects that are unknown by the Traverser base class, need to be considered for
           *  traversing.
           *  \note Any previously registered handler routine for the specified object code will be
           *  overridden by the new handler.
           *  \par Example:
           *  \code
           *    addObjectHandler( 
           *      OC_NEWGROUP   // object code for the new object NewGroup
           *                    // this must be provided by the author of NewGroup
           *      , &MyExclusiveTraverser::handleNewGroup // handler function  
           *    );    
           *  \endcode */
          template <typename T, typename U> 
          void addObjectHandler( unsigned int objectCode, void (T::*handler)(U*) );

          /*! \brief Template function to add a handler routine for a new class derived from Object.
           *  \param objectCode Object code to identify an object type at run-time.
           *  \param handler Specifies the address of the handler routine.
           *  \remarks The function registers the handler routine specified by \a handler to handle a
           *  concrete class, derived from Object, that is explicitly identified by \a objectCode.\n
           *  A handler routine must be a member function of a Traverser-derived class. It must return
           *  void and expect a pointer to the concrete object as a first parameter, and can have one
           *  additional (optional) argument of arbitrary type. A handler routine must not have a
           *  default argument! If a handler routine is intended to remain overloadable, it should be
           *  declared virtual.\n
           *  For derived classes intended to provide new operations for known objects arranged in a
           *  scene graph, it is strongly recommended not to add new handler routines for those objects
           *  but to override the corresponding handler routine. In this context, we refer to 'known
           *  objects' as objects that are known by the Traverser base class and for which a
           *  corresponding handler routine is provided.\n
           *  However, \c addObjectHandler is indispensable if newly invented objects, and hence, 
           *  objects that are unknown by the Traverser base class, need to be considered for
           *  traversing.
           *  \note Any previously registered handler routine for the specified object code will be
           *  overridden by the new handler.
           *  \par Example:
           *  \code
           *    addObjectHandler( 
           *      OC_NEWGROUP   // object code for the new object NewGroup
           *                    // this must be provided by the author of NewGroup
           *      , &MyExclusiveTraverser::handleNewGroup // handler function  
           *    );    
           *  \endcode */
          template <typename T, typename U, typename V> 
          void addObjectHandler( unsigned int objectCode, void (T::*handler)(U*, V) );

          /*! \brief Override of the traversal initiating interface.
           *  \param root The Node to start traversal at.
           *  \remarks This function is called from the framework after the Scene and the ViewState,
           *  if set, are locked. The actual traversal is done here.
           *  \sa preApply, postApply */
          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

          /*! \brief Function for performing the object traversal.
           *  \param p The Object to traverse.
           *  \remarks This function is called by the framework to really traverse an object.
           *  \sa traverseObject */
          DP_SG_ALGORITHM_API virtual void doTraverseObject( const dp::sg::core::ObjectSharedPtr & p );

          /*! \brief Handler function for a ParallelCamera.
           *  \param camera A pointer to the write-locked ParallelCamera being traversed.
           *  \remarks This function is called from the framework whenever a ParallelCamera is
           *  encountered on traversal. The ParallelCamera \a camera is already write-locked.\n
           *  The base implementation traverses through all the head lights of this camera.
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa handleMatrixCamera, handlePerspectiveCamera */
          DP_SG_ALGORITHM_API virtual void handleParallelCamera( dp::sg::core::ParallelCamera * camera );

          /*! \brief Handler function for a PerspectiveCamera.
           *  \param camera A pointer to the write-locked PerspectiveCamera being traversed.
           *  \remarks This function is called from the framework whenever a PerspectiveCamera is
           *  encountered on traversal. The PerspectiveCamera \a camera is already write-locked.\n
           *  The base implementation traverses through all the head lights of this camera.
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa handleMatrixCamera, handleParallelCamera */
          DP_SG_ALGORITHM_API virtual void handlePerspectiveCamera( dp::sg::core::PerspectiveCamera * camera );

          /*! \brief Handler function for a MatrixCamera.
           *  \param camera A pointer to the write-locked MatrixCamera being traversed.
           *  \remarks This function is called from the framework whenever a MatrixCamera is
           *  encountered on traversal. The MatrixCamera \a camera is already read-locked.\n
           *  The base implementation traverses through all the head lights of this camera.
           *  \note When this function is overridden by a traverser derived from SharedTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa handleParallelCamera, handlePerspectiveCamera */
          DP_SG_ALGORITHM_API virtual void handleMatrixCamera( dp::sg::core::MatrixCamera * camera );

          /*! \brief Handler function for a Group.
           *  \param group A pointer to the write-locked Group being traversed.
           *  \remarks This function is called from the framework whenever a Group is encountered on
           *  traversal. The Group \a group is already write-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup */
          DP_SG_ALGORITHM_API virtual void handleGroup( dp::sg::core::Group * group );

          /*! \brief Handler function for a Transform.
           *  \param trafo A pointer to the write-locked Transform being traversed.
           *  \remarks This function is called from the framework whenever a Transform is encountered
           *  on traversal. The Transform \a trafo is already write-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup, handleGroup */
          DP_SG_ALGORITHM_API virtual void handleTransform( dp::sg::core::Transform * trafo );

          /*! \brief Handler function for an LOD.
           *  \param lod A pointer to the write-locked LOD being traversed.
           *  \remarks This function is called from the framework whenever an LOD is encountered on
           *  traversal. The LOD \a lod is already write-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup, handleGroup */
          DP_SG_ALGORITHM_API virtual void handleLOD( dp::sg::core::LOD * lod );

          /*! \brief Handler function for a Switch.
           *  \param swtch A pointer to the write-locked Switch being traversed.
           *  \remarks This function is called from the framework whenever a Switch is encountered on
           *  traversal. The Switch \a switch is already write-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup, handleGroup */
          DP_SG_ALGORITHM_API virtual void handleSwitch( dp::sg::core::Switch * swtch );

          /*! \brief Handler function for a Billboard.
           *  \param billboard A pointer to the write-locked Billboard being traversed.
           *  \remarks This function is called from the framework whenever a Billboard is encountered on
           *  traversal. The Billboard \a billboard is already write-locked.\n
           *  The base implementation just calls Traverser::traverseGroup().
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGroup, handleGroup */
          DP_SG_ALGORITHM_API virtual void handleBillboard( dp::sg::core::Billboard * billboard );

          DP_SG_ALGORITHM_API virtual void handleLightSource( dp::sg::core::LightSource * light );

          /*! \brief Handler function for a GeoNode.
           *  \param gnode A pointer to the write-locked GeoNode being traversed.
           *  \remarks This function is called from the framework whenever a GeoNode is encountered on
           *  traversal. The GeoNode \a gnode is already write-locked.\n
           *  The base implementation just calls Traverser::traverseGeoNode().
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traverseGeoNode */
          DP_SG_ALGORITHM_API virtual void handleGeoNode( dp::sg::core::GeoNode * gnode );

          /*! \brief Handler function for a Primitive
           *  \param primitive A pointer to the write-locked Primitive being traversed.
           *  \remarks This function is called from the framework whenever a Primitive is encountered on
           *  traversal. The Primitive \a primitive is already write-locked.\n
           *  The base implementation just calls Traverser::traversePrimitive().
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa traversePrimitive */
          DP_SG_ALGORITHM_API virtual void handlePrimitive( dp::sg::core::Primitive * primitive );

          DP_SG_ALGORITHM_API virtual void handleEffectData( dp::sg::core::EffectData * ed );

          DP_SG_ALGORITHM_API virtual void handleParameterGroupData( dp::sg::core::ParameterGroupData * pgd );
          DP_SG_ALGORITHM_API virtual void handleSampler( dp::sg::core::Sampler * p );

          /*! \brief Handler function for an IndexSet.
           *  \param iset A pointer to the write-locked IndexSet being traversed.
           *  \remarks This function is called from the framework whenever an IndexSet is encountered
           *  on traversal. The IndexSet \a p is already write-locked.\n
           *  The base implementation just does nothing.
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it is
           *  recommended to always call the base class implementation in order to properly continue traversal. */
          DP_SG_ALGORITHM_API virtual void handleIndexSet( dp::sg::core::IndexSet * iset );

          /*! \brief Handler function for a VertexAttributeSet.
           *  \param vas A pointer to the write-locked VertexAttributeSet being traversed.
           *  \remarks This function is called from the framework whenever a VertexAttributeSet is
           *  encountered on traversal. The VertexAttributeSet \a vas is already write-locked.\n
           *  The base implementation just does nothing.
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal. */
          DP_SG_ALGORITHM_API virtual void handleVertexAttributeSet( dp::sg::core::VertexAttributeSet *vas );

          /*! \brief Function for common handling of Camera classes.
           *  \param camera A pointer to the write-locked Camera being traversed.
           *  \remarks This function is called from the framework whenever a Camera is encountered
           *  on traversal. The Camera \a camera is already write-locked.\n
           *  The base implementation traverses the headlights of \a camera.
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa LightSource */
          DP_SG_ALGORITHM_API virtual void traverseCamera( dp::sg::core::Camera * camera );

          /*! \brief Function for common handling of FrustumCamera classes.
           *  \param camera A pointer to the write-locked FrustumCamera being traversed.
           *  \remarks This function is called from the framework whenever a FrustumCamera is encountered
           *  on traversal. The FrustumCamera \a camera is already write-locked.\n
           *  The base implementation traverses the headlights of \a camera.
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa LightSource */
          DP_SG_ALGORITHM_API virtual void traverseFrustumCamera( dp::sg::core::FrustumCamera * camera );

          /*! \brief Function for common handling of LightSource classes.
           *  \param light A pointer to the write-locked LightSource being traversed.
           *  \remarks This function is called from the framework whenever a LightSource is encountered
           *  on traversal. The LightSource \a light is already write-locked.\n
           *  The base implementation traverses the Animation associated with \a light, if there is one.
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa TrafoAnimation */
          DP_SG_ALGORITHM_API virtual void traverseLightSource( dp::sg::core::LightSource * light );

          /*! \brief Function for common handling of Primitive classes.
           *  \param p A pointer to the write-locked Primitive being traversed.
           *  \remarks This function is called from the framework whenever a Primitive is encountered
           *  on traversal. The Primitive \a p is already write-locked.\n
           *  The base implementation first traverses the Skin, if there is one, and then the
           *  VertexAttributeSet of \a p, if there is one. 
           *  \note When this function is overridden by a traverser derived from ExclusiveTraverser, it
           *  is recommended to always call the base class implementation in order to properly continue
           *  traversal.
           *  \sa VertexAttributeSet, Skin */
          DP_SG_ALGORITHM_API virtual void traversePrimitive( dp::sg::core::Primitive * p );

          DP_SG_ALGORITHM_API void setTreeModified();

        private:
          bool                  m_treeModified;
      };

      //--- Inline functions ------------------

      inline void Traverser::postTraverseObject( const dp::sg::core::Object * object )
      {
        DP_ASSERT( object );
      }

      inline void Traverser::setTraversalMask( unsigned int mask )
      {
        m_traversalMask = mask;
      }

      inline void Traverser::setTraversalMaskOverride( unsigned int mask )
      {
        m_traversalMaskOverride = mask;
      }

      inline unsigned int Traverser::getTraversalMaskOverride() const
      {
        return m_traversalMaskOverride;
      }

      inline unsigned int Traverser::getTraversalMask() const
      {
        return m_traversalMask;
      }

      inline unsigned int SharedTraverser::getTraversalMask() const
      {
        return( m_viewState ? m_viewState->getTraversalMask() : Traverser::getTraversalMask() );
      }

      inline unsigned int ExclusiveTraverser::getTraversalMask() const
      {
        return( m_viewState ? m_viewState->getTraversalMask() : Traverser::getTraversalMask() );
      }

      inline void Traverser::traverseObject( const dp::sg::core::ObjectSharedPtr & obj )
      {
        doTraverseObject( obj );
      }

      inline void Traverser::traverseLockedObject(const dp::sg::core::Object * obj)
      {
        unsigned int oc = getObjectTraversalCode( obj );
        if ( (dp::sg::core::OC_INVALID != oc) && 
             (( obj->getTraversalMask() | getTraversalMaskOverride() ) & getTraversalMask()) 
           )
        {
          (this->*(void (Traverser::*)(const dp::sg::core::Object*))m_mftbl[oc])(obj);
        }
      }

      inline unsigned int Traverser::getCurrentTextureUnit() const
      {
        return( m_currentTextureUnit );
      }

      template <typename T, typename U> 
      inline void SharedTraverser::addObjectHandler(unsigned int objectCode, void (T::*handler)(const U*))
      {
        DP_STATIC_ASSERT(( boost::is_base_of<SharedTraverser,T>::value ));
        m_mftbl.addEntry(objectCode, handler);
      }

      template <typename T, typename U, typename V> 
      inline void SharedTraverser::addObjectHandler(unsigned int objectCode, void (T::*handler)(const U*, V))
      {
        DP_STATIC_ASSERT(( boost::is_base_of<SharedTraverser,T>::value ));
        m_mftbl.addEntry(objectCode, handler);
      }

      inline void SharedTraverser::doTraverseObject( const dp::sg::core::ObjectSharedPtr & p )
      {
        traverseLockedObject( p.getWeakPtr() );
      }

      template <typename T, typename U> 
      inline void ExclusiveTraverser::addObjectHandler(unsigned int objectCode, void (T::*handler)(U*))
      {
        DP_STATIC_ASSERT(( boost::is_base_of<ExclusiveTraverser,T>::value ));
        m_mftbl.addEntry(objectCode, handler);
      }

      template <typename T, typename U, typename V> 
      inline void ExclusiveTraverser::addObjectHandler(unsigned int objectCode, void (T::*handler)(U*, V))
      {
        DP_STATIC_ASSERT(( boost::is_base_of<ExclusiveTraverser,T>::value ));
        m_mftbl.addEntry(objectCode, handler);
      }

      inline void ExclusiveTraverser::doTraverseObject( const dp::sg::core::ObjectSharedPtr & p )
      {
        traverseLockedObject( p.getWeakPtr() );
      }

      inline void ExclusiveTraverser::setTreeModified()
      {
        m_treeModified = true;
      }

      inline bool ExclusiveTraverser::getTreeModified( void ) const
      {
        return( m_treeModified );
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
