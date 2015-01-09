// Copyright NVIDIA Corporation 2012
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
/** @file */

#include <typeinfo>
#include <dp/sg/core/nvsgapi.h> // commonly used stuff
#include <dp/math/Boxnt.h>
#include <dp/math/Spherent.h>
#include <dp/sg/core/ConstIterator.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/Event.h>
#include "dp/sg/core/HandledObject.h"
#include <dp/util/BitMask.h>
#include <dp/util/HashGenerator.h>
#include <dp/util/Types.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief 64-bit save DataID.
        * \remarks
        * The DataID is used to uniquely identify an Object's embedded data. 
        * \sa Object::getDataID
        */
      typedef unsigned long long DataID;

      /*! \brief Object Codes for DPSg Object Type Identification 
        */
      enum ObjectCode
      {
          OC_BILLBOARD            //!< Billboard
        , OC_CLIPPLANE            //!< ClipPlane
        , OC_EFFECT_DATA          //!< EffectData
        , OC_GEONODE              //!< GeoNode
        , OC_GROUP                //!< Group
        , OC_INDEX_SET            //!< IndexSet
        , OC_LIGHT_SOURCE         //!< LightSource
        , OC_LOD                  //!< LOD
        , OC_MATRIXCAMERA         //!< Matrix Camera
        , OC_PARALLELCAMERA       //!< ParallelCamera
        , OC_PARAMETER_GROUP_DATA //!< ParameterGroupData
        , OC_PERSPECTIVECAMERA    //!< PerspectiveCamera
        , OC_PRIMITIVE            //!< Generic Primitive
        , OC_SAMPLER              //!< Sampler
        , OC_SCENE                //!< Scene
        , OC_SWITCH               //!< Switch
        , OC_TRANSFORM            //!< Transform
        , OC_VERTEX_ATTRIBUTE_SET //!< VertexAttributeSet
        
        , OC_INVALID              //!< invalid object code
      };

      DP_SG_CORE_API std::string objectCodeToName( ObjectCode oc );

      /*! \brief Serves as base class for all traversable objects.
       * \par Namespace: dp::sg::core
       * \remarks
       * Besides being reference counted and providing the read/write locking interface
       * an Object has an object code that identifies its concrete class type. Every concrete
       * class derived from Object has to set an unique object code in its constructors. This object code
       * is used in the Traverser to determine the concrete type of the Object. If there is a handler
       * function registered for that specific object code, this handler is called. Otherwise, a higher
       * level object code is queried, that also has to be provided by the concrete classes derived from
       * Object, until an object code is encountered a handler function is registered for. */
      class Object : public HandledObject, public dp::util::Observer
      {
      public:
        class Event;

      public:
        enum
        {
          /*! \brief Flags the Object as always visible
           * \remarks A visible-flagged Object will not be considered by the
           * frameworks cull step. By default, this hint is not set. */
          DP_SG_HINT_ALWAYS_VISIBLE                    = BIT0,    // always visible

          /*! \brief Flags the Object as always invisible
           * \remarks An invisible-flagged Object will not be processed by the
           * render framework.
           * By default, this hint is not set. */
          DP_SG_HINT_ALWAYS_INVISIBLE                  = BIT1,    // never visible
          DP_SG_HINT_VISIBILITY_MASK = DP_SG_HINT_ALWAYS_VISIBLE | DP_SG_HINT_ALWAYS_INVISIBLE,

          /*! \brief Flags the Object not to cast shadows
           * \remarks Shadow algorithms can consider this hint for optimized 
           * strategy.
           * The built-in shaders will ignore intersections 
           * with geometry with this hint set. */
          DP_SG_HINT_NO_SHADOW_CASTER                  = BIT2,
 
          /*! \brief Flags the Object not to receive shadows
           * \remarks Shadow algorithms can consider this hint for optimized 
           * strategy. 
           * By default, this hint is not set. */
          DP_SG_HINT_NO_SHADOW_RECEIVER                = BIT3,

          /*! \brief Flags the Object as overlay Object
           * \remarks An overlay Object will always be rendered on top of other
           * non-overlay Objects independent of the Objects z-depth. By default,
           * this hint is not set. */
          DP_SG_HINT_OVERLAY                           = BIT4,

          /*! \brief Flags the Object as frequently altering
           * \remarks The processing framework considers dynamic-flagged objects
           * different from ordinary non-dynamic Objects to yield optimized render 
           * performance. By default, an Object is not flagged dynamic. */
          DP_SG_HINT_DYNAMIC                           = BIT5,

          /*! \brief Flags the Object as not being considered for clipping
           * \remarks During rendering, active clip planes have no effect on Objects 
           * that have the DP_SG_HINT_DONT_CLIP bit set. This hint only applies to 
           * Node-derived Objects. */
          DP_SG_HINT_DONT_CLIP                         = BIT6,

          DP_SG_HINT_ASSEMBLY                          = BIT10,

          DP_SG_HINT_LAST_HINT                         = DP_SG_HINT_ASSEMBLY,

          /*! \brief All possible hints */
          DP_SG_HINT_ALL_HINTS                         = (DP_SG_HINT_LAST_HINT << 1) - 1
        };

        enum
        {
          // bounding volume state
          NVSG_BOUNDING_BOX                   = DP_SG_HINT_LAST_HINT          << 1,
          NVSG_BOUNDING_SPHERE                = NVSG_BOUNDING_BOX             << 1,
      
          NVSG_BOUNDING_VOLUMES               = NVSG_BOUNDING_BOX | 
                                                NVSG_BOUNDING_SPHERE,

          // hash key state
          NVSG_HASH_KEY                       = NVSG_BOUNDING_SPHERE    << 1
        };

        class Event;

      public:
        /*! \brief Destructs an Object. 
          */
        DP_SG_CORE_API virtual ~Object();

        /*! \brief Returns the object code, which is unique per object type.
          * \return
          * The function returns the object code enum for this object.
          * \remarks
          * The object code, which is unique per object type, is used for fast object type 
          * identification at runtime. A dp::sg::algorithm::Traverser, for example, uses the object code to 
          * fast lookup the handler function for a particular object type. 
          * \n\n
          * Object-derived classes must override the protected member 
          * \link Object::m_objectCode m_objectCode \endlink, with the correct value for the
          * respective type. It is recommended to do this at instantiation time of the customized object.
          * \sa \ref howtoderiveanvsgobject, Object::getHigherLevelObjectCode
          */
        ObjectCode getObjectCode() const;

        /*! \brief Assigns new content from another Object. 
          * \param
          * rhs Reference to an Object from which to assign the new content.
          * \return
          * A reference to this object.
          * \remarks
          * The assignment operator unreferences the old content before assigning the new content. The new
          * content will be a deep-copy of the content of right-hand-sided object. 
          */
        DP_SG_CORE_API Object & operator=(const Object & rhs);

        /*! \brief Returns the higher-level object code for a given object code.
          * \param
          * oc %Object code for which to retrieve the associated higher-level object code.
          * \return
          * The default implementation returns OC_INVALID, which causes the traverser framework to 
          * immediately proceed without handling the object when it comes across an unknown object code.
          * \remarks
          * This function will be called from the object traverser framework if the object code \a oc 
          * is unknown. This would be the case if for a custom object a certain traverser did not 
          * register an appropriate handler function.
          * \n\n
          * The framework expects this function to return a higher-level object code in terms of a 
          * custom object hierarchy. The function will be repeatedly called with the returned 
          * higher-level object code, until either a known object code or OC_INVALID will be returned. 
          * That is - by repeatedly calling this function, the traverser framework moves up the custom 
          * object hierarchy to find an object for which a handler function was registered. If the 
          * traverser framework recognizes a known object code, it calls the appropriate handler 
          * function and proceeds. If OC_INVALID was returned, the traverser framework proceeds without 
          * handling the object.
          * \n\n
          * The runtime behavior is undefined if the function returns an object code of an object that 
          * is not part of the object hierarchy of the primary object!
          * \n\n
          * The framework might end up in a deadlock if the function never returns either OC_INVALID or 
          * an object code of a concrete object known by the traverser!
          * \sa Object::getObjectCode
          */
        DP_SG_CORE_API virtual ObjectCode getHigherLevelObjectCode(ObjectCode oc) const; 

        /*! \brief Returns whether this object shares its embedded data with other objects.
          * \return
          * The function should return \c true if the object shares its embedded data with other 
          * objects. Otherwise, the function should return \c false.
          * \n\n
          * The default implementation always returns \c false.
          * \remarks
          * For Object-derived classes that are capable of sharing their embedded data with other 
          * objects, it is recommended to override this function. For all other objects, the default
          * implementation will be sufficient.
          * \n\n
          * An object, capable of data sharing, shares its data with another object if it was either 
          * instantiated as a copy of the other object, or if its content was re-assigned to by the 
          * content of the other object using the appropriate assignment operator.
          * \n\n
          * Only a few classes of the SceniX core implement data sharing. These mainly are classes where
          * data sharing probably could save a lot of memory because of the large amount of data these
          * classes might contain. VertexAttributeSet is an example of \ref coreobjects that
          * implement data sharing.
          * \sa Object::getDataID
          */
        DP_SG_CORE_API virtual bool isDataShared() const;

        /*! \brief Returns the unique data identifier for this object.
          * \return
          * The function returns a 64-bit value which uniquely identifies the embedded data.
          * \remarks
          * This function in particular is useful to identify different objects that share their embedded
          * data. Two objects that share the same data always have the same DataID.
          * \n\n
          * Identifying objects that share their data with other objects is useful for saving objects
          * to a file and later reloading it without losing the data sharing. 
          * \sa Object::isDataShared
          */
        DP_SG_CORE_API virtual DataID getDataID() const;

        /*! \brief Specifies the name for an object.
          * \param
          * name Reference to a STL sting holding the name to specify. 
          * \remarks
          * A previously specified name gets lost when specifying a new name using this function.
          * \n\n
          * Use Object::getName to retrieve the actual name of an Object.  
          */
        DP_SG_CORE_API void setName(const std::string& name);

        /*! \brief Returns the name of the object.
          * \return
          * A reference to a STL string holding the name of the object. The function returns a reference
          * to an empty string if no name was previously specified for the object. 
          * \remarks
          * The function retrieves the object's name previously specified through Object::setName. 
          */
        DP_SG_CORE_API const std::string& getName() const;

        /*! \brief Lets you append an annotation to this Object.
         *  \param anno Annotation to append to this Object.
         *  \remarks
         *  The function lets you append an optional annotation. 
         *  A previously appended annotation gets lost.
         *  \sa Object::getAnnotation
         */
        DP_SG_CORE_API void setAnnotation(const std::string& anno);

        /*! \brief Retrieves the Object's annotation. 
         *  \return
         *  The function returns the annotation as last specified through setAnnotation. 
         *  If no annotation was specified before, the function returns an empty string.
         *  \sa Object::setAnnotation
         */
        DP_SG_CORE_API const std::string& getAnnotation() const;

        /*! \brief Tests whether this object is equivalent to another object.  
          * \param
          * object Pointer to the object to test for equivalence with this object.
          * \param
          * ignoreNames Object names will be ignored while testing if this is \c true.
          * \param
          * deepCompare The function performs a deep-compare if this is \c true.
          * Otherwise the function only performs a shallow compare.
          * \return
          * The function returns \c true if the object pointed to by \a object is detected to be 
          * equivalent to this object. Otherwise the function returns \c false.
          * \remarks
          * The test will be performed considering the optional control parameters ignoreNames and deepCompare. 
          * If you omit these two, the function ignores object names
          * and performs a shallow compare only.
          */
        DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames = true, bool deepCompare = false ) const;

        /*! \brief Attaches arbitrary user data to the object.
         * \param 
         * userData Specifies the address of the arbitrary user data.
         * \remarks
         * Use setUserData to store the address to arbitrary user data with the object.
         * The object does not interpret the data, nor does it take over responsibility 
         * for managing the memory occupied by the data behind \a userData. It just keeps 
         * the address to the user data until the address will be overwritten by a 
         * subsequent call to setUserData.
         * \n\n
         * User data will not be considered for storing to a file! 
         * \n\n
         * Use getUserData to get back the address to the user data that was last specified 
         * through setUserData.
         * \sa getUserData
         */
        DP_SG_CORE_API void setUserData(const void* userData);

        /*! \brief Returns a pointer to the attached arbitrary user data.
         * \return
         * The function returns the pointer to the arbitrary user data that was last 
         * specified by a call to setUserData.
         * \sa setUserData
         */
        DP_SG_CORE_API const void* getUserData() const; 
    
        /*! \brief Get the hints that are set for the object.
         * \return 
         * The function returns the hints for the object.
         * \sa setHints
         */
        unsigned int getHints() const;

        /*! \brief Get the hints that are set for the object, filtered by \a mask.
         * \param mask The mask the hints are filtered against.
         * \returm 
         * The function returns the hints for the object, filtered by \a mask.
         */
        unsigned int getHints( unsigned int mask ) const;

        /*! \brief Set the object's hints
         * \param hints The hints to be set for the object.
         */
        void setHints( unsigned int hints );

        /*! \brief Add hints to the object's hints.
         * \param hints The hints to be added to the object's hints.
         */
        void addHints( unsigned int hints );

        /*! \brief Remove hints from the object's hints.
         * \param hints The hints to be removed from the object's hints.
         */
        void removeHints( unsigned int hints );

        /*! \brief Determine which hints are set in the hierarchy below and including the object, filtered by \a mask.
         * \param mask The mask the hints are filtered against.
         * \return The hints that are set in the hierarchy below and including the object, filtered by \a mask
         * \remarks The getHints functions work locally on the object, whereas this function determines the hints set
         * for the object and the objects below it, should the object be parent of a set of objects. 
         */
        unsigned int getContainedHints( unsigned int mask ) const;

        /*! \brief Determine whether \a hints are set in the objects or the hierarchy below the object.
         * \param hints The hints that need to be set for the function to return true
         * \return Returns true if exactly \a hints are set for the object or the hierarchy beneath it.
         */
        bool containsHints( unsigned int hints ) const;

        /*! \brief Get the hash key of this Object.
         *  \return The hash key of this Object.
         *  \remarks If the hash key is not valid, the virtual function feedHashGenerator() is called to
         *  recursively determine a hash string, using a HashGenerator, which then is converted into
         *  a HashKey.
         *  \sa feedHashGenerator */
        DP_SG_CORE_API dp::util::HashKey getHashKey() const;

        /*! \brief Set a TraversalMask to be used with this Object.
         *  \param mask The mask to be used.
         *  \remarks The traversal mask is used in conjuction with Traverser-derived and Renderer-derived objects to
         *  determine whether this node, and therefore any of this node's children, are traversed, and/or rendered.
         *  See Traverser::setTraversalMask and SceneRenderer::setTraversalMask for more info.
         *  \note The default traversal mask is ~0
         *  \sa getTraversalMask, ViewState::setTraversalMask, Traverser::setTraversalMask, Traverser::setTraversalMaskOverride,
         *  \sa SceneRenderer::setTraversalMask, SceneRenderer::setTraversalMaskOverride */
        void setTraversalMask( unsigned int mask );

        /*  \brief Get the Traversal Mask
         *  \return The current traversal mask.
         *  \sa setTraversalMask, ViewState::getTraversalMask */
        unsigned int getTraversalMask() const;

          /*! \brief Feed the data of this object into the provied HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
        DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        /*! \brief virtual function overload from dp::util::Observer
         *  \param event A constant reference to the notifying event
         *  \param payload A pointer to the payload specified on attach
         *  \note This function is called whenever a Subject that is oberserved by this notifies an event.
         *  The standard behaviour of an Object is, to modify the dirty mask according to the Event.
         *  Then it calls notify with that same event to notify its own Observers.
         *  \sa dp::util::Observer, dp::util::Subject */
        DP_SG_CORE_API virtual void onNotify( dp::util::Event const & event, dp::util::Payload * payload );

        /*! \brief virtual function overload from dp::util::Observer
         *  \param subject A constant reference to the subject currently being destroyed
         *  \param payload A pointer to the payload specified on attach
         *  \note This function is called whenever a Subject that is observed by this is going to be destroyed.
         *  The standard behaviour of an Object is, to just ignore that notification. */
        DP_SG_CORE_API virtual void onDestroyed( dp::util::Subject const & subject, dp::util::Payload* payload );

      public:
      // exposed properties for reflection
        REFLECTION_INFO_API( DP_SG_CORE_API, Object );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Name );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Annotation );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( Hints );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( TraversalMask );
          END_DECLARE_STATIC_PROPERTIES

      protected: // accessible for derived classes
        /*! \brief Constructs an Object.
          */
        DP_SG_CORE_API Object();

        /*! \brief Constructs an Object as a copy of another Object.
          */
        DP_SG_CORE_API Object( const Object& rhs );

        DP_SG_CORE_API virtual unsigned int determineHintsContainment(unsigned int which) const;

      protected:
        /*! \brief Per-object type identifier.
          * \remarks
          * Concrete derived objects must initialize this object code with an identifier,
          * which must be a unique per-object type. It is recommended to do this at object 
          * instantiation.
          * \n\n
          * These object codes can be used for fast object type identification at runtime.
          */
        ObjectCode m_objectCode;

        mutable unsigned int      m_flags; // containment, contained hints
        mutable dp::util::HashKey m_hashKey;

        friend class BoundingVolumeObject;

      private:
        std::string     * m_name;          // optional name
        std::string     * m_annotation;    // optional annotation specified for the object

        const void      * m_userData;      // optional pointer to arbitrary user data
        unsigned int      m_hints;         // object exclusive hints
        unsigned int      m_traversalMask; // object traversal mask

        mutable unsigned int m_dirtyState;
      };


      class Object::Event : public core::Event
      {
      public:
        Event( Object const* object )
          : core::Event( core::Event::OBJECT )
          , m_object( object )
        {
        }

        Object const* getObject() const { return m_object; }

      private:
        Object const* m_object;
      };

      DP_SG_CORE_API void copy( ObjectSharedPtr const& src, ObjectSharedPtr const& dst );

      inline ObjectCode Object::getObjectCode() const
      {
        return m_objectCode;
      }

      inline void Object::setUserData(const void * userData)
      {
        if ( m_userData != userData )
        {
          m_userData = userData; // no sanity check!
          notify( Event(this ) );
        }
      }

      inline const void * Object::getUserData() const
      {
        return m_userData;
      }

      inline unsigned int Object::getHints() const
      {
        return( getHints(~0) );
      }

      inline unsigned int Object::getHints( unsigned int mask ) const
      {
        return( m_hints & mask );
      }

      inline void Object::setHints( unsigned int hints )
      {
        if ( m_hints != hints )
        {
          unsigned int changedHints = ( hints & ~m_hints ) | ( ~hints & m_hints );
          m_hints = hints;
          notify( PropertyEvent( this, PID_Hints ) );
        }
      }

      inline void Object::addHints( unsigned int hints )
      {
        unsigned int changedHints = hints & ~m_hints;
        if ( changedHints )
        {
          m_hints |= hints;
          notify( PropertyEvent( this, PID_Hints ) );
        }
      }

      inline void Object::removeHints( unsigned int hints )
      {
        unsigned int changedHints = hints & m_hints;
        if ( changedHints )
        {
          m_hints &= ~hints;
          notify( PropertyEvent( this, PID_Hints ) );
        }
      }

      inline unsigned int Object::getContainedHints( unsigned int hints ) const
      {
        if ( m_dirtyState & hints )
        {
          // The hints for this mask of hints have not been evaluated or are invalidated.
          // Therefore, evaluate them
          m_flags &= ~hints;
          m_flags |= determineHintsContainment(hints);
          m_dirtyState &= ~hints;
        }

        // Hints containment is valid at this point
        return m_flags & DP_SG_HINT_ALL_HINTS;
      }

      inline bool Object::containsHints( unsigned int hints ) const
      {
        return( (getContainedHints(hints) & hints) == hints );
      }

      inline unsigned int Object::getTraversalMask() const
      {
        return( m_traversalMask );
      }

      inline void Object::setTraversalMask( unsigned int mask )
      {
        if ( m_traversalMask != mask )
        {
          m_traversalMask = mask;
          notify( PropertyEvent( this, PID_TraversalMask ) );
        }
      }

    }//namespace core
  }//namespace sg
}//namespace dp
