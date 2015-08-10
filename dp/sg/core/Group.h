// Copyright NVIDIA Corporation 2002-2011
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

#include <dp/sg/core/Config.h>
#include <dp/sg/core/ClipPlane.h>
#include <dp/sg/core/LightSource.h>
#include <dp/sg/core/Node.h>
#include <dp/sg/core/ConstIterator.h>
#include <dp/util/HashGenerator.h>
#include <dp/math/Planent.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Provides grouping of Node objects
        * \par Namespace: dp::sg::core
        * \remarks
        * A Group is a special Node that provides grouping of all kinds of Node-derived objects.
        * Nodes that are grouped underneath a Group are referred to as children of the Group.
        */
      class Group : public Node
      {
        public:
          DP_SG_CORE_API static GroupSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~Group();

          class Event;

        public:
          /*! \brief The container type of the Groups children */
          typedef std::vector<NodeSharedPtr>                                 ChildrenContainer;

          /*! \brief The iterator over the ChildrenContainer */
          typedef ConstIterator<Group,ChildrenContainer::iterator>           ChildrenIterator;

          /*! \brief The const iterator over the ChildrenContainer */
          typedef ConstIterator<Group,ChildrenContainer::const_iterator>     ChildrenConstIterator;

          /*! \brief The container type of the Groups clip planes */
          typedef std::vector<ClipPlaneSharedPtr>                            ClipPlaneContainer;

          /*! \brief The iterator over the ClipPlaneContainer */
          typedef ConstIterator<Group,ClipPlaneContainer::iterator>          ClipPlaneIterator;

          /*! \brief The const iterator over the ClipPlaneContainer */
          typedef ConstIterator<Group,ClipPlaneContainer::const_iterator>    ClipPlaneConstIterator;

        public:
          /*! \brief Get the number of children nodes in this Group
           *  \return The number of children nodes in this Group.
           *  \sa beginChildren, endChildren, addChild, insertChild, removeChild, replaceChild, clearChildren, findChild */
          DP_SG_CORE_API unsigned int getNumberOfChildren() const;

          /*! \brief Get a const iterator to the first child in this Group.
           *  \return A const iterator to the first child in this Group.
           *  \sa getNumberOfChildren, endChildren, addChild, insertChild, removeChild, replaceChild, clearChildren, findChild */
          ChildrenConstIterator beginChildren() const;

          /*! \brief Get an iterator to the first child in this Group.
           *  \return An iterator to the first child in this Group.
           *  \sa getNumberOfChildren, endChildren, addChild, insertChild, removeChild, replaceChild, clearChildren, findChild */
          ChildrenIterator beginChildren();

          /*! \brief Get a const iterator that points just beyond the end of the children in this Group.
           *  \return A const iterator that points just beyond the end of the children in this Group.
           *  \sa getNumberOfChildren, beginChildren, addChild, insertChild, removeChild, replaceChild, clearChildren, findChild */
          ChildrenConstIterator endChildren() const;

          /*! \brief Get an iterator that points just beyond the end of the children in this Group.
           *  \return An iterator that points just beyond the end of the children in this Group.
           *  \sa getNumberOfChildren, beginChildren, addChild, insertChild, removeChild, replaceChild, clearChildren, findChild */
          ChildrenIterator endChildren();

          /*! \brief Add a child node at the end of the children in this Group.
           *  \param child The child node to add.
           *  \return An iterator that points to the position where \a child was added to the children.
           *  \sa getNumberOfChildren, beginChildren, endChildren, insertChild, removeChild, replaceChild, clearChildren, findChild */
          DP_SG_CORE_API ChildrenIterator addChild( const NodeSharedPtr & child );

          /*! \brief Insert a child node at a specified position in this Group.
           *  \param gci An iterator that points to the position where \a child is inserted.
           *  \param child The child node to add.
           *  \return An iterator that points to the position where \a child was inserted.
           *  \sa getNumberOfChildren, beginChildren, endChildren, addChild, removeChild, replaceChild, clearChildren, findChild */
          DP_SG_CORE_API ChildrenIterator insertChild( const ChildrenIterator & gci, const NodeSharedPtr & child );

          /*! \brief Remove all occurances of a specified Node from this Group.
           *  \param child The node to remove from this Group.
           *  \return \c true, if at least one Node has been removed from this Group, otherwise \c false.
           *  \note A Group might hold the same Node multiple times. This function removes all of them.
           *  \sa getNumberOfChildren, beginChildren, endChildren, addChild, insertChild, replaceChild, clearChildren, findChild */
          DP_SG_CORE_API bool removeChild( const NodeSharedPtr & child );

          /*! \brief Remove a specified Node from this Group.
           *  \param childIterator An iterator that points to the child to be removed.
           *  \return An iterator pointing to the new location of the child that followed the Node removed by this
           *  function call, which is endChildren() if the operation removed the last child.
           *  \sa getNumberOfChildren, beginChildren, endChildren, addChild, insertChild, replaceChild, clearChildren, findChild */
          DP_SG_CORE_API ChildrenIterator removeChild( const ChildrenIterator & childIterator );

          /*! \brief Replace all occurances of a specified Node in this Group by an other.
           *  \param newChild The new Node that replaces the old ones.
           *  \param oldChild The old Node to be replaced by the new one.
           *  \return \c true, if at least one Node has been replaced, otherwise \c false.
           *  \sa getNumberOfChildren, beginChildren, endChildren, addChild, insertChild, removeChild, clearChildren, findChild */
          DP_SG_CORE_API bool replaceChild( const NodeSharedPtr & newChild, const NodeSharedPtr & oldChild );

          /*! \brief Replace a specified Node in this Group by an other.
           *  \param newChild The new Node that replaces the old one.
           *  \param oldChildIterator An iterator pointing to the Node to be replaced.
           *  \return \c true if the specified Node has been replaced, otherwise \c false.
           *  \sa getNumberOfChildren, beginChildren, endChildren, addChild, insertChild, removeChild, clearChildren, findChild */
          DP_SG_CORE_API bool replaceChild( const NodeSharedPtr & newChild, ChildrenIterator & oldChildIterator );

          /*! \brief Remove all the children from this Group.
           *  \sa getNumberOfChildren, beginChildren, endChildren, addChild, insertChild, removeChild, replaceChild, findChild */
          DP_SG_CORE_API void clearChildren();

          /*! \brief Find the first occurance of a specified Node in this Group.
           *  \param start A const iterator into the children in this Group, where the search is to start.
           *  \param node The Node to be found in this Group.
           *  \return A const iterator pointing to the first occurance of the specified node in this Group.
           *  \note To cound the multiplicity of a child in a Group, one could do something like that:
           *  \code
           *  unsigned int count = 0;
           *  for ( Group::ChildrenConstIterator gcci = group->findChild( group->beginChildren(), child )
           *      ; gcci != group->endChildren()
           *      ; gcci = group->findChild( ++gcci, child ) )
           *  {
           *    count++;
           *  }
           *  \endcode
           *  \sa getNumberOfChildren, beginChildren, endChildren, addChild, insertChild, removeChild, replaceChild, clearChildren */
          DP_SG_CORE_API ChildrenConstIterator findChild( const ChildrenConstIterator & start, const NodeSharedPtr & node ) const;

          /*! \brief Find the first occurance of a specified Node in this Group.
           *  \param start An iterator into the children in this Group, where the search is to start.
           *  \param node The Node to be found in this Group.
           *  \return An iterator pointing to the first occurance of the specified node in this Group.
           *  \note To cound the multiplicity of a child in a Group, one could do something like that:
           *  \code
           *  unsigned int count = 0;
           *  for ( Group::ChildrenIterator gci = group->findChild( group->beginChildren(), child )
           *      ; gci != group->endChildren()
           *      ; gci = group->findChild( ++gci, child ) )
           *  {
           *    count++;
           *  }
           *  \endcode
           *  \sa getNumberOfChildren, beginChildren, endChildren, addChild, insertChild, removeChild, replaceChild, clearChildren */
          DP_SG_CORE_API ChildrenIterator findChild( const ChildrenIterator & start, const NodeSharedPtr & node );

          /*! \brief Get the number of clip planes in this Group.
           *  \return The number of clip planes on this Group.
           *  \sa getNumberOfActiveClipPlanes, beginClipPlanes, endClipPlanes, addClipPlane, removeClipPlane, clearClipPlanes, findClipPlane */
          DP_SG_CORE_API unsigned int getNumberOfClipPlanes() const;

          /*! \brief Get the number of active clip planes in this Group.
           *  \return The number of active clip planes in this Group.
           *  \sa getNumberOfClipPlanes, beginClipPlanes, endClipPlanes, addClipPlane, removeClipPlane, clearClipPlanes, findClipPlane */
          DP_SG_CORE_API unsigned int getNumberOfActiveClipPlanes() const;

          /*! \brief Get a const iterator to the first clip plane in this Group.
           *  \return A const iterator to the first clip plane in this Group.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, endClipPlanes, addClipPlane, removeClipPlane, clearClipPlanes, findClipPlane */
          DP_SG_CORE_API ClipPlaneConstIterator beginClipPlanes() const;

          /*! \brief Get an iterator to the first clip plane in this Group.
           *  \return An iterator to the first clip plane in this Group.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, endClipPlanes, addClipPlane, removeClipPlane, clearClipPlanes, findClipPlane */
          DP_SG_CORE_API ClipPlaneIterator beginClipPlanes();

          /*! \brief Get a const iterator that points just beyond the end of the clip planes in this Group.
           *  \return A const iterator that points just beyond the end of the clip planes in this Group.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, beginClipPlanes, addClipPlane, removeClipPlane, clearClipPlanes, findClipPlane */
          DP_SG_CORE_API ClipPlaneConstIterator endClipPlanes() const;

          /*! \brief Get an iterator that points just beyond the end of the clip planes in this Group.
           *  \return An iterator that points just beyond the end of the clip planes in this Group.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, beginClipPlanes, addClipPlane, removeClipPlane, clearClipPlanes, findClipPlane */
          DP_SG_CORE_API ClipPlaneIterator endClipPlanes();

          /*! \brief Adds a user clipping plane to this Group.
           *  \param plane Specifies the ClipPlane to add
           *  \return An iterator that points to the position where \a plane was added.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, beginClipPlanes, endClipPlanes, removeClipPlane, clearClipPlanes, findClipPlane */
          DP_SG_CORE_API ClipPlaneIterator addClipPlane( const ClipPlaneSharedPtr & plane );

          /*! \brief Remove a clip plane from this Group.
           *  \param plane The clip plane to remove from this Group.
           *  \return \c true, if the clip plane has been removed from this Group, otherwise \c false.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, beginClipPlanes, endClipPlanes, addClipPlane, clearClipPlanes, findClipPlane */
          DP_SG_CORE_API bool removeClipPlane( const ClipPlaneSharedPtr & plane );

          /*! \brief Remove a clip plane from this Group.
           *  \param cpi An iterator to the clip plane to remove from this Group.
           *  \return An iterator pointing to the new location of the clip plane that followed the one removed by
           *  this function call, which is endClipPlanes() if the operation removed the last clip plane.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, beginClipPlanes, endClipPlanes, addClipPlane, clearClipPlanes, findClipPlane */
          DP_SG_CORE_API ClipPlaneIterator removeClipPlane( const ClipPlaneIterator & cpi );

          /*! \brief Remove all clip planes from this Group.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, beginClipPlanes, endClipPlanes, addClipPlane, removeClipPlane, findClipPlane */
          DP_SG_CORE_API void clearClipPlanes();

          /*  \brief Find a specified clip plane in this Group.
           *  \param plane The clip plane to find.
           *  \return A const iterator to the found clip plane in this Group.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, beginClipPlanes, endClipPlanes, addClipPlane, removeClipPlane, clearClipPlanes */
          DP_SG_CORE_API ClipPlaneConstIterator findClipPlane( const ClipPlaneSharedPtr & plane ) const;

          /*  \brief Find a specified clip plane in this Group.
           *  \param plane The clip plane to find.
           *  \return An iterator to the found clip plane in this Group.
           *  \sa getNumberOfClipPlanes, getNumberOfActiveClipPlanes, beginClipPlanes, endClipPlanes, addClipPlane, removeClipPlane, clearClipPlanes */
          DP_SG_CORE_API ClipPlaneIterator findClipPlane( const ClipPlaneSharedPtr & plane );

          /*! \brief Overrides Object::isEquivalent.  
            * \param
            * p Pointer to the Object to test for equivalence with this Group object.
            * \param
            * ignoreNames Object names will be ignored while testing if this is \c true.
            * \param
            * deepCompare The function performs a deep-compare instead of a shallow compare if this is \c true.
            * \return
            * The function returns \c true if the Object pointed to by \a p is detected to be equivalent
            * to this Group object.
            * \remarks
            * The test will be performed considering the optional control parameters ignoreNames and deepCompare. 
            * If you omit these two, the function ignores the object names
            * and performs a shallow compare only.
            * \sa Object::isEquivalent
            */
          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const;

          /*! \brief Assigns new content from another Group object. 
            * \param
            * rhs Group object from which to assign the new content.
            * \return
            * A reference to this Group object.
            * \remarks
            * The assignment operator unreferences the old content before assigning the new content. The new
            * content will be a deep-copy of the right-hand-sided object's content.
            * \sa Group::clone
            */
          DP_SG_CORE_API Group & operator=(const Group & rhs);

          REFLECTION_INFO_API( DP_SG_CORE_API, Group );

        protected:
          /*! \brief Default-constructs a Group object.
            */
          DP_SG_CORE_API Group();

          /*! \brief Constructs a Group as a copy of another Group.
            * \param
            * rhs Group to serve as source for the newly constructed Group. 
            */
          DP_SG_CORE_API Group( const Group& rhs );

          /*! \brief Called from the framework immediately before a child will be removed from the Group.
            * \param
            * position Zero-based position of the child that immediately will be removed from the Group.
            * \remarks
            * It is recommended, for Group-derived classes that need to keep track of child nodes removal, 
            * to override this function to perform necessary tasks immediately before a child node will be 
            * removed from the Group.   
            * \n\n
            * The function will be called with the zero-based index that references the child that
            * will be removed immediately after this call.
            * \sa Group::postRemoveChild
            */
          DP_SG_CORE_API virtual void preRemoveChild(unsigned int position);

          /*! \brief Called from the framework immediately after a child has been removed from the Group.
            * \param
            * position Zero-based position that previously referenced the removed child node.
            * \remarks
            * It is recommended, for Group-derived classes that need to keep track of child nodes removal, 
            * to override this notifier to perform necessary tasks immediately after a child node has been 
            * removed from the Group.   
            * \n\n
            * The function will be called with the zero-based position that previously referenced the 
            * removed child. Note that this position can no longer be used to reference the child node, 
            * because the child node already has been removed from this Group.
            * \sa Group::preRemoveChild
            */
          DP_SG_CORE_API virtual void postRemoveChild(unsigned int position);

          /*! \brief Called from the framework immediately before a child will be added to the Group.
            * \param
            * position Zero-based position where the child node will be added.
            * \remarks
            * It is recommended, for Group-derived classes that need to keep track of child nodes insertion,
            * to override this notifier to perform necessary tasks immediately before a child node will be 
            * added to the Group.   
            * \n\n
            * The function will be called with the zero-based position where the child node will be added.
            * Note that this position cannot yet be used to reference the child node, because the child
            * node has not been added to this Group at this point.
            * \sa Group::postAddChild
            */
          DP_SG_CORE_API virtual void preAddChild(unsigned int position);

          /*! \brief Called from the framework immediately after a child was added to the Group.
            * \param
            * position Zero-based position where the child node was added.
            * \remarks
            * It is recommended, for Group-derived classes that need to keep track of child nodes insertion,
            * to override this notifier to perform necessary tasks immediately after a child node has been 
            * added to the Group.   
            * \n\n
            * The function will be called with the zero-based position where the child node has been added
            * immediately before this call.
            * \sa Group::preAddChild
            */
          DP_SG_CORE_API virtual void postAddChild(unsigned int position);

          /*! \brief Called from the framework if re-calculation of the Group's bounding box is required.
           * \return
           * The function returns a dp::math::Box3f that represents the actual bounding box of this Group.
           * \remarks
           * The function calculates the bounding box by accumulating the bounding boxes of all 
           * available child nodes. 
           * \n\n
           * For Group-derived classes to specialize bounding box calculation, it is recommended to 
           * override this function. 
           * \sa dp::math::Box3f
           */
          DP_SG_CORE_API virtual dp::math::Box3f calculateBoundingBox() const;

          /*! \brief Called from the framework if re-calculation of the Group's bounding sphere is required.
            * \return
            * The function returns a dp::math::Sphere3f that represents the actual bounding sphere of this 
            * Group.
            * \remarks
            * The function calculates the bounding sphere by accumulating the bounding spheres of all 
            * available child nodes. 
            * \n\n
            * For Group-derived classes to specialize bounding sphere calculation, it is recommended to 
            * override this function. 
            * \sa dp::math::Sphere3f
            */
          DP_SG_CORE_API virtual dp::math::Sphere3f calculateBoundingSphere() const;

          /*! \brief Determines Object hints contained in the hierarchy
           * \param hints
           * Specifies a but mask to indicate the Object hints for which containment should be determined.
           * \return 
           * The function returns a bit field indicating all hints contained in the tree hierarchy
           * having this Group as root. The bit field returned is filtered by the input bit mask.
           * \remarks
           * The function override the base class implementation to trigger determination for all
           * children of the Group, as well as for all maintained LightSources.
           * \sa Object::determineHintsContainment
           */
          DP_SG_CORE_API virtual unsigned int determineHintsContainment( unsigned int hints ) const;
      
          /*! \brief Feed the data of this object into the provied HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          // The following copy* and remove* functions do not notify observers about changes
          // Functions calling these functions must ensure that the events are being passed up.
          void copyChildren( const ChildrenContainer & children );
          void copyClipPlanes( const ClipPlaneContainer & clipPlanes );

          void removeChildren();
          void removeClipPlanes();

        private:
          DP_SG_CORE_API ChildrenContainer::iterator doInsertChild( const ChildrenContainer::iterator & gcci, const NodeSharedPtr & child );
          DP_SG_CORE_API ChildrenContainer::iterator doRemoveChild( const ChildrenContainer::iterator & cci );
          DP_SG_CORE_API ChildrenContainer::iterator doReplaceChild( ChildrenContainer::iterator & cci, const NodeSharedPtr & newChild );
          DP_SG_CORE_API ClipPlaneContainer::iterator doRemoveClipPlane( const ClipPlaneContainer::iterator & cpci );

        private:
          ChildrenContainer       m_children;
          ClipPlaneContainer      m_clipPlanes;
      };

      class Group::Event : public core::Event
      {
      public:
        enum Type
        {
            POST_CHILD_ADD
          , PRE_CHILD_REMOVE
          , POST_GROUP_EXCHANGED
          , LIGHT_SOURCES_CHANGED
          , CLIP_PLANES_CHANGED
        };

        Event( GroupSharedPtr const& group, Type type, const NodeSharedPtr& child, unsigned int index = 0 )
          : core::Event( core::Event::GROUP )
          , m_group( group )
          , m_child( child )
          , m_type( type )
          , m_index( index )
        {
        }

        Event( GroupSharedPtr const& group, Type type )
          : core::Event( core::Event::GROUP )
          , m_group( group )
          , m_child( NodeSharedPtr::null )
          , m_index( ~0 )
          , m_type( type )
        {
        }

        GroupSharedPtr const& getGroup() const  { return m_group; }
        NodeSharedPtr const& getChild() const   { return m_child; }
        unsigned int  getIndex() const          { return m_index; }

        // override getType from dp::sg::core::Event
        Type          getType() const         { return m_type; }

      private:
        GroupSharedPtr const& m_group;
        Type                  m_type;
        const NodeSharedPtr&  m_child;
        unsigned int          m_index;
      };


      inline unsigned int Group::getNumberOfChildren() const
      {
        return( dp::checked_cast<unsigned int>(m_children.size()) );
      }

      inline Group::ChildrenConstIterator Group::beginChildren() const
      {
        return( ChildrenConstIterator( m_children.begin() ) );
      }

      inline Group::ChildrenIterator Group::beginChildren()
      {
        return( ChildrenIterator( m_children.begin() ) );
      }

      inline Group::ChildrenConstIterator Group::endChildren() const
      {
        return( ChildrenConstIterator( m_children.end() ) );
      }

      inline Group::ChildrenIterator Group::endChildren()
      {
        return( ChildrenIterator( m_children.end() ) );
      }

      inline Group::ChildrenIterator Group::addChild( const NodeSharedPtr & child )
      {
        return( ChildrenIterator( doInsertChild( m_children.end(), child ) ) );
      }

      inline Group::ChildrenConstIterator Group::findChild( const ChildrenConstIterator & start, const NodeSharedPtr & node ) const
      {

        return( ChildrenConstIterator( find( start.m_iter, m_children.end(), node ) ) );
      }

      inline Group::ChildrenIterator Group::findChild( const ChildrenIterator & start, const NodeSharedPtr & node )
      {

        return( ChildrenIterator( find( start.m_iter, m_children.end(), node ) ) );
      }

      inline unsigned int Group::getNumberOfClipPlanes() const
      {
        return( dp::checked_cast<unsigned int>(m_clipPlanes.size()) );
      }

      inline Group::ClipPlaneConstIterator Group::beginClipPlanes() const
      {
        return( ClipPlaneConstIterator( m_clipPlanes.begin() ) );
      }

      inline Group::ClipPlaneIterator Group::beginClipPlanes()
      {
        return( ClipPlaneIterator( m_clipPlanes.begin() ) );
      }

      inline Group::ClipPlaneConstIterator Group::endClipPlanes() const
      {
        return( ClipPlaneConstIterator( m_clipPlanes.end() ) );
      }

      inline Group::ClipPlaneIterator Group::endClipPlanes()
      {
        return( ClipPlaneIterator( m_clipPlanes.end() ) );
      }

      inline Group::ClipPlaneConstIterator Group::findClipPlane( const ClipPlaneSharedPtr & plane ) const
      {

        return( ClipPlaneConstIterator( find( m_clipPlanes.begin(), m_clipPlanes.end(), plane ) ) );
      }

      inline Group::ClipPlaneIterator Group::findClipPlane( const ClipPlaneSharedPtr & plane )
      {
        return( ClipPlaneIterator( find( m_clipPlanes.begin(), m_clipPlanes.end(), plane ) ) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
