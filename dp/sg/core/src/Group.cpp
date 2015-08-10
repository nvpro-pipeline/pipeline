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


#include <dp/sg/core/Group.h>
#include <dp/sg/core/ClipPlane.h>
#include <dp/sg/core/LightSource.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      BEGIN_REFLECTION_INFO( Group )
        DERIVE_STATIC_PROPERTIES( Group, Node )
      END_REFLECTION_INFO

      GroupSharedPtr Group::create()
      {
        return( std::shared_ptr<Group>( new Group() ) );
      }

      HandledObjectSharedPtr Group::clone() const
      {
        return( std::shared_ptr<Group>( new Group( *this ) ) );
      }

      Group::Group()
      {
        m_objectCode = OC_GROUP;
      }

      Group::Group(const Group& rhs)
      : Node(rhs)
      {
        m_objectCode = OC_GROUP;
    
        copyChildren(rhs.m_children);
        copyClipPlanes(rhs.m_clipPlanes);
      }

      Group::~Group()
      {
        removeChildren();
        removeClipPlanes();
      }

      void Group::preRemoveChild(unsigned int index)
      {
        notify( Event( this->getSharedPtr<Group>(), Event::PRE_CHILD_REMOVE, m_children[index], index ) );
      }

      void Group::postRemoveChild(unsigned int index)
      {
      }

      void Group::preAddChild(unsigned int index)
      {
      }

      void Group::postAddChild(unsigned int index)
      {
        notify( Event( this->getSharedPtr<Group>(), Event::POST_CHILD_ADD, m_children[index], index ) );
      }

      Group::ChildrenContainer::iterator Group::doInsertChild( const ChildrenContainer::iterator & gcci, const NodeSharedPtr & child )
      {
        DP_ASSERT( child );
        unsigned int idx = dp::checked_cast<unsigned int>(std::distance( m_children.begin(), gcci ));
        preAddChild( idx );
        ChildrenContainer::iterator position = m_children.insert( gcci, child );
        child->attach( this );
        postAddChild( idx );
        return( position );
      }

      Group::ChildrenContainer::iterator Group::doRemoveChild( const ChildrenContainer::iterator & cci )
      {
        unsigned int idx = dp::checked_cast<unsigned int>(std::distance<ChildrenContainer::iterator>( m_children.begin(), cci ));
        preRemoveChild( idx );
        (*cci)->detach( this );
        ChildrenContainer::iterator ret = m_children.erase( cci );
        postRemoveChild( idx );
        return( ret );
      }

      Group::ChildrenContainer::iterator Group::doReplaceChild( ChildrenContainer::iterator & cci, const NodeSharedPtr & newChild )
      {
        DP_ASSERT( newChild );
        newChild->attach( this );
        (*cci)->detach( this );

        unsigned int idx = dp::checked_cast<unsigned int>(std::distance<ChildrenContainer::iterator>( m_children.begin(), cci ));
        notify( Event( this->getSharedPtr<Group>(), Event::PRE_CHILD_REMOVE, m_children[idx], idx ) );
        *cci = newChild;
        notify( Event( this->getSharedPtr<Group>(), Event::POST_CHILD_ADD, m_children[idx], idx ) );

        return( cci );
      }

      Group::ChildrenIterator Group::insertChild( const ChildrenIterator & gci, const NodeSharedPtr & child )
      {
        return( ChildrenIterator( doInsertChild( gci.m_iter, child ) ) );
      }

      bool Group::removeChild( const NodeSharedPtr & child )
      {
        bool removed = false;
        for ( ChildrenContainer::iterator cci = find( m_children.begin(), m_children.end(), child )
            ; cci != m_children.end()
            ; cci = find( cci, m_children.end(), child ) // continue searching to find all occurrences
            )
        {
          cci = doRemoveChild( cci );
          removed = true;
        }
        return( removed );
      }

      Group::ChildrenIterator Group::removeChild( const ChildrenIterator & ci )
      {
        return( ChildrenIterator( ci.m_iter != m_children.end() ? doRemoveChild( ci.m_iter ) : m_children.end() ) );
      }

      bool Group::replaceChild( const NodeSharedPtr & newChild, const NodeSharedPtr & oldChild )
      {
        if ( newChild != oldChild )
        {
          bool replaced = false;
          for ( ChildrenContainer::iterator cci = find( m_children.begin(), m_children.end(), oldChild )
              ; cci != m_children.end()
              ; cci = find( cci, m_children.end(), oldChild ) // continue searching to find all occurrences
              )
          {
            cci = doReplaceChild( cci, newChild );
            replaced = true;
          }
          return( replaced );
        }
        return( false );
      }

      bool Group::replaceChild( const NodeSharedPtr & newChild, ChildrenIterator & oldChildIterator )
      {
        if ( ( oldChildIterator.m_iter != m_children.end() ) && ( newChild != *oldChildIterator ) )
        {
          doReplaceChild( oldChildIterator.m_iter, newChild );
          return( true );
        }
        return( false );
      }

      void Group::clearChildren()
      {
        for ( ChildrenContainer::iterator cci = m_children.begin() ; cci != m_children.end() ; ++cci )
        {
          // this is nearly a nop if the bounding volumes are already dirty and cheap in comparison to what happens int he background
          // if the observer calls getBounding*() it's necessary that the dirty flag has already been set.
          notify( Event( this->getSharedPtr<Group>(), Event::PRE_CHILD_REMOVE, m_children[0], 0 ) );
          (*cci)->detach( this );
        }
        m_children.clear();
      }

      Box3f Group::calculateBoundingBox() const
      {
        Box3f bbox;
        for ( ChildrenContainer::const_iterator it = m_children.begin() ; it!=m_children.end() ; ++it )
        {
          bbox = boundingBox(bbox, (*it)->getBoundingBox());
        }
        return( bbox );
      }

      Sphere3f Group::calculateBoundingSphere() const
      {
        Sphere3f sphere;
        if ( ! m_children.empty() )
        {
          std::vector<dp::math::Sphere3f> spheres;
          spheres.reserve( m_children.size() );
          for ( ChildrenContainer::const_iterator it = m_children.begin() ; it!=m_children.end() ; ++it )
          {
            spheres.push_back( (*it)->getBoundingSphere() );
          }
          sphere = boundingSphere( &spheres[0], dp::checked_cast<unsigned int>(spheres.size()) );
        }
        return( sphere );
      }

      unsigned int Group::determineHintsContainment( unsigned int hints ) const
      {
        unsigned int containment = Node::determineHintsContainment( hints );
        for ( size_t i=0 ; ( (containment & hints ) != hints ) && i<m_children.size() ; i++ )
        {
          containment |= m_children[i]->getContainedHints( hints );
        }
        return( containment );
      }

      void Group::copyChildren(const ChildrenContainer & children)
      {
        DP_ASSERT(m_children.empty());

        // copy children from source object and add reference for each
        m_children.resize(children.size()); // allocate destination range
        for ( size_t i=0; i<children.size(); ++i )
        {
          m_children[i] = children[i].clone();
          m_children[i]->attach( this );
        }
      }

      void Group::copyClipPlanes(const ClipPlaneContainer & clipPlanes)
      {
        DP_ASSERT(m_clipPlanes.empty());

        // copy clipping planes from source object and add reference for each
        m_clipPlanes.resize(clipPlanes.size());   // allocate destination range
        for ( size_t i=0 ; i<clipPlanes.size() ; i++ )
        {
          m_clipPlanes[i] = clipPlanes[i].clone();
          m_clipPlanes[i]->attach( this );
        }
      }

      Group & Group::operator=(const Group & rhs)
      {
        if (&rhs != this)
        {
          Node::operator=(rhs);

          removeChildren();
          removeClipPlanes();

          copyChildren(rhs.m_children);
          copyClipPlanes(rhs.m_clipPlanes);

          notify( Event( this->getSharedPtr<Group>(), Event::POST_GROUP_EXCHANGED ) );
        }
        return *this;
      }

      bool Group::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<Group>() && Node::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          GroupSharedPtr const& g = object.staticCast<Group>();
          equi =   ( m_children.size()   == g->m_children.size() )
                && ( m_clipPlanes.size() == g->m_clipPlanes.size() );
          if ( deepCompare )
          {
            for ( ChildrenContainer::const_iterator lhsit = m_children.begin() ; equi && lhsit != m_children.end() ; ++lhsit )
            {
              bool found = false;
              for ( ChildrenContainer::const_iterator rhsit = g->m_children.begin() ; !found && rhsit != g->m_children.end() ; ++rhsit )
              {
                found = (*lhsit)->isEquivalent( *rhsit, ignoreNames, true );
              }
              equi = found;
            }
            for ( ClipPlaneContainer::const_iterator lhsit = m_clipPlanes.begin() ; equi && lhsit != m_clipPlanes.end() ; ++lhsit )
            {
              bool found = false;
              for ( ClipPlaneContainer::const_iterator rhsit = g->m_clipPlanes.begin() ; !found && rhsit != g->m_clipPlanes.end() ; ++rhsit )
              {
                found = (*lhsit)->isEquivalent( *rhsit, ignoreNames, true );
              }
              equi = found;
            }
          }
          else
          {
            for ( ChildrenContainer::const_iterator lhsit = m_children.begin() ; equi && lhsit != m_children.end() ; ++lhsit )
            {
              bool found = false;
              for ( ChildrenContainer::const_iterator rhsit = g->m_children.begin() ; !found && rhsit != g->m_children.end() ; ++rhsit )
              {
                found = ( *lhsit == *rhsit );
              }
              equi = found;
            }
            for ( ClipPlaneContainer::const_iterator lhsit = m_clipPlanes.begin() ; equi && lhsit != m_clipPlanes.end() ; ++lhsit )
            {
              bool found = false;
              for ( ClipPlaneContainer::const_iterator rhsit = g->m_clipPlanes.begin() ; !found && rhsit != g->m_clipPlanes.end() ; ++rhsit )
              {
                found = ( *lhsit == *rhsit );
              }
              equi = found;
            }
          }
        }
        return( equi );
      }

      unsigned int Group::getNumberOfActiveClipPlanes() const
      {
        unsigned int noacp = 0;
        DP_ASSERT( m_clipPlanes.size() <= UINT_MAX );
        for ( unsigned int i=0 ; i<m_clipPlanes.size() ; i++ )
        {
          if ( m_clipPlanes[i]->isEnabled() )
          {
            noacp++;
          }
        }
        return( noacp );
      }

      Group::ClipPlaneIterator Group::addClipPlane( const ClipPlaneSharedPtr & plane )
      {
        ClipPlaneContainer::iterator cpci = find( m_clipPlanes.begin(), m_clipPlanes.end(), plane );
        if ( cpci == m_clipPlanes.end() )
        {
          plane->attach( this );
          m_clipPlanes.push_back( plane );
          cpci = m_clipPlanes.end() - 1;
      
          notify( Group::Event( this->getSharedPtr<Group>(), Group::Event::CLIP_PLANES_CHANGED ) );
        }
        return( ClipPlaneIterator( cpci ) );
      }

      Group::ClipPlaneContainer::iterator Group::doRemoveClipPlane( const ClipPlaneContainer::iterator & cpci )
      {
        (*cpci)->detach( this );

        Group::ClipPlaneContainer::iterator it = m_clipPlanes.erase( cpci );
        notify( Group::Event( this->getSharedPtr<Group>(), Group::Event::CLIP_PLANES_CHANGED ) );

        return it;
      }

      bool Group::removeClipPlane( const ClipPlaneSharedPtr & plane )
      {
        DP_ASSERT( plane );

        ClipPlaneContainer::iterator cpci = find( m_clipPlanes.begin(), m_clipPlanes.end(), plane );
        if ( cpci != m_clipPlanes.end() )
        {
          doRemoveClipPlane( cpci );
          return( true );
        }
        return( false );
      }

      Group::ClipPlaneIterator Group::removeClipPlane( const ClipPlaneIterator & cpi )
      {
        if ( cpi.m_iter != m_clipPlanes.end() )
        {
          return( ClipPlaneIterator( doRemoveClipPlane( cpi.m_iter ) ) );
        }
        return( cpi );
      }

      void Group::clearClipPlanes()
      {
        unsigned int dirtyBits = 0;
        for ( ClipPlaneContainer::iterator cpci = m_clipPlanes.begin() ; cpci != m_clipPlanes.end() ; ++cpci )
        {
          (*cpci)->detach( this );
        }
        m_clipPlanes.clear();
        notify( Group::Event( this->getSharedPtr<Group>(), Group::Event::CLIP_PLANES_CHANGED ) );
      }

      void Group::removeChildren()
      {
        for ( ChildrenContainer::iterator it = m_children.begin(); it != m_children.end(); ++it )
        {
          (*it)->detach( this );
        }
        m_children.clear();
      }

      void Group::removeClipPlanes()
      {
        for ( ClipPlaneContainer::iterator it = m_clipPlanes.begin(); it != m_clipPlanes.end(); ++it )
        {
          (*it)->detach( this );
        }
        m_clipPlanes.clear();
      }

      void Group::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Node::feedHashGenerator( hg );
        for ( size_t i=0 ; i<m_children.size() ; ++i )
        {
          hg.update( m_children[i] );
        }
        for ( size_t i=0 ; i<m_clipPlanes.size() ; ++i )
        {
          hg.update( m_clipPlanes[i] );
        }
      }

    } // namespace core
  } // namespace sg
} // namespace dp
