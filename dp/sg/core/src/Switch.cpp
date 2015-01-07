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


#include <dp/sg/core/Switch.h>

#include <iterator>

using namespace dp::math;
using namespace dp::util;
using std::binary_function;
using std::map;
using std::pair;
using std::set;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( Switch, ActiveMaskKey );
      DEFINE_STATIC_PROPERTY( Switch, ActiveSwitchMask );

      BEGIN_REFLECTION_INFO( Switch )
        DERIVE_STATIC_PROPERTIES( Switch, Group );

        INIT_STATIC_PROPERTY_RW( Switch, ActiveMaskKey, MaskKey, SEMANTIC_VALUE, value, value );
        INIT_STATIC_PROPERTY_RO( Switch, ActiveSwitchMask, SwitchMask, SEMANTIC_VALUE, const_reference );
      END_REFLECTION_INFO

      struct DecrementGreater : public binary_function<unsigned int, unsigned int, unsigned int>
      {
        unsigned int operator()(unsigned int i, unsigned int ref) const
        {
          if ( i > ref )
          {
            i--;
          }
          return i;
        }
      };

      struct IncrementGreaterEqual : public binary_function<unsigned int, unsigned int, unsigned int>
      {
        unsigned int operator()(unsigned int i, unsigned int ref) const
        {
          if ( i >= ref )
          {
            i++;
          }
          return i;
        }
      };

      void Switch::init( void )
      {
        // add default
        SwitchMask empty;

        m_masks[ DEFAULT_MASK_KEY ] = empty;
        m_activeMaskKey = DEFAULT_MASK_KEY;
      }

      SwitchSharedPtr Switch::create()
      {
        return( std::shared_ptr<Switch>( new Switch() ) );
      }

      HandledObjectSharedPtr Switch::clone() const
      {
        return( std::shared_ptr<Switch>( new Switch( *this ) ) );
      }

      Switch::Switch()
      {
        m_objectCode = OC_SWITCH;

        init();
      }

      Switch::Switch( const Switch &rhs )
      : Group(rhs)
      , m_masks(rhs.m_masks)
      , m_activeMaskKey(rhs.m_activeMaskKey)
      {
        m_objectCode = OC_SWITCH;

        // set active mask after init
        // do not add default mask, as it will already be there
        setActiveMaskKey( m_activeMaskKey );
      }

      Switch::~Switch()
      {
      }

      void Switch::postRemoveChild(unsigned int index)
      {
        //
        // Loop through all masks and fix them up
        //
        for ( NonConstMaskIterator iter = m_masks.begin() ; iter != m_masks.end() ; ++iter )
        {
          SwitchMask & activeMask = (*iter).second;

          if ( !activeMask.empty() )
          {
            activeMask.erase(index);

            // adjust remaining actives after a child has been removed
            set<unsigned int> actives;
            transform( activeMask.begin()
                     , activeMask.end()
                     , inserter(actives, actives.begin())
                     , bind2nd(DecrementGreater(), index) );

            activeMask.swap(actives);
            if( iter->first == getActiveMaskKey() )
            {
              notify( PropertyEvent( this, PID_ActiveSwitchMask) );
            }
          }
        }
        Group::postRemoveChild(index);
      }

      void Switch::postAddChild(unsigned int index)
      {
        DP_ASSERT(index<getNumberOfChildren());

        //
        // Loop through all masks and fix them up
        //
        NonConstMaskIterator iter = m_masks.begin();

        while( iter != m_masks.end() )
        {
          SwitchMask & activeMask = (*iter).second;
          ++iter;

          if ( !activeMask.empty() )
          {
            // adjust remaining actives after a child has been added
            set<unsigned int> actives;
            transform( activeMask.begin()
                     , activeMask.end()
                     , inserter(actives, actives.begin())
                     , bind2nd(IncrementGreaterEqual(), index) );
        
            activeMask.swap(actives);
            if( iter->first == getActiveMaskKey() )
            {
              notify( PropertyEvent( this, PID_ActiveSwitchMask) );
            }
          }
        }
        Group::postAddChild(index);
      }

      void Switch::postActivateChild(unsigned int index)
      {
        notify( PropertyEvent( this, PID_ActiveSwitchMask) );
      }

      void Switch::postDeactivateChild(unsigned int index)
      {
        notify( PropertyEvent( this, PID_ActiveSwitchMask) );
      }

      void  Switch::setActive()
      {
        for (unsigned int i=0; i<getNumberOfChildren(); i++)
        {
          setActive(i);
        }
      }

      void  Switch::setInactive()
      {
        for (unsigned int i=0; i<getNumberOfChildren(); i++)
        {
          setInactive(i);
        }
      }

      void Switch::setActive(unsigned int index)
      {
        SwitchMask & aMask = activeMask();

        //  if the index previously was inactive, invalidate the caches
        pair< set<unsigned int>::iterator, bool > pr = aMask.insert(index);
        if ( pr.second )    //  true <=> index was inserted
        {
          postActivateChild(index);
        }
      }
        
      void Switch::setInactive( unsigned int index )
      {
        //  if the index previously was active, invalidate the caches
        if ( activeMask().erase(index) == 1 )
        {
          postDeactivateChild(index);
        }
      }

      unsigned int  Switch::getNumberOfActive() const
      {
        return( checked_cast<unsigned int>(activeMask().size()) );
      }

      unsigned int Switch::getActive( vector<unsigned int>& indices ) const
      {
        indices.clear();

        const SwitchMask & aMask = activeMask();

        copy(aMask.begin(), aMask.end(), inserter(indices, indices.begin()));
        return( checked_cast<unsigned int>(indices.size()) );
      }

      bool Switch::isActive() const
      {
        return activeMask().size() > 0;
      }

      bool Switch::isActive(unsigned int index) const
      {
        const SwitchMask & aMask = activeMask();
        return aMask.find(index) != aMask.end();
      }

      Box3f Switch::calculateBoundingBox() const
      {
        Box3f bbox;
        Group::ChildrenConstIterator gcci = beginChildren();
        unsigned int previousIndex = 0;
        const SwitchMask & aMask = activeMask();

        for ( set<unsigned int>::const_iterator it=aMask.begin() ; it!=aMask.end() ; ++it )
        {
          DP_ASSERT( *it < getNumberOfChildren() );

          std::advance( gcci, *it - previousIndex );
          previousIndex = *it;

          bbox = boundingBox( bbox, (*gcci)->getBoundingBox() );
        }

        return( bbox );
      }

      Sphere3f Switch::calculateBoundingSphere() const
      {
        Sphere3f sphere;
        Group::ChildrenConstIterator gcci = beginChildren();
        unsigned int previousIndex = 0;
        const SwitchMask & aMask = activeMask();

        for ( set<unsigned int>::const_iterator it=aMask.begin() ; it!=aMask.end() ; ++it )
        {
          DP_ASSERT( *it < getNumberOfChildren() );

          std::advance( gcci, *it - previousIndex );
          previousIndex = *it;

          sphere = boundingSphere( sphere, (*gcci)->getBoundingSphere() );
        }
        return( sphere );
      }

      unsigned short Switch::determineHintsContainment( unsigned short hints ) const
      {
        Group::ChildrenConstIterator gcci = beginChildren();
        unsigned int previousIndex = 0;
        const SwitchMask & aMask = activeMask();
        unsigned short containment = Node::determineHintsContainment( hints );

        for ( set<unsigned int>::const_iterator it=aMask.begin()
            ; ( (containment & hints ) != hints ) && it!=aMask.end()
            ; ++it )
        {
          DP_ASSERT( *it < getNumberOfChildren() );

          std::advance( gcci, *it - previousIndex );
          previousIndex = *it;

          containment |= (*gcci)->getContainedHints( hints );
        }
        return( containment );
      }

      bool Switch::determineTransparencyContainment() const
      {
        Group::ChildrenConstIterator gcci = beginChildren();
        unsigned int previousIndex = 0;
        const SwitchMask & aMask = activeMask();
        bool containsTransparent = false;

        for ( set<unsigned int>::const_iterator it=aMask.begin() ; !containsTransparent && it!=aMask.end() ; ++it )
        {
          DP_ASSERT( *it < getNumberOfChildren() );

          std::advance( gcci, *it - previousIndex );
          previousIndex = *it;

          containsTransparent = (*gcci)->containsTransparency();
        }
        return( containsTransparent );
      }

      Switch & Switch::operator=(const Switch & rhs)
      {
        if (&rhs != this)
        {
          Group::operator=(rhs);
          m_masks = rhs.m_masks;
          m_activeMaskKey = rhs.m_activeMaskKey;
          setActiveMaskKey( m_activeMaskKey );
          notify( PropertyEvent( this, PID_ActiveSwitchMask) );
          notify( PropertyEvent( this, PID_ActiveMaskKey) );
        }
        return *this;
      }

      bool Switch::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<Switch>() && Group::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          SwitchSharedPtr const& s = object.staticCast<Switch>();
          equi = ( m_activeMaskKey == s->m_activeMaskKey &&
                   m_masks == s->m_masks );
        }
        return( equi );
      }

      void 
      Switch::setActiveMaskKey( MaskKey k )
      {
        map<MaskKey,SwitchMask>::iterator iter = m_masks.find( k );
        SwitchMask & aMask = activeMask();

        // make sure we found it!!
        if( iter != m_masks.end() && 
            k != m_activeMaskKey )
        {
          SwitchMask & newActiveMask = (*iter).second;
          SwitchMask uunion, deactivate, activate;

          // get differences between two masks
          set_union( aMask.begin(), aMask.end(),
                     newActiveMask.begin(), newActiveMask.end(),
                     inserter( uunion, uunion.begin() ) );

          // difference between union and old will be those to activate
          set_difference( uunion.begin(), uunion.end(),
                          aMask.begin(), aMask.end(),
                          inserter( activate, activate.begin() ) );

          // difference between union and new will be those to deactivate
          set_difference( uunion.begin(), uunion.end(),
                          newActiveMask.begin(), newActiveMask.end(),
                          inserter( deactivate, deactivate.begin() ) );

          SwitchMask::iterator siter;

          // deactivate these
          siter = deactivate.begin();
          while( siter != deactivate.end() )
          {
            postDeactivateChild(*siter);
            ++siter;
          }

          // set current
          m_activeMaskKey = k;

          // activate these
          siter = activate.begin();
          while( siter != activate.end() )
          {
            postActivateChild(*siter);
            ++siter;
          }
          notify( PropertyEvent( this, PID_ActiveMaskKey ) );
        }
      }

      void 
      Switch::addMask( MaskKey key, const SwitchMask & sm )
      {
      #if !defined(NDEBUG)
        if ( m_masks.find(key)!=m_masks.end() ) 
        {
    #if 0
          NVSG_TRACE_OUT_F(("replacing switch mask %X", key))
    #endif
        }
      #endif
        // insert it 
        m_masks[ key ] = sm;
    
        if ( key == m_activeMaskKey )
        {
          //TODO: should this only notify if the mask really changed? expensive to test.
          notify( PropertyEvent( this, PID_ActiveMaskKey ) );
        }
        else
        {
          notify( Object::Event( this ) );
        }
      }

      bool 
      Switch::removeMask( MaskKey key )
      {
        // you may not remove the default mask
        DP_ASSERT( key != DEFAULT_MASK_KEY );

        map<MaskKey,SwitchMask>::iterator iter = m_masks.find( key );

        // make sure we found it!!
        if( iter != m_masks.end() )
        {
          // if they erase the current mask, set the mask to
          // be default.
          if( key == m_activeMaskKey )
          {
            setActiveMaskKey( DEFAULT_MASK_KEY );
          }

          // remove it
          m_masks.erase( iter );
          notify( Object::Event( this ) );

          return true;
        }
        else
        {
          return false;
        }
      }

      const Switch::SwitchMask& Switch::getActiveSwitchMask() const
      {
        return activeMask();
      }


      unsigned int Switch::getNumberOfMasks() const 
      { 
        return( checked_cast<unsigned int>(m_masks.size()) ); 
      }

      Switch::MaskIterator 
      Switch::getFirstMaskIterator() const 
      { 
        return m_masks.begin(); 
      }

      Switch::MaskIterator 
      Switch::getLastMaskIterator() const 
      { 
        return m_masks.end(); 
      }

      Switch::MaskIterator
      Switch::getCurrentMaskIterator() const
      {
        MaskIterator it = m_masks.find( m_activeMaskKey );
        DP_ASSERT( it != m_masks.end() );
        return it;
      }

      Switch::MaskIterator 
      Switch::getNextMaskIterator(Switch::MaskIterator it) const 
      {
        return ++it;
      }

      Switch::MaskKey 
      Switch::getMaskKey(Switch::MaskIterator it) const 
      { 
        return it->first; 
      }

      const Switch::SwitchMask &
      Switch::getSwitchMask(Switch::MaskIterator it) const 
      { 
        return it->second; 
      }

      Switch::MaskKey 
      Switch::getActiveMaskKey() const 
      { 
        return m_activeMaskKey; 
      }

      void Switch::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Group::feedHashGenerator( hg );
        for ( std::map<MaskKey,SwitchMask>::const_iterator it = m_masks.begin() ; it != m_masks.end() ; ++it )
        {
          hg.update( reinterpret_cast<const unsigned char *>(&it->first), sizeof(it->first) );
          for ( SwitchMask::const_iterator smit = it->second.begin() ; smit != it->second.end() ; ++smit )
          {
            hg.update( reinterpret_cast<const unsigned char *>(&*smit), sizeof(*smit) );
          }
        }
        hg.update( reinterpret_cast<const unsigned char *>(&m_activeMaskKey), sizeof(m_activeMaskKey) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
