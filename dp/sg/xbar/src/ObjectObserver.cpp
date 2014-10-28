// Copyright NVIDIA Corporation 2010-2013
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


#include <dp/sg/xbar/inc/ObjectObserver.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      ObjectObserver::~ObjectObserver()
      {
      }

      void ObjectObserver::attach( const dp::sg::core::ObjectWeakPtr& obj, ObjectTreeIndex index )
      {
        DP_ASSERT( m_indexMap.find(index) == m_indexMap.end() );

        SmartPayload payload( Payload::create( index ) );
        Observer<ObjectTreeIndex>::attach( obj, payload );

        // fill cache data entry with current data
        CacheData data;
        data.m_hints = obj->getHints();
        data.m_mask  = obj->getTraversalMask();

        m_newCacheData[index] = data;
      }

      void ObjectObserver::onDetach( ObjectTreeIndex index )
      {
        // remove NewCacheData entry
        NewCacheData::iterator it = m_newCacheData.find( index );
        if (it != m_newCacheData.end() )
        {
          m_newCacheData.erase( it );
        }
      }

      void ObjectObserver::onNotify( const dp::util::Event &event, dp::util::Payload *payload )
      {
        switch ( event.getType() )
        {
        case dp::util::Event::PROPERTY:
          {
            dp::util::Reflection::PropertyEvent const& propertyEvent = static_cast<dp::util::Reflection::PropertyEvent const&>(event);
            dp::util::PropertyId propertyId = propertyEvent.getPropertyId();
            if( propertyId == dp::sg::core::Object::PID_Hints || propertyId == dp::sg::core::Object::PID_TraversalMask )
            {
              dp::sg::core::Object const* o = static_cast<dp::sg::core::Object const*>(propertyEvent.getSource());

              CacheData data;
              data.m_hints = o->getHints();
              data.m_mask  = o->getTraversalMask();

              m_newCacheData[static_cast<Payload*>(payload)->m_index] = data;
            }
          }
          break;
        case dp::util::Event::DP_SG_CORE:
          {
            dp::sg::core::Event const& coreEvent = static_cast<dp::sg::core::Event const&>(event);
            if ( coreEvent.getType() == dp::sg::core::Event::GROUP )
            {
              dp::sg::core::Group::Event const& groupEvent = static_cast<dp::sg::core::Group::Event const&>(coreEvent);
              Payload* groupPayload = static_cast<Payload*>(payload);

              switch ( groupEvent.getType() )
              {
              case dp::sg::core::Group::Event::POST_CHILD_ADD:
                onPostAddChild( groupEvent.getGroup(), groupEvent.getChild(), groupEvent.getIndex(), groupPayload );
                break;
              case dp::sg::core::Group::Event::PRE_CHILD_REMOVE:
                onPreRemoveChild( groupEvent.getGroup(), groupEvent.getChild(), groupEvent.getIndex(), groupPayload );
                break;
              case dp::sg::core::Group::Event::POST_GROUP_EXCHANGED:
                m_sceneTree->replaceSubTree( dp::util::getWeakPtr<dp::sg::core::Node>(groupEvent.getGroup()), groupPayload->m_index );
                break;
              case dp::sg::core::Group::Event::CLIP_PLANES_CHANGED:
                DP_ASSERT( !"clipplanes not supported" );
                break;
              }
            }
          }
          break;
        }
      }

      void ObjectObserver::onPreRemoveChild( const dp::sg::core::Group *group, dp::sg::core::NodeSharedPtr const & child, unsigned int index, Payload* payload )
      {
        ObjectTreeIndex objectIndex = payload->m_index;

        // determine the index of child in the ObjectTree3
        ObjectTree& tree = m_sceneTree->getObjectTree();
        unsigned int i=0;
        ObjectTreeIndex childIndex = tree[objectIndex].m_firstChild;

        bool handleAsGroup = true;
        if ( group->getObjectCode() == dp::sg::core::OC_SWITCH )
        {
          dp::sg::core::Switch const * s = static_cast<dp::sg::core::Switch const *>( group );
          handleAsGroup = !!s->getHints( dp::sg::core::Object::DP_SG_HINT_DYNAMIC );
        }

        if ( handleAsGroup )
        {
          DP_ASSERT( childIndex != ~0 );
          while( i != index )
          {
            childIndex = tree[childIndex].m_nextSibling;
            DP_ASSERT( childIndex != ~0 );
            ++i;
          }
        }
        else
        {
          DP_ASSERT( group->getObjectCode() == dp::sg::core::OC_SWITCH );
          dp::sg::core::Switch const * s = static_cast<dp::sg::core::Switch const*>( group );
          if ( ! s->isActive( index ) )
          {
            childIndex = ~0;
          }
          while ( ( childIndex != ~0 ) && ( i != index ) )
          {
            if ( s->isActive( i ) )
            {
              childIndex = tree[childIndex].m_nextSibling;
            }
            ++i;
          }
        }

        if ( childIndex != ~0 )
        {
          m_sceneTree->removeObjectTreeIndex( childIndex );
        }
      }

      void ObjectObserver::onPostAddChild( dp::sg::core::Group const *group, dp::sg::core::NodeSharedPtr const & child, unsigned int index, Payload* payload )
      {
        ObjectTreeIndex objectIndex = payload->m_index;

        // find the left sibling and the left transform of our new child
        ObjectTree& tree = m_sceneTree->getObjectTree();
        TransformTreeIndex parentTransformIndex = tree[objectIndex].m_transformIndex;
        ObjectTreeIndex leftSibling = tree[objectIndex].m_firstChild; 
        TransformTreeIndex leftSiblingTransformIndex = ~0;
        ObjectTreeIndex currentLeftSibling = ~0;

        while ( index > 0 )
        {
          if( tree[leftSibling].m_transformIndex != parentTransformIndex )
          {
            leftSiblingTransformIndex = tree[leftSibling].m_transformIndex;
          }
          currentLeftSibling = leftSibling;
          leftSibling = tree[leftSibling].m_nextSibling;
          --index;
        }

        // leftSibling advanced one step too far. 
        leftSibling = currentLeftSibling;

        m_sceneTree->addSubTree( child.getWeakPtr(), objectIndex, leftSibling, parentTransformIndex, leftSiblingTransformIndex );
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
