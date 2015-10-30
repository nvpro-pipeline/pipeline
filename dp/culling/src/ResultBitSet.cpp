// Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/culling/ResultBitSet.h>
#include <dp/util/BitArray.h>

#include <dp/util/FrameProfiler.h>

namespace dp
{
  namespace culling
  {
      ResultBitSetSharedPtr ResultBitSet::create( GroupBitSetSharedPtr const& parentGroup )
      {
        return( std::shared_ptr<ResultBitSet>( new ResultBitSet( parentGroup ) ) );
      }

      ResultBitSet::ResultBitSet( GroupBitSetSharedPtr const& parentGroup )
        : m_groupParent( parentGroup )
        , m_objectIncarnation(~0)
        , m_results(0)
      {
        DP_ASSERT( m_groupParent );

        m_groupParent->attach( this );
      }

      ResultBitSet::~ResultBitSet()
      {
        m_groupParent->detach( this );
      }

      std::vector<ObjectSharedPtr> const & ResultBitSet::getChangedObjects() const
      {
        return m_changedObjects;
      }

      void ResultBitSet::updateChanged( uint32_t const* visibility )
      {
        dp::util::ProfileEntry p("ResultBitSet::updateChanged");

        if ( m_objectIncarnation != m_groupParent->getObjectIncarnation() )
        {
          m_objectIncarnation = m_groupParent->getObjectIncarnation();
          m_groupParent->getObjectCount();

          size_t numberOfResultChunks = ( m_groupParent->getObjectCount() + 31 ) / 32;
          size_t oldSize = m_results.getSize();
          m_results.resize( m_groupParent->getObjectCount() );

          // objects are visible by default, TODO required?
          for ( size_t index = oldSize; index < m_results.getSize(); ++index )
          {
            m_results.enableBit( index );
          }
        }
        /** \brief Visitor which adds changed objects to the changed group **/
        struct Visitor
        {
          inline Visitor( GroupBitSetSharedPtr const & group, std::vector<ObjectSharedPtr> & changed )
            : m_group( group )
            , m_changed( changed )
          {
            m_changed.clear();
          }

          inline void operator()( size_t index )
          {
            const ObjectBitSetSharedPtr& objectImpl = m_group->getObject( index );
            m_changed.push_back( objectImpl );
          }
        private:
          GroupBitSetSharedPtr const & m_group;
          std::vector<ObjectSharedPtr> & m_changed;
        };

        dp::util::BitArray newVisible( m_groupParent->getObjectCount() );
        newVisible.setBits( visibility, m_groupParent->getObjectCount() );

        dp::util::BitArray changed = newVisible ^ m_results;

        Visitor visitorBase( m_groupParent, m_changedObjects );
        changed.traverseBits(visitorBase);
        m_results = newVisible;
      }

      void ResultBitSet::onNotify( dp::util::Event const & event, dp::util::Payload * payload )
      {
        // If an object is being moved in the internal array move the visibility bit in the result to the new location.
        GroupBitSet::Event const& groupEvent = static_cast<GroupBitSet::Event const&>(event);

        if ( groupEvent.getNewIndex() < m_results.getSize() )
        {
          if ( groupEvent.getOldIndex() < m_results.getSize() )
          {
            // transfer visibility bit
            m_results.setBit( groupEvent.getNewIndex(), m_results.getBit( groupEvent.getOldIndex() ) );
          }
          else
          {
            // object from not yet known location. assume visiblity true
            m_results.enableBit( groupEvent.getNewIndex() );
          }
        }
      }

      void ResultBitSet::onDestroyed( dp::util::Subject const & subject, dp::util::Payload * payload )
      {
        throw std::runtime_error("The method or operation is not implemented.");
      }

  } // namespace culling
} // namespace dp
