// Copyright NVIDIA Corporation 2010-2015
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

#include <dp/sg/xbar/SceneTree.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {

      template <typename IndexType>
      class Observer : public dp::util::Observer
      {
      public:
        DEFINE_PTR_TYPES( Payload );

        class Payload : public dp::util::Payload
        {
        public:
          static PayloadSharedPtr create( IndexType index )
          {
            return( std::shared_ptr<Payload>( new Payload( index ) ) );
          }

        public:
          IndexType m_index;

        protected:
          Payload( IndexType index )
            : m_index( index )
          {
          }
        };

        Observer( SceneTreeSharedPtr const& sceneTree );
        virtual ~Observer();

        void attach( dp::util::SubjectSharedPtr const& subject, PayloadSharedPtr const& payload );
        void detach( IndexType index );
        void detachAll();

        virtual void onDestroyed( dp::util::Subject const& subject, dp::util::Payload * payload );
      protected:
        virtual void onDetach( IndexType index ) {};

        typedef std::multimap<ObjectTreeIndex, std::pair<dp::util::SubjectWeakPtr, PayloadSharedPtr> > IndexMap;
        IndexMap m_indexMap;
        SceneTreeSharedPtr m_sceneTree;
      };

      template <typename IndexType>
      Observer<IndexType>::Observer( SceneTreeSharedPtr const& sceneTree )
        : m_sceneTree( sceneTree )
      {
      }

      template <typename IndexType>
      Observer<IndexType>::~Observer()
      {
        detachAll();
      }

      template <typename IndexType>
      void Observer<IndexType>::attach( dp::util::SubjectSharedPtr const& subject, PayloadSharedPtr const& payload )
      {
        m_indexMap.insert( std::make_pair(payload->m_index, std::make_pair( subject.getWeakPtr(), payload ) ) );
        subject->attach( this, payload.operator->() );    // BIG HACK!! we somehow need to align dp::util::Payload and dp::sg::xbar::Observer<IndexType::Payload
      }

      template <typename IndexType>
      void Observer<IndexType>::detach( IndexType index )
      {
        onDetach( index );

        typename IndexMap::iterator it = m_indexMap.find( index );
        DP_ASSERT( it != m_indexMap.end() );

        it->second.first->detach( this, it->second.second.operator->() );    // BIG HACK!! we somehow need to align dp::util::Payload and dp::sg::xbar::Observer<IndexType::Payload

        m_indexMap.erase( it );
      }

      template <typename IndexType>
      void Observer<IndexType>::detachAll( )
      {
        typename IndexMap::iterator it, it_end = m_indexMap.end();
        for( it = m_indexMap.begin(); it != it_end; ++it )
        {
          it->second.first->detach( this, it->second.second.operator->() );    // BIG HACK!! we somehow need to align dp::util::Payload and dp::sg::xbar::Observer<IndexType::Payload
        }
        m_indexMap.clear();
      }


      template <typename IndexType>
      void Observer<IndexType>::onDestroyed( dp::util::Subject const& subject, dp::util::Payload * payload )
      {
        DP_ASSERT( dynamic_cast<Payload*>(payload) );
        Payload const* p = static_cast<Payload const*>(payload);

        onDetach( p->m_index );

        typename IndexMap::iterator it = m_indexMap.find( p->m_index );
        DP_ASSERT( it != m_indexMap.end() );
        m_indexMap.erase( it );
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
