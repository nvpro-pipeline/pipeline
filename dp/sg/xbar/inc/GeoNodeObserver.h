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


#pragma once

#include <dp/sg/xbar/inc/Observer.h>
#include <dp/sg/core/GeoNode.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {
      DEFINE_PTR_TYPES( GeoNodeObserver );

      class GeoNodeObserver : public Observer<ObjectTreeIndex>
      {
      public:
        virtual ~GeoNodeObserver();

        static GeoNodeObserverSharedPtr create( const SceneTreeWeakPtr &sceneTree )
        {
          return( std::shared_ptr<GeoNodeObserver>( new GeoNodeObserver( sceneTree ) ) );
        }

        void attach( dp::sg::core::GeoNodeWeakPtr geoNode, ObjectTreeIndex index )
        {
          DP_ASSERT( m_indexMap.find( index ) == m_indexMap.end() );

          Observer<ObjectTreeIndex>::attach( geoNode, Payload::create( index ) );
        }

        virtual void onDetach( ObjectTreeIndex index )
        {
          m_dirtyGeoNodes.erase( index );
        }

        void popDirtyGeoNodes( ObjectTreeIndexSet & currentSet ) const 
        {
          currentSet = m_dirtyGeoNodes;
          m_dirtyGeoNodes.clear(); 
        }

      protected:        
        GeoNodeObserver( const SceneTreeWeakPtr &sceneTree ) : Observer<ObjectTreeIndex>( sceneTree )
        {
        }
        //TODO if the bounding volume has changed
        //TODO bounding volume does not get updated if any child of geonode changes
        virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload );

      private:
        mutable ObjectTreeIndexSet m_dirtyGeoNodes;
      };

    } // namespace xbar
  } // namespace sg
} // namespace dp
