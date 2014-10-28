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
#include <dp/util/Reflection.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Switch.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {
      SMART_TYPES( ObjectObserver );

      class ObjectObserver : public Observer<ObjectTreeIndex>
      {
      public:
        struct CacheData
        {
          unsigned int m_hints;
          unsigned int m_mask;
        };
        typedef std::map< ObjectTreeIndex, CacheData >  NewCacheData;

      public:
        virtual ~ObjectObserver();

        static SmartObjectObserver create( SceneTreeWeakPtr sceneTree )
        {
          return( std::shared_ptr<ObjectObserver>( new ObjectObserver( sceneTree ) ) );
        }

        void attach( const dp::sg::core::ObjectWeakPtr& obj, ObjectTreeIndex index );
        virtual void onDetach( ObjectTreeIndex index );

        void popNewCacheData( NewCacheData & currentData ) const 
        {
          currentData = m_newCacheData;
          m_newCacheData.clear();
        }

      protected:
        ObjectObserver( const SceneTreeWeakPtr& sceneTree ) : Observer<ObjectTreeIndex>( sceneTree )
        {
        }
        void onNotify( const dp::util::Event &event, dp::util::Payload *payload );
        virtual void onPreRemoveChild( const dp::sg::core::Group *group, dp::sg::core::NodeSharedPtr const & child, unsigned int index, Payload* payload );
        virtual void onPostAddChild( dp::sg::core::Group const *group, dp::sg::core::NodeSharedPtr const & child, unsigned int index, Payload* payload );

      private:
        mutable NewCacheData m_newCacheData;
      };

    } // namespace xbar
  } // namespace sg
} // namespace dp
