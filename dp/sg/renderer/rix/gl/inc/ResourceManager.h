// Copyright NVIDIA Corporation 2011
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

#include <dp/sg/xbar/xbar.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/util/RCObject.h>
#include <dp/util/SmartPtr.h>
#include <dp/sg/core/HandledObject.h>
#include <dp/sg/renderer/rix/gl/inc/BufferAllocator.h>
#include <dp/util/Observer.h>
#include <dp/fx/EffectDefs.h>

#include <set>

namespace dp
{
  namespace sg
  {
    namespace renderer
    {
      namespace rix
      {
        namespace gl
        {

          class ResourceManager;
          typedef dp::util::SmartPtr<ResourceManager> SmartResourceManager;

          class ResourceManager : public dp::util::RCObject
          {
          public:
            class Resource;

            class Payload : public dp::util::Payload, public dp::util::RCObject
            {
            public:
              Payload( Resource *resource );

              bool                     m_isDirty;
              Resource*                m_resource;
              dp::rix::core::Renderer* m_renderer;
            };

            typedef dp::util::SmartPtr<Payload> SmartPayload;

            /** Baseclass for all resources managed by the ResourceManager **/
            class Resource : public dp::util::RCObject
            {
            public:
              Resource( );
              Resource( size_t key, const SmartResourceManager &manager );
              virtual ~Resource();

              virtual void update(); // update resource
              /** \brief Called upon dp::util::Observer::notify
                  \param event The event passed over to notify
                  \return true if the event has been processed or false if Resource::update should be called during ResourceManager::update.
              **/
              virtual bool update( const dp::util::Event& event);

              void setPayload( const SmartPayload &payload );
              const SmartPayload &getPayload() const;
        
              virtual const dp::sg::core::HandledObjectSharedPtr& getHandledObject() const;
            protected:
              friend class ResourceManager;

              SmartResourceManager         m_resourceManager; // TODO smartptr here? keep resourcemanager as long as a resource exists or notify all resources that resourcemanager is no longer available
              SmartPayload                 m_payload;
              size_t                       m_key; // TODO remove key and move unregister to derived classes?
            };

            typedef dp::util::SmartPtr<Resource> SmartResource;
            typedef Resource*                  WeakResource;

            /********************/
            /* ResourceObserver */
            /********************/
            class ResourceObserver : public dp::util::Observer, public dp::util::RCObject
            {
            public:
              ResourceObserver();

              void subscribe( Resource* resource );
              void unsubscribe( Resource* resource );

              virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload );
              virtual void onDestroyed( const dp::util::Subject& /*subject*/, dp::util::Payload* /*payload*/ ) { }; // TODO make abstract once texture change had been submitted

              void updateResources();
            protected:
              std::set<Resource *>      m_subscribedResources;
              mutable std::vector<ResourceManager::SmartPayload> m_dirtyPayloads;
            private:
              virtual ResourceObserver* clone() const;
            };

            typedef dp::util::SmartPtr<ResourceObserver> SmartResourceObserver;
            typedef ResourceObserver*                    WeakResourceObserver;

            /*******************/
            /* ResourceManager */
            /*******************/
            ResourceManager( dp::rix::core::Renderer* renderer, dp::fx::Manager shaderManagerType );
            ~ResourceManager();

            template <typename ResourceType> ResourceType* getResource( size_t key );
            void registerResource( size_t key, const WeakResource &m_resource );
            void unregisterResource( size_t key );

            void subscribe( Resource* resource );
            void unsubscribe( Resource* resource );

            void updateResources();

            dp::rix::core::Renderer* getRenderer() const;
            BufferAllocator &getBufferAllocator() { return m_bufferAllocator; }

            dp::fx::Manager getShaderManagerType() const { return m_shaderManagerType; }

          private:
            typedef std::map<size_t, Resource*> ResourceMap;

            dp::fx::Manager                     m_shaderManagerType;
            ResourceMap                         m_resources;
            SmartResourceObserver               m_resourceObserver;
            dp::rix::core::Renderer*            m_renderer;
            BufferAllocator                     m_bufferAllocator;
          };

          // TODO move actual code to cpp and write template wrapper for cast? What about ABI compatibility
          template <typename ResourceType>
          ResourceType* ResourceManager::getResource( size_t key )
          {
            ResourceMap::iterator it = m_resources.find( key );
            DP_ASSERT( it == m_resources.end() || (it != m_resources.end() && dynamic_cast<ResourceType*>( it->second )));
            return it != m_resources.end() ? static_cast<ResourceType*>(it->second) : nullptr;
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

