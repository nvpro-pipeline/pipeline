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
          DEFINE_PTR_TYPES( ResourceManager );

          class ResourceManager : public std::enable_shared_from_this<ResourceManager>
          {
          public:
            class Resource;

            DEFINE_PTR_TYPES( Payload );

            class Payload : public dp::util::Payload
            {
            public:
              static PayloadSharedPtr create( Resource * resource );

            public:
              bool                     m_isDirty;
              Resource*                m_resource;
              dp::rix::core::Renderer* m_renderer;

            protected:
              Payload( Resource *resource );
            };

            DEFINE_PTR_TYPES( Resource );

            /** Baseclass for all resources managed by the ResourceManager **/
            class Resource : public std::enable_shared_from_this<Resource>
            {
            public:
              static ResourceSharedPtr create();
              static ResourceSharedPtr create( size_t key, ResourceManagerSharedPtr const& manager );
              virtual ~Resource();

              virtual void update(); // update resource
              /** \brief Called upon dp::util::Observer::notify
                  \param event The event passed over to notify
                  \return true if the event has been processed or false if Resource::update should be called during ResourceManager::update.
              **/
              virtual bool update( const dp::util::Event& event);

              void setPayload( const PayloadSharedPtr &payload );
              const PayloadSharedPtr &getPayload() const;

              virtual const dp::sg::core::HandledObjectSharedPtr& getHandledObject() const;

            protected:
              Resource( );
              Resource( size_t key, const ResourceManagerSharedPtr &manager );

            protected:
              friend class ResourceManager;

              ResourceManagerSharedPtr  m_resourceManager; // TODO smartptr here? keep resourcemanager as long as a resource exists or notify all resources that resourcemanager is no longer available
              PayloadSharedPtr          m_payload;
              size_t                    m_key; // TODO remove key and move unregister to derived classes?
            };

            typedef Resource*                  WeakResource;

            /********************/
            /* ResourceObserver */
            /********************/
            DEFINE_PTR_TYPES( ResourceObserver );

            class ResourceObserver : public dp::util::Observer
            {
            public:
              static ResourceObserverSharedPtr create();
              void subscribe( Resource* resource );
              void unsubscribe( Resource* resource );

              virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload );
              virtual void onDestroyed( const dp::util::Subject& /*subject*/, dp::util::Payload* /*payload*/ ) { }; // TODO make abstract once texture change had been submitted

              void updateResources();

            protected:
              ResourceObserver();

            protected:
              std::set<Resource *>                                    m_subscribedResources;
              mutable std::vector<ResourceManager::PayloadSharedPtr>  m_dirtyPayloads;

            private:
              virtual ResourceObserver* clone() const;
            };

            typedef ResourceObserver*                    WeakResourceObserver;

            /*******************/
            /* ResourceManager */
            /*******************/
            static ResourceManagerSharedPtr create( dp::rix::core::Renderer* renderer, dp::fx::Manager shaderManagerType );
            ~ResourceManager();

            template <typename ResourceType> dp::util::SharedPtr<ResourceType> getResource( size_t key );
            void registerResource( size_t key, const WeakResource &m_resource );
            void unregisterResource( size_t key );

            void subscribe( Resource* resource );
            void unsubscribe( Resource* resource );

            void updateResources();

            dp::rix::core::Renderer* getRenderer() const;
            BufferAllocator &getBufferAllocator() { return m_bufferAllocator; }

            dp::fx::Manager getShaderManagerType() const { return m_shaderManagerType; }

          protected:
            ResourceManager( dp::rix::core::Renderer* renderer, dp::fx::Manager shaderManagerType );

          private:
            typedef std::map<size_t, Resource*> ResourceMap;

            dp::fx::Manager             m_shaderManagerType;
            ResourceMap                 m_resources;
            ResourceObserverSharedPtr   m_resourceObserver;
            dp::rix::core::Renderer   * m_renderer;
            BufferAllocator             m_bufferAllocator;
          };

          // TODO move actual code to cpp and write template wrapper for cast? What about ABI compatibility
          template <typename ResourceType>
          dp::util::SharedPtr<ResourceType> ResourceManager::getResource( size_t key )
          {
            ResourceMap::iterator it = m_resources.find( key );
            DP_ASSERT( it == m_resources.end() || (it != m_resources.end() && dynamic_cast<ResourceType*>( it->second )));
            return it != m_resources.end() ? dp::util::SharedPtr<ResourceType>( *reinterpret_cast<std::shared_ptr<ResourceType>*>(&it->second->shared_from_this()) ) : dp::util::SharedPtr<ResourceType>::null;
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp

