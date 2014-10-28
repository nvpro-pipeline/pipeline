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


#include <dp/sg/renderer/rix/gl/Config.h>
#include <dp/sg/renderer/rix/gl/inc/ResourceManager.h>

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

          ResourceManager::ResourceObserver* ResourceManager::ResourceObserver::clone() const
          {
            return new ResourceManager::ResourceObserver( *this );
          }

          /*******************************/
          /** ResourceManager::Resource **/
          /*******************************/
          ResourceManager::SmartResource ResourceManager::Resource::create()
          {
            return( std::shared_ptr<ResourceManager::Resource>( new ResourceManager::Resource() ) );
          }

          ResourceManager::SmartResource ResourceManager::Resource::create( size_t key, SmartResourceManager const& resourceManager )
          {
            return( std::shared_ptr<ResourceManager::Resource>( new ResourceManager::Resource( key, resourceManager ) ) );
          }

          ResourceManager::Resource::Resource( )
            : m_resourceManager( nullptr )
          {
          }

          ResourceManager::Resource::Resource( size_t key, const SmartResourceManager& resourceManager )
            : m_resourceManager( resourceManager )
            , m_key( key )
          {
            assert( resourceManager );
            m_resourceManager->registerResource( m_key, this );
          }

          ResourceManager::Resource::~Resource( )
          {
            if ( m_resourceManager )
            {
              m_resourceManager->unregisterResource( m_key );
            }
            if ( m_payload )
            {
              m_payload->m_resource = nullptr;
              m_payload->m_isDirty = false;
            }
          }

          void ResourceManager::Resource::update()
          {
          }

          bool ResourceManager::Resource::update(const dp::util::Event& event )
          {
            return false;
          }

          void ResourceManager::Resource::setPayload( const SmartPayload &payload )
          {
            m_payload = payload;
            m_payload->m_resource = this;
          }

          const ResourceManager::SmartPayload& ResourceManager::Resource::getPayload() const
          {
            return m_payload;
          }

          const dp::sg::core::HandledObjectSharedPtr& ResourceManager::Resource::getHandledObject() const
          {
            return dp::sg::core::HandledObjectSharedPtr::null;
          }

          /*******************************/
          /** ResourceManager::Payload  **/
          /*******************************/
          ResourceManager::SmartPayload ResourceManager::Payload::create( ResourceManager::Resource * resource )
          {
            return( std::shared_ptr<ResourceManager::Payload>( new ResourceManager::Payload( resource ) ) );
          }

          ResourceManager::Payload::Payload( ResourceManager::Resource *resource )
            : m_resource( resource )
            , m_isDirty( false )
          {
          }

          /***************************************/
          /** ResourceManager::ResourceObserver **/
          /***************************************/
          ResourceManager::SmartResourceObserver ResourceManager::ResourceObserver::create()
          {
            return( std::shared_ptr<ResourceManager::ResourceObserver>( new ResourceManager::ResourceObserver() ) );
          }

          ResourceManager::ResourceObserver::ResourceObserver()
          {
          }


          void ResourceManager::ResourceObserver::subscribe( Resource *resource )
          {
            DP_ASSERT( resource );
            DP_ASSERT( resource->getHandledObject() );

            ResourceManager::SmartPayload payload = ResourceManager::Payload::create( resource );
            resource->setPayload( payload );

            dp::util::Subject* subject = dynamic_cast<dp::util::Subject*>(resource->getHandledObject().operator->());
            subject->attach( static_cast<dp::util::Observer*>(this), payload.getWeakPtr() );
          }

          void ResourceManager::ResourceObserver::unsubscribe( Resource *resource )
          {
            DP_ASSERT( resource );
            DP_ASSERT( resource->getHandledObject() );

            if ( resource->getHandledObject() )
            {
              dp::util::Subject* subject = dynamic_cast<dp::util::Subject*>(resource->getHandledObject().operator->());
              subject->detach( static_cast<dp::util::Observer*>(this), resource->getPayload().getWeakPtr() );
            }
          }

          void ResourceManager::ResourceObserver::onNotify( const dp::util::Event &event, dp::util::Payload *payload )
          {
            ResourceManager::Payload* p = static_cast<ResourceManager::Payload*>(payload);
            if ( !p->m_isDirty )
            {
              if ( !p->m_resource->update( event ) )
              {
                p->m_isDirty = true;
                m_dirtyPayloads.push_back( std::static_pointer_cast<ResourceManager::Payload>(p->shared_from_this()) );
              }
            }
          }

          void ResourceManager::ResourceObserver::updateResources()
          {
            std::vector<ResourceManager::SmartPayload>::iterator it, itEnd = m_dirtyPayloads.end();
            for ( it = m_dirtyPayloads.begin(); it != itEnd; ++it )
            {
              ResourceManager::SmartPayload &p = *it;
              if ( p->m_isDirty )
              {
                if ( p->m_resource ) // TODO might not be required if dirty is being set to false on destruction
                {
                  p->m_resource->update();
                }
                p->m_isDirty = false;
              }
            }
            m_dirtyPayloads.clear();
          }

          /*********************/
          /** ResourceManager **/
          /*********************/
          SmartResourceManager ResourceManager::create( dp::rix::core::Renderer* renderer, dp::fx::Manager shaderManagerType )
          {
            return( std::shared_ptr<ResourceManager>( new ResourceManager( renderer, shaderManagerType ) ) );
          }

          ResourceManager::ResourceManager( dp::rix::core::Renderer *renderer, dp::fx::Manager shaderManagerType )
            : m_resourceObserver( ResourceObserver::create() )
            , m_renderer ( renderer )
            , m_shaderManagerType( shaderManagerType )
            , m_bufferAllocator( renderer )
          {
          }

          ResourceManager::~ResourceManager()
          {
          }

          void ResourceManager::registerResource( size_t key, const WeakResource &resource )
          {
            resource->m_resourceManager = shared_from_this();
            m_resources[key] = resource;
          }

          void ResourceManager::unregisterResource( size_t key )
          {
            ResourceMap::iterator it = m_resources.find( key );
            if ( it != m_resources.end() )
            {
              DP_ASSERT( !shared_from_this().unique() && "There's no reference to the ResourceManager left!" );
              Resource* res = it->second;
              m_resources.erase( it );
              res->m_resourceManager = SmartResourceManager::null;
            }
          }

          void ResourceManager::subscribe( Resource *resource )
          {
            m_resourceObserver->subscribe( resource );
          }

          void ResourceManager::unsubscribe( Resource *resource )
          {
            m_resourceObserver->unsubscribe( resource );
          }


          void ResourceManager::updateResources()
          {
            m_resourceObserver->updateResources( );
          }

          dp::rix::core::Renderer* ResourceManager::getRenderer() const
          {
            return m_renderer;
          }

        } // namespace gl
      } // namespace rix
    } // namespace renderer
  } // namespace sg
} // namespace dp
