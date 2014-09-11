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


#include "RenderGroupGL.h"
#include "ContainerGL.h"
#include "GeometryInstanceGL.h"
#include "RenderEngineGL.h"

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      /************************************************************************/
      /* RenderGroupGL::Cache                                                 */
      /************************************************************************/

      void RenderGroupGL::Cache::addGeometryInstance( GeometryInstanceGLHandle gi )
      {
        gi->m_payload.m_position = dp::util::checked_cast<dp::util::Uint32>(m_geometryInstances.size());
        gi->m_payload.m_gi = gi;

        gi->m_renderGroup = m_renderGroup;
        gi->m_pipelineGroupCache = this;

        handleRef( gi );

        m_geometryInstances.push_back( gi );
        m_dirty = true;

        gi->attach( m_renderGroup, &gi->m_payload );
      }

      void RenderGroupGL::Cache::removeGeometryInstance( GeometryInstanceGLHandle gi )
      {
        gi->detach( m_renderGroup, &gi->m_payload );

        dp::util::Uint32 pos = gi->m_payload.m_position;
        m_dirty = true;

        // replace gi with the one at the back to avoid moving gis
        m_geometryInstances[pos] = m_geometryInstances.back();
        m_geometryInstances[pos]->m_payload.m_position = pos;
        m_geometryInstances.pop_back();

        gi->m_pipelineGroupCache = nullptr;
        gi->m_pipelineGroupCacheIndex = ~0;
        gi->m_renderGroup = nullptr;
        gi->m_payload.m_position = ~0;

        handleUnref( gi );
      }

      /************************************************************************/
      /* RenderGroupGL                                                        */
      /************************************************************************/
      RenderGroupGL::RenderGroupGL( RenderEngineGL *renderEngine )
        : m_renderEngine( renderEngine )
      {
      }

      RenderGroupGL::~RenderGroupGL()
      {
        std::map< std::pair<ProgramPipelineGLHandle,ContainerDescriptorGLHandle>, DescriptorCache * >::iterator it;
        for ( it = m_descriptorCacheObjects.begin(); it != m_descriptorCacheObjects.end(); ++it )
        {
          delete it->second;
        }
      }

      void RenderGroupGL::addGeometryInstance( GeometryInstanceGLHandle gi )
      {
        if ( gi->m_payload.m_position == ~0u )
        {
          DP_ASSERT( gi->m_programPipeline );
          ProgramPipelineCaches::iterator it = m_programPipelineCaches.find( gi->m_programPipeline.get() );
          if ( it == m_programPipelineCaches.end() )
          {
            it = m_programPipelineCaches.insert( std::make_pair( gi->m_programPipeline.get(), m_renderEngine->createCache( this, gi->m_programPipeline ) ) ).first;
          }

          SmartCache& cache = it->second;
          cache->addGeometryInstance(gi);

          markDirty( gi );
        }
        else
        {
          DP_ASSERT( !"gi already added" );
          // TODO throw exception
        }
      }

      void RenderGroupGL::markDirty( GeometryInstanceGLHandle gi )
      {
        m_dirtyList.insert( gi );
      }

      void RenderGroupGL::removeGeometryInstance( GeometryInstanceGLHandle gi )
      {
        DP_ASSERT( gi->m_payload.m_position != ~0u );
        DP_ASSERT( gi->m_renderGroup == this );
        if ( gi->m_renderGroup == this )
        {
          ProgramPipelineCaches::iterator it = m_programPipelineCaches.find( gi->m_programPipeline.get() );
          DP_ASSERT( it != m_programPipelineCaches.end() );

          SmartCache& cache = it->second;
          cache->removeGeometryInstance(gi);

          std::set< GeometryInstanceGLHandle >::iterator itDirty = m_dirtyList.find( gi );
          if ( itDirty != m_dirtyList.end() )
          {
            m_dirtyList.erase( itDirty );
          }

          if ( cache->getGeometryInstances().empty() )
          {
            m_programPipelineCaches.erase( gi->m_programPipeline.get() );
          }
        }
      }

      void RenderGroupGL::setProgramPipeline( ProgramPipelineGLHandle /*programPipeline*/ )
      {
        DP_ASSERT(!"Not yet supported.");
      }

      void RenderGroupGL::useContainer( ContainerGLHandle container )
      {
        DP_ASSERT( container );

        m_globalContainers[container->m_descriptor.get()] = container;

        // notify all caches that this container had been added/updated
        for ( ProgramPipelineCaches::iterator it = m_programPipelineCaches.begin(); it != m_programPipelineCaches.end(); ++it )
        {
          it->second->useContainer( container);
        }
      }

      void RenderGroupGL::onNotify( dp::util::Event const & event, dp::util::Payload * /*payload*/ )
      {
        // listening only to GeometryInstances here
        const GeometryInstanceGL::Event& eventGeometryInstance = static_cast<const GeometryInstanceGL::Event&>(event);
        switch ( eventGeometryInstance.getEventType() )
        {
          case GeometryInstanceGL::CHANGED_DATA:
            markDirty( eventGeometryInstance.getGeometryInstance() );
          break;
          case GeometryInstanceGL::CHANGED_CONTAINER:
            for ( ProgramPipelineCaches::iterator it = m_programPipelineCaches.begin(); it != m_programPipelineCaches.end(); ++it )
            {
              it->second->onContainerExchanged();
            }
          break;
          case GeometryInstanceGL::CHANGED_VISIBILITY:
            // nothing to do
          break;
        }
      }

      void RenderGroupGL::onDestroyed( const dp::util::Subject& /*subject*/, dp::util::Payload* /*payload*/ )
      {
        DP_ASSERT( !"need to detach from something?" );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
