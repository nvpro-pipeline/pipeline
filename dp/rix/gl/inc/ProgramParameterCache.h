// Copyright NVIDIA Corporation 2013-2015
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

#include <dp/rix/core/HandledObject.h>
#include <dp/rix/gl/inc/ParameterCacheStream.h>
#include <dp/rix/gl/inc/ProgramPipelineGL.h>
#include <dp/rix/gl/inc/RenderGroupGL.h>
#include <dp/rix/gl/inc/GeometryInstanceGL.h>
#include <vector>
#include <memory>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      /************************************************************************/
      /* ProgramParameterCache                                                */
      /************************************************************************/
      template <typename ParameterCacheType>
      class ProgramParameterCache
      {
      public:
        typedef typename ParameterCache<ParameterCacheType>::ContainerCacheEntry ContainerCacheEntry;
        typedef typename ParameterCache<ParameterCacheType>::ContainerLocations  ContainerLocations;

        ProgramParameterCache( RenderGroupGLHandle renderGroup, ProgramPipelineGLHandle pipeline
                             , bool useUniformBufferUnifiedMemory, BufferMode bufferMode, bool batchedUpdates );
        ~ProgramParameterCache();

        virtual void useContainer(  ContainerGLHandle container );

        void generateContainerCacheEntries( std::vector<GeometryInstanceGLHandle> const& sortedGIs );
        void updateConvertedCache();

        /** \brief Set the active containers for each ParameterState to nullptr. **/
        void resetParameterStatePointers();

        size_t getParameterStateCount() const { return m_activeContainers.size(); }
        ContainerCacheEntry const* getContainerCacheEntries() const;

        void renderGlobalParameters();
        void renderParameters( ContainerCacheEntry const* containerCacheEntry );

      protected:
        void allocateCacheData( std::vector<GeometryInstanceGLHandle> const& sortedGIs );
        void generateGlobalParameterCache();

        std::unique_ptr< ParameterCache<ParameterCacheType> > m_parameterCache;
        std::unique_ptr< ParameterCache<ParameterCacheType> > m_parameterCacheGlobal;

        // variables
        std::map< ContainerDescriptorGLHandle, unsigned int> m_containersPerDescriptor;
        std::vector< unsigned int >  m_activeContainers;

        // uniform data cache
        bool                         m_uniformDataDirty; // set to true if at least one container data has been changes

        // stream cache test
        std::vector< ContainerCacheEntry > m_variableContainerCache; // stream of 'per gi variable' containers
        std::vector< ContainerCacheEntry > m_containerCacheGlobal; // stream of global containers

        // Pair of ParameterInfos and corresponding container for global Parameters
        struct GlobalContainerInfo
        {
          ContainerGLHandle                container;
        };

        typedef std::vector<GlobalContainerInfo> GlobalParameters;

        GlobalParameters m_globalParameters;
        ProgramPipelineGLSharedHandle m_programPipeline;

        // TODO make private again
        //private:
        class ContainerObserver : public dp::util::Observer
        {
        public:
          ContainerObserver( ProgramParameterCache& parameterCache );

          // observer
          virtual void onNotify( dp::util::Event const& event, dp::util::Payload *payload );
          virtual void onDestroyed( dp::util::Subject const& subject, dp::util::Payload* payload );
        private:
          ProgramParameterCache& m_parameterCache;
        };

        friend class ContainerObserver;

        void onContainerNotify( dp::util::Event const& event, dp::util::Payload* payload );
        void onContainerDestroyed( dp::util::Subject const& subject, dp::util::Payload* payload );

        std::unique_ptr<ContainerObserver>   m_containerObserver;
        dp::util::BitArray                   m_containerDirty;                // specifies if a container is dirty
        dp::util::BitArray                   m_containerKnown;                // specifies if a container is already known 
        std::vector<ContainerGLSharedHandle> m_containers;                    // All known containers, shared as they're being observed
        bool                                 m_useUniformBufferUnifiedMemory; // Use unified_buffer_unified_memory extension for UBO bindings
        bool                                 m_batchedUpdates;                // Use shader to batch updates to buffers
        BufferMode                           m_bufferMode;                    // Method to use when switching between UBO or SSBO parameters
      };

      template <typename ParameterCacheType>
      ProgramParameterCache<ParameterCacheType>::ProgramParameterCache( RenderGroupGLHandle renderGroup, ProgramPipelineGLHandle programPipeline
                                                                      , bool useUniformBufferUnifiedMemory, BufferMode bufferMode, bool batchedUpdates)
        : m_uniformDataDirty( true )
        , m_programPipeline( programPipeline )
        , m_useUniformBufferUnifiedMemory(useUniformBufferUnifiedMemory)
        , m_batchedUpdates(batchedUpdates)
        , m_bufferMode(bufferMode)
      {
        m_containerObserver.reset( new ContainerObserver( *this ) );

        // generate ParameterCache for GIs
        std::vector<ContainerDescriptorGLHandle> descriptors;
        for ( ProgramPipelineGL::ContainerDescriptorData::iterator it = programPipeline->m_containerDescriptorData.begin();
              it != programPipeline->m_containerDescriptorData.end(); ++ it )
        {
          descriptors.push_back( (*it).m_descriptor );
        }
        m_parameterCache.reset( new ParameterCache<ParameterCacheType>( programPipeline, descriptors, useUniformBufferUnifiedMemory, bufferMode, batchedUpdates ) );

        const RenderGroupGL::ContainerMap& globalContainers = renderGroup->getGlobalContainers();
        for( RenderGroupGL::ContainerMap::const_iterator it = globalContainers.begin(); it != globalContainers.end(); ++it )
        {
          useContainer( it->second.get() );
        }
      }

      template <typename ParameterCacheType>
      ProgramParameterCache<ParameterCacheType>::~ProgramParameterCache()
      {
        m_containerKnown.traverseBits( [&](size_t index) { m_containers[index]->detach(m_containerObserver.get(), nullptr); } );
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::allocateCacheData( std::vector<GeometryInstanceGLHandle> const& m_sortedGIs )
      {
        // TODO The algorithm can be more clever by detaching only the containers which are no longer referenced.
        // detach all observed containers
        m_containerKnown.traverseBits( [&](size_t index) { m_containers[index]->detach(m_containerObserver.get(), nullptr); } );
        m_containerKnown.clear();
        m_containerDirty.clear();

        m_parameterCache->allocationBegin();
        if ( !m_sortedGIs.empty())
        {
          // second compute offsets for variable containers
          for ( std::vector< GeometryInstanceGLHandle >::const_iterator itGI = m_sortedGIs.begin(); itGI != m_sortedGIs.end(); ++itGI )
          {
            for ( size_t containerIndex = 0; containerIndex < m_activeContainers.size(); ++containerIndex)
            {
              ContainerGLHandle container = (*itGI)->m_containers[containerIndex].container.get();

              ID uniqueId = container->getUniqueID();

              // grow vectors if required...

              if (m_containerKnown.getSize() <= uniqueId)
              {
                size_t newSize = (uniqueId + 65536) &~65535;
                m_containerKnown.resize(newSize);
                m_containerDirty.resize(newSize);
                m_containers.resize(newSize);
              }

              if (!m_containerKnown.getBit(uniqueId))
              {
                // allocate
                m_parameterCache->allocateContainer(container, containerIndex);

                // mark as known, keep reference and observe
                m_containerKnown.enableBit(uniqueId);
                m_containers[uniqueId] = container;
                container->attach( m_containerObserver.get(), nullptr );

                // and make dirty
                m_containerDirty.enableBit(uniqueId);
              }
            }
          }

          m_uniformDataDirty = true;
        }

        m_parameterCache->allocationEnd();
      }

      template<typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::generateContainerCacheEntries( std::vector<GeometryInstanceGLHandle> const& m_sortedGIs )
      {
        allocateCacheData( m_sortedGIs );
        updateConvertedCache();

        m_variableContainerCache.clear();

        if ( !m_sortedGIs.empty() && m_activeContainers.size() )
        {
          m_variableContainerCache.resize( m_sortedGIs.size() * m_activeContainers.size() );
          ContainerCacheEntry* containerCacheEntry = &m_variableContainerCache[0];

          // setup container cache entries for containers
          for ( size_t j = 0; j < m_sortedGIs.size(); ++j )
          {
            if ( !m_activeContainers.empty() )
            {
              for ( std::vector<unsigned int>::iterator it = m_activeContainers.begin(); it != m_activeContainers.end(); ++it )
              {
                m_parameterCache->updateContainerCacheEntry( m_sortedGIs[j]->m_containers[*it].container.get(), containerCacheEntry++ );
              }
            }
          }
        }
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::updateConvertedCache()
      {
        if ( m_uniformDataDirty )
        {
          m_containerDirty.traverseBits([&](size_t index) {m_parameterCache->updateContainer(m_containers[index].get());});
          m_uniformDataDirty = false;
          m_containerDirty.clear();
        }
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::onContainerNotify( const dp::util::Event &event, dp::util::Payload* /*payload*/ )
      {
        const ContainerGL::Event& containerEvent = static_cast<const ContainerGL::Event&>(event);

        // mark dirty to run updateConvertedCache next frame
        m_uniformDataDirty = true;

        // mark container location dirty
        m_containerDirty.enableBit(containerEvent.getHandle()->getUniqueID());
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::onContainerDestroyed( const dp::util::Subject& subject, dp::util::Payload* /*payload*/ )
      {
        const ContainerGL& containerHandle = static_cast<const ContainerGL&>(subject);
        m_parameterCache->removeContainer( const_cast<ContainerGLHandle>( &containerHandle ) );
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::resetParameterStatePointers( )
      {
        m_parameterCache->resetParameterStateContainers();
      }

      template <typename ParameterCacheType>
      typename ProgramParameterCache<ParameterCacheType>::ContainerCacheEntry const* ProgramParameterCache<ParameterCacheType>::getContainerCacheEntries() const
      {
        return m_variableContainerCache.empty() ? nullptr : &m_variableContainerCache[0];
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::useContainer( ContainerGLHandle container )
      {
        // TODO check whether this descriptor contains any parameters relevant for this program
        if ( true )
        {
          typename GlobalParameters::iterator it;
          for ( it = m_globalParameters.begin(); it != m_globalParameters.end(); ++it )
          {
            // found container with same descriptor, update
            if ( it->container->m_descriptor == container->m_descriptor )
            {
              it->container = container;
              break;
            }
          }

          // new container, add
          if ( it == m_globalParameters.end() )
          {
            GlobalContainerInfo info;
            info.container = container;
            m_globalParameters.push_back( info );
            m_parameterCacheGlobal.reset();
          }
        }
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::generateGlobalParameterCache( )
      {
        // generate ParameterCache for GIs
        std::vector<ContainerDescriptorGLHandle> descriptors;
        for ( typename GlobalParameters::iterator it = m_globalParameters.begin(); it != m_globalParameters.end(); ++it )
        {
          descriptors.push_back( it->container->m_descriptor.get() );
        }
        m_parameterCacheGlobal.reset( new ParameterCache<ParameterCacheType>( m_programPipeline.get(), descriptors, m_useUniformBufferUnifiedMemory, m_bufferMode, m_batchedUpdates ) );

        m_parameterCacheGlobal->allocationBegin();
        size_t index = 0;
        for ( typename GlobalParameters::iterator it = m_globalParameters.begin(); it != m_globalParameters.end(); ++it )
        {
          m_parameterCacheGlobal->allocateContainer( it->container, index++);
        }
        m_parameterCacheGlobal->allocationEnd();
        m_containerCacheGlobal.resize( m_globalParameters.size() );
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::renderGlobalParameters()
      {
        if ( !m_parameterCacheGlobal )
        {
          generateGlobalParameterCache();
        }
        m_parameterCacheGlobal->activate();
        if ( m_globalParameters.size() )
        {
          size_t index = 0;
          for ( typename GlobalParameters::iterator it = m_globalParameters.begin(); it != m_globalParameters.end(); ++index, ++it )
          {
            m_parameterCacheGlobal->updateContainer( it->container );
            m_parameterCacheGlobal->updateContainerCacheEntry( it->container, &m_containerCacheGlobal[index] );
          }
          m_parameterCacheGlobal->resetParameterStateContainers();
          m_parameterCacheGlobal->renderParameters( &m_containerCacheGlobal[0] );
        }
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::renderParameters( ContainerCacheEntry const* containerCacheEntry )
      {
        m_parameterCache->renderParameters( containerCacheEntry );
      }

      /************************************************************************/
      /* ContainerObserver                                                    */
      /************************************************************************/

      template <typename ParameterCacheType>
      ProgramParameterCache<ParameterCacheType>::ContainerObserver::ContainerObserver( ProgramParameterCache& parameterCache )
        : m_parameterCache( parameterCache )
      {

      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::ContainerObserver::onNotify( const dp::util::Event &event, dp::util::Payload* payload )
      {
        m_parameterCache.onContainerNotify( event, payload );
      }

      template <typename ParameterCacheType>
      void ProgramParameterCache<ParameterCacheType>::ContainerObserver::onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload )
      {
        m_parameterCache.onContainerDestroyed( subject, payload );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
