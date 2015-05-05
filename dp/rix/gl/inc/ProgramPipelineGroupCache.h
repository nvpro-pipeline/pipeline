// Copyright NVIDIA Corporation 2012-2015
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
#include <dp/rix/gl/inc/ProgramPipelineGL.h>
#include <dp/rix/gl/inc/RenderGroupGL.h>
#include <dp/rix/gl/inc/GeometryInstanceGL.h>
#include <dp/rix/gl/inc/ProgramParameterCache.h>
#include <dp/rix/gl/inc/ParameterCacheStream.h>
#include <vector>
#include <memory>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      typedef ParameterCacheStream PCT;

      template <typename VertexCache>
      struct ProgramPipelineGroupCache : public RenderGroupGL::Cache, public ProgramParameterCache<PCT>
      {
        typedef typename VertexAttributeCache<VertexCache>::AttributeCacheEntry        AttributeCacheEntry;
        typedef typename VertexAttributeCache<VertexCache>::GeometryInstanceCacheEntry GeometryInstanceCacheEntry;
        using RenderGroupGL::Cache::m_programPipeline; // TODO hack, ambigious between ParameterCache and RenderGroupGL::Cache

        struct GeometryInstanceCache : public RenderGroupGL::GeometryInstanceCache
        {

          GeometryInstanceCache()
            : m_geometryInstanceCacheEntries( nullptr )
            , m_attributeCacheEntries( nullptr )
            , m_numGeometryInstanceCacheEntries( 0 )
            , m_numAttributeCacheEntries( 0 )
          {
          }

          ~GeometryInstanceCache()
          {
            delete[] m_geometryInstanceCacheEntries;
            delete[] m_attributeCacheEntries;
          }

          GeometryInstanceCacheEntry* m_geometryInstanceCacheEntries;
          AttributeCacheEntry*        m_attributeCacheEntries;
          ContainerCacheEntry*        m_containerCacheEntry; // pointer to first container cache entry

          size_t m_numGeometryInstanceCacheEntries;
          size_t m_numAttributeCacheEntries;
        };

        struct Location
        {
          Location()
            : m_offset( ~0 )
            , m_descriptorIndex( ~0 )
            , m_dirty( false )
          {

          }

          Location( size_t offset, size_t descriptorIndex )
            : m_offset( offset )
            , m_descriptorIndex( descriptorIndex )
            , m_dirty( true )
          {

          }

          size_t m_offset;
          size_t m_descriptorIndex;
          bool   m_dirty;
        };

        class GeometryInstanceObserver : public dp::util::Observer
        {
        public:
          GeometryInstanceObserver( ProgramPipelineGroupCache* programPipelineGroupCache )
            : m_programPipelineGroupCache( programPipelineGroupCache )
          {
          }

          virtual void onNotify( dp::util::Event const& event, dp::util::Payload* /*payload*/ )
          {
            const GeometryInstanceGL::Event& eventGeometryInstance = reinterpret_cast<const GeometryInstanceGL::Event&>(event);
            m_programPipelineGroupCache->onVisiblityChanged( eventGeometryInstance.getGeometryInstance() );
          }

          virtual void onDestroyed( dp::util::Subject const& /*subject*/, dp::util::Payload* /*payload*/ )
          {
            m_programPipelineGroupCache = nullptr;
          }

        private:
          ProgramPipelineGroupCache* m_programPipelineGroupCache;
        };

        ProgramPipelineGroupCache( RenderGroupGLHandle renderGroup, ProgramPipelineGLHandle programPipeline
                                 , bool useUniformBufferUnifiedMemory, BufferMode bufferMode, bool batchedUpdates);
        ~ProgramPipelineGroupCache();

        void activate();

        virtual void addGeometryInstance( GeometryInstanceGLHandle gi );
        virtual void removeGeometryInstance( GeometryInstanceGLHandle gi );
        virtual void onVisiblityChanged( GeometryInstanceGLHandle gi );
        virtual void useContainer(  ContainerGLHandle container );

        void generateParameterCache( );

        /** \brief render GeometryInstance for the given cacheIndex which resides
                   in (GeometryInstanceGL::m_pipelineGroupCacheIndex)
        **/
        void renderParameters( size_t cacheIndex );

        void sort();

        // variables
        std::vector< GeometryInstanceGLHandle > m_sortedGIs; // sorted list of geometry instances

        // stream cache
        std::unique_ptr<GeometryInstanceCache> m_geometryInstanceCache;

      private:
        std::unique_ptr<GeometryInstanceObserver> m_geometryInstanceObserver;
      };

      template <typename VertexCache>
      ProgramPipelineGroupCache<VertexCache>::ProgramPipelineGroupCache( RenderGroupGLHandle renderGroup, ProgramPipelineGLHandle programPipeline
                                                                       , bool useUniformBufferUnifiedMemory, BufferMode bufferMode, bool batchedUpdates )
        : RenderGroupGL::Cache( renderGroup, programPipeline )
        , ProgramParameterCache<PCT>( renderGroup, programPipeline, useUniformBufferUnifiedMemory, bufferMode, batchedUpdates )
        , m_geometryInstanceCache( nullptr )
      {
        const RenderGroupGL::ContainerMap& globalContainers = renderGroup->getGlobalContainers();
        for( RenderGroupGL::ContainerMap::const_iterator it = globalContainers.begin(); it != globalContainers.end(); ++it )
        {
          useContainer( it->second.get() );
        }

        m_geometryInstanceObserver.reset( new GeometryInstanceObserver(this) );
      }

      template <typename VertexCache>
      ProgramPipelineGroupCache<VertexCache>::~ProgramPipelineGroupCache()
      {
        // detach from all gis
        for ( std::vector<GeometryInstanceGLHandle>::iterator gis = m_geometryInstances.begin(); gis != m_geometryInstances.end(); ++gis )
        {
          GeometryInstanceGLHandle gi = *gis;
          gi->detach( m_renderGroup, &gi->m_payload ); // TODO refactor, this is wrong! Attach in rendergroup, detach in cache. Cache should observe.
          gi->detach( m_geometryInstanceObserver.get(), nullptr );
          handleUnref( gi );
        }
        m_geometryInstances.clear();

        m_geometryInstanceCache.reset();
      }

      template <typename VertexCache>
      void ProgramPipelineGroupCache<VertexCache>::activate()
      {
        // TODO gather parameter of all programs/indices
#if RIX_GL_SEPARATE_SHADER_OBJECTS_SUPPORT == 1
        glBindProgramPipeline( m_currentProgramPipeline->m_pipelineId );
#else
        // there's only a single program in the pipeline
        glUseProgram( m_programPipeline->m_programs[0]->getProgram()->getGLId() );
#endif
        resetParameterStatePointers();
        renderGlobalParameters();
        m_parameterCache->activate();
      }


      template <typename VertexCache>
      void ProgramPipelineGroupCache<VertexCache>::sort()
      {
        size_t numDescriptors = m_programPipeline->m_containerDescriptorData.size();

        // implementing a radix sort to sort gis by containers, starting with the least significant container, which is, by definition
        // the last container in the array

        std::vector< ContainerGLHandle > containers; // list of all encourted containers
        std::vector< size_t> sortingDescriptors;
        std::vector< unsigned int > containersPerDescriptor( numDescriptors );

        this->m_activeContainers.clear();
        this->m_activeContainers.reserve( numDescriptors );

        GeometryInstances geometryInstances = this->getGeometryInstances();
        size_t numGis = geometryInstances.size();

        /************************************************************************/
        /* Count number of containers per descriptor                            */
        /************************************************************************/
        for ( size_t i = 0; i < numGis; ++i )
        {
          std::vector<GeometryInstanceGL::ContainerStorage> &  containerStorage = geometryInstances[i]->m_containers;

          for ( unsigned int descriptorIndex = 0; descriptorIndex < numDescriptors; ++descriptorIndex )
          {
            ContainerGLHandle container = containerStorage[descriptorIndex].container.get();
            // found a new container, register and add to cleanup-list to reset the counter later
            if ( !container->m_count )
            {
              containersPerDescriptor[ descriptorIndex ]++;
              containers.push_back( container );
            }
            ++container->m_count;
          }
        }

        //init src bucket
        for ( size_t descriptorIndex = 0; descriptorIndex < numDescriptors; ++descriptorIndex )
        {
          this->m_activeContainers.push_back( static_cast<unsigned int>(descriptorIndex) );
          sortingDescriptors.push_back( static_cast<unsigned int>(descriptorIndex) );
        }

        // A containerdata currently contains the bucket
        // TODO move bucket out of containerdata and use only one vector with ranges for sort
        typedef std::vector<ContainerGLHandle> BucketList; 

        if ( sortingDescriptors.empty() )
        {
          m_sortedGIs = m_geometryInstances;
        }
        else // radix sort
        {
          BucketList buckets;
          // reverse the order to begin at the lowest level
          std::reverse( sortingDescriptors.begin(), sortingDescriptors.end() );

          // build up bucket list for lowest level
          for ( GeometryInstances::iterator it = m_geometryInstances.begin(); it != m_geometryInstances.end(); ++it )
          {
            // TODO ContainerStorage could be ContainerData now
            ContainerGLSharedHandle const& containerHandle = (*it)->m_containers[sortingDescriptors[0]].container;

            if ( containerHandle->m_bucket.empty() )
            {
              containerHandle->m_bucket.reserve( containerHandle->m_count );
              buckets.push_back( containerHandle.get() );
            }
            containerHandle->m_bucket.push_back( *it );
          }

          // walk up in the hierarchy generating the new buckets
          for ( size_t index = 1;index < sortingDescriptors.size(); ++index )
          {
            BucketList newBuckets;
            size_t currentDescriptorIndex = sortingDescriptors[index];

            for ( std::vector< ContainerGLHandle >::iterator bucketIterator = buckets.begin(); bucketIterator != buckets.end(); ++bucketIterator )
            {
              for ( GeometryInstances::iterator giIterator = (*bucketIterator)->m_bucket.begin(); 
                    giIterator != (*bucketIterator)->m_bucket.end();
                    ++giIterator
                  )
              {
                ContainerGLSharedHandle const& containerHandle = (*giIterator)->m_containers[currentDescriptorIndex].container;
                if ( containerHandle->m_bucket.empty() )
                {
                  containerHandle->m_bucket.reserve( containerHandle->m_count );
                  newBuckets.push_back( containerHandle.get() );
                }
                containerHandle->m_bucket.push_back( (*giIterator) );
              }
            }
            
            std::swap( buckets, newBuckets );
          }

          // generated sorted list from generated buckets
          m_sortedGIs.clear();
          for ( std::vector< ContainerGLHandle >::iterator bucketIterator = buckets.begin(); bucketIterator != buckets.end(); ++bucketIterator )
          {
            for ( GeometryInstances::iterator giIterator = (*bucketIterator)->m_bucket.begin(); 
                  giIterator != (*bucketIterator)->m_bucket.end();
                  ++giIterator
                )
            {
              m_sortedGIs.push_back( *giIterator );
            }
          }
        }

        // update cache position in gis
        for ( size_t index = 0;index < m_sortedGIs.size(); ++index) 
        {
          m_sortedGIs[index]->m_pipelineGroupCacheIndex = index;
        }

        // cleanup container counter & buckets
        for ( std::vector< ContainerGLHandle>::iterator it = containers.begin(); it != containers.end(); ++it )
        {
          (*it)->m_bucket.clear();
          (*it)->m_count = 0;
        }
      }

      template <typename VertexCache>
      void ProgramPipelineGroupCache<VertexCache>::generateParameterCache( )
      {
        ProgramParameterCache<PCT>::generateContainerCacheEntries( m_sortedGIs );
      }


      template <typename VertexCache>
      void ProgramPipelineGroupCache<VertexCache>::addGeometryInstance( GeometryInstanceGLHandle gi )
      {
        RenderGroupGL::Cache::addGeometryInstance( gi );
        gi->attach( m_geometryInstanceObserver.get(), nullptr );
      }

      template <typename VertexCache>
      void ProgramPipelineGroupCache<VertexCache>::removeGeometryInstance( GeometryInstanceGLHandle gi )
      {
        RenderGroupGL::Cache::removeGeometryInstance( gi );
        gi->detach( m_geometryInstanceObserver.get(), nullptr );
      }

      template <typename VertexCache>
      void ProgramPipelineGroupCache<VertexCache>::useContainer( ContainerGLHandle container )
      {
        ProgramParameterCache<PCT>::useContainer( container );
      }


      template <typename VertexCache>
      void ProgramPipelineGroupCache<VertexCache>::onVisiblityChanged( GeometryInstanceGLHandle geometryInstance )
      {
        if ( m_geometryInstanceCache && geometryInstance->m_pipelineGroupCacheIndex != ~size_t(0) )
        {
          m_geometryInstanceCache->m_geometryInstanceCacheEntries[geometryInstance->m_pipelineGroupCacheIndex].m_isVisible = geometryInstance->isVisible();
        }
      }

      template <typename VertexCache>
      void ProgramPipelineGroupCache<VertexCache>::renderParameters( size_t cacheIndex )
      {
        ProgramParameterCache<PCT>::renderParameters( getContainerCacheEntries() + cacheIndex * m_activeContainers.size() );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
