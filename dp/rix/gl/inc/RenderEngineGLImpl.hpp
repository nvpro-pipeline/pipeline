// Copyright NVIDIA Corporation 2011-2015
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

#include <dp/rix/gl/inc/RenderEngineGL.h>
#include <dp/rix/gl/inc/VertexCacheGL.h>
#include <dp/rix/gl/inc/ProgramPipelineGroupCache.h>
#include <GL/glew.h>

#if defined(WIN32)
#include <xmmintrin.h>
#endif

// TODO Usually this is bad style. In this case this is not a header, but more a implementation.
#define GL_CONSTANT_FRAME_RATE_HINT_NV 0x8E8B 

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      template <typename VertexCache> class RenderEngineGLImpl;

      template<typename VertexCache>
      RenderEngineGL* renderEngineCreate(std::map<std::string, std::string> const & options)
      {
        return new RenderEngineGLImpl<VertexCache>(options);
      }

      template <typename VertexCache>
      class RenderEngineGLImpl : public RenderEngineGL, public VertexAttributeCache<VertexCache>
      {
      public:
        RenderEngineGLImpl(std::map<std::string, std::string> const & options);

        virtual void beginRender();
        virtual void render( RenderGroupGLSharedHandle const & groupHandle, dp::rix::core::RenderOptions const & renderOptions );
        virtual void render( RenderGroupGLSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const * gis, size_t numGIs, dp::rix::core::RenderOptions const & renderOptions );
        virtual void endRender();
        virtual RenderGroupGL::SmartCache createCache( RenderGroupGLSharedHandle const & renderGroupGL, ProgramPipelineGLSharedHandle const & programPipeline );

      protected: 
        typedef typename ProgramPipelineGroupCache<VertexCache>::AttributeCacheEntry AttributeCacheEntry;
        typedef typename ProgramPipelineGroupCache<VertexCache>::GeometryInstanceCache GeometryInstanceCache;
        typedef typename ProgramPipelineGroupCache<VertexCache>::GeometryInstanceCacheEntry GeometryInstanceCacheEntry;
        typedef typename ProgramPipelineGroupCache<VertexCache>::ContainerCacheEntry ContainerCacheEntry;
        typedef ProgramPipelineGroupCache<VertexCache> RenderGroupCache;

      private:

        void preRender( RenderGroupGLSharedHandle const & groupHandle );
        void postRender( RenderGroupGLSharedHandle const & groupHandle );
        void render( GeometryInstanceGLHandle gi );

        void generateCache( RenderGroupCache* programPipelineGroupCache );
        void generateGeometryCache( RenderGroupCache* programPipelineCache );

        void changeProgramPipelineCache( RenderGroupCache* programPipelineGroupCache );

        bool                    m_useUniformBufferUnifiedMemory;
        bool                    m_batchedUpdates;
        BufferMode              m_bufferMode;
        RenderGroupCache*       m_currentProgramPipelineCache;

        // the following is needed for gcc name lookup 
        using VertexAttributeCache< VertexCache >::beginFrame;
        using VertexAttributeCache< VertexCache >::endFrame;
        using VertexAttributeCache< VertexCache >::renderGeometryInstance;
        using VertexAttributeCache< VertexCache >::setVertexFormatMask;
        using VertexAttributeCache< VertexCache >::updateGeometryInstanceCacheEntry;

        using VertexAttributeCache< VertexCache >::m_currentVertexFormatId;
        using VertexAttributeCache< VertexCache >::m_currentPrimitiveRestartIndex;
        using VertexAttributeCache< VertexCache >::m_currentArrayBuffer;
        using VertexAttributeCache< VertexCache >::m_currentElementBuffer;
        using VertexAttributeCache< VertexCache >::m_currentVA;
        using VertexAttributeCache< VertexCache >::m_currentIS;
        using VertexAttributeCache< VertexCache >::m_numInstances;
      };

      template <typename VertexCache>
      RenderEngineGLImpl<VertexCache>::RenderEngineGLImpl(std::map<std::string, std::string> const & options) 
        : m_currentProgramPipelineCache( nullptr )
      {
        auto itUniformBufferUnifiedMemory = options.find("uniformBufferUnifiedMemory");
        m_useUniformBufferUnifiedMemory = itUniformBufferUnifiedMemory != options.end() && itUniformBufferUnifiedMemory->second == "true";

        auto itBatchedUpdates = options.find("batchedUpdates");
        m_batchedUpdates = itBatchedUpdates != options.end() && itBatchedUpdates->second == "true";

        auto itBufferMode = options.find("bufferMode");
        if (itBufferMode != options.end())
        {
          if (itBufferMode->second == "bufferSubData")
          {
            m_bufferMode = BM_BUFFER_SUBDATA;
          }
          else if (itBufferMode->second == "bindBufferRange")
          {
            m_bufferMode = BM_BIND_BUFFER_RANGE;
          }
          else if (itBufferMode->second == "persistentBufferMapping")
          {
            m_bufferMode = BM_PERSISTENT_BUFFER_MAPPING;
          }
          else
          {
            throw std::runtime_error("Invalid bufferMode. Valid values are bufferSubData, bindBufferRange, and persistentBufferMapping");
          }
        }
        else
        {
          m_bufferMode = BM_BIND_BUFFER_RANGE;
        }

        if (m_bufferMode == BM_BUFFER_SUBDATA && m_useUniformBufferUnifiedMemory == true)
        {
          throw std::runtime_error("bufferMode=bufferSubData is currently not compatible with uniformUnifiedMemory=true");
        }
      }

      template<typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::generateCache( RenderGroupCache* renderGroupCache )
      {
        renderGroupCache->generateParameterCache( );
        generateGeometryCache( renderGroupCache );
      }

      template<typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::generateGeometryCache( RenderGroupCache* programPipelineCache )
      {
        // initialize gi streaming cache
        ProgramPipelineGLHandle programPipeline = programPipelineCache->getProgramPipeline();
        programPipelineCache->m_geometryInstanceCache.reset( new GeometryInstanceCache() );
        GeometryInstanceCache* giCache = static_cast<GeometryInstanceCache*>(programPipelineCache->m_geometryInstanceCache.get());
        giCache->m_numGeometryInstanceCacheEntries = programPipelineCache->m_sortedGIs.size();
        giCache->m_geometryInstanceCacheEntries = new GeometryInstanceCacheEntry[giCache->m_numGeometryInstanceCacheEntries];

        giCache->m_numAttributeCacheEntries = programPipelineCache->m_sortedGIs.size() * programPipeline->m_numberOfActiveAttributes;
        giCache->m_attributeCacheEntries = new AttributeCacheEntry[giCache->m_numAttributeCacheEntries];
        
        
        for ( size_t j = 0; j < programPipelineCache->m_sortedGIs.size(); ++j )
        {
          GeometryInstanceGLHandle& gi = programPipelineCache->m_sortedGIs[j];
          GeometryInstanceCacheEntry &geometryInstanceCacheEntry = giCache->m_geometryInstanceCacheEntries[j];
          geometryInstanceCacheEntry.m_isVisible = gi->isVisible(); // TODO move to next update-call?
          updateGeometryInstanceCacheEntry( gi, geometryInstanceCacheEntry, giCache->m_attributeCacheEntries + j * programPipeline->m_numberOfActiveAttributes );
        }
      }

      template<typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::beginRender( )
      {
        //glHint( GL_CONSTANT_FRAME_RATE_HINT_NV, GL_NICEST );
        //glGetError();

        m_currentProgramPipelineCache = nullptr;

        glPrimitiveRestartIndex( ~0 );
        m_currentPrimitiveRestartIndex = ~0;
        glEnable( GL_PRIMITIVE_RESTART );

        dp::gl::bind( GL_ARRAY_BUFFER, dp::gl::BufferSharedPtr::null );

        dp::gl::bind( GL_ELEMENT_ARRAY_BUFFER, dp::gl::BufferSharedPtr::null );
        m_currentArrayBuffer = ~0;
        m_currentElementBuffer = ~0;

        m_currentVertexFormatId = ~0;
        m_currentVA = nullptr; // DAR Needed? That is done in most of the beginFrame() calls.
        m_currentIS = nullptr;

        beginFrame();
      }

      template<typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::preRender( RenderGroupGLSharedHandle const & groupHandle )
      {
        m_currentProgramPipelineCache = nullptr;
        m_currentArrayBuffer = ~0;
        m_currentElementBuffer = ~0;

        m_currentVertexFormatId = ~0;
        m_currentVA = nullptr; // DAR Needed? That is done in most of the beginFrame() calls.
        m_currentIS = nullptr;

        // reset descriptor cache objects
        std::map< std::pair<ProgramPipelineGLHandle,ContainerDescriptorGLHandle>, RenderGroupGL::DescriptorCache * >::iterator dco;
        for ( dco = groupHandle->m_descriptorCacheObjects.begin(); dco != groupHandle->m_descriptorCacheObjects.end(); ++dco )
        {
          dco->second->m_lastContainer = nullptr;
        }

        // if dirty, update sorted list
        RenderGroupGL::ProgramPipelineCaches::iterator giIt;
        RenderGroupGL::ProgramPipelineCaches::iterator giIt_end = groupHandle->m_programPipelineCaches.end();
        for ( giIt = groupHandle->m_programPipelineCaches.begin(); giIt != giIt_end; ++giIt )
        {
          RenderGroupCache* renderGroupCache = giIt->second.get<RenderGroupCache>();
          if ( renderGroupCache->isDirty() )
          {
            renderGroupCache->sort();
            generateCache( renderGroupCache );
            renderGroupCache->resetDirty();
            groupHandle->m_dirtyList.clear();
          }
          else if ( !groupHandle->m_dirtyList.empty() )
          {
            renderGroupCache->updateConvertedCache();
            generateCache( renderGroupCache );
            groupHandle->m_dirtyList.clear();
          }
          else 
          {
            renderGroupCache->updateConvertedCache();
          }
        }

        if (m_useUniformBufferUnifiedMemory) {
          glEnableClientState(GL_UNIFORM_BUFFER_UNIFIED_NV);
        }
      }

      template <typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::endRender()
      {
        if (m_useUniformBufferUnifiedMemory) {
          glDisableClientState(GL_UNIFORM_BUFFER_UNIFIED_NV);
        }

        endFrame();
        setVertexFormatMask( 0 );

        dp::gl::bind( GL_ARRAY_BUFFER, dp::gl::BufferSharedPtr::null );
        m_currentArrayBuffer = 0;

        dp::gl::bind( GL_ELEMENT_ARRAY_BUFFER, dp::gl::BufferSharedPtr::null );
        m_currentElementBuffer = 0;

        GLenum err = glGetError();
        if( err != GL_NO_ERROR )
        {
          size_t i(0);
          ++i;
        }

      }

      template <typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::postRender( RenderGroupGLSharedHandle const & /*groupHandle*/ )
      {
      }

      template <typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::render( GeometryInstanceGLHandle gi )
      {
        if ( gi->isVisible() )
        {
          assert( gi->m_pipelineGroupCache );
          RenderGroupCache* pipelineCache = static_cast<RenderGroupCache*>(gi->m_pipelineGroupCache);
          changeProgramPipelineCache( pipelineCache );

          //m_currentProgramPipelineCache->renderParameters( m_currentProgramPipelineCache->getContainerCacheEntries() + gi->m_pipelineGroupCacheIndex * m_numVariableParameterStates);
          m_currentProgramPipelineCache->renderParameters( gi->m_pipelineGroupCacheIndex );

          GeometryInstanceCache* giCache = static_cast<GeometryInstanceCache*>(pipelineCache->m_geometryInstanceCache.get());
          GeometryInstanceCacheEntry* giCacheEntry = giCache->m_geometryInstanceCacheEntries;
          renderGeometryInstance(*(giCacheEntry + gi->m_pipelineGroupCacheIndex));
        }
      }

#if 1 // #if 1 -> optimized use streaming cache loop, if 0 -> loop over gis and render
      template <typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::render( RenderGroupGLSharedHandle const & groupHandle, dp::rix::core::RenderOptions const & renderOptions )
      {
        m_numInstances = (unsigned int)(renderOptions.m_numInstances);
        m_currentProgramPipelineCache = nullptr;

        preRender( groupHandle );

        RenderGroupGL::ProgramPipelineCaches::iterator it;
        RenderGroupGL::ProgramPipelineCaches::iterator it_end = groupHandle->m_programPipelineCaches.end();
        for ( it = groupHandle->m_programPipelineCaches.begin(); it != it_end; ++it )
        {
          RenderGroupCache* giList = it->second.get<RenderGroupCache>();
          const size_t numGIs = giList->m_sortedGIs.size();
          if ( numGIs )
          {
            changeProgramPipelineCache( giList );

            GeometryInstanceCache* giCache = static_cast<GeometryInstanceCache*>(giList->m_geometryInstanceCache.get());
            GeometryInstanceCacheEntry* giCacheEntry = giCache->m_geometryInstanceCacheEntries;

            //ContainerCacheEntry const* containerCache = m_currentProgramPipelineCache->getContainerCacheEntries();

            for ( size_t idx = 0; idx < numGIs; ++idx )
            {
#if 0 // prefetching currently not used. Might bring another performance boost when done on attributes.
              if ( idx + 4 < size )
              {
                //data_prefetch( (const char*) &caches[idx + 4], _MM_HINT_T2 );
                //data_prefetch( (const char*) caches[idx + 2], _MM_HINT_T2 );
              }
#endif
              if ( giCacheEntry->m_isVisible )
              {
                //m_currentProgramPipelineCache->renderParameters( containerCache);
                m_currentProgramPipelineCache->renderParameters( idx );
                renderGeometryInstance( *giCacheEntry );
              }
              ++giCacheEntry;
              //containerCache += m_numVariableParameterStates;
            }
          }
        }

        postRender( groupHandle );
      }
#else
      template <typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::render( RenderGroupGLHandle groupHandle, size_t numInstances )
      {
        m_numInstances = (unsigned int)(numInstances);

        preRender( groupHandle );
        RenderGroupGL::ProgramPipelineCaches::iterator it;
        RenderGroupGL::ProgramPipelineCaches::iterator it_end = groupHandle->m_programPipelineCaches.end();
        for ( it = groupHandle->m_programPipelineCaches.begin(); it != it_end; ++it )
        {
          RenderGroupCache* giList = it->second.get<RenderGroupCache>();
          const size_t numGIs = giList->m_sortedGIs.size();
          for ( size_t idx = 0;idx < numGIs; ++idx )
          {
            render( giList->m_sortedGIs[idx] );
          }
        }
        postRender( groupHandle );
      }

#endif

      template <typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::changeProgramPipelineCache( RenderGroupCache* programPipelineGroupCache )
      {
        if ( m_currentProgramPipelineCache != programPipelineGroupCache )
        {
          m_currentProgramPipelineCache = programPipelineGroupCache;
          m_currentProgramPipelineCache->activate();
        }
      }


      template <typename VertexCache>
      void RenderEngineGLImpl<VertexCache>::render( RenderGroupGLSharedHandle const & groupHandle, dp::rix::core::GeometryInstanceSharedHandle const * gis, size_t numGIs, dp::rix::core::RenderOptions const & renderOptions )
      {
        m_numInstances = (unsigned int)(renderOptions.m_numInstances);
        preRender( groupHandle );

        for ( size_t idx = 0; idx < numGIs; ++idx )
        {
          assert( dp::rix::core::handleIsTypeOf<GeometryInstanceGL>(gis[idx]) );
          assert( dp::rix::core::handleCast<GeometryInstanceGL>(gis[idx])->m_renderGroup == groupHandle.get() );

          // use a dumb ptr here to avoid the refcounting overhead
          GeometryInstanceGLHandle geometryInstance = dp::rix::core::handleCast<GeometryInstanceGL>(gis[idx].get());
          render( geometryInstance );
        }

        postRender( groupHandle );
      }


      template <typename VertexCache>
      RenderGroupGL::SmartCache RenderEngineGLImpl<VertexCache>::createCache( RenderGroupGLSharedHandle const & renderGroupGL, ProgramPipelineGLSharedHandle const & programPipeline )
      {
        return new RenderGroupCache( renderGroupGL.get(), programPipeline.get(), m_useUniformBufferUnifiedMemory, m_bufferMode, m_batchedUpdates );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
