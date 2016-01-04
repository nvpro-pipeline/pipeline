// Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/rix/gl/RiXGL.h>

#include <dp/rix/gl/inc/RendererGLConfig.h>
#include <dp/rix/gl/inc/VertexCacheGL.h>
#include <dp/rix/gl/inc/ProgramGL.h>
#include <dp/rix/gl/inc/ProgramPipelineGL.h>
#include <dp/rix/gl/inc/VertexCacheGL.h>
#include <dp/rix/gl/inc/RenderEngineGLDrawCall.h>

#include <dp/util/Observer.h>

#include <vector>
#include <map>

#include <GL/glew.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      class ContainerGL;
      class ParameterObject;

      class RenderGroupGL : public dp::rix::core::RenderGroup, public dp::util::Observer
      {
      public:
        typedef std::map<ContainerDescriptorGLHandle, ContainerGLSharedHandle> ContainerMap;

      public:
        class GeometryInstanceCache
        {
        public:
          virtual ~GeometryInstanceCache() {}
        };

        struct DescriptorCache
        {
          DescriptorCache( )
            : m_lastContainer( nullptr )
          {
          }

          ContainerGLHandle m_lastContainer;
        };

        RenderGroupGL( RenderEngineGL* renderEngine );
        ~RenderGroupGL();

        void addGeometryInstance( GeometryInstanceGLHandle gi );
        void removeGeometryInstance( GeometryInstanceGLHandle gi );
        void setProgramPipeline( ProgramPipelineGLHandle programPipeline );

        /** \brief container per group **/
        void useContainer( ContainerGLHandle container );

        void markDirty( GeometryInstanceGLHandle gi );

        virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload );
        virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload );

        const ContainerMap& getGlobalContainers() const { return m_globalContainers; }

        enum class EventType
        {
            CONTAINER_CHANGED
          , CONTAINER_ADDED
        };

        struct Cache : public dp::rix::core::HandledObject
        {
        public:
          typedef std::vector< GeometryInstanceGLHandle > GeometryInstances;

        public:
          Cache( RenderGroupGLHandle renderGroup, ProgramPipelineGLHandle programPipeline )
            : m_dirty( true )
            , m_renderGroup( renderGroup )
            , m_programPipeline( programPipeline )
          {
          };

          virtual ~Cache()
          {
            // m_geometryInstances are cleared in \dp\rix\gl\inc\ProgramPipelineGroupCache.h
            DP_ASSERT( m_geometryInstances.empty() );
          }

        public:
          virtual void addGeometryInstance( GeometryInstanceGLHandle gi );
          virtual void removeGeometryInstance( GeometryInstanceGLHandle gi );
          virtual void useContainer(  ContainerGLHandle container ) = 0;

          /** \brief container in gis exchanged **/
          void onContainerExchanged( ) { m_dirty = true; }

          ProgramPipelineGLHandle getProgramPipeline() const { return m_programPipeline; }
          const GeometryInstances& getGeometryInstances() const { return m_geometryInstances; }
          bool isDirty() const { return m_dirty; }

          // TODO implement sort/generate cache and remove the following functions afterwards
          GeometryInstances& getGeometryInstances() { return m_geometryInstances; }
          void resetDirty() { m_dirty = false; }

        protected:
          bool                     m_dirty;
          GeometryInstances        m_geometryInstances;
          RenderGroupGLHandle      m_renderGroup;
          ProgramPipelineGLHandle  m_programPipeline;
        };

        typedef dp::rix::core::SmartHandle<Cache> SmartCache;
        typedef std::map< ProgramPipelineGLHandle, SmartCache > ProgramPipelineCaches;

        RenderEngineGL *m_renderEngine; // TODO refactor so that this is not required anymore

        ProgramPipelineCaches m_programPipelineCaches;

        std::map< std::pair<ProgramPipelineGLHandle,ContainerDescriptorGLHandle>, DescriptorCache * > m_descriptorCacheObjects;

        std::set< GeometryInstanceGLHandle > m_dirtyList; // dirtyList for cache

        ContainerMap m_globalContainers;

      };
    } // namespace gl
  } // namespace rix
} // namespace dp
