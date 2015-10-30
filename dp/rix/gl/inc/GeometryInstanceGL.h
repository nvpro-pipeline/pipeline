// Copyright (c) 2011-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/rix/gl/inc/RenderGroupGL.h>

#include <dp/util/Observer.h>

#include <map>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      class ParameterObject;
      struct ContainerCacheEntry;

      class GeometryInstanceGL : public dp::rix::core::GeometryInstance, public dp::util::Subject, public dp::util::Observer
      {
      public:
        struct Payload : public dp::util::Payload
        {
          Payload( )
            : m_position(~0)
            , m_gi( nullptr )
          {
          }

          uint32_t m_position;
          GeometryInstanceGLHandle m_gi;
        };

        struct ContainerStorage
        {
          ContainerStorage()
            :container(nullptr)
          {
          }

          ContainerGLSharedHandle container;
        };

        enum EventType
        {
            CHANGED_VISIBILITY
          , CHANGED_DATA
          , CHANGED_CONTAINER
        };

        class Event : public dp::util::Event
        {
        public:
          Event( GeometryInstanceGLHandle geometryInstance, EventType eventType )
            : m_geometryInstance( geometryInstance )
            , m_container( nullptr )
            , m_eventType( eventType )
          {
          }

          Event( GeometryInstanceGLHandle geometryInstance, ContainerGLHandle container, EventType eventType )
            : m_geometryInstance( geometryInstance )
            , m_container( container )
            , m_eventType( eventType )
          {
          }

          GeometryInstanceGLHandle getGeometryInstance() const { return m_geometryInstance; }
          ContainerGLHandle        getContainer() const { return m_container; }
          EventType                getEventType() const { return m_eventType; }

        private:
          GeometryInstanceGLHandle m_geometryInstance;
          ContainerGLHandle        m_container;
          EventType                m_eventType;
        };

        struct Cache
        {
          ContainerCacheEntry *m_containerCacheEntry;
        };

      public:
        GeometryInstanceGL();
        ~GeometryInstanceGL();

        virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload );
        virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload );

        void setGeometry( GeometryGLSharedHandle const & geometry );
        GeometryGLSharedHandle const & getGeometry() const { return m_geometry; }

        void setProgramPipeline( ProgramPipelineGLSharedHandle const & programPipeline );

        bool useContainer( ContainerGLSharedHandle const & container );

        void setVisible( bool visible );
        bool isVisible() const { return m_isVisible; }

      public:
        RenderGroupGLHandle m_renderGroup;

        dp::rix::core::SmartHandledObject m_vertexCache; // data storage for vertex cache
        dp::rix::core::SmartHandledObject m_parameterCache; // data storage for parameter cache

        Payload m_payload;  // Payload for Observer and attachment to RenderGroupGL

        ProgramPipelineGLSharedHandle      m_programPipeline;
        dp::rix::core::HandledObjectHandle m_pipelineGroupCache; // TODO put this into cache object?
        size_t                             m_pipelineGroupCacheIndex;

        std::vector<ContainerStorage> m_containers; // TODO ContainerStorage no longer required. Can use ContainerGLHandle directly.

        unsigned int      m_numParams;

      protected:
        bool                    m_isVisible;
        GeometryGLSharedHandle  m_geometry;
      };

    } // namespace gl
  } // namespace rix
} // namespace dp
