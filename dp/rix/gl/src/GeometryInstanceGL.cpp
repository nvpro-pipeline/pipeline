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


#include <dp/rix/gl/inc/ContainerGL.h>
#include <dp/rix/gl/inc/RenderGroupGL.h>
#include <dp/rix/gl/inc/GeometryInstanceGL.h>
#include <dp/rix/gl/inc/GeometryGL.h>
#include <dp/rix/gl/inc/ProgramGL.h>
#include <dp/rix/gl/inc/ProgramPipelineGL.h>
#include <dp/rix/gl/inc/VertexAttributesGL.h>
#include <dp/rix/gl/inc/VertexFormatGL.h>
#include <dp/rix/gl/inc/GeometryDescriptionGL.h>
#include <dp/rix/gl/inc/IndicesGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      GeometryInstanceGL::GeometryInstanceGL()
        : m_renderGroup(nullptr)
        , m_programPipeline(nullptr)
        , m_pipelineGroupCache(nullptr)
        , m_pipelineGroupCacheIndex( ~0 )
        , m_numParams( 0 )
        , m_isVisible( true )
        , m_geometry(nullptr)
      {
      }

      GeometryInstanceGL::~GeometryInstanceGL()
      {
        if ( m_geometry )
        {
          m_geometry->detach( this );
        }

        m_containers.clear();
      }

      void GeometryInstanceGL::setGeometry( GeometryGLSharedHandle const & geometry )
      {
        if ( m_geometry != geometry )
        {
          if ( m_geometry )
          {
            m_geometry->detach( this );
          }

          m_geometry = geometry;

          if ( m_geometry )
          {
            m_geometry->attach( this );
          }

          notify( Event(this, CHANGED_DATA) );
        }
      }

      void GeometryInstanceGL::setProgramPipeline( ProgramPipelineGLSharedHandle const & programPipeline )
      {
        if ( m_programPipeline != programPipeline )
        {
          // Keep old render group so that the GeometryInstance can be reattached after changing the pipeline
          RenderGroupGLSharedHandle renderGroup = m_renderGroup;
          if ( renderGroup )
          {
            renderGroup->removeGeometryInstance( this );
          }

          std::vector<ContainerStorage> oldContainers = m_containers;
          m_containers.clear();
          m_containers.resize( programPipeline->m_containerDescriptorData.size() );

          ProgramPipelineGLSharedHandle oldProgramPipeline = m_programPipeline;
          m_programPipeline = programPipeline;

          if ( oldProgramPipeline && !oldContainers.empty() )
          {
            size_t oldNumDescriptors = oldProgramPipeline->m_containerDescriptorData.size();

            for ( size_t i = 0; i < oldNumDescriptors; ++i )
            {
              // try to use the container with the new program
              if ( (oldContainers[i].container) )
              {
                useContainer( oldContainers[i].container );
              }
            }
          }

          if ( renderGroup )
          {
            renderGroup->addGeometryInstance( this );
            renderGroup->markDirty( this );
          }
        }
      }

      bool GeometryInstanceGL::useContainer( ContainerGLSharedHandle const & container )
      {
        ProgramPipelineGL::ContainerDescriptorPositions::const_iterator it = m_programPipeline->m_containerDescriptorPositions.find( container->m_descriptor.get() );
        if ( it != m_programPipeline->m_containerDescriptorPositions.end() )
        {
          m_containers[it->second].container = container;
          notify( Event( this, container.get(), CHANGED_CONTAINER ) );
          return true;
        }
        return false;
      }


      void GeometryInstanceGL::setVisible( bool isVisible )
      {
        if ( isVisible != m_isVisible )
        {
          m_isVisible = isVisible;
          notify( Event(this, CHANGED_VISIBILITY) );
        }

      }

      void GeometryInstanceGL::onNotify( dp::util::Event const & /*event*/, dp::util::Payload* /*payload*/ )
      {
        // comes from Geometry-Object
        notify( Event( this, CHANGED_DATA ) );
      }

      void GeometryInstanceGL::onDestroyed( dp::util::Subject const & /*subject*/, dp::util::Payload* /*payload*/ )
      {
        DP_ASSERT( !"need to detach from something?" );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
