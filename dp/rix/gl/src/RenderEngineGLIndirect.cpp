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


#include <dp/rix/gl/inc/RenderEngineGLIndirect.h>


#include <iostream>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      RenderEngineGLIndirect::RenderEngineGLIndirect()
      {

      }

      RenderEngineGLIndirect::~RenderEngineGLIndirect()
      {

      }

      void prepareDrawArrayEntry( RenderEngineGLIndirect::RenderGroupGLCache::DrawArrayListEntry *entry )
      {
        entry->m_indirectCommandsBuffer = nullptr;
        handleAssign( entry->m_indirectCommandsBuffer, BufferGL::create( dp::rix::core::BufferDescription() ) );
        entry->m_indirectCommandsBuffer->updateData( 0, &entry->m_indirectCommands[0], entry->m_indirectCommands.size() * sizeof(RenderEngineGLIndirect::RenderGroupGLCache::DrawArraysIndirectCommand) );

        entry->m_indirectEntriesBuffer = nullptr;
        handleAssign( entry->m_indirectEntriesBuffer, BufferGL::create( dp::rix::core::BufferDescription() ) );
        entry->m_indirectEntriesBuffer->updateData( 0, &entry->m_indirectEntries[0], entry->m_indirectEntries.size() * sizeof(RenderEngineGLIndirect::RenderGroupGLCache::IndirectEntry));

        entry->m_indirectPointersBuffer = nullptr;
        handleAssign( entry->m_indirectPointersBuffer, BufferGL::create( dp::rix::core::BufferDescription() ) );
        std::vector<GLuint64EXT> pointers;

        DP_ASSERT( entry->m_indirectEntriesBuffer->getBuffer() );
        GLuint64EXT base = entry->m_indirectEntriesBuffer->getBuffer()->getAddress();
        for ( size_t index = 0;index < entry->m_indirectEntries.size(); ++index )
        {
          pointers.push_back( base + index * sizeof(RenderEngineGLIndirect::RenderGroupGLCache::IndirectEntry) );
        }
        entry->m_indirectPointersBuffer->updateData( 0, &pointers[0], pointers.size() * sizeof(GLuint64EXT) );

      }

      RenderEngineGLIndirect::RenderGroupGLCache::IndirectEntry generateIndirectEntry( GeometryInstanceGLHandle gi )
      {
        RenderEngineGLIndirect::RenderGroupGLCache::IndirectEntry entry;

        VertexAttributesGLSharedHandle const& vertexAttributes = gi->getGeometry()->getVertexAttributes().get();
        VertexDataGLHandle data = vertexAttributes->getVertexDataGLHandle();

        DP_ASSERT( data->m_data[0].m_buffer->getBuffer() );
        entry.position = data->m_data[0].m_buffer->getBuffer()->getAddress();
        if ( data->m_data[2].m_buffer )
        {
          DP_ASSERT( data->m_data[2].m_buffer->getBuffer() );
          entry.normal = data->m_data[2].m_buffer->getBuffer()->getAddress();
        }
        else
        {
          // HACK, some geometry does not have normals
          entry.normal = entry.position;
        }
        entry.transform = *((GLuint64EXT*)gi->m_containers[2].container->m_data);
        entry.material = *((GLuint64EXT*)gi->m_containers[3].container->m_data);

        return entry;
      }


      void addArrayIndirectCommand( GeometryInstanceGLHandle gi, RenderEngineGLIndirect::RenderGroupGLCache::DrawArrayListEntry &drawListEntry  )
      {
        drawListEntry.m_indirectEntries.push_back( generateIndirectEntry(gi) );

        VertexAttributesGLSharedHandle const& vertexAttributes = gi->getGeometry()->getVertexAttributes().get();
        VertexDataGLHandle data = vertexAttributes->getVertexDataGLHandle();

        RenderEngineGLIndirect::RenderGroupGLCache::DrawArraysIndirectCommand command;
        command.count = static_cast<GLuint>(data->m_numberOfVertices);
        command.primCount = 1;
        command.first = 0;
        command.baseInstance = static_cast<GLuint>(drawListEntry.m_indirectCommands.size());


        drawListEntry.m_indirectCommands.push_back( command );
      }

      void addDrawArrayList( RenderEngineGLIndirect::RenderGroupGLCache* cache, RenderEngineGLIndirect::RenderGroupGLCache::DrawArrayListEntry& entry, GLenum primitiveType )
      {
        if ( entry.m_indirectCommands.size() )
        {
          prepareDrawArrayEntry( &entry );
          entry.m_primitiveType = primitiveType;
          cache->m_drawArrayList.push_back(entry);
        }
      }

      void addElementIndirectCommand( RenderEngineGLIndirect::RenderGroupGLCache* /*cache*/, RenderEngineGLIndirect::RenderGroupGLCache::DrawElementsListEntry& entry, GeometryInstanceGLHandle gi )
      {
#if 1
        entry.m_indirectEntries.push_back( generateIndirectEntry( gi ) );
        RenderEngineGLIndirect::RenderGroupGLCache::DrawElementsIndirectCommand command;
        command.count = static_cast<GLuint>(gi->getGeometry()->getIndices()->getCount());
        command.primCount = 1;
        command.firstIndex = 0;
        command.baseInstance = 0;
        command.baseVertex = 0;
        command.baseInstance = static_cast<GLuint>(entry.m_indirectCommands.size());

        entry.m_indirectCommands.push_back(command);

        RenderEngineGLIndirect::RenderGroupGLCache::BindlessPointer pointer;
        DP_ASSERT( gi->getGeometry()->getIndices()->getBufferHandle()->getBuffer() );
        dp::gl::BufferSharedPtr const& buffer = gi->getGeometry()->getIndices()->getBufferHandle()->getBuffer();
        pointer.m_address = buffer->getAddress();
        pointer.m_range = buffer->getSize();
        entry.m_indexPointers.push_back( pointer );
#else
        dp::gl::MappedBuffer<unsigned int> indices( entry.m_indexBuffer, GL_MAP_READ_BIT );

        unsigned int first = 0;
        unsigned int current = 0;
        while ( current < gi->getGeometry()->getIndices()->getCount() )
        {
          if (indices[current] == ~0)
          {
            if ( current - first > 0 )
            {
              RenderEngineGLIndirect::RenderGroupGLCache::DrawElementsIndirectCommand command;
              command.primCount = 1;
              command.firstIndex = first;
              command.count = current - first;
              command.baseInstance = 0;
              command.baseVertex = 0;
              command.baseInstance = static_cast<GLuint>(drawListEntry.m_indirectCommands.size());

              drawListEntry.m_indirectCommands.push_back(command);
              drawListEntry.m_indirectEntries.push_back( generateIndirectEntry( gi ) );
            }

            first = current + 1;
          }
          ++current;
        }

        if ( current - first > 0 )
        {
          RenderEngineGLIndirect::RenderGroupGLCache::DrawElementsIndirectCommand command;
          command.primCount = 1;
          command.firstIndex = first;
          command.count = current - first;
          command.baseInstance = 0;
          command.baseVertex = 0;
          command.baseInstance = static_cast<GLuint>(drawListEntry.m_indirectCommands.size());

          drawListEntry.m_indirectCommands.push_back(command);
          drawListEntry.m_indirectEntries.push_back( generateIndirectEntry( gi ) );
        }
#endif

        //if ( drawListEntry.m_indirectCommands.size() > 1 )
        {
          //std::cout << "#strips " << drawListEntry.m_indirectCommands.size() << std::endl;
        }

      }

      void addDrawElementArrayList( RenderEngineGLIndirect::RenderGroupGLCache* cache, RenderEngineGLIndirect::RenderGroupGLCache::DrawElementsListEntry& entry, GLenum primitiveType )
      {
        entry.m_primitiveType = primitiveType;
        cache->m_drawElementsList.push_back( entry );
      }


      void prepareDrawElementsEntry( RenderEngineGLIndirect::RenderGroupGLCache::DrawElementsListEntry *entry )
      {
        entry->m_indirectCommandsBuffer = nullptr;
        handleAssign( entry->m_indirectCommandsBuffer, BufferGL::create( dp::rix::core::BufferDescription() ) );
        entry->m_indirectCommandsBuffer->updateData( 0, &entry->m_indirectCommands[0], entry->m_indirectCommands.size() * sizeof(RenderEngineGLIndirect::RenderGroupGLCache::DrawElementsIndirectCommand) );

        entry->m_indirectEntriesBuffer = nullptr;
        handleAssign( entry->m_indirectEntriesBuffer, BufferGL::create( dp::rix::core::BufferDescription() ) );
        entry->m_indirectEntriesBuffer->updateData( 0, &entry->m_indirectEntries[0], entry->m_indirectEntries.size() * sizeof(RenderEngineGLIndirect::RenderGroupGLCache::IndirectEntry));

        entry->m_indirectPointersBuffer = nullptr;
        handleAssign( entry->m_indirectPointersBuffer, BufferGL::create( dp::rix::core::BufferDescription() ) );
        std::vector<GLuint64EXT> pointers;

        DP_ASSERT( entry->m_indirectEntriesBuffer->getBuffer() );
        dp::gl::BufferSharedPtr const& buffer = entry->m_indirectEntriesBuffer->getBuffer();
        GLuint64EXT base = buffer->getAddress();
        for ( size_t index = 0;index < entry->m_indirectEntries.size(); ++index )
        {
          pointers.push_back( base + index * sizeof(RenderEngineGLIndirect::RenderGroupGLCache::IndirectEntry) );
        }

        entry->m_indirectPointersBuffer->updateData( 0, &pointers[0], pointers.size() * sizeof(GLuint64EXT) );

        entry->m_indirectPointersBufferAddress = base;
        entry->m_indirectPointersBufferRange = buffer->getSize();

      }

      void RenderEngineGLIndirect::render( RenderGroupGLHandle groupHandle, const dp::rix::core::RenderOptions& /*renderOptions*/ )
      {
        // if dirty, update sorted list
        RenderGroupGL::ProgramPipelineCaches::iterator glIt;
        RenderGroupGL::ProgramPipelineCaches::iterator glIt_end = groupHandle->m_programPipelineCaches.end();
        for ( glIt = groupHandle->m_programPipelineCaches.begin(); glIt != glIt_end; ++glIt )
        {
          ProgramPipelineGLHandle programPipeline = glIt->first;
          RenderGroupGLCache* cache = glIt->second.get<RenderGroupGLCache>();

#if RIX_GL_SEPARATE_SHADER_OBJECTS_SUPPORT == 1
          glBindProgramPipeline( programPipeline->m_pipelineId );
#else
          glUseProgram( programPipeline->m_programs[0]->getProgram()->getGLId() );
#endif

          //if ( giList->m_indirectEntries.size() != giList->m_geometryInstances.size())
          if ( !cache->m_initialized ) // TODO support changes
          {
            cache->m_initialized = true;

            RenderGroupGLCache::DrawArrayListEntry triangles;
            RenderGroupGLCache::DrawArrayListEntry lines;
            RenderGroupGLCache::DrawArrayListEntry linesStrips;

            //cache->m_indirectEntries.clear();
            //cache->m_indirectCommands.clear();
            for (size_t index = 0; index < cache->getGeometryInstances().size(); ++index )
            {
              GeometryInstanceGLHandle gi = cache->getGeometryInstances()[index];

              if ( !gi->getGeometry()->getIndices() )
              {
                switch ( gi->getGeometry()->getGeometryDescription()->getPrimitiveType() )
                {
                case GPT_TRIANGLES:
                  //drawEntry = &triangles;
                  addArrayIndirectCommand(gi, triangles);
                  break;
                case GPT_LINES:
                  addArrayIndirectCommand(gi, lines);
                  break;
                case GPT_LINE_STRIP:
                  addArrayIndirectCommand(gi, linesStrips);
                  break;
                default:
                  assert( 0 && "unsupported primitve types");
                }
              }
              else
              {
                //addElementIndirectCommand(cache, gi );
              }

            }

            addDrawArrayList( cache, triangles, GL_TRIANGLES);
            addDrawArrayList( cache, lines, GL_LINES );
            addDrawArrayList( cache, linesStrips, GL_LINE_STRIP );
            for ( size_t idx = 0;idx < cache->m_drawElementsList.size(); ++idx )
            {
              prepareDrawElementsEntry( &cache->m_drawElementsList[idx] );
            }

          }

          // prepare fixed containers
#if 0
          GeometryInstanceGLHandle gi = cache->getGeometryInstances()[0];
          for ( size_t containerIdx = 0; containerIdx < gi->m_containers.size(); ++containerIdx )
          {
            // debug info
            auto infos = gi->m_containers[containerIdx].container->m_descriptor->m_parameterInfos;
            for ( size_t infoIdx = 0; infoIdx < infos.size(); ++infoIdx )
            {
              std::cout << "c: " << containerIdx << ", infoIdX:" << infoIdx << ", name: " << infos[infoIdx].m_name << std::endl;
            }
          }
#endif
          assert( !"No implementation available atm");
          // TODO update to new interface
          //gi->m_containers[0].parameters[0]->update( gi->m_containers[0].container->m_data ); // sys_View
          //gi->m_containers[1].parameters[0]->update( gi->m_containers[1].container->m_data ); // lights


          for ( size_t idx = 0;idx < cache->m_drawArrayList.size(); ++idx )
          {
            RenderGroupGLCache::DrawArrayListEntry &entry = cache->m_drawArrayList[idx];

            glEnableVertexAttribArray( 12 );
            dp::gl::bind( GL_ARRAY_BUFFER, entry.m_indirectPointersBuffer->getBuffer() );
            glVertexAttribLPointerEXT( 12, 1, GL_UNSIGNED_INT64_NV, 0, 0 );
            glVertexAttribDivisor( 12, 1);
            // TODO update vertex attrib pointer to 64-bit values


            #define USE_DRAW_INDIRECT_BUFFER

#if 0

#if defined(USE_DRAW_INDIRECT_BUFFER) 
            dp::gl::bind( GL_DRAW_INDIRECT_BUFFER, entry.m_indirectCommandsBuffer->getBuffer() );
            size_t offset = 0;
            for (size_t index = 0; index < entry.m_indirectCommands.size(); ++index )
            {
              glDrawArraysIndirect( entry.m_primitiveType, (const void*)(offset) );
              offset += sizeof( RenderGroupGLCache::DrawArraysIndirectCommand );

            }
#else
            RenderGroupGLCache::DrawArraysIndirectCommand *command = &entry.m_indirectCommands[0];
            for (size_t index = 0; index < entry.m_indirectCommands.size(); ++index )
            {
              //glDrawArraysIndirect( entry.m_primitiveType, command );
              glDrawArraysInstancedBaseInstance( entry.m_primitiveType, command->first, command->count, command->primCount, command->baseInstance );
              //glDrawementsBaseVertex( entry.m_primitiveType, command->count, GL_UN)
              ++command;
            }
#endif

#else

#if defined(USE_DRAW_INDIRECT_BUFFER)
            dp::gl::bind( GL_DRAW_INDIRECT_BUFFER, entry.m_indirectCommandsBuffer->getBuffer() );
            glMultiDrawArraysIndirectAMD( entry.m_primitiveType, 0, static_cast<GLsizei>(entry.m_indirectCommands.size()), sizeof( RenderGroupGLCache::DrawArraysIndirectCommand ) );
#else
            RenderGroupGLCache::DrawArraysIndirectCommand *commands = &entry.m_indirectCommands[0];
            glMultiDrawArraysIndirectAMD( entry.m_primitiveType, commands, static_cast<GLuint>(entry.m_indirectCommands.size()), 0 );
#endif

#endif

          }


#if 0
          unsigned int indirectCommands = 0;

          glEnable( GL_PRIMITIVE_RESTART );
          //glDisable( GL_PRIMITIVE_RESTART );
          glPrimitiveRestartIndex( ~0 );

          glEnableClientState(GL_ELEMENT_ARRAY_UNIFIED_NV );
          glEnableClientState(GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV);
          glEnableVertexAttribArray( 12 );
          glVertexAttribDivisor( 12, 1);
          glVertexAttribLFormatNV( 12, 1, GL_UNSIGNED_INT64_NV, 0 );

          for ( size_t idx = 0;idx < cache->m_drawElementsList.size(); ++idx )
          {
            const RenderGroupGLCache::DrawElementsListEntry& entry = cache->m_drawElementsList[idx];

            //dp::gl::bind( GL_ARRAY_BUFFER, entry.m_indirectPointersBuffer );
            //glVertexAttribLPointerEXT( 12, 1, GL_UNSIGNED_INT64_NV, 0, 0 );
            glBufferAddressRangeNV( GL_VERTEX_ATTRIB_ARRAY_ADDRESS_NV, 12, entry.m_indirectPointersBufferAddress, entry.m_indirectPointersBufferRange );

            //dp::gl::bind( GL_ELEMENT_ARRAY_BUFFER, entry.m_indexBuffer );
            glBufferAddressRangeNV( GL_ELEMENT_ARRAY_ADDRESS_NV, 0, entry.m_indexBufferAddress, entry.m_indexBufferRange );
            //glMultiDrawElementsIndirectAMD( entry.m_primitiveType, GL_UNSIGNED_INT, &entry.m_indirectCommands[0], static_cast<GLuint>(entry.m_indirectCommands.size()), 0);
            //glDrawElementsIndirect( entry.m_primitiveType, GL_UNSIGNED_INT, &entry.m_indirectCommands[0] );
            glDrawElementsInstancedBaseVertexBaseInstance( entry.m_primitiveType, entry.m_indirectCommands[0].count, GL_UNSIGNED_INT, 0, 1, entry.m_indirectCommands[0].baseVertex, entry.m_indirectCommands[0].baseInstance );

            indirectCommands += static_cast<unsigned int>(entry.m_indirectCommands.size());
          }
          glDisableClientState(GL_ELEMENT_ARRAY_UNIFIED_NV );
          glDisableClientState(GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV);

#endif

          //std::cout << "indirect commands " << indirectCommands << ", " << cache->m_drawElementsList.size() << std::endl;
        }
      }

      void RenderEngineGLIndirect::render( RenderGroupGLHandle /*groupHandle*/, const dp::rix::core::GeometryInstanceHandle* /*gis*/, size_t /*numGIs*/, const dp::rix::core::RenderOptions& /*renderOptions*/ )
      {
      }

      RenderGroupGL::SmartCache RenderEngineGLIndirect::createCache( RenderGroupGLHandle )
      {
        assert( !"update me to new interface");
        return nullptr;
        //return RenderGroupGL::SmartCache(new RenderGroupGLCache);
      }

      RenderEngineGL* create(std::map<std::string, std::string> const & /*options*/)
      {
        assert( !"update me");
        return nullptr;
        //return new RenderEngineGLIndirect;
      }

      void RenderEngineGLIndirect::beginRender()
      {

      }

      void RenderEngineGLIndirect::endRender()
      {

      }

      static bool initializedVBO = registerRenderEngine( "Indirect", &create );

    } // namespace gl
  } // namespace rix
} // namespace dp
