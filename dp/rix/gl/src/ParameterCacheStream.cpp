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


#include <dp/rix/gl/inc/BufferGL.h>
#include <dp/rix/gl/inc/ParameterCacheStream.h>
#include <dp/rix/gl/inc/ParameterRendererUniform.h>
#include <dp/rix/gl/inc/ParameterRendererBuffer.h>
#include <dp/rix/gl/inc/ParameterRendererBufferAddressRange.h>
#include <dp/rix/gl/inc/ParameterRendererBufferDSA.h>
#include <dp/rix/gl/inc/ParameterRendererBufferRange.h>
#include <dp/rix/gl/inc/ParameterRendererPersistentBufferMapping.h>
#include <dp/rix/gl/inc/ParameterRendererPersistentBufferMappingUnifiedMemory.h>

#include <dp/Assert.h>
#include <dp/util/Array.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      /************************************************************************/
      /* ParameterCache                                                       */
      /************************************************************************/
      ParameterCache<ParameterCacheStream>::ParameterCache( ProgramPipelineGLHandle programPipeline, std::vector<ContainerDescriptorGLHandle> const &descriptors
                                                          , bool useUniformBufferUnifiedMemory, BufferMode bufferMode, bool batchedUpdates)
        : m_containerLocationsValid(0)
        , m_programPipeline( programPipeline )
        , m_descriptors( descriptors )
        , m_useUniformBufferUnifiedMemory(useUniformBufferUnifiedMemory)
        , m_batchedUpdates(batchedUpdates)
        , m_bufferMode(bufferMode)
      {
        switch(m_bufferMode)
        {
        case BM_PERSISTENT_BUFFER_MAPPING:
          m_uboDataUBO = dp::gl::Buffer::create(dp::gl::Buffer::PERSISTENT_BUFFER, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
          break;
        case BM_BIND_BUFFER_RANGE:
        case BM_BUFFER_SUBDATA:
          m_uboDataUBO = dp::gl::Buffer::create(dp::gl::Buffer::CORE, GL_DYNAMIC_DRAW);
          break;
        default:
          assert("unknown buffer mode");
          break;
        }

        generateParameterStates( );
      }

      ParameterCache<ParameterCacheStream>::~ParameterCache( )
      {
      }

      void ParameterCache<ParameterCacheStream>::generateParameterStates(  )
      {
        DP_ASSERT( m_ubos.empty() );

        m_parameterStates.clear();

        size_t numDescriptors = m_descriptors.size();

        // assemble a vector of ParameterState objects that belong to variable containers
        for ( size_t i = 0; i < numDescriptors; ++i )
        {
          DP_ASSERT( m_programPipeline->m_programs.size() == 1 );

          ProgramGLHandle program = m_programPipeline->m_programs[0].get();
          ParameterState parameterState;
          parameterState.m_program             = program;
          parameterState.m_uniqueDescriptorIdx = i;
            
          ProgramGL::UniformInfos uniformInfos = program->getUniformInfos( m_descriptors[i] );
          ProgramGL::UniformInfos bufferVariables = program->getBufferInfos( m_descriptors[i] );

          if ( uniformInfos.size() && uniformInfos.begin()->second.blockIndex != -1 )
          {
            GLint blockIndex = uniformInfos.begin()->second.blockIndex;
            GLint binding;
            GLint blockSize;

            GLint programId = parameterState.m_program->getProgram()->getGLId();
            glGetActiveUniformBlockiv( programId, blockIndex, GL_UNIFORM_BLOCK_BINDING, &binding );
            glGetActiveUniformBlockiv( programId, blockIndex, GL_UNIFORM_BLOCK_DATA_SIZE, &blockSize);

            ParameterCacheEntryStreamBuffers parameterCacheEntries = createParameterCacheEntriesStreamBuffer( program, m_descriptors[i], uniformInfos );
            parameterState.m_numParameterObjects = parameterCacheEntries.size();

            switch (m_bufferMode)
            {
            case BM_BUFFER_SUBDATA:
              {
                dp::gl::BufferSharedPtr ubo = dp::gl::Buffer::create(dp::gl::Buffer::CORE, GL_STREAM_DRAW, GL_UNIFORM_BUFFER);
                if ( glNamedBufferSubDataEXT /** GLEW_EXT_direct_state_access, glew has a bug not reporting this extension because the double versions are missing **/)
                {
                  parameterState.m_parameterRenderer.reset( new ParameterRendererBufferDSA( parameterCacheEntries, ubo, GL_UNIFORM_BUFFER, binding, 0, blockSize, m_useUniformBufferUnifiedMemory ) );
                }
                else
                {
                  parameterState.m_parameterRenderer.reset( new ParameterRendererBuffer( parameterCacheEntries, ubo, GL_UNIFORM_BUFFER, binding, 0, blockSize, m_useUniformBufferUnifiedMemory ) );
                }
                ubo->setSize(blockSize);

                m_ubos.push_back( ubo );
                m_isUBOData.push_back(false);
              }
              break;
            case BM_BIND_BUFFER_RANGE:
              if ( m_useUniformBufferUnifiedMemory )
              {
                parameterState.m_parameterRenderer.reset( new ParameterRendererBufferAddressRange( parameterCacheEntries, m_uboDataUBO, GL_UNIFORM_BUFFER_ADDRESS_NV, binding, blockSize, m_batchedUpdates ) );
              }
              else
              {
                parameterState.m_parameterRenderer.reset( new ParameterRendererBufferRange( parameterCacheEntries, m_uboDataUBO, GL_UNIFORM_BUFFER, binding, blockSize, m_batchedUpdates ) );
              }

              m_isUBOData.push_back(true);
              break;
            case BM_PERSISTENT_BUFFER_MAPPING:
              if ( m_useUniformBufferUnifiedMemory )
              {
                parameterState.m_parameterRenderer.reset(new ParameterRendererPersistentBufferMappingUnifiedMemory(parameterCacheEntries, m_uboDataUBO, GL_UNIFORM_BUFFER_ADDRESS_NV, binding, blockSize));
              }
              else
              {
                parameterState.m_parameterRenderer.reset(new ParameterRendererPersistentBufferMapping(parameterCacheEntries, m_uboDataUBO, GL_UNIFORM_BUFFER, binding, blockSize));
              }

              m_isUBOData.push_back(true);
              break;
            default:
              DP_ASSERT(!"unsupported buffermode");
              m_isUBOData.push_back(false);
              break;
            }
          }
          else if ( bufferVariables.size() )
          {
            // TODO unify with UBO version
            GLint blockIndex = bufferVariables.begin()->second.blockIndex;
            GLint binding;
            GLint blockSize;

            GLenum props[] = { GL_BUFFER_BINDING, GL_BUFFER_DATA_SIZE };
            GLint results[sizeof dp::util::array(props)];
            glGetProgramResourceiv(program->getProgram()->getGLId(), GL_SHADER_STORAGE_BLOCK, blockIndex, sizeof dp::util::array(props), props, sizeof dp::util::array(results), NULL, results);
            binding = results[0];
            blockSize = results[1];

            ParameterCacheEntryStreamBuffers parameterCacheEntries = createParameterCacheEntriesStreamBuffer( program, m_descriptors[i], bufferVariables );
            parameterState.m_numParameterObjects = parameterCacheEntries.size();

            switch (m_bufferMode)
            {
            case BM_BUFFER_SUBDATA:
              {
                dp::gl::BufferSharedPtr ubo = dp::gl::Buffer::create(dp::gl::Buffer::CORE, GL_STREAM_DRAW, GL_UNIFORM_BUFFER);
                if ( glNamedBufferSubDataEXT /** GLEW_EXT_direct_state_access, glew has a bug not reporting this extension because the double versions are missing **/)
                {
                  parameterState.m_parameterRenderer.reset( new ParameterRendererBufferDSA( parameterCacheEntries, ubo, GL_SHADER_STORAGE_BUFFER, binding, 0, blockSize, m_useUniformBufferUnifiedMemory ) );
                }
                else
                {
                  parameterState.m_parameterRenderer.reset( new ParameterRendererBuffer( parameterCacheEntries, ubo, GL_SHADER_STORAGE_BUFFER, binding, 0, blockSize, m_useUniformBufferUnifiedMemory ) );
                }
                ubo->setSize(blockSize);

                m_ubos.push_back( ubo );
                m_isUBOData.push_back(false);
              }
              break;
            case BM_BIND_BUFFER_RANGE:
              parameterState.m_parameterRenderer.reset( new ParameterRendererBufferRange( parameterCacheEntries, m_uboDataUBO, GL_SHADER_STORAGE_BUFFER, binding, blockSize, m_batchedUpdates) );
              m_isUBOData.push_back(true);
              break;
            default:
              DP_ASSERT(!"unsupported buffermode");
              m_isUBOData.push_back(false);
              break;
            }
          }
          else
          {
            ParameterCacheEntryStreams parameterCacheEntries = createParameterCacheEntryStreams( program, m_descriptors[i], m_useUniformBufferUnifiedMemory );
            parameterState.m_parameterRenderer.reset( new ParameterRendererUniform( parameterCacheEntries ) );
            parameterState.m_numParameterObjects = parameterCacheEntries.size();
            m_isUBOData.push_back(false);
          }

          // use ~0 as invalidate data ptr since this member is being used as offset to an ubo or ptr to data im memory
          // and 0 is a valid offset.
          parameterState.m_currentDataPtr = reinterpret_cast<void*>(~0);
          m_parameterStates.push_back( parameterState );
          m_dataSizes.push_back( parameterState.m_parameterRenderer->getCacheSize() );
        }
      }

      void ParameterCache<ParameterCacheStream>::renderParameters( ContainerCacheEntry const* containers )
      {
        ParameterRendererStream::CacheEntry const* cacheEntry = static_cast<ParameterRendererStream::CacheEntry const*>(containers);
        size_t const numParameterGroups = m_parameterStates.size();
        if ( !m_parameterStates.empty() )
        {
          // update shader uniforms
          for ( size_t idx = 0; idx < numParameterGroups;++idx )
          {
            if ( (m_parameterStates[idx].m_currentDataPtr != cacheEntry->m_containerData) )
            {
              m_parameterStates[idx].m_currentDataPtr = cacheEntry->m_containerData;

#if RIX_GL_SEPARATE_SHADER_OBJECTS_SUPPORT == 1
              glActiveShaderProgram( m_currentProgramPipeline->m_pipelineId, parameterState[idx].m_program->m_programId );
#endif
              m_parameterStates[idx].m_parameterRenderer->render( cacheEntry->m_containerData );
            } 
            ++cacheEntry;
          }
        }
      }

      void ParameterCache<ParameterCacheStream>::updateContainer(ContainerGLHandle container)
      {
        Location const& location = m_containerLocations[container->getUniqueID()];
        // cannot use m_uniformData[offset] since dummy might be the last one which has size of 0
        // this would cause an out of bounds exception
        unsigned char *basePtr = (m_uniformData.empty() | m_isUBOData[location.m_descriptorIndex] ) ? nullptr : &m_uniformData[0];
        m_parameterStates[location.m_descriptorIndex].m_parameterRenderer->update(basePtr + location.m_offset, container->m_data);
        //location.m_dirty = false;
      }

      void ParameterCache<ParameterCacheStream>::updateContainerCacheEntry( ContainerGLHandle container, ContainerCacheEntry* containerCacheEntry )
      {
        ParameterRendererStream::CacheEntry *cacheEntry = static_cast<ParameterRendererStream::CacheEntry*>(containerCacheEntry);
        Location & location = m_containerLocations[container->getUniqueID()];
        unsigned char *base = (m_uniformData.empty() | m_isUBOData[location.m_descriptorIndex] ) ? nullptr : &m_uniformData[0];
        size_t offset = location.m_offset;
        *cacheEntry = ParameterRendererStream::CacheEntry( base + offset );
      }

      void ParameterCache<ParameterCacheStream>::allocationBegin()
      {
        // TODO MTA Does it make sense to think about shrinking?
        m_containerLocationsValid.clear();
        m_currentUniformOffset = 0;
        m_currentUBOOffset = 0;
      }

      void ParameterCache<ParameterCacheStream>::allocateContainer(ContainerGLHandle container, size_t descriptorIndex)
      {
        ID containerId = container->getUniqueID();
        growContainerLocations(containerId);

        DP_ASSERT(m_containerLocationsValid.getBit(containerId) == false);

        size_t & offset = m_isUBOData[descriptorIndex] ? m_currentUBOOffset : m_currentUniformOffset;
        m_containerLocations[containerId] = Location(offset, descriptorIndex );
        offset += m_dataSizes[descriptorIndex];
        m_containerLocationsValid.enableBit(containerId);
      }

      void ParameterCache<ParameterCacheStream>::removeContainer(ContainerGLHandle container)
      {
        DP_ASSERT(container->getUniqueID() < m_containerLocationsValid.getSize())
        m_containerLocationsValid.disableBit(container->getUniqueID());
      }

      void ParameterCache<ParameterCacheStream>::allocationEnd()
      {
        m_uniformData.resize( m_currentUniformOffset );
        if (m_uboDataUBO->getSize() != m_currentUBOOffset) {

          if (m_useUniformBufferUnifiedMemory && m_uboDataUBO->isResident())
          {
            m_uboDataUBO->makeNonResident();
          }

          if (m_bufferMode == BM_PERSISTENT_BUFFER_MAPPING && m_uboDataUBO->isMapped())
          {
            m_uboDataUBO->unmap();
          }

          m_uboDataUBO->setSize(m_currentUBOOffset);

          if (m_bufferMode == BM_PERSISTENT_BUFFER_MAPPING)
          {
            m_uboDataUBO->map(GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
          }
        }
      }

      void ParameterCache<ParameterCacheStream>::resetParameterStateContainers()
      {
        for ( ParameterStates::iterator it = m_parameterStates.begin(); it != m_parameterStates.end(); ++it )
        {
          it->m_currentDataPtr = reinterpret_cast<void*>(~0);
        }
      }

      void ParameterCache<ParameterCacheStream>::activate()
      {
        size_t const numParameterGroups = m_parameterStates.size();
        // update shader uniforms
        for ( size_t idx = 0; idx < numParameterGroups;++idx )
        {
          m_parameterStates[idx].m_parameterRenderer->activate();
        }
      }

      void ParameterCache<ParameterCacheStream>::resizeContainerLocations(size_t containerLocations)
      {
        m_containerLocationsValid.resize(containerLocations);
        m_containerLocations.resize(containerLocations);
      }

      void ParameterCache<ParameterCacheStream>::growContainerLocations(size_t newIndex)
      {
        if (newIndex >= m_containerLocationsValid.getSize())
        {
          resizeContainerLocations((newIndex + 65536) & ~65535);
        }
      }


    } // namespace gl
  } // namespace rix
} // namespace dp
