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

#include <dp/rix/gl/inc/ParameterCache.h>
#include <dp/rix/gl/inc/ParameterCacheEntryStream.h>
#include <dp/rix/gl/inc/ParameterCacheEntryStreamBuffer.h>
#include <dp/rix/gl/inc/ParameterRendererStream.h>
#include <dp/util/BitArray.h>
#include <memory>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      template <>
      class ParameterCache<ParameterCacheStream>
      {
      public:
        typedef ParameterRendererStream::CacheEntry ContainerCacheEntry;

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

        struct ParameterState : public ParameterStateBase
        {
          std::shared_ptr<ParameterRendererStream> m_parameterRenderer;
        };

        typedef std::vector<ParameterState> ParameterStates;

        /** \brief Construct ParameterCache for the given ProgramPipeline
        **/
        ParameterCache( ProgramPipelineGLHandle programPipeline, std::vector<ContainerDescriptorGLHandle> const& descriptors
                      , bool useUniformBufferUnifiedMemory
                      , BufferMode bufferMode
                      , bool batchedUpdates);
        virtual ~ParameterCache();

        /** \brief Clear all allocations in the cache **/
        void clear();

        /** \brief render Parameters for the given set of ContainerCacheEntries **/
        void renderParameters( ContainerCacheEntry const* containers );

        /** \brief Set the active containers for each ParameterState to nullptr. **/
        void resetParameterStateContainers();

        /** update a given container **/
        void updateContainer( ContainerGLHandle container );

        /** \brief begin with allocation process **/
        void allocationBegin();

        /** \brief allocate a Location for the container
            \return true if new allocation took place or false if there was an allocation before
        **/
        void allocateContainer( ContainerGLHandle container, size_t containerIndex );

        /** \brief remove a container from the cache **/
        void removeContainer( ContainerGLHandle container );

        /** \brief end allocation process **/
        void allocationEnd();

        /** \brief activate required resources **/
        void activate();

        /** \brief Update the ContainerCacheEntry for the given ContainerGLHandle. **/
        void updateContainerCacheEntry( ContainerGLHandle container, ContainerCacheEntry* containerCacheEntry);

        typedef std::vector<Location> ContainerLocations;

        ContainerLocations& getContainerLocations() { return m_containerLocations; }
        dp::util::BitArray& getContainerLocationsValid() { return m_containerLocationsValid; }

      private:
        dp::util::BitArray m_containerLocationsValid;
        ContainerLocations m_containerLocations;

        void generateParameterStates( );
        void resizeContainerLocations(size_t newSize);

        // ensure that the m_containerLocations* data structure are big enough to hold newIndex
        void growContainerLocations(size_t newIndex);

        ParameterStates    m_parameterStates;

        std::vector< unsigned char >          m_uniformData;// data bytes for glUniform data
        std::vector<dp::gl::BufferSharedPtr>  m_ubos;       // one UBO for each ubo descriptor if glBufferSubData technique is being used

        std::vector< unsigned char >          m_uboData;    // data bytes for the UBO before uploading
        dp::gl::BufferSharedPtr               m_uboDataUBO; // single UBO for all uboData entries

        std::vector< bool >                   m_isUBOData; // determines for each descriptor if the corresponding data should be copied to the UBO
        std::vector< size_t >                 m_dataSizes; // required size for container data cache for given index

        // allocation state
        size_t m_currentUniformOffset;
        size_t m_currentUBOOffset;

        ProgramPipelineGLHandle      m_programPipeline;
        std::vector<ContainerDescriptorGLHandle> m_descriptors;

        bool       m_useUniformBufferUnifiedMemory; // GL_NV_uniform_buffer_unified_memory is enabled and must be used
        bool       m_batchedUpdates;                // use shader to batch buffer updates
        BufferMode m_bufferMode;                    // Mode to use when switching between parameters when using UBOs and SSBOs
      };


    } // namespace gl
  } // namespace rix
} // namespace dp
