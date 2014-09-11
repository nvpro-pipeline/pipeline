// Copyright NVIDIA Corporation 2012
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

#include <dp/rix/fx/inc/BufferManagerImpl.h>

namespace dp
{
  namespace rix
  {
    namespace fx
    {

      class BufferManagerIndexed;
      typedef dp::rix::core::SmartHandle<BufferManagerIndexed> SmartBufferManagerIndexed;

      class BufferManagerIndexed : public dp::rix::fx::BufferManagerImpl
      {
      public:
        virtual ~BufferManagerIndexed();

        /** \brief create a new BufferManager for index buffer access. The buffer will always point to the first element where the descriptorId
             contains the index within the array inside the buffer.
            \param blockSize number of bytes per block in the buffer
            \param chunkSize number of blocks in one chunk
        **/
        static SmartBufferManager create( dp::rix::core::Renderer* renderer
                                        , dp::rix::core::ContainerDescriptorSharedHandle const & descriptorBuffer, dp::rix::core::ContainerEntry entryBuffer // descriptor with buffer attached
                                        , dp::rix::core::ContainerDescriptorSharedHandle const & descriptorId, dp::rix::core::ContainerEntry entryId // descriptor with index attached
                                        , size_t blockSize, size_t chunkSize );

        virtual dp::rix::core::ContainerSharedHandle allocationGetBufferContainer( AllocationHandle allocation );
        virtual dp::rix::core::ContainerSharedHandle allocationGetIndexContainer( AllocationHandle allocation );

        virtual dp::rix::core::ContainerDescriptorSharedHandle getBufferDescriptor( );
        virtual dp::rix::core::ContainerEntry getBufferDescriptorEntry();
        virtual dp::rix::core::ContainerDescriptorSharedHandle getIndexDescriptor( );

        virtual void useContainers( dp::rix::core::GeometryInstanceSharedHandle const & gi, AllocationHandle allocation );
        virtual void useContainers( dp::rix::core::RenderGroupSharedHandle const & renderGroup, AllocationHandle allocation );

      protected:
        BufferManagerIndexed( dp::rix::core::Renderer* renderer
                                        , dp::rix::core::ContainerDescriptorSharedHandle const & descriptor, dp::rix::core::ContainerEntry entry // descriptor with buffer attached
                                        , dp::rix::core::ContainerDescriptorSharedHandle const & descriptorId, dp::rix::core::ContainerEntry entryId // descriptor with index attached
                                        , size_t blockSize, size_t chunkSize );

        virtual SmartChunk allocateChunk();

        class Chunk : public dp::rix::fx::BufferManagerImpl::Chunk
        {
        public:
          // TODO BufferManagerIndexed const& 
          Chunk( BufferManagerIndexed* manager, size_t blockSize, size_t numberOfBlocks );

          dp::rix::core::ContainerSharedHandle m_container;
        };

        dp::rix::core::ContainerDescriptorSharedHandle m_descriptorBuffer;
        dp::rix::core::ContainerEntry m_entryBuffer; // descriptor with buffer attached
        dp::rix::core::ContainerDescriptorSharedHandle m_descriptorId;
        dp::rix::core::ContainerEntry m_entryId; // descriptor with index attached

        std::vector<dp::rix::core::ContainerSharedHandle> m_containerIndices;
      };

    } // namespace fx
  } // namespace rix
} // namespace dp
