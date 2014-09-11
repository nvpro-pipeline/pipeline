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


#include <dp/rix/fx/inc/BufferManagerIndexed.h>
#include <boost/scoped_array.hpp>
#include <vector>

namespace dp
{
  namespace rix
  {
    namespace fx
    {

      BufferManagerIndexed::BufferManagerIndexed( dp::rix::core::Renderer* renderer
                                          , dp::rix::core::ContainerDescriptorSharedHandle const & descriptor, dp::rix::core::ContainerEntry entry // descriptor with buffer attached
                                          , dp::rix::core::ContainerDescriptorSharedHandle const & descriptorId, dp::rix::core::ContainerEntry entryId // descriptor with index attached
                                          , size_t blockSize, size_t chunkSize )
        : BufferManagerImpl( renderer, blockSize, 4 * sizeof(float), chunkSize )
        , m_descriptorBuffer( descriptor )
        , m_entryBuffer( entry )
        , m_descriptorId( descriptorId )
        , m_entryId( entryId )
      {
        m_containerIndices.resize( chunkSize );
      }

      BufferManagerIndexed::~BufferManagerIndexed()
      {

      }

      dp::rix::fx::SmartChunk BufferManagerIndexed::allocateChunk()
      {
        return new Chunk( this, getAlignedBlockSize(), getChunkSize() );
      }

      dp::rix::core::ContainerSharedHandle BufferManagerIndexed::allocationGetBufferContainer( AllocationHandle allocation )
      {
        AllocationImplHandle allocationImpl = dp::rix::core::handleCast<AllocationImpl>(allocation);
        Chunk* chunk = dp::rix::core::handleCast<Chunk>(allocationImpl->m_chunk.get());
        return chunk->m_container;
      }

      dp::rix::core::ContainerSharedHandle BufferManagerIndexed::allocationGetIndexContainer( AllocationHandle allocation )
      {
        AllocationImplHandle allocationImpl = dp::rix::core::handleCast<AllocationImpl>(allocation);
        // TODO ref to make code more readable
        if ( !m_containerIndices[allocationImpl->m_blockIndex] )
        {
          m_containerIndices[allocationImpl->m_blockIndex] = getRenderer()->containerCreate( m_descriptorId );
          int index = static_cast<int>(allocationImpl->m_blockIndex);
          getRenderer()->containerSetData( m_containerIndices[allocationImpl->m_blockIndex], m_entryId, dp::rix::core::ContainerDataRaw( 0, &index, sizeof(index) ) );
        }
        return m_containerIndices[allocationImpl->m_blockIndex];
      }

      dp::rix::core::ContainerDescriptorSharedHandle BufferManagerIndexed::getBufferDescriptor()
      {
        return m_descriptorBuffer;
      }

      dp::rix::core::ContainerDescriptorSharedHandle BufferManagerIndexed::getIndexDescriptor()
      {
        return m_descriptorId;
      }

      dp::rix::core::ContainerEntry BufferManagerIndexed::getBufferDescriptorEntry()
      {
        return m_entryBuffer;
      }

     void BufferManagerIndexed::useContainers( dp::rix::core::GeometryInstanceSharedHandle const& gi, AllocationHandle allocation )
     {
        getRenderer()->geometryInstanceUseContainer( gi, allocationGetBufferContainer( allocation ) );
        getRenderer()->geometryInstanceUseContainer( gi, allocationGetIndexContainer( allocation ) );
     }

     void BufferManagerIndexed::useContainers( dp::rix::core::RenderGroupSharedHandle const & renderGroup, AllocationHandle allocation )
     {
       getRenderer()->renderGroupUseContainer( renderGroup, allocationGetBufferContainer( allocation ) );
       getRenderer()->renderGroupUseContainer( renderGroup, allocationGetIndexContainer( allocation ) );
     }

     /************************************************************************/
      /* Chunk implementation                                                 */
      /************************************************************************/
      BufferManagerIndexed::Chunk::Chunk( BufferManagerIndexed* manager, size_t blockSize, size_t numberOfBlocks )
        : dp::rix::fx::Chunk( manager, blockSize, numberOfBlocks )
      {
        dp::rix::core::Renderer* renderer = manager->getRenderer();

        m_container = renderer->containerCreate( manager->getBufferDescriptor() );
        renderer->containerSetData( m_container, manager->getBufferDescriptorEntry(), dp::rix::core::ContainerDataBuffer( m_buffer) );
      }

      typedef dp::rix::core::SmartHandle<Chunk> SmartChunk;


      SmartBufferManager BufferManagerIndexed::create( dp::rix::core::Renderer* renderer
                                              , dp::rix::core::ContainerDescriptorSharedHandle const & descriptor, dp::rix::core::ContainerEntry entry // descriptor with buffer attached
                                              , dp::rix::core::ContainerDescriptorSharedHandle const & descriptorId, dp::rix::core::ContainerEntry entryId // descriptor with index attached
                                              , size_t blockSize, size_t chunkSize )
      {
        return new BufferManagerIndexed( renderer, descriptor, entry, descriptorId, entryId, blockSize, chunkSize );
      }

    } // namespace fx
  } // namespace rix
} // namespace dp
