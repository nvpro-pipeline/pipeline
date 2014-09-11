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


#include <dp/rix/fx/inc/BufferManagerImpl.h>
#include <boost/scoped_array.hpp>
#include <vector>

namespace dp
{
  namespace rix
  {
    namespace fx
    {

      AllocationImpl::AllocationImpl( SmartChunk chunk, size_t blockIndex )
        : m_chunk( chunk )
        , m_blockIndex( blockIndex )
        , m_dirty( false )
      {

      }


      BufferManagerImpl::BufferManagerImpl( dp::rix::core::Renderer* renderer , size_t blockSize, size_t alignment, size_t chunkSize )
        : m_blockSize( blockSize )
        , m_alignment( alignment )
        , m_chunkSize( chunkSize )
        , m_renderer( renderer )
      {
        if ( blockSize % alignment )
        {
          m_alignedBlockSize = blockSize + alignment - (blockSize %  alignment );
        }
        else
        {
          m_alignedBlockSize = blockSize;
        }
      }

      BufferManagerImpl::~BufferManagerImpl()
      {

      }

      AllocationHandle BufferManagerImpl::allocate()
      {
        if ( !m_currentAllocationChunk )
        {
          m_currentAllocationChunk = allocateChunk();
        }

        AllocationHandle allocation = m_currentAllocationChunk->allocate();
        if ( !allocation )
        {
          m_currentAllocationChunk = allocateChunk();
          allocation = m_currentAllocationChunk->allocate();
        }
        DP_ASSERT( allocation );
        return allocation;
      }

      dp::rix::core::BufferHandle BufferManagerImpl::allocationGetBuffer( AllocationHandle allocation )
      {
        AllocationImplHandle allocationImpl = dp::rix::core::handleCast<AllocationImpl>(allocation);
        return allocationImpl->m_chunk->m_buffer.get();
      }

      size_t BufferManagerImpl::allocationGetOffset( AllocationHandle allocation )
      {
        AllocationImplHandle allocationImpl = dp::rix::core::handleCast<AllocationImpl>(allocation);
        return allocationImpl->m_blockIndex * m_alignedBlockSize;
      }

      char* BufferManagerImpl::allocationGetPointer( AllocationHandle allocation )
      {
        AllocationImplHandle allocationImpl = dp::rix::core::handleCast<AllocationImpl>(allocation);
        return allocationImpl->m_chunk->m_shadowBuffer.get() + allocationImpl->m_blockIndex * m_alignedBlockSize;
      }

      void BufferManagerImpl::allocationMarkDirty( AllocationHandle allocation )
      {
        AllocationImplHandle allocationImpl = dp::rix::core::handleCast<AllocationImpl>(allocation);
        if ( !allocationImpl->m_chunk->m_dirty )
        {
          allocationImpl->m_chunk->m_dirty = true;
          m_dirtyChunks.push_back( allocationImpl->m_chunk );
        }
      }

      SmartChunk BufferManagerImpl::allocateChunk()
      {
        return new Chunk( this, m_alignedBlockSize, m_chunkSize );
      }

      dp::rix::core::Renderer* BufferManagerImpl::getRenderer() const
      {
        return m_renderer;
      }

      void BufferManagerImpl::update()
      {
        for ( std::vector<SmartChunk>::iterator it = m_dirtyChunks.begin(); it != m_dirtyChunks.end(); ++it )
        {
          m_renderer->bufferUpdateData( (*it)->m_buffer, 0, (*it)->m_shadowBuffer.get(), m_alignedBlockSize * m_chunkSize );
          (*it)->m_dirty = false;
        }
        m_dirtyChunks.clear();
      }


      /************************************************************************/
      /* Chunk implementation                                                 */
      /************************************************************************/
      Chunk::Chunk( BufferManagerImpl* manager, size_t blockSize, size_t numberOfBlocks )
        : m_numberOfBlocks( numberOfBlocks )
        , m_nextBlock(0)
        , m_dirty( false )
      {
        dp::rix::core::Renderer* renderer = manager->getRenderer();

        size_t bufferSize = blockSize * numberOfBlocks;
        m_buffer = renderer->bufferCreate();
        renderer->bufferSetSize( m_buffer, bufferSize );
        m_shadowBuffer.reset( new char[bufferSize] );
      }

      AllocationHandle Chunk::allocate()
      {
        if ( m_nextBlock < m_numberOfBlocks )
        {
          return new AllocationImpl( this, m_nextBlock++ );
        }
        else
        {
          return nullptr;
        }
      }

      typedef dp::rix::core::SmartHandle<Chunk> SmartChunk;



    } // namespace fx
  } // namespace rix
} // namespace dp
