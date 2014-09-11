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

#include <dp/rix/fx/inc/BufferManager.h>
#include <dp/rix/core/RiX.h>
#include <boost/scoped_array.hpp>
#include <vector>

namespace dp
{
  namespace rix
  {
    namespace fx
    {

      class Chunk;
      typedef dp::rix::core::SmartHandle<dp::rix::fx::Chunk> SmartChunk;

      class BufferManagerImpl;

      /************************************************************************/
      /* AllocationImpl                                                       */
      /************************************************************************/
      class AllocationImpl : public Allocation
      {
      public:
        AllocationImpl( SmartChunk chunk, size_t blockIndex );

        SmartChunk m_chunk;
        size_t     m_blockIndex;
        bool       m_dirty;
      };

      typedef AllocationImpl* AllocationImplHandle;

      /************************************************************************/
      /* Chunk                                                                */
      /************************************************************************/
      class Chunk : public dp::rix::core::HandledObject
      {
      public:
        Chunk( BufferManagerImpl *manager, size_t blockSize, size_t numberOfBlocks );
        AllocationHandle allocate();

        size_t                              m_numberOfBlocks;
        size_t                              m_nextBlock;
        dp::rix::core::BufferSharedHandle    m_buffer;
        boost::scoped_array<char>           m_shadowBuffer;

        bool                                m_dirty;
      };

      /************************************************************************/
      /* BufferManagerImpl                                                    */
      /************************************************************************/
      class BufferManagerImpl : public BufferManager
      {
      protected:
        typedef dp::rix::fx::Chunk Chunk;
        typedef dp::rix::core::SmartHandle<Chunk> SmartChunk;

      public:
        virtual ~BufferManagerImpl();

        virtual AllocationHandle allocate();

        virtual dp::rix::core::BufferHandle allocationGetBuffer( AllocationHandle allocation );
        virtual size_t                      allocationGetOffset( AllocationHandle allocation );
        virtual char*                       allocationGetPointer( AllocationHandle allocation );
        virtual void allocationMarkDirty( AllocationHandle allocation );

        dp::rix::core::Renderer* getRenderer() const;

        virtual void update();

        size_t getBlockSize() const;
        size_t getChunkSize() const;
        size_t getAlignedBlockSize() const;

        virtual void useContainers( dp::rix::core::GeometryInstanceSharedHandle const& gi, AllocationHandle allocation ) = 0;
        virtual void useContainers( dp::rix::core::RenderGroupSharedHandle const & gi, AllocationHandle allocation ) = 0;

      protected:
        BufferManagerImpl( dp::rix::core::Renderer* renderer, size_t blockSize, size_t alignment, size_t chunkSize );

        virtual SmartChunk allocateChunk();

      private:
        SmartChunk m_currentAllocationChunk;
        size_t m_blockSize;
        size_t m_alignedBlockSize;
        size_t m_alignment;
        size_t m_chunkSize;

        dp::rix::core::Renderer* m_renderer;

        std::vector<SmartChunk> m_dirtyChunks;
      };

      inline size_t BufferManagerImpl::getBlockSize() const
      {
        return m_blockSize;
      }

      inline size_t BufferManagerImpl::getChunkSize() const
      {
        return m_chunkSize;
      }

      inline size_t BufferManagerImpl::getAlignedBlockSize() const
      {
        return m_alignedBlockSize;
      }


    } // namespace fx
  } // namespace rix
} // namespace dp
