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

      class BufferManagerOffset;
      typedef dp::rix::core::SmartHandle<BufferManagerOffset> SmartBufferManagerOffset;

      class BufferManagerOffset : public BufferManagerImpl
      {
      public:
        virtual ~BufferManagerOffset();

        /** \brief create a new BufferManager for offset access. The buffer 
            \param blockSize number of bytes per block in the buffer
            \param blockAlignment align the begnning of each object to blockAlignment.
            \param chunkSize number of blocks in one chunk
        **/
        static SmartBufferManager create( dp::rix::core::Renderer* renderer
                                        , dp::rix::core::ContainerDescriptorSharedHandle const & descriptor, dp::rix::core::ContainerEntry entry // descriptor with buffer attached
                                        , size_t blockSize, size_t blockAlignment, size_t chunkSize );


        virtual dp::rix::core::ContainerSharedHandle allocationGetBufferContainer( AllocationHandle allocation );
        virtual dp::rix::core::ContainerDescriptorSharedHandle getBufferDescriptor();
        virtual dp::rix::core::ContainerEntry getBufferDescriptorEntry();

        virtual void useContainers( dp::rix::core::GeometryInstanceSharedHandle const& gi, AllocationHandle allocation );
        virtual void useContainers( dp::rix::core::RenderGroupSharedHandle const & renderGroup, AllocationHandle allocation );

      protected:
        BufferManagerOffset( dp::rix::core::Renderer* renderer
                           , dp::rix::core::ContainerDescriptorSharedHandle const & descriptor, dp::rix::core::ContainerEntry entry // descriptor with buffer attached
                           , size_t blockSize, size_t blockAlignment, size_t chunkSize );

        dp::rix::core::ContainerDescriptorSharedHandle m_descriptor;
        dp::rix::core::ContainerEntry                  m_entry;


        class Chunk : public dp::rix::fx::Chunk
        {
        public:
          Chunk( BufferManagerOffset *manager, size_t blockSize, size_t numberOfBlocks );

          boost::scoped_array<char>           m_shadowBuffer;

          bool                                m_dirty;
        };
      };

    } // namespace fx
  } // namespace rix
} // namespace dp
