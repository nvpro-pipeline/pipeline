// Copyright NVIDIA Corporation 2010
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


#include <dp/sg/core/Buffer.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      Buffer::Buffer()
        : m_lockCount(0)
        , m_mappedPtr(nullptr)
        , m_managedBySystem(true)
      {
      }

      Buffer::~Buffer( )
      {
      }

      void Buffer::getData( size_t src_offset, size_t size, void* dst_data) const
      {
        DP_ASSERT( dst_data );

        const void* from = mapRead( src_offset, size );
        memcpy( dst_data, from, size );
        unmapRead();
      }

      void Buffer::setData( size_t dst_offset, size_t size, const void* src_data)
      {
        DP_ASSERT( src_data );

        void* to = map( MAP_WRITE, dst_offset, size );
        memcpy( to, src_data, size );
        unmap();
      }


      void Buffer::getData( size_t offset, size_t size, const BufferSharedPtr &dst_buffer , size_t dst_offset ) const
      {
        DP_ASSERT( dst_buffer );

        void* to = dst_buffer->map( MAP_WRITE, dst_offset, size );
        const void* from = mapRead( offset, size );

        memcpy( to, from, size );

        dst_buffer->unmap();
        unmapRead();
      }

      void Buffer::setData( size_t offset, size_t size, const BufferSharedPtr &src_buffer , size_t src_offset )
      {
        DP_ASSERT( src_buffer );

        void* to = map( MAP_WRITE, offset, size );
        const void* from = src_buffer->mapRead( src_offset, size );

        memcpy( to, from, size );

        src_buffer->unmapRead();
        unmap();
      }

    } // namespace core
  } // namespace sg
} // namespace dp
