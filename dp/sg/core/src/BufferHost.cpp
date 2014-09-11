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


#include <dp/sg/core/BufferHost.h>
#include <dp/sg/core/Object.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      BufferHostSharedPtr BufferHost::create()
      {
        return( std::shared_ptr<BufferHost>( new BufferHost() ) );
      }

      HandledObjectSharedPtr BufferHost::clone() const
      {
        return( std::shared_ptr<BufferHost>( new BufferHost( *this ) ) );
      }

      BufferHost::BufferHost( )
        : m_sizeInBytes( 0 )
        , m_data( 0 )
        , m_mapMode( MAP_NONE )
        , m_managed( true )
      {
      }

      BufferHost::~BufferHost()
      {
        if ( m_managed )
        {
          if ( m_data )
          {
            DP_ASSERT( m_sizeInBytes );
            dp::util::Singleton<dp::util::Allocator>::instance()->dealloc( m_data, m_sizeInBytes );
          }
        }
      }

      void BufferHost::setUnmanagedDataPtr( void *data )
      {
        if (m_managed)
        {
          if ( m_data )
          {
            DP_ASSERT( m_sizeInBytes );
            dp::util::Singleton<dp::util::Allocator>::instance()->dealloc( m_data, m_sizeInBytes );
          }
          m_managed = false;
        }
        m_data = reinterpret_cast<char*>(data);
      }

      void *BufferHost::map( MapMode mapMode, size_t offset, size_t size )
      {
        DP_ASSERT( m_mapMode == MAP_NONE );
        DP_ASSERT( m_data );
        DP_ASSERT( (offset + size) >= offset && (offset + size) <= m_sizeInBytes );

        m_mapMode = mapMode;
        char* data = reinterpret_cast<char*>( m_data );
        return reinterpret_cast<void*>( data + offset );
      }

      void BufferHost::unmap( )
      {
        DP_ASSERT( m_mapMode != MAP_NONE );

        if ( m_mapMode & MAP_WRITE )
        {
          notify( Event( this ) );
        }
        m_mapMode = MAP_NONE;
      }

      const void *BufferHost::mapRead( size_t offset, size_t size ) const
      {
        DP_ASSERT( m_mapMode == MAP_NONE );
        DP_ASSERT( m_data );
        DP_ASSERT( (offset + size) >= offset && (offset + size) <= m_sizeInBytes );

        m_mapMode = MAP_READ;
        const char* data = reinterpret_cast<const char*>( m_data );
        return reinterpret_cast<const void*>( data + offset );
      }

      void BufferHost::unmapRead( ) const
      {
        DP_ASSERT( m_mapMode == MAP_READ );

        m_mapMode = MAP_NONE;
      }

      size_t BufferHost::getSize() const
      {
        return m_sizeInBytes;
      }

      void BufferHost::setSize( size_t size )
      {
        DP_ASSERT( m_mapMode == MAP_NONE );

        if ( m_sizeInBytes != size)
        {
          if ( m_managed )
          {
            if ( m_data )
            {
              DP_ASSERT( m_sizeInBytes );
              dp::util::Singleton<dp::util::Allocator>::instance()->dealloc( m_data, m_sizeInBytes );
            }
            m_data = (char *) dp::util::Singleton<dp::util::Allocator>::instance()->alloc( size );
          }
          m_sizeInBytes = size;
          // TODO: notify about changes? data is currently crap.
        }
      }

    } // namespace core
  } // namespace sg
} // namespace dp
