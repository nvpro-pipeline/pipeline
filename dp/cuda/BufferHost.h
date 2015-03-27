// Copyright NVIDIA Corporation 2015
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

#include <dp/cuda/Config.h>
#include <dp/cuda/Types.h>

namespace dp
{
  namespace cuda
  {

    class BufferHost
    {
      public:
        DP_CUDA_API static BufferHostSharedPtr create( size_t size, unsigned int flags );
        DP_CUDA_API virtual ~BufferHost();

      public:
        template <typename T> T * getDevicePointer() const;
        template <typename T> T * getPointer() const;
        DP_CUDA_API size_t getSize() const;

      protected:
        DP_CUDA_API BufferHost( size_t size, unsigned int flags );

      private:
        bool      m_isMapped;
        void    * m_pointer;
        size_t    m_size;
#if !defined(NDEBUG)
        int       m_deviceID;
        bool      m_isPortable;
#endif
    };

    template <typename T>
    inline T * BufferHost::getDevicePointer() const
    {
      DP_ASSERT( m_isMapped );
#if !defined(NDEBUG)
      if ( !m_isPortable )
      {
        int deviceID;
        CUDA_VERIFY( cudaGetDevice( &deviceID ) );
        DP_ASSERT( deviceID == m_deviceID );
      }
#endif
      T * devicePointer;
      CUDA_VERIFY( cudaHostGetDevicePointer( &devicePointer, m_pointer, 0 ) );
      return( devicePointer );
    }

    template <typename T>
    inline T * BufferHost::getPointer() const
    {
      return( static_cast<T *>( m_pointer ) );
    }

    inline size_t BufferHost::getSize() const
    {
      return( m_size );
    }

  } // namespace cuda
} // namespace dp
