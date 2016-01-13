// Copyright (c) 2010-2016, NVIDIA CORPORATION. All rights reserved.
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
/** @file */

#include <dp/sg/core/Buffer.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /** \brief Buffer implementation using memory on the host as storage.
       *  \sa Buffer
      **/
      class BufferHost : public Buffer
      {
      public:
        DP_SG_CORE_API static BufferHostSharedPtr create();

        DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

        DP_SG_CORE_API virtual ~BufferHost();

      public:
        DP_SG_CORE_API virtual void setUnmanagedDataPtr( void *data );

        DP_SG_CORE_API virtual void setSize(size_t size);
        DP_SG_CORE_API virtual size_t getSize() const;

      protected:
        DP_SG_CORE_API BufferHost( );

        using Buffer::map;
        DP_SG_CORE_API virtual void *map( MapMode mode, size_t offset, size_t length );
        DP_SG_CORE_API virtual void unmap( );

        using Buffer::mapRead;
        DP_SG_CORE_API virtual const void *mapRead(size_t offset, size_t length ) const;
        DP_SG_CORE_API virtual void unmapRead() const;


        size_t                      m_sizeInBytes;
        char*                       m_data;
        mutable Buffer::MapModeMask m_mapMode;
        bool                        m_managed;
      };

    } // namespace core
  } // namespace sg
} // namespace dp

