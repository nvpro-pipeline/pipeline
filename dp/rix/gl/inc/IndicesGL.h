// Copyright NVIDIA Corporation 2011
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

#include <dp/rix/gl/RiXGL.h>

#include <dp/util/Observer.h>

#include <GL/glew.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      class IndicesGL : public dp::rix::core::Indices, public dp::util::Subject, public dp::util::Observer
      {
      public:
        class Event : public dp::util::Event
        {
        public:
          Event( IndicesGLHandle indices );
          IndicesGLHandle m_indices;
        };

        IndicesGL();
        ~IndicesGL();

        void setData( dp::DataType dataType, dp::rix::core::BufferHandle bufferHandle, size_t offset, size_t count ) ;
        void upload();
        virtual void onNotify(  const dp::util::Event &event, dp::util::Payload *payload );
        virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload );

        dp::DataType    getDataType() const;
        BufferGLHandle  getBufferHandle() const;
        size_t          getOffset() const;
        size_t          getCount()  const;

        bool    m_markedForUpload;
        GLsizei m_bufferGLSize;

      private:
        dp::DataType    m_dataType;
        BufferGLHandle  m_bufferHandle;
        size_t          m_offset;
        size_t          m_count;

      };

      inline dp::DataType IndicesGL::getDataType() const
      {
        return m_dataType;
      }

      inline BufferGLHandle IndicesGL::getBufferHandle() const
      {
        return m_bufferHandle;
      }

      inline size_t IndicesGL::getOffset() const
      {
        return m_offset;
      }

      inline size_t IndicesGL::getCount() const
      {
        return m_count;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
