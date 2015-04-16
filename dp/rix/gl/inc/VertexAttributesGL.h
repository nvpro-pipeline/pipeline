// Copyright NVIDIA Corporation 2011-2015
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
#include <dp/rix/gl/inc/VertexFormatGL.h>
#include <dp/rix/gl/inc/VertexDataGL.h>

#include <dp/util/Observer.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      class VertexAttributesGL : public dp::rix::core::VertexAttributes, public dp::util::Observer, public dp::util::Subject
      {
      public:
        class Event : public dp::util::Event
        {
        public:
          Event( VertexAttributesGL *vertexAttributes );

          VertexAttributesGL *m_vertexAttributes;
        };

        VertexAttributesGL()
          : m_vertexData( nullptr )
          , m_vertexFormat( nullptr )
        {
        }

        // TODO observer!
        ~VertexAttributesGL();

        virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload );
        virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload );

        void setVertexDataGLHandle( VertexDataGLHandle m_vertexData );
        VertexDataGLHandle getVertexDataGLHandle() const;

        void setVertexFormatGLHandle( VertexFormatGLHandle m_VertexFormat );
        VertexFormatGLHandle getVertexFormatGLHandle() const;

      private:
        void detachObservers();

        VertexDataGLHandle   m_vertexData;
        VertexFormatGLHandle m_vertexFormat;
      };

      inline VertexFormatGLHandle VertexAttributesGL::getVertexFormatGLHandle() const
      {
        return m_vertexFormat;
      }

      inline VertexDataGLHandle VertexAttributesGL::getVertexDataGLHandle() const
      {
        return m_vertexData;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
