// Copyright NVIDIA Corporation 2013-2015
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

#include <dp/rix/gl/inc/ParameterRendererBuffer.h>
#include <dp/rix/gl/inc/BufferGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      ParameterRendererBuffer::ParameterRendererBuffer(ParameterCacheEntryStreamBuffers const& parameterCacheEntries, dp::gl::BufferSharedPtr const& ubo, GLenum target
                                                      , size_t uboBinding, size_t uboOffset, GLsizeiptr uboBlockSize
                                                      , bool useUniformBufferUnifiedMemory)
        : ParameterRendererStreamBuffer(parameterCacheEntries)
        , m_ubo(ubo)
        , m_target(target)
        , m_uboBinding(GLint(uboBinding))
        , m_uboOffset(GLsizeiptr(uboOffset))
        , m_uboBlockSize(uboBlockSize)
        , m_useUniformBufferUnifiedMemory(useUniformBufferUnifiedMemory)
      {
      }

      void ParameterRendererBuffer::activate()
      {
        if (m_useUniformBufferUnifiedMemory) {
            // TODO hack, query alignment from driver!
            glBufferAddressRangeNV(GL_UNIFORM_BUFFER_ADDRESS_NV, m_uboBinding, m_ubo->getAddress(), (m_uboBlockSize + 0xff) & ~0xff);
        }
        else
        {
            glBindBufferRange( m_target, m_uboBinding, m_ubo->getGLId(), m_uboOffset, m_uboBlockSize );
        }
      }

      void ParameterRendererBuffer::render( void const* cache )
      { 
        m_ubo->update(cache, m_uboOffset, m_uboBlockSize);
      }

      void ParameterRendererBuffer::update( void* cache, void const* container )
      {
        updateParameters(cache, container);
      }

      size_t ParameterRendererBuffer::getCacheSize( ) const
      {
        return m_uboBlockSize;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp

