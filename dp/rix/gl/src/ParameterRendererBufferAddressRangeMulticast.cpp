// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/rix/gl/inc/ParameterRendererBufferAddressRangeMulticast.h>
#include <dp/rix/gl/inc/BufferGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      ParameterRendererBufferAddressRangeMulticast::ParameterRendererBufferAddressRangeMulticast(ParameterCacheEntryStreamBuffers const& parameterCacheEntries, dp::gl::BufferSharedPtr const& buffer
                                                                                                , GLenum target, size_t bindingIndex, GLsizeiptr bindingLength
                                                                                                , bool /*batchedUpdates*/, size_t containerSize, uint32_t numberOfGPUs)
        : ParameterRendererStreamBuffer(parameterCacheEntries)
        , m_target(target)
        , m_buffer(buffer)
        , m_bindingIndex(GLint(bindingIndex))
        , m_baseAddress(0)
        , m_bindingLength(bindingLength)
        , m_containerSize(containerSize)
        , m_numberOfGPUs(numberOfGPUs)
        , m_cacheData(new uint8_t[m_bindingLength])
      {
        glLGPUNamedBufferSubDataNVX = (PFNGLLGPUNAMEDBUFFERSUBDATANVXPROC)glGetProcAddress("glLGPUNamedBufferSubDataNVX");
        assert(glLGPUNamedBufferSubDataNVX && "multicast extension not supported");
      }

      void ParameterRendererBufferAddressRangeMulticast::activate()
      {
        // update base address of buffer
        m_baseAddress = m_buffer->getAddress();
      }

      void ParameterRendererBufferAddressRangeMulticast::render( void const* cache )
      {
        GLsizeiptr const offset = reinterpret_cast<GLintptr>(cache);
        glBufferAddressRangeNV(m_target, m_bindingIndex,  m_baseAddress + offset, m_bindingLength);
      }

      void ParameterRendererBufferAddressRangeMulticast::update( void* cache, void const* container )
      {
        GLuint ubo = m_buffer->getGLId();
        for (uint32_t gpuId = 0; gpuId < m_numberOfGPUs; ++gpuId)
        {
          updateParameters(m_cacheData.get(), reinterpret_cast<char const*>(container)+gpuId * m_containerSize);
          glLGPUNamedBufferSubDataNVX(1 << gpuId, ubo, reinterpret_cast<size_t>(cache), m_bindingLength, m_cacheData.get());
        }
      }

      size_t ParameterRendererBufferAddressRangeMulticast::getCacheSize( ) const
      {
        // TODO determine alignment requirement of binding!
        return (m_bindingLength + 255) & ~255;
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
