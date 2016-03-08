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


#include <dp/rix/gl/inc/ParameterRendererBufferMulticast.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      ParameterRendererBufferMulticast::ParameterRendererBufferMulticast(ParameterCacheEntryStreamBuffers const& parameterCacheEntries, dp::gl::BufferSharedPtr const& ubo, GLenum target
                                                                        , size_t uboBinding, size_t uboOffset, GLsizeiptr uboBlockSize
                                                                        , bool useUniformBufferUnifiedMemory, size_t containerSize, uint32_t numberOfGPUs)
        : ParameterRendererBuffer(parameterCacheEntries, ubo, target, uboBinding, uboOffset, uboBlockSize, useUniformBufferUnifiedMemory)
        , m_containerSize(containerSize)
        , m_numberOfGPUs(numberOfGPUs)
      {
        glLGPUNamedBufferSubDataNVX = (PFNGLLGPUNAMEDBUFFERSUBDATANVXPROC)wglGetProcAddress("glLGPUNamedBufferSubDataNVX");
        assert(glLGPUNamedBufferSubDataNVX && "multicast extension not supported");
        glGetIntegerv(GL_MAX_LGPU_GPUS_NVX, (GLint*)(&m_numberOfGPUs));
      }

      void ParameterRendererBufferMulticast::render( void const* cache )
      {
        char const* basePtr = reinterpret_cast<char const*>(cache);
        for (uint32_t gpuId = 0;gpuId < m_numberOfGPUs;++gpuId)
        {
          glLGPUNamedBufferSubDataNVX(1 << gpuId, m_ubo->getGLId(), m_uboOffset, m_uboBlockSize, basePtr );
          basePtr += m_containerSize;
        }
      }

    } // namespace gl
  } // namespace rix
} // namespace dp



