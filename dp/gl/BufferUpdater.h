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

#include <dp/gl/Config.h>
#include <dp/gl/Buffer.h>
#include <dp/gl/Program.h>
#include <vector>

namespace dp
{
  namespace gl
  {

/** \class BufferUpdater
    \brief Do batched updates in a buffer.

    Instead of calling glBufferSubData per update directly
    this modules queues a series of updates, put them into
    a big buffer (per chunkSize) and uses a shader to scatter
    the update-data into a bigger buffer.

    In the current implementation all offsets/sizes passed must
    be a multiple of 16.

    *  \code
    *  BufferUpdater bu(ubo);
    *  bu.update(768, 64, &matrix1);
    *  bu.update(512, 64, &matrix2);
    *  bu.update(256, 128, &material1);
    *  bu.update(4096, 128, &material2);
    *  bu.executeUpdates();
    *  \endcode

    There's no redundancy filtering. if bu.update(offset, size, data)
    is called multiple times with the same offset the last update wins.
**/

    class BufferUpdater {
    public:
      /** \brief Create BufferUpdater class
          \param Buffer is the buffer which will be updated upon the executeUpdates call.
          \param bachedUpdates If true, updates will be gathered into a big buffer and scattered on the GPU with a shader.
          \remarks Updates that are not aligned to 4 bytes or have a size which is not a multiple of 4 will not be batched.
      **/
      DP_GL_API BufferUpdater(dp::gl::BufferSharedPtr const& buffer, bool batchedUpdates);
      DP_GL_API ~BufferUpdater();

      /** \brief Add an update to the update queue.
          \param offset Offset inside the buffer where to put the data
          \param size   Number of bytes to update. 
          \param data   A pointer to the data for the update

          \remarks If offset or size are not a multiple of 16 an exception will be thrown.
      **/
      DP_GL_API void update(size_t offset, size_t size, void const* data);

      /** \brief Process all updates passed to the driver
      **/
      DP_GL_API void executeUpdates();

    private:
      /** \brief There's one UpdateInfo for each updateSize **/
      struct UpdateInfo {
        std::vector<char>   data;
        std::vector<Uint32> offsets;
        Uint32              offsetMask; // value with all offsets or'ed.
      };

      bool                              m_batchedUpdates; // Use shader to do batched updates on GPU
      std::map<Uint32, UpdateInfo>      m_updateInfos;          // One updateInfo per update size
      dp::gl::BufferSharedPtr           m_buffer;               // Buffer to update
      dp::gl::BufferSharedPtr           m_bufferData;           // Buffer with data to update
      dp::gl::BufferSharedPtr           m_bufferChunkOffsets;   // Buffer with offsets of the chunks to update
      dp::gl::ProgramInstanceSharedPtr  m_programUpdate4;       // Program used to scatter 4-byte aligned data on the GPU.
      dp::gl::ProgramInstanceSharedPtr  m_programUpdate8;       // Program used to scatter 8-byte aligned data on the GPU.
      dp::gl::ProgramInstanceSharedPtr  m_programUpdate16;      // Program used to scatter 16-byte aligned data on the GPU.
    };

  } // namespace gl
} // namespace dp

