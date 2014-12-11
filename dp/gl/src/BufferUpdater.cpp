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

#include <dp/gl/BufferUpdater.h>
#include <dp/gl/ProgramInstance.h>

namespace dp
{
  namespace gl
  {

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define INPUT_BINDING 0
#define OUTPUT_BINDING 1
#define CHUNK_OFFSET_BINDING 2
#define LOCAL_SIZE_X 32

    static const char* shader =
      "#version 430\n"
      "uniform int numChunks;\n"
      "uniform int chunkSize;\n"
      "\n"
      "layout( std430, binding = " TOSTRING(INPUT_BINDING) ") buffer Input {\n"
      "  uvec4 inputBuffer[];\n"
      "};\n"
      "layout( std430, binding = " TOSTRING(OUTPUT_BINDING) " ) buffer Output {\n"
      "  uvec4 outputBuffer[];\n"
      "};\n"
      "layout( std430, binding = " TOSTRING(CHUNK_OFFSET_BINDING) " ) buffer Offsets {\n"
      "  int chunk_offsets[];\n"
      "};\n"
      "layout( local_size_x = " TOSTRING(LOCAL_SIZE_X) ", local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "void main() {\n"
      "  uint index = gl_GlobalInvocationID.x;\n"
      "  uint chunk = index / chunkSize;\n"
      "  if (chunk < numChunks) {\n"
      "    uint element = index % chunkSize;\n"
      "    outputBuffer[chunk_offsets[chunk] + element] = inputBuffer[index];\n"
      "  }\n"
      "}\n"
      ;

    BufferUpdater::BufferUpdater(dp::gl::BufferSharedPtr const& buffer)
      : m_buffer(buffer)
    {
      m_bufferData = dp::gl::Buffer::create(dp::gl::Buffer::CORE, GL_STREAM_DRAW, GL_SHADER_STORAGE_BUFFER);
      m_bufferChunkOffsets = dp::gl::Buffer::create(dp::gl::Buffer::CORE, GL_STREAM_DRAW, GL_SHADER_STORAGE_BUFFER);
      m_programUpdate = dp::gl::ProgramInstance::create( dp::gl::Program::create(dp::gl::ComputeShader::create(shader)) );
    }

    BufferUpdater::~BufferUpdater()
    {
    }

    void BufferUpdater::update(size_t offset, size_t size, void const* data)
    {
      if ((offset & 15) != 0)
      {
        throw std::runtime_error("Offset must be a multiple of 16.");
      }
      if ((size & 15) != 0)
      {
        throw std::runtime_error("Size must be a multiple of 16.");
      }
      if (offset + size >= m_buffer->getSize())
      {
        throw std::runtime_error("Offset + size exceeds buffer size.");
      }

      UpdateInfo &info = m_updateInfos[size];

      size_t dataOffset = info.data.size();
      info.data.resize(dataOffset + size);
      memcpy(info.data.data() + dataOffset, data, size);
      info.offsets.push_back(dp::checked_cast<dp::Uint32>(offset));
    }

    void BufferUpdater::executeUpdates()
    {
      // Query the current program id, so that it can be restored afterwards...
      GLint id;
      glGetIntegerv(GL_CURRENT_PROGRAM, &id);

      // iterate over all chunk sizes and update the data
      for (auto it = m_updateInfos.begin(); it != m_updateInfos.end(); ++it)
      {
        UpdateInfo &info = it->second;

        for (size_t index = 0; index < info.offsets.size();++index )
        {
          DP_ASSERT(info.offsets[index] % 16 == 0);
          info.offsets[index] /= 16;
        }

        m_bufferData->setSize(info.data.size());
        m_bufferData->update(info.data.data());

        m_bufferChunkOffsets->setSize(info.offsets.size() * 4);
        m_bufferChunkOffsets->update(info.offsets.data());

        m_programUpdate->setUniform("chunkSize", static_cast<int>(it->first / 16));
        m_programUpdate->setUniform("numChunks", static_cast<int>(info.offsets.size()));
        m_programUpdate->setShaderStorageBuffer( "Input", m_bufferData );
        m_programUpdate->setShaderStorageBuffer( "Output", m_buffer );
        m_programUpdate->setShaderStorageBuffer( "Offsets", m_bufferChunkOffsets );
        m_programUpdate->apply();

        size_t numberOfBytes = info.offsets.size() * it->first;
        size_t numberOfVec4s = numberOfBytes / 16;
        size_t numberOfWorkgroups = (numberOfVec4s + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        glDispatchCompute(dp::checked_cast<GLuint>(numberOfWorkgroups), 1, 1);
        GLenum error = glGetError();
      }

      // clear all update infos.
      m_updateInfos.clear();

      // restore old program binding
      glUseProgram(id);
    }

  } // namespace gl
} // namespace dp
