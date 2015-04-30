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
#include <sstream>

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
    

    static const char* copyShader =
      "uniform int numChunks;\n"
      "uniform int chunkSize;\n"
      "\n"
      "layout( std430, binding = " TOSTRING(INPUT_BINDING) ") buffer Input {\n"
      "  CopyType inputBuffer[];\n"
      "};\n"
      "layout( std430, binding = " TOSTRING(OUTPUT_BINDING) " ) buffer Output {\n"
      "  CopyType outputBuffer[];\n"
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


    std::string generateCopyShader(std::string const & copyType)
    {
      std::ostringstream oss;
      oss << "#version 430\n";
      oss << "#define CopyType " << copyType << "\n";
      oss << copyShader;
      return oss.str();
    }


    BufferUpdater::BufferUpdater(dp::gl::BufferSharedPtr const& buffer, bool batchedUpdates)
      : m_batchedUpdates(batchedUpdates)
      , m_buffer(buffer)
    {
      if (m_batchedUpdates)
      {
        m_bufferData = dp::gl::Buffer::create(dp::gl::Buffer::CORE, GL_STREAM_DRAW, GL_SHADER_STORAGE_BUFFER);
        m_bufferChunkOffsets = dp::gl::Buffer::create(dp::gl::Buffer::CORE, GL_STREAM_DRAW, GL_SHADER_STORAGE_BUFFER);
        m_programUpdate4 = dp::gl::ProgramInstance::create( dp::gl::Program::create(dp::gl::ComputeShader::create(generateCopyShader("uint"))) );
        m_programUpdate8 = dp::gl::ProgramInstance::create( dp::gl::Program::create(dp::gl::ComputeShader::create(generateCopyShader("uvec2"))) );
        m_programUpdate16 = dp::gl::ProgramInstance::create( dp::gl::Program::create(dp::gl::ComputeShader::create(generateCopyShader("uvec4"))) );
      }
    }

    BufferUpdater::~BufferUpdater()
    {
    }

    void BufferUpdater::update(size_t offset, size_t size, void const* data)
    {
      if (offset + size >= m_buffer->getSize())
      {
        throw std::runtime_error("Offset + size exceeds buffer size.");
      }

      if (m_batchedUpdates)
      {
        UpdateInfo &info = m_updateInfos[dp::checked_cast<dp::Uint32>(size)];

        size_t dataOffset = info.data.size();
        info.data.resize(dataOffset + size);
        memcpy(info.data.data() + dataOffset, data, size);

        dp::Uint32 offset32 = dp::checked_cast<dp::Uint32>(offset);
        info.offsets.push_back(offset32);
        info.offsetMask |= offset32;
      }
      else
      {
        m_buffer->update(data, offset, size);
      }
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
        GLint glId = m_buffer->getGLId();

        bool useShader = true;

        unsigned int alignment;
        dp::gl::ProgramInstanceSharedPtr program;

        // determine if it's possible to use the shader to scatter data on the GPU side.
        dp::Uint32 alignmentMask = it->first | it->second.offsetMask;
        if (!(alignmentMask & 15))
        {
          program = m_programUpdate16;
          alignment = 16;
        }

        
        if (!(alignmentMask & 7))
        {
          program = m_programUpdate8;
          alignment = 8;
        }
        else if (!(alignmentMask & 3))
        {
          program = m_programUpdate4;
          alignment = 4;
        }
        else
        {
          // cannot use shader if alignment requirements are not fulfilled.
          useShader = false;  
        }

        // use shader to scatter updates on the GPU
        if (useShader)
        {
          for (size_t index = 0; index < info.offsets.size();++index )
          {
            info.offsets[index] /= alignment;
          }

          m_bufferData->setSize(info.data.size());
          m_bufferData->update(info.data.data());

          m_bufferChunkOffsets->setSize(info.offsets.size() * 4);
          m_bufferChunkOffsets->update(info.offsets.data());

          program->setUniform("chunkSize", static_cast<int>(it->first / alignment));
          program->setUniform("numChunks", static_cast<int>(info.offsets.size()));
          program->setShaderStorageBuffer( "Input", m_bufferData );
          program->setShaderStorageBuffer( "Output", m_buffer );
          program->setShaderStorageBuffer( "Offsets", m_bufferChunkOffsets );
          program->apply();

          size_t numberOfBytes = info.offsets.size() * it->first;
          size_t numberOfElements = numberOfBytes / alignment;
          size_t numberOfWorkgroups = (numberOfElements + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
          glDispatchCompute(dp::checked_cast<GLuint>(numberOfWorkgroups), 1, 1);
        }
        else // fall back to glNamedBufferSubData for each update
        {
          char *basePtr = (char*)info.data.data();
          for (size_t index = 0; index < info.offsets.size();++index )
          {
            m_buffer->update(basePtr + (index * it->first), info.offsets[index], it->first);
          }
        }
      }

      // clear all update infos.
      m_updateInfos.clear();

      // restore old program binding
      glUseProgram(id);
    }

  } // namespace gl
} // namespace dp
