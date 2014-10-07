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

#include <string>
#include <map>
#include <vector>

#include "ContainerGL.h"
#include <dp/gl/Program.h>
#include <dp/util/Types.h>
#include <dp/rix/gl/RiXGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      class ProgramGL : public dp::rix::core::Program
      {
      public:
        typedef std::map< ContainerDescriptorGLHandle, unsigned int > DescriptorIndexMap;

        enum BufferBindingType
        {
            BBT_UBO
          , BBT_ATOMIC_COUNTER
          , BBT_SHADER_STORAGE_BUFFER
          , BBT_NONE
        };

        struct BufferBinding
        {
          BufferBinding()
            : bufferIndex( 0 )
            , bufferBindingType( BBT_NONE )
          {
          }

          int               bufferIndex;
          BufferBindingType bufferBindingType;
        };

      public:
        ProgramGL();
        ProgramGL( const dp::rix::core::ProgramDescription& description );
        ~ProgramGL();

        dp::gl::SharedProgram const& getProgram()  { return m_program; }

        unsigned int getPosition( const ContainerDescriptorGLHandle & descriptorHandle ) const;
        std::vector<ContainerDescriptorGLSharedHandle> const & getDescriptors() const          { return m_descriptors; }
        DescriptorIndexMap const &                             getDescriptorToIndexMap() const { return m_descriptorToPosition; }

        BufferBinding const& getBufferBinding( std::string const& name ) const;

        typedef std::map<dp::rix::core::ContainerEntry, dp::gl::Program::Uniform> UniformInfos;
        UniformInfos getUniformInfos( ContainerDescriptorGLHandle containerDescriptor ) const;
        UniformInfos getBufferInfos( ContainerDescriptorGLHandle containerDescriptor ) const;

      private:
        typedef std::map<std::string, BufferBinding> BufferBindingMap;
        BufferBindingMap m_bufferBindings;

      private:
        void registerBufferBinding( const std::string& name, int bufferIndex, BufferBindingType bindingType );

        void addDescriptor( ContainerDescriptorGLSharedHandle const& cd, unsigned int position );
        void initEffect( dp::rix::core::ProgramShaderCode const& psc );

      private:
        dp::gl::SharedProgram                          m_program;
        std::vector<ContainerDescriptorGLSharedHandle> m_descriptors;
        DescriptorIndexMap                             m_descriptorToPosition;
      };

    } // namespace gl
  } // namespace rix
} // namespace dp
