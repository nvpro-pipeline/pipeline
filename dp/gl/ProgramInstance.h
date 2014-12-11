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
#include <dp/gl/Program.h>
#include <dp/gl/Types.h>
#include <map>
#if !defined(NDEBUG)
#include <set>
#endif

namespace dp
{
  namespace gl
  {
    class ProgramInstance
    {
      public:
        struct ImageUniformData
        {
          GLenum            access;
          TextureSharedPtr  texture;
        };

        struct SamplerUniformData
        {
          SamplerSharedPtr  sampler;
          TextureSharedPtr  texture;
        };

      public:
        DP_GL_API static ProgramInstanceSharedPtr create( ProgramSharedPtr const& program );
        DP_GL_API virtual ~ProgramInstance();

        DP_GL_API void apply() const;
        DP_GL_API ProgramSharedPtr const& getProgram() const;

        DP_GL_API void setImageUniform( std::string const& uniformName, TextureSharedPtr const& texture, GLenum access );
        DP_GL_API void setImageUniform( size_t uniformIndex, TextureSharedPtr const& texture, GLenum access );
        DP_GL_API ImageUniformData const& getImageUniform( std::string const& uniformName ) const;
        DP_GL_API ImageUniformData const& getImageUniform( size_t uniformIndex ) const;

        DP_GL_API void setSamplerUniform( std::string const& uniformName, TextureSharedPtr const& texture, SamplerSharedPtr const& sampler );
        DP_GL_API void setSamplerUniform( size_t uniformIndex, TextureSharedPtr const& texture, SamplerSharedPtr const& sampler );
        DP_GL_API SamplerUniformData const& getSamplerUniform( std::string const& uniformName ) const;
        DP_GL_API SamplerUniformData const& getSamplerUniform( size_t uniformIndex ) const;

        DP_GL_API void setShaderStorageBuffer( std::string const& ssbName, BufferSharedPtr const& buffer );
        DP_GL_API void setShaderStorageBuffer( size_t ssbIndex, BufferSharedPtr const& buffer );
        DP_GL_API BufferSharedPtr const& getShaderStorageBuffer( std::string const& ssbName ) const;
        DP_GL_API BufferSharedPtr const& getShaderStorageBuffer( size_t ssbIndex ) const;

        template <typename T> void setUniform( std::string const& uniformName, T const& value );
        template <typename T> void setUniform( size_t uniformIndex, T const& value );
        template <typename T> bool getUniform( std::string const& uniformName, T & value ) const;
        template <typename T> bool getUniform( size_t uniformIndex, T & value ) const;

      protected:
        DP_GL_API ProgramInstance( ProgramSharedPtr const& program );

      private:
        ProgramSharedPtr                    m_program;
        std::map<size_t,ImageUniformData>   m_imageUniforms;
        std::map<size_t,SamplerUniformData> m_samplerUniforms;
        std::vector<BufferSharedPtr>        m_shaderStorageBuffers;
        std::vector<BufferSharedPtr>        m_uniformBuffers;
        std::map<size_t,std::vector<char>>  m_uniforms;
#if !defined(NDEBUG)
        std::set<size_t>  m_unsetShaderStorageBlocks;
        std::set<size_t>  m_unsetUniforms;
#endif
    };

    template <typename T>
    inline void ProgramInstance::setUniform( std::string const& uniformName, T const& value )
    {
      size_t uniformIndex = m_program->getActiveUniformIndex( uniformName );
      if ( uniformIndex != ~0 )
      {
        setUniform( uniformIndex, value );
      }
    }

    template <typename T>
    inline void ProgramInstance::setUniform( size_t uniformIndex, T const& value )
    {
      Program::Uniform const& uniform = m_program->getActiveUniform( uniformIndex );
      DP_ASSERT( !isImageType( uniform.type ) && !isSamplerType( uniform.type ) );
      DP_ASSERT( TypeTraits<T>::glType() == uniform.type );
      DP_ASSERT( uniform.arraySize == 1 );
      if ( uniform.location != -1 )
      {
        DP_ASSERT( uniform.blockIndex == -1 );
        std::map<size_t,std::vector<char>>::iterator it = m_uniforms.find( uniformIndex );
        DP_ASSERT( ( it != m_uniforms.end() ) && ( it->second.size() == sizeOfType( uniform.type ) ) );
        memcpy( it->second.data(), &value, sizeOfType( uniform.type ) );
      }
      else
      {
        DP_ASSERT( uniform.blockIndex != -1 );
        DP_ASSERT( uniform.blockIndex < m_uniformBuffers.size() );
        setBufferUniform( m_uniformBuffers[uniform.blockIndex], uniform, value );
      }
#if !defined(NDEBUG)
      m_unsetUniforms.erase( uniformIndex );
#endif
    }

    template <typename T>
    inline bool ProgramInstance::getUniform( std::string const& uniformName, T & value ) const
    {
      size_t uniformIndex = m_program->getActiveUniformIndex( uniformName );
      if ( uniformIndex != ~0 )
      {
        return( getUniform( uniformIndex, value ) );
      }
      return( false );
    }

    template <typename T>
    inline bool ProgramInstance::getUniform( size_t uniformIndex, T & value ) const
    {
      DP_ASSERT( m_unsetUniforms.find( uniformIndex ) == m_unsetUniforms.end() );
      Program::Uniform const& uniform = m_program->getActiveUniform( uniformIndex );
      DP_ASSERT( !isImageType( uniform.type ) && !isSamplerType( uniform.type ) );
      DP_ASSERT( TypeTraits<T>::glType() == uniform.type );
      DP_ASSERT( uniform.arraySize == 1 );
      if ( uniform.location != -1 )
      {
        DP_ASSERT( uniform.blockIndex == -1 );
        std::map<size_t,UniformData>::iterator it = m_uniforms.find( uniformIndex );
        DP_ASSERT( ( it != m_uniforms.end() ) && ( it->second.size() == sizeOfType( uniform.type ) ) );
        memcpy( &value, it->second.data(), sizeOfType( uniform.type ) );
      }
      else
      {
        DP_ASSERT( uniform.blockIndex != -1 );
        DP_ASSERT( uniform.blockIndex < m_uniformBuffers.size() );
        getBufferUniform( m_uniformBuffers[uniform.blockIndex], uniform, value );
      }
    }

    namespace
    {
      template <typename T>
      inline void setBufferUniform( BufferSharedPtr const& buffer, Program::Uniform const& uniform, T const& value )
      {
        DP_ASSERT( sizeOfType( uniform.type ) == sizeof(T) );
        buffer->update( &value, uniform.offset, sizeof(T) );
      }

      template <>
      inline void setBufferUniform( BufferSharedPtr const& buffer, Program::Uniform const& uniform, bool const& value )
      {
        setBufferUniform<int>( buffer, uniform, value );
      }

      template <>
      inline void setBufferUniform( BufferSharedPtr const& buffer, Program::Uniform const& uniform, dp::math::Mat33f const& value )
      {
        DP_ASSERT( uniform.matrixStride == 4 * sizeof(float) );
        buffer->update( &value[0], uniform.offset                         , 3*sizeof(float) );
        buffer->update( &value[1], uniform.offset +   uniform.matrixStride, 3*sizeof(float) );
        buffer->update( &value[2], uniform.offset + 2*uniform.matrixStride, 3*sizeof(float) );
      }

      template <typename T>
      inline void getBufferUniform( BufferSharedPtr const& buffer, Program::Uniform const& uniform, T & value )
      {
        DP_ASSERT( sizeOfType( uniform.type ) == sizeof(T) );
        void* ptr = buffer->map( GL_MAP_READ_BIT, uniform.offset, sizeof(T) );
        memcpy( &value, ptr, sizeof(T) );
      }

      template <>
      inline void getBufferUniform( BufferSharedPtr const& buffer, Program::Uniform const& uniform, bool & value )
      {
        int iValue;
        getBufferUniform( buffer, uniform, iValue );
        value = !!iValue;
      }

      template <>
      inline void getBufferUniform( BufferSharedPtr const& buffer, Program::Uniform const& uniform, dp::math::Mat33f & value )
      {
        DP_ASSERT( uniform.matrixStride == 4 * sizeof(float) );
        float* ptr = reinterpret_cast<float*>(buffer->map( GL_MAP_READ_BIT, uniform.offset, 12*sizeof(float) ));   // 3 * (3+1) * sizeof(float) !!
        memcpy( &value[0], ptr,     3*sizeof(float) );
        memcpy( &value[1], ptr + 4, 3*sizeof(float) );
        memcpy( &value[2], ptr + 8, 3*sizeof(float) );
      }

    }

  } // namespace gl
} // namespace dp