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


#include <dp/gl/ProgramInstance.h>
#include <dp/gl/Program.h>
#include <dp/gl/Sampler.h>

namespace dp
{
  namespace gl
  {
    namespace
    {

      void setProgramUniform( GLenum type, GLint programID, GLint location, std::vector<char> const& data )
      {
        switch( type )
        {
          case GL_BOOL :
            DP_ASSERT( data.size() == sizeof(int) );
            glProgramUniform1iv( programID, location, 1, reinterpret_cast<GLint const*>(data.data()) );
            break;
          case GL_FLOAT :
            glProgramUniform1fv( programID, location, 1, reinterpret_cast<GLfloat const*>(data.data()) );
            break;
          case GL_FLOAT_MAT3 :
            glProgramUniformMatrix3fv( programID, location, 1, GL_FALSE, reinterpret_cast<GLfloat const*>(data.data()) );
            break;
          case GL_FLOAT_MAT4 :
            glProgramUniformMatrix4fv( programID, location, 1, GL_FALSE, reinterpret_cast<GLfloat const*>(data.data()) );
            break;
          case GL_FLOAT_VEC3 :
            glProgramUniform3fv( programID, location, 1, reinterpret_cast<GLfloat const*>(data.data()) );
            break;
          case GL_FLOAT_VEC4 :
            glProgramUniform4fv( programID, location, 1, reinterpret_cast<GLfloat const*>(data.data()) );
            break;
          case GL_INT :
            glProgramUniform1iv( programID, location, 1, reinterpret_cast<GLint const*>(data.data()) );
            break;
          case GL_INT_VEC3 :
            glProgramUniform3iv( programID, location, 1, reinterpret_cast<GLint const*>(data.data()) );
            break;
          default :
            DP_ASSERT( !"setProgramUniform: encountered unhandled uniform type!" );
            break;
        }
      }

    }

    ProgramInstanceSharedPtr ProgramInstance::create( ProgramSharedPtr const& program )
    {
      return( std::shared_ptr<ProgramInstance>( new ProgramInstance( program ) ) );
    }

    ProgramInstance::ProgramInstance( ProgramSharedPtr const& program )
      : m_program( program )
    {
      GLint programID = program->getGLId();

      m_shaderStorageBuffers.resize( program->getShaderStorageBlocks().size() );
#if !defined(NDEBUG)
      for ( size_t i=0 ; i<m_shaderStorageBuffers.size() ; i++ )
      {
        m_unsetShaderStorageBlocks.insert( i );
      }
#endif

      std::vector<Program::Uniform> const& uniforms = program->getActiveUniforms();
      for ( size_t i=0 ; i<uniforms.size() ; i++ )
      {
        if ( isImageType( uniforms[i].type ) )
        {
          m_imageUniforms.insert( std::make_pair( i, ImageUniformData() ) );
        }
        else if ( isSamplerType( uniforms[i].type ) )
        {
          m_samplerUniforms.insert( std::make_pair( i, SamplerUniformData() ) );
        }
        else if ( uniforms[i].location != -1 )
        {
          std::map<size_t,std::vector<char>>::iterator it = m_uniforms.insert( std::make_pair( i, std::vector<char>() ) ).first;
          it->second.resize( sizeOfType( uniforms[i].type ) );
        }

#if !defined(NDEBUG)
        m_unsetUniforms.insert( i );
#endif
      }

      std::vector<Program::Block> const& blocks = program->getActiveUniformBlocks();
      m_uniformBuffers.resize( blocks.size() );
      for ( size_t i=0 ; i<m_uniformBuffers.size() ; i++ )
      {
        m_uniformBuffers[i] = dp::gl::Buffer::create( Buffer::CORE, GL_DYNAMIC_DRAW, GL_UNIFORM_BUFFER );
        m_uniformBuffers[i]->setSize( blocks[i].dataSize );
      }
    }

    ProgramInstance::~ProgramInstance( )
    {
    }

    void ProgramInstance::apply() const
    {
      DP_ASSERT( m_unsetShaderStorageBlocks.empty() && m_unsetUniforms.empty() );

      GLint programID = m_program->getGLId();
      glUseProgram( programID );

      for ( std::map<size_t,ImageUniformData>::const_iterator it = m_imageUniforms.begin() ; it != m_imageUniforms.end() ; ++it )
      {
        DP_ASSERT( it->second.texture );
        glBindImageTexture( m_program->getActiveUniform( it->first ).unit, it->second.texture->getGLId(), 0, GL_FALSE, 0, it->second.access, it->second.texture->getInternalFormat() );
      }

      for ( std::map<size_t,SamplerUniformData>::const_iterator it = m_samplerUniforms.begin() ; it != m_samplerUniforms.end() ; ++it )
      {
        DP_ASSERT( it->second.texture );
        GLint unit = m_program->getActiveUniform( it->first ).unit;
        glActiveTexture( GL_TEXTURE0 + unit );
        glBindTexture( it->second.texture->getTarget(), it->second.texture->getGLId() );
        glBindSampler( unit , it->second.sampler->getGLId() );
      }

      for ( size_t i=0 ; i<m_shaderStorageBuffers.size() ; i++ )
      {
        DP_ASSERT( m_shaderStorageBuffers[i] );
        glBindBufferBase( GL_SHADER_STORAGE_BUFFER, m_program->getShaderStorageBlocks()[i].binding, m_shaderStorageBuffers[i]->getGLId() );
      }

      for ( std::map<size_t,std::vector<char>>::const_iterator it = m_uniforms.begin() ; it != m_uniforms.end() ; ++it )
      {
        Program::Uniform const& uniform = m_program->getActiveUniform( it->first );
        setProgramUniform( uniform.type, programID, uniform.location, it->second );
      }

      for ( size_t i=0 ; i<m_uniformBuffers.size() ; i++ )
      {
        DP_ASSERT( m_uniformBuffers[i] );
        glBindBufferBase( GL_UNIFORM_BUFFER, m_program->getActiveUniformBlocks()[i].binding, m_uniformBuffers[i]->getGLId() );
      }
    }

    ProgramSharedPtr const& ProgramInstance::getProgram() const
    {
      return( m_program );
    }

    void ProgramInstance::setImageUniform( std::string const& uniformName, TextureSharedPtr const& texture, GLenum access )
    {
      size_t uniformIndex = m_program->getActiveUniformIndex( uniformName );
      setImageUniform( uniformIndex, texture, access );
    }

    void ProgramInstance::setImageUniform( size_t uniformIndex, TextureSharedPtr const& texture, GLenum access )
    {
      std::map<size_t,ImageUniformData>::iterator it = m_imageUniforms.find( uniformIndex );
      DP_ASSERT( it != m_imageUniforms.end() );
      it->second.access = access;
      it->second.texture = texture;
#if !defined(NDEBUG)
      m_unsetUniforms.erase( uniformIndex );
#endif
    }

    ProgramInstance::ImageUniformData const& ProgramInstance::getImageUniform( std::string const& uniformName ) const
    {
      size_t uniformIndex = m_program->getActiveUniformIndex( uniformName );
      return( getImageUniform( uniformIndex ) );
    }

    ProgramInstance::ImageUniformData const& ProgramInstance::getImageUniform( size_t uniformIndex ) const
    {
      DP_ASSERT( m_unsetUniforms.find( uniformIndex ) == m_unsetUniforms.end() );
      std::map<size_t,ImageUniformData>::const_iterator it = m_imageUniforms.find( uniformIndex );
      DP_ASSERT( it != m_imageUniforms.end() );
      return( it->second );
    }

    void ProgramInstance::setSamplerUniform( std::string const& uniformName, TextureSharedPtr const& texture, SamplerSharedPtr const& sampler )
    {
      size_t uniformIndex = m_program->getActiveUniformIndex( uniformName );
      setSamplerUniform( uniformIndex, texture, sampler );
    }

    void ProgramInstance::setSamplerUniform( size_t uniformIndex, TextureSharedPtr const& texture, SamplerSharedPtr const& sampler )
    {
      std::map<size_t,SamplerUniformData>::iterator it = m_samplerUniforms.find( uniformIndex );
      DP_ASSERT( it != m_samplerUniforms.end() );
      it->second.sampler = sampler;
      it->second.texture = texture;
#if !defined(NDEBUG)
      m_unsetUniforms.erase( uniformIndex );
#endif
    }

    ProgramInstance::SamplerUniformData const& ProgramInstance::getSamplerUniform( std::string const& uniformName ) const
    {
      size_t uniformIndex = m_program->getActiveUniformIndex( uniformName );
      return( getSamplerUniform( uniformIndex ) );
    }

    ProgramInstance::SamplerUniformData const& ProgramInstance::getSamplerUniform( size_t uniformIndex ) const
    {
      DP_ASSERT( m_unsetUniforms.find( uniformIndex ) == m_unsetUniforms.end() );
      std::map<size_t,SamplerUniformData>::const_iterator it = m_samplerUniforms.find( uniformIndex );
      DP_ASSERT( it != m_samplerUniforms.end() );
      return( it->second );
    }

    void ProgramInstance::setShaderStorageBuffer( std::string const& ssbName, BufferSharedPtr const& buffer )
    {
      size_t ssbIndex = m_program->getShaderStorageBlockIndex( ssbName );
      setShaderStorageBuffer( ssbIndex, buffer );
    }

    void ProgramInstance::setShaderStorageBuffer( size_t ssbIndex, BufferSharedPtr const& buffer )
    {
      DP_ASSERT( ssbIndex < m_shaderStorageBuffers.size() );
      m_shaderStorageBuffers[ssbIndex] = buffer;
#if !defined(NDEBUG)
      m_unsetShaderStorageBlocks.erase( ssbIndex );
#endif
    }

    BufferSharedPtr const& ProgramInstance::getShaderStorageBuffer( std::string const& ssbName ) const
    {
      size_t ssbIndex = m_program->getShaderStorageBlockIndex( ssbName );
      return( getShaderStorageBuffer( ssbIndex ) );
    }

    BufferSharedPtr const& ProgramInstance::getShaderStorageBuffer( size_t ssbIndex ) const
    {
      DP_ASSERT( m_unsetShaderStorageBlocks.find( ssbIndex ) == m_unsetShaderStorageBlocks.end() );
      DP_ASSERT( ssbIndex < m_shaderStorageBuffers.size() );
      return( m_shaderStorageBuffers[ssbIndex] );
    }

  } // namespace gl
} // namespace dp
