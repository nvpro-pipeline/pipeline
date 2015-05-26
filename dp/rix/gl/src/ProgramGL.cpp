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


#include <dp/rix/gl/inc/ProgramGL.h>

#include <dp/rix/gl/inc/BufferGL.h>
#include <dp/rix/gl/inc/TextureGL.h>
#include <dp/rix/gl/inc/ContainerGL.h>
#include <dp/rix/gl/inc/SamplerStateGL.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>
#include <dp/rix/gl/inc/ProgramParameterBuffer.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <fstream>
#include <memory>

//#define DUMP_SHADERS

#if defined(DUMP_SHADERS)
static size_t shaderId = 0;
#endif

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

      using dp::Int8;
      using dp::Int16;
      using dp::Int32;
      using dp::Int64;

      using dp::Uint8;
      using dp::Uint16;
      using dp::Uint32;
      using dp::Uint64;

      ProgramGL::ProgramGL( const ProgramDescription& descr )
      {
        DP_ASSERT( descr.m_shader.m_type == PST_CODE );
        DP_ASSERT( dynamic_cast<const ProgramShaderCode*>(&descr.m_shader) );
        const ProgramShaderCode& sc = static_cast<const ProgramShaderCode&>( descr.m_shader );

        initEffect( sc );

        m_descriptors.resize( descr.m_numDescriptors );
        for ( unsigned int i=0; i<descr.m_numDescriptors; ++i )
        {
          DP_ASSERT( handleIsTypeOf<ContainerDescriptorGL>( descr.m_descriptors[i] ) );
          ContainerDescriptorGLSharedHandle cgl = handleCast< ContainerDescriptorGL >( descr.m_descriptors[i] );
          addDescriptor( cgl, i );
        }
      }

      ProgramGL::ProgramGL() 
      {
      }

      ProgramGL::~ProgramGL()
      {
      }

      void ProgramGL::initEffect( dp::rix::core::ProgramShaderCode const& psc )
      {
        std::vector<dp::gl::ShaderSharedPtr> shaders;
        for ( unsigned int i=0 ; i<psc.m_numShaders ; i++ )
        {
          shaders.push_back( dp::gl::Shader::create( getGLProgramDomain( psc.m_shaderTypes[i] ), psc.m_codes[i] ) );
        }

        dp::gl::Program::Parameters pp;
#if defined( DUMP_SHADERS )
        pp.m_binaryRetrievableHint = true;
#endif
        pp.m_separable = true;

        m_program = dp::gl::Program::create( shaders, pp );
        DP_ASSERT( m_program );
        GLuint programID = m_program->getGLId();

#if defined(DUMP_SHADERS)
        {
          DP_ASSERT(getenv("TMP"));
          std::string tmp = getenv("TMP");

          std::ostringstream nameShader;
          nameShader << "shader";
          nameShader << std::setfill('0');
          nameShader << std::setw(4) << shaderId;
          nameShader << "s.txt";
          std::ofstream outputShader( tmp + "\\" + nameShader.str() );
          for ( std::vector<dp::gl::ShaderSharedPtr>::const_iterator it = shaders.begin() ; it != shaders.end() ; ++it )
          {
            outputShader << dp::gl::shaderTypeToName( (*it)->getType() ) << std::endl;
            outputShader << (*it)->getSource();
          }

          std::ostringstream nameBinary;
          nameBinary << "shader";
          nameBinary << std::setfill('0');
          nameBinary << std::setw(4) << shaderId;
          nameBinary << "b.txt";

          std::vector<char> binary = m_program->getBinary().second;
          std::ofstream outputBinary( tmp + "\\" + nameBinary.str(), std::ios::binary );
          outputBinary.write( binary.data(), binary.size() );

          ++shaderId;
        }
#endif

        std::vector<dp::gl::Program::Block> const& ubos = m_program->getActiveUniformBlocks();
        for ( size_t i=0 ; i<ubos.size() ; i++ )
        {
          registerBufferBinding( ubos[i].name, ubos[i].binding, BBT_UBO );
        }

        glUseProgram( m_program->getGLId() );

        // build map of active atomic counters
        GLint maxUniformNameLen  = 0;
        glGetProgramiv( programID, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxUniformNameLen );
        std::vector<char> name;
        name.resize(maxUniformNameLen); // atomic counters may not be grouped into blocks

        if( GLEW_ARB_shader_atomic_counters )
        {
          GLint numBuffers;
          glGetProgramiv( programID, GL_ACTIVE_ATOMIC_COUNTER_BUFFERS, &numBuffers );
          for( GLint bufferIndex = 0; bufferIndex < numBuffers; ++bufferIndex )
          {
            GLint bindingPoint;
            glGetActiveAtomicCounterBufferiv( programID, bufferIndex, GL_ATOMIC_COUNTER_BUFFER_BINDING, &bindingPoint );
            GLint numCounters;
            glGetActiveAtomicCounterBufferiv( programID, bufferIndex, GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS, &numCounters );
            if( numCounters )  // This should always be true, but see bug 971682
            {
              std::vector<GLint> counterIndices;
              counterIndices.resize(numCounters);
              glGetActiveAtomicCounterBufferiv( programID, bufferIndex, GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES, &counterIndices[0] );
              for( int counterIndex = 0; counterIndex < numCounters; ++counterIndex )
              {
                GLint  nameLen = 0;
                GLint  arraySize;
                GLenum type;
                glGetActiveUniform( programID, counterIndices[counterIndex], maxUniformNameLen, &nameLen, &arraySize, &type, &name[0] );
                std::string atomicName(&name[0],nameLen);
                registerBufferBinding( atomicName, bindingPoint, BBT_ATOMIC_COUNTER );
              }
            }
          }
        }
        name.clear();

        std::vector<dp::gl::Program::Block> const& ssbs = m_program->getShaderStorageBlocks();
        for ( size_t i=0 ; i<ssbs.size() ; i++ )
        {
          registerBufferBinding( ssbs[i].name, ssbs[i].binding, BBT_SHADER_STORAGE_BUFFER );
        }
      }

      void ProgramGL::addDescriptor( ContainerDescriptorGLSharedHandle const& cd, unsigned int position )
      {
        m_descriptorToPosition[ cd.get() ] = position;
        m_descriptors[position] = cd;
      }

      void ProgramGL::registerBufferBinding( const std::string& name, int bindingIndex, BufferBindingType bufferBindingType )
      {
        DP_ASSERT( m_bufferBindings.find( name ) == m_bufferBindings.end() );

        BufferBinding bufferBinding;
        bufferBinding.bufferIndex = bindingIndex;
        bufferBinding.bufferBindingType = bufferBindingType;
        m_bufferBindings[name] = bufferBinding;
      }

      const ProgramGL::BufferBinding& ProgramGL::getBufferBinding( const std::string &name ) const
      {
        BufferBindingMap::const_iterator it = m_bufferBindings.find( name );
        if ( it == m_bufferBindings.end() )
        {
          static BufferBinding defaultBinding;
          return defaultBinding;
        }
        else
        {
          return it->second;
        }
      }

      unsigned int ProgramGL::getPosition( const ContainerDescriptorGLHandle & descriptorHandle ) const
      {
        std::map< ContainerDescriptorGLHandle, unsigned int >::const_iterator it = m_descriptorToPosition.find( descriptorHandle );
        if ( it != m_descriptorToPosition.end() )
        {
          return it->second;
        }
        return ~(unsigned int)(0);
      }

      ProgramGL::UniformInfos ProgramGL::getUniformInfos( ContainerDescriptorGLHandle containerDescriptor ) const
      {
        size_t numParameters = containerDescriptor->m_parameterInfos.size();

        UniformInfos uniformInfos;

        for ( size_t index = 0; index < numParameters; ++index )
        {
          std::string name = containerDescriptor->m_parameterInfos[index].m_name;
          size_t uniformIndex = m_program->getActiveUniformIndex( name );
          if ( uniformIndex != ~size_t(0) )
          {
            uniformInfos[containerDescriptor->getEntry( name.c_str() )] = m_program->getActiveUniform( uniformIndex );
          }
        }
        return uniformInfos;
      }

      ProgramGL::UniformInfos ProgramGL::getBufferInfos( ContainerDescriptorGLHandle containerDescriptor ) const
      {
        size_t numParameters = containerDescriptor->m_parameterInfos.size();

        UniformInfos uniformInfos;

        for ( size_t index = 0; index < numParameters; ++index )
        {
          std::string name = containerDescriptor->m_parameterInfos[index].m_name;
          size_t bufferVariableIndex = m_program->getActiveBufferVariableIndex( name );
          if ( bufferVariableIndex != ~size_t(0) )
          {
            uniformInfos[containerDescriptor->getEntry( name.c_str() )] = m_program->getActiveBufferVariable( bufferVariableIndex );
          }
        }

        return uniformInfos;
      }


    } // namespace gl
  } // namespace rix
} // namespace dp
