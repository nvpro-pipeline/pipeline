// Copyright NVIDIA Corporation 2010-2014
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


#include <dp/gl/Program.h>
#include <dp/gl/Sampler.h>
#include <dp/util/Array.h>
#include <sstream>

namespace dp
{
  namespace gl
  {

    ProgramSharedPtr Program::create( std::vector<ShaderSharedPtr> const& shaders, Parameters const& programParameters )
    {
      return( std::shared_ptr<Program>( new Program( shaders, programParameters ) ) );
    }

    ProgramSharedPtr Program::create( VertexShaderSharedPtr const& vertexShader, FragmentShaderSharedPtr const& fragmentShader, Parameters const& programParameters )
    {
      std::vector<ShaderSharedPtr> shaders;
      shaders.push_back( vertexShader );
      shaders.push_back( fragmentShader );
      return( std::shared_ptr<Program>( new Program( shaders, programParameters ) ) );
    }

    ProgramSharedPtr Program::create( ComputeShaderSharedPtr const& computeShader, Parameters const& programParameters )
    {
      std::vector<ShaderSharedPtr> shaders;
      shaders.push_back( computeShader );
      return( std::shared_ptr<Program>( new Program( shaders, programParameters ) ) );
    }

    Program::Program( std::vector<ShaderSharedPtr> const& shaders, Parameters const& parameters )
      : m_activeAttributesCount(0)
      , m_activeAttributesMask(0)
      , m_shaders( shaders )
    {
#if !defined(NDEBUG)
      std::set<GLenum> types;
      for ( std::vector<ShaderSharedPtr>::const_iterator it = m_shaders.begin() ; it != m_shaders.end() ; ++it )
      {
        DP_ASSERT( *it && types.insert( (*it)->getType() ).second );
      }
#endif

      GLuint id = glCreateProgram();
      setGLId( id );

      DP_ASSERT( glIsProgram( id ) && "failed to create program" );

      glProgramParameteri( id, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, parameters.m_binaryRetrievableHint );
      glProgramParameteri( id, GL_PROGRAM_SEPARABLE, parameters.m_separable );

      for ( std::vector<ShaderSharedPtr>::const_iterator it = m_shaders.begin() ; it != m_shaders.end() ; ++it )
      {
        glAttachShader( id, (*it)->getGLId() );
      }
      glLinkProgram( id );

      GLint result;
      glGetProgramiv( id, GL_LINK_STATUS, &result );
      if ( !result )
      {
        GLint errorLen;
        glGetProgramiv( id, GL_INFO_LOG_LENGTH, &errorLen );

        std::string buffer;
        buffer.resize( errorLen, 0 );
        glGetProgramInfoLog( id, errorLen, NULL, &buffer[0] );

        std::ostringstream error;
        error << "failed to link program: " << std::endl << buffer << std::endl;
        throw std::runtime_error(error.str().c_str());
      }

      GLint maxNameLength = 0;
      glGetProgramiv( id, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxNameLength );
      // some drivers do not add the trailing 0 to the name length. increase by 1 to get all characters.
      std::vector<char> name( maxNameLength + 1 );

      GLint numActiveUniforms = 0;
      glGetProgramInterfaceiv( m_id, GL_UNIFORM, GL_ACTIVE_RESOURCES, &numActiveUniforms );
      m_uniforms.resize( numActiveUniforms );

      GLint unit = 0;
      for ( GLint i=0 ; i<numActiveUniforms; i++ )
      {
        GLenum const properties[] = {
            GL_BLOCK_INDEX  // 0
          , GL_LOCATION     // 1
          , GL_TYPE         // 2
          , GL_ARRAY_SIZE   // 3
          , GL_OFFSET       // 4
          , GL_ARRAY_STRIDE // 5
          , GL_MATRIX_STRIDE// 6
          , GL_IS_ROW_MAJOR // 7
        };

        GLsizei const numProperties = sizeof dp::util::array(properties);
        GLint values[numProperties];

        glGetProgramResourceiv( m_id, GL_UNIFORM, i, numProperties, properties, numProperties, NULL, values );

        // get uniform name
        glGetProgramResourceName( m_id, GL_UNIFORM, i, GLsizei(name.size()), NULL, name.data() );

        m_uniforms[i].blockIndex = values[0];
        m_uniforms[i].location = values[1];
        m_uniforms[i].type = values[2];
        m_uniforms[i].arraySize = values[3];
        m_uniforms[i].offset = values[4];
        m_uniforms[i].arrayStride = values[5];
        m_uniforms[i].matrixStride = values[6];
        m_uniforms[i].isRowMajor = (values[7] != 0);

        m_uniformsMap[name.data()] = i;

        if ( isImageType( m_uniforms[i].type ) || isSamplerType( m_uniforms[i].type ) )
        {
          DP_ASSERT( m_uniforms[i].arraySize == 1 );
          m_uniforms[i].unit = unit++;
          glProgramUniform1i( m_id, m_uniforms[i].location, m_uniforms[i].unit );
        }
      }

      GLint numActiveUniformBlocks = 0;
      glGetProgramInterfaceiv( m_id, GL_UNIFORM_BLOCK, GL_ACTIVE_RESOURCES, &numActiveUniformBlocks );
      if ( 0 < numActiveUniformBlocks )
      {
        GLint maxNameLength = 0;
        glGetProgramInterfaceiv( m_id, GL_UNIFORM_BLOCK, GL_MAX_NAME_LENGTH, &maxNameLength );
        name.resize( maxNameLength + 1 );

        std::set<GLint> usedBindings;
        std::set<GLint> unboundBlocks;

        m_uniformBlocks.resize( numActiveUniformBlocks );
        for ( GLint i=0 ; i<numActiveUniformBlocks ; i++ )
        {
          glGetProgramResourceName( m_id, GL_UNIFORM_BLOCK, i, GLsizei(name.size()), NULL, name.data() );
          m_uniformBlocksMap[name.data()] = i;
          m_uniformBlocks[i].name = name.data();

          GLenum const properties[] = {
              GL_BUFFER_BINDING   // 0
            , GL_BUFFER_DATA_SIZE // 1
          };
          GLsizei const numProperties = sizeof dp::util::array(properties);
          GLint values[numProperties];

          glGetProgramResourceiv( m_id, GL_UNIFORM_BLOCK, i, numProperties, properties, numProperties, NULL, values );
          m_uniformBlocks[i].dataSize = values[1];
          if ( values[0] )
          {
            usedBindings.insert( values[0] );
            m_uniformBlocks[i].binding = values[0];
          }
          else
          {
            unboundBlocks.insert( i );
          }
        }

        GLint currentBinding = 1;
        for ( std::set<GLint>::iterator it=unboundBlocks.begin() ; it!=unboundBlocks.end() ; ++it )
        {
          // search first free binding
          while ( usedBindings.find( currentBinding) != usedBindings.end() )
          {
            currentBinding++;
          }
          m_uniformBlocks[*it].binding = currentBinding;
          glUniformBlockBinding( m_id, *it, currentBinding++ );
        }
      }

      GLint numActiveBufferVariables;
      glGetProgramInterfaceiv( m_id, GL_BUFFER_VARIABLE, GL_ACTIVE_RESOURCES, &numActiveBufferVariables );
      m_bufferVariables.resize( numActiveBufferVariables );

      glGetProgramInterfaceiv( m_id, GL_BUFFER_VARIABLE, GL_MAX_NAME_LENGTH, &maxNameLength );
      name.resize( maxNameLength + 1 );

      for ( GLint i=0 ; i<numActiveBufferVariables; i++ )
      {
        GLenum const properties[] = {
            GL_TYPE         // 0
          , GL_ARRAY_SIZE   // 1
          , GL_OFFSET       // 2
          , GL_BLOCK_INDEX  // 3
          , GL_ARRAY_STRIDE // 4
          , GL_MATRIX_STRIDE// 5
          , GL_IS_ROW_MAJOR // 6
        };

        GLsizei const numProperties = sizeof dp::util::array(properties);
        GLint values[numProperties];

        glGetProgramResourceiv( m_id, GL_BUFFER_VARIABLE, i, numProperties, properties, numProperties, NULL, values );

        m_bufferVariables[i].location = -1;
        m_bufferVariables[i].type = values[0];
        m_bufferVariables[i].arraySize = values[1];
        m_bufferVariables[i].offset = values[2];
        m_bufferVariables[i].blockIndex = values[3];
        m_bufferVariables[i].arrayStride = values[4];
        m_bufferVariables[i].matrixStride = values[5];
        m_bufferVariables[i].isRowMajor = ( values[6] != 0 );

        glGetProgramResourceName( m_id, GL_BUFFER_VARIABLE, i, GLsizei(name.size()), NULL, name.data() );
        m_bufferVariablesMap[name.data()] = i;
      }

      GLint numShaderStorageBlocks = 0;
      glGetProgramInterfaceiv( m_id, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &numShaderStorageBlocks );

      if ( 0 < numShaderStorageBlocks )
      {
        m_shaderStorageBlocks.resize( numShaderStorageBlocks );

#if !defined(NDEBUG)
        GLint maxNameLength = 0;
        glGetProgramInterfaceiv( m_id, GL_SHADER_STORAGE_BLOCK, GL_MAX_NAME_LENGTH, &maxNameLength );
        DP_ASSERT( maxNameLength < name.size() );
#endif

        std::set<GLint> usedBindings;
        std::set<GLint> unboundBlocks;

        for ( GLint i=0 ; i<numShaderStorageBlocks ; i++ )
        {
          glGetProgramResourceName( m_id, GL_SHADER_STORAGE_BLOCK, i, static_cast<GLsizei>(name.size()), NULL, name.data() );
          m_shaderStorageBlocksMap[name.data()] = i;
          m_shaderStorageBlocks[i].name = name.data();

          GLenum const properties[] = {
              GL_BUFFER_BINDING     // 0
            , GL_BUFFER_DATA_SIZE   // 1
          };
          GLsizei const numProperties = sizeof dp::util::array(properties);
          GLint values[numProperties];

          glGetProgramResourceiv( m_id, GL_SHADER_STORAGE_BLOCK, i, numProperties, properties, numProperties, NULL, values );
          m_shaderStorageBlocks[i].dataSize = values[1];
          if ( values[0] )
          {
            usedBindings.insert( values[0] );
            m_shaderStorageBlocks[i].binding = values[0];
          }
          else
          {
            unboundBlocks.insert( i );
          }
        }

        GLint currentBinding = 1;
        for ( std::set<GLint>::iterator it=unboundBlocks.begin() ; it!=unboundBlocks.end() ; ++it )
        {
          // search first free binding
          while ( usedBindings.find( currentBinding) != usedBindings.end() )
          {
            currentBinding++;
          }
          m_shaderStorageBlocks[*it].binding = currentBinding;
          glShaderStorageBlockBinding( m_id, *it, currentBinding++ );
        }
      }

      // build mask of active vertex attributes
      GLint activeAttributeMaxLength;
      glGetProgramiv( id, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &activeAttributeMaxLength );
      name.resize( activeAttributeMaxLength );

      GLint activeAttributesCount;
      glGetProgramiv( id, GL_ACTIVE_ATTRIBUTES, &activeAttributesCount );

      // the driver reports 1 attribute while activeAttributeMaxLength is 0.
      // since attributes always have names? something strange is going on.
      if ( activeAttributeMaxLength == 0 )
      {
        //DP_ASSERT( false );
        activeAttributesCount = 0;
      }

      m_activeAttributesCount = activeAttributesCount;
      for ( GLint i=0 ; i<activeAttributesCount ; i++ )
      {
        GLint size;
        GLenum type;
        glGetActiveAttrib( id, i, activeAttributeMaxLength, nullptr, &size, &type, &name[0] );
        m_activeAttributesMask |= 1 << glGetAttribLocation( id, &name[0] );
      }
    }

    Program::~Program( )
    {
      if ( m_id )
      {
        if ( getShareGroup() )
        {
          DEFINE_PTR_TYPES( CleanupTask );
          class CleanupTask : public ShareGroupTask
          {
            public:
              static CleanupTaskSharedPtr create( GLuint id )
              {
                return( std::shared_ptr<CleanupTask>( new CleanupTask( id ) ) );
              }

              virtual void execute() { glDeleteProgram( m_id ); }

            protected:
              CleanupTask( GLuint id ) : m_id( id ) {}

            private:
              GLuint m_id;
          };

          // make destructor exception safe
          try
          {
            getShareGroup()->executeTask( CleanupTask::create( m_id ) );
          } catch (...) {}
        }
        else
        {
          glDeleteProgram( m_id );
        }
      }
    }

    unsigned int Program::getActiveAttributesCount() const
    {
      return( m_activeAttributesCount );
    }

    unsigned int Program::getActiveAttributesMask() const
    {
      return( m_activeAttributesMask );
    }

    GLint Program::getAttributeLocation( std::string const& name ) const
    {
      return( glGetAttribLocation( m_id, name.c_str() ) );
    }

    std::pair<GLenum,std::vector<char>> Program::getBinary() const
    {
      GLint binaryLength(0);
      glGetProgramiv( m_id, GL_PROGRAM_BINARY_LENGTH, &binaryLength );

      GLenum binaryFormat(0);
      std::vector<char> binary(binaryLength);
      glGetProgramBinary( m_id, binaryLength, nullptr, &binaryFormat, binary.data() );

      return( std::make_pair( binaryFormat, binary ) );
    }

    std::vector<ShaderSharedPtr> const& Program::getShaders() const
    {
      return( m_shaders );
    }

    std::vector<Program::Uniform> const& Program::getActiveUniforms() const
    {
      return( m_uniforms );
    }

    Program::Uniform const& Program::getActiveUniform( size_t index ) const
    {
      DP_ASSERT( index < m_uniforms.size() );
      return( m_uniforms[index] );
    }

    size_t Program::getActiveUniformIndex( std::string const& uniformName ) const
    {
      std::map<std::string,size_t>::const_iterator it = m_uniformsMap.find( uniformName );
      return( it == m_uniformsMap.end() ? ~0 : it->second );
    }

    std::vector<Program::Uniform> const& Program::getActiveBufferVariables() const
    {
      return( m_bufferVariables );
    }

    Program::Uniform const& Program::getActiveBufferVariable( size_t index ) const
    {
      DP_ASSERT( index < m_bufferVariables.size() );
      return( m_bufferVariables[index] );
    }

    size_t Program::getActiveBufferVariableIndex( std::string const& bufferVariableName ) const
    {
      std::map<std::string,size_t>::const_iterator it = m_bufferVariablesMap.find( bufferVariableName );
      return( it == m_bufferVariablesMap.end() ? ~0 : it->second );
    }

    std::vector<Program::Block> const& Program::getActiveUniformBlocks() const
    {
      return( m_uniformBlocks );
    }

    Program::Block const& Program::getActiveUniformBlock( size_t index ) const
    {
      DP_ASSERT( index < m_uniformBlocks.size() );
      return( m_uniformBlocks[index] );
    }

    size_t Program::getActiveUniformBlockIndex( std::string const& uniformName ) const
    {
      std::map<std::string,size_t>::const_iterator it = m_uniformBlocksMap.find( uniformName );
      return( it == m_uniformBlocksMap.end() ? ~0 : it->second );
    }

    std::vector<Program::Block> const& Program::getShaderStorageBlocks() const
    {
      return( m_shaderStorageBlocks );
    }

    Program::Block const& Program::getShaderStorageBlock( size_t index ) const
    {
      DP_ASSERT( index < m_shaderStorageBlocks.size() );
      return( m_shaderStorageBlocks[index] );
    }

    size_t Program::getShaderStorageBlockIndex( std::string const& ssbName ) const
    {
      std::map<std::string,size_t>::const_iterator it = m_shaderStorageBlocksMap.find( ssbName );
      return( it == m_shaderStorageBlocksMap.end() ? ~0 : it->second );
    }

    size_t sizeOfType( GLenum type )
    {
      switch( type )
      {
        case GL_BOOL :                return(      sizeof(int) );
        case GL_FLOAT :               return(      sizeof(float) );
        case GL_FLOAT_VEC3 :          return(  3 * sizeof(float) );
        case GL_FLOAT_VEC4 :          return(  4 * sizeof(float) );
        case GL_FLOAT_MAT3 :          return( 12 * sizeof(float) );    // 3 * (3+1) !!
        case GL_FLOAT_MAT4 :          return( 16 * sizeof(float) );
        case GL_INT :                 return(      sizeof(int) );
        case GL_INT_SAMPLER_BUFFER :  return(      sizeof(int) );
        case GL_INT_VEC3 :            return(  3 * sizeof(int) );
        case GL_INT_VEC4 :            return(  4 * sizeof(int) );
        case GL_SAMPLER_1D :          return(      sizeof(int) );
        case GL_SAMPLER_2D :          return(      sizeof(int) );
        case GL_SAMPLER_3D :          return(      sizeof(int) );
        case GL_SAMPLER_BUFFER :      return(      sizeof(int) );
        case GL_UNSIGNED_INT :        return(      sizeof(unsigned int) );
        default :
          DP_ASSERT( false );
          return( 0 );
      }
    }

  } // namespace gl
} // namespace dp
