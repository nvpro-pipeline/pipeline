// Copyright NVIDIA Corporation 2010
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
#include <dp/util/Array.h>

namespace dp
{
  namespace gl
  {

    SmartProgram Program::create( std::vector<SmartShader> const& shaders, Parameters const& programParameters )
    {
      return( new Program( shaders, programParameters ) );
    }

    SmartProgram Program::create( SmartVertexShader const& vertexShader, SmartFragmentShader const& fragmentShader, Parameters const& programParameters )
    {
      std::vector<SmartShader> shaders;
      shaders.push_back( vertexShader );
      shaders.push_back( fragmentShader );
      return( new Program( shaders, programParameters ) );
    }

    SmartProgram Program::create( SmartComputeShader const& computeShader, Parameters const& programParameters )
    {
      std::vector<SmartShader> shaders;
      shaders.push_back( computeShader );
      return( new Program( shaders, programParameters ) );
    }

    Program::Program( std::vector<SmartShader> const& shaders, Parameters const& parameters )
      : m_activeAttributesCount(0)
      , m_activeAttributesMask(0)
      , m_shaders( shaders )
    {
#if !defined(NDEBUG)
      std::set<GLenum> types;
      for ( std::vector<SmartShader>::const_iterator it = m_shaders.begin() ; it != m_shaders.end() ; ++it )
      {
        DP_ASSERT( *it && types.insert( (*it)->getType() ).second );
      }
#endif

      GLuint id = glCreateProgram();
      setGLId( id );

      DP_ASSERT( glIsProgram( id ) && "failed to create program" );

      glProgramParameteri( id, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, parameters.m_binaryRetrievableHint );
      glProgramParameteri( id, GL_PROGRAM_SEPARABLE, parameters.m_separable );

      for ( std::vector<SmartShader>::const_iterator it = m_shaders.begin() ; it != m_shaders.end() ; ++it )
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

      GLint activeUniformMaxLength = 0;
      glGetProgramiv( id, GL_ACTIVE_UNIFORM_MAX_LENGTH, &activeUniformMaxLength );
      std::vector<char> name( activeUniformMaxLength );

      GLint activeUniformsCount = 0;
      glGetProgramiv( id, GL_ACTIVE_UNIFORMS, &activeUniformsCount );
      m_uniforms.resize( activeUniformsCount );

      glUseProgram( id );
      for ( GLint i=0 ; i<activeUniformsCount ; i++ )
      {
        glGetActiveUniform( id, i, activeUniformMaxLength, nullptr, &m_uniforms[i].size, &m_uniforms[i].type, name.data() );
        m_uniforms[i].name = name.data();
        m_uniforms[i].location = glGetUniformLocation( id, m_uniforms[i].name.data() );

        if ( isSamplerType( m_uniforms[i].type ) )
        {
          DP_ASSERT( m_uniforms[i].size == 1 );
          glUniform1i( m_uniforms[i].location, dp::util::checked_cast<GLint>(m_samplerUniforms.size() ) );
          m_samplerUniforms.push_back( i );
        }
        else if ( isImageType( m_uniforms[i].type ) )
        {
          DP_ASSERT( m_uniforms[i].size == 1 );
          glUniform1i( m_uniforms[i].location, dp::util::checked_cast<GLint>(m_imageUniforms.size()) );
          m_imageUniforms.push_back( ImageData() );
          m_imageUniforms.back().index = i;
        }
      }
      glUseProgram( 0 );

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
      class CleanupTask : public ShareGroupTask
      {
        public:
          CleanupTask( GLuint id ) : m_id( id ) {}

          virtual void execute() { glDeleteProgram( m_id ); }

        private:
          GLuint m_id;
      };

      if ( getGLId() && getShareGroup() )
      {
        // make destructor exception safe
        try
        {
          getShareGroup()->executeTask( new CleanupTask( getGLId() ) );
        } catch (...) {}
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

    std::pair<GLenum,std::vector<char>> Program::getBinary() const
    {
      GLint binaryLength(0);
      glGetProgramiv( getGLId(), GL_PROGRAM_BINARY_LENGTH, &binaryLength );

      GLenum binaryFormat(0);
      std::vector<char> binary(binaryLength);
      glGetProgramBinary( getGLId(), binaryLength, nullptr, &binaryFormat, binary.data() );

      return( std::make_pair( binaryFormat, binary ) );
    }

    std::vector<SmartShader> const& Program::getShaders() const
    {
      return( m_shaders );
    }

    GLint Program::getUniformLocation( std::string const& name ) const
    {
      for ( std::vector<UniformData>::const_iterator it = m_uniforms.begin() ; it != m_uniforms.end() ; ++it )
      {
        if ( it->name == name )
        {
          return( it->location );
        }
      }
      return( -1 );
    }

    GLenum Program::getUniformType( GLint location ) const
    {
      for ( std::vector<UniformData>::const_iterator it = m_uniforms.begin() ; it != m_uniforms.end() ; ++it )
      {
        if ( it->location == location )
        {
          return( it->type );
        }
      }
      DP_ASSERT( false );
      return( GL_NONE );
    }

    void Program::setImageTexture( std::string const& textureName, SmartTexture const& texture, GLenum access )
    {
      bool found = false;
      for ( std::vector<ImageData>::iterator it = m_imageUniforms.begin() ; it != m_imageUniforms.end() && !found ; ++it )
      {
        if ( m_uniforms[it->index].name == textureName )
        {
          it->access  = access;
          it->texture = texture;
          found = true;
        }
      }
      DP_ASSERT( found );
    }

    std::vector<Program::ImageData>::const_iterator Program::beginImageUnits() const
    {
      return( m_imageUniforms.begin() );
    }

    std::vector<Program::ImageData>::const_iterator Program::endImageUnits() const
    {
      return( m_imageUniforms.end() );
    }

    Program::Uniforms Program::getActiveUniforms()
    {
      GLint numActiveUniforms;
      glGetProgramInterfaceiv( getGLId(), GL_UNIFORM, GL_ACTIVE_RESOURCES, &numActiveUniforms );

      std::vector<GLuint> indices;
      indices.resize(numActiveUniforms);
      for (GLint index = 0;index < numActiveUniforms;++index)
      {
        indices[index] = index;
      }
      return getActiveUniforms(indices);
    }

    Program::Uniforms Program::getActiveBufferVariables()
    {
      GLint numActiveVariables;
      glGetProgramInterfaceiv( getGLId(), GL_BUFFER_VARIABLE, GL_ACTIVE_RESOURCES, &numActiveVariables );

      std::vector<GLuint> indices;
      indices.resize(numActiveVariables);
      for (GLint index = 0;index < numActiveVariables;++index)
      {
        indices[index] = index;
      }
      return getActiveBufferVariables(indices);
    }

    Program::Uniforms Program::getActiveUniforms( std::vector<std::string> const & uniformNames )
    {
      std::vector<GLuint> indices;
      indices.reserve(uniformNames.size());
      for ( size_t idx = 0; idx < uniformNames.size(); ++idx )
      {
        GLuint locationIndex = glGetProgramResourceIndex( getGLId(), GL_UNIFORM, uniformNames[idx].c_str() );
        if ( locationIndex != GL_INVALID_INDEX )
        {
          indices.push_back(GLuint(locationIndex));
        }
      }
      return getActiveUniforms( indices );
    }

    Program::Uniforms Program::getActiveBufferVariables( std::vector<std::string> const & names )
    {
      std::vector<GLuint> indices;
      indices.reserve(names.size());
      for ( size_t idx = 0; idx < names.size(); ++idx )
      {
        GLuint locationIndex = glGetProgramResourceIndex( getGLId(), GL_BUFFER_VARIABLE, names[idx].c_str() );
        if ( locationIndex != GL_INVALID_INDEX )
        {
          indices.push_back(GLuint(locationIndex));
        }
      }
      return getActiveBufferVariables( indices );
    }

    Program::Uniforms Program::getActiveUniforms( std::vector<GLuint> const & locationIndices )
    {
      Program::Uniforms uniforms;

      for ( size_t idx = 0; idx < locationIndices.size(); ++idx )
      {
        GLuint index = locationIndices[idx];

        GLenum const properties[] = {
            GL_NAME_LENGTH  // 0
          , GL_BLOCK_INDEX  // 1
          , GL_LOCATION     // 2
          , GL_TYPE         // 3
          , GL_ARRAY_SIZE   // 4
          , GL_OFFSET       // 5
          , GL_ARRAY_STRIDE // 6
          , GL_MATRIX_STRIDE// 7
          , GL_IS_ROW_MAJOR // 8
        };

        GLsizei const numProperties = sizeof dp::util::array(properties);
        GLint values[numProperties];

        glGetProgramResourceiv( getGLId(), GL_UNIFORM, index, numProperties, properties, numProperties, NULL, values );

        Uniform uniform;
        
        // get uniform name
        uniform.name.resize(values[0]);
        glGetProgramResourceName( getGLId(), GL_UNIFORM, index, GLsizei(uniform.name.size()), NULL, &uniform.name[0]);

        uniform.blockIndex = values[1];
        uniform.uniformLocation = values[2];
        uniform.type = values[3];
        uniform.arraySize = values[4];
        uniform.offset = values[5];
        uniform.arrayStride = values[6];
        uniform.matrixStride = values[7];
        uniform.isRowMajor = (values[8] != 0);

        uniforms[uniform.name] = uniform;
      }
      return uniforms;
    }

    Program::Uniforms Program::getActiveBufferVariables( std::vector<GLuint> const & locationIndices )
    {
      Program::Uniforms uniforms;

      GLint numActiveVariables;
      glGetProgramInterfaceiv( getGLId(), GL_BUFFER_VARIABLE, GL_ACTIVE_RESOURCES, &numActiveVariables );

      for ( size_t idx = 0; idx < locationIndices.size(); ++idx )
      {
        GLuint index = locationIndices[idx];

        GLenum const properties[] = {
          GL_NAME_LENGTH  // 0
          , GL_BLOCK_INDEX  // 1
          , GL_TYPE         // 2
          , GL_ARRAY_SIZE   // 3
          , GL_OFFSET       // 4
          , GL_ARRAY_STRIDE // 5
          , GL_MATRIX_STRIDE// 6
          , GL_IS_ROW_MAJOR // 7
        };

        GLsizei const numProperties = sizeof dp::util::array(properties);
        GLint values[numProperties];

        glGetProgramResourceiv( getGLId(), GL_BUFFER_VARIABLE, index, numProperties, properties, numProperties, NULL, values );

        Uniform uniform;

        // get uniform name
        uniform.name.resize(values[0]);
        glGetProgramResourceName( getGLId(), GL_BUFFER_VARIABLE, index, GLsizei(uniform.name.size()), NULL, &uniform.name[0]);

        uniform.blockIndex = values[1];
        uniform.uniformLocation = -1;
        uniform.type = values[2];
        uniform.arraySize = values[3];
        uniform.offset = values[4];
        uniform.arrayStride = values[5];
        uniform.matrixStride = values[6];
        uniform.isRowMajor = (values[7] != 0);

        uniforms[uniform.name] = uniform;
      }
      return uniforms;
    }

    ProgramUseGuard::ProgramUseGuard( SmartProgram const& program, bool doBinding )
      : m_program( program )
      , m_binding( doBinding )
    {
      DP_ASSERT( m_program );
      glUseProgram( m_program->getGLId() );

      if ( m_binding )
      {
        GLuint unit = 0;
        for ( std::vector<Program::ImageData>::const_iterator it = m_program->beginImageUnits() ; it != m_program->endImageUnits() ; ++it, ++unit )
        {
          DP_ASSERT( it->texture );
          glBindImageTexture( unit, it->texture->getGLId(), 0, GL_FALSE, 0, it->access, it->texture->getInternalFormat() );
        }
      }
    }

    ProgramUseGuard::~ProgramUseGuard()
    {
      if ( m_binding )
      {
        GLuint unit = 0;
        for ( std::vector<Program::ImageData>::const_iterator it = m_program->beginImageUnits() ; it != m_program->endImageUnits() ; ++it, ++unit )
        {
          DP_ASSERT( it->texture );
          glBindImageTexture( unit, 0, 0, GL_FALSE, 0, it->access, it->texture->getInternalFormat() );
        }
      }

      glUseProgram( 0 );
    }

  } // namespace gl
} // namespace dp
