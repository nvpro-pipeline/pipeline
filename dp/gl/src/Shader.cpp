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


#include <dp/gl/Shader.h>

namespace dp
{
  namespace gl
  {
    std::string shaderTypeToName( GLenum type )
    {
      switch( type )
      {
        case GL_COMPUTE_SHADER :          return( "ComputeShader" );
        case GL_VERTEX_SHADER :           return( "VertexShader" );
        case GL_TESS_CONTROL_SHADER :     return( "TessControlShader" );
        case GL_TESS_EVALUATION_SHADER :  return( "TessEvaluationShader" );
        case GL_GEOMETRY_SHADER :         return( "GeometryShader" );
        case GL_FRAGMENT_SHADER :         return( "FragmentShader" );
        default :
          DP_ASSERT( false );
          return( "" );
      }
    }

    ShaderSharedPtr Shader::create( GLenum type, std::string const& source )
    {
      switch( type )
      {
        case GL_COMPUTE_SHADER :          return( ComputeShader::create( source ) );
        case GL_VERTEX_SHADER :           return( VertexShader::create( source ) );
        case GL_TESS_CONTROL_SHADER :     return( TessControlShader::create( source ) );
        case GL_TESS_EVALUATION_SHADER :  return( TessEvaluationShader::create( source ) );
        case GL_GEOMETRY_SHADER :         return( GeometryShader::create( source ) );
        case GL_FRAGMENT_SHADER :         return( FragmentShader::create( source ) );
        default :
          DP_ASSERT( false );
          return( ShaderSharedPtr::null );
      }
    }

    Shader::Shader( GLenum type, std::string const& source )
    {
      GLuint id = glCreateShader( type );
      setGLId( id );

      GLint length = dp::util::checked_cast<GLint>(source.length());
      GLchar const* src = source.c_str();
      glShaderSource( id, 1, &src, &length );
      glCompileShader( id );

#if !defined( NDEBUG )
      GLint result;
      glGetShaderiv( id, GL_COMPILE_STATUS, &result );
      if ( ! result )
      {
        GLint errorLen;
        glGetShaderiv( id, GL_INFO_LOG_LENGTH, &errorLen );

        std::string buffer;
        buffer.resize( errorLen, 0 );
        glGetShaderInfoLog( id, errorLen, NULL, &buffer[0] );
        DP_ASSERT( false );
      }
#endif
    }

    Shader::~Shader( )
    {
      if ( getGLId() )
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

              virtual void execute() { glDeleteShader( m_id ); }

            protected:
              CleanupTask( GLuint id ) : m_id( id ) {}

            private:
              GLuint m_id;
          };

          // make destructor exception safe
          try
          {
            getShareGroup()->executeTask( CleanupTask::create( getGLId() ) );
          } catch (...) {}
        }
        else
        {
          glDeleteShader( getGLId() );
        }
      }
    }

    std::string Shader::getSource() const
    {
      GLint sourceLength;
      glGetShaderiv( getGLId(), GL_SHADER_SOURCE_LENGTH, &sourceLength );

      std::vector<char> source( sourceLength );
      glGetShaderSource( getGLId(), sourceLength, nullptr, source.data() );
      return( source.data() );
    }


    VertexShaderSharedPtr VertexShader::create( std::string const& source )
    {
      return( std::shared_ptr<VertexShader>( new VertexShader( source ) ) );
    }

    VertexShader::VertexShader( std::string const& source )
      : Shader( GL_VERTEX_SHADER, source )
    {
    }

    GLenum VertexShader::getType() const
    {
      return( GL_VERTEX_SHADER );
    }


    TessControlShaderSharedPtr TessControlShader::create( std::string const& source )
    {
      return( std::shared_ptr<TessControlShader>( new TessControlShader( source ) ) );
    }

    TessControlShader::TessControlShader( std::string const& source )
      : Shader( GL_TESS_CONTROL_SHADER, source )
    {
    }

    GLenum TessControlShader::getType() const
    {
      return( GL_TESS_CONTROL_SHADER );
    }


    TessEvaluationShaderSharedPtr TessEvaluationShader::create( std::string const& source )
    {
      return( std::shared_ptr<TessEvaluationShader>( new TessEvaluationShader( source ) ) );
    }

    TessEvaluationShader::TessEvaluationShader( std::string const& source )
      : Shader( GL_TESS_EVALUATION_SHADER, source )
    {
    }

    GLenum TessEvaluationShader::getType() const
    {
      return( GL_TESS_EVALUATION_SHADER );
    }


    GeometryShaderSharedPtr GeometryShader::create( std::string const& source )
    {
      return( std::shared_ptr<GeometryShader>( new GeometryShader( source ) ) );
    }

    GeometryShader::GeometryShader( std::string const& source )
      : Shader( GL_GEOMETRY_SHADER, source )
    {
    }

    GLenum GeometryShader::getType() const
    {
      return( GL_GEOMETRY_SHADER );
    }


    FragmentShaderSharedPtr FragmentShader::create( std::string const& source )
    {
      return( std::shared_ptr<FragmentShader>( new FragmentShader( source ) ) );
    }

    FragmentShader::FragmentShader( std::string const& source )
      : Shader( GL_FRAGMENT_SHADER, source )
    {
    }

    GLenum FragmentShader::getType() const
    {
      return( GL_FRAGMENT_SHADER );
    }


    ComputeShaderSharedPtr ComputeShader::create( std::string const& source )
    {
      return( std::shared_ptr<ComputeShader>( new ComputeShader( source ) ) );
    }

    ComputeShader::ComputeShader( std::string const& source )
      : Shader( GL_COMPUTE_SHADER, source )
    {
    }

    GLenum ComputeShader::getType() const
    {
      return( GL_COMPUTE_SHADER );
    }

  } // namespace gl
} // namespace dp
