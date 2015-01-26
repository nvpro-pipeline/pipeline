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


#pragma once

#include <dp/gl/Config.h>
#include <dp/gl/Object.h>
#include <dp/gl/Shader.h>
#include <dp/gl/Texture.h>
#include <dp/math/Matmnt.h>
#include <GL/glew.h>

namespace dp
{
  namespace gl
  {
    class ProgramUseGuard;

    class Program : public Object
    {
      public:
        struct Parameters
        {
          Parameters( bool binaryRetrievableHint = false, bool separable = false)
            : m_binaryRetrievableHint(binaryRetrievableHint)
            , m_separable(separable)
          {}

          bool  m_binaryRetrievableHint;
          bool  m_separable;
        };

        struct Uniform
        {
          GLint   blockIndex;
          GLint   location;
          GLenum  type;
          GLint   arraySize;
          GLint   offset;
          GLint   arrayStride;
          GLint   matrixStride;
          bool    isRowMajor;
        };
        typedef std::vector<Uniform> Uniforms;

        struct UniformBlock
        {
          GLint                 bufferBinding;
          dp::gl::BufferSharedPtr  buffer;
        };
        typedef std::vector<UniformBlock> UniformBlocks;

      public:
        DP_GL_API static ProgramSharedPtr create( std::vector<ShaderSharedPtr> const& shaders, Parameters const& parameters = Parameters() );
        DP_GL_API static ProgramSharedPtr create( VertexShaderSharedPtr const& vertexShader, FragmentShaderSharedPtr const& fragmentShader, Parameters const& parameters = Parameters() );
        DP_GL_API static ProgramSharedPtr create( ComputeShaderSharedPtr const& computeShader, Parameters const& programParameters = Parameters() );
        DP_GL_API ~Program();

        DP_GL_API unsigned int getActiveAttributesCount() const;
        DP_GL_API unsigned int getActiveAttributesMask() const;
        DP_GL_API GLint getAttributeLocation( std::string const& name ) const;
        DP_GL_API std::pair<GLenum,std::vector<char>> getBinary() const;
        DP_GL_API std::vector<ShaderSharedPtr> const& getShaders() const;
        DP_GL_API void setImageTexture( std::string const& textureName, TextureSharedPtr const& texture, GLenum access );

        template <typename T> void setUniform( std::string const& name, T const& value );
        template <typename T> void setUniform( size_t uniformIndex, T const& value );

        DP_GL_API Uniforms const& getActiveUniforms() const;
        DP_GL_API Uniform const& getActiveUniform( size_t index ) const;
        DP_GL_API size_t getActiveUniformIndex( std::string const& uniformName ) const;

        DP_GL_API Uniforms const& getActiveBufferVariables() const;
        DP_GL_API Uniform const& getActiveBufferVariable( size_t index ) const;
        DP_GL_API size_t getActiveBufferVariableIndex( std::string const& bufferVariableName ) const;

        DP_GL_API UniformBlocks const& getActiveUniformBlocks() const;
        DP_GL_API UniformBlock const& getActiveUniformBlock( size_t index ) const;
        DP_GL_API size_t getActiveUniformBlockIndex( std::string const& uniformName ) const;

      protected:
        DP_GL_API Program( std::vector<ShaderSharedPtr> const& shaders, Parameters const & parameter );

      private:
        struct ImageData
        {
          GLuint            index;                        // index into m_uniforms
          GLenum            access;
          TextureSharedPtr  texture;
        };

      private:
        friend class ProgramUseGuard;

        DP_GL_API std::vector<ImageData>::const_iterator beginImageUnits() const;
        DP_GL_API std::vector<ImageData>::const_iterator endImageUnits() const;

      private:
        unsigned int                  m_activeAttributesCount;
        unsigned int                  m_activeAttributesMask;
        std::vector<ImageData>        m_imageUniforms;
        std::vector<GLuint>           m_samplerUniforms;  // indices into m_uniforms
        std::vector<ShaderSharedPtr>  m_shaders;
        Uniforms                      m_uniforms;
        std::map<std::string,size_t>  m_uniformsMap;
        Uniforms                      m_bufferVariables;
        std::map<std::string,size_t>  m_bufferVariablesMap;
        UniformBlocks                 m_uniformBlocks;
        std::map<std::string,size_t>  m_uniformBlocksMap;
    };

    class ProgramUseGuard
    {
      public:
        DP_GL_API ProgramUseGuard( ProgramSharedPtr const& program, bool doBinding = true );
        DP_GL_API ~ProgramUseGuard();

      private:
        ProgramSharedPtr  m_program;
        bool              m_binding;
    };

    template <typename T>
    inline void Program::setUniform( std::string const& name, T const& value )
    {
      size_t uniformIndex = getActiveUniformIndex( name );
      DP_ASSERT( uniformIndex != -1 );
      setUniform( uniformIndex, value );
    }

    template <typename T>
    inline void Program::setUniform( size_t uniformIndex, T const& value )
    {
      Uniform const& uniform = getActiveUniform( uniformIndex );
      DP_ASSERT( ( TypeTraits<T>::glType() == uniform.type ) || ( ( TypeTraits<T>::glType() == GL_INT ) && ( isSamplerType( uniform.type ) || isImageType( uniform.type ) ) ) );
      DP_ASSERT( uniform.arraySize == 1 );
      if ( uniform.location != -1 )
      {
        DP_ASSERT( uniform.blockIndex == -1 );
        setProgramUniform( getGLId(), uniform.location, value );
      }
      else
      {
        DP_ASSERT( uniform.blockIndex != -1 );
        DP_ASSERT( uniform.blockIndex < getActiveUniformBlocks().size() );
        UniformBlock const& uniformBlock = getActiveUniformBlock( uniform.blockIndex );
        glBindBufferBase( GL_UNIFORM_BUFFER, uniform.blockIndex, uniformBlock.buffer->getGLId() );
        setBufferData( getActiveUniformBlock( uniform.blockIndex ).buffer->getGLId(), uniform.offset, uniform.matrixStride, value );
      }
    }

    template <typename T>
    inline void setProgramUniform( GLint program, GLint location, T const& value )
    {
#if defined(_MSVC_VER)
      DP_STATIC_ASSERT( !"setProgramUniform: missing specialization for type T!" );
#endif
    }

    template <>
    inline void setProgramUniform( GLint program, GLint location, float const& value )
    {
      glProgramUniform1f( program, location, value );
    }

    template <>
    inline void setProgramUniform( GLint program, GLint location, int const& value )
    {
      glProgramUniform1i( program, location, value );
    }

    template <>
    inline void setProgramUniform( GLint program, GLint location, dp::math::Vec4f const& value )
    {
      glProgramUniform4fv( program, location, 1, reinterpret_cast<GLfloat const*>(&value) );
    }

    template <>
    inline void setProgramUniform( GLint program, GLint location, dp::math::Mat33f const& value )
    {
      glProgramUniformMatrix3fv( program, location, 1, GL_FALSE, reinterpret_cast<GLfloat const*>(&value) );
    }

    template <>
    inline void setProgramUniform( GLint program, GLint location, dp::math::Mat44f const& value )
    {
      glProgramUniformMatrix4fv( program, location, 1, GL_FALSE, reinterpret_cast<GLfloat const*>(&value) );
    }

    template <typename T>
    inline void setBufferData( GLint buffer, GLint offset, GLint matrixStride, T const& value )
    {
#if defined(_MSVC_VER)
      DP_STATIC_ASSERT( !"setProgramUniformBuffer: missing specialization for type T!" );
#endif
    }

    template <>
    inline void setBufferData( GLint buffer, GLint offset, GLint /*matrixStride*/, float const& value )
    {
#if defined(GL_VERSION_4_5)
      glNamedBufferSubData( buffer, offset, sizeof(float), &value );
#else
      glNamedBufferSubDataEXT( buffer, offset, sizeof(float), &value );
#endif
    }

    template <>
    inline void setBufferData( GLint buffer, GLint offset, GLint /*matrixStride*/, int const& value )
    {
#if defined(GL_VERSION_4_5)
      glNamedBufferSubData( buffer, offset, sizeof(int), &value );
#else
      glNamedBufferSubDataEXT( buffer, offset, sizeof(int), &value );
#endif
    }

    template <>
    inline void setBufferData( GLint buffer, GLint offset, GLint /*matrixStride*/, unsigned int const& value )
    {
#if defined(GL_VERSION_4_5)
      glNamedBufferSubData( buffer, offset, sizeof(unsigned int), &value );
#else
      glNamedBufferSubDataEXT( buffer, offset, sizeof(unsigned int), &value );
#endif
    }

    template <>
    inline void setBufferData( GLint buffer, GLint offset, GLint /*matrixStride*/, dp::math::Vec4f const& value )
    {
#if defined(GL_VERSION_4_5)
      glNamedBufferSubData( buffer, offset, sizeof(dp::math::Vec4f), &value );
#else
      glNamedBufferSubDataEXT( buffer, offset, sizeof(dp::math::Vec4f), &value );
#endif
    }

    template <>
    inline void setBufferData( GLint buffer, GLint offset, GLint matrixStride, dp::math::Mat33f const& value )
    {
      DP_ASSERT( matrixStride == 4 * sizeof(float) );
#if defined(GL_VERSION_4_5)
      glNamedBufferSubData( buffer, offset, 3*sizeof(float), &value[0] );
      glNamedBufferSubData( buffer, offset + matrixStride, 3*sizeof(float), &value[1] );
      glNamedBufferSubData( buffer, offset + 2*matrixStride, 3*sizeof(float), &value[2] );
#else
      glNamedBufferSubDataEXT( buffer, offset, 3*sizeof(float), &value[0] );
      glNamedBufferSubDataEXT( buffer, offset + matrixStride, 3*sizeof(float), &value[1] );
      glNamedBufferSubDataEXT( buffer, offset + 2*matrixStride, 3*sizeof(float), &value[2] );
#endif
    }

    template <>
    inline void setBufferData( GLint buffer, GLint offset, GLint matrixStride, dp::math::Mat44f const& value )
    {
      DP_ASSERT( matrixStride == 4 * sizeof(float) );
#if defined(GL_VERSION_4_5)
      glNamedBufferSubData( buffer, offset, sizeof(dp::math::Mat44f), &value );
#else
      glNamedBufferSubDataEXT( buffer, offset, sizeof(dp::math::Mat44f), &value );
#endif
    }

  } // namespace gl
} // namespace dp