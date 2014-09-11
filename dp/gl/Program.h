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

namespace dp
{
  namespace gl
  {
    class Program;
    typedef dp::util::SmartPtr<Program> SmartProgram;

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
          std::string name;
          GLint       blockIndex;
          GLint       uniformLocation;
          GLenum      type;
          GLint       arraySize;
          GLint       offset;
          GLint       arrayStride;
          GLint       matrixStride;
          bool        isRowMajor;
        };

        typedef std::map<std::string, Uniform> Uniforms;

      public:
        DP_GL_API static SmartProgram create( std::vector<SmartShader> const& shaders, Parameters const& parameters = Parameters() );
        DP_GL_API static SmartProgram create( SmartVertexShader const& vertexShader, SmartFragmentShader const& fragmentShader, Parameters const& parameters = Parameters() );
        DP_GL_API static SmartProgram create( SmartComputeShader const& computeShader, Parameters const& programParameters = Parameters() );

        DP_GL_API unsigned int getActiveAttributesCount() const;
        DP_GL_API unsigned int getActiveAttributesMask() const;
        DP_GL_API std::pair<GLenum,std::vector<char>> getBinary() const;
        DP_GL_API std::vector<SmartShader> const& getShaders() const;
        DP_GL_API GLint getUniformLocation( std::string const& name ) const;
        DP_GL_API GLenum getUniformType( GLint location ) const;
        DP_GL_API void setImageTexture( std::string const& textureName, SmartTexture const& texture, GLenum access );

        DP_GL_API Uniforms getActiveUniforms();
        DP_GL_API Uniforms getActiveUniforms( std::vector<std::string> const& uniformNames );

        DP_GL_API Uniforms getActiveBufferVariables();
        DP_GL_API Uniforms getActiveBufferVariables( std::vector<std::string> const& names );

      protected:
        DP_GL_API Program( std::vector<SmartShader> const& shaders, Parameters const & parameter );
        DP_GL_API ~Program();

        /** \brief Query all uniforms for the given location indices. Note that the location index is the resource index
                   of the OpenGL 4.3 interface query interface. For more information have a look at
                   http://www.opengl.org/wiki/Program_Introspection#Interface_query
        **/
        DP_GL_API Uniforms getActiveUniforms( std::vector<GLuint > const& locationIndices );
        DP_GL_API Uniforms getActiveBufferVariables( std::vector<GLuint > const& locationIndices );

      private:
        struct ImageData
        {
          GLuint        index;                        // index into m_uniforms
          GLenum        access;
          SmartTexture  texture;
        };

        struct UniformData
        {
          std::string name;
          GLint       size;
          GLenum      type;
          GLint       location;
        };

      private:
        friend class ProgramUseGuard;

        DP_GL_API std::vector<ImageData>::const_iterator beginImageUnits() const;
        DP_GL_API std::vector<ImageData>::const_iterator endImageUnits() const;

      private:
        unsigned int              m_activeAttributesCount;
        unsigned int              m_activeAttributesMask;
        std::vector<ImageData>    m_imageUniforms;
        std::vector<GLuint>       m_samplerUniforms;  // indices into m_uniforms
        std::vector<SmartShader>  m_shaders;
        std::vector<UniformData>  m_uniforms;
    };

    class ProgramUseGuard
    {
      public:
        DP_GL_API ProgramUseGuard( SmartProgram const& program, bool doBinding = true );
        DP_GL_API ~ProgramUseGuard();

      private:
        SmartProgram  m_program;
        bool          m_binding;
    };

  } // namespace gl
} // namespace dp