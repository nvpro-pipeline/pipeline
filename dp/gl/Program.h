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
#include <cstring>

namespace dp
{
  namespace gl
  {
    DP_GL_API size_t sizeOfType( GLenum type );

    template <typename T>
    void setProgramUniform( GLint program, GLint location, T const& value );

    template <typename T>
    void setBufferData( GLint buffer, GLint offset, GLint matrixStride, T const& value );

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

        struct Block
        {
          GLint       binding;
          GLint       dataSize;
          std::string name;
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
          GLint   unit;           // used for images and samplers
        };

      public:
        DP_GL_API static ProgramSharedPtr create( std::vector<ShaderSharedPtr> const& shaders, Parameters const& parameters = Parameters() );
        DP_GL_API static ProgramSharedPtr create( VertexShaderSharedPtr const& vertexShader, FragmentShaderSharedPtr const& fragmentShader, Parameters const& parameters = Parameters() );
        DP_GL_API static ProgramSharedPtr create( ComputeShaderSharedPtr const& computeShader, Parameters const& programParameters = Parameters() );
        DP_GL_API virtual ~Program();

        DP_GL_API unsigned int getActiveAttributesCount() const;
        DP_GL_API unsigned int getActiveAttributesMask() const;
        DP_GL_API GLint getAttributeLocation( std::string const& name ) const;
        DP_GL_API std::pair<GLenum,std::vector<char>> getBinary() const;
        DP_GL_API std::vector<ShaderSharedPtr> const& getShaders() const;

        DP_GL_API std::vector<Uniform> const& getActiveUniforms() const;
        DP_GL_API Uniform const& getActiveUniform( size_t index ) const;
        DP_GL_API size_t getActiveUniformIndex( std::string const& uniformName ) const;

        DP_GL_API std::vector<Uniform> const& getActiveBufferVariables() const;
        DP_GL_API Uniform const& getActiveBufferVariable( size_t index ) const;
        DP_GL_API size_t getActiveBufferVariableIndex( std::string const& bufferVariableName ) const;

        DP_GL_API std::vector<Block> const& getActiveUniformBlocks() const;
        DP_GL_API Block const& getActiveUniformBlock( size_t index ) const;
        DP_GL_API size_t getActiveUniformBlockIndex( std::string const& uniformName ) const;

        DP_GL_API std::vector<Block> const& getShaderStorageBlocks() const;
        DP_GL_API Block const& getShaderStorageBlock( size_t index ) const;
        DP_GL_API size_t getShaderStorageBlockIndex( std::string const& ssbName ) const;

      protected:
        DP_GL_API Program( std::vector<ShaderSharedPtr> const& shaders, Parameters const & parameter );

      private:
        unsigned int                  m_activeAttributesCount;
        unsigned int                  m_activeAttributesMask;
        std::vector<Uniform>          m_bufferVariables;
        std::map<std::string,size_t>  m_bufferVariablesMap;
        std::vector<ShaderSharedPtr>  m_shaders;
        std::vector<Block>            m_shaderStorageBlocks;
        std::map<std::string,size_t>  m_shaderStorageBlocksMap;
        std::vector<Uniform>          m_uniforms;
        std::map<std::string,size_t>  m_uniformsMap;
        std::vector<Block>            m_uniformBlocks;
        std::map<std::string,size_t>  m_uniformBlocksMap;
    };

  } // namespace gl
} // namespace dp