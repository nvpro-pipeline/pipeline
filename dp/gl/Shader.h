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

namespace dp
{
  namespace gl
  {
    DP_GL_API std::string shaderTypeToName( GLenum type );


    class Shader : public Object
    {
      public:
        DP_GL_API static ShaderSharedPtr create( GLenum type, std::string const& source );
        DP_GL_API virtual ~Shader();

      public:
        DP_GL_API std::string getSource() const;
        DP_GL_API virtual GLenum getType() const = 0;

      protected:
        DP_GL_API Shader( GLenum type, std::string const& source );
    };


    class VertexShader : public Shader
    {
      public:
        DP_GL_API static VertexShaderSharedPtr create( std::string const& source );

      public:
        DP_GL_API virtual GLenum getType() const;

      protected:
        DP_GL_API VertexShader( std::string const& source );
    };


    class TessControlShader : public Shader
    {
      public:
        DP_GL_API static TessControlShaderSharedPtr create( std::string const& source );

      public:
        DP_GL_API virtual GLenum getType() const;

      protected:
        DP_GL_API TessControlShader( std::string const& source );
    };


    class TessEvaluationShader : public Shader
    {
      public:
        DP_GL_API static TessEvaluationShaderSharedPtr create( std::string const& source );

      public:
        DP_GL_API virtual GLenum getType() const;

      protected:
        DP_GL_API TessEvaluationShader( std::string const& source );
    };


    class GeometryShader : public Shader
    {
      public:
        DP_GL_API static GeometryShaderSharedPtr create( std::string const& source );

      public:
        DP_GL_API virtual GLenum getType() const;

      protected:
        DP_GL_API GeometryShader( std::string const& source );
    };


    class FragmentShader : public Shader
    {
      public:
        DP_GL_API static FragmentShaderSharedPtr create( std::string const& source );

      public:
        DP_GL_API virtual GLenum getType() const;

      protected:
        DP_GL_API FragmentShader( std::string const& source );
    };

    class ComputeShader : public Shader
    {
      public:
        DP_GL_API static ComputeShaderSharedPtr create( std::string const& source );

      public:
        DP_GL_API virtual GLenum getType() const;

      protected:
        DP_GL_API ComputeShader( std::string const& source );
    };

  } // namespace gl
} // namespace dp