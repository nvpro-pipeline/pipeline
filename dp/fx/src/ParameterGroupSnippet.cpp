// Copyright NVIDIA Corporation 2012
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


#include <ParameterGroupSnippet.h>
#include <boost/scoped_ptr.hpp>

#include <dp/fx/glsl/UniformGenerator.h>
#include <dp/fx/glsl/UniformGeneratorGLSLStandard.h>
#include <dp/fx/glsl/UniformGeneratorGLSLShaderBufferLoad.h>
#include <dp/fx/glsl/UniformGeneratorUBOStd140.h>
#include <dp/fx/glsl/UniformGeneratorSSBOStd140.h>

namespace dp
{
  namespace fx
  {
    ParameterGroupSnippet::ParameterGroupSnippet( const ParameterGroupSpecSharedPtr& parameterGroupSpec )
      : m_parameterGroupSpec( parameterGroupSpec )
    {
      for ( ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs(); it != m_parameterGroupSpec->endParameterSpecs(); ++it )
      {
        if ( ( it->first.getType() & PT_SCALAR_TYPE_MASK ) == PT_ENUM )
        {
          addRequiredEnumSpec( it->first.getEnumSpec() );
        }
      }
    }

    std::string ParameterGroupSnippet::getSnippet( GeneratorConfiguration& configuration )
    {
      boost::scoped_ptr<dp::fx::UniformGenerator> ug;

      switch (configuration.manager)
      {
      case MANAGER_SHADERBUFFER:
        ug.reset( new dp::fx::glsl::UniformGeneratorGLSLShaderBufferLoad() );
        break;
      case MANAGER_UNIFORM:
        ug.reset( new dp::fx::glsl::UniformGeneratorGLSLStandard() );
        break;
      case MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX:
        ug.reset( new dp::fx::glsl::UniformGeneratorUBOStd140( true ) );
        break;
      case MANAGER_UNIFORM_BUFFER_OBJECT_RIX:
        ug.reset( new dp::fx::glsl::UniformGeneratorUBOStd140( false ) );
        break;
      case MANAGER_SHADER_STORAGE_BUFFER_OBJECT:
        ug.reset( new dp::fx::glsl::UniformGeneratorSSBOStd140(true) );
        break;
      case MANAGER_SHADER_STORAGE_BUFFER_OBJECT_RIX:
        ug.reset( new dp::fx::glsl::UniformGeneratorSSBOStd140(false) );
        break;
      default:
        DP_ASSERT(0 && "unsupported generator");
        return "";
      }

      return ug->generateUniforms( m_parameterGroupSpec );
    }

  } // namespace fx
} // namespace dp
