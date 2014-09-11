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


#include <dp/fx/glsl/UniformGeneratorGLSL.h>
#include <dp/fx/glsl/UniformGeneratorGLSLStandard.h>
#include <dp/fx/glsl/UniformGeneratorGLSLShaderBufferLoad.h>
#include <dp/fx/glsl/ParameterInfo.h>
#include <dp/fx/EffectLibrary.h>

using namespace std;

namespace dp
{
  namespace fx
  {
    namespace glsl
    {

      std::string UniformGeneratorGLSLShaderBufferLoad::generateUniforms( const dp::fx::SmartParameterGroupSpec& spec )
      {
        // TODO who decides if SBL is available for the group?
        bool blockShaderBufferLoad = false;
        for ( ParameterGroupSpec::iterator it = spec->beginParameterSpecs(); !blockShaderBufferLoad && it != spec->endParameterSpecs(); ++it )
        {
          unsigned int parameterType = it->first.getType();
          blockShaderBufferLoad |= !!(parameterType & PT_SAMPLER_TYPE_MASK);
          blockShaderBufferLoad |= !!(parameterType & PT_BUFFER_PTR);
        }

        if ( blockShaderBufferLoad )
        {
          UniformGeneratorGLSLStandard generatorStandard;
          return generatorStandard.generateUniforms( spec );
        }
        else
        {
          std::string structName;
          std::string uniformName = "uniform_" + spec->getName();

          std::string stringRegion = "// ParameterGroup: " + spec->getName() + "\n";
          std::string stringStruct = generateStruct(spec, structName);
          std::string stringDefines = generateParameterAccessors( spec, uniformName, "", "->" );
          std::string stringUniform = "uniform " + structName + "* " + uniformName + ";\n";

          return stringRegion + stringStruct + stringUniform + stringDefines + stringRegion + "\n";
        }
      }

      /************************************************************************/
      /* Compute Layout in Buffer for SBL                                     */
      /************************************************************************/
      dp::fx::SmartParameterGroupLayout UniformGeneratorGLSLShaderBufferLoad::getParameterGroupLayout( const dp::fx::SmartParameterGroupSpec& spec )
      {
        std::vector<ParameterGroupLayout::SmartParameterInfo> parameterInfos;

        bool blockBuffer = false;
        for ( ParameterGroupSpec::iterator it = spec->beginParameterSpecs(); !blockBuffer && it != spec->endParameterSpecs(); ++it )
        {
          unsigned int parameterType = it->first.getType();
          blockBuffer |= !!(parameterType & PT_SAMPLER_TYPE_MASK);
          blockBuffer |= !!(parameterType & PT_BUFFER_PTR);
        }

        if ( blockBuffer )
        {
          UniformGeneratorGLSLStandard ug;
          return ug.getParameterGroupLayout( spec );
        }
        else
        {
          size_t maxAlign = 4;
          size_t offset = 0;

          for ( ParameterGroupSpec::iterator it = spec->beginParameterSpecs(); it != spec->endParameterSpecs(); ++it )
          {
            const ParameterSpec& parameterSpec = it->first;
            ParameterGroupLayout::SmartParameterInfo pi = dp::fx::glsl::createParameterInfoShaderBufferLoad( parameterSpec.getType(), offset, parameterSpec.getArraySize() );
            parameterInfos.push_back(pi);
          }
          std::string groupName = std::string("uniform_") + spec->getName();
          return ParameterGroupLayout::create( dp::fx::MANAGER_SHADERBUFFER, parameterInfos, groupName, offset, false, spec );
        }

      }

    } // namespace glsl
  } // namespace fx
} // namespace dp
