// Copyright (c) 2012-2016, NVIDIA CORPORATION. All rights reserved.
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
#include <dp/fx/glsl/UniformGeneratorUBOStd140.h>
#include <dp/fx/glsl/ParameterInfo.h>
#include <dp/fx/EffectLibrary.h>
#include <sstream>

using namespace std;

namespace dp
{
  namespace fx
  {
    namespace glsl
    {


      UniformGeneratorUBOStd140::UniformGeneratorUBOStd140( bool generateLayout )
        : m_generateLayout( generateLayout )
      {
      }

      std::string UniformGeneratorUBOStd140::generateUniforms(const dp::fx::ParameterGroupSpecSharedPtr& spec)
      {
        // TODO who decides if SBL is available for the group?
        if ( !bufferAllowed(spec) )
        {
          UniformGeneratorGLSLStandard generatorStandard;
          return generatorStandard.generateUniforms( spec );
        }
        else
        {
          std::string specName = stripNameSpaces(spec->getName());
          std::string uboName = "ubo_" + specName;
          std::string uniformName = "uniform_" + specName;

          std::string stringRegion = "// ParameterGroup: " + specName + "\n";
          std::ostringstream stringUBO;
          std::string stringDefines;

          stringUBO << "layout(std140) uniform " << uboName << " {\n";
          for ( ParameterGroupSpec::iterator it = spec->beginParameterSpecs(); it != spec->endParameterSpecs(); ++it )
          {
            const std::string& name = it->first.getName();

            stringUBO << "  " << getGLSLTypename( it->first ) << " " << name;
            if ( it->first.getArraySize() )
            {
              stringUBO << string("[") << it->first.getArraySize() << "]";
            }
            stringUBO << ";\n";
          }
          stringUBO << "\n};\n";

          return stringRegion + stringUBO.str() + stringRegion + "\n";
        }
      }

      /************************************************************************/
      /* Compute Layout in Buffer for SBL                                     */
      /************************************************************************/
      dp::fx::ParameterGroupLayoutSharedPtr UniformGeneratorUBOStd140::getParameterGroupLayout( const dp::fx::ParameterGroupSpecSharedPtr& spec )
      {
        std::vector<ParameterGroupLayout::ParameterInfoSharedPtr> parameterInfos;

        if ( !bufferAllowed(spec) )
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
            ParameterGroupLayout::ParameterInfoSharedPtr pi = dp::fx::glsl::createParameterInfoUBOStd140( parameterSpec.getType(), offset, parameterSpec.getArraySize() );
            parameterInfos.push_back(pi);
          }
          std::string groupName = std::string("ubo_") + spec->getName();
          if ( m_generateLayout )
          {
            return ParameterGroupLayout::create( dp::fx::Manager::UNIFORM_BUFFER_OBJECT_RIX_FX, parameterInfos, groupName, offset, false, spec );
          }
          return( dp::fx::ParameterGroupLayout::create( dp::fx::Manager::UNIFORM, parameterInfos, "", 0, false, spec ) );
        }

      }

    } // namespace glsl
  } // namespace fx
} // namespace dp
