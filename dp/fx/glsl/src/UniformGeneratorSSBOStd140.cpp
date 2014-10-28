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
#include <dp/fx/glsl/UniformGeneratorSSBOStd140.h>
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

      UniformGeneratorSSBOStd140::UniformGeneratorSSBOStd140( bool generateLayout )
          : m_generateLayout( generateLayout )
      {
      }

      std::string UniformGeneratorSSBOStd140::generateUniforms( const dp::fx::SmartParameterGroupSpec& spec )
      {
        // TODO who decides if SBL is available for the group?
        if ( !bufferAllowed(spec) )
        {
          UniformGeneratorGLSLStandard generatorStandard;
          return generatorStandard.generateUniforms( spec );
        }
        else
        {
          std::string stringRegion = "// ParameterGroup: " + spec->getName() + "\n";
          std::ostringstream stringSSBO;
          std::string bufferName = "buffer_" + spec->getName();

          // use arrays for layout version
          if ( m_generateLayout ) {

            std::string structName;
            std::string stringStruct = generateStruct(spec, structName);

            std::string uniformName = std::string("u_") + structName;
            std::string bufferIndex = bufferName + "Id";

            std::string stringDefines = generateParameterAccessors( spec, uniformName, bufferIndex, "." );

            stringSSBO << "layout(std140) buffer " << bufferName << " {\n";
            stringSSBO << "  " << structName << " " << uniformName << "[256];\n";
            stringSSBO << "};\n";
            stringSSBO << "uniform int " << bufferIndex << ";\n";

            return stringRegion + stringStruct + stringSSBO.str() + stringDefines + stringRegion + "\n";
          }
          else
          {
            // and no arrays when not using layout version
            stringSSBO << "layout(std140) buffer " << bufferName << " {\n";
            for ( ParameterGroupSpec::iterator it = spec->beginParameterSpecs(); it != spec->endParameterSpecs(); ++it )
            {
              const std::string& name = it->first.getName();

              stringSSBO << "  " << getGLSLTypename( it->first ) << " " << name;
              if ( it->first.getArraySize() )
              {
                stringSSBO << string("[") << it->first.getArraySize() << "]";
              }
              stringSSBO << ";\n";
            }
            stringSSBO << "\n};\n";
          }

          return stringRegion + stringSSBO.str() + stringRegion + "\n";
        }
      }

      /************************************************************************/
      /* Compute Layout in Buffer for SBL                                     */
      /************************************************************************/
      dp::fx::SmartParameterGroupLayout UniformGeneratorSSBOStd140::getParameterGroupLayout( const dp::fx::SmartParameterGroupSpec& spec )
      {
        std::vector<ParameterGroupLayout::SmartParameterInfo> parameterInfos;

        if ( !bufferAllowed(spec) )
        {
          UniformGeneratorGLSLStandard ug;
          return ug.getParameterGroupLayout( spec );
        }
        else
        {
          size_t offset = 0;

          for ( ParameterGroupSpec::iterator it = spec->beginParameterSpecs(); it != spec->endParameterSpecs(); ++it )
          {
            const ParameterSpec& parameterSpec = it->first;
            // can use the same layout as ubostd140
            ParameterGroupLayout::SmartParameterInfo pi = dp::fx::glsl::createParameterInfoUBOStd140( parameterSpec.getType(), offset, parameterSpec.getArraySize() );
            parameterInfos.push_back(pi);
          }
          std::string groupName = std::string("buffer_") + spec->getName();

          if ( m_generateLayout )
          {
            return ParameterGroupLayout::create( dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT, parameterInfos, groupName, offset, false, spec );
          }
          return( dp::fx::ParameterGroupLayout::create( dp::fx::MANAGER_UNIFORM, parameterInfos, "", 0, false, spec ) );
        }

      }

    } // namespace glsl
  } // namespace fx
} // namespace dp
