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


#include <dp/fx/EffectLibrary.h>
#include <dp/fx/ParameterGroupLayout.h>
#include <dp/fx/glsl/UniformGeneratorGLSL.h>
#include <dp/fx/glsl/UniformGeneratorGLSLStandard.h>
#include <sstream>

namespace dp
{
  namespace fx
  {
    namespace glsl
    {

      std::string UniformGeneratorGLSLStandard::generateUniforms( const dp::fx::ParameterGroupSpecSharedPtr& spec )
      {
        std::ostringstream oss;
        oss << "// ParameterGroup: " << spec->getName() << "\n";
        for ( ParameterGroupSpec::iterator it = spec->beginParameterSpecs(); it != spec->endParameterSpecs(); ++it )
        {
          oss << "uniform " << getGLSLTypename( it->first ) << " " << it->first.getName();
          if ( it->first.getArraySize() )
          {
            oss << "[" << it->first.getArraySize() << "]";
          }
          oss << ";\n";
        }
        oss << "\n";
        return( oss.str() );
      }

      dp::fx::ParameterGroupLayoutSharedPtr UniformGeneratorGLSLStandard::getParameterGroupLayout( const dp::fx::ParameterGroupSpecSharedPtr& spec )
      {
        std::vector<dp::fx::ParameterGroupLayout::ParameterInfoSharedPtr> parameterInfos;
        return( dp::fx::ParameterGroupLayout::create( dp::fx::MANAGER_UNIFORM, parameterInfos, "", 0, false, spec ) );
      }


    } // namespace glsl
  } // namespace fx
} // namespace dpfx
