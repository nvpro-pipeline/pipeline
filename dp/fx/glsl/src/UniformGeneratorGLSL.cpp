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
#include <dp/fx/ParameterSpec.h>
#include <dp/fx/ParameterGroupSpec.h>
#include <map>
#include <sstream>

namespace dp
{
  namespace fx
  {
    namespace glsl
    {

      typedef std::map< unsigned int, std::string > GLSLTypeMap;

      GLSLTypeMap initTypeMap()
      {
        GLSLTypeMap typeMap;

        typeMap.insert(std::make_pair(static_cast<unsigned int>(PT_FLOAT32), "float"));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_VECTOR2   ,"vec2" ) );
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_VECTOR3   ,"vec3"  ));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_VECTOR4   ,"vec4"  ));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_INT8) ,"int"   ));
        typeMap.insert(std::make_pair( PT_INT8 | PT_VECTOR2     ,"ivec2" ));
        typeMap.insert(std::make_pair( PT_INT8 | PT_VECTOR3     ,"ivec3" ));
        typeMap.insert(std::make_pair( PT_INT8 | PT_VECTOR4     ,"ivec4" ));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_INT16) ,"int"   ));
        typeMap.insert(std::make_pair( PT_INT16 | PT_VECTOR2     ,"ivec2" ));
        typeMap.insert(std::make_pair( PT_INT16 | PT_VECTOR3     ,"ivec3" ));
        typeMap.insert(std::make_pair( PT_INT16 | PT_VECTOR4     ,"ivec4" ));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_INT32) ,"int"   ));
        typeMap.insert(std::make_pair( PT_INT32 | PT_VECTOR2     ,"ivec2" ));
        typeMap.insert(std::make_pair( PT_INT32 | PT_VECTOR3     ,"ivec3" ));
        typeMap.insert(std::make_pair( PT_INT32 | PT_VECTOR4     ,"ivec4" ));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_INT64) ,"int64_t"   ));
        typeMap.insert(std::make_pair( PT_INT64 | PT_VECTOR2     ,"i64vec2" ));
        typeMap.insert(std::make_pair( PT_INT64 | PT_VECTOR3     ,"i64vec3" ));
        typeMap.insert(std::make_pair( PT_INT64 | PT_VECTOR4     ,"i64vec4" ));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_UINT8) ,"uint"  ));
        typeMap.insert(std::make_pair( PT_UINT8 | PT_VECTOR2    ,"uvec2" ));
        typeMap.insert(std::make_pair( PT_UINT8 | PT_VECTOR3    ,"uvec3" ));
        typeMap.insert(std::make_pair( PT_UINT8 | PT_VECTOR4    ,"uvec4" ));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_UINT16) ,"uint"  ));
        typeMap.insert(std::make_pair( PT_UINT16 | PT_VECTOR2    ,"uvec2" ));
        typeMap.insert(std::make_pair( PT_UINT16 | PT_VECTOR3    ,"uvec3" ));
        typeMap.insert(std::make_pair( PT_UINT16 | PT_VECTOR4    ,"uvec4" ));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_UINT32) ,"uint"  ));
        typeMap.insert(std::make_pair( PT_UINT32 | PT_VECTOR2    ,"uvec2" ));
        typeMap.insert(std::make_pair( PT_UINT32 | PT_VECTOR3    ,"uvec3" ));
        typeMap.insert(std::make_pair( PT_UINT32 | PT_VECTOR4    ,"uvec4" ));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_UINT64) ,"uint64_t"  ));
        typeMap.insert(std::make_pair( PT_UINT64 | PT_VECTOR2    ,"u64vec2" ));
        typeMap.insert(std::make_pair( PT_UINT64 | PT_VECTOR3    ,"u64vec3" ));
        typeMap.insert(std::make_pair( PT_UINT64 | PT_VECTOR4    ,"u64vec4" ));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_BOOL) ,"bool"  ));
        typeMap.insert(std::make_pair( PT_BOOL | PT_VECTOR2      ,"bvec2" ));
        typeMap.insert(std::make_pair( PT_BOOL | PT_VECTOR3      ,"bvec3" ));
        typeMap.insert(std::make_pair( PT_BOOL | PT_VECTOR4      ,"bvec4" ));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX2x2 ,"mat2"  ));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX2x2 ,"mat2x2"));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX2x3 ,"mat2x3"));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX2x4 ,"mat2x4"));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX3x2 ,"mat3x2"));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX3x3 ,"mat3"  ));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX3x3 ,"mat3x3"));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX3x4 ,"mat3x4"));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX4x2 ,"mat4x2"));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX4x3 ,"mat4x3"));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX4x4 ,"mat4"  ));
        typeMap.insert(std::make_pair( PT_FLOAT32 | PT_MATRIX4x4 ,"mat4x4"));
        typeMap.insert(std::make_pair( static_cast<unsigned int>(PT_FLOAT64) ,"double" ));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_VECTOR2   ,"dvec2"  ));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_VECTOR3   ,"dvec3"  ));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_VECTOR4   ,"dvec4"  ));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX2x2 ,"dmat2"  ));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX2x2 ,"dmat2x2"));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX2x3 ,"dmat2x3"));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX2x4 ,"dmat2x4"));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX3x2 ,"dmat3x2"));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX3x3 ,"dmat3"  ));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX3x3 ,"dmat3x3"));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX3x4 ,"dmat3x4"));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX4x2 ,"dmat4x2"));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX4x3 ,"dmat4x3"));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX4x4 ,"dmat4"  ));
        typeMap.insert(std::make_pair( PT_FLOAT64 | PT_MATRIX4x4 ,"dmat4x4"));
        // m_mapGLStoPT.insert(std::make_pair( PT_UNIFORM_BUFFER ,"buffer?")); // DAR TODO How is this defined?
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_1D                    ,"sampler1D"             ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_2D                    ,"sampler2D"             ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_3D                    ,"sampler3D"             ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_CUBE                  ,"samplerCube"           ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_2D_RECT               ,"sampler2DRect"         ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_1D_ARRAY              ,"sampler1DArray"        ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_2D_ARRAY              ,"sampler2DArray"        ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_BUFFER                ,"samplerBuffer"         ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_2D_MULTI_SAMPLE       ,"sampler2DMS"           ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_2D_MULTI_SAMPLE_ARRAY ,"sampler2DMSArray"      ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_CUBE_ARRAY            ,"samplerCubeArray"      )); // DAR Only exists in GLSL 4.00++
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_1D_SHADOW             ,"sampler1DShadow"       ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_2D_SHADOW             ,"sampler2DShadow"       ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_2D_RECT_SHADOW        ,"sampler2DRectShadow"   ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_1D_ARRAY_SHADOW       ,"sampler1DArrayShadow"  ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_2D_ARRAY_SHADOW       ,"sampler2DArrayShadow"  ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_CUBE_SHADOW           ,"samplerCubeShadow"     ));
        typeMap.insert(std::make_pair( PT_SAMPLER_PTR | PT_SAMPLER_CUBE_ARRAY_SHADOW     ,"samplerCubeArrayShadow"));
        return typeMap;
      }

      static GLSLTypeMap typemap = initTypeMap();

      std::string getGLSLTypename( const dp::fx::ParameterSpec& parameterSpec )
      {
        if ( parameterSpec.getType() == PT_ENUM )
        {
          return "int";
        }

        GLSLTypeMap::iterator itTypeName = typemap.find( parameterSpec.getType() );
        if ( itTypeName != typemap.end() )
        {
          return itTypeName->second;
        }

        DP_ASSERT( !"unsupported GLSL type" );
        return "";
      }

      std::string generateStruct( const dp::fx::ParameterGroupSpecSharedPtr& parameterGroupSpec, std::string& generatedStructName )
      {
        std::ostringstream stringStruct;
        generatedStructName = "pgs_" + parameterGroupSpec->getName();

        stringStruct << "struct " << generatedStructName << " {\n";
        for ( ParameterGroupSpec::iterator it = parameterGroupSpec->beginParameterSpecs(); it != parameterGroupSpec->endParameterSpecs(); ++it )
        {
          const std::string& name = it->first.getName();

          stringStruct << "  " << getGLSLTypename( it->first ) << " " << name;
          if ( it->first.getArraySize() )
          {
            stringStruct << std::string("[") << it->first.getArraySize() << "]";
          }
          stringStruct << ";\n";
        }
        stringStruct << "\n};\n";

        return stringStruct.str();
      }

      std::string generateParameterAccessors( const dp::fx::ParameterGroupSpecSharedPtr& parameterGroupSpec, const std::string& uniformName, const std::string& arrayIndex, const std::string& accessor )
      {
        std::ostringstream stringDefines;
        std::string arrayAccess = arrayIndex.empty() ? arrayIndex : std::string("[") + arrayIndex + std::string("]");
        for ( ParameterGroupSpec::iterator it = parameterGroupSpec->beginParameterSpecs(); it != parameterGroupSpec->endParameterSpecs(); ++it )
        {
          const std::string& name = it->first.getName();
          const std::string glslTypename = getGLSLTypename( it->first );

          if ( it->first.getArraySize() )
          {
            DP_ASSERT( !"arrays not yet tested!");
            stringDefines << glslTypename << " " << name << " = {" << std::endl;
            for ( size_t index = 0; index < it->first.getArraySize(); ++index )
            {
              if ( index )
              {
                stringDefines << ", ";
              }
              // uniform_struct[uniform_structId].variable[index]
              stringDefines << uniformName << arrayAccess << accessor << name << "[" << index << "]" << std::endl;
            }
            stringDefines << "};" <<std::endl;
          }
          else
          {
            // int variable = uniform_struct[uniform_structId].variable[index];
            stringDefines << glslTypename << " " << name << " = " << uniformName << arrayAccess << accessor << name << ";\n";
          }
        }
        return stringDefines.str();
      }

    } // namespace glsl
  } // namespace fx
} // namespace dp
