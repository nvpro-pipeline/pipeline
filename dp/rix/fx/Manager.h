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


#pragma once

#include <dp/rix/fx/Config.h>
#include <dp/rix/core/RiX.h>
#include <dp/fx/EffectLibrary.h>

namespace dp
{
  namespace rix
  {
    namespace fx
    {

      DEFINE_RIX_HANDLE(GroupData);
      DEFINE_RIX_HANDLE(Instance);
      DEFINE_RIX_HANDLE(Program);

      DEFINE_PTR_TYPES( Manager );

      typedef std::map<dp::fx::Domain, std::vector<std::string> > SourceFragments;

      class Manager
      {
      public:
        struct EffectSpecInfo
        {
          EffectSpecInfo() {};

          EffectSpecInfo( dp::fx::EffectSpecSharedPtr const & effectSpec, bool isGlobal )
            : m_effectSpec( effectSpec )
            , m_isGlobal( isGlobal)
          {
          }

          dp::fx::EffectSpecSharedPtr m_effectSpec;
          bool                        m_isGlobal;
        };

        typedef std::map<std::string, EffectSpecInfo> SystemSpecs;

      public:
        RIX_FX_API static ManagerSharedPtr     create(dp::fx::Manager managerType, dp::rix::core::Renderer* rdr, unsigned int uniformBufferOffsetAlign=256);

        RIX_FX_API virtual void                runPendingUpdates() = 0;

        RIX_FX_API virtual dp::rix::fx::ProgramSharedHandle programCreate( const dp::fx::EffectSpecSharedPtr& effectSpec
                                                                         , Manager::SystemSpecs const & systemSpecs
                                                                         , const char *technique
                                                                         , dp::rix::core::ContainerDescriptorHandle *userDescriptors
                                                                         , size_t numDescriptors
                                                                         , SourceFragments const& sourceFragments = SourceFragments() ) = 0;

        RIX_FX_API virtual std::map<dp::fx::Domain,std::string> getShaderSources( const dp::fx::EffectSpecSharedPtr & effectSpec
                                                                                , bool depthPass
                                                                                , dp::rix::fx::Manager::SystemSpecs const & systemSpecs
                                                                                , SourceFragments const& sourceFragments = SourceFragments() ) const = 0;

        RIX_FX_API virtual GroupDataSharedHandle groupDataCreate( dp::fx::ParameterGroupSpecSharedPtr const& group ) = 0;
        RIX_FX_API virtual void                  groupDataSetValue( GroupDataSharedHandle const& groupdata
                                                                  , const dp::fx::ParameterGroupSpec::iterator& parameter
                                                                  , const core::ContainerData& data ) = 0;

        RIX_FX_API virtual dp::rix::fx::InstanceHandle instanceCreate(core::GeometryInstanceHandle gi) = 0;
        RIX_FX_API virtual dp::rix::fx::InstanceHandle instanceCreate(core::RenderGroupHandle renderGroup) = 0;
        RIX_FX_API virtual void                        instanceSetProgram( dp::rix::fx::InstanceHandle instance
                                                                         , dp::rix::fx::ProgramHandle program ) = 0;
        RIX_FX_API virtual bool                        instanceUseGroupData( dp::rix::fx::InstanceHandle instanceHandle
                                                                           , dp::rix::fx::GroupDataHandle groupHandle) = 0;

        RIX_FX_API virtual dp::fx::Manager getManagerType() const = 0;

      protected:
        RIX_FX_API Manager();
      };
    
        
      // Utility function to convert parameter type dp::fx::PT_* back to dp::rix::core::ContainerParameterType CPT_*.
      inline dp::rix::core::ContainerParameterType getContainerParameterType( unsigned int parameterType )
      {
        switch ( parameterType )
        {
        case dp::fx::PT_BOOL                      : return dp::rix::core::CPT_BOOL;
        case dp::fx::PT_BOOL | dp::fx::PT_VECTOR2 : return dp::rix::core::CPT_BOOL2;
        case dp::fx::PT_BOOL | dp::fx::PT_VECTOR3 : return dp::rix::core::CPT_BOOL3;
        case dp::fx::PT_BOOL | dp::fx::PT_VECTOR4 : return dp::rix::core::CPT_BOOL4;

        case dp::fx::PT_ENUM                      : return dp::rix::core::CPT_INT_32;
        case dp::fx::PT_ENUM | dp::fx::PT_VECTOR2 : return dp::rix::core::CPT_INT2_32;
        case dp::fx::PT_ENUM | dp::fx::PT_VECTOR3 : return dp::rix::core::CPT_INT3_32;
        case dp::fx::PT_ENUM | dp::fx::PT_VECTOR4 : return dp::rix::core::CPT_INT4_32;

        case dp::fx::PT_INT8                      : return dp::rix::core::CPT_INT_8;
        case dp::fx::PT_INT8 | dp::fx::PT_VECTOR2 : return dp::rix::core::CPT_INT2_8;
        case dp::fx::PT_INT8 | dp::fx::PT_VECTOR3 : return dp::rix::core::CPT_INT3_8;
        case dp::fx::PT_INT8 | dp::fx::PT_VECTOR4 : return dp::rix::core::CPT_INT4_8;

        case dp::fx::PT_UINT8                      : return dp::rix::core::CPT_UINT_8;
        case dp::fx::PT_UINT8 | dp::fx::PT_VECTOR2 : return dp::rix::core::CPT_UINT2_8;
        case dp::fx::PT_UINT8 | dp::fx::PT_VECTOR3 : return dp::rix::core::CPT_UINT3_8;
        case dp::fx::PT_UINT8 | dp::fx::PT_VECTOR4 : return dp::rix::core::CPT_UINT4_8;

        case dp::fx::PT_INT16                      : return dp::rix::core::CPT_INT_16;
        case dp::fx::PT_INT16 | dp::fx::PT_VECTOR2 : return dp::rix::core::CPT_INT2_16;
        case dp::fx::PT_INT16 | dp::fx::PT_VECTOR3 : return dp::rix::core::CPT_INT3_16;
        case dp::fx::PT_INT16 | dp::fx::PT_VECTOR4 : return dp::rix::core::CPT_INT4_16;

        case dp::fx::PT_UINT16                      : return dp::rix::core::CPT_UINT_16;
        case dp::fx::PT_UINT16 | dp::fx::PT_VECTOR2 : return dp::rix::core::CPT_UINT2_16;
        case dp::fx::PT_UINT16 | dp::fx::PT_VECTOR3 : return dp::rix::core::CPT_UINT3_16;
        case dp::fx::PT_UINT16 | dp::fx::PT_VECTOR4 : return dp::rix::core::CPT_UINT4_16;

        case dp::fx::PT_INT32                      : return dp::rix::core::CPT_INT_32;
        case dp::fx::PT_INT32 | dp::fx::PT_VECTOR2 : return dp::rix::core::CPT_INT2_32;
        case dp::fx::PT_INT32 | dp::fx::PT_VECTOR3 : return dp::rix::core::CPT_INT3_32;
        case dp::fx::PT_INT32 | dp::fx::PT_VECTOR4 : return dp::rix::core::CPT_INT4_32;

        case dp::fx::PT_UINT32                      : return dp::rix::core::CPT_UINT_32;
        case dp::fx::PT_UINT32 | dp::fx::PT_VECTOR2 : return dp::rix::core::CPT_UINT2_32;
        case dp::fx::PT_UINT32 | dp::fx::PT_VECTOR3 : return dp::rix::core::CPT_UINT3_32;
        case dp::fx::PT_UINT32 | dp::fx::PT_VECTOR4 : return dp::rix::core::CPT_UINT4_32;

        case dp::fx::PT_FLOAT32                      : return dp::rix::core::CPT_FLOAT;
        case dp::fx::PT_FLOAT32 | dp::fx::PT_VECTOR2 : return dp::rix::core::CPT_FLOAT2;
        case dp::fx::PT_FLOAT32 | dp::fx::PT_VECTOR3 : return dp::rix::core::CPT_FLOAT3;
        case dp::fx::PT_FLOAT32 | dp::fx::PT_VECTOR4 : return dp::rix::core::CPT_FLOAT4;

        case dp::fx::PT_FLOAT32 | dp::fx::PT_MATRIX2x2 : return dp::rix::core::CPT_MAT2X2;
        case dp::fx::PT_FLOAT32 | dp::fx::PT_MATRIX2x3 : return dp::rix::core::CPT_MAT2X3;
        case dp::fx::PT_FLOAT32 | dp::fx::PT_MATRIX2x4 : return dp::rix::core::CPT_MAT2X4;

        case dp::fx::PT_FLOAT32 | dp::fx::PT_MATRIX3x2 : return dp::rix::core::CPT_MAT3X2;
        case dp::fx::PT_FLOAT32 | dp::fx::PT_MATRIX3x3 : return dp::rix::core::CPT_MAT3X3;
        case dp::fx::PT_FLOAT32 | dp::fx::PT_MATRIX3x4 : return dp::rix::core::CPT_MAT3X4;

        case dp::fx::PT_FLOAT32 | dp::fx::PT_MATRIX4x2 : return dp::rix::core::CPT_MAT4X2;
        case dp::fx::PT_FLOAT32 | dp::fx::PT_MATRIX4x3 : return dp::rix::core::CPT_MAT4X3;
        case dp::fx::PT_FLOAT32 | dp::fx::PT_MATRIX4x4 : return dp::rix::core::CPT_MAT4X4;

        // DAR FIXME Add doubles here once the ContainerParameterTypes exist.

        // DAR ParameterGroups don't support buffers and probably don't need to.
        // That's a renderer specific thing and in the hands of the developer.
        //case dp::fx::PT_BUFFER_PTR | dp::fx::PT_BUFFER_1D:
        //case dp::fx::PT_BUFFER_PTR | dp::fx::PT_BUFFER_2D:
        //case dp::fx::PT_BUFFER_PTR | dp::fx::PT_BUFFER_3D:
        //  return dp::rix::core::CPT_BUFFER;

        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_1D:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_2D:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_3D:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_CUBE:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_2D_RECT:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_1D_ARRAY:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_2D_ARRAY:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_BUFFER:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_2D_MULTI_SAMPLE:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_2D_MULTI_SAMPLE_ARRAY:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_CUBE_ARRAY:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_1D_SHADOW:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_2D_SHADOW:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_2D_RECT_SHADOW:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_1D_ARRAY_SHADOW:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_2D_ARRAY_SHADOW:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_CUBE_SHADOW:
        case dp::fx::PT_SAMPLER_PTR | dp::fx::PT_SAMPLER_CUBE_ARRAY_SHADOW:
          return dp::rix::core::CPT_SAMPLER;

        default:
          DP_ASSERT(!"getContainerParameterType(): Implement this parameter type!");
          return dp::rix::core::CPT_NUM_PARAMETERTYPES; // Return something which will definitely fail later on.
        }
      }
        
    }
  }
}
