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

#include <dp/fx/EffectLibrary.h>
#include <dp/fx/ParameterGroupSpec.h>
#include <dp/rix/core/RiX.h>
#include <BufferManager.h>
#include <vector>

namespace dp
{
  namespace rix
  {
    namespace fx
    {

      class ParameterGroupSpecInfo : public dp::rix::core::HandledObject
      {
      public:
        ParameterGroupSpecInfo( const dp::fx::ParameterGroupSpecSharedPtr& parameterGroupSpec, dp::fx::Manager manager, dp::rix::core::Renderer *renderer );

        struct ParameterMapping
        {
          dp::rix::core::ContainerEntry m_entry;
        };

        dp::fx::ParameterGroupSpecSharedPtr             m_parameterGroupSpec;
        dp::rix::core::ContainerDescriptorSharedHandle  m_descriptor; // rix descriptor for group
        dp::rix::core::ContainerDescriptorSharedHandle  m_descriptorId; // rix descriptor for group id

        // for uniform
        std::vector<ParameterMapping>     m_mapping; //! Mapping between n-th entry of spec and rix container

        // for buffers
        SmartBufferManager                    m_bufferManager;
        dp::fx::ParameterGroupLayoutSharedPtr m_groupLayout;
      };

      typedef dp::rix::core::HandleTrait<ParameterGroupSpecInfo>::Type ParameterGroupSpecInfoHandle;
      typedef dp::rix::core::SmartHandle<ParameterGroupSpecInfo>       SmartParameterGroupSpecInfoHandle;

    } // namespace fx
  } // namespace rix
} // namespace dp
