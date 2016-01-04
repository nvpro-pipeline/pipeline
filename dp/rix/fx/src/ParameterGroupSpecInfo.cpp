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


#include <dp/rix/fx/Manager.h>
#include <dp/rix/fx/inc/BufferManagerIndexed.h>
#include <dp/rix/fx/inc/BufferManagerOffset.h>
#include <ParameterGroupSpecInfo.h>

using namespace dp::rix::core;

#define CHUNK_SIZE 256
namespace dp
{
  namespace rix
  {
    namespace fx
    {

      ParameterGroupSpecInfo::ParameterGroupSpecInfo( const dp::fx::ParameterGroupSpecSharedPtr& parameterGroupSpec, dp::fx::Manager manager, dp::rix::core::Renderer *renderer )
        : m_parameterGroupSpec( parameterGroupSpec )
      {
        // TODO get manager from somewhere else
        m_groupLayout = dp::fx::EffectLibrary::instance()->getParameterGroupLayout( m_parameterGroupSpec, manager );

        /************************************************************************/
        /* Create rix container for group                                       */
        /************************************************************************/
        std::vector<dp::rix::core::ProgramParameter> programParameters;
        switch ( m_groupLayout->getManager() )
        {
        case dp::fx::Manager::UNIFORM:
          {
            for ( dp::fx::ParameterGroupSpec::iterator itps = parameterGroupSpec->beginParameterSpecs();
                  itps != parameterGroupSpec->endParameterSpecs();
                  ++itps )
            {
              const dp::fx::ParameterSpec& spec = (*itps).first;
              programParameters.push_back( dp::rix::core::ProgramParameter( spec.getName().c_str(), getContainerParameterType( spec.getType() ), spec.getArraySize() ) );
            }

            if ( programParameters.size() )  // Means m_descriptor == null for effects without parameters.
            {
              m_descriptor = renderer->containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptorCommon( &programParameters[0], programParameters.size() ) );
            }

            /************************************************************************/
            /* Create parameter mapping table                                       */
            /************************************************************************/
            for ( dp::fx::ParameterGroupSpec::iterator it = parameterGroupSpec->beginParameterSpecs(); it != parameterGroupSpec->endParameterSpecs(); ++it )
            {
              ParameterMapping mapping;
              mapping.m_entry = renderer->containerDescriptorGetEntry( m_descriptor, it->first.getName().c_str() );

              m_mapping.push_back(mapping);
            }

          }
          break;
        case dp::fx::Manager::SHADERBUFFER:
        case dp::fx::Manager::UNIFORM_BUFFER_OBJECT_RIX_FX:
        case dp::fx::Manager::SHADER_STORAGE_BUFFER_OBJECT:
          {
            size_t alignment = 16;

            programParameters.push_back( dp::rix::core::ProgramParameter( m_groupLayout->getGroupName().c_str(), ContainerParameterType::BUFFER, 0 ) );
            m_descriptor = renderer->containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptorCommon( &programParameters[0], programParameters.size() ) );

            if ( m_groupLayout->isInstanced() )
            {
              dp::rix::core::ContainerEntry entry = renderer->containerDescriptorGetEntry( m_descriptor, m_groupLayout->getGroupName().c_str() );

              std::string groupId = m_groupLayout->getGroupName() + "Id";
              programParameters.clear();
              programParameters.push_back( dp::rix::core::ProgramParameter( groupId.c_str(), ContainerParameterType::INT_32, 0 ) );
              m_descriptorId = renderer->containerDescriptorCreate( dp::rix::core::ProgramParameterDescriptorCommon( &programParameters[0], programParameters.size() ) );
              dp::rix::core::ContainerEntry entryId = renderer->containerDescriptorGetEntry( m_descriptorId, groupId.c_str());

              m_bufferManager = BufferManagerIndexed::create( renderer, m_descriptor, entry, m_descriptorId, entryId, m_groupLayout->getBufferSize(), CHUNK_SIZE );
            }
            else
            {
              if ( manager == dp::fx::Manager::SHADERBUFFER )
              {
                dp::rix::core::ContainerEntry entry = renderer->containerDescriptorGetEntry( m_descriptor, m_groupLayout->getGroupName().c_str() );
                m_bufferManager = BufferManagerOffset::create( renderer, m_descriptor, entry, m_groupLayout->getBufferSize(), alignment, CHUNK_SIZE );
              }
              else if ( manager == dp::fx::Manager::UNIFORM_BUFFER_OBJECT_RIX_FX )
              {
#if 0
                // TODO no gl available here. where to get alignment?
                GLint glAlignment;
                glGetIntegerv( GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &glAlignment );
                alignment = dp::checked_cast<size_t>(glAlignment);
#else
                // TODO hard code 256 since this can be assumed on current hardware
                size_t alignment = 256;
#endif

                dp::rix::core::ContainerEntry entry = renderer->containerDescriptorGetEntry( m_descriptor, m_groupLayout->getGroupName().c_str() );
                m_bufferManager = BufferManagerOffset::create( renderer, m_descriptor, entry, m_groupLayout->getBufferSize(), alignment, CHUNK_SIZE );

              }
            }
          }
          break;
        default:
          DP_ASSERT( !"Unsupported Manager");
        }
      }

    } // namespace fx
  } // namespace rix
} // namespace dp
