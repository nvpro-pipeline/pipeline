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


#include <GroupDataBuffered.h>

namespace dp
{
  namespace rix
  {
    namespace fx
    {

      GroupDataBuffered::GroupDataBuffered( ManagerUniform* manager, const SmartParameterGroupSpecInfoHandle& groupSpecInfo)
        : m_groupSpecInfo( groupSpecInfo )
        , m_shadowBuffer( new char[groupSpecInfo->m_groupLayout->getBufferSize()] )
        , m_dirty( false )
      {
        m_renderer = manager->getRenderer();
        m_buffer = m_renderer->bufferCreate();
        m_container = m_renderer->containerCreate( m_groupSpecInfo->m_descriptor );

        // initialize buffer with required size
        m_renderer->bufferSetSize( m_buffer, m_groupSpecInfo->m_groupLayout->getBufferSize() );

        // pass buffer over to container
        dp::rix::core::ContainerEntry entry = m_renderer->containerDescriptorGetEntry( groupSpecInfo->m_descriptor, groupSpecInfo->m_groupLayout->getGroupName().c_str() );
        m_renderer->containerSetData( m_container, entry, dp::rix::core::ContainerDataBuffer( m_buffer ) );
      }

      bool GroupDataBuffered::setValue( const dp::fx::ParameterGroupSpec::iterator& parameter, const dp::rix::core::ContainerData& data )
      {
        const dp::rix::core::ContainerDataRaw& containerDataRaw = reinterpret_cast<const dp::rix::core::ContainerDataRaw&>(data);
        const dp::fx::ParameterGroupLayout::ParameterInfoSharedPtr& info = m_groupSpecInfo->m_groupLayout->getParameterInfo(parameter);
        info->convert( m_shadowBuffer.get(), containerDataRaw.m_data );
        if ( !m_dirty )
        {
          m_dirty = true;
          return true;
        }
        return false;
      }

      void GroupDataBuffered::update( )
      {
        if ( m_dirty )
        {
          m_renderer->bufferUpdateData( m_buffer, 0, m_shadowBuffer.get(), m_groupSpecInfo->m_groupLayout->getBufferSize() );
          m_dirty = false;
        }
      }

    } // namespace fx
  } // namespace rix
} // namespace dp
