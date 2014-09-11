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


#include <GroupDataBufferedCombined.h>
#include <dp/rix/fx/inc/BufferManagerIndexed.h>
#include <dp/rix/fx/inc/BufferManagerOffset.h>

namespace dp
{
  namespace rix
  {
    namespace fx
    {

      GroupDataBufferedCombined::GroupDataBufferedCombined( ManagerUniform* manager, const SmartParameterGroupSpecInfoHandle& groupSpecInfo)
        : m_groupSpecInfo( groupSpecInfo )
      {
        m_renderer = manager->getRenderer();
        m_allocation = groupSpecInfo->m_bufferManager->allocate();
      }

      bool GroupDataBufferedCombined::setValue( const dp::fx::ParameterGroupSpec::iterator& parameter, const dp::rix::core::ContainerData& data )
      {
        const dp::rix::core::ContainerDataRaw& containerDataRaw = reinterpret_cast<const dp::rix::core::ContainerDataRaw&>(data);
        const dp::fx::ParameterGroupLayout::SmartParameterInfo& info = m_groupSpecInfo->m_groupLayout->getParameterInfo(parameter);
        info->convert( m_groupSpecInfo->m_bufferManager->allocationGetPointer( m_allocation.get() ), containerDataRaw.m_data );
        m_groupSpecInfo->m_bufferManager->allocationMarkDirty( m_allocation.get() );
        return false;
      }

      void GroupDataBufferedCombined::update( )
      {
        // dirty handling is managed by buffer manager
        DP_ASSERT( !"should never be called since");
      }

      void GroupDataBufferedCombined::useContainers( dp::rix::core::GeometryInstanceSharedHandle const & gi )
      {
        DP_ASSERT( dp::rix::core::handleIsTypeOf<BufferManagerImpl>( m_groupSpecInfo->m_bufferManager.get() ) );
        BufferManagerImpl* bufferManager = dp::rix::core::handleCast<BufferManagerImpl>(m_groupSpecInfo->m_bufferManager.get());
        bufferManager->useContainers( gi, m_allocation.get() );
      }

      void GroupDataBufferedCombined::useContainers( dp::rix::core::RenderGroupSharedHandle const & renderGroup )
      {
        DP_ASSERT( dp::rix::core::handleIsTypeOf<BufferManagerImpl>( m_groupSpecInfo->m_bufferManager.get() ) );
        BufferManagerImpl* bufferManager = dp::rix::core::handleCast<BufferManagerImpl>(m_groupSpecInfo->m_bufferManager.get());
        bufferManager->useContainers( renderGroup, m_allocation.get() );
      }

    } // namespace fx
  } // namespace rix
} // namespace dp
