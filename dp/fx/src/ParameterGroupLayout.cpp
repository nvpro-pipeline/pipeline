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


#include <dp/fx/ParameterGroupLayout.h>

namespace dp
{
  namespace fx
  {
    SmartParameterGroupLayout ParameterGroupLayout::create( dp::fx::Manager manager, const std::vector<SmartParameterInfo>& parameterInfos, const std::string& groupName, size_t bufferSize, bool isInstanced, const SmartParameterGroupSpec& spec )
    {
      return( std::shared_ptr<ParameterGroupLayout>( new ParameterGroupLayout( manager, parameterInfos, groupName, bufferSize, isInstanced, spec ) ) );
    }

    ParameterGroupLayout::ParameterGroupLayout( dp::fx::Manager manager, const std::vector<SmartParameterInfo>& parameterInfos, const std::string& groupName, size_t bufferSize, bool isInstanced, const SmartParameterGroupSpec& spec )
      : m_manager( manager )
      , m_parameterInfos( parameterInfos )
      , m_groupName( groupName )
      , m_spec( spec )
      , m_bufferSize( bufferSize )
      , m_isInstanced( isInstanced )
    {
      DP_ASSERT( spec );
    }

    bool ParameterGroupLayout::isInstanced() const
    {
      return m_isInstanced;
    }

    dp::fx::Manager ParameterGroupLayout::getManager() const
    {
      return m_manager;
    }

    const std::string& ParameterGroupLayout::getGroupName() const
    {
      return m_groupName;
    }

    size_t ParameterGroupLayout::getBufferSize() const
    {
      return m_bufferSize;
    }


  } // namespace fx
} // namespace dp

