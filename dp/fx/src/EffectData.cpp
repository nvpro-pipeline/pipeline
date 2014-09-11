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


#include <dp/fx/EffectData.h>

namespace dp
{
  namespace fx
  {

    EffectData::EffectData( const SmartEffectSpec& effectSpec, const std::string& name )
      : m_effectSpec( effectSpec)
      , m_name( name )
      , m_transparent( effectSpec->getTransparent() )
    {
      DP_ASSERT( m_effectSpec ); // actually should be an exception
      m_parameterGroupDatas.reset( new SmartParameterGroupData[ effectSpec->getNumberOfParameterGroupSpecs() ] );
    }

    const SmartParameterGroupData& EffectData::getParameterGroupData( EffectSpec::iterator it ) const
    {
      DP_ASSERT( it != m_effectSpec->endParameterGroupSpecs() );
      return m_parameterGroupDatas[ std::distance( m_effectSpec->beginParameterGroupSpecs(), it ) ];
    }

    const std::string& EffectData::getName() const
    {
      return m_name;
    }

    const SmartEffectSpec& EffectData::getEffectSpec( ) const
    {
      return m_effectSpec;
    }

    bool EffectData::getTransparent() const
    {
      return m_transparent;
    }

    void EffectData::setTransparent( bool transparent )
    {
      m_transparent = transparent;
    }

    bool EffectData::operator==( const EffectData& rhs ) const
    {
      if ( this == &rhs )
      {
        return true;
      }
      else
      {
        DP_ASSERT( m_effectSpec );
        DP_ASSERT( m_parameterGroupDatas );

        bool equal = m_effectSpec == rhs.m_effectSpec && m_name == rhs.m_name;
        for ( size_t index = 0; equal && index < m_effectSpec->getNumberOfParameterGroupSpecs(); ++index )
        {
          equal = m_parameterGroupDatas[index] == rhs.m_parameterGroupDatas[index];
        }
        return equal;
      }
    }

  } // namespace fx
} // namespace dp
