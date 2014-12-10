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


#include <dp/fx/ParameterGroupData.h>

namespace dp
{
  namespace fx
  {
    ParameterGroupDataSharedPtr ParameterGroupData::create( ParameterGroupSpecSharedPtr const& parameterGroupSpec, std::string const& name )
    {
      return( std::shared_ptr<ParameterGroupData>( new ParameterGroupData( parameterGroupSpec, name ) ) );
    }

    ParameterGroupData::ParameterGroupData( const ParameterGroupSpecSharedPtr& parameterGroupSpec, const std::string& name )
      : m_parameterGroupSpec( parameterGroupSpec )
      , m_name( name )
    {
      DP_ASSERT( m_parameterGroupSpec );
      m_data.resize( parameterGroupSpec->getDataSize() );
      DP_ASSERT( ( reinterpret_cast<size_t>(m_data.data()) & 0x7 ) == 0 );
    }

    ParameterGroupData::~ParameterGroupData()
    {
      // decrease refcount of textures and buffers
      for ( ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
      {
        if ( it->first.getType() & PT_POINTER_TYPE_MASK )
        {
          DP_ASSERT( it->second < m_data.size() );
          size_t numPtrs = it->first.getArraySize() ? it->first.getArraySize() : 1;
          DP_ASSERT( it->second + numPtrs * sizeof(char*) <= m_data.size() );
          for ( unsigned int i=0 ; i < numPtrs ; i++ )
          {
            char*& ptr = *reinterpret_cast<char**>(&m_data[it->second+i*sizeof(char*)]);
            delete[] ptr;
          }
        }
      }
    }

    bool ParameterGroupData::operator ==( const ParameterGroupData& rhs) const
    {
      if ( this == &rhs )
      {
        return true;
      }
      else
      {
        DP_ASSERT( m_parameterGroupSpec );

        bool equal = m_name == rhs.m_name;
        for ( ParameterGroupSpec::iterator it = m_parameterGroupSpec->beginParameterSpecs() ; equal && it != m_parameterGroupSpec->endParameterSpecs() ; ++it )
        {
          const void *pLhs = getParameter( it );
          const void *pRhs = rhs.getParameter( it );
          if ( isParameterPointer( it->first) )
          {
            if ( pLhs != pRhs )
            {
              // both pointers are non-zero. Do a string compare
              if ( pLhs != nullptr && pRhs != nullptr )
              {
                equal = strcmp( reinterpret_cast<const char *>(pLhs), reinterpret_cast<const char*>(pRhs) ) == 0;
              }
              // either one of the pointers is zero while the other is not. different.
              else
              {
                equal = false;
              }
            }
          }
          else
          {
            equal = ( memcmp( pLhs, pRhs, it->first.getSizeInByte() ) == 0 );
          }
        }
        return( equal );
      }

    }

    const dp::fx::ParameterGroupSpecSharedPtr& ParameterGroupData::getParameterGroupSpec() const
    {
      return m_parameterGroupSpec;
    }

    const std::string& ParameterGroupData::getName() const
    {
      return m_name;
    }

    const void * ParameterGroupData::getParameter( dp::fx::ParameterGroupSpec::iterator it ) const
    {
      DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
      //DP_ASSERT( ( it->first.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == it->first.getType() );
      DP_ASSERT( it->second + it->first.getSizeInByte() <= m_data.size() );
      if ( isParameterPointer( it->first ) )
      {
        // return a pointer to a constant pointer of char
        return( *reinterpret_cast<char *const *>(&m_data[it->second]) );
      }
      else
      {
        return( &m_data[it->second] );
      }
    }

  } // namespace fx
} // namespace dp

