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


#include <dp/fx/inc/ParameterGroupDataPrivate.h>
#include <algorithm>

namespace dp
{
  namespace fx
  {

    ParameterGroupDataPrivateSharedPtr ParameterGroupDataPrivate::create( ParameterGroupSpecSharedPtr const& parameterGroupSpec, std::string const& name )
    {
      return( std::shared_ptr<ParameterGroupDataPrivate>( new ParameterGroupDataPrivate( parameterGroupSpec, name ) ) );
    }

    ParameterGroupDataPrivate::ParameterGroupDataPrivate( const ParameterGroupSpecSharedPtr& parameterGroupSpec, const std::string& name )
      : ParameterGroupData( parameterGroupSpec, name )
    {
      // fill the data area with the defaults from the spec
      for ( ParameterGroupSpec::iterator it = getParameterGroupSpec()->beginParameterSpecs() ; it != getParameterGroupSpec()->endParameterSpecs() ; ++it )
      {
        if ( it->first.getDefaultValue() )
        {
          setParameter( it, it->first.getDefaultValue() );
        }
      }
    }

    ParameterGroupDataPrivate::~ParameterGroupDataPrivate( )
    {
      for ( ParameterGroupSpec::iterator it = getParameterGroupSpec()->beginParameterSpecs() ; it != getParameterGroupSpec()->endParameterSpecs() ; ++it )
      {
        if ( isParameterPointer( it->first ) )
        {
          size_t arraySize = std::max( 1u, it->first.getArraySize() );
          for ( size_t index = 0; index < arraySize; ++index )
          {
            updateString( it, index, nullptr );
          }
        }
      }
    }

    void ParameterGroupDataPrivate::updateString( dp::fx::ParameterGroupSpec::iterator it, size_t index, const char* value )
    {
      char*& destination = *reinterpret_cast<char**>(&m_data[it->second+index*sizeof(char*)]);
      delete[] destination;
      
      if ( value )
      {
        size_t valueLength = strlen( value );
        destination = new char[valueLength + 1];
        strcpy( destination, value );
      }
      else
      {
        destination = nullptr;
      }
    }

    void ParameterGroupDataPrivate::updateValue( dp::fx::ParameterGroupSpec::iterator it, size_t index, const void* value )
    {
      unsigned int size = it->first.getElementSizeInBytes();
      if ( value )
      {
        memcpy( &m_data[it->second] + index * size, value, size );
      }
      else
      {
        memset( &m_data[it->second] + index * size, 0, size );
      }
    }

    void ParameterGroupDataPrivate::setParameter( dp::fx::ParameterGroupSpec::iterator it, const void * value )
    {
      DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );

      if ( isParameterPointer( it->first ) )
      {
        DP_ASSERT( it->first.getArraySize() == 0 );
        updateString( it, 0, reinterpret_cast<const char*>(value) );
      }
      else
      {
        unsigned int elementSize = it->first.getElementSizeInBytes();
        const char * charValue = reinterpret_cast<const char *>(value);
        unsigned int count = std::max( dp::util::checked_cast<unsigned int>(it->first.getArraySize()), (unsigned int)(1) );
        for ( unsigned int i=0 ; i<count ; i++ )
        {
          updateValue( it, i, charValue );
          charValue += elementSize;
        }
      }
    }

    void *ParameterGroupDataPrivate::getValuePointer( dp::fx::ParameterGroupSpec::iterator it )
    {
      DP_ASSERT( it != m_parameterGroupSpec->endParameterSpecs() );
      return &m_data[it->second];
    }

  } // namespace fx
} // namespace dp

