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


#include <dp/fx/ParameterConversion.h>
#include <dp/fx/ParameterSpec.h>
#include <dp/util/Tokenizer.h>
#include <dp/util/Types.h>

using namespace dp::util;
using std::max;
using std::string;
using std::vector;

namespace dp
{
  namespace fx
  {

    ParameterSpec::ParameterSpec( const string & name, unsigned int type, Semantic semantic, unsigned int arraySize, const string & defaultString, const string & annotation )
      : m_name(name)
      , m_type(type)
      , m_annotation(annotation)
      , m_semantic(semantic)
      , m_arraySize(arraySize)
      , m_defaultValue(nullptr)
      , m_sizeInBytes(std::max( 1u, m_arraySize ) * getParameterCount( m_type ) * getParameterSize( m_type ) )
    {
      DP_ASSERT( !!( type & PT_SCALAR_TYPE_MASK ) ^ !!( type & PT_POINTER_TYPE_MASK ) );
      DP_ASSERT( !( type & PT_POINTER_TYPE_MASK ) || !( type & PT_SCALAR_MODIFIER_MASK ) );
      DP_ASSERT( ( type & PT_SCALAR_TYPE_MASK ) != PT_ENUM );
      if ( !defaultString.empty() )
      {
        if ( type & PT_POINTER_TYPE_MASK )
        {
          DP_ASSERT( arraySize == 0 );
          size_t sib = defaultString.size() + 1;
          m_defaultValue = malloc( sib );
          memcpy( m_defaultValue, defaultString.c_str(), sib );
        }
        else
        {
          unsigned int sib = getSizeInByte();
          m_defaultValue = malloc( sib );
          getValueFromString( m_type, max(1u,m_arraySize), defaultString, m_defaultValue );
        }
      }
    }

    ParameterSpec::ParameterSpec( const std::string & name, unsigned int type, Semantic semantic, unsigned int arraySize, const void * defaultValue, const std::string & annotation )
      : m_name(name)
      , m_type(type)
      , m_annotation(annotation)
      , m_semantic(semantic)
      , m_arraySize(arraySize)
      , m_defaultValue(nullptr)
      , m_sizeInBytes(std::max( 1u, m_arraySize ) * getParameterCount( m_type ) * getParameterSize( m_type ) )
    {
      DP_ASSERT( !!( type & PT_SCALAR_TYPE_MASK ) ^ !!( type & PT_POINTER_TYPE_MASK ) );
      DP_ASSERT( !( type & PT_POINTER_TYPE_MASK ) || !( type & PT_SCALAR_MODIFIER_MASK ) );
      DP_ASSERT( !( type & PT_POINTER_TYPE_MASK ) || ( arraySize == 0 ) );
      DP_ASSERT( ( type & PT_SCALAR_TYPE_MASK ) != PT_ENUM );

      if ( defaultValue )
      {
        size_t sib = ( type & PT_POINTER_TYPE_MASK ) ? strlen( static_cast<const char *>( defaultValue ) ) + 1 : getSizeInByte();
        m_defaultValue = malloc( sib );
        memcpy( m_defaultValue, defaultValue, sib );
      }
    }

    ParameterSpec::ParameterSpec( const std::string & name, const EnumSpecSharedPtr & enumSpec, unsigned int arraySize, const string & defaultString, const std::string & annotation )
      : m_name(name)
      , m_type(PT_ENUM)
      , m_annotation(annotation)
      , m_semantic(SEMANTIC_VALUE)
      , m_enumSpec(enumSpec)
      , m_arraySize(arraySize)
      , m_defaultValue(nullptr)
      , m_sizeInBytes(std::max( 1u, m_arraySize ) * getParameterCount( m_type ) * getParameterSize( m_type ) )
    {
      DP_ASSERT( m_enumSpec );
      if ( !defaultString.empty() )
      {
        unsigned int sib = getSizeInByte();
        m_defaultValue = malloc( sib );
        getValueFromString( m_enumSpec, max(1u,m_arraySize), defaultString, (EnumSpec::StorageType *)m_defaultValue );
      }
    }

    ParameterSpec::ParameterSpec( const ParameterSpec & spec )
      : m_name( spec.m_name )
      , m_type( spec.m_type )
      , m_annotation( spec.m_annotation )
      , m_semantic( spec.m_semantic )
      , m_enumSpec( spec.m_enumSpec )
      , m_arraySize( spec.m_arraySize )
      , m_defaultValue( nullptr )
      , m_sizeInBytes(std::max( 1u, m_arraySize ) * getParameterCount( m_type ) * getParameterSize( m_type ) )
    {
      if ( spec.m_defaultValue )
      {
        size_t sib = ( m_type & PT_POINTER_TYPE_MASK ) ? strlen( static_cast<const char *>( spec.m_defaultValue ) ) + 1 : spec.getSizeInByte();
        m_defaultValue = malloc( sib );
        memcpy( m_defaultValue, spec.m_defaultValue, sib );
      }
    }

    ParameterSpec::~ParameterSpec()
    {
      free( m_defaultValue );
    }

  } // namespace fx
} // namespace cp
