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
#include <boost/tokenizer.hpp>
#include <sstream>

using namespace dp::util;
using std::max;
using std::string;
using std::vector;

namespace dp
{
  namespace fx
  {
      /************************************************************************/
      /* Conversion from string to value                                      */
      /************************************************************************/
      template<typename T>
      T convert( std::istringstream& inputStream  )
      {
        T value;
        inputStream >> value;
        return value;
      }

      template<>
      bool convert<bool>( std::istringstream& inputStream )
      {
        std::string token;
        inputStream >> token;

        DP_ASSERT( _stricmp(token.c_str(), "true") == 0 || _stricmp(token.c_str(), "false") == 0 || token == "0" || token == "1" );
        return _stricmp( token.c_str(), "true" ) == 0 || token == "1";
      }

      template<typename T>
      void convert( unsigned int count, std::istringstream& inputStream, T * value )
      {
        DP_ASSERT( count );
        for ( unsigned int i=0 ; i<count ; i++ )
        {
          DP_ASSERT( !inputStream.eof() && !inputStream.bad() );
          value[i] = convert<T>( inputStream );
        }
      }

      void getValueFromString( unsigned int type, unsigned int arraySize, const std::string & valueString, void * value )
      {
        DP_ASSERT( !valueString.empty() );
        std::istringstream inputStream( valueString );
        if ( type & ( PT_SCALAR_TYPE_MASK | PT_SCALAR_MODIFIER_MASK ) )
        {
          unsigned int count = std::max( (unsigned int)1, arraySize ) * getParameterCount( type );
          unsigned int size = count * getParameterSize( type );
          switch( type & PT_SCALAR_TYPE_MASK )
          {
          case PT_BOOL:
            convert( count, inputStream, static_cast<dp::Bool*>(value) );
            break;
          case PT_ENUM:
            DP_ASSERT( false );    // enums are handled by a separate function!
            break;
          case PT_INT8:
            convert( count, inputStream, static_cast<dp::Int8*>(value) );
            break;
          case PT_UINT8:
            convert( count, inputStream, static_cast<dp::Uint8*>(value) );
            break;
          case PT_INT16:
            convert( count, inputStream, static_cast<dp::Int16*>(value) );
            break;
          case PT_UINT16:
            convert( count, inputStream, static_cast<dp::Uint16*>(value) );
            break;
          case PT_FLOAT32:
            convert( count, inputStream, static_cast<dp::Float32*>(value) );
            break;
          case PT_INT32:
            convert( count, inputStream, static_cast<dp::Int32*>(value) );
            break;
          case PT_UINT32:
            convert( count, inputStream, static_cast<dp::Uint32*>(value) );
            break;
          case PT_FLOAT64:
            convert( count, inputStream, static_cast<dp::Float64*>(value) );
            break;
          case PT_INT64:
            convert( count, inputStream, static_cast<dp::Int64*>(value) );
            break;
          case PT_UINT64:
            convert( count, inputStream, static_cast<dp::Uint64*>(value) );
            break;
          }
        }
      }

      void getValueFromString( const EnumSpecSharedPtr & enumSpec, unsigned int arraySize, const std::string & valueString, EnumSpec::StorageType * valueArray )
      {
        DP_ASSERT( !valueString.empty() );

        unsigned int count = std::max( (unsigned int)1, arraySize );

        boost::tokenizer<boost::char_separator<char>> tokenizer( valueString, boost::char_separator<char>( " " ) );
        boost::tokenizer<boost::char_separator<char>>::const_iterator it = tokenizer.begin();
        for ( unsigned int i=0 ; i<count && it != tokenizer.end() ; ++it, i++ )
        {
          valueArray[i] = enumSpec->getValue( *it );
        }
      }

      /************************************************************************/
      /* Conversion from value to string                                      */
      /************************************************************************/
      template<typename T>
      void convert( T const value, std::ostringstream & outputStream  )
      {
        outputStream << value;
      }

      template<>
      void convert<bool>( bool const value, std::ostringstream & outputStream )
      {
        outputStream << (value ? "true" : "false");
      }

      template<typename T>
      void convert( unsigned int count, T* value, std::ostringstream & outputStream  )
      {
        DP_ASSERT( count );
        convert<T>( value[0], outputStream );
        for ( unsigned int index = 1 ; index < count ; index++ )
        {
          outputStream << " ";
          convert<T>( value[index], outputStream );
        }
      }


      std::string getStringFromValue( unsigned int type, unsigned int arraySize, void const * value )
      {
        unsigned int count = std::max( (unsigned int)1, arraySize ) * getParameterCount( type );
        unsigned int size = count * getParameterSize( type );

        std::ostringstream outputStream;
        if ( type & ( PT_SCALAR_TYPE_MASK | PT_SCALAR_MODIFIER_MASK ) )
        {
          switch( type & PT_SCALAR_TYPE_MASK )
          {
          case PT_BOOL:
            convert( count, reinterpret_cast<dp::Bool const*>(value), outputStream );
            break;
          case PT_ENUM:
            DP_ASSERT( false );    // enums are handled by a separate function!
            break;
          case PT_INT8:
            convert( count, reinterpret_cast<dp::Int8 const*>(value), outputStream );
            break;
          case PT_UINT8:
            convert( count, reinterpret_cast<dp::Uint8 const*>(value), outputStream );
            break;
          case PT_INT16:
            convert( count, reinterpret_cast<dp::Int16 const*>(value), outputStream );
            break;
          case PT_UINT16:
            convert( count, reinterpret_cast<dp::Uint16 const*>(value), outputStream );
            break;
          case PT_FLOAT32:
            convert( count, reinterpret_cast<dp::Float32 const*>(value), outputStream );
            break;
          case PT_INT32:
            convert( count, reinterpret_cast<dp::Int32 const*>(value), outputStream );
            break;
          case PT_UINT32:
            convert( count, reinterpret_cast<dp::Uint32 const*>(value), outputStream );
            break;
          case PT_FLOAT64:
            convert( count, reinterpret_cast<dp::Float64 const*>(value), outputStream );
            break;
          case PT_INT64:
            convert( count, reinterpret_cast<dp::Int64 const*>(value), outputStream );
            break;
          case PT_UINT64:
            convert( count, reinterpret_cast<dp::Uint64 const*>(value), outputStream );
            break;
          }
        }
        else if ( type & PT_POINTER_TYPE_MASK )
        {
          switch( type & PT_POINTER_TYPE_MASK )
          {
          case PT_BUFFER_PTR:
            DP_ASSERT( !"getStringFromValue: unhandled pointer type PT_BUFFER_PTR encountered!" );
            break;
            break;
          case PT_SAMPLER_PTR:
            DP_ASSERT( type & PT_SAMPLER_TYPE_MASK );
            outputStream << reinterpret_cast<char const*>(value);
            break;
          case PT_TEXTURE_PTR:
            DP_ASSERT( !"getStringFromValue: unhandled pointer type PT_TEXTURE_PTR encounterd!" );
            break;
          default:
            DP_ASSERT( !"getStringFromValue: unknown pointer type encountered!" );
            break;
          }
        }
        return outputStream.str();
      }

      std::string getStringFromValue( const EnumSpecSharedPtr & enumSpec, unsigned int arraySize,  EnumSpec::StorageType const * valueArray )
      {
        std::ostringstream outputStream;
        unsigned int count = std::max( (unsigned int)1, arraySize );
        
        outputStream << enumSpec->getValueName( valueArray[0] );
        for ( size_t index = 1; index < count; ++index )
        {
          outputStream << " ";
          outputStream << enumSpec->getValueName( valueArray[index] );
        }
        return outputStream.str();
      }

  } // namespace fx
} // namespace cp
