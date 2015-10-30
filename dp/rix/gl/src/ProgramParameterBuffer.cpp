// Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/rix/gl/inc/ContainerGL.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>
#include <dp/rix/gl/inc/ProgramParameterBuffer.h>
#include <cstring>


using namespace dp::util;

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      /************************************************************************/
      /* ParameterntBuffer<n, T>                                              */
      /************************************************************************/
      template<unsigned int n, typename T>
      ParameterntBuffer<n, T>::ParameterntBuffer( unsigned int offsetContainerData, unsigned int offsetBuffer, unsigned int arraySize )
        : ParameterObject( offsetContainerData, sizeof( T ) * n * arraySize, static_cast<ConversionFunction>(&ParameterntBuffer<n, T>::doUpdateConverted) )
        , m_offsetBuffer( offsetBuffer )
        , m_arraySize( arraySize )
      {
      }

      template<unsigned int n, typename T>
      void ParameterntBuffer<n, T>::update(const void * /*data*/ )
      {
        DP_ASSERT( !"not supported");
      }

      template<unsigned int n, typename T>
      void ParameterntBuffer<n, T>::copy( const void* containerData, void* destination ) const
      {
        memcpy( reinterpret_cast<char*>(destination) + m_offsetBuffer, reinterpret_cast<const char*>(containerData) + m_offset, getConvertedSize() );
      }

      template<unsigned int n, typename T>
      void ParameterntBuffer<n, T>::doUpdateConverted( void const* /*convertedData*/ ) const
      {
        DP_ASSERT( !"not supported");
      }


      /************************************************************************/
      /* DataType conversion copy                                             */
      /************************************************************************/
      template <typename SourceType, typename DestinationType, int numComponents>
      struct UniformConversion
      {
        static void convert( void *destinationData, const void* containerData, unsigned int arraySize )
        {
          assert( arraySize == 1 );
          if ( arraySize == 1 )
          {
            const SourceType* source = reinterpret_cast<const SourceType*>(containerData);
            DestinationType* destination = reinterpret_cast<DestinationType*>(destinationData);
            for ( size_t component = 0; component < numComponents; ++component )
            {
              destination[component] = source[component];
            }
          }
        }
      };


      /************************************************************************/
      /* ParameterntBufferConversion<n,T,SourceType>                          */
      /************************************************************************/
      template<unsigned int n, typename T, typename SourceType>
      ParameterntBufferConversion<n, T, SourceType>::ParameterntBufferConversion( unsigned int offsetContainerData, unsigned int offsetBuffer, unsigned int arraySize )
        : ParameterObject( offsetContainerData, n * sizeof(T), static_cast<ConversionFunction>(&ParameterntBufferConversion<n,T,SourceType>::doUpdateConverted) )
        , m_offsetBuffer( offsetBuffer )
        , m_arraySize( arraySize )
      {
      }

      template<unsigned int n, typename T, typename SourceType>
      void ParameterntBufferConversion<n, T, SourceType>::update(const void * /*data*/)
      {
        DP_ASSERT( !"not supported");
      }

      template<unsigned int n, typename T, typename SourceType>
      void ParameterntBufferConversion<n, T, SourceType>::copy( const void* containerData, void* destination ) const
      {
        UniformConversion<SourceType, T, n>::convert(reinterpret_cast<char*>(destination) + m_offsetBuffer
                                                    , reinterpret_cast<const char*>(containerData) + m_offset, m_arraySize );
      }

      template<unsigned int n, typename T, typename SourceType>
      void ParameterntBufferConversion<n, T, SourceType>::doUpdateConverted( void const * /*convertedData*/ ) const
      {
        DP_ASSERT( !"not supported");
      }


      /************************************************************************/
      /* ParameternmtBuffer<n,m,T>                                                  */
      /************************************************************************/
      template<unsigned int n, unsigned int m, typename T>
      ParameternmtBuffer<n, m, T>::ParameternmtBuffer( unsigned int offsetContainerData, unsigned int offsetBuffer, unsigned int arraySize )
        : ParameterObject( offsetContainerData, arraySize * m * n * sizeof(T), static_cast<ConversionFunction>(&ParameternmtBuffer<n,m,T>::doUpdateConverted) )
        , m_offsetBuffer( offsetBuffer )
        , m_arraySize( arraySize )
      {
      }

      template<unsigned int n, unsigned int m, typename T>
      void ParameternmtBuffer<n, m, T>::update(const void * /*data*/)
      {
        DP_ASSERT( !"not supported");
      }

      template<unsigned int n, unsigned int m, typename T>
      void ParameternmtBuffer<n, m, T>::copy( const void* containerData, void* destination ) const
      {
        memcpy( reinterpret_cast<char*>(destination) + m_offsetBuffer, reinterpret_cast<const char*>(containerData) + m_offset, getConvertedSize() );
      }

      template<unsigned int n, unsigned int m, typename T>
      void ParameternmtBuffer<n, m, T>::doUpdateConverted( void const * /*convertedData*/ ) const
      {
        DP_ASSERT( !"not supported");
      }


      /************************************************************************/
      /* Instantiate templates                                                */
      /************************************************************************/
      template class ParameterntBuffer<1, float>;
      template class ParameterntBuffer<2, float>;
      template class ParameterntBuffer<3, float>;
      template class ParameterntBuffer<4, float>;

      template class ParameterntBuffer<1, double>;
      template class ParameterntBuffer<2, double>;
      template class ParameterntBuffer<3, double>;
      template class ParameterntBuffer<4, double>;

      template class ParameterntBufferConversion<1, int32_t, int8_t>;
      template class ParameterntBufferConversion<2, int32_t, int8_t>;
      template class ParameterntBufferConversion<3, int32_t, int8_t>;
      template class ParameterntBufferConversion<4, int32_t, int8_t>;

      template class ParameterntBufferConversion<1, int32_t, int16_t>;
      template class ParameterntBufferConversion<2, int32_t, int16_t>;
      template class ParameterntBufferConversion<3, int32_t, int16_t>;
      template class ParameterntBufferConversion<4, int32_t, int16_t>;

      template class ParameterntBuffer<1, int32_t>;
      template class ParameterntBuffer<2, int32_t>;
      template class ParameterntBuffer<3, int32_t>;
      template class ParameterntBuffer<4, int32_t>;

      template class ParameterntBuffer<1, int64_t>;
      template class ParameterntBuffer<2, int64_t>;
      template class ParameterntBuffer<3, int64_t>;
      template class ParameterntBuffer<4, int64_t>;

      template class ParameterntBufferConversion<1, uint32_t, uint8_t>;
      template class ParameterntBufferConversion<2, uint32_t, uint8_t>;
      template class ParameterntBufferConversion<3, uint32_t, uint8_t>;
      template class ParameterntBufferConversion<4, uint32_t, uint8_t>;

      template class ParameterntBufferConversion<1, uint32_t, uint16_t>;
      template class ParameterntBufferConversion<2, uint32_t, uint16_t>;
      template class ParameterntBufferConversion<3, uint32_t, uint16_t>;
      template class ParameterntBufferConversion<4, uint32_t, uint16_t>;

      template class ParameterntBuffer<1, uint32_t>;
      template class ParameterntBuffer<2, uint32_t>;
      template class ParameterntBuffer<3, uint32_t>;
      template class ParameterntBuffer<4, uint32_t>;

      template class ParameterntBuffer<1, uint64_t>;
      template class ParameterntBuffer<2, uint64_t>;
      template class ParameterntBuffer<3, uint64_t>;
      template class ParameterntBuffer<4, uint64_t>;

      template class ParameternmtBuffer<2, 2, float>;
      template class ParameternmtBuffer<2, 3, float>;
      template class ParameternmtBuffer<2, 4, float>;
      template class ParameternmtBuffer<3, 2, float>;
      template class ParameternmtBuffer<3, 3, float>;
      template class ParameternmtBuffer<3, 4, float>;
      template class ParameternmtBuffer<4, 2, float>;
      template class ParameternmtBuffer<4, 3, float>;
      template class ParameternmtBuffer<4, 4, float>;

      template class ParameternmtBuffer<2, 2, double>;
      template class ParameternmtBuffer<2, 3, double>;
      template class ParameternmtBuffer<2, 4, double>;
      template class ParameternmtBuffer<3, 2, double>;
      template class ParameternmtBuffer<3, 3, double>;
      template class ParameternmtBuffer<3, 4, double>;
      template class ParameternmtBuffer<4, 2, double>;
      template class ParameternmtBuffer<4, 3, double>;
      template class ParameternmtBuffer<4, 4, double>;

    } // namespace gl
  } // namespace rix
} // namespace dp
