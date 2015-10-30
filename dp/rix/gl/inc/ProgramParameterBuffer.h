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


#pragma once

#include <ProgramParameterGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {

      /************************************************************************/
      /* ParameterntBuffer<n, T>                                                    */
      /************************************************************************/
      template<unsigned int n, typename T>
      class ParameterntBuffer : public ParameterObject
      {
      public:
        ParameterntBuffer( unsigned int offsetContainerData, unsigned int offsetBuffer, unsigned int arraySize );
        virtual void update( const void* data );
        virtual void copy( const void* containerData, void* destination ) const;
        void doUpdateConverted( void const* converted ) const;

        unsigned int m_offsetBuffer;
        unsigned int m_arraySize;
      };

      /************************************************************************/
      /* ParameterntBufferConversion                                                */
      /************************************************************************/
      template<unsigned int n, typename T, typename SourceType>
      class ParameterntBufferConversion : public ParameterObject
      {
      public:
        ParameterntBufferConversion( unsigned int offsetContainerData, unsigned int offsetBuffer, unsigned int arraySize );
        virtual void update( const void* data );
        virtual void copy( const void* containerData, void* destination ) const;
        void doUpdateConverted( void const* converted ) const;

        unsigned int m_offsetBuffer;
        unsigned int m_arraySize;
      };

      /************************************************************************/
      /* Parametermnt                                                         */
      /************************************************************************/
      template<unsigned int n, unsigned m, typename T>
      class ParameternmtBuffer : public ParameterObject
      {
      public:
        ParameternmtBuffer( unsigned int offsetContainerData, unsigned int offsetBuffer, unsigned int arraySize );
        virtual void update( const void* data );
        virtual void copy( const void* containerData, void* destination ) const;
        void doUpdateConverted( void const* converted ) const;

        unsigned int m_offsetBuffer;
        unsigned int m_arraySize;
      };

      /************************************************************************/
      /* Typedefs                                                             */
      /************************************************************************/

      typedef ParameterntBuffer<1, float> ParameterBufferFloat;
      typedef ParameterntBuffer<2, float> ParameterBufferFloat2;
      typedef ParameterntBuffer<3, float> ParameterBufferFloat3;
      typedef ParameterntBuffer<4, float> ParameterBufferFloat4;

      typedef ParameterntBuffer<1, double> ParameterBufferDouble;
      typedef ParameterntBuffer<2, double> ParameterBufferDouble2;
      typedef ParameterntBuffer<3, double> ParameterBufferDouble3;
      typedef ParameterntBuffer<4, double> ParameterBufferDouble4;

      typedef ParameterntBufferConversion<1, int32_t, int8_t> ParameterBufferInt_8;
      typedef ParameterntBufferConversion<2, int32_t, int8_t> ParameterBufferInt2_8;
      typedef ParameterntBufferConversion<3, int32_t, int8_t> ParameterBufferInt3_8;
      typedef ParameterntBufferConversion<4, int32_t, int8_t> ParameterBufferInt4_8;

      typedef ParameterntBufferConversion<1, int32_t, int16_t> ParameterBufferInt_16;
      typedef ParameterntBufferConversion<2, int32_t, int16_t> ParameterBufferInt2_16;
      typedef ParameterntBufferConversion<3, int32_t, int16_t> ParameterBufferInt3_16;
      typedef ParameterntBufferConversion<4, int32_t, int16_t> ParameterBufferInt4_16;

      typedef ParameterntBuffer<1, int32_t> ParameterBufferInt_32;
      typedef ParameterntBuffer<2, int32_t> ParameterBufferInt2_32;
      typedef ParameterntBuffer<3, int32_t> ParameterBufferInt3_32;
      typedef ParameterntBuffer<4, int32_t> ParameterBufferInt4_32;

      typedef ParameterntBuffer<1, int64_t> ParameterBufferInt_64;
      typedef ParameterntBuffer<2, int64_t> ParameterBufferInt2_64;
      typedef ParameterntBuffer<3, int64_t> ParameterBufferInt3_64;
      typedef ParameterntBuffer<4, int64_t> ParameterBufferInt4_64;

      typedef ParameterntBufferConversion<1, uint32_t, uint8_t> ParameterBufferUint_8;
      typedef ParameterntBufferConversion<2, uint32_t, uint8_t> ParameterBufferUint2_8;
      typedef ParameterntBufferConversion<3, uint32_t, uint8_t> ParameterBufferUint3_8;
      typedef ParameterntBufferConversion<4, uint32_t, uint8_t> ParameterBufferUint4_8;

      typedef ParameterntBufferConversion<1, uint32_t, uint16_t> ParameterBufferUint_16;
      typedef ParameterntBufferConversion<2, uint32_t, uint16_t> ParameterBufferUint2_16;
      typedef ParameterntBufferConversion<3, uint32_t, uint16_t> ParameterBufferUint3_16;
      typedef ParameterntBufferConversion<4, uint32_t, uint16_t> ParameterBufferUint4_16;

      typedef ParameterntBuffer<1, uint32_t> ParameterBufferUint_32;
      typedef ParameterntBuffer<2, uint32_t> ParameterBufferUint2_32;
      typedef ParameterntBuffer<3, uint32_t> ParameterBufferUint3_32;
      typedef ParameterntBuffer<4, uint32_t> ParameterBufferUint4_32;

      typedef ParameterntBuffer<1, uint64_t> ParameterBufferUint_64;
      typedef ParameterntBuffer<2, uint64_t> ParameterBufferUint2_64;
      typedef ParameterntBuffer<3, uint64_t> ParameterBufferUint3_64;
      typedef ParameterntBuffer<4, uint64_t> ParameterBufferUint4_64;

      typedef ParameternmtBuffer<2, 2, float> ParameterBufferFloat2x2;
      typedef ParameternmtBuffer<2, 3, float> ParameterBufferFloat2x3;
      typedef ParameternmtBuffer<2, 4, float> ParameterBufferFloat2x4;
      typedef ParameternmtBuffer<3, 2, float> ParameterBufferFloat3x2;
      typedef ParameternmtBuffer<3, 3, float> ParameterBufferFloat3x3;
      typedef ParameternmtBuffer<3, 4, float> ParameterBufferFloat3x4;
      typedef ParameternmtBuffer<4, 2, float> ParameterBufferFloat4x2;
      typedef ParameternmtBuffer<4, 3, float> ParameterBufferFloat4x3;
      typedef ParameternmtBuffer<4, 4, float> ParameterBufferFloat4x4;

      typedef ParameternmtBuffer<2, 2, double> ParameterBufferDouble2x2;
      typedef ParameternmtBuffer<2, 3, double> ParameterBufferDouble2x3;
      typedef ParameternmtBuffer<2, 4, double> ParameterBufferDouble2x4;
      typedef ParameternmtBuffer<3, 2, double> ParameterBufferDouble3x2;
      typedef ParameternmtBuffer<3, 3, double> ParameterBufferDouble3x3;
      typedef ParameternmtBuffer<3, 4, double> ParameterBufferDouble3x4;
      typedef ParameternmtBuffer<4, 2, double> ParameterBufferDouble4x2;
      typedef ParameternmtBuffer<4, 3, double> ParameterBufferDouble4x3;
      typedef ParameternmtBuffer<4, 4, double> ParameterBufferDouble4x4;

    } // namespace gl
  } // namespace rix
} // namespace dp
