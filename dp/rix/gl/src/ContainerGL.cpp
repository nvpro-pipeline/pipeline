// Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
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
#include <dp/rix/gl/inc/BufferGL.h>
#include <dp/rix/gl/inc/Sampler.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>
#include <algorithm>
#include <cstring>
#include <typeinfo>
#include <dp/rix/gl/RiXGL.h>

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      using namespace dp::rix::core;

      size_t getSizeOfComponent( ContainerParameterType parameterType )
      {
        switch ( parameterType )
        {
        case ContainerParameterType::FLOAT:
        case ContainerParameterType::FLOAT2:
        case ContainerParameterType::FLOAT3:
        case ContainerParameterType::FLOAT4:
          return sizeof(float);

        case ContainerParameterType::INT_8:
        case ContainerParameterType::INT2_8:
        case ContainerParameterType::INT3_8:
        case ContainerParameterType::INT4_8:
          return sizeof(int8_t);

        case ContainerParameterType::INT_16:
        case ContainerParameterType::INT2_16:
        case ContainerParameterType::INT3_16:
        case ContainerParameterType::INT4_16:
          return sizeof(int16_t);

        case ContainerParameterType::INT_32:
        case ContainerParameterType::INT2_32:
        case ContainerParameterType::INT3_32:
        case ContainerParameterType::INT4_32:
          return sizeof(int32_t);

        case ContainerParameterType::INT_64:
        case ContainerParameterType::INT2_64:
        case ContainerParameterType::INT3_64:
        case ContainerParameterType::INT4_64:
          return sizeof(int64_t);

        case ContainerParameterType::UINT_8:
        case ContainerParameterType::UINT2_8:
        case ContainerParameterType::UINT3_8:
        case ContainerParameterType::UINT4_8:
          return sizeof(uint8_t);

        case ContainerParameterType::UINT_16:
        case ContainerParameterType::UINT2_16:
        case ContainerParameterType::UINT3_16:
        case ContainerParameterType::UINT4_16:
          return sizeof(uint16_t);

        case ContainerParameterType::UINT_32:
        case ContainerParameterType::UINT2_32:
        case ContainerParameterType::UINT3_32:
        case ContainerParameterType::UINT4_32:
          return sizeof(uint32_t);

        case ContainerParameterType::UINT_64:
        case ContainerParameterType::UINT2_64:
        case ContainerParameterType::UINT3_64:
        case ContainerParameterType::UINT4_64:
          return sizeof(uint64_t);

        case ContainerParameterType::BOOL:
        case ContainerParameterType::BOOL2:
        case ContainerParameterType::BOOL3:
        case ContainerParameterType::BOOL4:
          return sizeof(uint8_t);

        case ContainerParameterType::MAT2X2:
        case ContainerParameterType::MAT2X3:
        case ContainerParameterType::MAT2X4:
          return sizeof(float);

        case ContainerParameterType::MAT3X2:
        case ContainerParameterType::MAT3X3:
        case ContainerParameterType::MAT3X4:
          return sizeof(float);

        case ContainerParameterType::MAT4X2:
        case ContainerParameterType::MAT4X3:
        case ContainerParameterType::MAT4X4:
          return sizeof(float);

        case ContainerParameterType::BUFFER_ADDRESS:
          return sizeof(long long); // 64 bit value

        case ContainerParameterType::SAMPLER:
          return sizeof(ContainerGL::ParameterDataSampler);

        case ContainerParameterType::IMAGE:
          return sizeof(ContainerGL::ParameterDataImage);

        case ContainerParameterType::BUFFER:
          return sizeof(ContainerGL::ParameterDataBuffer);

        case ContainerParameterType::CALLBACK_:
          return sizeof(CallbackObject);

        default:
          DP_ASSERT( !"unsupported parameter type" );
          return 0;
        }
      }

      size_t getSizeOf( ContainerParameterType parameterType )
      {
        switch ( parameterType )
        {
        case ContainerParameterType::FLOAT:
          return sizeof(float);
        case ContainerParameterType::FLOAT2:
          return sizeof(float) * 2;
        case ContainerParameterType::FLOAT3:
          return sizeof(float) * 3;
        case ContainerParameterType::FLOAT4:
          return sizeof(float) * 4;

        case ContainerParameterType::INT_8:
          return sizeof(int8_t);
        case ContainerParameterType::INT2_8:
          return sizeof(int8_t) * 2;
        case ContainerParameterType::INT3_8:
          return sizeof(int8_t) * 3;
        case ContainerParameterType::INT4_8:
          return sizeof(int8_t) * 4;

        case ContainerParameterType::INT_16:
          return sizeof(int16_t);
        case ContainerParameterType::INT2_16:
          return sizeof(int16_t) * 2;
        case ContainerParameterType::INT3_16:
          return sizeof(int16_t) * 3;
        case ContainerParameterType::INT4_16:
          return sizeof(int16_t) * 4;

        case ContainerParameterType::INT_32:
          return sizeof(int32_t);
        case ContainerParameterType::INT2_32:
          return sizeof(int32_t) * 2;
        case ContainerParameterType::INT3_32:
          return sizeof(int32_t) * 3;
        case ContainerParameterType::INT4_32:
          return sizeof(int32_t) * 4;

        case ContainerParameterType::INT_64:
          return sizeof(int64_t);
        case ContainerParameterType::INT2_64:
          return sizeof(int64_t) * 2;
        case ContainerParameterType::INT3_64:
          return sizeof(int64_t) * 3;
        case ContainerParameterType::INT4_64:
          return sizeof(int64_t) * 4;

        case ContainerParameterType::UINT_8:
          return sizeof(uint8_t);
        case ContainerParameterType::UINT2_8:
          return sizeof(uint8_t) * 2;
        case ContainerParameterType::UINT3_8:
          return sizeof(uint8_t) * 3;
        case ContainerParameterType::UINT4_8:
          return sizeof(uint8_t) * 4;

        case ContainerParameterType::UINT_16:
          return sizeof(uint16_t);
        case ContainerParameterType::UINT2_16:
          return sizeof(uint16_t) * 2;
        case ContainerParameterType::UINT3_16:
          return sizeof(uint16_t) * 3;
        case ContainerParameterType::UINT4_16:
          return sizeof(uint16_t) * 4;

        case ContainerParameterType::UINT_32:
          return sizeof(uint32_t);
        case ContainerParameterType::UINT2_32:
          return sizeof(uint32_t) * 2;
        case ContainerParameterType::UINT3_32:
          return sizeof(uint32_t) * 3;
        case ContainerParameterType::UINT4_32:
          return sizeof(uint32_t) * 4;

        case ContainerParameterType::UINT_64:
          return sizeof(uint64_t);
        case ContainerParameterType::UINT2_64:
          return sizeof(uint64_t) * 2;
        case ContainerParameterType::UINT3_64:
          return sizeof(uint64_t) * 3;
        case ContainerParameterType::UINT4_64:
          return sizeof(uint64_t) * 4;

        case ContainerParameterType::BOOL:
          return sizeof(uint8_t);
        case ContainerParameterType::BOOL2:
          return sizeof(uint8_t) * 2;
        case ContainerParameterType::BOOL3:
          return sizeof(uint8_t) * 3;
        case ContainerParameterType::BOOL4:
          return sizeof(uint8_t) * 4;

        case ContainerParameterType::MAT2X2:
          return sizeof(float) * 4;
        case ContainerParameterType::MAT2X3:
          return sizeof(float) * 6;
        case ContainerParameterType::MAT2X4:
          return sizeof(float) * 8;

        case ContainerParameterType::MAT3X2:
          return sizeof(float) * 6;
        case ContainerParameterType::MAT3X3:
          return sizeof(float) * 9;
        case ContainerParameterType::MAT3X4:
          return sizeof(float) * 12;

        case ContainerParameterType::MAT4X2:
          return sizeof(float) * 8;
        case ContainerParameterType::MAT4X3:
          return sizeof(float) * 12;
        case ContainerParameterType::MAT4X4:
          return sizeof(float) * 16;

        case ContainerParameterType::BUFFER_ADDRESS:
          return sizeof(long long); // 64 bit value

        case ContainerParameterType::SAMPLER:
          return sizeof(ContainerGL::ParameterDataSampler);

        case ContainerParameterType::IMAGE:
          return sizeof(ContainerGL::ParameterDataImage);

        case ContainerParameterType::BUFFER:
          return sizeof(ContainerGL::ParameterDataBuffer);

        case ContainerParameterType::CALLBACK_:
          return sizeof(CallbackObject);

        default:
          DP_ASSERT( !"unsupported parameter type" );
          return 0;
        }
      }

      // initialize the first used id to something > 0
      unsigned int ContainerDescriptorGL::m_freeId = 42;

      ContainerDescriptorGL::ContainerDescriptorGL( RiXGL* renderer, size_t numParameters, ProgramParameter* parameters, bool multicast )
        : m_id( m_freeId++ << 16 )
        , m_size(0)
        , m_multicast(multicast)
        , m_renderer(renderer)
      {
        for ( size_t i = 0; i < numParameters; ++i )
        {
          ContainerDescriptorGL::ParameterInfo pi;

          pi.m_type      = parameters[i].m_type;
          pi.m_name      = parameters[i].m_name;
          pi.m_offset    = m_size;
          pi.m_arraySize = parameters[i].m_arraySize;

          // TODO: move all size types from unsigned int to size_t?
          // CK: why? hardly likely we have single arrays/types above 4 gb, keeps thing tight in cache
          pi.m_elementSize  = (unsigned int)getSizeOf( pi.m_type );
          pi.m_componentSize = (unsigned int)getSizeOfComponent( pi.m_type );

          pi.m_size = pi.m_elementSize * std::max<unsigned int>(pi.m_arraySize, 1);
          m_parameterInfos.push_back( pi );
          m_size += pi.m_size;
        }
      }

      ContainerGL::ContainerGL(dp::rix::gl::RiXGL *renderer, ContainerDescriptorGLSharedHandle const & desc)
        : m_descriptor( desc )
        , m_size( desc->m_size )
        , m_data( nullptr )
        , m_count( 0 )
        , m_renderer(renderer)
      {
        m_uniqueID = renderer->aquireContainerID();
        if ( m_size )
        {
          m_data = malloc( m_size * renderer->getNumberOfGPUs());
          // must be initialized with 0 so that pointer references can
          // be detected properly
          memset( m_data, 0, m_size );
          DP_ASSERT( m_data );
        }
      }

      ContainerGL::~ContainerGL()
      {
        m_renderer->releaseUniqueContainerID(m_uniqueID);
        for (size_t index = 0; index < m_descriptor->m_parameterInfos.size(); index++)
        {
          const ContainerDescriptorGL::ParameterInfo &descr = m_descriptor->m_parameterInfos[index];
          switch( descr.m_type )
          {
          case ContainerParameterType::SAMPLER:
            {
              char* selfdata = static_cast<char*>(m_data) + descr.m_offset;
              ParameterDataSampler* parameterData = reinterpret_cast<ParameterDataSampler*>(selfdata);

              if ( parameterData->m_samplerHandle )
              {
                parameterData->m_samplerHandle->detach( this );
                handleReset( parameterData->m_samplerHandle );
              }
            }
            break;
          case ContainerParameterType::IMAGE:
            {
              char* selfdata = static_cast<char*>(m_data) + descr.m_offset;
              ParameterDataImage* parameterImage = reinterpret_cast<ParameterDataImage*>(selfdata);

              handleReset( parameterImage->m_textureHandle );
            }
            break;
          case ContainerParameterType::BUFFER:
            {
              DP_ASSERT( (descr.m_size / descr.m_elementSize) == 1);

              char* selfdata =  static_cast<char*>(m_data) + descr.m_offset;
              ParameterDataBuffer *parameterData = reinterpret_cast<ParameterDataBuffer*>(selfdata);
              if ( parameterData->m_bufferHandle )
              {
                parameterData->m_bufferHandle->detach( this );
                handleReset(parameterData->m_bufferHandle);
              }
            }
            break;
          default:
            break;
          }
        }

        free(m_data);
      }

      void ContainerGL::setData( ContainerEntry entry, const ContainerData& containerData )
      {
        unsigned short index = m_descriptor->getIndex( entry );
        if ( index != (unsigned short)(~0) )
        {
          DP_ASSERT(index < m_descriptor->m_parameterInfos.size());

          const ContainerDescriptorGL::ParameterInfo &parameterInfo = m_descriptor->m_parameterInfos[index];

          switch ( containerData.getContainerDataType() )
          {
          case ContainerDataType::RAW:
            DP_ASSERT( dynamic_cast<const ContainerDataRaw*>(&containerData) );
            setData( parameterInfo, static_cast<const ContainerDataRaw&>( containerData ) );
            break;
          case ContainerDataType::BUFFER:
            DP_ASSERT( dynamic_cast<const ContainerDataBuffer*>(&containerData) );
            setData( parameterInfo, static_cast<const ContainerDataBuffer&>( containerData ) );
            break;
          case ContainerDataType::SAMPLER:
            DP_ASSERT( dynamic_cast<const ContainerDataSampler*>(&containerData) );
            setData( parameterInfo, static_cast<const ContainerDataSampler&>( containerData ) );
            break;
          case ContainerDataType::IMAGE:
            DP_ASSERT( dynamic_cast< const ContainerDataImage*>(&containerData) );
            setData( parameterInfo, static_cast<const ContainerDataImage&>( containerData ) );
            break;
          default:
            DP_ASSERT(!"Unsupported ContainerData type.");
            // returning here. Since nothing has changes it's not necessary to notify.
            return;
          }
          notify( ContainerGL::Event(this) );
        }
      }

      void ContainerGL::setData( const ContainerDescriptorGL::ParameterInfo& parameterInfo, const ContainerDataRaw& containerData )
      {
        size_t size = containerData.m_size;
        size_t offset = containerData.m_offset;
        void const* data = containerData.m_data;
        char* basePtr = reinterpret_cast<char*>(m_data) + containerData.m_gpuId * m_size;

        if ( size + offset <= parameterInfo.m_size )
        {
          DP_ASSERT ( (size   % parameterInfo.m_componentSize == 0) && "update size not multiple of component size" );
          DP_ASSERT ( (offset % parameterInfo.m_componentSize == 0) && "update offset not multiple of component size" );

          switch( parameterInfo.m_type )
          {
          case ContainerParameterType::FLOAT:
          case ContainerParameterType::FLOAT2:
          case ContainerParameterType::FLOAT3:
          case ContainerParameterType::FLOAT4:
          case ContainerParameterType::INT_8:
          case ContainerParameterType::INT2_8:
          case ContainerParameterType::INT3_8:
          case ContainerParameterType::INT4_8:
          case ContainerParameterType::INT_16:
          case ContainerParameterType::INT2_16:
          case ContainerParameterType::INT3_16:
          case ContainerParameterType::INT4_16:
          case ContainerParameterType::INT_32:
          case ContainerParameterType::INT2_32:
          case ContainerParameterType::INT3_32:
          case ContainerParameterType::INT4_32:
          case ContainerParameterType::INT_64:
          case ContainerParameterType::INT2_64:
          case ContainerParameterType::INT3_64:
          case ContainerParameterType::INT4_64:
          case ContainerParameterType::UINT_8:
          case ContainerParameterType::UINT2_8:
          case ContainerParameterType::UINT3_8:
          case ContainerParameterType::UINT4_8:
          case ContainerParameterType::UINT_16:
          case ContainerParameterType::UINT2_16:
          case ContainerParameterType::UINT3_16:
          case ContainerParameterType::UINT4_16:
          case ContainerParameterType::UINT_32:
          case ContainerParameterType::UINT2_32:
          case ContainerParameterType::UINT3_32:
          case ContainerParameterType::UINT4_32:
          case ContainerParameterType::UINT_64:
          case ContainerParameterType::UINT2_64:
          case ContainerParameterType::UINT3_64:
          case ContainerParameterType::UINT4_64:
          case ContainerParameterType::BOOL:
          case ContainerParameterType::BOOL2:
          case ContainerParameterType::BOOL3:
          case ContainerParameterType::BOOL4:
          case ContainerParameterType::MAT2X2: // DAR What about transpose handling?
          case ContainerParameterType::MAT2X3:
          case ContainerParameterType::MAT2X4:
          case ContainerParameterType::MAT3X2:
          case ContainerParameterType::MAT3X3:
          case ContainerParameterType::MAT3X4:
          case ContainerParameterType::MAT4X2:
          case ContainerParameterType::MAT4X3:
          case ContainerParameterType::MAT4X4:
          case ContainerParameterType::BUFFER_ADDRESS:
          case ContainerParameterType::CALLBACK_:
            {
              memcpy( basePtr + parameterInfo.m_offset + offset, data, size );
            }
            break;
          default:
            DP_ASSERT( !"unsupported ContainerParameterType!" );
            // returning here. Since nothing has changes it's not necessary to notify.
            return;
          }
          notify( ContainerGL::Event(this) );
        }
        else
        {
          DP_ASSERT(!"Container buffer not big enough to hold data.");
        }
      }

      void ContainerGL::setData( const ContainerDescriptorGL::ParameterInfo& parameterInfo, const ContainerDataBuffer& containerData )
      {
        switch (parameterInfo.m_type)
        {
        case ContainerParameterType::BUFFER:
          {
            char* selfdata =  static_cast<char*>(m_data) + parameterInfo.m_offset;
            ParameterDataBuffer* parameterData = reinterpret_cast<ParameterDataBuffer*>(selfdata);

            if ( parameterData->m_bufferHandle != containerData.m_bufferHandle.get() )
            {
              BufferGLHandle bufferGL = containerData.m_bufferHandle ? handleCast<BufferGL>(containerData.m_bufferHandle.get()) : nullptr;

              // observe buffer to get notified about buffer size changes
              if ( parameterData->m_bufferHandle )
              {
                parameterData->m_bufferHandle->detach( this );
              }

              if ( bufferGL )
              {
                bufferGL->attach( this );
              }

              // assign new buffer
              handleAssign( parameterData->m_bufferHandle, bufferGL );
            }

            parameterData->m_offset   = containerData.m_offset;
            parameterData->m_length   = containerData.m_length;

          }
          break;
        default:
          // returning here. Since nothing has changes it's not necessary to notify.
          DP_ASSERT(!"Unsupported combination of ContainerParameterType and ContainerDataType.");
          return;
        }
        notify( ContainerGL::Event(this) );
      }

      void ContainerGL::setData( const ContainerDescriptorGL::ParameterInfo& parameterInfo, const ContainerDataSampler& containerData )
      {
        switch ( parameterInfo.m_type )
        {
        case ContainerParameterType::SAMPLER:
          {
            char* selfdata = static_cast<char*>(m_data) + parameterInfo.m_offset;
            ParameterDataSampler* parameterData = reinterpret_cast<ParameterDataSampler*>(selfdata);

            if ( parameterData->m_samplerHandle )
            {
              parameterData->m_samplerHandle->detach( this );
            }

            handleAssign( parameterData->m_samplerHandle, handleCast<Sampler>( containerData.m_samplerHandle ) );

            if ( parameterData->m_samplerHandle )
            {
              parameterData->m_samplerHandle->attach( this );
            }

          }
          break;
        default:
          // returning here. Since nothing has changes it's not necessary to notify.
          DP_ASSERT(!"Unsupported combination of ContainerParameterType and ContainerDataType.");
          return;
        }
        notify( ContainerGL::Event(this) );
      }


      void ContainerGL::setData( const ContainerDescriptorGL::ParameterInfo& parameterInfo, const ContainerDataImage& containerData )
      {
        switch ( parameterInfo.m_type )
        {
        case ContainerParameterType::IMAGE:
          {
            char* selfdata = static_cast<char*>(m_data) + parameterInfo.m_offset;
            ParameterDataImage* parameterDataImage = reinterpret_cast<ParameterDataImage*>(selfdata);
            if( containerData.m_textureHandle )
            {
              handleAssign( parameterDataImage->m_textureHandle, handleCast<TextureGL>(containerData.m_textureHandle) );
            }
            else
            {
              handleReset( parameterDataImage->m_textureHandle );
            }
            parameterDataImage->m_level   = containerData.m_level;
            parameterDataImage->m_layered = containerData.m_layered;
            parameterDataImage->m_layer   = containerData.m_layer;
            parameterDataImage->m_access  = getGLAccessMode( containerData.m_access );
          }
          break;
        default:
          DP_ASSERT(!"Unsupported combination of ContainerParameterType and ContainerDataType.");
          // returning here. Since nothing has changes it's not necessary to notify.
          return;
        }
        notify( ContainerGL::Event(this) );
      }

      void ContainerGL::onNotify( const dp::util::Event &event, dp::util::Payload* /*payload*/ )
      {
        if ( typeid(event) == typeid( const BufferGL::Event& ) )
        {
          // Buffer Events are only of interest if the size does change
          const BufferGL::Event& bufferEvent = static_cast<const BufferGL::Event&>(event);
          if ( bufferEvent.m_type == BufferGL::Event::Type::DATA_AND_SIZE_CHANGED )
          {
            notify( ContainerGL::Event(this) );
          }
        }
        else
        {
          notify( ContainerGL::Event( this ) );
        }
      }

      void ContainerGL::onDestroyed( const dp::util::Subject& /*subject*/, dp::util::Payload* /*payload*/ )
      {
        DP_ASSERT( !"need to detach something?" );
      }

    } // namespace gl
  } // namespace rix
} // namespace dp
