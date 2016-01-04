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


#include <dp/rix/core/RiX.h>

#include <iostream>

namespace dp
{
  namespace rix
  {
    namespace core
    {

      Renderer::~Renderer()
      {
      }

      BufferDescription::BufferDescription( BufferDescriptionType type /* = BufferDescriptionType::COMMON */ )
        : m_type( type )
      {
      }

      BufferDescription::~BufferDescription()
      {
      }

      TextureDescription::TextureDescription( TextureType type, InternalTextureFormat internalFormat,
                                              dp::PixelFormat pixelFormat, dp::DataType dataType,
                                              size_t width /*= 0*/, size_t height /*= 0*/, size_t depth /*= 0*/, 
                                              size_t layers /*= 0*/, bool mipmaps /*= false */ )
        : m_type( type )
        , m_internalFormat( internalFormat )
        , m_pixelFormat( pixelFormat )
        , m_dataType( dataType )
        , m_width( width )
        , m_height( height )
        , m_depth( depth )
        , m_layers( layers )
        , m_mipmaps( mipmaps )
      {
      }

      TextureDescription::~TextureDescription()
      {
      }

      TextureData::TextureData( TextureDataType type )
        : m_type( type )
      {
      }

      TextureData::~TextureData()
      {
      }

      TextureDataBuffer::TextureDataBuffer( BufferSharedHandle const & buffer )
        : TextureData( TextureDataType::BUFFER )
        , m_buffer( buffer )
      {
      }

      TextureDataPtr::TextureDataPtr( void const * data, dp::PixelFormat pixelFormat, dp::DataType pixelDataType )
        : TextureData( TextureDataType::POINTER )
        , m_pData( data )
        , m_data( &m_pData )
        , m_numLayers( 1 )
        , m_numMipMapLevels( 0 )
        , m_pixelFormat( pixelFormat )
        , m_pixelDataType( pixelDataType )
      {
      }

      TextureDataPtr::TextureDataPtr( void const * const * data, unsigned int numMipMapLevels, dp::PixelFormat pixelFormat, dp::DataType pixelDataType )
        : TextureData( TextureDataType::POINTER )
        , m_pData( nullptr )
        , m_data( data )
        , m_numLayers( 1 )
        , m_numMipMapLevels( numMipMapLevels )
        , m_pixelFormat( pixelFormat )
        , m_pixelDataType( pixelDataType )
      {
      }

      TextureDataPtr::TextureDataPtr( void const * const * data, unsigned int numMipMapLevels, unsigned int numLayers, dp::PixelFormat pixelFormat, dp::DataType pixelDataType )
        : TextureData( TextureDataType::POINTER )
        , m_pData( nullptr )
        , m_data( data )
        , m_numLayers( numLayers )
        , m_numMipMapLevels( numMipMapLevels )
        , m_pixelFormat( pixelFormat )
        , m_pixelDataType( pixelDataType )
      {
      }

      GeometryInstanceDescription::GeometryInstanceDescription( GeometryInstanceDescriptionType type )
        : m_type( type )
      {
      }

      GeometryInstanceDescription::~GeometryInstanceDescription()
      {
      }


      ProgramParameter::ProgramParameter( const char* name, ContainerParameterType type, unsigned int arraySize /*= 0 */ )
        : m_name(name)
        , m_type(type)
        , m_arraySize(arraySize)
      {
      }

      ProgramParameter::~ProgramParameter()
      {
      }

      BufferObject::~BufferObject()
      {
      }
      
      ProgramShaderCode::ProgramShaderCode( const char* code, ShaderType shaderType )
        : ProgramShader( ProgramShaderType::CODE )
        , m_numShaders( 1 )
      {
        m_codeData = code;
        m_codes = &m_codeData;
        m_shaderTypeData = shaderType;
        m_shaderTypes = &m_shaderTypeData;
      }

      ProgramShaderCode::ProgramShaderCode( size_t numShaders, const char** codes, ShaderType* shaderTypes )
        : ProgramShader( ProgramShaderType::CODE )
        , m_codeData( nullptr )
        , m_shaderTypeData( ShaderType::NUM_SHADERTYPES )
        , m_numShaders( numShaders )
        , m_codes( codes )
        , m_shaderTypes( shaderTypes )
      {
      }

      ProgramShaderCode::~ProgramShaderCode()
      {
      }

      ProgramDescription::ProgramDescription( const ProgramShader& programShader, ContainerDescriptorSharedHandle* descriptors, size_t numDescriptors )
        : m_shader( programShader )
        , m_descriptors( descriptors )
        , m_numDescriptors( numDescriptors )
      {
      }

      ProgramDescription::~ProgramDescription()
      {
      }

      ProgramParameterDescriptor::~ProgramParameterDescriptor()
      {
      }

    } // core
  } // rix
} // dp

