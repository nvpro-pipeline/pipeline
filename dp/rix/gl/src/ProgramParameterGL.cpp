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


#include <dp/rix/gl/inc/BufferGL.h>
#include <dp/rix/gl/inc/TextureGL.h>
#include <dp/rix/gl/inc/ContainerGL.h>
#include <dp/rix/gl/inc/Sampler.h>
#include <dp/rix/gl/inc/SamplerStateGL.h>
#include <dp/rix/gl/inc/DataTypeConversionGL.h>
#include <dp/rix/gl/inc/ProgramGL.h>
#include <cstring>
#include <dp/rix/gl/inc/UniformUpdate.h>


using namespace dp::util;

namespace dp
{
  namespace rix
  {
    namespace gl
    {

#if 0
      /************************************************************************/
      /* glVertexAttrib wrappers                                              */
      /************************************************************************/
      // glVertexAttrib* template.
      template<unsigned int n, typename T>
      void setVertexAttribute( unsigned int attribute, const void *data ); // GENERAL CASE NOT SUPPORTED

      // glVertexAttribI{1,2,3,4}bv
      template<> void setVertexAttribute<1, int8_t>(unsigned int attributeIndex, const void *data )
      {
        int8_t tmp[4];
        tmp[0] = (((const int8_t*)data)[0]);
        glVertexAttribI4bv( attributeIndex, tmp );
      }
      template<> void setVertexAttribute<2, int8_t>(unsigned int attributeIndex, const void *data )
      {
        int8_t tmp[4];
        tmp[0] = ((const int8_t*)data)[0];
        tmp[1] = ((const int8_t*)data)[1];
        glVertexAttribI4bv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<3, int8_t>(unsigned int attributeIndex, const void *data )
      {
        int8_t tmp[4];
        tmp[0] = ((const int8_t*)data)[0];
        tmp[1] = ((const int8_t*)data)[1];
        tmp[2] = ((const int8_t*)data)[2];
        glVertexAttribI4bv( attributeIndex, tmp);
      }

      template<> void setVertexAttribute<4, int8_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribI4bv( attributeIndex, static_cast< const int8_t* >(data) );
      }

      // glVertexAttribI{1,2,3,4}sv
      template<> void setVertexAttribute<1, int16_t>(unsigned int attributeIndex, const void *data )
      {
        int16_t tmp[4];
        tmp[0] = ((const int16_t*)data)[0];
        glVertexAttribI4sv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<2, int16_t>(unsigned int attributeIndex, const void *data )
      {
        int16_t tmp[4];
        tmp[0] = ((const int16_t*)data)[0];
        tmp[1] = ((const int16_t*)data)[1];
        glVertexAttribI4sv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<3, int16_t>(unsigned int attributeIndex, const void *data )
      {
        int16_t tmp[4];
        tmp[0] = ((const int16_t*)data)[0];
        tmp[1] = ((const int16_t*)data)[1];
        tmp[2] = ((const int16_t*)data)[2];
        glVertexAttribI4sv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<4, int16_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribI4sv( attributeIndex, static_cast< const int16_t* >(data) );
      }

      // glVertexAttrib{1,2,3,4}iv
      template<> void setVertexAttribute<1, int32_t>(unsigned int attributeIndex, const void *data )
      {
        GLint tmp[4];
        tmp[0] = ((const GLint*)data)[0];
        glVertexAttribI4iv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<2, int32_t>(unsigned int attributeIndex, const void *data )
      {
        GLint tmp[4];
        tmp[0] = ((const GLint*)data)[0];
        tmp[1] = ((const GLint*)data)[1];
        glVertexAttribI4iv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<3, int32_t>(unsigned int attributeIndex, const void *data )
      {
        GLint tmp[4];
        tmp[0] = ((const GLint*)data)[0];
        tmp[1] = ((const GLint*)data)[1];
        tmp[2] = ((const GLint*)data)[2];
        glVertexAttribI4iv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<4, int32_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribI4iv( attributeIndex, static_cast<const GLint *>( data ));
      }

      // glVertexAttrib{1,2,3,4}Liv
      template<> void setVertexAttribute<1, int64_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribL1i64NV( attributeIndex, *static_cast<const int64_t *>( data ));
      }

      template<> void setVertexAttribute<2, int64_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribL2i64vNV( attributeIndex, static_cast<const int64_t *>( data ));
      }

      template<> void setVertexAttribute<3, int64_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribL3i64vNV( attributeIndex, static_cast<const int64_t *>( data ));
      }

      template<> void setVertexAttribute<4, int64_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribL4i64vNV( attributeIndex, static_cast<const int64_t *>( data ));
      }

      // glVertexAttrib{1,2,3,4}ubv
      template<> void setVertexAttribute<1, uint8_t>(unsigned int attributeIndex, const void *data )
      {
        uint8_t tmp[4];
        tmp[0] = ((const uint8_t*)data)[0];
        glVertexAttrib4ubv( attributeIndex, tmp );
      }
      template<> void setVertexAttribute<2, uint8_t>(unsigned int attributeIndex, const void *data )
      {
        uint8_t tmp[4];
        tmp[0] = ((const uint8_t*)data)[0];
        tmp[1] = ((const uint8_t*)data)[1];
        glVertexAttrib4ubv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<3, uint8_t>(unsigned int attributeIndex, const void *data )
      {
        uint8_t tmp[4];
        tmp[0] = ((const uint8_t*)data)[0];
        tmp[1] = ((const uint8_t*)data)[1];
        tmp[2] = ((const uint8_t*)data)[2];
        glVertexAttrib4ubv( attributeIndex, tmp );
      }
      template<> void setVertexAttribute<4, uint8_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttrib4ubv( attributeIndex, static_cast< const uint8_t* >(data) );
      }

      // glVertexAttrib{1,2,3,4}usb
      template<> void setVertexAttribute<1, uint16_t>(unsigned int attributeIndex, const void *data )
      {
        uint16_t tmp[4];
        tmp[0] = ((const uint16_t*)data)[0];
        glVertexAttribI4usv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<2, uint16_t>(unsigned int attributeIndex, const void *data )
      {
        uint16_t tmp[4];
        tmp[0] = ((const uint16_t*)data)[0];
        tmp[1] = ((const uint16_t*)data)[1];
        glVertexAttribI4usv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<3, uint16_t>(unsigned int attributeIndex, const void *data )
      {
        uint16_t tmp[4];
        tmp[0] = (((const uint16_t*)data)[0]);
        tmp[1] = (((const uint16_t*)data)[1]);
        tmp[2] = (((const uint16_t*)data)[2]);
        glVertexAttribI4usv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<4, uint16_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribI4usv( attributeIndex, static_cast<const uint16_t*>( data ));
      }

      // glVertexAttrib{1,2,3,4}uiv
      template<> void setVertexAttribute<1, uint32_t>(unsigned int attributeIndex, const void *data )
      {
        GLuint tmp[4];
        tmp[0] = (((const GLuint*)data)[0]);
        glVertexAttribI4uiv( attributeIndex, tmp );
      }
      template<> void setVertexAttribute<2, uint32_t>(unsigned int attributeIndex, const void *data )
      {
        GLuint tmp[4];
        tmp[0] = (((const GLuint*)data)[0]);
        tmp[1] = (((const GLuint*)data)[1]);
        glVertexAttribI4uiv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<3, uint32_t>(unsigned int attributeIndex, const void *data )
      {
        GLuint tmp[4];
        tmp[0] = (((const GLuint*)data)[0]);
        tmp[1] = (((const GLuint*)data)[1]);
        tmp[2] = (((const GLuint*)data)[2]);
        glVertexAttribI4uiv( attributeIndex, tmp );
      }

      template<> void setVertexAttribute<4, uint32_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribI4uiv( attributeIndex, static_cast<const GLuint*>( data ));
      }

      // glVertexAttrib{1,2,3,4}Luiv
      template<> void setVertexAttribute<1, uint64_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribL1ui64NV( attributeIndex, *static_cast<const GLuint64 *>( data ));
      }

      template<> void setVertexAttribute<2, uint64_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribL2ui64vNV( attributeIndex, static_cast<const GLuint64 *>( data ));
      }

      template<> void setVertexAttribute<3, uint64_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribL3ui64vNV( attributeIndex, static_cast<const GLuint64 *>( data ));
      }
      template<> void setVertexAttribute<4, uint64_t>(unsigned int attributeIndex, const void *data )
      {
        glVertexAttribL4ui64vNV( attributeIndex, static_cast<const GLuint64 *>( data ));
      }

      /************************************************************************/
      /* AttributeParameternt<n,T>                                            */
      /************************************************************************/
      template<unsigned int n, typename T>
      AttributeParameternt<n, T>::AttributeParameternt( unsigned int offset, unsigned int attributeIndex )
        : ParameterObject( offset, n * sizeof(T), static_cast<ConversionFunction>(&AttributeParameternt<n, T>::doUpdateConverted) )
        , m_attributeIndex( attributeIndex )
      {
      }

      template<unsigned int n, typename T>
      void AttributeParameternt<n, T>::update(const void *data)
      {
        const void* offsetData = static_cast<const char *>(data) + m_offset;
        setVertexAttribute<n, T>( m_attributeIndex, offsetData );
      }

      template<unsigned int n, typename T>
      void AttributeParameternt<n, T>::copy( const void* containerData, void* destination ) const
      {
        memcpy(destination, reinterpret_cast<const char*>(containerData) + m_offset, getConvertedSize() );
      }

      template<unsigned int n, typename T>
      void AttributeParameternt<n, T>::doUpdateConverted( void const* convertedData ) const
      {
        const void* offsetData = static_cast<const char *>(convertedData);
        setVertexAttribute<n, T>( m_attributeIndex, offsetData );
      }

      /************************************************************************/
      /* AttributeParameterBufferAddress                                      */
      /************************************************************************/
      AttributeParameterBufferAddress::AttributeParameterBufferAddress( unsigned int offset, unsigned int attribute, unsigned int arraySize )
        : ParameterObject( offset, sizeof(GLuint64EXT), static_cast<ConversionFunction>(&AttributeParameterBufferAddress::doUpdateConverted) )
        , m_attribute( attribute )
        , m_arraySize( arraySize )
      {
        DP_ASSERT( m_arraySize <= 0 && "only arraysize 0 (no array) is supported for AttributeParameterBufferAddress");
      }

      void AttributeParameterBufferAddress::update( const void* data )
      {
        const void* offsetData = static_cast<const char *>(data) + m_offset;
        GLuint64EXT ptr = *(const GLuint64EXT*)offsetData;
        glVertexAttribL1ui64NV(m_attribute, ptr );
      }

      void AttributeParameterBufferAddress::copy( const void* containerData, void* destination ) const
      {
        memcpy(destination, reinterpret_cast<const char*>(containerData) + m_offset, getConvertedSize() );
      }

      void AttributeParameterBufferAddress::doUpdateConverted( void const* convertedData ) const
      {
        const void* offsetData = static_cast<const char *>(convertedData);
        GLuint64EXT ptr = *(const GLuint64EXT*)offsetData;
        glVertexAttribL1ui64NV( m_attribute, ptr );
      }

      template class AttributeParameternt<1,int8_t>;
      template class AttributeParameternt<2,int8_t>;
      template class AttributeParameternt<3,int8_t>;
      template class AttributeParameternt<4,int8_t>;

      template class AttributeParameternt<1,int16_t>;
      template class AttributeParameternt<2,int16_t>;
      template class AttributeParameternt<3,int16_t>;
      template class AttributeParameternt<4,int16_t>;

      template class AttributeParameternt<1,int32_t>;
      template class AttributeParameternt<2,int32_t>;
      template class AttributeParameternt<3,int32_t>;
      template class AttributeParameternt<4,int32_t>;

      template class AttributeParameternt<1,int64_t>;
      template class AttributeParameternt<2,int64_t>;
      template class AttributeParameternt<3,int64_t>;
      template class AttributeParameternt<4,int64_t>;

      template class AttributeParameternt<1,uint8_t>;
      template class AttributeParameternt<2,uint8_t>;
      template class AttributeParameternt<3,uint8_t>;
      template class AttributeParameternt<4,uint8_t>;

      template class AttributeParameternt<1,uint16_t>;
      template class AttributeParameternt<2,uint16_t>;
      template class AttributeParameternt<3,uint16_t>;
      template class AttributeParameternt<4,uint16_t>;

      template class AttributeParameternt<1,uint32_t>;
      template class AttributeParameternt<2,uint32_t>;
      template class AttributeParameternt<3,uint32_t>;
      template class AttributeParameternt<4,uint32_t>;

      template class AttributeParameternt<1,uint64_t>;
      template class AttributeParameternt<2,uint64_t>;
      template class AttributeParameternt<3,uint64_t>;
      template class AttributeParameternt<4,uint64_t>;
#endif

    } // namespace gl
  } // namespace rix
} // namespace dp
