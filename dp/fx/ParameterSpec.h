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


#pragma once
/** \file */

#include <cstring>
#include <memory>
#include <dp/fx/Config.h>
#include <dp/fx/EnumSpec.h>
#include <dp/math/Matmnt.h>
#include <dp/math/Vecnt.h>
#include <dp/util/Semantic.h>
#include <dp/util/Types.h>

namespace dp
{
  namespace fx
  {
    const unsigned int PT_UNDEFINED = 0;

    const unsigned int PT_BOOL    =  1;
    const unsigned int PT_ENUM    =  2;
    const unsigned int PT_INT8    =  3;
    const unsigned int PT_UINT8   =  4;
    const unsigned int PT_INT16   =  5;
    const unsigned int PT_UINT16  =  6;
    const unsigned int PT_FLOAT32 =  7;
    const unsigned int PT_INT32   =  8;
    const unsigned int PT_UINT32  =  9;
    const unsigned int PT_FLOAT64 = 10;
    const unsigned int PT_INT64   = 11;
    const unsigned int PT_UINT64  = 12;
    const unsigned int PT_SCALAR_TYPE_MASK = 0xFF;

    const unsigned int PT_VECTOR2   =  1 << 8;
    const unsigned int PT_VECTOR3   =  2 << 8;
    const unsigned int PT_VECTOR4   =  3 << 8;
    const unsigned int PT_MATRIX2x2 =  4 << 8;
    const unsigned int PT_MATRIX2x3 =  5 << 8;
    const unsigned int PT_MATRIX2x4 =  6 << 8;
    const unsigned int PT_MATRIX3x2 =  7 << 8;
    const unsigned int PT_MATRIX3x3 =  8 << 8;
    const unsigned int PT_MATRIX3x4 =  9 << 8;
    const unsigned int PT_MATRIX4x2 = 10 << 8;
    const unsigned int PT_MATRIX4x3 = 11 << 8;
    const unsigned int PT_MATRIX4x4 = 12 << 8;
    const unsigned int PT_SCALAR_MODIFIER_MASK = 0xFF << 8;

    const unsigned int PT_BUFFER_PTR  = 1 << 16;
    const unsigned int PT_SAMPLER_PTR = 2 << 16;
    const unsigned int PT_TEXTURE_PTR = 3 << 16;
    const unsigned int PT_POINTER_TYPE_MASK = 0x0F << 16;

    // These are modifiers on const unsigned int PT_SAMPLER_PTR transporting the GLSL sampler type.
    const unsigned int PT_SAMPLER_1D                    =  1 << 24;
    const unsigned int PT_SAMPLER_2D                    =  2 << 24;
    const unsigned int PT_SAMPLER_3D                    =  3 << 24;
    const unsigned int PT_SAMPLER_CUBE                  =  4 << 24;
    const unsigned int PT_SAMPLER_2D_RECT               =  5 << 24;
    const unsigned int PT_SAMPLER_1D_ARRAY              =  6 << 24;
    const unsigned int PT_SAMPLER_2D_ARRAY              =  7 << 24;
    const unsigned int PT_SAMPLER_BUFFER                =  8 << 24;
    const unsigned int PT_SAMPLER_2D_MULTI_SAMPLE       =  9 << 24;
    const unsigned int PT_SAMPLER_2D_MULTI_SAMPLE_ARRAY = 10 << 24;
    const unsigned int PT_SAMPLER_CUBE_ARRAY            = 11 << 24;
    const unsigned int PT_SAMPLER_1D_SHADOW             = 12 << 24;
    const unsigned int PT_SAMPLER_2D_SHADOW             = 13 << 24;
    const unsigned int PT_SAMPLER_2D_RECT_SHADOW        = 14 << 24;
    const unsigned int PT_SAMPLER_1D_ARRAY_SHADOW       = 15 << 24;
    const unsigned int PT_SAMPLER_2D_ARRAY_SHADOW       = 16 << 24;
    const unsigned int PT_SAMPLER_CUBE_SHADOW           = 17 << 24;
    const unsigned int PT_SAMPLER_CUBE_ARRAY_SHADOW     = 18 << 24;
    const unsigned int PT_SAMPLER_TYPE_MASK             = 0xFF << 24;

    template <typename T> struct ParameterTraits
    {
      enum { type = PT_UNDEFINED };
    };

    template <> struct ParameterTraits<bool>
    {
      enum { type = PT_BOOL };
    };
    template <> struct ParameterTraits<char>
    {
      enum { type = PT_INT8 };
    };
    template <> struct ParameterTraits<unsigned char>
    {
      enum { type = PT_UINT8 };
    };
    template <> struct ParameterTraits<short>
    {
      enum { type = PT_INT16 };
    };
    template <> struct ParameterTraits<unsigned short>
    {
      enum { type = PT_UINT16 };
    };
    template <> struct ParameterTraits<int>
    {
      enum { type = PT_INT32 };
    };
    template <> struct ParameterTraits<unsigned int>
    {
      enum { type = PT_UINT32 };
    };
    template <> struct ParameterTraits<long long>
    {
      enum { type = PT_INT64 };
    };
    template <> struct ParameterTraits<unsigned long long>
    {
      enum { type = PT_UINT64 };
    };
    template <> struct ParameterTraits<float>
    {
      enum { type = PT_FLOAT32 };
    };
    template <> struct ParameterTraits<double>
    {
      enum { type = PT_FLOAT64 };
    };

    template <typename T> struct ParameterTraits<dp::math::Vecnt<2,T> >
    {
      enum { type = ParameterTraits<T>::type | PT_VECTOR2 };
    };
    template <typename T> struct ParameterTraits<dp::math::Vecnt<3,T> >
    {
      enum { type = ParameterTraits<T>::type | PT_VECTOR3 };
    };
    template <typename T> struct ParameterTraits<dp::math::Vecnt<4,T> >
    {
      enum { type = ParameterTraits<T>::type | PT_VECTOR4 };
    };

    template <typename T> struct ParameterTraits<dp::math::Matmnt<2,2,T> >
    {
      enum { type = ParameterTraits<T>::type | PT_MATRIX2x2 };
    };

    template <typename T> struct ParameterTraits<dp::math::Matmnt<3,3,T> >
    {
      enum { type = ParameterTraits<T>::type | PT_MATRIX3x3 };
    };

    template <typename T> struct ParameterTraits<dp::math::Matmnt<4,4,T> >
    {
      enum { type = ParameterTraits<T>::type | PT_MATRIX4x4 };
    };

    unsigned int getParameterAlignment( unsigned int type );
    unsigned int getParameterSize( unsigned int type );
    unsigned int getParameterCount( unsigned int type );

    class ParameterSpec
    {
      public:
        DP_FX_API ParameterSpec( const std::string & name, unsigned int type, dp::util::Semantic semantic, unsigned int arraySize = 0, const std::string & defaultString = "", const std::string & annotation = "" );
        DP_FX_API ParameterSpec( const std::string & name, unsigned int type, dp::util::Semantic semantic, unsigned int arraySize, const void* defaultValue, const std::string & annotation = "" );
        DP_FX_API ParameterSpec( const std::string & name, const SmartEnumSpec & enumSpec, unsigned int arraySize = 0, const std::string & defaultString = "", const std::string & annotation = "" );
        DP_FX_API ParameterSpec( const ParameterSpec & spec );
        DP_FX_API ~ParameterSpec();

        const std::string & getName() const;
        unsigned int getType() const;
        const std::string & getAnnotation() const;
        dp::util::Semantic getSemantic() const;
        unsigned int getArraySize() const;
        const void * getDefaultValue() const;
        template<typename T> const T & getDefaultValue( unsigned int idx = 0 ) const;
        const SmartEnumSpec & getEnumSpec() const;

        unsigned int getSizeInByte() const;
        unsigned int getElementSizeInBytes() const;

        bool operator==( const ParameterSpec & rhs ) const;

      private:
        std::string           m_name;
        unsigned int          m_type;
        SmartEnumSpec         m_enumSpec;
        std::string           m_annotation;
        dp::util::Semantic    m_semantic;
        unsigned int          m_arraySize;
        void                * m_defaultValue;
        // cached values
        unsigned int          m_sizeInBytes;
    };


    inline const std::string & ParameterSpec::getName() const
    {
      return( m_name );
    }

    inline unsigned int ParameterSpec::getType() const
    {
      return( m_type );
    }

    inline const std::string & ParameterSpec::getAnnotation() const
    {
      return( m_annotation );
    }

    inline dp::util::Semantic ParameterSpec::getSemantic() const
    {
      return( m_semantic );
    }

    inline unsigned int ParameterSpec::getArraySize() const
    {
      return( m_arraySize );
    }

    inline const void * ParameterSpec::getDefaultValue() const
    {
      return( m_defaultValue );
    }

    template<typename T>
    inline const T & ParameterSpec::getDefaultValue( unsigned int idx ) const
    {
      DP_ASSERT( ( ParameterTraits<T>::type == m_type ) || ( ( ParameterTraits<T>::type == PT_INT32 ) && ( m_type == PT_ENUM ) ) );
      DP_ASSERT( idx < std::max(1u,m_arraySize) );
      DP_ASSERT( m_defaultValue );
      return( (static_cast<const T *>(m_defaultValue))[idx] );
    }

    inline const SmartEnumSpec & ParameterSpec::getEnumSpec() const
    {
      return( m_enumSpec );
    }

    inline unsigned int ParameterSpec::getSizeInByte() const
    {
      return( m_sizeInBytes );
    }

    inline unsigned int ParameterSpec::getElementSizeInBytes() const
    {
      return( getParameterCount( m_type ) * getParameterSize( m_type ) );
    }

    inline bool ParameterSpec::operator==( const ParameterSpec & rhs ) const
    {
      return(   ( m_name == rhs.m_name )
            &&  ( m_type == rhs.m_type )
            &&  ( m_annotation == rhs.m_annotation )
            &&  ( m_semantic == rhs.m_semantic )
            &&  ( ( m_type & PT_SCALAR_TYPE_MASK ) != PT_ENUM || ( m_enumSpec == rhs.m_enumSpec ) )
            &&  ( m_arraySize == rhs.m_arraySize )
            &&  ( !!m_defaultValue == !!rhs.m_defaultValue )
            &&  ( !m_defaultValue || memcmp( m_defaultValue, rhs.m_defaultValue, getSizeInByte() ) == 0 ) );
    }

    inline bool isParameterScalar( const ParameterSpec& parameterSpec )
    {
      return (parameterSpec.getType() & ( dp::fx::PT_SCALAR_TYPE_MASK | dp::fx::PT_SCALAR_MODIFIER_MASK ) ) == parameterSpec.getType();
    }

    inline bool isParameterPointer( const ParameterSpec& parameterSpec )
    {
      return (parameterSpec.getType() & ( dp::fx::PT_POINTER_TYPE_MASK | dp::fx::PT_SAMPLER_TYPE_MASK ) ) == parameterSpec.getType();
    }

    inline bool isParameterArray( const ParameterSpec& parameterSpec )
    {
      return parameterSpec.getArraySize() != 0;
    }

    inline bool isParameterEnum( const ParameterSpec& parameterSpec )
    {
      return ( parameterSpec.getType() & PT_SCALAR_TYPE_MASK ) == PT_ENUM;
    }

    inline unsigned int getParameterAlignment( unsigned int type )
    {
      unsigned int alignment = 0;
      if ( type & PT_SCALAR_TYPE_MASK )
      {
        switch( type & PT_SCALAR_TYPE_MASK )
        {
        case PT_BOOL :
          DP_STATIC_ASSERT( sizeof(bool) == 1 );
        case PT_INT8 :
        case PT_UINT8 :
          alignment = 1;
          break;
        case PT_INT16 :
        case PT_UINT16 :
          alignment = 2;
          break;
        case PT_ENUM :
        case PT_FLOAT32 :
        case PT_INT32 :
        case PT_UINT32 :
          alignment = 4;
          break;
        case PT_FLOAT64 :
        case PT_INT64 :
        case PT_UINT64 :
          alignment = 8;
          break;
        default :
          DP_ASSERT( false );
          break;
        }
      }
      else if ( type & PT_POINTER_TYPE_MASK )
      {
        alignment = sizeof(void*);
      }
      return( alignment );
    }

    inline unsigned int getParameterSize( unsigned int type )
    {
      unsigned int size = 0;
      if ( type & PT_SCALAR_TYPE_MASK )
      {
        switch( type & PT_SCALAR_TYPE_MASK )
        {
        case PT_ENUM :
          size = dp::util::checked_cast<unsigned int>( sizeof( EnumSpec::StorageType ) );
          break;
        case PT_BOOL :
          DP_STATIC_ASSERT( sizeof(bool) == 1 );
        case PT_INT8 :
        case PT_UINT8 :
          size = 1;
          break;
        case PT_INT16 :
        case PT_UINT16 :
          size = 2;
          break;
        case PT_FLOAT32 :
        case PT_INT32 :
        case PT_UINT32 :
          size = 4;
          break;
        case PT_FLOAT64 :
        case PT_INT64 :
        case PT_UINT64 :
          size = 8;
          break;
        default :
          DP_ASSERT( false );
          break;
        }
      }
      else if ( type & PT_POINTER_TYPE_MASK )
      {
        size = sizeof( std::shared_ptr<void> ); //sizeof( dp::sg::core::HandledObjectSharedPtr );
      }
      return( size );
    }

    inline unsigned int getParameterCount( unsigned int type )
    {
      unsigned int count = 1;
      switch( type & PT_SCALAR_MODIFIER_MASK )
      {
      case PT_VECTOR2 :
        count = 2;
        break;
      case PT_VECTOR3 :
        count = 3;
        break;
      case PT_VECTOR4 :
      case PT_MATRIX2x2 :
        count = 4;
        break;
      case PT_MATRIX2x3 :
      case PT_MATRIX3x2 :
        count = 6;
        break;
      case PT_MATRIX2x4 :
      case PT_MATRIX4x2 :
        count = 8;
        break;
      case PT_MATRIX3x3 :
        count = 9;
        break;
      case PT_MATRIX3x4 :
      case PT_MATRIX4x3 :
        count = 12;
        break;
      case PT_MATRIX4x4 :
        count = 16;
        break;
      default :
        break;
      }
      return( count );
    }

  } // namespace fx
} // namespace dp
