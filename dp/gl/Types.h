// Copyright NVIDIA Corporation 2014
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
/** @file */

#include <memory>
#include <GL/glew.h>
#include <dp/math/Vecnt.h>
#include <dp/math/Matmnt.h>
#include <dp/util/DPAssert.h>

// required declaration
namespace dp
{
  namespace gl
  {
#define TYPES(T)                        \
    class T;                            \
    typedef std::shared_ptr<T> Shared##T

    TYPES(Buffer);
    TYPES(ComputeShader);
    TYPES(DisplayList);
    TYPES(FragmentShader);
    TYPES(GeometryShader);
    TYPES(Object);
    TYPES(Program);
    TYPES(Renderbuffer);
    TYPES(RenderContext);
    TYPES(RenderTarget);
    TYPES(RenderTargetFB);
    TYPES(RenderTargetFBO);
    TYPES(Shader);
    TYPES(ShareGroup);
    TYPES(TessControlShader);
    TYPES(TessEvaluationShader);
    TYPES(Texture);
    TYPES(Texture1D);
    TYPES(Texture1DArray);
    TYPES(Texture2D);
    TYPES(Texture2DArray);
    TYPES(Texture3D);
    TYPES(TextureBuffer);
    TYPES(TextureCubemap);
    TYPES(TextureCubemapArray);
    TYPES(Texture2DMultisample);
    TYPES(Texture2DMultisampleArray);
    TYPES(TextureRectangle);
    TYPES(VertexArrayObject);
    TYPES(VertexShader);

#undef TYPES

    template <typename T>
    class TypeTraits
    {
      public:
        typedef T componentType;

      public:
        static unsigned int componentCount();
        static GLenum glType();
        static bool isInteger();
    };

    template <typename T>
    inline unsigned int TypeTraits<T>::componentCount()
    {
      DP_STATIC_ASSERT( std::numeric_limits<T>::is_specialized );
      return( 1 );
    }

    template <typename T>
    inline GLenum TypeTraits<T>::glType()
    {
      DP_STATIC_ASSERT( !"TypeTraits::glType: specialization for type T is missing" );
      return( GL_ERROR );
    }

    template <>
    inline GLenum TypeTraits<float>::glType()
    {
      return( GL_FLOAT );
    }

    template <>
    inline GLenum TypeTraits<int>::glType()
    {
      return( GL_INT );
    }

    template <>
    inline GLenum TypeTraits<short>::glType()
    {
      return( GL_SHORT );
    }

    template <>
    inline GLenum TypeTraits<unsigned int>::glType()
    {
      return( GL_UNSIGNED_INT );
    }

    template <typename T>
    inline bool TypeTraits<T>::isInteger()
    {
      DP_STATIC_ASSERT( std::numeric_limits<T>::is_specialized );
      return( std::numeric_limits<T>::is_integer );
    }


    template <unsigned int n, typename T>
    class TypeTraits<dp::math::Vecnt<n,T>>
    {
      public:
        typedef T componentType;

      public:
        static unsigned int componentCount()  { return n; }
        static GLenum glType();
        static bool isInteger()               { return( n == 1 ? TypeTraits<T>::isInteger() : false ); }
    };

    template <unsigned int n, typename T>
    inline GLenum TypeTraits<dp::math::Vecnt<n,T>>::glType()
    {
      DP_STATIC_ASSERT( !"TypeTraits::glType: specialization for type dp::math::Vecnt<n,T> is missing" );
      return( GL_ERROR );
    }

    template <>
    inline GLenum TypeTraits<dp::math::Vec4f>::glType()
    {
      return( GL_FLOAT_VEC4 );
    }


    template<unsigned int m, unsigned int n, typename T>
    class TypeTraits<dp::math::Matmnt<m,n,T>>
    {
      public:
        typedef T componentType;

      public:
        static unsigned int componentCount()  { return m * n; }
        static GLenum glType();
        static bool isInteger()               { return( m * n == 1 ? TypeTraits<T>::isInteger() : false ); }
    };

    template<unsigned int m, unsigned int n, typename T>
    inline GLenum TypeTraits<dp::math::Matmnt<m,n,T>>::glType()
    {
      DP_STATIC_ASSERT( !"TypeTraits::glType: specialization for type dp::math::Matmnt<m,n,T> is missing" );
      return( GL_ERROR );
    }

    template <>
    inline GLenum TypeTraits<dp::math::Mat33f>::glType()
    {
      return( GL_FLOAT_MAT3 );
    }

    template <>
    inline GLenum TypeTraits<dp::math::Mat44f>::glType()
    {
      return( GL_FLOAT_MAT4 );
    }


  } // namespace gl
} // namespace dp
