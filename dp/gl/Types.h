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
#include <dp/ui/RenderTarget.h>
#include <dp/util/DPAssert.h>
#include <dp/util/SharedPtr.h>

// required declaration
namespace dp
{
  namespace gl
  {
    DEFINE_PTR_TYPES(Buffer);
    DEFINE_PTR_TYPES(ComputeShader);
    DEFINE_PTR_TYPES(DisplayList);
    DEFINE_PTR_TYPES(FragmentShader);
    DEFINE_PTR_TYPES(GeometryShader);
    DEFINE_PTR_TYPES(Object);
    DEFINE_PTR_TYPES(Program);
    DEFINE_PTR_TYPES(Renderbuffer);
    DEFINE_PTR_TYPES(RenderContext);
    DEFINE_PTR_TYPES(RenderTarget);
    DEFINE_PTR_TYPES(RenderTargetFB);
    DEFINE_PTR_TYPES(RenderTargetFBO);
    DEFINE_PTR_TYPES(Shader);
    DEFINE_PTR_TYPES(ShareGroup);
    DEFINE_PTR_TYPES(TessControlShader);
    DEFINE_PTR_TYPES(TessEvaluationShader);
    DEFINE_PTR_TYPES(Texture);
    DEFINE_PTR_TYPES(Texture1D);
    DEFINE_PTR_TYPES(Texture1DArray);
    DEFINE_PTR_TYPES(Texture2D);
    DEFINE_PTR_TYPES(Texture2DArray);
    DEFINE_PTR_TYPES(Texture3D);
    DEFINE_PTR_TYPES(TextureBuffer);
    DEFINE_PTR_TYPES(TextureCubemap);
    DEFINE_PTR_TYPES(TextureCubemapArray);
    DEFINE_PTR_TYPES(Texture2DMultisample);
    DEFINE_PTR_TYPES(Texture2DMultisampleArray);
    DEFINE_PTR_TYPES(TextureRectangle);
    DEFINE_PTR_TYPES(VertexArrayObject);
    DEFINE_PTR_TYPES(VertexShader);

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
#if defined(_MSVC_VER)
      DP_STATIC_ASSERT( std::numeric_limits<T>::is_specialized );
#endif
      return( 1 );
    }

      template <typename T>
    inline GLenum TypeTraits<T>::glType()
    {
#if defined(_MSVC_VER)
      DP_STATIC_ASSERT( !"TypeTraits::glType: specialization for type T is missing" );
#endif
      return( GL_INVALID_VALUE );
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
#if defined(_MSVC_VER)
      DP_STATIC_ASSERT( std::numeric_limits<T>::is_specialized );
#endif
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
#if defined(_MSVC_VER)
      DP_STATIC_ASSERT( !"TypeTraits::glType: specialization for type dp::math::Vecnt<n,T> is missing" );
#endif
      return( GL_INVALID_VALUE );
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
#if defined(_MSVC_VER)
      DP_STATIC_ASSERT( !"TypeTraits::glType: specialization for type dp::math::Matmnt<m,n,T> is missing" );
#endif
      return( GL_INVALID_VALUE );
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


namespace dp
{
  namespace util
  {

    /*! \brief Macro to define ObjectType and our four standard types of a base type T as part of a templated struct.
     *  \remark Using this struct, the standard types Handle, SharedPtr, WeakPtr and Lock, as well as
     *  the ObjectType itself, are easily available within a template context. */
    #define OBJECT_TRAITS_BASE(T)                 \
    template <> struct ObjectTraits<T>            \
    {                                             \
      typedef T                       ObjectType; \
      typedef dp::util::SharedPtr<T>  SharedPtr;  \
      typedef T*                      WeakPtr;    \
    }

    /*! \brief Macro to define ObjectType and our five standard types of a type T, with base type BT, as part of a templated struct.
     *  \remark Using this struct, the standard types Handle, SharedPtr, WeakPtr and Lock, as well as
     *  the ObjectType itself, are easily available within a template context. */
    #define OBJECT_TRAITS(T, BT)                  \
    template <> struct ObjectTraits<T>            \
    {                                             \
      typedef T                       ObjectType; \
      typedef BT                      Base;       \
      typedef dp::util::SharedPtr<T>  SharedPtr;  \
      typedef T*                      WeakPtr;    \
    }

    OBJECT_TRAITS_BASE( dp::gl::Object );
      OBJECT_TRAITS( dp::gl::Buffer, dp::gl::Object );
      OBJECT_TRAITS( dp::gl::DisplayList, dp::gl::Object );
      OBJECT_TRAITS( dp::gl::Program, dp::gl::Object );
      OBJECT_TRAITS( dp::gl::Renderbuffer, dp::gl::Object );
      OBJECT_TRAITS( dp::gl::Shader, dp::gl::Object );
        OBJECT_TRAITS( dp::gl::ComputeShader, dp::gl::Shader );
        OBJECT_TRAITS( dp::gl::FragmentShader, dp::gl::Shader );
        OBJECT_TRAITS( dp::gl::GeometryShader, dp::gl::Shader );
        OBJECT_TRAITS( dp::gl::TessControlShader, dp::gl::Shader );
        OBJECT_TRAITS( dp::gl::TessEvaluationShader, dp::gl::Shader );
        OBJECT_TRAITS( dp::gl::VertexShader, dp::gl::Shader );
      OBJECT_TRAITS( dp::gl::Texture, dp::gl::Object );
        OBJECT_TRAITS( dp::gl::Texture1D, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::Texture1DArray, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::Texture2D, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::Texture2DArray, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::Texture3D, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::TextureBuffer, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::TextureCubemap, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::TextureCubemapArray, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::Texture2DMultisample, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::Texture2DMultisampleArray, dp::gl::Texture );
        OBJECT_TRAITS( dp::gl::TextureRectangle, dp::gl::Texture );
      OBJECT_TRAITS( dp::gl::VertexArrayObject, dp::gl::Object );
    OBJECT_TRAITS_BASE( dp::gl::RenderContext );
    OBJECT_TRAITS_BASE( dp::gl::ShareGroup );

      OBJECT_TRAITS( dp::gl::RenderTarget, dp::ui::RenderTarget );
        OBJECT_TRAITS( dp::gl::RenderTargetFB, dp::gl::RenderTarget );
        OBJECT_TRAITS( dp::gl::RenderTargetFBO, dp::gl::RenderTarget );

  } // namespace util
} // namespace dp
