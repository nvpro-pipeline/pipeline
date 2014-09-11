// Copyright NVIDIA Corporation 2011-2013
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

#include <xmmintrin.h>
#include <smmintrin.h>
#include <cstring>

namespace dp
{
  namespace math
  {
    namespace sse
    {
      /************************/
      /* Vec4f specialization */
      /************************/
      /*! \brief Vector class of fixed size and type.
       *  \remarks This class is templated by size and type. It holds \a n values of type \a T. There
       *  are typedefs for the most common usage with 2, 3, and 4 values of type \c float and \c double:
       *  Vec2f, Vec2d, Vec3f, Vec3d, Vec4f, Vec4d. */
      template<> class Vecnt<4, float>
      {
        public:
          /*! \brief Default constructor.
           *  \remarks For performance reasons, no initialization is performed. */
          Vecnt();

          /*! \brief Copy constructor from a vector with one less value than \c this, and an explicit last value.
           *  \param rhs A vector with \a m values of type \a S, where \a m has to be one less than \a n.
           *  \param last A single value of type \a R, that will be set as the last value of \c this.
           *  \remarks This constructor contains a compile-time NVSG_ASSERTion, to make sure that \a m is one
           *  less than \a n. The values of \a rhs of type \a S are converted to type \a T and assigned
           *  as the first values of \c this. The value \a last of type \a R also is converted to type
           *  \a T and assigned as the last value of \c this.
           *  \par Example:
           *  \code
           *    Vec3f v3f(0.0f,0.0f,0.0f);
           *    Vec4f v4f(v3f,1.0f);
           *  \endcode */
          Vecnt( const Vec3f & rhs, float last );

          /*! \brief Constructor for a single Element written to x,y,z,w
              \param \param u Value to use for all 4 components.
          **/
          explicit Vecnt( float u );

          /*! \brief Constructor for a four-element vector.
           *  \param x First element of the vector.
           *  \param y Second element of the vector.
           *  \param z Third element of the vector.
           *  \param w Fourth element of the vector.
           *  \remarks This constructor contains a compile-time NVSG_ASSERTion, to make sure it is used for
           *  four-element vectors, like Vec4f, only.
           *  \par Example:
           *  \code
           *    Vec4f         v4f( 1.0f, 2.0f, 3.0f, 4.0f );
           *    Vecnt<4,int>  v4i( 0, 1, 2, 3 );
           *  \endcode */
          Vecnt( float x, float y, float z, float w );

          /*! \brief Constructor for a four-element array.
              \param Array with 4 float values
          **/
          explicit Vecnt( const float rhs[4] );

          /*! \brief Constructor for a sse value
              \param 128-bit sse value with 4 floats
          **/
          Vecnt( __m128 sse )
            : m_sse( sse )
          {
          }

        public:
          /*! \brief Get a pointer to the constant values of this vector.
           *  \return A pointer to the constant values of this vector.
           *  \remarks It is assured, that the values of a vector are contiguous.
           *  \par Example:
           *  \code
           *    GLColor3fv( p->getDiffuseColor().getPtr() );
           *  \endcode */
          const float* getPtr() const;

          /*! \brief Direct access to the sse datatype of this vector
          **/
          __m128& sse();

          /*! \brief Direct access to the sse datatype of this vector
          **/
          const __m128& sse() const;

          /*! \brief Access operator to the values of a vector.
           *  \param i The index of the value to access.
           *  \return A reference to the value at position \a i in this vector.
           *  \remarks The index \a i has to be less than 4.
           *  \note The behavior is undefined if \ i is greater or equal to \a n. */
          float& operator[]( size_t i );

          /*! \brief Constant access operator to the values of a vector.
           *  \param i The index of the value to access.
           *  \return A constant reference to the value at position \a i in this vector.
           *  \remarks The index \a i has to be less than 4.
           *  \note The behavior is undefined if \ i is greater or equal to \a n. */
          const float& operator[]( size_t i ) const;

          /*! \brief Vector assignment operator.
           *  \param rhs A constant reference to the vector to assign to \c this.
           *  \return A reference to \c this.
           *  \remarks The values of \a rhs are component-wise assigned to the values of \c this. */
          Vecnt<4,float>& operator=( const Vecnt<4,float>& rhs );

          /*! \brief Vector addition and assignment operator with a vector of possibly different type.
           *  \param rhs A constant reference to the vector to add to \c this.
           *  \return A reference to \c this.
           *  \remarks The values of \a rhs are component-wise added to the values of \c this. */
          Vecnt<4,float>& operator+=( const Vecnt<4,float>& rhs );

          /*! \brief Vector subtraction and assignment operator with a vector of possibly different type.
           *  \param rhs A constant reference to the vector to subtract from \c this.
           *  \return A reference to \c this.
           *  \remarks The values of \a rhs are component-wise subtracted from the values of \c this. */
          Vecnt<4,float>& operator-=( const Vecnt<4,float>& rhs );

          /*! \brief Scalar multiplication and assignment operator with a scalar of same type.
           *  \param s A scalar to multiply \c this with.
           *  \return A reference to \c this.
           *  \remarks The values of \c this are component-wise multiplied with \a s. */
          Vecnt<4,float> & operator*=( float s );

          /*! \brief Scalar division and assignment operator with a scalar of same type.
           *  \param s A scalar to divide \c this by.
           *  \return A reference to \c this.
           *  \remarks The values of \c this are component-wise divided by \a s.
           *  \note The behavior is undefined if \a s is less than the type-dependent epsilon. */
          Vecnt<4,float> & operator/=( float s );

        /*! \brief Vector assignment operator with a vector of possibly different type.
         *  \param rhs A constant reference to the vector to assign to \c this.
         *  \return A reference to \c this.
         *  \remarks The values of \a rhs are component-wise assigned to the values of \c this. */
          template<typename S>
          Vecnt<4,float>& operator=( const Vecnt<4,S> & rhs );

          /*! \brief Normalize this vector and get it's previous length.
           *  \return The length of the vector before the normalization. */
          float normalize();

        private:
          union
          {
            __m128 m_sse;     // SSE is little endian
            float  m_vector[4];
          };
      };

      /*! \brief Default constructor.
        *  \remarks For performance reasons, no initialization is performed. */
      inline Vecnt<4, float>::Vecnt()
      {
      }

      inline Vecnt<4, float>::Vecnt( const Vec3f& rhs, float last )
      {
        m_sse = _mm_set_ps( last, rhs[2], rhs[1], rhs[0] );
      }

      inline Vecnt<4, float>::Vecnt( float u )
      {
        __m128 tmp = _mm_set_ss( u );
        m_sse = _mm_shuffle_ps( tmp, tmp, _MM_SHUFFLE( 0, 0, 0, 0 ) );
      }

      inline Vecnt<4, float>::Vecnt( float x, float y, float z, float w )
      {
        m_sse = _mm_set_ps( w, z, y, x );
      }

      inline Vecnt<4, float>::Vecnt( const float rhs[4] )
      {
        memcpy( m_vector, rhs, 4 * sizeof(float) );
      }

      inline const float* Vecnt<4, float>::getPtr() const
      {
        return m_vector;
      }

      /*! \brief Direct access to the sse datatype of this vector
      **/
      inline __m128& Vecnt<4, float>::sse()
      {
        return m_sse;
      }

      /*! \brief Direct access to the sse datatype of this vector
      **/
      inline const __m128& Vecnt<4, float>::sse() const
      {
        return m_sse;
      }

      /*! \brief Access operator to the values of a vector.
        *  \param i The index of the value to access.
        *  \return A reference to the value at position \a i in this vector.
        *  \remarks The index \a i has to be less than 4.
        *  \note The behavior is undefined if \ i is greater or equal to \a n. */
      inline float& Vecnt<4, float>::operator[]( size_t i )
      {
        DP_ASSERT( i < 4 );
        return m_vector[i];
      }

      /*! \brief Constant access operator to the values of a vector.
        *  \param i The index of the value to access.
        *  \return A constant reference to the value at position \a i in this vector.
        *  \remarks The index \a i has to be less than 4.
        *  \note The behavior is undefined if \ i is greater or equal to \a n. */
      inline const float& Vecnt<4, float>::operator[]( size_t i ) const
      {
        DP_ASSERT( i < 4 );
        return m_vector[i];
      }

      /*! \brief Vector assignment operator.
        *  \param rhs A constant reference to the vector to assign to \c this.
        *  \return A reference to \c this.
        *  \remarks The values of \a rhs are component-wise assigned to the values of \c this. */
      inline Vecnt<4,float>& Vecnt<4, float>::operator=( const Vecnt<4,float>& rhs )
      {
        m_sse = rhs.m_sse;
        return *this;
      }

      template<typename S>
      Vecnt<4,float>& Vecnt<4,float>::operator=( const Vecnt<4,S> & rhs )
      {
        m_vector[0] = (float)rhs[0];
        m_vector[1] = (float)rhs[1];
        m_vector[2] = (float)rhs[2];
        m_vector[3] = (float)rhs[3];
        return *this;
      }


      /*! \brief Vector addition and assignment operator with a vector of possibly different type.
        *  \param rhs A constant reference to the vector to add to \c this.
        *  \return A reference to \c this.
        *  \remarks The values of \a rhs are component-wise added to the values of \c this. */
      inline Vecnt<4,float>& Vecnt<4, float>::operator+=( const Vecnt<4,float>& rhs )
      {
        m_sse = _mm_add_ps( m_sse, rhs.m_sse );
        return *this;
      }

      /*! \brief Vector subtraction and assignment operator with a vector of possibly different type.
        *  \param rhs A constant reference to the vector to subtract from \c this.
        *  \return A reference to \c this.
        *  \remarks The values of \a rhs are component-wise subtracted from the values of \c this. */
      inline Vecnt<4,float>& Vecnt<4, float>::operator-=( const Vecnt<4,float>& rhs )
      {
        m_sse = _mm_sub_ps( m_sse, rhs.m_sse );
        return *this;
      }

      /*! \brief Scalar multiplication and assignment operator with a scalar of same type.
        *  \param s A scalar to multiply \c this with.
        *  \return A reference to \c this.
        *  \remarks The values of \c this are component-wise multiplied with \a s. */
      inline Vecnt<4,float> & Vecnt<4, float>::operator*=( float s )
      {
        __m128 scale = _mm_set_ss( s );
        scale = _mm_shuffle_ps( scale, scale, _MM_SHUFFLE(0,0,0,0) );
        m_sse = _mm_mul_ps( m_sse, scale );
        return *this;
      }

      /*! \brief Scalar division and assignment operator with a scalar of same type.
        *  \param s A scalar to divide \c this by.
        *  \return A reference to \c this.
        *  \remarks The values of \c this are component-wise divided by \a s.
        *  \note The behavior is undefined if \a s is less than the type-dependent epsilon. */
      inline Vecnt<4,float> & Vecnt<4, float>::operator/=( float s )
      {
        m_sse = _mm_div_ps( m_sse, Vec4f(s).sse() );
        return *this;
      }

      inline Vecnt< 4, float> operator*( const Vecnt< 4, float > &v1, float s )
      {
        Vec4f result = v1;
        result *= s;
        return result;
      }

      inline Vecnt< 4, float> operator*( float s, const Vecnt< 4, float > &v1 )
      {
        return( v1 * s );
      }

      inline float operator*( const Vecnt< 4, float > &v1, const Vecnt< 4, float > &v2 )
      {
        float result;
        _MM_EXTRACT_FLOAT( result, _mm_dp_ps( v1.sse(), v2.sse(), 0xf1 ),0 );
        return result;
      }

      inline Vecnt< 4, float > multiply( const Vecnt< 4, float > &v1, const Vecnt< 4, float > &v2 )
      {
        return Vecnt< 4, float >( _mm_mul_ps( v1.sse(), v2.sse() ) );
      }

      inline Vecnt< 4, float > operator+( const Vecnt< 4, float > &v1, const Vecnt< 4, float > &v2 )
      {
        return Vecnt< 4, float >( _mm_add_ps( v1.sse(), v2.sse() ) );
      }

      inline Vecnt< 4, float > operator-( const Vecnt< 4, float > &v1, const Vecnt< 4, float > &v2 )
      {
        return Vecnt< 4, float >( _mm_sub_ps( v1.sse(), v2.sse() ) );
      }

      template <int i3, int i2, int i1, int i0>
      inline Vecnt< 4, float > swizzle( const Vecnt< 4, float > &vector )
      {
        return Vecnt< 4, float >( _mm_shuffle_ps(vector.sse(), vector.sse(), _MM_SHUFFLE( i3, i2, i1, i0 )));
      }

      template<>
      inline float lengthSquared<4, float>( Vecnt<4,float> const & v)
      {
        return v*v;
      }

      template<>
      inline float length<4, float>( Vecnt<4,float> const & v)
      {
        __m128 dot = v.sse();
        dot = _mm_dp_ps( dot, dot, 0xf1 );

        float result;
        _MM_EXTRACT_FLOAT( result, dot,0 );
        return result;
      }

      inline float Vecnt<4,float>::normalize()
      {
        float l = length<4,float>(*this);
        *this /= l;
        return l;
      }

    } // sse
  } // math
} // namespace dp
