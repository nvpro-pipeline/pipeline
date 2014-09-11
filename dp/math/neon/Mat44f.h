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

#include <dp/math/neon/Matmnt.h>
#include <dp/math/neon/Vecnt.h>

namespace dp
{
  namespace math
  {
    namespace neon
    {

      inline Matmnt< 4, 4, float > operator*( const Matmnt< 4, 4, float > &m1, const Matmnt< 4, 4, float > &m2 );

      /**************************/
      /* Mat44f specialization **/
      /**************************/

      /*! \brief Matrix class of fixed size and type.
       *  \remarks This class is templated by size and type. It holds \a n times \a n values of type \a
       *  T. There are typedefs for the most common usage with 3 and 4 values of type \c float and \c
       *  double: Mat33f, Mat33d, Mat44f, Mat44d. */
      template<> class Matmnt<4, 4, float>
      {
        public:
          /*! \brief Default constructor.
           *  \remarks For performance reasons, no initialization is performed. */
          Matmnt();

          /*! \brief Copy constructor from a matrix of same size and type
           *  \param rhs A 4x4 matrix of type \a float.
           **/
           Matmnt( const Matmnt<4, 4, float> & rhs );

          /*! \brief Constructor for a 4x4 matrix out of an array with 16 floats
          *   \param mat 16 floats in row major.
          **/
          Matmnt( const float mat[16] );

          /*! \brief Constructor for a 4 by 4 matrix out of 16 scalar values.
           *  \param m00 First element of the first row of the matrix.
           *  \param m01 Second element of the first row of the matrix.
           *  \param m02 Third element of the first row of the matrix.
           *  \param m03 Fourth element of the first row of the matrix.
           *  \param m10 First element of the second row of the matrix.
           *  \param m11 Second element of the second row of the matrix.
           *  \param m12 Third element of the second row of the matrix.
           *  \param m13 Fourth element of the second row of the matrix.
           *  \param m20 First element of the third row of the matrix.
           *  \param m21 Second element of the third row of the matrix.
           *  \param m22 Third element of the third row of the matrix.
           *  \param m23 Fourth element of the third row of the matrix.
           *  \param m30 First element of the fourth row of the matrix.
           *  \param m31 Second element of the fourth row of the matrix.
           *  \param m32 Third element of the fourth row of the matrix.
           *  \param m33 Fourth element of the fourth row of the matrix.
           *  \remarks This constructor can only be used with 4 by 4 matrices, like Mat44f.
           *  \par Example:
           *  \code
           *    Mat44f m44f(  1.0f,  2.0f,  3.0f,  4.0f
           *               ,  5.0f,  6.0f,  7.0f,  8.0f
           *               ,  9.0f, 10.0f, 11.0f, 12.0f
           *               , 13.0f, 14.0f, 15.0f, 16.0f );
           *  \endcode */
          Matmnt( float m00, float m01, float m02, float m03
                , float m10, float m11, float m12, float m13
                , float m20, float m21, float m22, float m23
                , float m30, float m31, float m32, float m33 );

          /*! \brief Constructor for a 4 by 4 matrix out of four vectors of four values each.
           *  \param v0 First row of the matrix.
           *  \param v1 Second row of the matrix.
           *  \param v2 Third row of the matrix.
           *  \param v3 Fourth row of the matrix.
           *  \remarks This constructor can only be used with 4 by 4 matrices, like Mat44f.
           **/
          Matmnt( const Vec4f& v0, const Vec4f& v1, const Vec4f& v2, const Vec4f& v3 );

#if 0
          /*! \brief Constructor for a 4 by 4 transformation matrix out of a quaternion and a translation.
           *  \param ori A reference to the normalized quaternion representing the rotational part.
           *  \param trans A reference to the vector representing the translational part.
           *  \note The behavior is undefined, if \ ori is not normalized. */
          Matmnt( const Quatt<float> & ori, const Vecnt<3,float> & trans );

          /*! \brief Constructor for a 4 by 4 transformation matrix out of a quaternion, a translation,
           *  and a scaling.
           *  \param ori A reference to the normalized quaternion representing the rotational part.
           *  \param trans A reference to the vector representing the translational part.
           *  \param scale A reference to the vector representing the scaling along the three axis directions.
           *  \note The behavior is undefined, if \ ori is not normalized. */
          Matmnt( const Quatt<float> & ori, const Vecnt<3,float> & trans, const Vecnt<3,float> & scale );
#endif

        public:
          /*! \brief Get a constant pointer to the 16 values of the matrix.
           *  \return A constant pointer to the matrix elements.
           *  \remarks The matrix elements are stored in row-major order. This function returns the
           *  address of the first element of the first row. It is assured, that the other elements of
           *  the matrix follow linearly.
           */
          const float* getPtr() const;

          /*! \brief Invert the matrix.
           *  \return \c true, if the matrix was successfully inverted, otherwise \c false. */
          bool invert();

          /*! \brief Makes the current matrix a matrix identity
          **/
          void setIdentity();

          /*! \brief Non-constant subscript operator.
           *  \param i Index of the row to address.
           *  \return A reference to the \a i th row of the matrix. */
          Vec4f& operator[]( size_t i );

          /*! \brief Constant subscript operator.
           *  \param i Index of the row to address.
           *  \return A constant reference to the \a i th row of the matrix. */
          const Vec4f& operator[]( size_t i ) const;

          /*! \brief Matrix addition and assignment operator.
           *  \param m A constant reference to the matrix to add.
           *  \return A reference to \c this.
           *  \remarks The matrix \a m has to be of the same size as \c this, but may hold values of a
           *  different type. The matrix elements of type \a S of \a m are converted to type \a T and
           *  added to the corresponding matrix elements of \c this. */
          Matmnt<4, 4, float> & operator+=( const Matmnt<4, 4, float>& m );

          /*! \brief Matrix subtraction and assignment operator.
           *  \param m A constant reference to the matrix to subtract.
           *  \return A reference to \c this.
           *  \remarks The matrix \a m has to be of the same size and type as \c this.
          */
          Matmnt<4, 4, float>& operator-=( const Matmnt<4, 4, float> & m );

          /*! \brief Scalar multiplication and assignment operator.
           *  \param s A scalar value to multiply with.
           *  \return A reference to \c this.
           *  \remarks The type of \a s may be of different type as the elements of the \c this. \a s is
           *  converted to type \a T and each element of \c this is multiplied with it. */
          Matmnt<4, 4, float>& operator*=( float s );

          /*! \brief Matrix multiplication and assignment operator.
           *  \param m A constant reference to the matrix to multiply with.
           *  \return A reference to \c this.
           *  \remarks The matrix multiplication \code *this * m \endcode is calculated and assigned to
           *  \c this. */
          Matmnt<4, 4, float>& operator*=( const Matmnt<4, 4, float>& m );

          /*! \brief Scalar division and assignment operator.
           *  \param s A scalar value to divide by.
           *  \return A reference to \c this.
           *  \note The behavior is undefined if \a s is very close to zero. */
          Matmnt<4, 4, float> & operator/=( float s );

        private:
          Vec4f m_mat[4];
      };

#if 0
    /*! \brief Set the values of a 4 by 4 matrix by the constituents of a transformation.
     *  \param mat A reference to the matrix to set.
     *  \param ori A constant reference of the rotation part of the transformation.
     *  \param trans An optional constant reference to the translational part of the transformation. The
     *  default is a null vector.
     *  \param scale An optional constant reference to the scaling part of the transformation. The default
     *  is the identity scaling.
     *  \return A reference to \a mat. */
    template<typename T>
      Matmnt<4,4,T> & setMat( Matmnt<4,4,T> & mat, const Quatt<T> & ori
                                                 , const Vecnt<3,T> & trans = Vecnt<3,T>(0,0,0)
                                                 , const Vecnt<3,T> & scale = Vecnt<3,T>(1,1,1) )
    {
      Matmnt<3,3,T> m3( ori );
      mat[0] = Vecnt<4,T>( scale[0] * m3[0], 0 );
      mat[1] = Vecnt<4,T>( scale[1] * m3[1], 0 );
      mat[2] = Vecnt<4,T>( scale[2] * m3[2], 0 );
      mat[3] = Vecnt<4,T>( trans, 1 );
      return( mat );
    }
#endif


      inline Matmnt<4, 4, float>::Matmnt()
      {

      }

      inline Matmnt<4, 4, float>::Matmnt( const Matmnt<4, 4, float> & rhs )
      {
        m_mat[0] = rhs.m_mat[0];
        m_mat[1] = rhs.m_mat[1];
        m_mat[2] = rhs.m_mat[2];
        m_mat[3] = rhs.m_mat[3];
      }

      inline Matmnt<4, 4, float>::Matmnt( const float mat[16] )
      {
        m_mat[0] = Vec4f( mat[0], mat[1], mat[2], mat[3] );
        m_mat[1] = Vec4f( mat[4], mat[5], mat[6], mat[7] );
        m_mat[2] = Vec4f( mat[8], mat[9], mat[10], mat[11] );
        m_mat[3] = Vec4f( mat[12], mat[13], mat[14], mat[15] );
      }

      inline Matmnt<4, 4, float>::Matmnt( float m00, float m01, float m02, float m03
            , float m10, float m11, float m12, float m13
            , float m20, float m21, float m22, float m23
            , float m30, float m31, float m32, float m33 )
      {
        m_mat[0] = Vec4f( m03, m02, m01, m00 );
        m_mat[1] = Vec4f( m13, m12, m11, m10 );
        m_mat[2] = Vec4f( m23, m22, m21, m20 );
        m_mat[3] = Vec4f( m33, m32, m31, m30 );
      }

      inline Matmnt<4, 4, float>::Matmnt( const Vec4f& v0, const Vec4f& v1, const Vec4f& v2, const Vec4f& v3 )
      {
        m_mat[0] = v0;
        m_mat[1] = v1;
        m_mat[2] = v2;
        m_mat[3] = v3;
      }

#if 0
      inline Matmnt<4, 4, float>::Matmnt( const Quatt<float> & ori, const Vecnt<3,float> & trans )
      {
        setMat( *this, ori, trans );
      }

      inline Matmnt<4, 4, float>::Matmnt( const Quatt<float> & ori, const Vecnt<3,float> & trans, const Vecnt<3,float> & scale )
      {
        setMat( *this, ori, trans, scale );
      }
#endif


      inline const float* Matmnt<4, 4, float>::getPtr() const
      {
        return m_mat[0].getPtr();
      }

#if 0
      inline bool Matmnt<4, 4, float>::invert()
      {
        Matmnt< 4, 4, float > tmp;
        bool ok = dp::math::invert( *this, tmp );
        if ( ok )
        {
          *this = tmp;
        }
        return( ok );
      }
#endif

      inline void Matmnt<4, 4, float>::setIdentity()
      {
        // TODO might be much faster with _mm_shuffe_ps since the vector needs to be loaded only once from memory
        m_mat[0] = Vec4f( 1.0, 0.0, 0.0, 0.0 );
        m_mat[1] = Vec4f( 0.0, 1.0, 0.0, 0.0 );
        m_mat[2] = Vec4f( 0.0, 0.0, 1.0, 0.0 );
        m_mat[3] = Vec4f( 0.0, 0.0, 0.0, 1.0 );
      }

      inline Vec4f& Matmnt<4, 4, float>::operator[]( size_t i )
      {
        return m_mat[i];
      }

      inline const Vec4f& Matmnt<4, 4, float>::operator[]( size_t i ) const
      {
        return m_mat[i];
      }

      inline Matmnt<4, 4, float>& Matmnt<4, 4, float>::operator+=( const Matmnt<4, 4, float>& m )
      {
        m_mat[0] += m.m_mat[0];
        m_mat[1] += m.m_mat[1];
        m_mat[2] += m.m_mat[2];
        m_mat[3] += m.m_mat[3];
        return *this;
      }

      inline Matmnt<4, 4, float>& Matmnt<4, 4, float>::operator-=( const Matmnt<4, 4, float> & m )
      {
        m_mat[0] -= m.m_mat[0];
        m_mat[1] -= m.m_mat[1];
        m_mat[2] -= m.m_mat[2];
        m_mat[3] -= m.m_mat[3];
        return *this;
      }

      inline Matmnt<4, 4, float>& Matmnt<4, 4, float>::operator*=( float s )
      {
        m_mat[0] *= s;
        m_mat[1] *= s;
        m_mat[2] *= s;
        m_mat[3] *= s;
        return *this;
      }

      inline Matmnt<4, 4, float>& Matmnt<4, 4, float>::operator*=( const Matmnt<4, 4, float>& m )
      {
        *this = *this * m;
        return *this;
      }

      inline Matmnt<4, 4, float>& Matmnt<4, 4, float>::operator/=( float s )
      {
        m_mat[0] /= s;
        m_mat[1] /= s;
        m_mat[2] /= s;
        m_mat[3] /= s;
        return *this;
      }

      inline Matmnt< 4, 4, float > operator*( const Matmnt< 4, 4, float > &m1, const Matmnt< 4, 4, float > &m2 )
      {
        Matmnt< 4, 4, float > result;

#if 0
        const Vec4f v0 = m1[0];
        const Vec4f u0 = m2[0];
        const Vec4f u1 = m2[1];
        const Vec4f u2 = m2[2];
        const Vec4f u3 = m2[3];
        result[0] = multiply(Vec4f(v0[0]), u0);
        result[0] += multiply(Vec4f(v0[1]), u1);
        result[0] += multiply(Vec4f(v0[2]), u2);
        result[0] += multiply(Vec4f(v0[3]), u3);

        const Vec4f v1 = m1[1];
        result[1] = multiply(Vec4f(v1[0]), u0);
        result[1] += multiply(Vec4f(v1[1]), u1);
        result[1] += multiply(Vec4f(v1[2]), u2);
        result[1] += multiply(Vec4f(v1[3]), u3);

        const Vec4f v2 = m1[2];
        result[2] = multiply(Vec4f(v2[0]), u0);
        result[2] += multiply(Vec4f(v2[1]), u1);
        result[2] += multiply(Vec4f(v2[2]), u2);
        result[2] += multiply(Vec4f(v2[3]), u3);

        const Vec4f v3 = m1[3];
        result[3] = multiply(Vec4f(v3[0]), u0);
        result[3] += multiply(Vec4f(v3[1]), u1);
        result[3] += multiply(Vec4f(v3[2]), u2);
        result[3] += multiply(Vec4f(v3[3]), u3);
#else
        const Vec4f u0 = m2[0];
        const Vec4f u1 = m2[1];
        const Vec4f u2 = m2[2];
        const Vec4f u3 = m2[3];

        const Vec4f v0 = m1[0];
        float32x4_t result0;
        result0 = vmulq_f32(         vdupq_n_f32(v0[0]), u0.neon());
        result0 = vmlaq_f32(result0, vdupq_n_f32(v0[1]), u1.neon());
        result0 = vmlaq_f32(result0, vdupq_n_f32(v0[2]), u2.neon());
        result0 = vmlaq_f32(result0, vdupq_n_f32(v0[3]), u3.neon());
        result[0].neon() = result0;

        const Vec4f v1 = m1[1];
        float32x4_t result1;
        result1 = vmulq_f32(         vdupq_n_f32(v1[0]), u0.neon());
        result1 = vmlaq_f32(result1, vdupq_n_f32(v1[1]), u1.neon());
        result1 = vmlaq_f32(result1, vdupq_n_f32(v1[2]), u2.neon());
        result1 = vmlaq_f32(result1, vdupq_n_f32(v1[3]), u3.neon());
        result[1].neon() = result1;

        const Vec4f v2 = m1[2];
        float32x4_t result2;
        result2 = vmulq_f32(         vdupq_n_f32(v2[0]), u0.neon());
        result2 = vmlaq_f32(result2, vdupq_n_f32(v2[1]), u1.neon());
        result2 = vmlaq_f32(result2, vdupq_n_f32(v2[2]), u2.neon());
        result2 = vmlaq_f32(result2, vdupq_n_f32(v2[3]), u3.neon());
        result[2].neon() = result2;

        const Vec4f v3 = m1[3];
        float32x4_t result3;
        result3 = vmulq_f32(         vdupq_n_f32(v3[0]), u0.neon());
        result3 = vmlaq_f32(result3, vdupq_n_f32(v3[1]), u1.neon());
        result3 = vmlaq_f32(result3, vdupq_n_f32(v3[2]), u2.neon());
        result3 = vmlaq_f32(result3, vdupq_n_f32(v3[3]), u3.neon());
        result[3].neon() = result3;

#endif

        return result;
      }

      inline Vec4f operator*( const Vec4f& v, const Matmnt< 4, 4, float >& m )
      {
        float32x4_t result;
        result = vmulq_f32(        vdupq_n_f32(v[0]), m[0].neon());
        result = vmlaq_f32(result, vdupq_n_f32(v[1]), m[1].neon());
        result = vmlaq_f32(result, vdupq_n_f32(v[2]), m[2].neon());
        result = vmlaq_f32(result, vdupq_n_f32(v[3]), m[3].neon());

#if 0
        result  = multiply(m[0],  Vec4f(v[0]));
        result += multiply( m[1], Vec4f(v[1]));
        result += multiply( m[2], Vec4f(v[2]));
        result += multiply( m[3], Vec4f(v[3]));
#endif

        Vec4f tmp(result);

        return tmp;
      }

    } // namespace neon
  } // namespace math
} // namespace dp
