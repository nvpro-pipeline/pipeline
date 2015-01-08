// Copyright NVIDIA Corporation 2002-2011
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

#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cassert>

#include "RendererGLConfig.h"

#if defined(RENDERERGL_SSE)
  #include <xmmintrin.h>
  #include <emmintrin.h>
#endif

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      /*! \brief Vector class of fixed size and type.
       *  \remarks This class is templated by size and type. It holds \a n values of type \a T. There
       *  are typedefs for the most common usage with 2, 3, and 4 values of type \c float and \c double:
       *  Vec2f, Vec2d, Vec3f, Vec3d, Vec4f, Vec4d. */
      template<unsigned int n, typename T> class Vecnt
      {
        public:
          /*! \brief Default constructor.
           *  \remarks For performance reasons, no initialization is performed. */
          Vecnt();

          /*! \brief Copy constructor from a vector of possibly different size and type.
           *  \param rhs A vector with \a m values of type \a S.
           *  \remarks The minimum \a k of \a n and \a m is determined. The first \a k values of type \a
           *  S from \a rhs are converted to type \a T and assigned as the first \a k values of \c this.
           *  If \a k is less than \a n, the \a n - \a k last values of \c this are not initialized. */
          template<unsigned int m, typename S>
            explicit Vecnt( const Vecnt<m,S> & rhs );

          /*! \brief Copy constructor from a vector with one less value than \c this, and an explicit last value.
           *  \param rhs A vector with \a m values of type \a S, where \a m has to be one less than \a n.
           *  \param last A single value of type \a R, that will be set as the last value of \c this.
           *  \remarks This constructor contains a compile-time assertion, to make sure that \a m is one
           *  less than \a n. The values of \a rhs of type \a S are converted to type \a T and assigned
           *  as the first values of \c this. The value \a last of type \a R also is converted to type
           *  \a T and assigned as the last value of \c this.
           *  \par Example:
           *  \code
           *    Vec3f v3f(0.0f,0.0f,0.0f);
           *    Vec4f v4f(v3f,1.0f);
           *  \endcode */
          template<typename S, typename R>
            Vecnt( const Vecnt<n-1,S> & rhs, R last );

          /*! \brief Constructor for a one-element vector.
           *  \param x First element of the vector.
           *  \remarks This constructor can only be used with one-element vectors.
           *  \par Example:
           *  \code
           *    Vec1f         v1f( 1.0f );
           *    Vecnt<1,int>  v1i( 0 );
           *  \endcode */
          Vecnt( T x);

          /*! \brief Constructor for a two-element vector.
           *  \param x First element of the vector.
           *  \param y Second element of the vector.
           *  \remarks This constructor can only be used with two-element vectors.
           *  \par Example:
           *  \code
           *    Vec2f         v2f( 1.0f, 2.0f );
           *    Vecnt<2,int>  v2i( 0, 1 );
           *  \endcode */
          Vecnt( T x, T y );

          /*! \brief Constructor for a three-element vector.
           *  \param x First element of the vector.
           *  \param y Second element of the vector.
           *  \param z Third element of the vector.
           *  \remarks This constructor contains a compile-time assertion, to make sure it is used for
           *  three-element vectors, like Vec3f, only.
           *  \par Example:
           *  \code
           *    Vec3f         v3f( 1.0f, 2.0f, 3.0f );
           *    Vecnt<3,int>  v3i( 0, 1, 2 );
           *  \endcode */
          Vecnt( T x, T y, T z );

          /*! \brief Constructor for a four-element vector.
           *  \param x First element of the vector.
           *  \param y Second element of the vector.
           *  \param z Third element of the vector.
           *  \param w Fourth element of the vector.
           *  \remarks This constructor contains a compile-time assertion, to make sure it is used for
           *  four-element vectors, like Vec4f, only.
           *  \par Example:
           *  \code
           *    Vec4f         v4f( 1.0f, 2.0f, 3.0f, 4.0f );
           *    Vecnt<4,int>  v4i( 0, 1, 2, 3 );
           *  \endcode */
          Vecnt( T x, T y, T z, T w );

        public:
          /*! \brief Get a pointer to the constant values of this vector.
           *  \return A pointer to the constant values of this vector.
           *  \remarks It is assured, that the values of a vector are contiguous.
           *  \par Example:
           *  \code
           *    GLColor3fv( p->getDiffuseColor().getPtr() );
           *  \endcode */
          const T * getPtr() const;

          /*! \brief Normalize this vector and get it's previous length.
           *  \return The length of the vector before the normalization. */
          T normalize();

          /*! \brief Access operator to the values of a vector.
           *  \param i The index of the value to access.
           *  \return A reference to the value at position \a i in this vector.
           *  \remarks The index \a i has to be less than the size of the vector, given by the template
           *  argument \a n.
           *  \note The behavior is undefined if \ i is greater or equal to \a n. */
          T & operator[]( size_t i );

          /*! \brief Constant access operator to the values of a vector.
           *  \param i The index of the value to access.
           *  \return A constant reference to the value at position \a i in this vector.
           *  \remarks The index \a i has to be less than the size of the vector, given by the template
           *  argument \a n.
           *  \note The behavior is undefined if \ i is greater or equal to \a n. */
          const T & operator[]( size_t i ) const;

          /*! \brief Vector assignment operator with a vector of possibly different type.
           *  \param rhs A constant reference to the vector to assign to \c this.
           *  \return A reference to \c this.
           *  \remarks The values of \a rhs are component-wise assigned to the values of \c this. */
          template<typename S>
            Vecnt<n,T> & operator=( const Vecnt<n,S> & rhs );

          /*! \brief Vector addition and assignment operator with a vector of possibly different type.
           *  \param rhs A constant reference to the vector to add to \c this.
           *  \return A reference to \c this.
           *  \remarks The values of \a rhs are component-wise added to the values of \c this. */
          template<typename S>
            Vecnt<n,T> & operator+=( const Vecnt<n,S> & rhs );

          /*! \brief Vector subtraction and assignment operator with a vector of possibly different type.
           *  \param rhs A constant reference to the vector to subtract from \c this.
           *  \return A reference to \c this.
           *  \remarks The values of \a rhs are component-wise subtracted from the values of \c this. */
          template<typename S>
            Vecnt<n,T> & operator-=( const Vecnt<n,S> & rhs );

          /*! \brief Scalar multiplication and assignment operator with a scalar of possibly different type.
           *  \param s A scalar to multiply \c this with.
           *  \return A reference to \c this.
           *  \remarks The values of \c this are component-wise multiplied with \a s. */
          template<typename S>
            Vecnt<n,T> & operator*=( S s );

          /*! \brief Scalar division and assignment operator with a scalar of possibly different type.
           *  \param s A scalar to divide \c this by.
           *  \return A reference to \c this.
           *  \remarks The values of \c this are component-wise divided by \a s.
           *  \note The behavior is undefined if \a s is less than the type-dependent epsilon. */
          template<typename S>
            Vecnt<n,T> & operator/=( S s );

          /*! \brief Orthonormalize \c this with respect to the vector \a v.
           *  \param v A constant reference to the vector to orthonormalize \c this to.
           *  \remarks Subtracts the orthogonal projection of \c this on \a v from \c this and
           *  normalizes the it, resulting in a normalized vector that is orthogonal to \a v. */
          void orthonormalize( const Vecnt<n,T> & v );

        private:
          T m_vec[n];
      };


      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // non-member functions
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

       /*! \brief Determine if two vectors point in opposite directions.
       *  \param v0 A constant reference to the first normalized vector to use.
       *  \param v1 A constant reference to the second normalized vector to use.
       *  \param eps An optional type dependent epsilon, defining the acceptable deviation.
       *  \return \c true, if \a v0 and \a v1 are anti-parallel, otherwise \c false.
       *  \note The behavior is undefined if \a v0 or \a v1 are not normalized.
       *  \sa areCollinear, isNormalized */
      template<unsigned int n, typename T>
        bool areOpposite( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1
                        , T eps = std::numeric_limits<T>::epsilon() )
      {
        DP_PRIVATE_ASSERT( isNormalized( v0 ) && isNormalized( v1 ) );
        return( ( 1 + v0 * v1 ) <= eps );
      }

      /*! \brief Determine if two vectors are orthogonal.
       *  \param v0 A constant reference to the first vector to use.
       *  \param v1 A constant reference to the second vector to use.
       *  \param eps An optional deviation from orthonormality. The default is the type dependent
       *  epsilon.
       *  \return \c true, if \a v0 and \a v1 are orthogonal, otherwise \c false.
       *  \sa areOrthonormal */
      template<unsigned int n, typename T>
        bool areOrthogonal( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1
                          , T eps = 2 * std::numeric_limits<T>::epsilon() )
      {
        return( abs(v0*v1) <= std::max(T(1),length(v0)) * std::max(T(1),length(v1)) * eps );
      }

      /*! \brief Determine if two vectors are orthonormal.
       *  \param v0 A constant reference to the first normalized vector to use.
       *  \param v1 A constant reference to the second normalized vector to use.
       *  \param eps An optional deviation from orthonormality. The default is the type dependent
       *  epsilon.
       *  \return \c true, if \a v0 and \a v1 are orthonormal, otherwise \c false.
       *  \note The behavior is undefined if \a v0 or \a v1 are not normalized.
       *  \sa areOrthogonal */
      template<unsigned int n, typename T>
        bool areOrthonormal( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1
                           , T eps = 2 * std::numeric_limits<T>::epsilon() )
      {
        return( isNormalized(v0) && isNormalized(v1) && ( abs(v0*v1) <= eps ) );
      }

      /*! \brief Determine if two vectors differ less than a given epsilon in each component.
       *  \param v0 A constant reference to the first vector to use.
       *  \param v1 A constant reference to the second vector to use.
       *  \param eps The acceptable deviation for each component.
       *  \return \c true, if \ v0 and \a v1 differ less than or equal to \a eps in each component, otherwise \c
       *  false.
       *  \sa distance, operator==() */
      template<unsigned int n, typename T>
        bool areSimilar( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1, T eps );

      /*! \brief Determine the distance between vectors.
       *  \param v0 A constant reference to the first vector.
       *  \param v1 A constant reference to the second vector.
       *  \return The euclidean distance between \a v0 and \a v1.
       *  \sa length, lengthSquared */
      template<unsigned int n, typename T>
        T distance( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 );

      template<unsigned int n, typename T>
        T intensity( const Vecnt<n,T> & v );

      /*! \brief Determine if a vector is normalized.
       *  \param v A constant reference to the vector to test.
       *  \param eps An optional type dependent epsilon, defining the acceptable deviation.
       *  \return \c true, if \a v is normalized, otherwise \c false.
       *  \sa isNull, length, lengthSquared, normalize */
      template<unsigned int n, typename T>
        bool isNormalized( const Vecnt<n,T> & v, T eps = 2 * std::numeric_limits<T>::epsilon() )
      {
        return( abs( length( v ) - 1 ) <= eps );
      }

      /*! \brief Determine if a vector is the null vector.
       *  \param v A constant reference to the vector to test.
       *  \param eps An optional type dependent epsilon, defining the acceptable deviation.
       *  \return \c true, if \a v is the null vector, otherwise \c false.
       *  \sa isNormalized, length, lengthSquared */
      template<unsigned int n, typename T>
        bool isNull( const Vecnt<n,T> & v, T eps = std::numeric_limits<T>::epsilon() )
      {
        return( length( v ) <= eps );
      }

      /*! \brief Determine if a vector is uniform, that is, all its components are equal.
       *  \param v A constant reference to the vector to test.
       *  \param eps An optional type dependent epsilon, defining the acceptable deviation.
       *  \return \c true, if all components of \a v are equal, otherwise \c false. */
      template<unsigned int n, typename T>
      bool isUniform( const Vecnt<n,T> & v, T eps = std::numeric_limits<T>::epsilon() )
      {
        bool uniform = true;
        for ( unsigned int i=1 ; i<n && uniform ; i++ )
        {
          uniform = ( abs( v[0] - v[i] ) <= eps );
        }
        return( uniform );
      }

      /*! \brief Determine the length of a vector.
       *  \param v A constant reference to the vector to use.
       *  \return The length of the vector \a v.
       *  \sa lengthSquared */
      template<unsigned int n, typename T>
        T length( const Vecnt<n,T> & v );

      /*! \brief Determine the squared length of a vector.
       *  \param v A constant reference to the vector to use.
       *  \return The squared length of the vector \a v.
       *  \sa length */
      template<unsigned int n, typename T>
        T lengthSquared( const Vecnt<n,T> & v );

      /*! \brief Determine the maximal element of a vector.
       *  \param v A constant reference to the vector to use.
       *  \return The largest absolute value of \a v.
       *  \sa minElement */
      template<unsigned int n, typename T>
        T maxElement( const Vecnt<n,T> & v );

      /*! \brief Determine the minimal element of a vector.
       *  \param v A constant reference to the vector to use.
       *  \return The smallest absolute value of \a v.
       *  \sa maxElement */
      template<unsigned int n, typename T>
        T minElement( const Vecnt<n,T> & v );

      /*! \brief Normalize a vector.
       *  \param v A reference to the vector to normalize.
       *  \return The length of the unnormalized vector.
       *  \sa isNormalized, length */
      template<unsigned int n, typename T>
        T  normalize( Vecnt<n,T> & v );

      /*! \brief Test for equality of two vectors.
       *  \param v0 A constant reference to the first vector to test.
       *  \param v1 A constant reference to the second vector to test.
       *  \return \c true, if the two vectors component-wise differ less than the type dependent
       *  epsilon, otherwise \c false.
       *  \sa operator!=() */
      template<unsigned int n, typename T>
        bool operator==( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 );

      /*! \brief Test for inequality of two vectors.
       *  \param v0 A constant reference to the first vector to test.
       *  \param v1 A constant reference to the second vector to test.
       *  \return \c true, if the two vectors component-wise differ more than the type dependent
       *  epsilon in at least one component, otherwise \c false.
       *  \sa operator==() */
      template<unsigned int n, typename T>
        bool operator!=( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 );

      /*! \brief Vector addition operator
       *  \param v0 A constant reference to the left operand.
       *  \param v1 A second reference to the right operand.
       *  \return A vector holding the component-wise sum of \a v0 and \a v1.
       *  \sa operator-() */
      template<unsigned int n, typename T>
        Vecnt<n,T> operator+( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 );

      /*! \brief Unary vector negation operator.
       *  \param v A constant reference to a vector.
       *  \return A vector holding the component-wise negation of \a v.
       *  \sa operator-() */
      template<unsigned int n, typename T>
        Vecnt<n,T> operator-( const Vecnt<n,T> & v );

      /*! \brief Vector subtraction operator.
       *  \param v0 A constant reference to the left operand.
       *  \param v1 A second reference to the right operand.
       *  \return A vector holding the component-wise difference of \a v0 and \a v1.
       *  \sa operator+() */
      template<unsigned int n, typename T>
        Vecnt<n,T> operator-( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 );

      /*! \brief Scalar multiplication of a vector.
       *  \param v A constant reference to the left operand.
       *  \param s A scalar as the right operand.
       *  \return A vector holding the component-wise product with s.
       *  \sa operator/() */
      template<unsigned int n, typename T, typename S>
        Vecnt<n,T> operator*( const Vecnt<n,T> & v, S s );

      /*! \brief Scalar multiplication of a vector.
       *  \param s A scalar as the left operand.
       *  \param v A constant reference to the right operand.
       *  \return A vector holding the component-wise product with \a s.
       *  \sa operator/() */
      template<unsigned int n, typename T, typename S>
        Vecnt<n,T> operator*( S s, const Vecnt<n,T> & v );

      /*! \brief Vector multiplication.
       *  \param v0 A constant reference to the left operand.
       *  \param v1 A constant reference to the right operand.
       *  \return The dot product of \a v0 and \a v1.
       *  \sa operator^() */
      template<unsigned int n, typename T>
        T operator*( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 );

      /*! \brief Scalar division of a vector.
       *  \param v A constant reference to the left operand.
       *  \param s A scalar as the right operand.
       *  \return A vector holding the component-wise division by \a s.
       *  \sa operator*() */
      template<unsigned int n, typename T, typename S>
        Vecnt<n,T> operator/( const Vecnt<n,T> & v, S s );

      /*! \brief Determine a vector that's orthonormal to \a v.
       *  \param v A vector to determine an orthonormal vector to.
       *  \return A vector that's orthonormal to \a v.
       *  \note The result of this function is not uniquely defined. In two dimensions, the orthonormal
       *  to a vector can be one of two anti-parallel vectors. In higher dimensions, there are infinitely
       *  many possible results. This function just select one of them.
       *  \sa orthonormalize */
      template<unsigned int n, typename T>
        Vecnt<n,T> orthonormal( const Vecnt<n,T> & v );

      /*! \brief Determine the orthonormal vector of \a v1 with respect to \a v0.
       *  \param v0 A constant reference to the normalized vector to orthonormalize against.
       *  \param v1 A constant reference to the normalized vector to orthonormalize.
       *  \return A normalized vector representing the orthonormalized version of \a v1 with respect to
       *  \a v0.
       *  \note The behavior is undefined if \a v0 or \a v1 are not normalized.
       *  \par Example:
       *  \code
       *    Vec3f newYAxis = orthonormalize( newZAxis, oldYAxis );
       *  \endcode
       *  \sa normalize */
      template<unsigned int n, typename T>
        Vecnt<n,T> orthonormalize( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 );

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // non-member functions, specialized for n == 1
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      /*! \brief Set the values of a two-component vector.
       *  \param v A reference to the vector to set with \a x.
       *  \param x The first component to set. */
      template<typename T>
        Vecnt<1,T> & setVec( Vecnt<1,T> & v, T x );

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // non-member functions, specialized for n == 2
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      /*! \brief Set the values of a two-component vector.
       *  \param v A reference to the vector to set with \a x and \a y.
       *  \param x The first component to set.
       *  \param y The second component to set. */
      template<typename T>
        Vecnt<2,T> & setVec( Vecnt<2,T> & v, T x, T y );

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // non-member functions, specialized for n == 3
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      /*! \brief Determine if two vectors are collinear.
       *  \param v0 A constant reference to the first vector.
       *  \param v1 A constant reference to the second vector.
       *  \param eps An optional type dependent epsilon, defining the acceptable deviation.
       *  \return \c true, if \a v0 and \a v1 are collinear, otherwise \c false.
       *  \sa areOpposite */
      template<typename T>
        bool areCollinear( const Vecnt<3,T> &v0, const Vecnt<3,T> &v1
                         , T eps = std::numeric_limits<T>::epsilon() )
      {
        return( length( v0 ^ v1 ) < eps );
      }

      /*! \brief Cross product operator.
       *  \param v0 A constant reference to the left operand.
       *  \param v1 A constant reference to the right operand.
       *  \return A vector that is the cross product of v0 and v1.
       *  \sa operator*() */
      template<typename T>
        Vecnt<3,T> operator^( const Vecnt<3,T> &v0, const Vecnt<3,T> &v1 );

      /*! \brief Set the values of a three-component vector.
       *  \param v A reference to the vector to set with \a x, \a y and \a z.
       *  \param x The first component to set.
       *  \param y The second component to set.
       *  \param z The third component to set. */
      template<typename T>
        Vecnt<3,T> & setVec( Vecnt<3,T> & v, T x, T y, T z );

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // non-member functions, specialized for n == 4
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      /*! \brief Set the values of a four-component vector.
       *  \param v A reference to the vector to set with \a x, \a y, \a z, and \a w.
       *  \param x The first component to set.
       *  \param y The second component to set.
       *  \param z The third component to set.
       *  \param w The fourth component to set. */
      template<typename T>
        Vecnt<4,T> & setVec( Vecnt<4,T> & v, T x, T y, T z, T w );

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // inlined member functions
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      template<unsigned int n, typename T>
      inline Vecnt<n,T>::Vecnt()
      {
      }

      template<unsigned int n, typename T>
      template<unsigned int m, typename S>
      inline Vecnt<n,T>::Vecnt( const Vecnt<m,S> & rhs )
      {
        for ( unsigned int i=0 ; i<std::min( n, m ) ; ++i )
        {
          m_vec[i] = (T)rhs[i];
        }
      }

      template<unsigned int n, typename T>
      template<typename S, typename R>
      inline Vecnt<n,T>::Vecnt( const Vecnt<n-1,S> & rhs, R last )
      {
        for ( unsigned int i=0 ; i<n-1 ; ++i )
        {
          m_vec[i] = (T)rhs[i];
        }
        m_vec[n-1] = (T)last;
      }

      template<unsigned int n, typename T>
      inline Vecnt<n,T>::Vecnt( T x )
      {
        setVec( *this, x );
      }

      template<unsigned int n, typename T>
      inline Vecnt<n,T>::Vecnt( T x, T y )
      {
        setVec( *this, x, y );
      }

      template<unsigned int n, typename T>
      inline Vecnt<n,T>::Vecnt( T x, T y, T z )
      {
        setVec( *this, x, y, z );
      }

      template<unsigned int n, typename T>
      inline Vecnt<n,T>::Vecnt( T x, T y, T z, T w )
      {
        setVec( *this, x, y, z, w );
      }

      template<unsigned int n, typename T>
      inline const T * Vecnt<n,T>::getPtr() const
      {
        return( &m_vec[0] );
      }

      template<unsigned int n, typename T>
      inline T Vecnt<n,T>::normalize()
      {
        return( normalize( *this ) );
      }

      // TODO why is S used as template parameter. It could be just size_t which is the type used for ptr operations.
      template<unsigned int n, typename T>
      inline T & Vecnt<n,T>::operator[]( size_t i )
      {
        assert( i < n );
        return( m_vec[i] );
      }

      // TODO why is S used as template parameter. It could be just size_t which is the type used for ptr operations.
      template<unsigned int n, typename T>
      inline const T & Vecnt<n,T>::operator[]( size_t i ) const
      {
        assert( i < n );
        return( m_vec[i] );
      }

      template<unsigned int n, typename T>
      template<typename S>
      inline Vecnt<n,T> & Vecnt<n,T>::operator=( const Vecnt<n,S> & rhs )
      {
        for ( unsigned int i=0 ; i<n ; ++i )
        {
          m_vec[i] = T(rhs[i]);
        }
        return( *this );
      }

      template<unsigned int n, typename T>
      template<typename S>
      inline Vecnt<n,T> & Vecnt<n,T>::operator+=( const Vecnt<n,S> & rhs )
      {
        for ( unsigned int i=0 ; i<n ; ++i )
        {
          m_vec[i] += T(rhs[i]);
        }
        return( *this );
      }

      template<unsigned int n, typename T>
      template<typename S>
      inline Vecnt<n,T> & Vecnt<n,T>::operator-=( const Vecnt<n,S> & rhs )
      {
        for ( unsigned int i=0 ; i<n ; ++i )
        {
          m_vec[i] -= T(rhs[i]);
        }
        return( *this );
      }

      template<unsigned int n, typename T>
      template<typename S>
      inline Vecnt<n,T> & Vecnt<n,T>::operator*=( S s )
      {
        for ( unsigned int i=0 ; i<n ; ++i )
        {
          m_vec[i] *= T(s);
        }
        return( *this );
      }

      template<unsigned int n, typename T>
      template<typename S>
      inline Vecnt<n,T> & Vecnt<n,T>::operator/=( S s )
      {
        DP_PRIVATE_ASSERT( std::numeric_limits<S>::epsilon() < abs(s) );
        for ( unsigned int i=0 ; i<n ; ++i )
        {
          m_vec[i] /= T(s);
        }
        return( *this );
      }

      template<unsigned int n, typename T>
      inline void Vecnt<n,T>::orthonormalize( const Vecnt<n,T> & v )
      {
        *this = orthonormalize( v, *this );
      }


      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // inlined non-member functions
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      template<unsigned int n, typename T>
      inline bool  areSimilar( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1, T eps )
      {
        for ( unsigned int i=0 ; i<n ; ++i )
        {
          if ( ( v0[i] != v1[i] ) && ( ( v0[i] < v1[i] ) ? ( v0[i] + eps < v1[i] ) : ( v1[i] + eps < v0[i] ) ) )
          {
            return( false );
          }
        }
        return( true );
      }

      template<unsigned int n, typename T>
      inline T distance( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 )
      {
        return( length( v0 - v1 ) );
      }

      template<unsigned int n, typename T>
      inline T intensity( const Vecnt<n,T> & v )
      {
        T intens(0);
        for ( unsigned int i=0 ; i<n ; i++ )
        {
          intens += v[i];
        }
        return( intens / n );
      }

      template<unsigned int n, typename T>
      inline T length( const Vecnt<n,T> & v )
      {
        return( sqrt( lengthSquared( v ) ) );
      }

      template<unsigned int n, typename T>
      inline T lengthSquared( const Vecnt<n,T> & v )
      {
        return( v * v );
      }

      template<unsigned int n, typename T>
      inline T maxElement( const Vecnt<n,T> & v )
      {
        T me = abs( v[0] );
        for ( unsigned int i=1 ; i<n ; ++i )
        {
          T t = abs( v[i] );
          if ( me < t )
          {
            me = t;
          }
        }
        return( me );
      }

      template<unsigned int n, typename T>
      inline T minElement( const Vecnt<n,T> & v )
      {
        T me = abs( v[0] );
        for ( unsigned int i=1 ; i<n ; ++i )
        {
          T t = abs( v[i] );
          if ( t < me )
          {
            me = t;
          }
        }
        return( me );
      }

      template<unsigned int n, typename T>
      inline T normalize( Vecnt<n,T> & v )
      {
        T norm = length( v );
        if ( std::numeric_limits<T>::epsilon() < norm )
        {
          v /= norm;
          //  DP_PRIVATE_ASSERT( isNormalized( v ) ); // due to rounding errors this assertion might fail !
        }
        return( norm );
      }

      template<unsigned int n>
      inline float normalize( Vecnt<n,float> & v )
      {
        Vecnt<n,double> vd(v);
        double norm = normalize( vd );
        v = vd;
        DP_PRIVATE_ASSERT( ( norm <= std::numeric_limits<double>::epsilon() ) || isNormalized(v) );
        return( (float)norm );
      }

      template<unsigned int n, typename T>
      inline bool  operator==( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 )
      {
        return( areSimilar( v0, v1, std::numeric_limits<T>::epsilon() ) );
      }

      template<unsigned int n, typename T>
      inline bool  operator!=( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 )
      {
        return( ! ( v0 == v1 ) );
      }

      template<unsigned int n, typename T>
      inline Vecnt<n,T>  operator+( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 )
      {
        Vecnt<n,T> ret(v0);
        ret += v1;
        return( ret );
      }

      template<unsigned int n, typename T>
      inline Vecnt<n,T>  operator-( const Vecnt<n,T> & v )
      {
        Vecnt<n,T> ret;
        for ( unsigned int i=0 ; i<n ; ++i )
        {
          ret[i] = - v[i];
        }
        return( ret );
      }

      template<unsigned int n, typename T>
      inline Vecnt<n,T>  operator-( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 )
      {
        Vecnt<n,T> ret(v0);
        ret -= v1;
        return( ret );
      }

      template<unsigned int n, typename T, typename S>
      inline Vecnt<n,T>  operator*( const Vecnt<n,T> & v, S s )
      {
        Vecnt<n,T> ret(v);
        ret *= s;
        return( ret );
      }

      template<unsigned int n, typename T, typename S>
      inline Vecnt<n,T>  operator*( S s, const Vecnt<n,T> & v )
      {
        return( v * s );
      }

      template<unsigned int n, typename T>
      inline T operator*( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 )
      {
        T ret(0);
        for ( unsigned int i=0 ; i<n ; ++i )
        {
          ret += v0[i] * v1[i];
        }
        return( ret );
      }

      template<unsigned int n, typename T, typename S>
      inline Vecnt<n,T> operator/( const Vecnt<n,T> & v, S s )
      {
        Vecnt<n,T> ret(v);
        ret /= s;
        return( ret );
      }

      // generic less than and greater than comparison not implemented!
      template<unsigned int n, typename T>
      bool operator<(const Vecnt<n,T>& lhs, const Vecnt<n,T>& rhs);

      template<unsigned int n, typename T>
      bool operator>(const Vecnt<n,T>& lhs, const Vecnt<n,T>& rhs);

      template<unsigned int n, typename T>
      inline bool operator<(const Vecnt<n,T>& lhs, const Vecnt<n,T>& rhs)
      {
        for ( unsigned int i=0 ; i<n ; i++)
        {
          // here, for j<i, lhs[j] == rhs[j]
          if ( lhs[i] < rhs[i] )
          {
            // i is first element different: less
            return( true );
          }
          else if ( rhs[i] < lhs[i] )
          {
            // i is first element different: greater
            return( false );
          }
        }
        // all elements equal
        return( false );
      }

      template<unsigned int n, typename T>
      inline bool operator>(const Vecnt<n,T>& lhs, const Vecnt<n,T>& rhs)
      {
        for ( unsigned int i=0 ; i<n ; i++)
        {
          // here, for j<i, lhs[j] == rhs[j]
          if ( lhs[i] < rhs[i] )
          {
            // i is first element different: less
            return( false );
          }
          else if ( rhs[i] < lhs[i] )
          {
            // i is first element different: greater
            return( true );
          }
        }
        // all elements equal
        return( false );
      }

      template<unsigned int n, typename T>
      inline Vecnt<n,T> orthonormal( const Vecnt<n,T> & v )
      {
        //DP_STATIC_ASSERT( 1 < n );
        //DP_STATIC_ASSERT( ! std::numeric_limits<T>::is_integer );

        T firstV(-1), secondV(-1);
        unsigned int firstI(0), secondI(0);
        Vecnt<n,T> result;
        for ( unsigned int i=0 ; i<n ; i++ )
        {
          T avi = abs( v[i] );
          if ( firstV < avi )
          {
            firstV = avi;
            firstI = i;
          }
          else if ( secondV < avi )
          {
            secondV = avi;
            secondI = i;
          }
          result[i] = T(0);
        }
        DP_ASSERT( ( T(0) < firstV ) && ( firstI != secondI ) );
        result[firstI] = secondV;
        result[secondI] = -firstV;
        result.normalize();
        return( result );
      }

      template<unsigned int n, typename T>
      inline Vecnt<n,T>  orthonormalize( const Vecnt<n,T> & v0, const Vecnt<n,T> & v1 )
      {
        DP_PRIVATE_ASSERT( isNormalized( v0 ) && isNormalized( v1 ) && !areCollinear( v0, v1 ) );
        //  determine the orthogonal projection of v1 on v0 : ( v0 * v1 ) * v0
        //  and subtract it from v1 resulting in the orthogonalized version of v1
        Vecnt<n,T> vr = v1 - ( v0 * v1 ) * v0;
        vr.normalize();
        //  don't assert the general case, because this orthonormalization is far from exact
        //  DP_PRIVATE_ASSERT( areOrthonormal( vr, v0 ) );
        return( vr );
      }

      template<typename T>
      inline Vecnt<3,T> orthonormalize( const Vecnt<3,T> & v0, const Vecnt<3,T> & v1 )
      {
        DP_PRIVATE_ASSERT( isNormalized( v0 ) && isNormalized( v1 ) && !areCollinear( v0, v1 ) );
        Vecnt<3,T> vr = v0 ^ ( v1 ^ v0 );
        vr.normalize();
        DP_PRIVATE_ASSERT( areOrthonormal( vr, v0 ) );
        return( vr );
      }

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // inlined non-member functions, specialized for n == 1
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      template<typename T>
      inline Vecnt<1,T> & setVec( Vecnt<1,T> & v, T x )
      {
        v[0] = x;
        return( v );
      }


      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // inlined non-member functions, specialized for n == 2
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      template<typename T>
      inline Vecnt<2,T> & setVec( Vecnt<2,T> & v, T x, T y )
      {
        v[0] = x;
        v[1] = y;
        return( v );
      }


      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // inlined non-member functions, specialized for n == 3
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      template<typename T>
      inline Vecnt<3,T> operator^( const Vecnt<3,T> &v0, const Vecnt<3,T> &v1 )
      {
        Vecnt<3,T> ret;
        ret[0] = v0[1] * v1[2] - v0[2] * v1[1];
        ret[1] = v0[2] * v1[0] - v0[0] * v1[2];
        ret[2] = v0[0] * v1[1] - v0[1] * v1[0];
        DP_PRIVATE_ASSERT( areOrthogonal( ret, v0 ) && areOrthogonal( ret, v1 ) );
        return( ret );
      }

      template<typename T>
      inline Vecnt<3,T> & setVec( Vecnt<3,T> & v, T x, T y, T z )
      {
        v[0] = x;
        v[1] = y;
        v[2] = z;
        return( v );
      }

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // inlined non-member functions, specialized for n == 4
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      template<typename T>
      inline Vecnt<4,T> & setVec( Vecnt<4,T> & v, T x, T y, T z, T w )
      {
        v[0] = x;
        v[1] = y;
        v[2] = z;
        v[3] = w;
        return( v );
      }

  #if defined(RENDERERGL_SSE)
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
           *  \remarks This constructor contains a compile-time assertion, to make sure that \a m is one
           *  less than \a n. The values of \a rhs of type \a S are converted to type \a T and assigned
           *  as the first values of \c this. The value \a last of type \a R also is converted to type
           *  \a T and assigned as the last value of \c this.
           *  \par Example:
           *  \code
           *    Vec3f v3f(0.0f,0.0f,0.0f);
           *    Vec4f v4f(v3f,1.0f);
           *  \endcode */
          Vecnt( const Vecnt<3,float> & rhs, float last );

          /*! \brief Constructor for a single Element written to x,y,z,w
              \param \param u Value to use for all 4 components.
          **/
          Vecnt( float u );

          /*! \brief Constructor for a four-element vector.
           *  \param x First element of the vector.
           *  \param y Second element of the vector.
           *  \param z Third element of the vector.
           *  \param w Fourth element of the vector.
           *  \remarks This constructor contains a compile-time assertion, to make sure it is used for
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

        private:
          union
          {
            __m128 m_sse;
            float  m_vector[4];
          };
      };

      /*! \brief Default constructor.
        *  \remarks For performance reasons, no initialization is performed. */
      inline Vecnt<4, float>::Vecnt()
      {
        //m_sse = _mm_setzero_ps();
      }

      inline Vecnt<4, float>::Vecnt( const Vecnt<3,float>& rhs, float last )
      {
        m_sse = _mm_set_ps( rhs[0], rhs[1], rhs[2], last );
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
        assert( i < 4 );
        return m_vector[i];
      }

      /*! \brief Constant access operator to the values of a vector.
        *  \param i The index of the value to access.
        *  \return A constant reference to the value at position \a i in this vector.
        *  \remarks The index \a i has to be less than 4.
        *  \note The behavior is undefined if \ i is greater or equal to \a n. */
      inline const float& Vecnt<4, float>::operator[]( size_t i ) const
      {
        assert( i < 4 );
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
        __m128 scale = _mm_set_ss( s );
        scale = _mm_shuffle_ps( scale, scale, _MM_SHUFFLE(0,0,0,0) );
        m_sse = _mm_div_ps( m_sse, scale );
        return *this;
      }

      inline Vecnt< 4, float > operator*( const Vecnt< 4, float > &v1, const Vecnt< 4, float > &v2 )
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
      inline Vecnt< 4, float > shuffle( const Vecnt< 4, float > &vector )
      {
        return Vecnt< 4, float >( _mm_shuffle_ps(vector.sse(), vector.sse(), _MM_SHUFFLE( i3, i2, i1, i0 )));
      }

  #endif

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // Convenience type definitions
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      typedef Vecnt<1,float>  Vec1f;
      typedef Vecnt<1,double> Vec1d;
      typedef Vecnt<2,float>  Vec2f;
      typedef Vecnt<2,double> Vec2d;
      typedef Vecnt<3,float>  Vec3f;
      typedef Vecnt<3,double> Vec3d;
      typedef Vecnt<4,float>  Vec4f;
      typedef Vecnt<4,double> Vec4d;
      typedef Vecnt<1,int> Vec1i;
      typedef Vecnt<2,int> Vec2i;
      typedef Vecnt<3,int> Vec3i;
      typedef Vecnt<4,int> Vec4i;
      typedef Vecnt<1,unsigned int> Vec1ui;
      typedef Vecnt<2,unsigned int> Vec2ui;
      typedef Vecnt<3,unsigned int> Vec3ui;
      typedef Vecnt<4,unsigned int> Vec4ui;

    } // namespace gl
  } // namespace rix
} // namespace dp
