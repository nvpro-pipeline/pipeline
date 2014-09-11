// Copyright NVIDIA Corporation 2002-2006
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

#include <dp/math/Vecnt.h>
#include <climits>

namespace dp
{
  namespace math
  {
    /*! \brief Plane class
     *  \remarks A Plane is a Hyperplane in n-space. It's defined by a normal and an offset, such that
     *  for any point x in the plane, the equation \code normal * x + offset = 0 \endcode holds. */
    template<unsigned int n,typename T> class Planent
    {
      public:
        /*! \brief Default constructor.
         *  \remarks For performance reasons no initialization is performed. */
        Planent();

        /*! \brief Constructor by normal and an offset.
         *  \param normal The normal of the plane. 
         *  \param offset The offset of the plane.
         *  \note It is assumed that \a normal is normalized. */
        Planent( const dp::math::Vecnt<n,T> & normal, T offset );

        /*! \brief Constructor by normal and a point on the plane.
         *  \param normal The normal of the plane.
         *  \param p An arbitrary point on the plane.
         *  \remarks The plane containing the point \a p, with the plane normal \a normal is created. */
        Planent( const dp::math::Vecnt<n,T> & normal, const dp::math::Vecnt<n,T> & p );

        /*! \brief Constructor by three points on the plane.
         *  \param p0 A first point on the plane.
         *  \param p1 A second point on the plane.
         *  \param p2 A third point on the plane.
         *  \remarks Three points uniquely define a plane. */
        Planent( const dp::math::Vecnt<n,T> & p0, const dp::math::Vecnt<n,T> & p1, const dp::math::Vecnt<n,T> & p2 );

      public:
        /*! \brief Get the normal of the plane.
         *  \return A constant reference to the normal of the plane.
         *  \sa getOffset, setNormal */
        const dp::math::Vecnt<n,T> & getNormal() const;

        /*! \brief Get the offset of the plane.
         *  \return The offset of the plane.
         *  \sa getNormal, setOffset */
        T getOffset() const;

        /*! \brief Function call operator.
         *  \param p A reference to a constant point.
         *  \return The signed distance of the point \a p from the plane. */
        T operator()( const dp::math::Vecnt<n,T> & p ) const;

        /*! \brief Set the normal of the plane.
         *  \param normal A reference to the constant normal to use.
         *  \sa getNormal, setOffset */
        void setNormal( const dp::math::Vecnt<n,T> & normal );

        /*! \brief Set the offset of the plane.
         *  \param offset The offset to use.
         *  \sa getOffset, setNormal */
        void setOffset( T offset );

      private:
        dp::math::Vecnt<n,T>  m_normal;
        T                     m_offset;
    };

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    /*! \brief Test if two points are on opposite sides of a plane.
     *  \param pl A reference to a constant plane.
     *  \param p0 A reference to the constant first point.
     *  \param p1 A reference to the constant second point.
     *  \return \c true, if \a p0 and \a p1 are on opposite sides of \a pl.
     *  \sa areOnSameSide */
    template<unsigned int n, typename T>
      bool areOnOppositeSides( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p0, const dp::math::Vecnt<n,T> & p1 );

    /*! \brief Test if two points are on the same side of a plane.
     *  \param pl A reference to a constant plane.
     *  \param p0 A reference to the constant first point.
     *  \param p1 A reference to the constant second point.
     *  \return \c true, if \a p0 and \a p1 are on the same side of \a pl.
     *  \sa areOnOppositeSide */
    template<unsigned int n, typename T>
      bool areOnSameSide( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p0, const dp::math::Vecnt<n,T> & p1 );

    /*! \brief Calculate the distance of a point to a plane.
     *  \param pl A reference to the constant plane.
     *  \param p A reference to the constant point.
     *  \return The distance of the point \a p to the plane \a pl.
     *  \sa signedDistance */
    template<unsigned int n, typename T>
      T distance( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p );

    /*! \brief Calculate the point on a plane nearest to a give point.
     *  \param pl A reference to the constant plane.
     *  \param p A reference to the constant point.
     *  \return The point on the plane \a pl nearest to the point \a p.
     *  \sa distance, signedDistance */
    template<unsigned int n, typename T>
      dp::math::Vecnt<n,T> nearestPoint( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p );

    /*! \brief Test for equality of two planes.
     *  \param pl0 A reference to the constant first plane.
     *  \param pl1 A reference to the constant second plane.
     *  \return \c true, if the normals of the two planes are equal and the offsets differ less than
     *  the type specific epsilon. Otherwise \c false.
     *  \sa operator!=() */
    template<unsigned int n, typename T>
      bool operator==( const Planent<n,T> & pl0, const Planent<n,T> & pl1 );

    /*! \brief Test for inequality of two planes.
     *  \param pl0 A reference to the constant first plane.
     *  \param pl1 A reference to the constant second plane.
     *  \return \c true, if the normals of the two planes are not equal or the offsets differ more
     *  than the type specific epsilon. Otherwise \c false.
     *  \sa operator==() */
    template<unsigned int n, typename T>
      bool operator!=( const Planent<n,T> & pl0, const Planent<n,T> & pl1 );

    /*! \brief Calculate the signed distance of a point to a plane.
     *  \param pl A reference to the constant plane.
     *  \param p A reference to the constant point.
     *  \return The signed distance of the point \a p to the plane \a pl.
     *  \remarks The signed distance of a point \a p to a plane \a pl is positive, if \a p is in the
     *  half space the normal a \a pl points to. If \a p is in the other half space, the signed
     *  distance is negative.
     *  \sa distance */
    template<unsigned int n, typename T>
      T signedDistance( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p );

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // Convenience type definitions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    typedef Planent<3,float>  Plane3f;
    typedef Planent<3,double> Plane3d;

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // inlined member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    template<unsigned int n, typename T>
    inline Planent<n,T>::Planent()
    {
    }

    template<unsigned int n, typename T>
    inline Planent<n,T>::Planent( const dp::math::Vecnt<n,T> & normal, T offset )
      : m_normal(normal)
      , m_offset(offset)
    {
    }

    template<unsigned int n, typename T>
    inline Planent<n,T>::Planent( const dp::math::Vecnt<n,T> & normal, const dp::math::Vecnt<n,T> & p )
      : m_normal(normal)
      , m_offset(-p*normal)
    {
    }

    template<unsigned int n, typename T>
    inline Planent<n,T>::Planent( const dp::math::Vecnt<n,T> & p0, const dp::math::Vecnt<n,T> & p1, const dp::math::Vecnt<n,T> & p2 )
    {
      m_normal = ( p1 - p0 ) ^ ( p2 - p0 );
      m_normal.normalize();
      m_offset = - m_normal * p0;
    }

    template<unsigned int n, typename T>
    inline const dp::math::Vecnt<n,T> & Planent<n,T>::getNormal() const
    {
      return( m_normal );
    }

    template<unsigned int n, typename T>
    inline T Planent<n,T>::getOffset() const
    {
      return( m_offset );
    }

    template<unsigned int n, typename T>
    inline T Planent<n,T>::operator()( const dp::math::Vecnt<n,T> & p ) const
    {
      return( m_normal * p + m_offset );
    }

    template<unsigned int n, typename T>
    inline void Planent<n,T>::setNormal( const dp::math::Vecnt<n,T> & normal )
    {
      m_normal = normal;
    }

    template<unsigned int n, typename T>
    inline void Planent<n,T>::setOffset( T offset )
    {
      m_offset = offset;
    }

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // inlined non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    template<unsigned int n, typename T>
    bool areOnOppositeSides( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p0, const dp::math::Vecnt<n,T> & p1 )
    {
      return( pl( p0 ) * pl( p1 ) < 0 );
    }

    template<unsigned int n, typename T>
    bool areOnSameSide( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p0, const dp::math::Vecnt<n,T> & p1 )
    {
      return( 0 < pl( p0 ) * pl( p1 ) );
    }

    template<unsigned int n, typename T>
    T distance( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p )
    {
      return( abs( pl( p ) ) );
    }

    template<unsigned int n, typename T>
    dp::math::Vecnt<n,T> nearestPoint( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p )
    {
      return( p - pl( p ) * pl.getNormal() );
    }

    template<unsigned int n, typename T>
    bool operator==( const Planent<n,T> & pl0, const Planent<n,T> & pl1 )
    {
      return(   ( pl0.getNormal() == pl1.getNormal() )
            &&  ( abs( pl0.getOffset() - pl1.getOffset() ) < FLT_EPSILON ) );
    }

    template<unsigned int n, typename T>
    bool operator!=( const Planent<n,T> & pl0, const Planent<n,T> & pl1 )
    {
      return( ! ( pl0 == pl1 ) );
    }

    template<unsigned int n, typename T>
    T signedDistance( const Planent<n,T> & pl, const dp::math::Vecnt<n,T> & p )
    {
      return( pl( p ) );
    }
  } // namespace math
} // namespace dp

